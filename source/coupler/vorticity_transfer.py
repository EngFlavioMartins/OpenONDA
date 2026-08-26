"""Absolute FVM-state replacement inside the FVM--VPM overlap."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
import logging
from pathlib import Path
from typing import cast

import numpy as np
from scipy.spatial import cKDTree  # type: ignore[missing-module-attribute]

from source import log_style
from source.coupler.interpolation import FVMVelocityInterpolator
from source.coupler.lattice_transfer import (
    RenewalLattice,
    blend_fvm_vpm_circulation_in_renewal_belt,
    build_renewal_lattice,
    first_vorticity_moment,
    state_blend_weight,
)
from source.coupler.renewal_projection import (
    GBDRenewalProjectionResult,
    SparseRenewalProjectionResult,
    evaluate_sparse_gaussian_vorticity,
    gbd_guard_width,
    geometric_renewal_mask,
    select_residual_support_positions,
    solve_sparse_renewal_projection,
)
from source.coupler.reporting import format_coupler_log
from source.coupler.stable_renewal import (
    StableRenewalLattice,
    build_stable_renewal_lattice,
    renew_stable_overlap,
    vortex_strength_from_velocity_trace,
)
from source.solvers.vpm.diagnostics.resolution import discretization_health

logger = logging.getLogger("coupler")


def _smoothstep(values: np.ndarray | float, lower: float, upper: float) -> np.ndarray:
    """C1 Hermite ramp used by the proven solid and mesh confidence tapers."""
    span = float(upper) - float(lower)
    if abs(span) < np.finfo(np.float64).tiny:
        return np.where(np.asarray(values, dtype=np.float64) >= upper, 1.0, 0.0)
    phase = np.clip((np.asarray(values, dtype=np.float64) - lower) / span, 0.0, 1.0)
    return phase * phase * (3.0 - 2.0 * phase)


def replacement_eta(
    points: np.ndarray,
    box: np.ndarray | list[float] | tuple[float, ...],
    blend_width: float,
) -> np.ndarray:
    """Return the FVM state-replacement weight at ``points``.

    ``blend_width == 0`` selects a hard ownership boundary: every point inside
    ``box`` has ``eta = 1`` and every point outside has ``eta = 0``. A positive
    width replaces the hard jump by a C1 cosine ramp measured inward from the
    six box faces. This is a state partition, not an additive correction.
    """
    return state_blend_weight(points, box, blend_width)


def required_renewal_buffer_length(
    freestream_velocity: np.ndarray | list[float] | tuple[float, ...],
    coupling_time_step: float,
    particle_spacing: float,
    *,
    advection_safety_factor: float = 1.5,
) -> float:
    """Return the release travel plus complete M4' support required by renewal."""
    velocity = np.asarray(freestream_velocity, dtype=np.float64).reshape(3)
    dt = float(coupling_time_step)
    spacing = float(particle_spacing)
    safety = float(advection_safety_factor)
    if (
        not np.all(np.isfinite(velocity))
        or not np.isfinite(dt)
        or dt < 0.0
        or not np.isfinite(spacing)
        or spacing <= 0.0
        or not np.isfinite(safety)
        or safety < 1.0
    ):
        raise ValueError("renewal buffer inputs must be finite and physically valid")
    return safety * float(np.linalg.norm(velocity)) * dt + 2.0 * spacing


@dataclass(frozen=True)
class TransferResult:
    """Particle and circulation budget for one absolute state replacement."""

    n_particles_before: int
    n_particles_retained: int
    n_particles_removed: int
    n_particles_blended: int
    n_particles_injected: int
    n_particles_after: int
    injected_vortex_strength_l1: float
    injected_vortex_strength_net: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    replaced_vortex_strength_l1: float = 0.0
    replaced_vortex_strength_net: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    state_change_vortex_strength_net: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    eta_blending_enabled: bool = False
    transfer_method: str = "fvm_cell_centres"
    mapped_target_nodes: int = 0
    excluded_solid_target_nodes: int = 0
    excluded_solid_active_nodes: int = 0
    excluded_solid_vortex_strength_net: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    excluded_solid_vortex_strength_l1: float = 0.0
    excluded_solid_first_moment: np.ndarray = field(
        default_factory=lambda: np.zeros((3, 3), dtype=np.float64)
    )
    mapped_first_moment: np.ndarray = field(
        default_factory=lambda: np.zeros((3, 3), dtype=np.float64)
    )
    mapped_target_vortex_strength_l1: float = 0.0
    mapped_target_vortex_strength_net: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    maximum_mapped_vortex_strength: float = 0.0
    fvm_donor_vortex_strength_net: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    fvm_mapped_vortex_strength_net: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    fvm_donor_first_moment: np.ndarray = field(
        default_factory=lambda: np.zeros((3, 3), dtype=np.float64)
    )
    fvm_mapped_first_moment: np.ndarray = field(
        default_factory=lambda: np.zeros((3, 3), dtype=np.float64)
    )
    blend_cross_divergence_l2_before: float = 0.0
    blend_cross_divergence_l2_after: float = 0.0
    blend_cross_divergence_relative: float = 0.0
    mapped_vorticity_divergence_error: float | None = None
    mapped_vortex_strength_misalignment_degrees: float | None = None
    mapped_mean_overlap_ratio: float | None = None
    projection_vorticity_relative_error: float = 0.0
    projection_velocity_relative_error: float | None = None
    projection_condition_number: float = 0.0
    selective_support_births: int = 0
    renewal_guard_width: float = 0.0
    renewal_diffusion_substeps: int = 0
    renewed_input_particles: int = 0
    renewed_output_particles: int = 0
    preserved_outer_particles: int = 0
    pruned_lattice_nodes: int = 0
    pruned_vortex_strength_l1: float = 0.0
    pruned_vortex_strength_fraction: float = 0.0
    population_pruned_particles: int = 0
    population_pruned_vortex_strength_fraction: float = 0.0
    population_pruned_velocity_bound: float = 0.0
    renewal_cfl: float = 0.0
    renewal_conservation_error: float = 0.0
    renewal_linear_impulse_error: float = 0.0
    representation_residual_before_prune: float | None = None
    representation_residual_after_prune: float | None = None
    maximum_transfer_amplification: float = 0.0


def _validate_particle_sources(
    position: np.ndarray,
    cell_volume: np.ndarray,
    vorticity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_position = np.asarray(position, dtype=np.float64).reshape(-1, 3)
    volume = np.asarray(cell_volume, dtype=np.float64).reshape(-1)
    source_vorticity = np.asarray(vorticity, dtype=np.float64).reshape(-1, 3)
    if len(source_position) != len(volume) or len(source_position) != len(source_vorticity):
        raise ValueError("FVM position, volume, and vorticity counts must match")
    if not np.all(np.isfinite(source_position)):
        raise RuntimeError("FVM cell positions contain non-finite values")
    if not np.all(np.isfinite(volume)) or np.any(volume <= 0.0):
        raise RuntimeError("FVM cell volumes must be finite and positive")
    if not np.all(np.isfinite(source_vorticity)):
        raise RuntimeError("FVM cell vorticity contains non-finite values")
    return source_position, volume, source_vorticity


def _particle_state_snapshot(vpm) -> dict[str, np.ndarray]:
    """Download every mutable particle field required for a transfer rollback."""
    particles = vpm.particles
    count = int(particles.n_particles_total)
    dtype = np.dtype(getattr(vpm, "np_dtype", np.float64))

    def read(name: str, shape: tuple[int, ...], default: float | int, *, integer: bool = False):
        accessor = getattr(particles, f"{name}_cpu", None)
        value_dtype = np.int32 if integer else dtype
        if callable(accessor):
            return np.ascontiguousarray(np.asarray(accessor(), dtype=value_dtype).copy())
        return np.full(shape, default, dtype=value_dtype)

    return {
        "position": read("position", (count, 3), 0.0),
        "velocity": read("velocity", (count, 3), 0.0),
        "vortex_strength": read("vortex_strength", (count, 3), 0.0),
        "core_radius": read("core_radius", (count,), 0.0),
        "particle_volume": read("particle_volume", (count,), 0.0),
        "kinematic_viscosity": read("kinematic_viscosity", (count,), 0.0),
        "eddy_viscosity": read("eddy_viscosity", (count,), 0.0),
        "group_id": read("group_id", (count,), 0, integer=True),
        "zone_id": read("zone_id", (count,), 0, integer=True),
        "velocity_gradient": read("velocity_gradient", (count, 3, 3), 0.0),
        "strain_rate": read("strain_rate", (count, 3, 3), 0.0),
    }


def _restore_particle_state(vpm, snapshot: dict[str, np.ndarray]) -> None:
    """Restore a transfer snapshot through VPM's atomic field replacement API."""
    replace = getattr(vpm, "replace_vortex_particles", None)
    if not callable(replace):
        raise RuntimeError(
            "FVM/VPM transfer needs replace_vortex_particles to roll back a failed mutation"
        )
    replace(report_removal=False, **snapshot)


def replace_particles_from_buffered_m4_renewal(
    vpm,
    *,
    lattice: StableRenewalLattice,
    fvm_vortex_strength_at_node: Callable[[np.ndarray], np.ndarray],
    particle_fluid_weight: Callable[[np.ndarray], np.ndarray] | None,
    particle_in_solid: Callable[[np.ndarray], np.ndarray] | None,
    prune_threshold: float,
    core_radius_ratio: float,
    amplification_cap: float,
    boundary_prune_multiplier: float,
    kinematic_viscosity: float,
    freestream_speed: float,
    time_step_size: float,
    compute_diagnostics: bool,
) -> TransferResult:
    """Atomically apply the recovered whole-belt M4' renewal to a GBD cloud."""
    if str(getattr(vpm, "viscous_scheme", "")).upper() != "GBD":
        raise ValueError("buffered_m4_renewal requires the GBD viscous scheme")
    viscosity = float(kinematic_viscosity)
    if not np.isfinite(viscosity) or viscosity < 0.0:
        raise ValueError("kinematic_viscosity must be finite and non-negative")

    particles = vpm.particles
    n_before = int(particles.n_particles_total)
    existing_position = np.asarray(particles.position_cpu(), dtype=np.float64).reshape(-1, 3)
    existing_strength = np.asarray(particles.vortex_strength_cpu(), dtype=np.float64).reshape(-1, 3)
    if len(existing_position) != n_before or len(existing_strength) != n_before:
        raise RuntimeError("VPM particle arrays do not match the active particle count")
    if not np.all(np.isfinite(existing_position)) or not np.all(np.isfinite(existing_strength)):
        raise RuntimeError("VPM particle state contains non-finite values")

    result = renew_stable_overlap(
        existing_position,
        existing_strength,
        lattice,
        fvm_vortex_strength_at_node=fvm_vortex_strength_at_node,
        particle_fluid_weight=particle_fluid_weight,
        particle_in_solid=particle_in_solid,
        prune_threshold=prune_threshold,
        core_radius_ratio=core_radius_ratio,
        amplification_cap=amplification_cap,
        boundary_prune_multiplier=boundary_prune_multiplier,
        maximum_particle_count=int(particles.capacity),
        freestream_speed=freestream_speed,
        time_step_size=time_step_size,
        compute_diagnostics=compute_diagnostics,
    )
    n_after = result.particle_count
    if n_after > int(particles.capacity):
        raise RuntimeError(
            f"buffered M4' renewal produced {n_after:,} particles, exceeding "
            f"the VPM capacity {int(particles.capacity):,}"
        )

    dtype = np.dtype(vpm.np_dtype)
    vpm.replace_vortex_particles(
        position=np.ascontiguousarray(result.position, dtype=dtype),
        velocity=np.zeros((n_after, 3), dtype=dtype),
        vortex_strength=np.ascontiguousarray(result.vortex_strength, dtype=dtype),
        core_radius=np.ascontiguousarray(result.core_radius, dtype=dtype),
        particle_volume=np.ascontiguousarray(result.particle_volume, dtype=dtype),
        kinematic_viscosity=np.full(n_after, viscosity, dtype=dtype),
        eddy_viscosity=np.zeros(n_after, dtype=dtype),
        group_id=np.zeros(n_after, dtype=np.int32),
        zone_id=np.zeros(n_after, dtype=np.int32),
        report_removal=False,
    )
    if int(particles.n_particles_total) != n_after:
        raise RuntimeError(
            f"VPM particle count after buffered renewal is "
            f"{int(particles.n_particles_total)}, expected {n_after}"
        )

    renewed_output_count = min(int(result.renewed_output_count), n_after)
    renewed_strength = result.vortex_strength[:renewed_output_count]
    removed_mask = np.ones(n_before, dtype=bool)
    if n_before:
        valid = np.ones(n_before, dtype=bool)
        if particle_in_solid is not None:
            valid &= ~np.asarray(particle_in_solid(existing_position), dtype=bool).reshape(-1)
        preserved = valid & ~np.all(
            (existing_position >= lattice.renewal_bounds[::2])
            & (existing_position <= lattice.renewal_bounds[1::2]),
            axis=1,
        )
        removed_mask = ~preserved
    replaced_strength = existing_strength[removed_mask]
    injected_net = renewed_strength.sum(axis=0, dtype=np.float64)
    output_net = result.vortex_strength.sum(axis=0, dtype=np.float64)
    input_net = existing_strength.sum(axis=0, dtype=np.float64)
    conservation = result.conservation_residual
    maximum_strength = float(np.linalg.norm(result.vortex_strength, axis=1).max(initial=0.0))
    return TransferResult(
        n_particles_before=n_before,
        n_particles_retained=int(result.preserved_outer_count),
        n_particles_removed=int(np.count_nonzero(removed_mask)),
        n_particles_blended=int(result.renewed_input_count),
        n_particles_injected=renewed_output_count,
        n_particles_after=n_after,
        injected_vortex_strength_l1=float(
            np.linalg.norm(renewed_strength, axis=1).sum(dtype=np.float64)
        ),
        injected_vortex_strength_net=injected_net,
        replaced_vortex_strength_l1=float(
            np.linalg.norm(replaced_strength, axis=1).sum(dtype=np.float64)
        ),
        replaced_vortex_strength_net=replaced_strength.sum(axis=0, dtype=np.float64),
        state_change_vortex_strength_net=output_net - input_net,
        eta_blending_enabled=bool(
            np.any((lattice.fvm_authority > 0.0) & (lattice.fvm_authority < 1.0))
        ),
        transfer_method="buffered_m4_renewal",
        mapped_target_nodes=renewed_output_count,
        excluded_solid_target_nodes=int(np.count_nonzero(lattice.solid_interior)),
        excluded_solid_vortex_strength_l1=float(result.excluded_target_vortex_strength_l1),
        mapped_target_vortex_strength_l1=float(
            np.linalg.norm(renewed_strength, axis=1).sum(dtype=np.float64)
        ),
        mapped_target_vortex_strength_net=injected_net,
        maximum_mapped_vortex_strength=maximum_strength,
        renewed_input_particles=int(result.renewed_input_count),
        renewed_output_particles=renewed_output_count,
        preserved_outer_particles=int(result.preserved_outer_count),
        pruned_lattice_nodes=int(result.pruned_node_count),
        pruned_vortex_strength_l1=float(result.pruned_vortex_strength_l1),
        pruned_vortex_strength_fraction=float(result.pruned_vortex_strength_fraction),
        population_pruned_particles=int(result.population_pruned_count),
        population_pruned_vortex_strength_fraction=float(
            result.population_pruned_vortex_strength_fraction
        ),
        population_pruned_velocity_bound=float(result.population_pruned_velocity_bound),
        renewal_cfl=float(result.transfer_cfl),
        renewal_conservation_error=float(conservation.get("total_vortex_strength", 0.0)),
        renewal_linear_impulse_error=float(conservation.get("linear_impulse", 0.0)),
        representation_residual_before_prune=result.representation_residual_before_prune,
        representation_residual_after_prune=result.representation_residual_after_prune,
        maximum_transfer_amplification=float(result.maximum_transfer_amplification),
    )


def apply_projected_gbd_renewal(
    vpm,
    projection: GBDRenewalProjectionResult,
    *,
    particle_spacing: float,
    kinematic_viscosity: float,
) -> TransferResult:
    """Apply one certified absolute projection to the current post-GBD cloud.

    The projection has already classified the current geometry and subtracted
    the complete preserved Gaussian field.  This mutation therefore updates
    only renewable strengths and appends only support positions selected by a
    failed current-basis field gate.  It never performs blanket hole filling or
    asks GBD for an additional zero-time remesh.
    """
    if getattr(vpm, "viscous_scheme", None) != "GBD":
        raise ValueError("apply_projected_gbd_renewal requires a GBD VPM solver")
    spacing = float(particle_spacing)
    viscosity = float(kinematic_viscosity)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("particle_spacing must be finite and positive")
    if not np.isfinite(viscosity) or viscosity < 0.0:
        raise ValueError("kinematic_viscosity must be finite and non-negative")

    particles = vpm.particles
    n_before = int(particles.n_particles_total)
    existing_strength = np.asarray(particles.vortex_strength_cpu(), dtype=np.float64).reshape(-1, 3)
    renewable = np.asarray(projection.renewable_mask, dtype=bool).reshape(-1)
    preserved = np.asarray(projection.preserved_mask, dtype=bool).reshape(-1)
    updated = np.asarray(projection.updated_vortex_strength, dtype=np.float64).reshape(-1, 3)
    birth_position = np.asarray(projection.birth_position, dtype=np.float64).reshape(-1, 3)
    birth_strength = np.asarray(projection.birth_vortex_strength, dtype=np.float64).reshape(-1, 3)
    birth_radius = np.asarray(projection.birth_core_radius, dtype=np.float64).reshape(-1)
    if (
        len(existing_strength) != n_before
        or len(renewable) != n_before
        or len(preserved) != n_before
        or len(updated) != n_before
    ):
        raise RuntimeError("projected renewal arrays do not match the active particle count")
    if np.any(renewable & preserved) or not np.all(renewable | preserved):
        raise RuntimeError("renewable and preserved masks must be complementary")
    if len(birth_position) != len(birth_strength) or len(birth_position) != len(birth_radius):
        raise RuntimeError("projected renewal birth arrays must have matching counts")
    if not np.all(np.isfinite(updated)) or not np.all(np.isfinite(birth_strength)):
        raise RuntimeError("projected renewal strengths must be finite")
    if not np.all(np.isfinite(birth_position)) or not np.all(np.isfinite(birth_radius)):
        raise RuntimeError("projected renewal birth geometry must be finite")
    if np.any(birth_radius <= 0.0):
        raise RuntimeError("projected renewal birth core radii must be positive")
    # The numerical solver promises exact preservation; enforce that contract
    # again at the mutation boundary before touching device state.
    if not np.array_equal(updated[preserved], existing_strength[preserved]):
        raise RuntimeError("projected renewal attempted to modify preserved strengths")

    n_injected = len(birth_position)
    n_after = n_before + n_injected
    if n_after > int(particles.capacity):
        raise RuntimeError(
            "Projected renewal requires "
            f"{n_after:,} particles ({n_before:,} existing, {n_injected:,} births), "
            f"exceeding the VPM capacity {int(particles.capacity):,}."
        )
    delta = updated[renewable] - existing_strength[renewable]
    snapshot = _particle_state_snapshot(vpm)
    try:
        if np.any(renewable):
            vpm.update_particle_vortex_strength(renewable, delta)
        if n_injected:
            dtype = vpm.np_dtype
            vpm.add_vortex_particles(
                position=np.ascontiguousarray(birth_position, dtype=dtype),
                velocity=np.zeros((n_injected, 3), dtype=dtype),
                vortex_strength=np.ascontiguousarray(birth_strength, dtype=dtype),
                core_radius=np.ascontiguousarray(birth_radius, dtype=dtype),
                particle_volume=np.full(n_injected, spacing**3, dtype=dtype),
                kinematic_viscosity=np.full(n_injected, viscosity, dtype=dtype),
                eddy_viscosity=np.zeros(n_injected, dtype=dtype),
                group_id=np.zeros(n_injected, dtype=np.int32),
                zone_id=np.zeros(n_injected, dtype=np.int32),
            )
        if int(particles.n_particles_total) != n_after:
            raise RuntimeError(
                f"VPM particle count after projected renewal is "
                f"{int(particles.n_particles_total)}, expected {n_after}"
            )
    except Exception:
        _restore_particle_state(vpm, snapshot)
        raise

    replaced_strength = existing_strength[renewable]
    injected_net = birth_strength.sum(axis=0, dtype=np.float64)
    state_change = (
        updated.sum(axis=0, dtype=np.float64)
        + injected_net
        - existing_strength.sum(axis=0, dtype=np.float64)
    )
    return TransferResult(
        n_particles_before=n_before,
        n_particles_retained=n_before,
        n_particles_removed=0,
        n_particles_blended=int(np.count_nonzero(renewable)),
        n_particles_injected=n_injected,
        n_particles_after=n_after,
        injected_vortex_strength_l1=float(np.linalg.norm(birth_strength, axis=1).sum()),
        injected_vortex_strength_net=injected_net,
        replaced_vortex_strength_l1=float(np.linalg.norm(replaced_strength, axis=1).sum()),
        replaced_vortex_strength_net=replaced_strength.sum(axis=0, dtype=np.float64),
        state_change_vortex_strength_net=state_change,
        eta_blending_enabled=False,
        transfer_method="projected_gbd_renewal",
        mapped_target_nodes=int(np.count_nonzero(renewable)),
        projection_vorticity_relative_error=(projection.projection.vorticity_relative_error),
        projection_velocity_relative_error=projection.projection.velocity_relative_error,
        projection_condition_number=projection.projection.condition_number,
        selective_support_births=n_injected,
    )


def replace_particles_from_fvm(
    vpm,
    *,
    transfer_box: np.ndarray | list[float] | tuple[float, ...],
    eta_blend_width: float,
    fvm_position: np.ndarray,
    fvm_cell_volume: np.ndarray,
    fvm_vorticity: np.ndarray,
    core_radius_ratio: float,
    kinematic_viscosity: float,
    fvm_solid_mask: np.ndarray | None = None,
) -> TransferResult:
    r"""Replace the overlap state with literal FVM cell circulation.

    Existing particles are attenuated by ``1 - eta`` and FVM cell particles
    carry ``eta * V_cell * omega_F``. Consequently ``eta = 1`` is a hard
    delete/reinject operation, ``eta = 0`` leaves the VPM state untouched, and
    intermediate values form a partition-of-unity state blend. Particles
    outside ``transfer_box`` are never mutated.
    """
    ratio = float(core_radius_ratio)
    viscosity = float(kinematic_viscosity)
    if not np.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("core_radius_ratio must be finite and positive")
    if not np.isfinite(viscosity) or viscosity < 0.0:
        raise ValueError("kinematic_viscosity must be finite and non-negative")

    source_position, source_volume, source_vorticity = _validate_particle_sources(
        fvm_position,
        fvm_cell_volume,
        fvm_vorticity,
    )
    source_eta = replacement_eta(source_position, transfer_box, eta_blend_width)
    if fvm_solid_mask is not None:
        solid = np.asarray(fvm_solid_mask, dtype=bool).reshape(-1)
        if len(solid) != len(source_position):
            raise ValueError("fvm_solid_mask must match the FVM cell count")
        source_eta[solid] = 0.0

    source_strength = source_volume[:, None] * source_vorticity
    inject = (source_eta > 0.0) & np.any(source_strength != 0.0, axis=1)
    injected_position = source_position[inject]
    injected_volume = source_volume[inject]
    injected_strength = source_eta[inject, None] * source_strength[inject]
    injected_core_radius = ratio * np.cbrt(injected_volume)

    particles = vpm.particles
    n_before = int(particles.n_particles_total)
    existing_position = np.asarray(particles.position_cpu(), dtype=np.float64).reshape(-1, 3)
    existing_strength = np.asarray(particles.vortex_strength_cpu(), dtype=np.float64).reshape(-1, 3)
    if len(existing_position) != n_before or len(existing_strength) != n_before:
        raise RuntimeError("VPM particle arrays do not match the active particle count")
    if not np.all(np.isfinite(existing_position)) or not np.all(np.isfinite(existing_strength)):
        raise RuntimeError("VPM particle state contains non-finite values")

    existing_eta = replacement_eta(existing_position, transfer_box, eta_blend_width)
    tolerance = 32.0 * np.finfo(np.float64).eps
    remove = existing_eta >= 1.0 - tolerance
    blend = (existing_eta > tolerance) & ~remove
    remove_index = np.flatnonzero(remove)
    n_removed = int(len(remove_index))
    n_blended = int(np.count_nonzero(blend))
    n_injected = int(len(injected_position))
    n_after = n_before - n_removed + n_injected
    capacity = int(particles.capacity)
    if n_after > capacity:
        raise RuntimeError(
            "FVM overlap replacement requires "
            f"{n_after:,} particles ({n_before:,} before, {n_removed:,} removed, "
            f"{n_injected:,} injected), exceeding the VPM capacity {capacity:,}."
        )

    replaced_strength = existing_eta[:, None] * existing_strength
    replaced_net = replaced_strength.sum(axis=0)
    injected_net = injected_strength.sum(axis=0)

    # All checks above occur before the first mutation.  Any later API failure
    # is rolled back through the solver's full-field replacement API.
    snapshot = _particle_state_snapshot(vpm)
    try:
        if n_blended:
            vpm.update_particle_vortex_strength(
                blend,
                -existing_eta[blend, None] * existing_strength[blend],
            )
        if n_removed:
            vpm.remove_particles(particle_indices=remove_index.tolist())
        if n_injected:
            dtype = vpm.np_dtype
            vpm.add_vortex_particles(
                position=np.ascontiguousarray(injected_position, dtype=dtype),
                velocity=np.zeros((n_injected, 3), dtype=dtype),
                vortex_strength=np.ascontiguousarray(injected_strength, dtype=dtype),
                core_radius=np.ascontiguousarray(injected_core_radius, dtype=dtype),
                particle_volume=np.ascontiguousarray(injected_volume, dtype=dtype),
                kinematic_viscosity=np.full(n_injected, viscosity, dtype=dtype),
                eddy_viscosity=np.zeros(n_injected, dtype=dtype),
                group_id=np.zeros(n_injected, dtype=np.int32),
                zone_id=np.zeros(n_injected, dtype=np.int32),
            )
        actual_after = int(particles.n_particles_total)
        if actual_after != n_after:
            raise RuntimeError(
                f"VPM particle count after replacement is {actual_after}, expected {n_after}"
            )
    except Exception:
        _restore_particle_state(vpm, snapshot)
        raise
    return TransferResult(
        n_particles_before=n_before,
        n_particles_retained=n_before - n_removed,
        n_particles_removed=n_removed,
        n_particles_blended=n_blended,
        n_particles_injected=n_injected,
        n_particles_after=n_after,
        injected_vortex_strength_l1=float(np.linalg.norm(injected_strength, axis=1).sum()),
        injected_vortex_strength_net=injected_net,
        replaced_vortex_strength_l1=float(np.linalg.norm(replaced_strength, axis=1).sum()),
        replaced_vortex_strength_net=replaced_net,
        state_change_vortex_strength_net=injected_net - replaced_net,
        eta_blending_enabled=float(eta_blend_width) > 0.0,
    )


def replace_particles_from_lattice_blend(
    vpm,
    *,
    transfer_box: np.ndarray | list[float] | tuple[float, ...],
    eta_blend_width: float,
    fvm_position: np.ndarray,
    fvm_cell_volume: np.ndarray,
    fvm_vorticity: np.ndarray,
    lattice_anchor: np.ndarray,
    particle_spacing: float,
    core_radius_ratio: float,
    kinematic_viscosity: float,
    fvm_solid_mask: np.ndarray | None = None,
    solid_contains=None,
    renewal_lattice: RenewalLattice | None = None,
    renewal_buffer_length: float | None = None,
    target_solid_mask: np.ndarray | None = None,
    compute_divergence_diagnostic: bool = True,
    discretization_error_limit: float | None = None,
) -> TransferResult:
    """Renew a buffered overlap from one absolute common-lattice state."""
    spacing = float(particle_spacing)
    ratio = float(core_radius_ratio)
    viscosity = float(kinematic_viscosity)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("particle_spacing must be finite and positive")
    if not np.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("core_radius_ratio must be finite and positive")
    if not np.isfinite(viscosity) or viscosity < 0.0:
        raise ValueError("kinematic_viscosity must be finite and non-negative")
    if discretization_error_limit is not None and (
        not np.isfinite(discretization_error_limit) or not 0.0 < discretization_error_limit <= 1.0
    ):
        raise ValueError("discretization_error_limit must lie in (0, 1]")

    particles = vpm.particles
    n_before = int(particles.n_particles_total)
    existing_position = np.asarray(particles.position_cpu(), dtype=np.float64).reshape(-1, 3)
    existing_strength = np.asarray(particles.vortex_strength_cpu(), dtype=np.float64).reshape(-1, 3)
    existing_core_radius = np.asarray(particles.core_radius_cpu(), dtype=np.float64).reshape(-1)
    if (
        len(existing_position) != n_before
        or len(existing_strength) != n_before
        or len(existing_core_radius) != n_before
    ):
        raise RuntimeError("VPM particle arrays do not match the active particle count")
    if (
        not np.all(np.isfinite(existing_position))
        or not np.all(np.isfinite(existing_strength))
        or not np.all(np.isfinite(existing_core_radius))
        or np.any(existing_core_radius <= 0.0)
    ):
        raise RuntimeError("VPM particle state contains non-finite values")

    if renewal_lattice is None:
        buffer_length = (
            2.0 * spacing if renewal_buffer_length is None else float(renewal_buffer_length)
        )
        if not np.isfinite(buffer_length) or buffer_length < 2.0 * spacing:
            raise ValueError("renewal_buffer_length must cover at least the complete M4' support")
        renewal_bounds = np.asarray(transfer_box, dtype=np.float64).reshape(6).copy()
        renewal_bounds[::2] -= buffer_length
        renewal_bounds[1::2] += buffer_length
        renewal_lattice = build_renewal_lattice(
            renewal_bounds,
            lattice_anchor=lattice_anchor,
            spacing=spacing,
        )
    else:
        anchor = np.asarray(lattice_anchor, dtype=np.float64).reshape(3)
        if not np.isclose(renewal_lattice.spacing, spacing, rtol=0.0, atol=0.0):
            raise ValueError("renewal lattice spacing does not match particle_spacing")
        if not np.allclose(
            renewal_lattice.lattice_anchor,
            anchor,
            rtol=0.0,
            atol=32.0 * np.finfo(np.float64).eps * np.maximum(1.0, np.abs(anchor)),
        ):
            raise ValueError("renewal lattice anchor does not match lattice_anchor")
    state = blend_fvm_vpm_circulation_in_renewal_belt(
        fvm_position=fvm_position,
        fvm_cell_volume=fvm_cell_volume,
        fvm_vorticity=fvm_vorticity,
        vpm_position=existing_position,
        vpm_vortex_strength=existing_strength,
        transfer_box=transfer_box,
        blend_width=eta_blend_width,
        lattice=renewal_lattice,
        fvm_solid_mask=fvm_solid_mask,
        vpm_position_dtype=np.dtype(vpm.np_dtype),
        # Periodic spectral divergence is not a valid acceptance metric for
        # this finite free-space wake.  The Gaussian particle-field diagnostic
        # below is the production gate.
        compute_divergence_diagnostic=False,
    )
    if target_solid_mask is not None:
        target_solid = np.asarray(target_solid_mask, dtype=bool).reshape(-1)
    else:
        target_solid = (
            np.asarray(solid_contains(state.position), dtype=bool).reshape(-1)
            if solid_contains is not None
            else np.zeros(len(state.position), dtype=bool)
        )
    if len(target_solid) != len(state.position):
        raise ValueError("solid_contains must return one flag per target lattice node")
    excluded_solid_strength = state.vortex_strength[target_solid]
    mapped_strength = state.vortex_strength.copy()
    mapped_strength[target_solid] = 0.0
    strength_magnitude = np.linalg.norm(mapped_strength, axis=1)
    nonzero = strength_magnitude > 0.0
    target_position = state.position[nonzero]
    target_strength = mapped_strength[nonzero]

    health: dict[str, float] | None = None
    if compute_divergence_diagnostic:
        # Sources represented by the common lattice must not be counted a
        # second time.  Retained particles outside that source set remain part
        # of the physical post-transfer Gaussian field.
        retained = ~state.vpm_source_mask
        health_position = np.vstack((target_position, existing_position[retained]))
        health_strength = np.vstack((target_strength, existing_strength[retained]))
        health_radius = np.concatenate(
            (
                np.full(len(target_position), ratio * spacing, dtype=np.float64),
                existing_core_radius[retained],
            )
        )
        evaluated = discretization_health(health_position, health_strength, health_radius)
        divergence_error = evaluated["vorticity_divergence_error"]
        if np.isfinite(divergence_error):
            health = evaluated
            if (
                discretization_error_limit is not None
                and divergence_error >= discretization_error_limit
            ):
                raise RuntimeError(
                    "common-lattice transfer failed its Gaussian-vorticity divergence gate: "
                    f"error={divergence_error:.6e}, limit={discretization_error_limit:.6e}"
                )
            if discretization_error_limit is not None and divergence_error > 0.05:
                logger.warning(
                    "common-lattice Gaussian-vorticity divergence is in the short-run "
                    "acceptance band: error=%.6e",
                    divergence_error,
                )

    renewable = state.vpm_replace_mask
    # Every source represented in the absolute lattice state is topologically
    # managed by this handoff.  This includes regular nodes in the complete
    # outer support guard: if their new absolute target is zero, retaining them
    # would leave stale zero/cancelled particles and grow the cloud on repeated
    # renewals.  Off-lattice particles beyond the physical belt are not source
    # state and remain fully persistent.
    managed = state.vpm_source_mask
    anchor = np.asarray(lattice_anchor, dtype=np.float64).reshape(3)
    existing_index = np.rint((existing_position - anchor) / spacing).astype(np.int64)
    existing_lattice_position = anchor + spacing * existing_index
    storage_dtype = np.dtype(vpm.np_dtype)
    if not np.issubdtype(storage_dtype, np.floating):
        raise RuntimeError("VPM particle position dtype must be floating point")
    stored_position = np.asarray(existing_position, dtype=storage_dtype)
    stored_lattice_position = np.asarray(existing_lattice_position, dtype=storage_dtype)
    coordinate_ulp = np.maximum(
        np.abs(np.spacing(stored_position)),
        np.abs(np.spacing(stored_lattice_position)),
    ).astype(np.float64)
    coordinate_tolerance = np.maximum(
        8.0 * coordinate_ulp,
        32.0 * np.finfo(np.float64).eps * np.maximum(1.0, np.abs(existing_lattice_position)),
    )
    regular = np.all(
        np.abs(existing_position - existing_lattice_position) <= coordinate_tolerance, axis=1
    )
    target_index = np.rint((target_position - anchor) / spacing).astype(np.int64)
    existing_by_node: dict[tuple[int, int, int], list[int]] = {}
    for particle_index in np.flatnonzero(regular):
        existing_by_node.setdefault(tuple(existing_index[particle_index]), []).append(
            int(particle_index)
        )

    inject = np.ones(len(target_position), dtype=bool)
    remove = managed.copy()
    update_index: list[int] = []
    update_increment: list[np.ndarray] = []
    tolerance = 32.0 * np.finfo(np.float64).eps
    for target_number, node in enumerate(target_index):
        matches = existing_by_node.get(tuple(node), [])
        renewable_matches = [index for index in matches if renewable[index]]
        persistent_matches = [index for index in matches if not renewable[index]]
        if renewable_matches:
            # Reconcile an existing lattice node to the new absolute state.
            # Any duplicates on the renewable side collapse into this one node.
            particle_index = renewable_matches[0]
            remove[particle_index] = False
            delta = target_strength[target_number] - existing_strength[particle_index]
            if np.any(delta != 0.0):
                update_index.append(particle_index)
                update_increment.append(delta)
            inject[target_number] = False
            continue
        if persistent_matches:
            if len(persistent_matches) != 1:
                raise RuntimeError("VPM state contains duplicate persistent lattice nodes")
            particle_index = persistent_matches[0]
            if state.vpm_source_mask[particle_index]:
                # A regular release-belt node was embedded directly in the
                # absolute VPM lattice state.  Reconcile it, never add it.
                delta = target_strength[target_number] - existing_strength[particle_index]
            else:
                # Outer complete-stencil support from a renewable off-lattice
                # source reaches an otherwise independent persistent node.
                delta = target_strength[target_number]
            if np.any(delta != 0.0):
                update_index.append(particle_index)
                update_increment.append(delta)
            inject[target_number] = False

    remove_index = np.flatnonzero(remove)
    n_injected = int(np.count_nonzero(inject))
    n_removed = int(len(remove_index))
    n_after = n_before - n_removed + n_injected
    capacity = int(particles.capacity)
    if n_after > capacity:
        raise RuntimeError(
            "FVM/VPM lattice blend requires "
            f"{n_after:,} particles ({n_before:,} before, {n_removed:,} removed, "
            f"{n_injected:,} injected), exceeding the VPM capacity {capacity:,}."
        )

    # All checks above occur before the first mutation.  Roll back every VPM
    # field if a later mutation or final-count assertion fails.
    snapshot = _particle_state_snapshot(vpm)
    try:
        if update_index:
            order = np.argsort(np.asarray(update_index, dtype=np.int64))
            ordered_index = np.asarray(update_index, dtype=np.int64)[order]
            update_mask = np.zeros(n_before, dtype=bool)
            update_mask[ordered_index] = True
            vpm.update_particle_vortex_strength(
                update_mask,
                np.asarray(update_increment, dtype=np.float64)[order],
            )
        if n_removed:
            vpm.remove_particles(particle_indices=remove_index.tolist())
        if n_injected:
            dtype = vpm.np_dtype
            vpm.add_vortex_particles(
                position=np.ascontiguousarray(target_position[inject], dtype=dtype),
                velocity=np.zeros((n_injected, 3), dtype=dtype),
                vortex_strength=np.ascontiguousarray(target_strength[inject], dtype=dtype),
                core_radius=np.full(n_injected, ratio * spacing, dtype=dtype),
                particle_volume=np.full(n_injected, spacing**3, dtype=dtype),
                kinematic_viscosity=np.full(n_injected, viscosity, dtype=dtype),
                eddy_viscosity=np.zeros(n_injected, dtype=dtype),
                group_id=np.zeros(n_injected, dtype=np.int32),
                zone_id=np.zeros(n_injected, dtype=np.int32),
            )
        actual_after = int(particles.n_particles_total)
        if actual_after != n_after:
            raise RuntimeError(
                f"VPM particle count after lattice blend is {actual_after}, expected {n_after}"
            )
    except Exception:
        _restore_particle_state(vpm, snapshot)
        raise
    removed_strength = existing_strength[remove]
    replaced_strength = existing_strength[managed]
    update_delta = (
        np.asarray(update_increment, dtype=np.float64).sum(axis=0, dtype=np.float64)
        if update_increment
        else np.zeros(3, dtype=np.float64)
    )
    mapped_net = target_strength.sum(axis=0, dtype=np.float64)
    actually_injected_strength = target_strength[inject]
    injected_net = actually_injected_strength.sum(axis=0, dtype=np.float64)
    state_change = (
        target_strength[inject].sum(axis=0, dtype=np.float64)
        + update_delta
        - removed_strength.sum(axis=0, dtype=np.float64)
    )
    existing_eta = replacement_eta(existing_position, transfer_box, eta_blend_width)
    blended = (existing_eta > tolerance) & (existing_eta < 1.0 - tolerance)
    return TransferResult(
        n_particles_before=n_before,
        n_particles_retained=n_before - n_removed,
        n_particles_removed=n_removed,
        n_particles_blended=int(np.count_nonzero(blended)),
        n_particles_injected=n_injected,
        n_particles_after=n_after,
        injected_vortex_strength_l1=float(np.linalg.norm(actually_injected_strength, axis=1).sum()),
        injected_vortex_strength_net=injected_net,
        replaced_vortex_strength_l1=float(np.linalg.norm(replaced_strength, axis=1).sum()),
        replaced_vortex_strength_net=replaced_strength.sum(axis=0, dtype=np.float64),
        state_change_vortex_strength_net=state_change,
        eta_blending_enabled=float(eta_blend_width) > 0.0,
        transfer_method="common_m4_lattice_blend",
        mapped_target_nodes=int(len(target_position)),
        excluded_solid_target_nodes=int(np.count_nonzero(target_solid)),
        excluded_solid_active_nodes=int(
            np.count_nonzero(np.linalg.norm(excluded_solid_strength, axis=1) > 0.0)
        ),
        excluded_solid_vortex_strength_net=excluded_solid_strength.sum(axis=0, dtype=np.float64),
        excluded_solid_vortex_strength_l1=float(
            np.linalg.norm(excluded_solid_strength, axis=1).sum(dtype=np.float64)
        ),
        excluded_solid_first_moment=first_vorticity_moment(
            state.position[target_solid], excluded_solid_strength
        ),
        mapped_first_moment=first_vorticity_moment(target_position, target_strength),
        mapped_target_vortex_strength_l1=float(np.linalg.norm(target_strength, axis=1).sum()),
        mapped_target_vortex_strength_net=mapped_net,
        maximum_mapped_vortex_strength=float(strength_magnitude[nonzero].max(initial=0.0)),
        fvm_donor_vortex_strength_net=state.fvm_donor_vortex_strength_net,
        fvm_mapped_vortex_strength_net=state.fvm_vortex_strength.sum(axis=0, dtype=np.float64),
        fvm_donor_first_moment=state.fvm_donor_first_moment,
        fvm_mapped_first_moment=first_vorticity_moment(state.position, state.fvm_vortex_strength),
        blend_cross_divergence_l2_before=state.cross_divergence_l2_before,
        blend_cross_divergence_l2_after=state.cross_divergence_l2_after,
        blend_cross_divergence_relative=state.cross_divergence_relative,
        mapped_vorticity_divergence_error=(
            None if health is None else health["vorticity_divergence_error"]
        ),
        mapped_vortex_strength_misalignment_degrees=(
            None if health is None else health["vortex_strength_misalignment_degrees"]
        ),
        mapped_mean_overlap_ratio=(None if health is None else health["mean_overlap_ratio"]),
    )


def _transfer_log_record(step: int, result: TransferResult) -> str:
    rows: list[log_style.Row] = [
        ("method", result.transfer_method),
        ("blend, eta", "on" if result.eta_blending_enabled else "off"),
        ("particles, before", f"{result.n_particles_before:,}"),
        ("particles, removed", f"{result.n_particles_removed:,}"),
        ("particles, blended", f"{result.n_particles_blended:,}"),
        ("particles, injected", f"{result.n_particles_injected:,}"),
        ("particles, after", f"{result.n_particles_after:,}"),
        ("lattice nodes, active", f"{result.mapped_target_nodes:,}"),
        ("solid nodes, excluded", f"{result.excluded_solid_active_nodes:,}"),
        (
            "solid strength, excluded l1",
            f"{result.excluded_solid_vortex_strength_l1:.3e}",
            "m^3/s",
        ),
        (
            "fvm map, net error",
            f"{float(np.linalg.norm(result.fvm_mapped_vortex_strength_net - result.fvm_donor_vortex_strength_net)):.3e}",
            "m^3/s",
        ),
        (
            "fvm map, first-moment error",
            f"{float(np.linalg.norm(result.fvm_mapped_first_moment - result.fvm_donor_first_moment)):.3e}",
            "m^4/s",
        ),
        ("vortex strength replaced, l1", f"{result.replaced_vortex_strength_l1:.3e}", "m^3/s"),
        (
            "vortex strength mapped, l1",
            f"{result.mapped_target_vortex_strength_l1:.3e}",
            "m^3/s",
        ),
        ("vortex strength born, l1", f"{result.injected_vortex_strength_l1:.3e}", "m^3/s"),
        ("vortex strength, max node", f"{result.maximum_mapped_vortex_strength:.3e}", "m^3/s"),
        (
            "vortex strength, net change",
            f"{float(np.linalg.norm(result.state_change_vortex_strength_net)):.3e}",
            "m^3/s",
        ),
    ]
    if result.mapped_vorticity_divergence_error is not None:
        rows.extend(
            (
                (
                    "particle field, divergence error",
                    f"{result.mapped_vorticity_divergence_error:.3e}",
                ),
                (
                    "particle field, misalignment",
                    f"{result.mapped_vortex_strength_misalignment_degrees:.3f}",
                    "deg",
                ),
                (
                    "particle field, mean overlap",
                    f"{result.mapped_mean_overlap_ratio:.3f}",
                ),
            )
        )
    if result.transfer_method == "projected_gbd_renewal":
        velocity_error = result.projection_velocity_relative_error
        rows.extend(
            (
                (
                    "projection error, omega",
                    f"{result.projection_vorticity_relative_error:.3e}",
                ),
                (
                    "projection error, normal velocity",
                    f"{0.0 if velocity_error is None else velocity_error:.3e}",
                ),
                ("projection condition number", f"{result.projection_condition_number:.3e}"),
                ("gbd guard width", f"{result.renewal_guard_width:.4g}", "m"),
                ("gbd diffusion substeps", result.renewal_diffusion_substeps),
                ("selective support births", f"{result.selective_support_births:,}"),
            )
        )
    if result.transfer_method == "buffered_m4_renewal":
        rows.extend(
            (
                ("renewal belt, input", f"{result.renewed_input_particles:,}"),
                ("renewal belt, output", f"{result.renewed_output_particles:,}"),
                ("outer wake, preserved", f"{result.preserved_outer_particles:,}"),
                ("lattice nodes, pruned", f"{result.pruned_lattice_nodes:,}"),
                (
                    "pruned strength, fraction",
                    f"{100.0 * result.pruned_vortex_strength_fraction:.3f}",
                    "%",
                ),
                ("renewal CFL", f"{result.renewal_cfl:.3f}"),
                (
                    "renewal conservation, strength",
                    f"{result.renewal_conservation_error:.3e}",
                ),
                (
                    "renewal conservation, impulse",
                    f"{result.renewal_linear_impulse_error:.3e}",
                ),
                (
                    "transfer amplification, max",
                    f"{result.maximum_transfer_amplification:.3f}",
                ),
            )
        )
        if result.representation_residual_after_prune is not None:
            rows.extend(
                (
                    (
                        "representation residual, before prune",
                        f"{0.0 if result.representation_residual_before_prune is None else result.representation_residual_before_prune:.3e}",
                    ),
                    (
                        "representation residual, after prune",
                        f"{result.representation_residual_after_prune:.3e}",
                    ),
                )
            )
    return format_coupler_log(f"state replacement, step {step:,}", *rows)


class VorticityTransfer:
    """Synchronize the inner VPM cloud with the absolute FVM vorticity state."""

    def __init__(self, coupler):
        cfg = coupler.setup
        if coupler.kinematic_viscosity is None or coupler.fvm_box is None:
            raise RuntimeError("VorticityTransfer requires initialized FVM and VPM state")
        self.config = cfg
        self.transfer_method = str(cfg.transfer_method)
        candidate_vpm = getattr(coupler, "vpm_solver", None)
        if (
            self.transfer_method == "buffered_m4_renewal"
            and candidate_vpm is not None
            and str(getattr(candidate_vpm, "viscous_scheme", "")).upper() != "GBD"
        ):
            raise ValueError("buffered_m4_renewal currently requires the GBD viscous scheme")
        if not np.isfinite(coupler.vpm_core_radius_ratio):
            raise RuntimeError("VorticityTransfer requires the resolved VPM core-radius ratio")
        self.core_radius_ratio = float(coupler.vpm_core_radius_ratio)
        if not np.isfinite(coupler.vpm_particle_spacing):
            raise RuntimeError("VorticityTransfer requires the resolved VPM particle spacing")
        self.particle_spacing = float(coupler.vpm_particle_spacing)
        self.eta_blend_width = float(cfg.eta_blend_width)
        self.vpm_only_width = float(cfg.vpm_only_width)
        self.transfer_prune_threshold_abs = (
            float(cfg.transfer_vorticity_cutoff) * self.particle_spacing**3
        )
        self.transfer_boundary_prune_multiplier = float(cfg.transfer_boundary_prune_multiplier)
        self.transfer_amplification_cap = float(cfg.transfer_amplification_cap)
        self.kinematic_viscosity = float(coupler.kinematic_viscosity)
        coupling_time_step = getattr(coupler, "vpm_time_step_size", None)
        if coupling_time_step is None and self.transfer_method in {
            "buffered_m4_renewal",
            "common_lattice",
        }:
            raise RuntimeError("VorticityTransfer requires the resolved VPM time step")
        self.coupling_time_step = 0.0 if coupling_time_step is None else float(coupling_time_step)
        self.renewal_buffer_length = required_renewal_buffer_length(
            cfg.freestream_velocity,
            self.coupling_time_step,
            self.particle_spacing,
        )
        self.diagnostic_interval = int(cfg.transfer_diagnostic_interval_steps)
        self.discretization_error_limit = float(cfg.transfer_discretization_error_limit)
        self.renewal_vorticity_error_limit = float(cfg.renewal_vorticity_error_limit)
        self.renewal_velocity_error_limit = float(cfg.renewal_velocity_error_limit)
        self.renewal_gaussian_tail_cutoff = float(cfg.renewal_gaussian_tail_cutoff)
        self.renewal_solver_tolerance = float(cfg.renewal_solver_tolerance)
        self._fvm_box = np.asarray(coupler.fvm_box, dtype=np.float64)
        self._box: np.ndarray | None = None
        self._cell_tree: cKDTree | None = None
        self._cell_centre: np.ndarray | None = None
        self._cell_volume: np.ndarray | None = None
        self._velocity_trace: FVMVelocityInterpolator | None = None
        self._fvm_solid_mask: np.ndarray | None = None
        self._body_bounds: np.ndarray | None = None
        self._solid_bodies: tuple = ()
        self._lattice_anchor: np.ndarray | None = None
        self._renewal_lattice: RenewalLattice | None = None
        self._renewal_target_solid_mask: np.ndarray | None = None
        self._stable_renewal_lattice: StableRenewalLattice | None = None
        self._face_cells: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._authority_cell_mask: np.ndarray | None = None
        self.step = 0
        self.last_interface_flow: dict[str, float] = {}
        self.last_vortex_line_closure: dict[str, float] = {}

    @staticmethod
    def _vorticity_from_gradient(gradient: np.ndarray) -> np.ndarray:
        """Curl for the FVM layout ``G[i,j] = d(u_j)/d(x_i)``."""
        g = np.asarray(gradient, dtype=np.float64).reshape(-1, 3, 3)
        return np.stack(
            [
                g[:, 1, 2] - g[:, 2, 1],
                g[:, 2, 0] - g[:, 0, 2],
                g[:, 0, 1] - g[:, 1, 0],
            ],
            axis=1,
        )

    def _points_in_solid(self, points, *, include_boundary: bool) -> np.ndarray:
        query = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        inside = np.zeros(len(query), dtype=bool)
        for body in self._solid_bodies:
            inside |= np.asarray(
                body.contains(query, include_boundary=include_boundary), dtype=bool
            ).reshape(-1)
        if self._body_bounds is not None:
            lower = self._body_bounds[[0, 2, 4]]
            upper = self._body_bounds[[1, 3, 5]]
            comparison = (query >= lower) & (query <= upper)
            if not include_boundary:
                comparison = (query > lower) & (query < upper)
            inside |= np.all(comparison, axis=1)
        return inside

    def _signed_solid_distance(self, points: np.ndarray) -> np.ndarray:
        """Approximate signed distance, positive in fluid and negative in solid."""
        query = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        distance = np.full(len(query), np.inf, dtype=np.float64)
        for body in self._solid_bodies:
            distance = np.minimum(
                distance,
                np.asarray(body.signed_distance(query), dtype=np.float64).reshape(-1),
            )
        if self._body_bounds is not None:
            lower = self._body_bounds[[0, 2, 4]]
            upper = self._body_bounds[[1, 3, 5]]
            outward = np.maximum(lower - query, query - upper)
            outside_distance = np.linalg.norm(np.maximum(outward, 0.0), axis=1)
            inside_depth = np.max(outward, axis=1)
            box_distance = np.where(outside_distance > 0.0, outside_distance, inside_depth)
            distance = np.minimum(distance, box_distance)
        return distance

    def _face_cell_index(self, bounds: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        faces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if self._cell_centre is None:
            return faces
        centres = self._cell_centre
        scale = (
            np.cbrt(self._cell_volume) if self._cell_volume is not None else np.zeros(len(centres))
        )
        for axis in range(3):
            for side, (bound, sign) in enumerate(
                ((bounds[2 * axis], -1.0), (bounds[2 * axis + 1], 1.0))
            ):
                inside = np.ones(len(centres), dtype=bool)
                for other in range(3):
                    if other != axis:
                        inside &= (centres[:, other] >= bounds[2 * other]) & (
                            centres[:, other] <= bounds[2 * other + 1]
                        )
                index = np.flatnonzero(inside & (np.abs(centres[:, axis] - bound) <= scale))
                if index.size:
                    normal = np.zeros(3)
                    normal[axis] = sign
                    name = f"{'xyz'[axis]}{'min' if side == 0 else 'max'}"
                    faces[name] = (index, normal)
        return faces

    def _build_face_cell_index(self) -> None:
        self._face_cells = {} if self._box is None else self._face_cell_index(self._box)

    def check_interface_flow(self, velocity: np.ndarray) -> dict[str, float]:
        values = np.asarray(velocity, dtype=np.float64).reshape(-1, 3)
        return {
            name: float(np.mean(values[index] @ normal))
            for name, (index, normal) in self._face_cells.items()
            if index.max(initial=-1) < len(values)
        }

    def check_vortex_line_closure(self, velocity_gradient: np.ndarray) -> dict[str, float]:
        vorticity = self._vorticity_from_gradient(velocity_gradient)
        scale = float(np.linalg.norm(vorticity, axis=1).mean()) + np.finfo(float).tiny
        return {
            name: float(np.mean(np.abs(vorticity[index] @ normal)) / scale)
            for name, (index, normal) in self._face_cells.items()
            if index.max(initial=-1) < len(vorticity)
        }

    def setup(self, fvm) -> None:
        self._box = np.asarray(
            self.config.transfer_region_bounds or self._fvm_box, dtype=np.float64
        )
        self._cell_centre = np.asarray(fvm.get_cell_centre_coordinates(), dtype=np.float64).reshape(
            -1, 3
        )
        self._cell_volume = np.asarray(fvm.get_cell_volume(), dtype=np.float64).reshape(-1)

        # These partitioned getters are collective, even though only rank zero
        # receives the assembled arrays. Keep their call order identical.
        wall_patches = [
            boundary_condition.name
            for boundary_condition in fvm.setup.boundaries
            if boundary_condition.mesh_type == "wall"
        ]
        wall_faces = None
        if len(wall_patches) == 1:
            wall_faces = np.asarray(
                fvm.get_boundary_face_centre_coordinates(wall_patches[0]), dtype=np.float64
            ).reshape(-1, 3)

        if len(self._cell_centre) == 0:
            self._build_face_cell_index()
            return
        if len(self._cell_volume) != len(self._cell_centre):
            raise RuntimeError("FVM cell-centre and cell-volume counts do not match")
        _validate_particle_sources(
            self._cell_centre,
            self._cell_volume,
            np.zeros_like(self._cell_centre),
        )
        if self.transfer_method == "buffered_m4_renewal":
            self._cell_tree = cKDTree(self._cell_centre)
            self._velocity_trace = FVMVelocityInterpolator(
                self._cell_centre,
                self._cell_tree,
                neighbour_count=4,
            )

        if wall_faces is not None and len(wall_faces):
            bounds = np.array(
                [
                    wall_faces[:, 0].min(),
                    wall_faces[:, 0].max(),
                    wall_faces[:, 1].min(),
                    wall_faces[:, 1].max(),
                    wall_faces[:, 2].min(),
                    wall_faces[:, 2].max(),
                ]
            )
            on_planes = np.zeros(len(wall_faces), dtype=bool)
            for axis in range(3):
                on_planes |= np.isclose(wall_faces[:, axis], bounds[2 * axis], atol=1.0e-9)
                on_planes |= np.isclose(wall_faces[:, axis], bounds[2 * axis + 1], atol=1.0e-9)
            if on_planes.all():
                self._body_bounds = bounds
                self._lattice_anchor = bounds[[0, 2, 4]]
                if self.transfer_method == "buffered_m4_renewal":
                    # This is the lattice phase used by the 20-second renewal
                    # run: no particle centre lies directly on a cube face.
                    self._lattice_anchor -= 0.5 * self.particle_spacing

        ibm = getattr(fvm, "ibm", None)
        bodies = () if ibm is None else tuple(ibm.bodies)
        self._solid_bodies = tuple(body for body in bodies if body.has_solid_geometry)
        if self._solid_bodies:
            self._body_bounds = None
            self._lattice_anchor = self._cell_centre[0].copy()
        if self._lattice_anchor is None:
            self._lattice_anchor = self._cell_centre[0].copy()

        if self.transfer_method == "buffered_m4_renewal":
            if self._cell_tree is None:
                raise RuntimeError("buffered M4' renewal requires an FVM cell tree")

            def mesh_weight_at_node(points: np.ndarray) -> np.ndarray:
                distance, _ = self._cell_tree.query(points, workers=-1)
                return 1.0 - _smoothstep(
                    distance,
                    self.particle_spacing,
                    2.0 * self.particle_spacing,
                )

            has_solid = bool(self._solid_bodies) or self._body_bounds is not None

            def fluid_weight_at_node(points: np.ndarray) -> np.ndarray:
                return _smoothstep(
                    self._signed_solid_distance(points),
                    -self.particle_spacing,
                    0.0,
                )

            def interior_at_node(points: np.ndarray) -> np.ndarray:
                return self._points_in_solid(points, include_boundary=False)

            self._stable_renewal_lattice = build_stable_renewal_lattice(
                self._box,
                self.particle_spacing,
                buffer_length=self.renewal_buffer_length,
                authority_ramp_width=self.eta_blend_width,
                vpm_dead_zone=self.vpm_only_width,
                lattice_anchor=self._lattice_anchor,
                mesh_weight_at_node=mesh_weight_at_node,
                fluid_weight_at_node=fluid_weight_at_node if has_solid else None,
                interior_at_node=interior_at_node if has_solid else None,
            )
        else:
            renewal_bounds = self._box.copy()
            renewal_bounds[::2] -= self.renewal_buffer_length
            renewal_bounds[1::2] += self.renewal_buffer_length
            self._renewal_lattice = build_renewal_lattice(
                renewal_bounds,
                lattice_anchor=self._lattice_anchor,
                spacing=self.particle_spacing,
            )
            self._renewal_target_solid_mask = self._points_in_solid(
                self._renewal_lattice.position,
                include_boundary=False,
            )
        self._fvm_solid_mask = self._points_in_solid(
            self._cell_centre,
            include_boundary=True,
        )
        donor_eta = replacement_eta(self._cell_centre, self._box, 0.0)
        self._authority_cell_mask = (donor_eta > 0.0) & ~self._fvm_solid_mask
        donor_count = int(np.count_nonzero(self._authority_cell_mask))
        if donor_count == 0:
            raise ValueError("FVM transfer region contains no fluid cell centres")
        self._build_face_cell_index()
        lattice_count = (
            len(self._stable_renewal_lattice.positions)
            if self._stable_renewal_lattice is not None
            else (0 if self._renewal_lattice is None else len(self._renewal_lattice.position))
        )
        logger.info(
            format_coupler_log(
                "replacement region",
                ("method", self.transfer_method),
                ("fvm fluid cells", f"{donor_count:,}"),
                *(
                    (("authority ramp width, eta", "off"),)
                    if self.eta_blend_width == 0.0
                    else (("authority ramp width, eta", f"{self.eta_blend_width:.4g}", "m"),)
                ),
                ("renewal buffer", f"{self.renewal_buffer_length:.4g}", "m"),
                ("renewal lattice nodes", f"{lattice_count:,}"),
                (
                    "state",
                    (
                        "synchronized fvm velocity trace"
                        if self.transfer_method == "buffered_m4_renewal"
                        else "cell volume x fvm vorticity"
                    ),
                ),
            )
        )

    @staticmethod
    def _bounded_sample(index: np.ndarray, maximum_count: int) -> np.ndarray:
        """Return a deterministic, domain-spanning subset of integer indices."""
        values = np.asarray(index, dtype=np.int64).reshape(-1)
        if len(values) <= maximum_count:
            return values
        selection = np.linspace(0, len(values) - 1, maximum_count, dtype=np.int64)
        return values[selection]

    def _renewal_vorticity_floor(self, vpm, vorticity: np.ndarray) -> float:
        """Convert the configured absolute GBD strength floor to vorticity."""
        magnitude = np.linalg.norm(np.asarray(vorticity, dtype=np.float64), axis=1)
        peak = float(magnitude.max(initial=0.0))
        numerical_floor = 128.0 * np.finfo(np.float64).eps * max(peak, 1.0)
        viscous = getattr(getattr(vpm, "setup", None), "viscous", None)
        mode = str(getattr(viscous, "gbd_threshold_mode", "budget")).lower()
        threshold = float(getattr(viscous, "gbd_threshold", 0.0))
        if mode == "absolute" and np.isfinite(threshold) and threshold > 0.0:
            return max(numerical_floor, threshold / self.particle_spacing**3)
        return max(numerical_floor, 1.0e-8 * peak)

    def _gbd_renewal_bounds(self, vpm) -> tuple[np.ndarray, float, int]:
        """Expand the FVM-fit box by the numerical reach of the last GBD step."""
        if self._box is None:
            raise RuntimeError("projected renewal requires initialized authority bounds")
        last_substeps = int(
            getattr(getattr(vpm, "physics", None), "last_gbd_diffusion_substeps", 1)
        )
        predicted_substeps = 1
        counter = getattr(getattr(vpm, "physics", None), "gbd_diffusion_substep_count", None)
        particles = getattr(vpm, "particles", None)
        if callable(counter) and particles is not None:
            stage_counter = cast(Callable[[float, float, float], int], counter)
            count = int(particles.n_particles_total)
            effective_viscosity = (
                np.asarray(particles.effective_viscosity_cpu(), dtype=np.float64).reshape(-1)
                if count
                else np.empty(0, dtype=np.float64)
            )
            max_diffusivity = float(
                effective_viscosity[:count].max(initial=self.kinematic_viscosity)
            )
            predicted_substeps = int(
                stage_counter(
                    max_diffusivity,
                    float(vpm.time_step_size),
                    self.particle_spacing,
                )
            )
        diffusion_substeps = max(last_substeps, predicted_substeps)
        guard = gbd_guard_width(
            particle_spacing=self.particle_spacing,
            diffusion_substeps=diffusion_substeps,
        )
        renewal = self._box.copy()
        renewal[::2] -= guard
        renewal[1::2] += guard
        tolerance = 64.0 * np.finfo(np.float64).eps * np.maximum(1.0, np.abs(self._fvm_box))
        if np.any(renewal[::2] < self._fvm_box[::2] - tolerance[::2]) or np.any(
            renewal[1::2] > self._fvm_box[1::2] + tolerance[1::2]
        ):
            raise RuntimeError(
                "projected renewal needs an FVM guard around transfer_region_bounds: "
                f"guard={guard:.6g} m from {diffusion_substeps} GBD diffusion substeps"
            )
        return renewal, guard, diffusion_substeps

    def _renewal_fit_and_verification_points(
        self,
        fvm_vorticity: np.ndarray,
        *,
        activity_floor: float,
        authority_cell_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Select disjoint FVM fit and verification cells across the belt."""
        if self._cell_centre is None or self._fvm_solid_mask is None:
            raise RuntimeError("projected renewal requires prepared FVM authority cells")
        authority = np.flatnonzero(np.asarray(authority_cell_mask, dtype=bool))
        if not len(authority):
            raise RuntimeError("projected renewal region contains no FVM fluid cells")
        magnitude = np.linalg.norm(fvm_vorticity, axis=1)
        active = authority[magnitude[authority] >= activity_floor]
        if len(active) > 20_000:
            strongest = np.argpartition(magnitude[active], -12_000)[-12_000:]
            active = np.unique(
                np.concatenate((active[strongest], self._bounded_sample(active, 8_000)))
            )
        background = self._bounded_sample(
            authority[magnitude[authority] < activity_floor],
            4_000,
        )
        pool = np.unique(np.concatenate((active, background)))
        fit_index = pool[::2]
        verification_index = pool[1::2]
        if not len(verification_index):
            verification_index = fit_index
        return (
            self._cell_centre[fit_index],
            fvm_vorticity[fit_index],
            self._cell_centre[verification_index],
            fvm_vorticity[verification_index],
        )

    def _sparse_particle_vorticity(
        self,
        evaluation_position: np.ndarray,
        particle_position: np.ndarray,
        vortex_strength: np.ndarray,
        core_radius: np.ndarray,
    ) -> np.ndarray:
        return evaluate_sparse_gaussian_vorticity(
            evaluation_position,
            particle_position,
            vortex_strength,
            core_radius,
            relative_tail_cutoff=self.renewal_gaussian_tail_cutoff,
        )

    @staticmethod
    def _field_relative_error(
        actual: np.ndarray,
        expected: np.ndarray,
        *,
        absolute_scale: float,
    ) -> float:
        residual_rms = float(np.sqrt(np.mean((actual - expected) ** 2)))
        expected_rms = float(np.sqrt(np.mean(expected**2)))
        return residual_rms / max(expected_rms, absolute_scale, np.finfo(float).tiny)

    def _solve_current_sparse_basis(
        self,
        *,
        collocation_position: np.ndarray,
        collocation_target: np.ndarray,
        solve_position: np.ndarray,
        solve_radius: np.ndarray,
        solve_prior: np.ndarray,
        preserved_position: np.ndarray,
        preserved_strength: np.ndarray,
        preserved_radius: np.ndarray,
    ) -> SparseRenewalProjectionResult:
        target = np.asarray(collocation_target, dtype=np.float64).reshape(-1, 3).copy()
        target -= self._sparse_particle_vorticity(
            collocation_position,
            preserved_position,
            preserved_strength,
            preserved_radius,
        )
        return solve_sparse_renewal_projection(
            collocation_position=collocation_position,
            target_vorticity=target,
            particle_position=solve_position,
            core_radius=solve_radius,
            prior_vortex_strength=solve_prior,
            prior_weight=0.0,
            relative_tail_cutoff=self.renewal_gaussian_tail_cutoff,
            relative_tolerance=self.renewal_solver_tolerance,
        )

    def _verify_sparse_projection(
        self,
        *,
        verification_position: np.ndarray,
        verification_target: np.ndarray,
        solve_position: np.ndarray,
        solve_strength: np.ndarray,
        solve_radius: np.ndarray,
        preserved_position: np.ndarray,
        preserved_strength: np.ndarray,
        preserved_radius: np.ndarray,
        activity_floor: float,
    ) -> float:
        actual = self._sparse_particle_vorticity(
            verification_position,
            solve_position,
            solve_strength,
            solve_radius,
        )
        actual += self._sparse_particle_vorticity(
            verification_position,
            preserved_position,
            preserved_strength,
            preserved_radius,
        )
        return self._field_relative_error(
            actual,
            verification_target,
            absolute_scale=activity_floor,
        )

    def _select_projected_support_births(
        self,
        *,
        fvm_vorticity: np.ndarray,
        renewal_bounds: np.ndarray,
        authority_cell_mask: np.ndarray,
        existing_position: np.ndarray,
        solve_position: np.ndarray,
        solve_strength: np.ndarray,
        solve_radius: np.ndarray,
        preserved_position: np.ndarray,
        preserved_strength: np.ndarray,
        preserved_radius: np.ndarray,
        activity_floor: float,
        capacity_remaining: int,
    ) -> np.ndarray:
        if (
            self._box is None
            or self._cell_centre is None
            or self._lattice_anchor is None
            or capacity_remaining <= 0
        ):
            return np.empty((0, 3), dtype=np.float64)
        magnitude = np.linalg.norm(fvm_vorticity, axis=1)
        active = np.asarray(authority_cell_mask, dtype=bool) & (magnitude >= activity_floor)
        if not np.any(active):
            return np.empty((0, 3), dtype=np.float64)
        base_lattice_index = np.rint(
            (self._cell_centre[active] - self._lattice_anchor) / self.particle_spacing
        ).astype(np.int64)
        base_lattice_index = np.unique(base_lattice_index, axis=0)
        neighbour_offsets = np.array(
            [
                [0, 0, 0],
                [-1, 0, 0],
                [1, 0, 0],
                [0, -1, 0],
                [0, 1, 0],
                [0, 0, -1],
                [0, 0, 1],
            ],
            dtype=np.int64,
        )
        lattice_index = np.unique(
            (base_lattice_index[:, None, :] + neighbour_offsets[None, :, :]).reshape(-1, 3),
            axis=0,
        )
        candidate = self._lattice_anchor + self.particle_spacing * lattice_index
        candidate = candidate[
            geometric_renewal_mask(
                candidate,
                renewal_bounds,
                particle_spacing=self.particle_spacing,
            )
        ]
        candidate = candidate[~self._points_in_solid(candidate, include_boundary=True)]
        if not len(candidate):
            return candidate

        active_index = np.flatnonzero(active)
        if len(active_index) > 30_000:
            strongest = np.argpartition(magnitude[active_index], -20_000)[-20_000:]
            active_index = np.unique(
                np.concatenate(
                    (active_index[strongest], self._bounded_sample(active_index, 10_000))
                )
            )
        residual_position = self._cell_centre[active_index]
        target = fvm_vorticity[active_index]
        represented = self._sparse_particle_vorticity(
            residual_position,
            solve_position,
            solve_strength,
            solve_radius,
        )
        represented += self._sparse_particle_vorticity(
            residual_position,
            preserved_position,
            preserved_strength,
            preserved_radius,
        )
        residual = target - represented
        return select_residual_support_positions(
            candidate_position=candidate,
            existing_position=existing_position,
            collocation_position=residual_position,
            vorticity_residual=residual,
            renewal_bounds=renewal_bounds,
            particle_spacing=self.particle_spacing,
            residual_fraction=1.0e-3,
            maximum_births=capacity_remaining,
        )

    def _projected_boundary_velocity_error(
        self,
        vpm,
        fvm_velocity: np.ndarray,
        *,
        renewal_bounds: np.ndarray,
    ) -> float | None:
        position_parts: list[np.ndarray] = []
        normal_parts: list[np.ndarray] = []
        target_parts: list[np.ndarray] = []
        if self._cell_centre is None:
            return None
        for index, normal in self._face_cell_index(renewal_bounds).values():
            position_parts.append(self._cell_centre[index])
            normal_parts.append(np.tile(normal, (len(index), 1)))
            target_parts.append(fvm_velocity[index])
        if not position_parts:
            return None
        position = np.vstack(position_parts)
        normal = np.vstack(normal_parts)
        target = np.vstack(target_parts)
        refresh = getattr(vpm, "refresh_boundary_element_solution", None)
        if callable(refresh):
            refresh()
        actual = np.asarray(
            vpm.compute_velocity_at_points(position, include_freestream=True, include_body=True),
            dtype=np.float64,
        ).reshape(-1, 3)
        actual_normal = np.einsum("ij,ij->i", actual, normal)
        target_normal = np.einsum("ij,ij->i", target, normal)
        return self._field_relative_error(
            actual_normal,
            target_normal,
            absolute_scale=max(float(np.linalg.norm(self.config.freestream_velocity)), 1.0e-12),
        )

    def _write_projection_failure_oracle(
        self,
        vpm,
        *,
        renewal_bounds: np.ndarray,
        authority_cell_mask: np.ndarray,
        fvm_velocity: np.ndarray,
        fvm_vorticity: np.ndarray,
        fit_position: np.ndarray,
        fit_target: np.ndarray,
        verification_position: np.ndarray,
        verification_target: np.ndarray,
        solve_position: np.ndarray,
        solve_radius: np.ndarray,
        solve_prior: np.ndarray,
        solve_strength: np.ndarray,
        birth_position: np.ndarray,
        fit_error: float,
        verification_error: float,
        condition_number: float,
        iteration_count: int,
    ) -> Path:
        """Persist the exact failed cube field and basis for an offline oracle."""
        if self._cell_centre is None or self._fvm_solid_mask is None:
            raise RuntimeError("cannot export an uninitialized projected-renewal oracle")
        path = Path(vpm.case_dir) / "renewal_projection_failure_oracle.npz"
        np.savez(
            path,
            fvm_position=self._cell_centre,
            fvm_velocity=np.asarray(fvm_velocity, dtype=np.float64),
            fvm_vorticity=np.asarray(fvm_vorticity, dtype=np.float64),
            fvm_solid_mask=self._fvm_solid_mask,
            authority_cell_mask=np.asarray(authority_cell_mask, dtype=bool),
            renewal_bounds=np.asarray(renewal_bounds, dtype=np.float64),
            body_bounds=(
                np.empty(0, dtype=np.float64)
                if self._body_bounds is None
                else np.asarray(self._body_bounds, dtype=np.float64)
            ),
            lattice_anchor=(
                np.empty(0, dtype=np.float64)
                if self._lattice_anchor is None
                else np.asarray(self._lattice_anchor, dtype=np.float64)
            ),
            fit_position=np.asarray(fit_position, dtype=np.float64),
            fit_target=np.asarray(fit_target, dtype=np.float64),
            verification_position=np.asarray(verification_position, dtype=np.float64),
            verification_target=np.asarray(verification_target, dtype=np.float64),
            solve_position=np.asarray(solve_position, dtype=np.float64),
            solve_radius=np.asarray(solve_radius, dtype=np.float64),
            solve_prior=np.asarray(solve_prior, dtype=np.float64),
            solve_strength=np.asarray(solve_strength, dtype=np.float64),
            birth_position=np.asarray(birth_position, dtype=np.float64),
            particle_spacing=np.asarray(self.particle_spacing),
            core_radius_ratio=np.asarray(self.core_radius_ratio),
            activity_floor=np.asarray(self._renewal_vorticity_floor(vpm, fvm_vorticity)),
            fit_error=np.asarray(fit_error),
            verification_error=np.asarray(verification_error),
            condition_number=np.asarray(condition_number),
            iteration_count=np.asarray(iteration_count),
        )
        logger.error("projected-renewal failure oracle written to %s", path)
        return path

    def _transfer_projected_gbd(
        self,
        vpm,
        *,
        fvm_velocity: np.ndarray,
        fvm_vorticity: np.ndarray,
    ) -> TransferResult:
        if getattr(vpm, "viscous_scheme", None) != "GBD":
            raise RuntimeError("projected_renewal currently requires the GBD viscous scheme")
        if self._box is None or self._cell_centre is None or self._fvm_solid_mask is None:
            raise RuntimeError("projected renewal requires initialized authority bounds")
        renewal_bounds, guard_width, diffusion_substeps = self._gbd_renewal_bounds(vpm)
        authority_cell_mask = (
            replacement_eta(self._cell_centre, renewal_bounds, 0.0) > 0.0
        ) & ~self._fvm_solid_mask
        particles = vpm.particles
        n_before = int(particles.n_particles_total)
        position = np.asarray(particles.position_cpu(), dtype=np.float64).reshape(-1, 3)
        strength = np.asarray(particles.vortex_strength_cpu(), dtype=np.float64).reshape(-1, 3)
        radius = np.asarray(particles.core_radius_cpu(), dtype=np.float64).reshape(-1)
        renewable = geometric_renewal_mask(
            position,
            renewal_bounds,
            particle_spacing=self.particle_spacing,
        )
        preserved = ~renewable
        activity_floor = self._renewal_vorticity_floor(vpm, fvm_vorticity)
        (
            fit_position,
            fit_target,
            verification_position,
            verification_target,
        ) = self._renewal_fit_and_verification_points(
            fvm_vorticity,
            activity_floor=activity_floor,
            authority_cell_mask=authority_cell_mask,
        )

        solve_position = position[renewable]
        solve_radius = radius[renewable]
        solve_prior = strength[renewable]
        preserved_position = position[preserved]
        preserved_strength = strength[preserved]
        preserved_radius = radius[preserved]
        sparse_result = self._solve_current_sparse_basis(
            collocation_position=fit_position,
            collocation_target=fit_target,
            solve_position=solve_position,
            solve_radius=solve_radius,
            solve_prior=solve_prior,
            preserved_position=preserved_position,
            preserved_strength=preserved_strength,
            preserved_radius=preserved_radius,
        )
        fit_error = float(sparse_result.vorticity_relative_error)
        verification_error = self._verify_sparse_projection(
            verification_position=verification_position,
            verification_target=verification_target,
            solve_position=solve_position,
            solve_strength=sparse_result.vortex_strength,
            solve_radius=solve_radius,
            preserved_position=preserved_position,
            preserved_strength=preserved_strength,
            preserved_radius=preserved_radius,
            activity_floor=activity_floor,
        )
        sparse_result = replace(
            sparse_result,
            vorticity_relative_error=verification_error,
        )

        birth_position = np.empty((0, 3), dtype=np.float64)
        used_births = False
        if verification_error > self.renewal_vorticity_error_limit:
            birth_position = self._select_projected_support_births(
                fvm_vorticity=fvm_vorticity,
                renewal_bounds=renewal_bounds,
                authority_cell_mask=authority_cell_mask,
                existing_position=position,
                solve_position=solve_position,
                solve_strength=sparse_result.vortex_strength,
                solve_radius=solve_radius,
                preserved_position=preserved_position,
                preserved_strength=preserved_strength,
                preserved_radius=preserved_radius,
                activity_floor=activity_floor,
                capacity_remaining=int(particles.capacity) - n_before,
            )
            if len(birth_position):
                birth_radius = np.full(
                    len(birth_position),
                    self.core_radius_ratio * self.particle_spacing,
                )
                solve_position = np.vstack((solve_position, birth_position))
                solve_radius = np.concatenate((solve_radius, birth_radius))
                solve_prior = np.vstack((solve_prior, np.zeros((len(birth_position), 3))))
                sparse_result = self._solve_current_sparse_basis(
                    collocation_position=fit_position,
                    collocation_target=fit_target,
                    solve_position=solve_position,
                    solve_radius=solve_radius,
                    solve_prior=solve_prior,
                    preserved_position=preserved_position,
                    preserved_strength=preserved_strength,
                    preserved_radius=preserved_radius,
                )
                fit_error = float(sparse_result.vorticity_relative_error)
                verification_error = self._verify_sparse_projection(
                    verification_position=verification_position,
                    verification_target=verification_target,
                    solve_position=solve_position,
                    solve_strength=sparse_result.vortex_strength,
                    solve_radius=solve_radius,
                    preserved_position=preserved_position,
                    preserved_strength=preserved_strength,
                    preserved_radius=preserved_radius,
                    activity_floor=activity_floor,
                )
                sparse_result = replace(
                    sparse_result,
                    vorticity_relative_error=verification_error,
                )
                used_births = True

        if verification_error > self.renewal_vorticity_error_limit:
            oracle_path = self._write_projection_failure_oracle(
                vpm,
                renewal_bounds=renewal_bounds,
                authority_cell_mask=authority_cell_mask,
                fvm_velocity=fvm_velocity,
                fvm_vorticity=fvm_vorticity,
                fit_position=fit_position,
                fit_target=fit_target,
                verification_position=verification_position,
                verification_target=verification_target,
                solve_position=solve_position,
                solve_radius=solve_radius,
                solve_prior=solve_prior,
                solve_strength=sparse_result.vortex_strength,
                birth_position=birth_position,
                fit_error=fit_error,
                verification_error=verification_error,
                condition_number=sparse_result.condition_number,
                iteration_count=sparse_result.iteration_count,
            )
            raise RuntimeError(
                "projected renewal failed its independent vorticity gate: "
                f"error={verification_error:.6e}, "
                f"fit_error={fit_error:.6e}, "
                f"limit={self.renewal_vorticity_error_limit:.6e}, "
                f"renewable={int(np.count_nonzero(renewable))}, "
                f"births={len(birth_position)}, "
                f"oracle={oracle_path}"
            )
        if not sparse_result.converged:
            raise RuntimeError("projected renewal sparse LSMR solve did not converge")

        renewable_count = int(np.count_nonzero(renewable))
        updated = strength.copy()
        updated[renewable] = sparse_result.vortex_strength[:renewable_count]
        projection = GBDRenewalProjectionResult(
            renewable_mask=renewable,
            preserved_mask=preserved,
            updated_vortex_strength=updated,
            birth_position=birth_position,
            birth_vortex_strength=sparse_result.vortex_strength[renewable_count:],
            birth_core_radius=solve_radius[renewable_count:],
            projection=sparse_result,
            used_selective_births=used_births,
        )
        snapshot = _particle_state_snapshot(vpm)
        result = apply_projected_gbd_renewal(
            vpm,
            projection,
            particle_spacing=self.particle_spacing,
            kinematic_viscosity=self.kinematic_viscosity,
        )
        velocity_error = self._projected_boundary_velocity_error(
            vpm,
            fvm_velocity,
            renewal_bounds=renewal_bounds,
        )
        if velocity_error is not None and velocity_error > self.renewal_velocity_error_limit:
            _restore_particle_state(vpm, snapshot)
            refresh = getattr(vpm, "refresh_boundary_element_solution", None)
            if callable(refresh):
                refresh()
            raise RuntimeError(
                "projected renewal failed its boundary normal-velocity gate: "
                f"error={velocity_error:.6e}, "
                f"limit={self.renewal_velocity_error_limit:.6e}"
            )
        return replace(
            result,
            projection_velocity_relative_error=velocity_error,
            renewal_guard_width=guard_width,
            renewal_diffusion_substeps=diffusion_substeps,
        )

    def _transfer_buffered_m4_renewal(
        self,
        vpm,
        *,
        fvm_velocity: np.ndarray,
        fvm_velocity_gradient: np.ndarray,
    ) -> TransferResult:
        """Run the recovered synchronized velocity-trace whole-belt renewal."""
        if self._stable_renewal_lattice is None or self._velocity_trace is None:
            raise RuntimeError("buffered M4' renewal lattice was not initialized")
        has_solid = bool(self._solid_bodies) or self._body_bounds is not None

        def fluid_weight(points: np.ndarray) -> np.ndarray:
            return _smoothstep(
                self._signed_solid_distance(points),
                -self.particle_spacing,
                0.0,
            )

        def in_solid(points: np.ndarray) -> np.ndarray:
            return self._points_in_solid(points, include_boundary=False)

        def velocity_at(points: np.ndarray) -> np.ndarray:
            query = np.asarray(points, dtype=np.float64).reshape(-1, 3)
            sampled = self._velocity_trace.sample(
                query,
                fvm_velocity,
                fvm_velocity_gradient,
            )
            if has_solid:
                sampled *= _smoothstep(
                    self._signed_solid_distance(query),
                    0.0,
                    self.particle_spacing,
                )[:, None]
            return sampled

        def target_at_node(points: np.ndarray) -> np.ndarray:
            return vortex_strength_from_velocity_trace(
                points,
                self.particle_spacing,
                velocity_at,
            )

        return replace_particles_from_buffered_m4_renewal(
            vpm,
            lattice=self._stable_renewal_lattice,
            fvm_vortex_strength_at_node=target_at_node,
            particle_fluid_weight=fluid_weight if has_solid else None,
            particle_in_solid=in_solid if has_solid else None,
            prune_threshold=self.transfer_prune_threshold_abs,
            core_radius_ratio=self.core_radius_ratio,
            amplification_cap=self.transfer_amplification_cap,
            boundary_prune_multiplier=self.transfer_boundary_prune_multiplier,
            kinematic_viscosity=self.kinematic_viscosity,
            freestream_speed=float(np.linalg.norm(self.config.freestream_velocity_vector)),
            time_step_size=self.coupling_time_step,
            compute_diagnostics=(self.step == 1 or self.step % self.diagnostic_interval == 0),
        )

    def transfer(self, vpm, velocity, velocity_gradient) -> TransferResult:
        """Replace the inner particle state and preserve the outer particle cloud."""
        self.step += 1
        if self._box is None or self._cell_centre is None or self._cell_volume is None:
            raise RuntimeError("VorticityTransfer.setup() has not prepared the FVM donor cells")
        velocity_values = np.asarray(velocity, dtype=np.float64).reshape(-1, 3)
        gradient_values = np.asarray(velocity_gradient, dtype=np.float64).reshape(-1, 3, 3)
        if len(velocity_values) != len(self._cell_centre) or len(gradient_values) != len(
            self._cell_centre
        ):
            raise ValueError("FVM velocity, gradient, and cell-centre counts must match")

        self.last_interface_flow = self.check_interface_flow(velocity_values)
        self.last_vortex_line_closure = self.check_vortex_line_closure(gradient_values)
        if self._lattice_anchor is None:
            raise RuntimeError("VorticityTransfer.setup() did not resolve a lattice anchor")
        fvm_vorticity = self._vorticity_from_gradient(gradient_values)
        if self.transfer_method == "buffered_m4_renewal":
            result = self._transfer_buffered_m4_renewal(
                vpm,
                fvm_velocity=velocity_values,
                fvm_velocity_gradient=gradient_values,
            )
        elif self.transfer_method == "projected_renewal":
            result = self._transfer_projected_gbd(
                vpm,
                fvm_velocity=velocity_values,
                fvm_vorticity=fvm_vorticity,
            )
        else:
            if self._renewal_lattice is None or self._renewal_target_solid_mask is None:
                raise RuntimeError("VorticityTransfer.setup() did not build the renewal lattice")
            result = replace_particles_from_lattice_blend(
                vpm,
                transfer_box=self._box,
                eta_blend_width=self.eta_blend_width,
                fvm_position=self._cell_centre,
                fvm_cell_volume=self._cell_volume,
                fvm_vorticity=fvm_vorticity,
                lattice_anchor=self._lattice_anchor,
                particle_spacing=self.particle_spacing,
                core_radius_ratio=self.core_radius_ratio,
                kinematic_viscosity=self.kinematic_viscosity,
                fvm_solid_mask=self._fvm_solid_mask,
                renewal_lattice=self._renewal_lattice,
                target_solid_mask=self._renewal_target_solid_mask,
                compute_divergence_diagnostic=(
                    self.step == 1 or self.step % self.diagnostic_interval == 0
                ),
                discretization_error_limit=self.discretization_error_limit,
            )
        if self.step % self.diagnostic_interval == 0:
            logger.info(_transfer_log_record(self.step, result))
        return result


__all__ = [
    "TransferResult",
    "VorticityTransfer",
    "apply_projected_gbd_renewal",
    "replace_particles_from_buffered_m4_renewal",
    "replace_particles_from_fvm",
    "replace_particles_from_lattice_blend",
    "replacement_eta",
    "required_renewal_buffer_length",
]
