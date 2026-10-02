"""Absolute FVM-state replacement inside the FVM--VPM overlap."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import cKDTree  # type: ignore[missing-module-attribute]

from source import log_style
from source.coupler.geometry import SolidBoundary, TriangulatedWall
from source.coupler.interpolation import FVMVelocityInterpolator
from source.coupler.reporting import format_coupler_log
from source.coupler.stable_renewal import (
    StableRenewalLattice,
    VortexInvariants,
    build_stable_renewal_lattice,
    renew_stable_overlap,
)

if TYPE_CHECKING:
    from source.coupler.solver import FVMVPMCoupler
    from source.solvers.fvm import FVMSolver
    from source.solvers.vpm import VPMSolver

logger = logging.getLogger("coupler")

_PARTICLE_REDUCTION_CHUNK_SIZE = 65_536


def _smoothstep(values: np.ndarray | float, lower: float, upper: float) -> np.ndarray:
    """C1 Hermite ramp for the solid and mesh confidence tapers."""
    span = float(upper) - float(lower)
    if abs(span) < np.finfo(np.float64).tiny:
        return np.where(np.asarray(values, dtype=np.float64) >= upper, 1.0, 0.0)
    phase = np.clip((np.asarray(values, dtype=np.float64) - lower) / span, 0.0, 1.0)
    return phase * phase * (3.0 - 2.0 * phase)


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
    """Particle population, represented-state and conservation budgets for renewal.

    Strengths are in m³/s, first moments in m⁴/s, and relative errors are
    dimensionless. Raw mismatch and applied correction are recorded separately
    from the final circulation/impulse residual.
    """

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
    transfer_method: str = "buffered_m4_renewal"
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
    renewed_input_particles: int = 0
    renewed_output_particles: int = 0
    preserved_outer_particles: int = 0
    coalesced_outer_particles: int = 0
    pruned_lattice_nodes: int = 0
    pruned_vortex_strength_l1: float = 0.0
    pruned_vortex_strength_fraction: float = 0.0
    population_pruned_particles: int = 0
    population_pruned_vortex_strength_fraction: float = 0.0
    population_pruned_velocity_bound: float = 0.0
    renewal_cfl: float = 0.0
    renewal_raw_vortex_strength_error: float = 0.0
    renewal_applied_vortex_strength_correction: float = 0.0
    renewal_conservation_error: float = 0.0
    renewal_vortex_strength_tolerance: float = 0.0
    renewal_raw_linear_impulse_error: float = 0.0
    renewal_applied_linear_impulse_correction: float = 0.0
    renewal_linear_impulse_error: float = 0.0
    renewal_linear_impulse_tolerance: float = 0.0
    renewal_raw_angular_impulse_error: float = 0.0
    renewal_applied_angular_impulse_correction: float = 0.0
    renewal_angular_impulse_error: float = 0.0
    renewal_applied_particle_strength_fraction: float = 0.0
    population_renewal_raw_vortex_strength_error: float = 0.0
    population_renewal_applied_vortex_strength_correction: float = 0.0
    population_renewal_conservation_error: float = 0.0
    population_renewal_raw_linear_impulse_error: float = 0.0
    population_renewal_applied_linear_impulse_correction: float = 0.0
    population_renewal_linear_impulse_error: float = 0.0
    population_renewal_applied_particle_strength_fraction: float = 0.0
    representation_residual_before_prune: float | None = None
    representation_residual_after_prune: float | None = None
    maximum_transfer_amplification: float = 0.0


def _validate_particle_sources(
    position: np.ndarray,
    cell_volume: np.ndarray,
    vorticity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize and validate the FVM donor arrays used by a transfer.

    Parameters
    ----------
    position : ndarray, shape (M, 3)
        FVM cell centres in metres.
    cell_volume : ndarray, shape (M,)
        Positive donor-cell volumes in m³.
    vorticity : ndarray, shape (M, 3)
        Cell-centred vorticity in 1/s.

    Returns
    -------
    tuple of ndarray
        Float64 position, volume, and vorticity arrays with the same leading
        length and contiguous-compatible reshape semantics.

    Raises
    ------
    ValueError
        If the three leading dimensions do not match.
    RuntimeError
        If any coordinate, volume, or vorticity value is non-finite, or if a
        donor volume is not strictly positive.
    """
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


def _particle_state_snapshot(vpm, *, slot: str = "coupler-transfer"):
    """Capture every rollback field through VPM's typed snapshot API."""
    return vpm.capture_particle_snapshot(slot=slot)


def _restore_particle_state(vpm, snapshot) -> None:
    """Restore a transfer snapshot through VPM's typed snapshot API."""
    vpm.restore_particle_snapshot(snapshot)


def _vortex_invariant_residual(
    target: VortexInvariants,
    actual: VortexInvariants,
) -> dict[str, float]:
    """Return normed residuals for the three conserved vector budgets.

    Parameters
    ----------
    target, actual : VortexInvariants
        Desired and measured total particle-strength, linear-impulse, and
        angular-impulse invariants.

    Returns
    -------
    dict[str, float]
        Euclidean residual norms keyed by ``total_vortex_strength`` (m³/s),
        ``linear_impulse`` (m⁴/s), and ``angular_impulse`` (m⁵/s). No input is
        mutated.
    """
    return {
        "total_vortex_strength": float(
            np.linalg.norm(target.total_vortex_strength - actual.total_vortex_strength)
        ),
        "linear_impulse": float(np.linalg.norm(target.linear_impulse - actual.linear_impulse)),
        "angular_impulse": float(np.linalg.norm(target.angular_impulse - actual.angular_impulse)),
    }


@dataclass(frozen=True)
class _AppliedParticleReductions:
    """Bounded-memory reductions of the particle state about to be uploaded."""

    renewed_invariants: VortexInvariants
    renewed_vortex_strength_l1: float
    cast_correction_l1: float
    renewed_position_scale: float
    output_vortex_strength_net: np.ndarray
    maximum_output_vortex_strength: float


def _reduce_applied_particle_state(
    position: np.ndarray,
    vortex_strength: np.ndarray,
    *,
    renewed_count: int,
    reference_vortex_strength: np.ndarray,
) -> _AppliedParticleReductions:
    """Reduce an applied storage-precision cloud without full float64 copies."""
    applied_position = np.asarray(position)
    applied_strength = np.asarray(vortex_strength)
    reference_strength = np.asarray(reference_vortex_strength, dtype=np.float64).reshape(-1, 3)
    if applied_position.ndim != 2 or applied_position.shape[1:] != (3,):
        raise ValueError("applied particle position must have shape (N, 3)")
    if applied_strength.shape != applied_position.shape:
        raise ValueError("applied position and vortex strength must have one common shape")
    if reference_strength.shape != applied_strength.shape:
        raise ValueError("reference vortex strength must match the applied particle state")
    count = len(applied_position)
    renewed = int(renewed_count)
    if not 0 <= renewed <= count:
        raise ValueError("renewed_count must lie inside the applied particle state")

    total = np.zeros(3, dtype=np.float64)
    linear_impulse = np.zeros(3, dtype=np.float64)
    angular_impulse = np.zeros(3, dtype=np.float64)
    output_net = np.zeros(3, dtype=np.float64)
    renewed_l1 = 0.0
    cast_correction_l1 = 0.0
    position_scale = 0.0
    maximum_strength = 0.0
    for start in range(0, count, _PARTICLE_REDUCTION_CHUNK_SIZE):
        stop = min(start + _PARTICLE_REDUCTION_CHUNK_SIZE, count)
        position_storage_chunk = applied_position[start:stop]
        if not np.all(np.isfinite(position_storage_chunk)):
            raise RuntimeError("buffered M4' renewal overflows VPM position storage precision")
        strength_chunk = applied_strength[start:stop].astype(np.float64, copy=False)
        if not np.all(np.isfinite(strength_chunk)):
            raise RuntimeError("buffered M4' renewal overflows VPM storage precision")
        magnitude = np.linalg.norm(strength_chunk, axis=1)
        output_net += strength_chunk.sum(axis=0, dtype=np.float64)
        maximum_strength = max(maximum_strength, float(magnitude.max(initial=0.0)))

        renewed_stop = min(stop, renewed)
        if renewed_stop <= start:
            continue
        local_count = renewed_stop - start
        renewed_strength = strength_chunk[:local_count]
        renewed_position = position_storage_chunk[:local_count].astype(np.float64, copy=False)
        position_cross_strength = np.cross(renewed_position, renewed_strength)
        total += renewed_strength.sum(axis=0, dtype=np.float64)
        linear_impulse += 0.5 * position_cross_strength.sum(axis=0, dtype=np.float64)
        angular_impulse += (1.0 / 3.0) * np.cross(
            renewed_position,
            position_cross_strength,
        ).sum(axis=0, dtype=np.float64)
        renewed_l1 += float(magnitude[:local_count].sum(dtype=np.float64))
        cast_correction_l1 += float(
            np.linalg.norm(
                renewed_strength - reference_strength[start:renewed_stop],
                axis=1,
            ).sum(dtype=np.float64)
        )
        position_scale = max(
            position_scale,
            float(np.linalg.norm(renewed_position, axis=1).max(initial=0.0)),
        )

    return _AppliedParticleReductions(
        renewed_invariants=VortexInvariants(
            total_vortex_strength=total,
            linear_impulse=linear_impulse,
            angular_impulse=angular_impulse,
        ),
        renewed_vortex_strength_l1=renewed_l1,
        cast_correction_l1=cast_correction_l1,
        renewed_position_scale=position_scale,
        output_vortex_strength_net=output_net,
        maximum_output_vortex_strength=maximum_strength,
    )


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
    kinematic_viscosity: float,
    freestream_speed: float,
    time_step_size: float,
    compute_diagnostics: bool,
    maximum_closure_correction_fraction: float | None = None,
    release_prune_threshold: float | None = None,
) -> TransferResult:
    """Atomically apply the recovered whole-belt M4' renewal to a GBD cloud."""
    if str(getattr(vpm, "viscous_scheme", "")).upper() != "GBD":
        raise ValueError("buffered_m4_renewal requires the GBD viscous scheme")
    viscosity = float(kinematic_viscosity)
    if not np.isfinite(viscosity) or viscosity < 0.0:
        raise ValueError("kinematic_viscosity must be finite and non-negative")
    if maximum_closure_correction_fraction is not None and (
        not np.isfinite(maximum_closure_correction_fraction)
        or not 0.0 < maximum_closure_correction_fraction <= 1.0
    ):
        raise ValueError("maximum_closure_correction_fraction must lie in (0, 1]")

    particles = vpm.particles
    n_before = int(particles.n_particles_total)
    # Renewal needs the actual storage precision to distinguish node roundoff
    # from genuinely off-lattice particles before using visible M4 support.
    existing_position = np.asarray(particles.position_cpu()).reshape(-1, 3)
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
        release_prune_threshold=release_prune_threshold,
        core_radius_ratio=core_radius_ratio,
        amplification_cap=amplification_cap,
        maximum_particle_count=int(particles.capacity),
        freestream_speed=freestream_speed,
        time_step_size=time_step_size,
        compute_diagnostics=compute_diagnostics,
    )
    if result.population_pruned_count:
        raise RuntimeError(
            "buffered M4' renewal reached the VPM particle capacity and would "
            f"discard {result.population_pruned_count:,} particles; increase capacity "
            "or reduce the physical particle population"
        )
    n_after = result.particle_count
    if n_after > int(particles.capacity):
        raise RuntimeError(
            f"buffered M4' renewal produced {n_after:,} particles, exceeding "
            f"the VPM capacity {int(particles.capacity):,}"
        )

    dtype = np.dtype(vpm.np_dtype)
    applied_position = np.ascontiguousarray(result.position, dtype=dtype)
    applied_strength = np.ascontiguousarray(result.vortex_strength, dtype=dtype)
    applied_core_radius = np.ascontiguousarray(result.core_radius, dtype=dtype)
    applied_particle_volume = np.ascontiguousarray(result.particle_volume, dtype=dtype)
    renewed_output_count = min(int(result.renewed_output_count), n_after)
    target_invariants = result.conservation_target_invariants
    raw_invariants = result.conservation_raw_invariants
    if target_invariants is None or raw_invariants is None:
        raise RuntimeError("buffered M4' renewal omitted its conservation reference state")
    reductions = _reduce_applied_particle_state(
        applied_position,
        applied_strength,
        renewed_count=renewed_output_count,
        reference_vortex_strength=result.vortex_strength,
    )
    applied_invariants = reductions.renewed_invariants
    raw_conservation = _vortex_invariant_residual(target_invariants, raw_invariants)
    applied_conservation = _vortex_invariant_residual(raw_invariants, applied_invariants)
    conservation = _vortex_invariant_residual(target_invariants, applied_invariants)
    closure_correction_fraction = float(
        result.conservation_applied_particle_strength_fraction
        + reductions.cast_correction_l1 / (result.conservation_reference_strength_l1 + 1.0e-30)
    )
    if not np.isfinite(closure_correction_fraction):
        raise RuntimeError("buffered M4' renewal produced a non-finite closure correction")
    if (
        maximum_closure_correction_fraction is not None
        and closure_correction_fraction > maximum_closure_correction_fraction
    ):
        raise RuntimeError(
            "buffered M4' renewal closure correction is excessive: "
            f"fraction={closure_correction_fraction:.6e}, "
            f"limit={maximum_closure_correction_fraction:.6e}"
        )
    storage_epsilon = float(np.finfo(dtype).eps)
    reference_strength_l1 = float(result.conservation_reference_strength_l1)
    position_scale = max(
        reductions.renewed_position_scale,
        float(lattice.particle_spacing),
    )
    renewal_vortex_strength_tolerance = 32.0 * storage_epsilon * reference_strength_l1
    renewal_linear_impulse_tolerance = (
        32.0 * storage_epsilon * reference_strength_l1 * position_scale
    )
    strength_error = float(conservation["total_vortex_strength"])
    impulse_error = float(conservation["linear_impulse"])
    if strength_error > renewal_vortex_strength_tolerance + np.finfo(np.float64).tiny:
        raise RuntimeError(
            "buffered M4' renewal does not conserve vortex strength in storage precision: "
            f"error={strength_error:.6e}, "
            f"tolerance={renewal_vortex_strength_tolerance:.6e}"
        )
    if impulse_error > renewal_linear_impulse_tolerance + np.finfo(np.float64).tiny:
        raise RuntimeError(
            "buffered M4' renewal does not conserve linear impulse in storage precision: "
            f"error={impulse_error:.6e}, "
            f"tolerance={renewal_linear_impulse_tolerance:.6e}"
        )

    # Prepare and validate the complete replacement budget before mutating the
    # VPM. A malformed renewal result must leave the prior cloud untouched.
    removed_mask = np.ones(n_before, dtype=bool)
    if n_before:
        valid = np.ones(n_before, dtype=bool)
        if particle_in_solid is not None:
            solid_mask = np.asarray(
                particle_in_solid(existing_position),
                dtype=bool,
            ).reshape(-1)
            if len(solid_mask) != n_before:
                raise RuntimeError("particle_in_solid must return one value per existing particle")
            valid &= ~solid_mask
        preserved = valid & ~np.all(
            (existing_position >= lattice.renewal_bounds[::2])
            & (existing_position <= lattice.renewal_bounds[1::2]),
            axis=1,
        )
        removed_mask = ~preserved
    coalesced_input = np.asarray(
        result.coalesced_outer_input_indices,
        dtype=np.int64,
    ).reshape(-1)
    if len(coalesced_input):
        if np.any((coalesced_input < 0) | (coalesced_input >= n_before)):
            raise RuntimeError("renewal returned an invalid coalesced input index")
        removed_mask[coalesced_input] = True
    removed_count = int(np.count_nonzero(removed_mask))
    if n_before - removed_count + renewed_output_count != n_after:
        raise RuntimeError("renewal returned an inconsistent particle budget")
    replaced_strength = existing_strength[removed_mask]
    input_net = existing_strength.sum(axis=0, dtype=np.float64)

    vpm.replace_vortex_particles(
        position=applied_position,
        velocity=np.zeros((n_after, 3), dtype=dtype),
        vortex_strength=applied_strength,
        core_radius=applied_core_radius,
        particle_volume=applied_particle_volume,
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

    injected_net = reductions.renewed_invariants.total_vortex_strength
    output_net = reductions.output_vortex_strength_net
    population_raw_conservation = result.population_conservation_raw_mismatch
    population_applied_conservation = result.population_conservation_applied_correction
    population_conservation = result.population_conservation_residual
    maximum_strength = reductions.maximum_output_vortex_strength
    return TransferResult(
        n_particles_before=n_before,
        n_particles_retained=int(result.preserved_outer_count),
        n_particles_removed=removed_count,
        n_particles_blended=int(result.renewed_input_count),
        n_particles_injected=renewed_output_count,
        n_particles_after=n_after,
        injected_vortex_strength_l1=reductions.renewed_vortex_strength_l1,
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
        mapped_target_vortex_strength_l1=reductions.renewed_vortex_strength_l1,
        mapped_target_vortex_strength_net=injected_net,
        maximum_mapped_vortex_strength=maximum_strength,
        renewed_input_particles=int(result.renewed_input_count),
        renewed_output_particles=renewed_output_count,
        preserved_outer_particles=int(result.preserved_outer_count),
        coalesced_outer_particles=int(result.coalesced_outer_count),
        pruned_lattice_nodes=int(result.pruned_node_count),
        pruned_vortex_strength_l1=float(result.pruned_vortex_strength_l1),
        pruned_vortex_strength_fraction=float(result.pruned_vortex_strength_fraction),
        population_pruned_particles=int(result.population_pruned_count),
        population_pruned_vortex_strength_fraction=float(
            result.population_pruned_vortex_strength_fraction
        ),
        population_pruned_velocity_bound=float(result.population_pruned_velocity_bound),
        renewal_cfl=float(result.transfer_cfl),
        renewal_raw_vortex_strength_error=float(raw_conservation.get("total_vortex_strength", 0.0)),
        renewal_applied_vortex_strength_correction=float(
            applied_conservation.get("total_vortex_strength", 0.0)
        ),
        renewal_conservation_error=float(conservation.get("total_vortex_strength", 0.0)),
        renewal_vortex_strength_tolerance=renewal_vortex_strength_tolerance,
        renewal_raw_linear_impulse_error=float(raw_conservation.get("linear_impulse", 0.0)),
        renewal_applied_linear_impulse_correction=float(
            applied_conservation.get("linear_impulse", 0.0)
        ),
        renewal_linear_impulse_error=float(conservation.get("linear_impulse", 0.0)),
        renewal_linear_impulse_tolerance=renewal_linear_impulse_tolerance,
        renewal_raw_angular_impulse_error=float(raw_conservation.get("angular_impulse", 0.0)),
        renewal_applied_angular_impulse_correction=float(
            applied_conservation.get("angular_impulse", 0.0)
        ),
        renewal_angular_impulse_error=float(conservation.get("angular_impulse", 0.0)),
        renewal_applied_particle_strength_fraction=closure_correction_fraction,
        population_renewal_raw_vortex_strength_error=float(
            population_raw_conservation.get("total_vortex_strength", 0.0)
        ),
        population_renewal_applied_vortex_strength_correction=float(
            population_applied_conservation.get("total_vortex_strength", 0.0)
        ),
        population_renewal_conservation_error=float(
            population_conservation.get("total_vortex_strength", 0.0)
        ),
        population_renewal_raw_linear_impulse_error=float(
            population_raw_conservation.get("linear_impulse", 0.0)
        ),
        population_renewal_applied_linear_impulse_correction=float(
            population_applied_conservation.get("linear_impulse", 0.0)
        ),
        population_renewal_linear_impulse_error=float(
            population_conservation.get("linear_impulse", 0.0)
        ),
        population_renewal_applied_particle_strength_fraction=float(
            result.population_conservation_applied_particle_strength_fraction
        ),
        representation_residual_before_prune=result.representation_residual_before_prune,
        representation_residual_after_prune=result.representation_residual_after_prune,
        maximum_transfer_amplification=float(result.maximum_transfer_amplification),
    )


def _transfer_log_record(step: int, result: TransferResult) -> log_style.Event:
    """Retain the immutable transfer result until an enabled report needs rows."""
    return log_style.Event("interface transfer", lambda: _transfer_log_rows(step, result))


def _transfer_log_rows(step: int, result: TransferResult) -> tuple[log_style.Row, ...]:
    """Build host rows for one auditable particle-strength transfer.

    Parameters
    ----------
    step : int
        Coupled accepted-step index associated with the transfer.
    result : TransferResult
        Immutable transfer accounting and error diagnostics.

    Returns
    -------
    tuple[log_style.Row, ...]
        Labelled scalar measurements, units and short host-vector norms. Row
        construction leaves the transferred state and stored result unchanged.
    """
    rows: list[log_style.Row] = [
        ("coupling step", step),
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
    ]
    rows.extend(
        (
            (
                "vortex strength replaced, l1",
                f"{result.replaced_vortex_strength_l1:.3e}",
                "m^3/s",
            ),
            (
                "vortex strength mapped, l1",
                f"{result.mapped_target_vortex_strength_l1:.3e}",
                "m^3/s",
            ),
            (
                "vortex strength born, l1",
                f"{result.injected_vortex_strength_l1:.3e}",
                "m^3/s",
            ),
            (
                "vortex strength, max node",
                f"{result.maximum_mapped_vortex_strength:.3e}",
                "m^3/s",
            ),
            (
                "vortex strength, net change",
                f"{float(np.linalg.norm(result.state_change_vortex_strength_net)):.3e}",
                "m^3/s",
            ),
        )
    )
    rows.extend(
        (
            ("renewal belt, input", f"{result.renewed_input_particles:,}"),
            ("renewal belt, output", f"{result.renewed_output_particles:,}"),
            ("outer wake, preserved", f"{result.preserved_outer_particles:,}"),
            ("outer wake, coalesced at support seam", f"{result.coalesced_outer_particles:,}"),
            ("lattice nodes, pruned", f"{result.pruned_lattice_nodes:,}"),
            (
                "pruned strength, fraction",
                f"{100.0 * result.pruned_vortex_strength_fraction:.3f}",
                "%",
            ),
            ("renewal CFL", f"{result.renewal_cfl:.3f}"),
            (
                "renewal closure, raw strength mismatch",
                f"{result.renewal_raw_vortex_strength_error:.3e}",
            ),
            (
                "renewal closure, applied strength correction",
                f"{result.renewal_applied_vortex_strength_correction:.3e}",
            ),
            (
                "renewal closure, corrected strength mismatch",
                f"{result.renewal_conservation_error:.3e}",
            ),
            (
                "renewal closure, strength tolerance",
                f"{result.renewal_vortex_strength_tolerance:.3e}",
            ),
            (
                "renewal closure, raw impulse mismatch",
                f"{result.renewal_raw_linear_impulse_error:.3e}",
            ),
            (
                "renewal closure, applied impulse correction",
                f"{result.renewal_applied_linear_impulse_correction:.3e}",
            ),
            (
                "renewal closure, corrected impulse mismatch",
                f"{result.renewal_linear_impulse_error:.3e}",
            ),
            (
                "renewal closure, impulse tolerance",
                f"{result.renewal_linear_impulse_tolerance:.3e}",
            ),
            (
                "renewal closure, particle-strength correction",
                f"{100.0 * result.renewal_applied_particle_strength_fraction:.3f}",
                "%",
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
                    result.representation_residual_before_prune,
                ),
                (
                    "representation residual, after prune",
                    f"{result.representation_residual_after_prune:.3e}",
                ),
            )
        )
    return tuple(rows)


class VorticityTransfer:
    """Synchronize the inner VPM cloud with absolute cell-centred FVM state.

    This runtime component is constructed internally by
    :class:`FVMVPMCoupler` after both solver discretizations are known. It
    derives vorticity ``omega = curl(u)`` from the accepted FVM gradient,
    represents ``Gamma = omega * V`` on the renewal lattice,
    and atomically replaces the FVM-authoritative portion of the particle
    cloud while preserving the outer wake.

    Parameters
    ----------
    coupler : FVMVPMCoupler
        Initialized driver providing the transfer policy, FVM box, shared
        viscosity in m²/s, VPM spacing/core ratio, and coupling time step in s.

    Attributes
    ----------
    particle_spacing : float
        VPM lattice spacing ``h`` in m.
    core_radius_ratio : float
        Dimensionless ``sigma/h`` ratio used for injected particles.
    coupling_time_step : float
        VPM macro-step represented by each transfer, in s.
    step : int
        Number of transfer calls attempted. Incremented at call entry.
    last_interface_flow : dict[str, float]
        Mean outward normal velocity by transfer face in m/s.
    last_vortex_line_closure : dict[str, float]
        Dimensionless normal-vorticity closure error by transfer face.

    Raises
    ------
    RuntimeError
        If the coupler has not resolved the FVM box, fluid properties, VPM
        spacing/core ratio, or coupling time step.
    ValueError
        If buffered M4-prime renewal is selected without GBD diffusion.

    Notes
    -----
    Users normally interact with this object through the driver. Call
    :meth:`setup` once for an FVM mesh, then :meth:`transfer` only at accepted
    synchronized coupling states. Transfer mutates VPM particles and caches;
    setup itself only builds donor/lattice metadata.
    """

    def __init__(self, coupler: FVMVPMCoupler) -> None:
        """Resolve transfer controls from an initialized coupling driver."""
        cfg = coupler.setup
        if coupler.kinematic_viscosity is None or coupler.fvm_box is None:
            raise RuntimeError("VorticityTransfer requires initialized FVM and VPM state")
        self.config = cfg
        candidate_vpm = getattr(coupler, "vpm_solver", None)
        self._wall_scatter_correction = getattr(
            getattr(candidate_vpm, "physics", None), "_m4_wall_corrections", None
        )
        if (
            candidate_vpm is not None
            and str(getattr(candidate_vpm, "viscous_scheme", "")).upper() != "GBD"
        ):
            raise ValueError("buffered_m4_renewal currently requires the GBD viscous scheme")
        if not np.isfinite(coupler.vpm_core_radius_ratio):
            raise RuntimeError("VorticityTransfer requires the resolved VPM core-radius ratio")
        self.core_radius_ratio = float(coupler.vpm_core_radius_ratio)
        if not np.isfinite(coupler.vpm_particle_spacing):
            raise RuntimeError("VorticityTransfer requires the resolved VPM particle spacing")
        self.particle_spacing = float(coupler.vpm_particle_spacing)
        induction = getattr(candidate_vpm, "induction", None)
        self._planar_span = getattr(induction, "planar_span", None)
        self._slip_slab = getattr(induction, "method", None) == "SLIP_SLAB"
        self._slip_z = (
            (float(induction.z_min), float(induction.z_max))
            if self._slip_slab and induction is not None
            else None
        )
        self._planar_induction = induction if self._planar_span is not None else None
        self.eta_blend_width = float(cfg.eta_blend_width)
        self.vpm_only_width = float(cfg.vpm_only_width)
        self.transfer_prune_threshold_abs = (
            float(cfg.transfer_vorticity_cutoff)
            * self.particle_spacing**2
            * (self.particle_spacing if self._planar_span is None else self._planar_span)
        )
        self.transfer_release_prune_threshold_abs = self.transfer_prune_threshold_abs
        if candidate_vpm is not None:
            viscous = candidate_vpm.setup.viscous
            if str(viscous.gbd_threshold_mode).lower() != "absolute":
                raise ValueError(
                    "buffered_m4_renewal requires GBD threshold_mode='absolute' so "
                    "release-belt pruning has a resolved physical floor"
                )
            self.transfer_release_prune_threshold_abs = float(viscous.gbd_threshold)
            if self.transfer_release_prune_threshold_abs > self.transfer_prune_threshold_abs:
                raise ValueError(
                    "transfer_vorticity_cutoff must be at least the VPM GBD vorticity floor"
                )
        self.transfer_amplification_cap = float(cfg.transfer_amplification_cap)
        self.kinematic_viscosity = float(coupler.kinematic_viscosity)
        coupling_time_step = getattr(coupler, "vpm_time_step_size", None)
        if coupling_time_step is None:
            raise RuntimeError("VorticityTransfer requires the resolved VPM time step")
        self.coupling_time_step = 0.0 if coupling_time_step is None else float(coupling_time_step)
        self.renewal_buffer_length = required_renewal_buffer_length(
            cfg.freestream_velocity,
            self.coupling_time_step,
            self.particle_spacing,
        )
        self.diagnostic_interval = int(cfg.transfer_diagnostic_interval_steps)
        self.discretization_error_limit = float(cfg.transfer_discretization_error_limit)
        self._fvm_box = np.asarray(coupler.fvm_box, dtype=np.float64)
        self._box: np.ndarray | None = None
        self._cell_tree: cKDTree | None = None
        self._cell_centre: np.ndarray | None = None
        self._cell_volume: np.ndarray | None = None
        self._velocity_trace: FVMVelocityInterpolator | None = None
        self._buffered_trace_stencils: list | None = None
        self._fvm_solid_mask: np.ndarray | None = None
        self._solid_bodies: tuple = ()
        self.solid_boundary: SolidBoundary | None = None
        self._lattice_anchor: np.ndarray | None = None
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
        """Classify transfer points against configured solid geometry.

        Parameters
        ----------
        points : array_like, shape (K, 3)
            Candidate particle or lattice-node coordinates in metres.
        include_boundary : bool
            Include points exactly on a solid boundary when true; use a
            strict interior test when false.

        Returns
        -------
        ndarray, shape (K,), dtype=bool
            ``True`` for points excluded from the fluid transfer region.
            The input is converted to float64 but is not modified.
        """
        query = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if self.solid_boundary is not None:
            return self.solid_boundary.contains(query, include_boundary=include_boundary)
        return np.zeros(len(query), dtype=bool)

    def _signed_solid_distance(self, points: np.ndarray) -> np.ndarray:
        """Signed distance from the configured wall, positive in fluid."""
        query = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if self.solid_boundary is not None:
            return self.solid_boundary.signed_distance(query)
        return np.full(len(query), np.inf, dtype=np.float64)

    def _face_cell_index(self, bounds: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Index donor cells lying on each face of a transfer box.

        Parameters
        ----------
        bounds : ndarray, shape (6,)
            ``(xmin, xmax, ymin, ymax, zmin, zmax)`` bounds in metres.

        Returns
        -------
        dict[str, tuple[ndarray, ndarray]]
            Face-name to ``(cell_indices, outward_normal)`` mapping. Indices
            refer to the FVM cell-centre array and normals are dimensionless.
            Faces without donor cells are omitted.
        """
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
        """Refresh the cached transfer-box face-to-cell lookup."""
        self._face_cells = {} if self._box is None else self._face_cell_index(self._box)

    def check_interface_flow(self, velocity: np.ndarray) -> dict[str, float]:
        """Return mean outward velocity on each indexed transfer-box face.

        Parameters
        ----------
        velocity : ndarray, shape (n_cells, 3)
            Accepted cell-centred Cartesian FVM velocity in m/s, ordered like
            the cell centres supplied to :meth:`setup`.

        Returns
        -------
        dict[str, float]
            Mean ``u dot n`` in m/s keyed by ``xmin`` through ``zmax`` for
            every face with donor cells.
        """
        values = np.asarray(velocity, dtype=np.float64).reshape(-1, 3)
        return {
            name: float(np.mean(values[index] @ normal))
            for name, (index, normal) in self._face_cells.items()
            if index.max(initial=-1) < len(values)
        }

    def check_vortex_line_closure(self, velocity_gradient: np.ndarray) -> dict[str, float]:
        """Measure normal-vorticity leakage on each transfer-box face.

        Parameters
        ----------
        velocity_gradient : ndarray, shape (n_cells, 3, 3)
            Accepted FVM gradient in 1/s using ``G[i, j] = d(u_j)/d(x_i)``.

        Returns
        -------
        dict[str, float]
            Mean absolute ``omega dot n`` divided by the global mean vorticity
            magnitude, keyed by face. Values are dimensionless.
        """
        vorticity = self._vorticity_from_gradient(velocity_gradient)
        scale = float(np.linalg.norm(vorticity, axis=1).mean()) + np.finfo(float).tiny
        return {
            name: float(np.mean(np.abs(vorticity[index] @ normal)) / scale)
            for name, (index, normal) in self._face_cells.items()
            if index.max(initial=-1) < len(vorticity)
        }

    def setup(self, fvm: FVMSolver) -> None:
        """Build FVM donor indices, solid masks, and renewal lattices.

        Parameters
        ----------
        fvm : FVMSolver
            Initialized solver providing global cell-centre coordinates
            ``(M, 3)`` in m, cell volumes ``(M,)`` in m³, boundary geometry,
            and optional immersed-body geometry.

        Raises
        ------
        RuntimeError
            If cell-centre/volume counts disagree, values are non-finite, or a
            required lattice/solid representation cannot be built.
        ValueError
            If transfer bounds or method-specific geometry is invalid.

        Notes
        -----
        Partitioned FVM getters are collective. This method mutates only
        transfer metadata and interpolation caches; it does not modify FVM
        fields, VPM particles, or either accepted clock.
        """
        self._buffered_trace_stencils = None
        self._box = np.asarray(
            self.config.transfer_region_bounds or self._fvm_box, dtype=np.float64
        )
        if self._slip_z is not None and not np.allclose(
            self._box[4:6], self._slip_z, rtol=0.0, atol=1e-8
        ):
            raise ValueError("slip-slab transfer z faces must match induction planes")
        self._cell_centre = np.asarray(fvm.get_cell_centre_coordinates(), dtype=np.float64).reshape(
            -1, 3
        )
        self._cell_volume = np.asarray(fvm.get_cell_volume(), dtype=np.float64).reshape(-1)

        # These partitioned getters are collective, even though only rank zero
        # receives the assembled arrays. Keep their call order identical.
        resolved_setup = getattr(fvm, "_resolved_setup", fvm.setup)
        wall_patches = [
            boundary_condition.name
            for boundary_condition in resolved_setup.boundaries
            if boundary_condition.mesh_type == "wall"
        ]
        get_wall_triangles = getattr(fvm, "get_wall_surface_triangles", None)
        wall_triangles = get_wall_triangles() if callable(get_wall_triangles) else None

        from source.simulation.parallel import collective_phase

        comm = getattr(getattr(fvm, "parallel", None), "comm", None)
        with collective_phase(comm, "transfer geometry"):
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
            self._cell_tree = cKDTree(self._cell_centre)

            ibm = getattr(fvm, "ibm", None)
            bodies = () if ibm is None else tuple(ibm.bodies)
            if any(not body.has_solid_geometry for body in bodies):
                raise ValueError("Coupled immersed bodies require explicit solid geometry")
            self._solid_bodies = bodies
            if wall_triangles is not None and len(wall_triangles):
                self._solid_bodies += (TriangulatedWall(wall_triangles, self._fvm_box),)
            elif wall_patches:
                raise RuntimeError("Body-fitted transfer requires the FVM wall surface triangles")
            self.solid_boundary = SolidBoundary(self._solid_bodies) if self._solid_bodies else None
            # One mesh-derived phase for every geometry, independent of donor
            # ordering, wall patch names, and inferred body dimensions.
            self._lattice_anchor = self._cell_centre.min(axis=0)
            self._velocity_trace = FVMVelocityInterpolator(
                self._cell_centre,
                self._cell_tree,
                neighbour_count=4,
                solid_boundary=self.solid_boundary,
            )
            if self._slip_z is not None:
                # Keep both physical slip planes halfway between VPM nodes.
                # The FVM cell-centre z phase need not match the particle h.
                self._lattice_anchor[2] = self._slip_z[0] + 0.5 * self.particle_spacing

            if self._cell_tree is None:
                raise RuntimeError("buffered M4' renewal requires an FVM cell tree")

            def mesh_weight_at_node(points: np.ndarray) -> np.ndarray:
                # Distance to a donor centre is not a fluid-domain test:
                # coarse/anisotropic cells legitimately span many particles.
                # Solid exclusion uses wall geometry, independently of h.
                roundoff = 64 * np.finfo(float).eps * np.maximum(1.0, np.abs(self._fvm_box))
                return np.all(
                    (points >= self._fvm_box[::2] - roundoff[::2])
                    & (points <= self._fvm_box[1::2] + roundoff[1::2]),
                    axis=1,
                ).astype(np.float64)

            has_solid = self.solid_boundary is not None

            def fluid_weight_at_node(points: np.ndarray) -> np.ndarray:
                return _smoothstep(
                    self._signed_solid_distance(points),
                    -self.particle_spacing,
                    0.0,
                )

            def interior_at_node(points: np.ndarray) -> np.ndarray:
                return self._points_in_solid(points, include_boundary=False)

            if self._planar_induction is not None:
                _, self._planar_groups, self._planar_group_counts = np.unique(
                    np.round(self._cell_centre[:, :2], 11),
                    axis=0,
                    return_inverse=True,
                    return_counts=True,
                )
                if np.any(self._planar_group_counts < 2):
                    raise ValueError("Planar coupling requires complete extruded FVM cell stacks")
                self.last_spanwise_metrics = {}
                self._planar_induction.lattice_anchor = self._lattice_anchor.copy()
                self._planar_induction.solid_at = interior_at_node if has_solid else None
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
                solid_boundary=self.solid_boundary,
                wall_scatter_correction=self._wall_scatter_correction,
                planar_span=self._planar_span,
                slip_slab=self._slip_slab,
                plane_z=getattr(self._planar_induction, "plane_z", 0.0),
            )
            self._fvm_solid_mask = self._points_in_solid(
                self._cell_centre,
                include_boundary=True,
            )
            inside = np.all(
                (self._cell_centre >= self._box[::2]) & (self._cell_centre <= self._box[1::2]),
                axis=1,
            )
            self._authority_cell_mask = inside & ~self._fvm_solid_mask
            donor_count = int(np.count_nonzero(self._authority_cell_mask))
            if donor_count == 0:
                raise ValueError("FVM transfer region contains no fluid cell centres")
            self._build_face_cell_index()
            lattice_count = (
                len(self._stable_renewal_lattice.positions)
                if self._stable_renewal_lattice is not None
                else 0
            )
            geometry_rows: list[log_style.Row] = []
            if self._stable_renewal_lattice is not None:
                velocity = np.asarray(self.config.freestream_velocity, dtype=np.float64)
                streamwise_axis = int(np.argmax(np.abs(velocity)))
                downstream_is_maximum = velocity[streamwise_axis] >= 0.0
                axis_name = "xyz"[streamwise_axis]
                lattice = self._stable_renewal_lattice
                planes = lattice.origin[streamwise_axis] + self.particle_spacing * np.arange(
                    lattice.shape[streamwise_axis]
                )
                authority_grid = lattice.fvm_authority.reshape(lattice.shape)
                plane_authority = np.max(
                    np.moveaxis(authority_grid, streamwise_axis, 0).reshape(len(planes), -1),
                    axis=1,
                )
                authoritative_planes = planes[plane_authority > 0.0]
                face = self._box[2 * streamwise_axis + int(downstream_is_maximum)]
                renewal_edge = lattice.renewal_bounds[
                    2 * streamwise_axis + int(downstream_is_maximum)
                ]
                beyond_face = planes > face if downstream_is_maximum else planes < face
                inside_renewal = (
                    planes <= renewal_edge if downstream_is_maximum else planes >= renewal_edge
                )
                beyond_renewal = ~inside_renewal
                support_edge = renewal_edge + (
                    2.0 * self.particle_spacing
                    if downstream_is_maximum
                    else -2.0 * self.particle_spacing
                )
                inside_m4_support = (
                    planes <= support_edge if downstream_is_maximum else planes >= support_edge
                )

                def downstream_extreme(values: np.ndarray) -> float:
                    return float(values.max() if downstream_is_maximum else values.min())

                def upstream_extreme(values: np.ndarray) -> float:
                    return float(values.min() if downstream_is_maximum else values.max())

                donor_coordinate = self._cell_centre[self._authority_cell_mask, streamwise_axis]
                geometry_rows.extend(
                    (
                        (
                            f"fvm last donor centre, {axis_name}",
                            f"{downstream_extreme(donor_coordinate):.6f}",
                            "m",
                        ),
                        (
                            f"fvm authority boundary, {axis_name}",
                            f"{face:.6f}",
                            "m",
                        ),
                        (
                            f"last fvm-authoritative plane, {axis_name}",
                            f"{downstream_extreme(authoritative_planes):.6f}",
                            "m",
                        ),
                        (
                            f"first vpm-only release plane, {axis_name}",
                            f"{upstream_extreme(planes[beyond_face]):.6f}",
                            "m",
                        ),
                        (
                            f"last renewed input plane, {axis_name}",
                            f"{downstream_extreme(planes[inside_renewal]):.6f}",
                            "m",
                        ),
                        (
                            f"first persistent input plane, {axis_name}",
                            f"{upstream_extreme(planes[beyond_renewal]):.6f}",
                            "m",
                        ),
                        (
                            f"last m4-prime-reachable plane, {axis_name}",
                            f"{downstream_extreme(planes[inside_m4_support]):.6f}",
                            "m",
                        ),
                        (
                            f"allocated m4-prime guard endpoint, {axis_name}",
                            f"{downstream_extreme(planes):.6f}",
                            "m",
                        ),
                    )
                )
            logger.info(
                format_coupler_log(
                    "replacement region",
                    ("method", "buffered_m4_renewal"),
                    ("fvm fluid cells", f"{donor_count:,}"),
                    *(
                        (("authority ramp width, eta", "off"),)
                        if self.eta_blend_width == 0.0
                        else (("authority ramp width, eta", f"{self.eta_blend_width:.4g}", "m"),)
                    ),
                    ("renewal buffer", f"{self.renewal_buffer_length:.4g}", "m"),
                    ("renewal lattice nodes", f"{lattice_count:,}"),
                    *geometry_rows,
                    (
                        "state",
                        "synchronized fvm velocity trace",
                    ),
                )
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
        if self._planar_induction is not None:
            groups = self._planar_groups
            counts = self._planar_group_counts
            mean = np.column_stack(
                [
                    np.bincount(groups, weights=fvm_velocity[:, axis], minlength=len(counts))
                    / counts
                    for axis in range(3)
                ]
            )
            variation = float(np.max(np.abs(fvm_velocity - mean[groups]), initial=0))
            span_velocity = float(np.max(np.abs(fvm_velocity[:, 2]), initial=0))
            scale = max(
                float(np.linalg.norm(self.config.freestream_velocity_vector)),
                float(np.linalg.norm(mean, axis=1).max(initial=0)),
                1e-12,
            )
            self.last_spanwise_metrics = {
                "span_velocity_max": span_velocity,
                "span_variation_max": variation,
                "velocity_scale": scale,
            }
            if max(variation, span_velocity) > self._planar_induction.spanwise_tolerance * scale:
                raise RuntimeError(
                    f"Planar FVM donor state lost spanwise invariance: {self.last_spanwise_metrics}"
                )
        has_solid = self.solid_boundary is not None

        def fluid_weight(points: np.ndarray) -> np.ndarray:
            return _smoothstep(
                self._signed_solid_distance(points),
                -self.particle_spacing,
                0.0,
            )

        def in_solid(points: np.ndarray) -> np.ndarray:
            return self._points_in_solid(points, include_boundary=False)

        lattice = self._stable_renewal_lattice
        slab_box = self._box if self._slip_slab else None
        if self._slip_slab and slab_box is None:
            raise RuntimeError("slip-slab renewal requires resolved FVM bounds")
        # Only geometry is pinned: every interface sweep supplies fresh fields.
        needed = (lattice.mesh_weight > 0.0) | (lattice.fluid_weight < 1.0)
        if slab_box is not None:
            needed &= (lattice.positions[:, 2] >= slab_box[4] - 1e-10) & (
                lattice.positions[:, 2] <= slab_box[5] + 1e-10
            )
        if self._buffered_trace_stencils is None:
            stencils = []
            positions = lattice.positions[needed]
            for axis in range(3 if self._planar_span is None else 2):
                for sign in (1.0, -1.0):
                    query = positions.copy()
                    query[:, axis] += sign * 0.5 * self.particle_spacing
                    reflected = np.zeros(len(query), dtype=bool)
                    if slab_box is not None:
                        below = query[:, 2] < slab_box[4]
                        above = query[:, 2] > slab_box[5]
                        query[below, 2] = 2.0 * slab_box[4] - query[below, 2]
                        query[above, 2] = 2.0 * slab_box[5] - query[above, 2]
                        reflected = below | above
                        if np.any((query[:, 2] < slab_box[4]) | (query[:, 2] > slab_box[5])):
                            raise RuntimeError("renewal trace needs more than one slab reflection")
                    wall_weight = (
                        _smoothstep(self._signed_solid_distance(query), 0.0, self.particle_spacing)
                        if has_solid
                        else np.ones(len(query))
                    )
                    fluid = wall_weight > 0.0
                    stencil = self._velocity_trace.prepare(query[fluid])
                    stencils.append((axis, sign, fluid, wall_weight[fluid], reflected, stencil))
            self._buffered_trace_stencils = stencils

        def target_at_node(points: np.ndarray) -> np.ndarray:
            # Integrate n cross u on the same six midpoint faces as the
            # uncached velocity-trace formula. In-solid velocity remains zero.
            strength = np.zeros((np.count_nonzero(needed), 3), dtype=np.float64)
            for start in range(0, len(self._buffered_trace_stencils), 2):
                velocities = []
                axis = self._buffered_trace_stencils[start][0]
                for (
                    _axis,
                    _sign,
                    fluid,
                    weight,
                    reflected,
                    stencil,
                ) in self._buffered_trace_stencils[start : start + 2]:
                    velocity = np.zeros_like(strength)
                    velocity[fluid] = (
                        stencil.sample(fvm_velocity, fvm_velocity_gradient) * weight[:, None]
                    )
                    velocity[reflected, 2] *= -1.0
                    velocities.append(velocity)
                normal = np.zeros(3)
                normal[axis] = 1.0
                strength += self.particle_spacing**2 * np.cross(
                    normal, velocities[0] - velocities[1]
                )
            if self._planar_span is not None:
                strength[:, :2] = 0.0
                strength *= self._planar_span / self.particle_spacing
            target = np.zeros_like(points)
            target[needed] = strength
            return target

        return replace_particles_from_buffered_m4_renewal(
            vpm,
            lattice=self._stable_renewal_lattice,
            fvm_vortex_strength_at_node=target_at_node,
            particle_fluid_weight=fluid_weight if has_solid else None,
            particle_in_solid=in_solid if has_solid else None,
            prune_threshold=self.transfer_prune_threshold_abs,
            release_prune_threshold=self.transfer_release_prune_threshold_abs,
            core_radius_ratio=self.core_radius_ratio,
            amplification_cap=self.transfer_amplification_cap,
            kinematic_viscosity=self.kinematic_viscosity,
            freestream_speed=float(np.linalg.norm(self.config.freestream_velocity_vector)),
            time_step_size=self.coupling_time_step,
            compute_diagnostics=(self.step == 1 or self.step % self.diagnostic_interval == 0),
            maximum_closure_correction_fraction=self.discretization_error_limit,
        )

    def transfer(
        self,
        vpm: VPMSolver,
        velocity: np.ndarray,
        velocity_gradient: np.ndarray,
    ) -> TransferResult:
        """Replace inner particles from one accepted FVM donor state.

        Parameters
        ----------
        vpm : VPMSolver
            Active rank-zero particle solver. Its particle fields and source
            caches are mutated atomically on success.
        velocity : ndarray, shape (M, 3)
            Accepted cell-centred FVM velocity in m/s, ordered like the donor
            cells captured by :meth:`setup`.
        velocity_gradient : ndarray, shape (M, 3, 3)
            Accepted FVM gradient in 1/s with
            ``G[i, j] = d(u_j)/d(x_i)``. Curl is converted to pointwise
            vorticity and then integrated particle strength ``Gamma=omega*V``.

        Returns
        -------
        TransferResult
            Immutable population, circulation, moment, divergence, projection,
            and pruning budget for the completed replacement.

        Raises
        ------
        RuntimeError
            If :meth:`setup` has not run, an invariant/quality gate fails, or
            the selected transfer cannot construct a valid replacement.
        ValueError
            If velocity or gradient donor counts disagree with the FVM cells.

        Notes
        -----
        The method increments :attr:`step`, updates interface diagnostics, and
        replaces the accepted VPM particle state. Transfer implementations
        snapshot all mutable particle fields and restore them if a later
        quality gate fails; no FVM arrays are modified.
        """
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
        result = self._transfer_buffered_m4_renewal(
            vpm,
            fvm_velocity=velocity_values,
            fvm_velocity_gradient=gradient_values,
        )
        if self.step % self.diagnostic_interval == 0:
            logger.info(_transfer_log_record(self.step, result))
        return result


__all__ = [
    "TransferResult",
    "VorticityTransfer",
    "replace_particles_from_buffered_m4_renewal",
    "required_renewal_buffer_length",
]
