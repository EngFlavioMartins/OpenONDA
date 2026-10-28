"""Vorticity transfer: FVM vorticity replaces the particle vorticity inside the transfer region."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import cKDTree  # type: ignore[missing-module-attribute]

from openonda.core import log_style
from openonda.coupler.frame import MeshFrame
from openonda.coupler.geometry import SolidBoundary, TriangulatedWall
from openonda.coupler.interpolation import FVMVelocityInterpolator
from openonda.coupler.reporting import format_coupler_log
from openonda.coupler.transfer_lattice import (
    TransferLattice,
    VortexMoments,
    build_transfer_lattice,
    transfer_on_lattice,
    vortex_moment_residual,
)

if TYPE_CHECKING:
    from openonda.coupler.solver import CoupledSolver
    from openonda.fvm import FVMSolver
    from openonda.vpm import VPMSolver

logger = logging.getLogger("openonda.coupler")

_PARTICLE_REDUCTION_CHUNK_SIZE = 65_536


def _smoothstep(values: np.ndarray | float, lower: float, upper: float) -> np.ndarray:
    """C1 Hermite ramp of the fluid weight next to the wall."""
    span = float(upper) - float(lower)
    if abs(span) < np.finfo(np.float64).tiny:
        return np.where(np.asarray(values, dtype=np.float64) >= upper, 1.0, 0.0)
    phase = np.clip((np.asarray(values, dtype=np.float64) - lower) / span, 0.0, 1.0)
    return phase * phase * (3.0 - 2.0 * phase)


def remeshing_extension_length(
    freestream_velocity: np.ndarray | list[float] | tuple[float, ...],
    coupling_time_step_size: float,
    particle_spacing: float,
) -> float:
    """Return the outward extension ``L_remesh`` of the remeshing region, in m.

    It holds the distance a particle travels in one coupling interval, with a
    safety factor of 1.5, plus the two-spacing M4' support.
    """
    velocity = np.asarray(freestream_velocity, dtype=np.float64).reshape(3)
    dt = float(coupling_time_step_size)
    spacing = float(particle_spacing)
    if (
        not np.all(np.isfinite(velocity))
        or not np.isfinite(dt)
        or dt < 0.0
        or not np.isfinite(spacing)
        or spacing <= 0.0
    ):
        raise ValueError(
            "the free-stream velocity, coupling time step and particle spacing must be finite, "
            "with a non-negative time step and a positive spacing"
        )
    return 1.5 * float(np.linalg.norm(velocity)) * dt + 2.0 * spacing


@dataclass(frozen=True)
class TransferResult:
    """Particle counts, representation residuals and conservation of one vorticity transfer.

    Strengths are in m³/s, linear impulse in m⁴/s, angular impulse in m⁵/s and
    relative errors are dimensionless. The ``transfer_*`` fields describe the
    transfer-lattice particles: the raw mismatch before the moment correction,
    the applied correction and the remaining error of the total strength and
    impulse. The ``particle_set_*`` fields describe the removal of the weakest
    particles when the particle capacity is reached.
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
    blending_enabled: bool = False
    mapped_target_nodes: int = 0
    excluded_solid_target_nodes: int = 0
    excluded_solid_vortex_strength_l1: float = 0.0
    mapped_target_vortex_strength_l1: float = 0.0
    mapped_target_vortex_strength_net: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    maximum_mapped_vortex_strength: float = 0.0
    transfer_input_particles: int = 0
    transfer_output_particles: int = 0
    preserved_outer_particles: int = 0
    coalesced_outer_particles: int = 0
    removed_lattice_nodes: int = 0
    removed_vortex_strength_l1: float = 0.0
    removed_vortex_strength_fraction: float = 0.0
    particle_set_removed_particles: int = 0
    particle_set_removed_vortex_strength_fraction: float = 0.0
    particle_set_removed_velocity_bound: float = 0.0
    transfer_cfl: float = 0.0
    transfer_raw_vortex_strength_error: float = 0.0
    transfer_applied_vortex_strength_correction: float = 0.0
    transfer_conservation_error: float = 0.0
    transfer_vortex_strength_tolerance: float = 0.0
    transfer_raw_linear_impulse_error: float = 0.0
    transfer_applied_linear_impulse_correction: float = 0.0
    transfer_linear_impulse_error: float = 0.0
    transfer_linear_impulse_tolerance: float = 0.0
    transfer_raw_angular_impulse_error: float = 0.0
    transfer_applied_angular_impulse_correction: float = 0.0
    transfer_angular_impulse_error: float = 0.0
    transfer_applied_particle_strength_fraction: float = 0.0
    particle_set_transfer_raw_vortex_strength_error: float = 0.0
    particle_set_transfer_applied_vortex_strength_correction: float = 0.0
    particle_set_transfer_conservation_error: float = 0.0
    particle_set_transfer_raw_linear_impulse_error: float = 0.0
    particle_set_transfer_applied_linear_impulse_correction: float = 0.0
    particle_set_transfer_linear_impulse_error: float = 0.0
    particle_set_transfer_applied_particle_strength_fraction: float = 0.0
    representation_residual_before_removal: float | None = None
    representation_residual_after_removal: float | None = None
    strength_ratio: float = 0.0


def _validate_particle_sources(
    position: np.ndarray,
    cell_volume: np.ndarray,
    vorticity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize and check the FVM cell arrays used by a vorticity transfer.

    Parameters
    ----------
    position : ndarray, shape (M, 3)
        FVM cell centres in metres.
    cell_volume : ndarray, shape (M,)
        Positive cell volumes in m³.
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
        cell volume is not strictly positive.
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
    """Capture every particle field needed to restore the particle state."""
    return vpm.capture_particle_snapshot(slot=slot)


def _restore_particle_state(vpm, snapshot) -> None:
    """Restore a particle state captured by :func:`_particle_state_snapshot`."""
    vpm.restore_particle_snapshot(snapshot)


@dataclass(frozen=True)
class _AppliedParticleReductions:
    """Moments and norms of the particle state about to be written to the VPM."""

    transfer_moments: VortexMoments
    transfer_vortex_strength_l1: float
    cast_correction_l1: float
    transfer_position_scale: float
    output_vortex_strength_net: np.ndarray
    maximum_output_vortex_strength: float


def _reduce_applied_particle_state(
    position: np.ndarray,
    vortex_strength: np.ndarray,
    *,
    transfer_count: int,
    reference_vortex_strength: np.ndarray,
) -> _AppliedParticleReductions:
    """Reduce the particle state in storage precision, chunk by chunk, without float64 copies."""
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
    n_transfer = int(transfer_count)
    if not 0 <= n_transfer <= count:
        raise ValueError("transfer_count must lie inside the applied particle state")

    total = np.zeros(3, dtype=np.float64)
    linear_impulse = np.zeros(3, dtype=np.float64)
    angular_impulse = np.zeros(3, dtype=np.float64)
    output_net = np.zeros(3, dtype=np.float64)
    transfer_l1 = 0.0
    cast_correction_l1 = 0.0
    position_scale = 0.0
    maximum_strength = 0.0
    for start in range(0, count, _PARTICLE_REDUCTION_CHUNK_SIZE):
        stop = min(start + _PARTICLE_REDUCTION_CHUNK_SIZE, count)
        position_storage_chunk = applied_position[start:stop]
        if not np.all(np.isfinite(position_storage_chunk)):
            raise RuntimeError(
                "the vorticity transfer overflows the VPM position storage precision"
            )
        strength_chunk = applied_strength[start:stop].astype(np.float64, copy=False)
        if not np.all(np.isfinite(strength_chunk)):
            raise RuntimeError(
                "the vorticity transfer overflows the VPM strength storage precision"
            )
        magnitude = np.linalg.norm(strength_chunk, axis=1)
        output_net += strength_chunk.sum(axis=0, dtype=np.float64)
        maximum_strength = max(maximum_strength, float(magnitude.max(initial=0.0)))

        transfer_stop = min(stop, n_transfer)
        if transfer_stop <= start:
            continue
        local_count = transfer_stop - start
        transfer_strength = strength_chunk[:local_count]
        transfer_position = position_storage_chunk[:local_count].astype(np.float64, copy=False)
        position_cross_strength = np.cross(transfer_position, transfer_strength)
        total += transfer_strength.sum(axis=0, dtype=np.float64)
        linear_impulse += 0.5 * position_cross_strength.sum(axis=0, dtype=np.float64)
        angular_impulse += (1.0 / 3.0) * np.cross(
            transfer_position,
            position_cross_strength,
        ).sum(axis=0, dtype=np.float64)
        transfer_l1 += float(magnitude[:local_count].sum(dtype=np.float64))
        cast_correction_l1 += float(
            np.linalg.norm(
                transfer_strength - reference_strength[start:transfer_stop],
                axis=1,
            ).sum(dtype=np.float64)
        )
        position_scale = max(
            position_scale,
            float(np.linalg.norm(transfer_position, axis=1).max(initial=0.0)),
        )

    return _AppliedParticleReductions(
        transfer_moments=VortexMoments(
            total_vortex_strength=total,
            linear_impulse=linear_impulse,
            angular_impulse=angular_impulse,
        ),
        transfer_vortex_strength_l1=transfer_l1,
        cast_correction_l1=cast_correction_l1,
        transfer_position_scale=position_scale,
        output_vortex_strength_net=output_net,
        maximum_output_vortex_strength=maximum_strength,
    )


def transfer_vorticity_to_particles(
    vpm,
    *,
    lattice: TransferLattice,
    fvm_target_strength_at_node: Callable[[np.ndarray], np.ndarray],
    particle_fluid_weight: Callable[[np.ndarray], np.ndarray] | None,
    particle_in_solid: Callable[[np.ndarray], np.ndarray] | None,
    strength_threshold: float,
    core_radius_ratio: float,
    max_strength_ratio: float,
    kinematic_viscosity: float,
    freestream_speed: float,
    time_step_size: float,
    compute_diagnostics: bool,
    max_moment_correction_fraction: float | None = None,
    surface_strength_threshold: float | None = None,
) -> TransferResult:
    """Transfer the FVM vorticity to the particles of a GBD particle set in one step.

    :func:`transfer_on_lattice` builds the new particle state; it is checked for
    capacity and for conservation of the total strength and linear impulse in
    storage precision before it replaces the VPM particles.
    """
    if vpm.viscous_scheme != "gbd":
        raise ValueError("the vorticity transfer requires the GBD viscous scheme")
    viscosity = float(kinematic_viscosity)
    if not np.isfinite(viscosity) or viscosity < 0.0:
        raise ValueError("kinematic_viscosity must be finite and non-negative")
    if max_moment_correction_fraction is not None and (
        not np.isfinite(max_moment_correction_fraction)
        or not 0.0 < max_moment_correction_fraction <= 1.0
    ):
        raise ValueError("max_moment_correction_fraction must lie in (0, 1]")

    particles = vpm.particles
    n_before = int(particles.n_particles_total)
    # The storage precision distinguishes node roundoff from particles that are
    # off the lattice and need the fluid-side M4' stencil.
    existing_position = np.asarray(particles.position_cpu()).reshape(-1, 3)
    existing_strength = np.asarray(particles.vortex_strength_cpu(), dtype=np.float64).reshape(-1, 3)
    if len(existing_position) != n_before or len(existing_strength) != n_before:
        raise RuntimeError("VPM particle arrays do not match the active particle count")
    if not np.all(np.isfinite(existing_position)) or not np.all(np.isfinite(existing_strength)):
        raise RuntimeError("VPM particle state contains non-finite values")

    result = transfer_on_lattice(
        existing_position,
        existing_strength,
        lattice,
        fvm_target_strength_at_node=fvm_target_strength_at_node,
        particle_fluid_weight=particle_fluid_weight,
        particle_in_solid=particle_in_solid,
        strength_threshold=strength_threshold,
        surface_strength_threshold=surface_strength_threshold,
        core_radius_ratio=core_radius_ratio,
        max_strength_ratio=max_strength_ratio,
        maximum_particle_count=int(particles.capacity),
        freestream_speed=freestream_speed,
        time_step_size=time_step_size,
        compute_diagnostics=compute_diagnostics,
    )
    if result.particle_set_removed_count:
        raise RuntimeError(
            "the vorticity transfer reached the VPM particle capacity and would "
            f"remove {result.particle_set_removed_count:,} particles; increase the capacity "
            "or reduce the number of particles"
        )
    n_after = result.particle_count
    if n_after > int(particles.capacity):
        raise RuntimeError(
            f"the vorticity transfer produced {n_after:,} particles, exceeding "
            f"the VPM capacity {int(particles.capacity):,}"
        )

    dtype = np.dtype(vpm.np_dtype)
    applied_position = np.ascontiguousarray(result.position, dtype=dtype)
    applied_strength = np.ascontiguousarray(result.vortex_strength, dtype=dtype)
    applied_core_radius = np.ascontiguousarray(result.core_radius, dtype=dtype)
    applied_particle_volume = np.ascontiguousarray(result.particle_volume, dtype=dtype)
    transfer_output_count = min(int(result.transfer_output_count), n_after)
    target_moments = result.conservation_target_moments
    raw_moments = result.conservation_raw_moments
    if target_moments is None or raw_moments is None:
        raise RuntimeError("the vorticity transfer returned no reference moments")
    reductions = _reduce_applied_particle_state(
        applied_position,
        applied_strength,
        transfer_count=transfer_output_count,
        reference_vortex_strength=result.vortex_strength,
    )
    applied_moments = reductions.transfer_moments
    raw_conservation = vortex_moment_residual(target_moments, raw_moments)
    applied_conservation = vortex_moment_residual(raw_moments, applied_moments)
    conservation = vortex_moment_residual(target_moments, applied_moments)
    moment_correction_fraction = float(
        result.conservation_applied_particle_strength_fraction
        + reductions.cast_correction_l1 / (result.conservation_reference_strength_l1 + 1.0e-30)
    )
    if not np.isfinite(moment_correction_fraction):
        raise RuntimeError("the vorticity transfer produced a non-finite moment correction")
    if (
        max_moment_correction_fraction is not None
        and moment_correction_fraction > max_moment_correction_fraction
    ):
        raise RuntimeError(
            "the moment correction of the vorticity transfer exceeds its limit: "
            f"fraction={moment_correction_fraction:.6e}, "
            f"limit={max_moment_correction_fraction:.6e}"
        )
    storage_epsilon = float(np.finfo(dtype).eps)
    # Strengths below the strength threshold are unresolved; this floor keeps a
    # round-off field, such as a uniform free stream, from tightening the check.
    reference_strength_l1 = max(
        float(result.conservation_reference_strength_l1), float(strength_threshold)
    )
    position_scale = max(
        reductions.transfer_position_scale,
        float(lattice.particle_spacing),
    )
    transfer_vortex_strength_tolerance = 32.0 * storage_epsilon * reference_strength_l1
    transfer_linear_impulse_tolerance = (
        32.0 * storage_epsilon * reference_strength_l1 * position_scale
    )
    strength_error = float(conservation["total_vortex_strength"])
    impulse_error = float(conservation["linear_impulse"])
    if strength_error > transfer_vortex_strength_tolerance + np.finfo(np.float64).tiny:
        raise RuntimeError(
            "the vorticity transfer does not conserve the total strength in storage precision: "
            f"error={strength_error:.6e}, "
            f"tolerance={transfer_vortex_strength_tolerance:.6e}"
        )
    if impulse_error > transfer_linear_impulse_tolerance + np.finfo(np.float64).tiny:
        raise RuntimeError(
            "the vorticity transfer does not conserve the linear impulse in storage precision: "
            f"error={impulse_error:.6e}, "
            f"tolerance={transfer_linear_impulse_tolerance:.6e}"
        )

    # Check the complete particle count before changing the VPM, so a wrong
    # result leaves the previous particle state untouched.
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
            (existing_position >= lattice.remeshing_region_bounds[::2])
            & (existing_position <= lattice.remeshing_region_bounds[1::2]),
            axis=1,
        )
        removed_mask = ~preserved
    coalesced_input = np.asarray(
        result.coalesced_outer_input_indices,
        dtype=np.int64,
    ).reshape(-1)
    if len(coalesced_input):
        if np.any((coalesced_input < 0) | (coalesced_input >= n_before)):
            raise RuntimeError("the vorticity transfer returned an invalid merged-particle index")
        removed_mask[coalesced_input] = True
    removed_count = int(np.count_nonzero(removed_mask))
    if n_before - removed_count + transfer_output_count != n_after:
        raise RuntimeError("the vorticity transfer returned an inconsistent particle count")
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
            f"VPM particle count after the vorticity transfer is "
            f"{int(particles.n_particles_total)}, expected {n_after}"
        )

    injected_net = reductions.transfer_moments.total_vortex_strength
    output_net = reductions.output_vortex_strength_net
    particle_set_raw_conservation = result.particle_set_conservation_raw_mismatch
    particle_set_applied_conservation = result.particle_set_conservation_applied_correction
    particle_set_conservation = result.particle_set_conservation_residual
    maximum_strength = reductions.maximum_output_vortex_strength
    return TransferResult(
        n_particles_before=n_before,
        n_particles_retained=int(result.preserved_outer_count),
        n_particles_removed=removed_count,
        n_particles_blended=int(result.transfer_input_count),
        n_particles_injected=transfer_output_count,
        n_particles_after=n_after,
        injected_vortex_strength_l1=reductions.transfer_vortex_strength_l1,
        injected_vortex_strength_net=injected_net,
        replaced_vortex_strength_l1=float(
            np.linalg.norm(replaced_strength, axis=1).sum(dtype=np.float64)
        ),
        replaced_vortex_strength_net=replaced_strength.sum(axis=0, dtype=np.float64),
        state_change_vortex_strength_net=output_net - input_net,
        blending_enabled=bool(
            np.any((lattice.blending_weight > 0.0) & (lattice.blending_weight < 1.0))
        ),
        mapped_target_nodes=transfer_output_count,
        excluded_solid_target_nodes=int(np.count_nonzero(lattice.solid_interior)),
        excluded_solid_vortex_strength_l1=float(result.excluded_target_vortex_strength_l1),
        mapped_target_vortex_strength_l1=reductions.transfer_vortex_strength_l1,
        mapped_target_vortex_strength_net=injected_net,
        maximum_mapped_vortex_strength=maximum_strength,
        transfer_input_particles=int(result.transfer_input_count),
        transfer_output_particles=transfer_output_count,
        preserved_outer_particles=int(result.preserved_outer_count),
        coalesced_outer_particles=int(result.coalesced_outer_count),
        removed_lattice_nodes=int(result.removed_node_count),
        removed_vortex_strength_l1=float(result.removed_vortex_strength_l1),
        removed_vortex_strength_fraction=float(result.removed_vortex_strength_fraction),
        particle_set_removed_particles=int(result.particle_set_removed_count),
        particle_set_removed_vortex_strength_fraction=float(
            result.particle_set_removed_vortex_strength_fraction
        ),
        particle_set_removed_velocity_bound=float(result.particle_set_removed_velocity_bound),
        transfer_cfl=float(result.transfer_cfl),
        transfer_raw_vortex_strength_error=float(
            raw_conservation.get("total_vortex_strength", 0.0)
        ),
        transfer_applied_vortex_strength_correction=float(
            applied_conservation.get("total_vortex_strength", 0.0)
        ),
        transfer_conservation_error=float(conservation.get("total_vortex_strength", 0.0)),
        transfer_vortex_strength_tolerance=transfer_vortex_strength_tolerance,
        transfer_raw_linear_impulse_error=float(raw_conservation.get("linear_impulse", 0.0)),
        transfer_applied_linear_impulse_correction=float(
            applied_conservation.get("linear_impulse", 0.0)
        ),
        transfer_linear_impulse_error=float(conservation.get("linear_impulse", 0.0)),
        transfer_linear_impulse_tolerance=transfer_linear_impulse_tolerance,
        transfer_raw_angular_impulse_error=float(raw_conservation.get("angular_impulse", 0.0)),
        transfer_applied_angular_impulse_correction=float(
            applied_conservation.get("angular_impulse", 0.0)
        ),
        transfer_angular_impulse_error=float(conservation.get("angular_impulse", 0.0)),
        transfer_applied_particle_strength_fraction=moment_correction_fraction,
        particle_set_transfer_raw_vortex_strength_error=float(
            particle_set_raw_conservation.get("total_vortex_strength", 0.0)
        ),
        particle_set_transfer_applied_vortex_strength_correction=float(
            particle_set_applied_conservation.get("total_vortex_strength", 0.0)
        ),
        particle_set_transfer_conservation_error=float(
            particle_set_conservation.get("total_vortex_strength", 0.0)
        ),
        particle_set_transfer_raw_linear_impulse_error=float(
            particle_set_raw_conservation.get("linear_impulse", 0.0)
        ),
        particle_set_transfer_applied_linear_impulse_correction=float(
            particle_set_applied_conservation.get("linear_impulse", 0.0)
        ),
        particle_set_transfer_linear_impulse_error=float(
            particle_set_conservation.get("linear_impulse", 0.0)
        ),
        particle_set_transfer_applied_particle_strength_fraction=float(
            result.particle_set_conservation_applied_particle_strength_fraction
        ),
        representation_residual_before_removal=result.representation_residual_before_removal,
        representation_residual_after_removal=result.representation_residual_after_removal,
        strength_ratio=float(result.strength_ratio),
    )


def _transfer_log_record(step: int, result: TransferResult) -> log_style.Event:
    """Keep the immutable transfer result until an enabled report needs its rows."""
    return log_style.Event("vorticity transfer", lambda: _transfer_log_rows(step, result))


def _transfer_log_rows(step: int, result: TransferResult) -> tuple[log_style.Row, ...]:
    """Build the log rows of one vorticity transfer.

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
        ("blending", "on" if result.blending_enabled else "off"),
        ("particles, before", f"{result.n_particles_before:,}"),
        ("particles, removed", f"{result.n_particles_removed:,}"),
        ("particles, blended", f"{result.n_particles_blended:,}"),
        ("particles, injected", f"{result.n_particles_injected:,}"),
        ("particles, after", f"{result.n_particles_after:,}"),
        ("lattice nodes, active", f"{result.mapped_target_nodes:,}"),
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
            ("remeshing region, particles in", f"{result.transfer_input_particles:,}"),
            ("transfer-lattice particles", f"{result.transfer_output_particles:,}"),
            ("outer wake, kept", f"{result.preserved_outer_particles:,}"),
            ("outer wake, merged at lattice nodes", f"{result.coalesced_outer_particles:,}"),
            ("lattice nodes, removed", f"{result.removed_lattice_nodes:,}"),
            (
                "removed strength, fraction",
                f"{100.0 * result.removed_vortex_strength_fraction:.3f}",
                "%",
            ),
            ("transfer CFL, U dt / L_remesh", f"{result.transfer_cfl:.3f}"),
            (
                "moment correction, total strength before",
                f"{result.transfer_raw_vortex_strength_error:.3e}",
            ),
            (
                "moment correction, total strength applied",
                f"{result.transfer_applied_vortex_strength_correction:.3e}",
            ),
            (
                "moment correction, total strength after",
                f"{result.transfer_conservation_error:.3e}",
            ),
            (
                "moment correction, total strength tolerance",
                f"{result.transfer_vortex_strength_tolerance:.3e}",
            ),
            (
                "moment correction, linear impulse before",
                f"{result.transfer_raw_linear_impulse_error:.3e}",
            ),
            (
                "moment correction, linear impulse applied",
                f"{result.transfer_applied_linear_impulse_correction:.3e}",
            ),
            (
                "moment correction, linear impulse after",
                f"{result.transfer_linear_impulse_error:.3e}",
            ),
            (
                "moment correction, linear impulse tolerance",
                f"{result.transfer_linear_impulse_tolerance:.3e}",
            ),
            (
                "moment correction, strength fraction",
                f"{100.0 * result.transfer_applied_particle_strength_fraction:.3f}",
                "%",
            ),
            (
                "strength ratio, kappa",
                f"{result.strength_ratio:.3f}",
            ),
        )
    )
    if result.representation_residual_after_removal is not None:
        rows.extend(
            (
                (
                    "representation residual, before removal",
                    result.representation_residual_before_removal,
                ),
                (
                    "representation residual, after removal",
                    f"{result.representation_residual_after_removal:.3e}",
                ),
            )
        )
    return tuple(rows)


class VorticityTransfer:
    """Vorticity transfer from the accepted cell-centred FVM state to the particles.

    :class:`CoupledSolver` constructs it once both discretizations are known.
    The FVM target strength of each lattice node is the Stokes integral of the
    reconstructed FVM velocity over the faces of its lattice cube, ``Γ^FVM_j``.
    Inside the transfer region the FVM vorticity replaces the particle vorticity
    according to the blending weight; the outer wake is kept.

    Parameters
    ----------
    coupler : CoupledSolver
        Initialized coupled solver providing the transfer settings, the FVM domain
        bounds, the kinematic viscosity in m²/s, the particle spacing, the core
        radius ratio and the coupling time step in s.

    Attributes
    ----------
    particle_spacing : float
        Particle and transfer-lattice spacing ``h`` in m.
    core_radius_ratio : float
        Dimensionless ``sigma/h`` of the transfer-lattice particles.
    coupling_time_step_size : float
        Coupling interval ``Δt_VPM``, in s.
    step : int
        Number of transfer calls attempted. Incremented at call entry.
    last_transfer_surface_flow : dict[str, float]
        Mean outward normal FVM velocity on each face of the transfer surface, in m/s.
    last_transfer_surface_vorticity : dict[str, float]
        Mean normal FVM vorticity on each face of the transfer surface, relative to the
        mean vorticity magnitude.

    Raises
    ------
    RuntimeError
        If the coupled solver has not resolved the FVM domain bounds, the fluid
        properties, the particle spacing and core radius ratio, or the coupling
        time step.
    ValueError
        If the VPM does not use GBD diffusion with an absolute threshold.

    Notes
    -----
    It is used through the coupled solver. Call :meth:`setup` once for an FVM
    mesh, then :meth:`transfer` at accepted coupling times only. A transfer
    changes the VPM particles and their caches; :meth:`setup` only builds the
    transfer lattice, the interpolation stencils and the solid masks.
    """

    def __init__(self, coupler: CoupledSolver) -> None:
        """Read the transfer settings from an initialized coupled solver."""
        cfg = coupler.config
        if coupler.kinematic_viscosity is None or coupler.fvm_domain_bounds is None:
            raise RuntimeError("VorticityTransfer requires initialized FVM and VPM state")
        self.config = cfg
        numerics = coupler.vpm_case.numerics
        viscous = numerics.viscous
        # Fluid-side M4' remeshing of off-lattice particles, from the VPM GBD grid.
        vpm = getattr(coupler, "vpm_solver", None)
        self._wall_scatter_correction = (
            None if vpm is None else vpm.physics.compute_m4_wall_corrections
        )
        if viscous.scheme != "gbd":
            raise ValueError("the vorticity transfer requires the GBD viscous scheme")
        if not np.isfinite(coupler.vpm_core_radius_ratio):
            raise RuntimeError("VorticityTransfer requires the resolved VPM core-radius ratio")
        self.core_radius_ratio = float(coupler.vpm_core_radius_ratio)
        if not np.isfinite(coupler.vpm_particle_spacing):
            raise RuntimeError("VorticityTransfer requires the resolved VPM particle spacing")
        self.particle_spacing = float(coupler.vpm_particle_spacing)
        induction = numerics.induction
        self._slip_slab = getattr(induction, "method", None) == "slip_slab"
        self._slip_z = (float(induction.z_min), float(induction.z_max)) if self._slip_slab else None
        self.blend_start_depth, self.blend_end_depth = cfg.blend_depths(self.particle_spacing)
        self.strength_threshold = float(cfg.transfer_removal_threshold) * self.particle_spacing**3
        if str(viscous.gbd_threshold_mode).lower() != "absolute":
            raise ValueError(
                "the vorticity transfer requires GBD threshold_mode='absolute', so the "
                "removal threshold at the transfer surface is a resolved physical value"
            )
        self.surface_strength_threshold = float(viscous.gbd_threshold)
        if self.surface_strength_threshold > self.strength_threshold:
            raise ValueError(
                "the strength threshold h³ transfer_removal_threshold must be at least the "
                "GBD removal threshold of the VPM"
            )
        self.max_strength_ratio = float(cfg.max_strength_ratio)
        self.kinematic_viscosity = float(coupler.kinematic_viscosity)
        self.coupling_time_step_size = float(coupler.vpm_time_step_size)
        self.remeshing_extension_length = remeshing_extension_length(
            cfg.freestream_velocity,
            self.coupling_time_step_size,
            self.particle_spacing,
        )
        self.diagnostic_interval = int(cfg.transfer_diagnostic_interval_steps)
        self.max_moment_correction_fraction = float(cfg.max_moment_correction_fraction)
        self._fvm_domain_bounds = np.asarray(coupler.fvm_domain_bounds, dtype=np.float64)
        self._transfer_region_bounds: np.ndarray | None = None
        self._cell_tree: cKDTree | None = None
        self._cell_centre: np.ndarray | None = None
        self._cell_volume: np.ndarray | None = None
        self._velocity_interpolator: FVMVelocityInterpolator | None = None
        self._target_stencils: list | None = None
        self._fvm_solid_mask: np.ndarray | None = None
        self._solid_bodies: tuple = ()
        self.solid_boundary: SolidBoundary | None = None
        self._lattice_anchor: np.ndarray | None = None
        self._transfer_lattice: TransferLattice | None = None
        self._face_cells: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._fvm_transfer_cell_mask: np.ndarray | None = None
        self.step = 0
        self.last_transfer_surface_flow: dict[str, float] = {}
        self.last_transfer_surface_vorticity: dict[str, float] = {}

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
            Particle or lattice-node coordinates in metres.
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
        """Index the FVM cells that lie on each face of the transfer surface.

        Parameters
        ----------
        bounds : ndarray, shape (6,)
            ``(xmin, xmax, ymin, ymax, zmin, zmax)`` bounds in metres.

        Returns
        -------
        dict[str, tuple[ndarray, ndarray]]
            Face-name to ``(cell_indices, outward_normal)`` mapping. Indices
            refer to the FVM cell-centre array and normals are dimensionless.
            Faces without FVM cells are omitted.
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
        """Refresh the cached cells of each face of the transfer surface."""
        bounds = self._transfer_region_bounds
        self._face_cells = {} if bounds is None else self._face_cell_index(bounds)

    def check_transfer_surface_flow(self, velocity: np.ndarray) -> dict[str, float]:
        """Return the mean outward normal velocity on each face of the transfer surface.

        Parameters
        ----------
        velocity : ndarray, shape (n_cells, 3)
            Accepted cell-centred Cartesian FVM velocity in m/s, ordered like
            the cell centres supplied to :meth:`setup`.

        Returns
        -------
        dict[str, float]
            Mean ``u dot n`` in m/s keyed by ``xmin`` through ``zmax`` for
            every face with FVM cells.
        """
        values = np.asarray(velocity, dtype=np.float64).reshape(-1, 3)
        return {
            name: float(np.mean(values[index] @ normal))
            for name, (index, normal) in self._face_cells.items()
            if index.max(initial=-1) < len(values)
        }

    def check_transfer_surface_vorticity(self, velocity_gradient: np.ndarray) -> dict[str, float]:
        """Return the mean normal vorticity on each face of the transfer surface.

        Vorticity crossing the transfer surface is cut by the replacement; a
        small value means the vortex lines leave the transfer region mostly
        through the outer wake.

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
        """Build the transfer lattice, the interpolation stencils and the solid masks.

        Parameters
        ----------
        fvm : FVMSolver
            Initialized solver providing global cell-centre coordinates
            ``(M, 3)`` in m, cell volumes ``(M,)`` in m³, and boundary
            geometry.

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
        self._target_stencils = None
        # FVM cells and walls are read in the frame of a translating mesh.
        frame = MeshFrame(fvm)
        self._transfer_region_bounds = self.config.transfer_region(
            self._fvm_domain_bounds, self.particle_spacing, self._slip_slab
        )
        self._cell_centre = frame.to_frame(
            np.asarray(fvm.gather_cell_centre_coordinates(), dtype=np.float64).reshape(-1, 3)
        )
        self._cell_volume = np.asarray(fvm.gather_cell_volume(), dtype=np.float64).reshape(-1)

        # These partitioned getters are collective, even though only rank zero
        # receives the assembled arrays. Keep their call order identical.
        wall_patches = [
            boundary_condition.name
            for boundary_condition in fvm.case.boundaries
            if boundary_condition.mesh_type == "wall"
        ]
        wall_triangles = fvm.gather_wall_surface_triangles()
        if wall_triangles is not None:
            wall_triangles = frame.to_frame(wall_triangles)

        from openonda.core.simulation.parallel import collective_phase

        comm = fvm.parallel.comm
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

            self._solid_bodies = ()
            if wall_triangles is not None and len(wall_triangles):
                self._solid_bodies = (TriangulatedWall(wall_triangles, self._fvm_domain_bounds),)
            elif wall_patches:
                raise RuntimeError("Body-fitted transfer requires the FVM wall surface triangles")
            self.solid_boundary = SolidBoundary(self._solid_bodies) if self._solid_bodies else None
            # One mesh-derived lattice phase for every geometry, independent of cell
            # ordering, wall patch names and inferred body dimensions.
            self._lattice_anchor = self._cell_centre.min(axis=0)
            self._velocity_interpolator = FVMVelocityInterpolator(
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
                raise RuntimeError("the vorticity transfer requires an FVM cell search tree")

            def mesh_weight_at_node(points: np.ndarray) -> np.ndarray:
                # The distance to a cell centre is not a fluid-domain test:
                # coarse or anisotropic cells span many particles. Solid
                # exclusion uses the wall geometry, independently of h.
                domain = self._fvm_domain_bounds
                roundoff = 64 * np.finfo(float).eps * np.maximum(1.0, np.abs(domain))
                return np.all(
                    (points >= domain[::2] - roundoff[::2])
                    & (points <= domain[1::2] + roundoff[1::2]),
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

            self._transfer_lattice = build_transfer_lattice(
                self._transfer_region_bounds,
                self.particle_spacing,
                remeshing_extension_length=self.remeshing_extension_length,
                blend_start_depth=self.blend_start_depth,
                blend_end_depth=self.blend_end_depth,
                lattice_anchor=self._lattice_anchor,
                mesh_weight_at_node=mesh_weight_at_node,
                fluid_weight_at_node=fluid_weight_at_node if has_solid else None,
                interior_at_node=interior_at_node if has_solid else None,
                solid_boundary=self.solid_boundary,
                wall_scatter_correction=self._wall_scatter_correction,
                slip_slab=self._slip_slab,
            )
            self._fvm_solid_mask = self._points_in_solid(
                self._cell_centre,
                include_boundary=True,
            )
            inside = np.all(
                (self._cell_centre >= self._transfer_region_bounds[::2])
                & (self._cell_centre <= self._transfer_region_bounds[1::2]),
                axis=1,
            )
            self._fvm_transfer_cell_mask = inside & ~self._fvm_solid_mask
            fluid_cell_count = int(np.count_nonzero(self._fvm_transfer_cell_mask))
            if fluid_cell_count == 0:
                raise ValueError("the transfer region contains no FVM fluid cell centre")
            self._build_face_cell_index()
            lattice_count = (
                len(self._transfer_lattice.positions) if self._transfer_lattice is not None else 0
            )
            geometry_rows: list[log_style.Row] = []
            if self._transfer_lattice is not None:
                velocity = np.asarray(self.config.freestream_velocity, dtype=np.float64)
                streamwise_axis = int(np.argmax(np.abs(velocity)))
                downstream_is_maximum = velocity[streamwise_axis] >= 0.0
                axis_name = "xyz"[streamwise_axis]
                lattice = self._transfer_lattice
                planes = lattice.origin[streamwise_axis] + self.particle_spacing * np.arange(
                    lattice.shape[streamwise_axis]
                )
                blend_weight_grid = lattice.blending_weight.reshape(lattice.shape)
                plane_blend_weight = np.max(
                    np.moveaxis(blend_weight_grid, streamwise_axis, 0).reshape(len(planes), -1),
                    axis=1,
                )
                fvm_transfer_planes = planes[plane_blend_weight > 0.0]
                face_index = 2 * streamwise_axis + int(downstream_is_maximum)
                face = self._transfer_region_bounds[face_index]
                remeshing_edge = lattice.remeshing_region_bounds[face_index]
                beyond_face = planes > face if downstream_is_maximum else planes < face
                inside_remeshing = (
                    planes <= remeshing_edge if downstream_is_maximum else planes >= remeshing_edge
                )
                beyond_remeshing = ~inside_remeshing
                support_edge = remeshing_edge + (
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

                fluid_cell_coordinate = self._cell_centre[
                    self._fvm_transfer_cell_mask, streamwise_axis
                ]
                geometry_rows.extend(
                    (
                        (
                            f"last FVM cell centre, {axis_name}",
                            f"{downstream_extreme(fluid_cell_coordinate):.6f}",
                            "m",
                        ),
                        (
                            f"transfer surface, {axis_name}",
                            f"{face:.6f}",
                            "m",
                        ),
                        (
                            f"last plane with FVM vorticity, {axis_name}",
                            f"{downstream_extreme(fvm_transfer_planes):.6f}",
                            "m",
                        ),
                        (
                            f"first plane of the inset band, {axis_name}",
                            f"{upstream_extreme(planes[beyond_face]):.6f}",
                            "m",
                        ),
                        (
                            f"last plane of the remeshing region, {axis_name}",
                            f"{downstream_extreme(planes[inside_remeshing]):.6f}",
                            "m",
                        ),
                        (
                            f"first plane of the kept particles, {axis_name}",
                            f"{upstream_extreme(planes[beyond_remeshing]):.6f}",
                            "m",
                        ),
                        (
                            f"last plane of the M4' support, {axis_name}",
                            f"{downstream_extreme(planes[inside_m4_support]):.6f}",
                            "m",
                        ),
                        (
                            f"last transfer-lattice plane, {axis_name}",
                            f"{downstream_extreme(planes):.6f}",
                            "m",
                        ),
                    )
                )
            logger.info(
                format_coupler_log(
                    "replacement region",
                    ("fvm fluid cells", f"{fluid_cell_count:,}"),
                    *(
                        (("blending", "off"),)
                        if self.blend_end_depth == 0.0
                        else (
                            ("blend start depth, delta_0", f"{self.blend_start_depth:.4g}", "m"),
                            ("blend end depth, delta_1", f"{self.blend_end_depth:.4g}", "m"),
                        )
                    ),
                    (
                        "remeshing extension, L_remesh",
                        f"{self.remeshing_extension_length:.4g}",
                        "m",
                    ),
                    ("transfer-lattice nodes", f"{lattice_count:,}"),
                    *geometry_rows,
                    ("fvm target strength", "Stokes integral of the reconstructed FVM velocity"),
                )
            )

    def _transfer_on_lattice(
        self,
        vpm,
        *,
        fvm_velocity: np.ndarray,
        fvm_velocity_gradient: np.ndarray,
    ) -> TransferResult:
        """Transfer the FVM vorticity of one accepted FVM state on the transfer lattice."""
        if self._transfer_lattice is None or self._velocity_interpolator is None:
            raise RuntimeError("the transfer lattice was not initialized")
        has_solid = self.solid_boundary is not None

        def fluid_weight(points: np.ndarray) -> np.ndarray:
            return _smoothstep(
                self._signed_solid_distance(points),
                -self.particle_spacing,
                0.0,
            )

        def in_solid(points: np.ndarray) -> np.ndarray:
            return self._points_in_solid(points, include_boundary=False)

        lattice = self._transfer_lattice
        slab_bounds = self._transfer_region_bounds if self._slip_slab else None
        # Only the stencil geometry is cached: every coupling iteration supplies new fields.
        needed = (lattice.mesh_weight > 0.0) | (lattice.fluid_weight < 1.0)
        if slab_bounds is not None:
            needed &= (lattice.positions[:, 2] >= slab_bounds[4] - 1e-10) & (
                lattice.positions[:, 2] <= slab_bounds[5] + 1e-10
            )
        if self._target_stencils is None:
            stencils = []
            positions = lattice.positions[needed]
            for axis in range(3):
                for sign in (1.0, -1.0):
                    query = positions.copy()
                    query[:, axis] += sign * 0.5 * self.particle_spacing
                    reflected = np.zeros(len(query), dtype=bool)
                    if slab_bounds is not None:
                        below = query[:, 2] < slab_bounds[4]
                        above = query[:, 2] > slab_bounds[5]
                        query[below, 2] = 2.0 * slab_bounds[4] - query[below, 2]
                        query[above, 2] = 2.0 * slab_bounds[5] - query[above, 2]
                        reflected = below | above
                        if np.any((query[:, 2] < slab_bounds[4]) | (query[:, 2] > slab_bounds[5])):
                            raise RuntimeError(
                                "the FVM target strength needs more than one slab reflection"
                            )
                    wall_weight = (
                        _smoothstep(self._signed_solid_distance(query), 0.0, self.particle_spacing)
                        if has_solid
                        else np.ones(len(query))
                    )
                    fluid = wall_weight > 0.0
                    stencil = self._velocity_interpolator.prepare(query[fluid])
                    stencils.append((axis, sign, fluid, wall_weight[fluid], reflected, stencil))
            self._target_stencils = stencils

        def fvm_target_at_node(points: np.ndarray) -> np.ndarray:
            # Integrate n x u on the six face midpoints of each lattice cube, as
            # transfer_lattice.fvm_target_strength does. The velocity is zero in the solid.
            strength = np.zeros((np.count_nonzero(needed), 3), dtype=np.float64)
            for start in range(0, len(self._target_stencils), 2):
                velocities = []
                axis = self._target_stencils[start][0]
                for (
                    _axis,
                    _sign,
                    fluid,
                    weight,
                    reflected,
                    stencil,
                ) in self._target_stencils[start : start + 2]:
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
            target = np.zeros_like(points)
            target[needed] = strength
            return target

        return transfer_vorticity_to_particles(
            vpm,
            lattice=self._transfer_lattice,
            fvm_target_strength_at_node=fvm_target_at_node,
            particle_fluid_weight=fluid_weight if has_solid else None,
            particle_in_solid=in_solid if has_solid else None,
            strength_threshold=self.strength_threshold,
            surface_strength_threshold=self.surface_strength_threshold,
            core_radius_ratio=self.core_radius_ratio,
            max_strength_ratio=self.max_strength_ratio,
            kinematic_viscosity=self.kinematic_viscosity,
            freestream_speed=float(np.linalg.norm(self.config.freestream_velocity_vector)),
            time_step_size=self.coupling_time_step_size,
            compute_diagnostics=(self.step == 1 or self.step % self.diagnostic_interval == 0),
            max_moment_correction_fraction=self.max_moment_correction_fraction,
        )

    def transfer(
        self,
        vpm: VPMSolver,
        velocity: np.ndarray,
        velocity_gradient: np.ndarray,
    ) -> TransferResult:
        """Transfer the vorticity of one accepted FVM state to the particles.

        Parameters
        ----------
        vpm : VPMSolver
            Particle solver of the root rank. Its particle fields and source
            caches are replaced in one step on success.
        velocity : ndarray, shape (M, 3)
            Accepted cell-centred FVM velocity in m/s, ordered like the FVM
            cells read by :meth:`setup`.
        velocity_gradient : ndarray, shape (M, 3, 3)
            Accepted FVM gradient in 1/s with
            ``G[i, j] = d(u_j)/d(x_i)``; with the velocity it reconstructs the
            FVM velocity at the faces of each lattice cube.

        Returns
        -------
        TransferResult
            Particle counts, representation residuals, weak-strength removal and
            conservation of the total strength and impulse of this transfer.

        Raises
        ------
        RuntimeError
            If :meth:`setup` has not run, or a conservation or capacity check
            of the new particle state fails.
        ValueError
            If the velocity or gradient count differs from the FVM cell count.

        Notes
        -----
        The method increments :attr:`step`, updates the transfer-surface
        diagnostics and replaces the accepted VPM particle state. The new state
        is checked before the particles are replaced, so a failed check leaves
        them unchanged; no FVM array is modified.
        """
        self.step += 1
        if (
            self._transfer_region_bounds is None
            or self._cell_centre is None
            or self._cell_volume is None
        ):
            raise RuntimeError("VorticityTransfer.setup() has not read the FVM cells")
        velocity_values = np.asarray(velocity, dtype=np.float64).reshape(-1, 3)
        gradient_values = np.asarray(velocity_gradient, dtype=np.float64).reshape(-1, 3, 3)
        if len(velocity_values) != len(self._cell_centre) or len(gradient_values) != len(
            self._cell_centre
        ):
            raise ValueError("FVM velocity, gradient, and cell-centre counts must match")

        self.last_transfer_surface_flow = self.check_transfer_surface_flow(velocity_values)
        self.last_transfer_surface_vorticity = self.check_transfer_surface_vorticity(
            gradient_values
        )
        if self._lattice_anchor is None:
            raise RuntimeError("VorticityTransfer.setup() did not resolve a lattice anchor")
        result = self._transfer_on_lattice(
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
    "remeshing_extension_length",
    "transfer_vorticity_to_particles",
]
