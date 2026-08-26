"""Stable fixed-lattice FVM-to-VPM renewal used by the long cube-flow runs.

This module preserves the numerical mechanism that existed immediately before
commit ``34828ccc`` while presenting it as a small, side-effect-free API.  It
does not mutate a VPM solver and is therefore safe to certify before wiring it
into the production coupler.

The renewal is deliberately compatible with grid-based diffusion (GBD): every
renewed particle is placed on the fixed lattice with the base core radius.
There is no core-spreading/particle-age transport in this implementation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from numba import njit
import numpy as np
from scipy.ndimage import gaussian_filter

ArrayFunction = Callable[[np.ndarray], np.ndarray]

CORE_RADIUS_RATIO = 1.0
M4_PRIME_SUPPORT_CELLS = 2.0
DEFAULT_AMPLIFICATION_CAP = 2.0
_ALIGNMENT_TOLERANCE_CELLS = 1.0e-5


def required_buffer_length(
    freestream_speed: float,
    time_step_size: float,
    particle_spacing: float,
    safety_factor: float = 1.5,
) -> float:
    """Return the historical advection buffer plus complete M4' support."""
    spacing = _positive_finite("particle_spacing", particle_spacing)
    safety = _positive_finite("safety_factor", safety_factor)
    speed = _nonnegative_finite("freestream_speed", abs(float(freestream_speed)))
    time_step = _nonnegative_finite("time_step_size", abs(float(time_step_size)))
    return float(safety * speed * time_step + M4_PRIME_SUPPORT_CELLS * spacing)


def maximum_stable_time_step(
    freestream_speed: float,
    buffer_length: float,
    particle_spacing: float,
    safety_factor: float = 1.5,
) -> float:
    """Invert :func:`required_buffer_length` for a supplied renewal buffer."""
    spacing = _positive_finite("particle_spacing", particle_spacing)
    safety = _positive_finite("safety_factor", safety_factor)
    speed = _nonnegative_finite("freestream_speed", abs(float(freestream_speed)))
    buffer = _nonnegative_finite("buffer_length", buffer_length)
    if speed < np.finfo(np.float64).tiny:
        return float("inf")
    return float(max(buffer - M4_PRIME_SUPPORT_CELLS * spacing, 0.0) / (safety * speed))


def m4_prime(distance: np.ndarray | float) -> np.ndarray:
    """Evaluate the interpolating M4' kernel on dimensionless distances."""
    q = np.abs(np.asarray(distance, dtype=np.float64))
    weight = np.zeros_like(q)
    inner = q < 1.0
    outer = (q >= 1.0) & (q < 2.0)
    weight[inner] = 1.0 - 2.5 * q[inner] ** 2 + 1.5 * q[inner] ** 3
    weight[outer] = 0.5 * (1.0 - q[outer]) * (2.0 - q[outer]) ** 2
    return weight


@njit(cache=True, fastmath=False)
def _m4_prime_scalar(distance: float) -> float:
    q = abs(distance)
    if q < 1.0:
        return 1.0 - 2.5 * q * q + 1.5 * q * q * q
    if q < 2.0:
        return 0.5 * (1.0 - q) * (2.0 - q) * (2.0 - q)
    return 0.0


@njit(cache=True, fastmath=False)
def _scatter_unaligned_m4_prime(
    relative_position: np.ndarray,
    lower_stencil_index: np.ndarray,
    vortex_strength: np.ndarray,
    shape: np.ndarray,
) -> np.ndarray:
    result = np.zeros((shape[0], shape[1], shape[2], 3), dtype=np.float64)
    for donor in range(len(relative_position)):
        for offset_x in range(4):
            index_x = lower_stencil_index[donor, 0] + offset_x
            weight_x = _m4_prime_scalar(relative_position[donor, 0] - index_x)
            for offset_y in range(4):
                index_y = lower_stencil_index[donor, 1] + offset_y
                weight_y = _m4_prime_scalar(relative_position[donor, 1] - index_y)
                for offset_z in range(4):
                    index_z = lower_stencil_index[donor, 2] + offset_z
                    weight_z = _m4_prime_scalar(relative_position[donor, 2] - index_z)
                    weight = weight_x * weight_y * weight_z
                    result[index_x, index_y, index_z, 0] += weight * vortex_strength[donor, 0]
                    result[index_x, index_y, index_z, 1] += weight * vortex_strength[donor, 1]
                    result[index_x, index_y, index_z, 2] += weight * vortex_strength[donor, 2]
    return result


def vortex_strength_from_velocity_trace(
    positions: np.ndarray,
    particle_spacing: float,
    velocity_at: ArrayFunction,
) -> np.ndarray:
    """Integrate ``n x u`` on the six faces of each particle control cell.

    ``velocity_at`` is the hook used by the stable coupler to sample the
    synchronized FVM velocity (and its local reconstruction) at face centres.
    """
    position = _vectors("positions", positions)
    spacing = _positive_finite("particle_spacing", particle_spacing)
    strength = np.zeros_like(position)
    offset = np.zeros(3, dtype=np.float64)
    for axis in range(3):
        offset.fill(0.0)
        offset[axis] = 0.5 * spacing
        upper_velocity = _vectors("velocity_at", velocity_at(position + offset))
        lower_velocity = _vectors("velocity_at", velocity_at(position - offset))
        if len(upper_velocity) != len(position) or len(lower_velocity) != len(position):
            raise ValueError("velocity_at must return one velocity per query point")
        normal = np.zeros(3, dtype=np.float64)
        normal[axis] = 1.0
        strength += spacing**2 * np.cross(normal, upper_velocity - lower_velocity)
    return strength


def inward_cosine_authority(
    positions: np.ndarray,
    transfer_box: np.ndarray | list[float] | tuple[float, ...],
    ramp_width: float,
    vpm_dead_zone: float = 0.0,
) -> np.ndarray:
    """Return the historical FVM authority: one inside, zero at/outside faces.

    The authority is zero through ``vpm_dead_zone`` measured inward from the
    transfer surface, rises with a cosine, and reaches one at ``ramp_width``.
    """
    position = _vectors("positions", positions)
    bounds = _bounds(transfer_box)
    width = _nonnegative_finite("ramp_width", ramp_width)
    dead_zone = _nonnegative_finite("vpm_dead_zone", vpm_dead_zone)
    if width > 0.0 and dead_zone >= width:
        raise ValueError("vpm_dead_zone must be smaller than ramp_width")

    face_distance = np.minimum.reduce(
        [
            position[:, 0] - bounds[0],
            bounds[1] - position[:, 0],
            position[:, 1] - bounds[2],
            bounds[3] - position[:, 1],
            position[:, 2] - bounds[4],
            bounds[5] - position[:, 2],
        ]
    )
    authority = np.zeros(len(position), dtype=np.float64)
    if width == 0.0:
        authority[face_distance > 0.0] = 1.0
        return authority

    authority[face_distance >= width] = 1.0
    ramp = (face_distance > dead_zone) & (face_distance < width)
    if np.any(ramp):
        phase = (face_distance[ramp] - dead_zone) / (width - dead_zone)
        authority[ramp] = 0.5 * (1.0 - np.cos(np.pi * phase))
    return authority


@dataclass(frozen=True)
class VortexInvariants:
    """Integral vortex strength and the first two impulse measures."""

    total_vortex_strength: np.ndarray
    linear_impulse: np.ndarray
    angular_impulse: np.ndarray


def vortex_invariants(
    positions: np.ndarray,
    vortex_strength: np.ndarray,
) -> VortexInvariants:
    """Compute the invariants used to close remeshing and population pruning."""
    position = _vectors("positions", positions)
    strength = _vectors("vortex_strength", vortex_strength)
    if len(position) != len(strength):
        raise ValueError("positions and vortex_strength must have the same length")
    if len(position) == 0:
        zero = np.zeros(3, dtype=np.float64)
        return VortexInvariants(zero.copy(), zero.copy(), zero.copy())
    position_cross_strength = np.cross(position, strength)
    return VortexInvariants(
        total_vortex_strength=strength.sum(axis=0, dtype=np.float64),
        linear_impulse=0.5 * position_cross_strength.sum(axis=0, dtype=np.float64),
        angular_impulse=(1.0 / 3.0)
        * np.cross(position, position_cross_strength).sum(axis=0, dtype=np.float64),
    )


def recover_vortex_invariants(
    positions: np.ndarray,
    vortex_strength: np.ndarray,
    target: VortexInvariants,
    *,
    volumes: np.ndarray,
) -> np.ndarray:
    """Recover total strength and linear impulse with minimum volume weighting."""
    position = _vectors("positions", positions)
    strength = _vectors("vortex_strength", vortex_strength)
    volume = np.asarray(volumes, dtype=np.float64).reshape(-1)
    count = len(position)
    if len(strength) != count or volume.shape != (count,):
        raise ValueError("positions, vortex_strength, and volumes must have one common length")
    if np.any(~np.isfinite(volume)) or np.any(volume <= 0.0):
        raise ValueError("volumes must be finite and positive")

    target_strength = np.asarray(target.total_vortex_strength, dtype=np.float64).reshape(3)
    target_impulse = np.asarray(target.linear_impulse, dtype=np.float64).reshape(3)
    residual_scale = np.linalg.norm(np.concatenate((target_strength, target_impulse)))
    if count == 0:
        if residual_scale > 1.0e-14:
            raise ValueError("cannot recover non-zero invariants without particles")
        return strength
    if count < 2:
        raise ValueError("at least two particles are required for invariant recovery")

    reference = np.average(position, weights=volume, axis=0)
    relative = position - reference
    current_strength = strength.sum(axis=0, dtype=np.float64)
    current_impulse = 0.5 * np.cross(relative, strength).sum(axis=0, dtype=np.float64)
    relative_target_impulse = target_impulse - 0.5 * np.cross(reference, target_strength)
    residual = np.concatenate(
        (target_strength - current_strength, relative_target_impulse - current_impulse)
    )
    if np.linalg.norm(residual) <= 1.0e-14:
        return strength

    matrix = np.zeros((6, 6), dtype=np.float64)
    for column, probe in enumerate(np.eye(6)):
        delta = volume[:, None] * (probe[:3] + 0.5 * np.cross(relative, probe[3:]))
        matrix[:3, column] = delta.sum(axis=0, dtype=np.float64)
        matrix[3:, column] = 0.5 * np.cross(relative, delta).sum(axis=0, dtype=np.float64)
    condition = float(np.linalg.cond(matrix))
    if not np.isfinite(condition) or condition > 1.0e12:
        raise np.linalg.LinAlgError(
            f"invariant recovery matrix is ill-conditioned ({condition:.3e})"
        )
    multiplier = np.linalg.solve(matrix, residual)
    correction = volume[:, None] * (multiplier[:3] + 0.5 * np.cross(relative, multiplier[3:]))
    return strength + correction


@dataclass(frozen=True)
class StableRenewalLattice:
    """Fixed buffered lattice and its time-independent transfer masks."""

    transfer_box: np.ndarray
    renewal_bounds: np.ndarray
    origin: np.ndarray
    shape: tuple[int, int, int]
    positions: np.ndarray
    particle_spacing: float
    buffer_length: float
    lattice_anchor: np.ndarray | None
    mesh_weight: np.ndarray
    fluid_weight: np.ndarray
    solid_interior: np.ndarray
    fvm_authority: np.ndarray

    @property
    def particle_volume(self) -> float:
        return self.particle_spacing**3


def build_stable_renewal_lattice(
    transfer_box: np.ndarray | list[float] | tuple[float, ...],
    particle_spacing: float,
    *,
    buffer_length: float,
    authority_ramp_width: float,
    vpm_dead_zone: float = 0.0,
    lattice_anchor: np.ndarray | None = None,
    mesh_weight_at_node: ArrayFunction | None = None,
    fluid_weight_at_node: ArrayFunction | None = None,
    interior_at_node: ArrayFunction | None = None,
) -> StableRenewalLattice:
    """Build the reusable lattice for a whole-belt stable renewal.

    The physical renewal belt is ``transfer_box + buffer_length``.  The
    returned lattice has another two cells on every side so every donor in the
    closed belt owns its complete 4x4x4 M4' stencil.
    """
    bounds = _bounds(transfer_box)
    spacing = _positive_finite("particle_spacing", particle_spacing)
    buffer = _nonnegative_finite("buffer_length", buffer_length)
    width = _nonnegative_finite("authority_ramp_width", authority_ramp_width)
    dead_zone = _nonnegative_finite("vpm_dead_zone", vpm_dead_zone)
    if width > 0.0 and dead_zone >= width:
        raise ValueError("vpm_dead_zone must be smaller than authority_ramp_width")

    renewal_bounds = bounds.copy()
    renewal_bounds[::2] -= buffer
    renewal_bounds[1::2] += buffer
    lower = renewal_bounds[::2] - M4_PRIME_SUPPORT_CELLS * spacing
    upper = renewal_bounds[1::2] + M4_PRIME_SUPPORT_CELLS * spacing
    anchor_array: np.ndarray | None = None
    if lattice_anchor is not None:
        anchor_array = np.asarray(lattice_anchor, dtype=np.float64).reshape(3)
        if np.any(~np.isfinite(anchor_array)):
            raise ValueError("lattice_anchor must be finite")
        lower = anchor_array + np.floor((lower - anchor_array) / spacing) * spacing
    shape = (
        int(np.ceil((upper[0] - lower[0]) / spacing)) + 1,
        int(np.ceil((upper[1] - lower[1]) / spacing)) + 1,
        int(np.ceil((upper[2] - lower[2]) / spacing)) + 1,
    )
    positions = _regular_grid_positions(lower, spacing, shape)
    count = len(positions)

    mesh_weight = _evaluate_weight("mesh_weight_at_node", mesh_weight_at_node, positions)
    fluid_weight = _evaluate_weight("fluid_weight_at_node", fluid_weight_at_node, positions)
    solid_interior = (
        np.zeros(count, dtype=bool)
        if interior_at_node is None
        else np.asarray(interior_at_node(positions), dtype=bool).reshape(-1)
    )
    if solid_interior.shape != (count,):
        raise ValueError(f"interior_at_node returned {solid_interior.shape}, expected ({count},)")
    authority = inward_cosine_authority(positions, bounds, width, dead_zone) * mesh_weight
    return StableRenewalLattice(
        transfer_box=bounds,
        renewal_bounds=renewal_bounds,
        origin=lower,
        shape=shape,
        positions=positions,
        particle_spacing=spacing,
        buffer_length=buffer,
        lattice_anchor=anchor_array,
        mesh_weight=mesh_weight,
        fluid_weight=fluid_weight,
        solid_interior=solid_interior,
        fvm_authority=authority,
    )


def scatter_m4_prime_to_lattice(
    positions: np.ndarray,
    vortex_strength: np.ndarray,
    lattice: StableRenewalLattice,
) -> np.ndarray:
    """Scatter donors with aligned direct insertion and complete M4' support."""
    position = _vectors("positions", positions)
    strength = _vectors("vortex_strength", vortex_strength)
    if len(position) != len(strength):
        raise ValueError("positions and vortex_strength must have the same length")
    if len(position) == 0:
        return np.zeros((len(lattice.positions), 3), dtype=np.float64)
    if np.any(position < lattice.renewal_bounds[::2]) or np.any(
        position > lattice.renewal_bounds[1::2]
    ):
        raise ValueError("M4' donors must lie inside the physical renewal belt")

    relative = (position - lattice.origin) / lattice.particle_spacing
    nearest = np.rint(relative).astype(np.int64)
    shape_array = np.asarray(lattice.shape, dtype=np.int64)
    aligned = np.max(np.abs(relative - nearest), axis=1) <= _ALIGNMENT_TOLERANCE_CELLS
    aligned &= np.all((nearest >= 0) & (nearest < shape_array), axis=1)
    field = np.zeros((*lattice.shape, 3), dtype=np.float64)
    if np.any(aligned):
        index = nearest[aligned]
        np.add.at(field, (index[:, 0], index[:, 1], index[:, 2]), strength[aligned])

    if np.any(~aligned):
        relative_free = relative[~aligned]
        lower_stencil = np.floor(relative_free).astype(np.int64) - 1
        upper_stencil = lower_stencil + 3
        if np.any(lower_stencil < 0) or np.any(upper_stencil >= shape_array):
            raise RuntimeError("fixed renewal lattice does not contain a complete M4' stencil")
        field += _scatter_unaligned_m4_prime(
            relative_free,
            lower_stencil,
            strength[~aligned],
            shape_array,
        )
    return field.reshape(-1, 3)


def gaussian_represented_vortex_strength(
    lattice_vortex_strength: np.ndarray,
    shape: tuple[int, int, int],
    particle_spacing: float,
    *,
    core_radius: float,
) -> np.ndarray:
    """Return the vortex strength represented by Gaussian lattice particles."""
    spacing = _positive_finite("particle_spacing", particle_spacing)
    radius = _positive_finite("core_radius", core_radius)
    strength = _vectors("lattice_vortex_strength", lattice_vortex_strength)
    expected = int(np.prod(shape))
    if len(strength) != expected:
        raise ValueError(f"lattice_vortex_strength has {len(strength)} rows, expected {expected}")
    standard_deviation_cells = radius / (np.sqrt(2.0) * spacing)
    grid = strength.reshape(*shape, 3)
    return np.stack(
        [
            gaussian_filter(
                grid[..., component],
                standard_deviation_cells,
                mode="constant",
                truncate=5.0,
            )
            for component in range(3)
        ],
        axis=-1,
    ).reshape(-1, 3)


@dataclass(frozen=True)
class RepresentedStateBlend:
    """Result of blending in represented-vorticity space."""

    vortex_strength: np.ndarray
    physical_target: np.ndarray
    represented_vortex_strength: np.ndarray | None
    residual_before_correction: float
    residual_after_correction: float | None
    maximum_amplification: float


def blend_represented_state(
    vpm_vortex_strength: np.ndarray,
    fvm_target_vortex_strength: np.ndarray,
    fvm_authority: np.ndarray,
    shape: tuple[int, int, int],
    particle_spacing: float,
    *,
    core_radius: float,
    amplification_cap: float = DEFAULT_AMPLIFICATION_CAP,
    output_weight: np.ndarray | None = None,
    compute_final_representation: bool = True,
) -> RepresentedStateBlend:
    """Blend the Gaussian represented VPM state and apply one local correction."""
    cap = float(amplification_cap)
    if not np.isfinite(cap) or cap < 1.0:
        raise ValueError("amplification_cap must be finite and at least one")
    vpm_strength = _vectors("vpm_vortex_strength", vpm_vortex_strength)
    fvm_strength = _vectors("fvm_target_vortex_strength", fvm_target_vortex_strength)
    authority = np.asarray(fvm_authority, dtype=np.float64).reshape(-1)
    if vpm_strength.shape != fvm_strength.shape or authority.shape != (len(vpm_strength),):
        raise ValueError("blend inputs must share one lattice shape")
    weight: np.ndarray | None = None
    if output_weight is not None:
        weight = np.asarray(output_weight, dtype=np.float64).reshape(-1)
        if weight.shape != (len(vpm_strength),):
            raise ValueError("output_weight must share the transfer lattice shape")

    represented_vpm = gaussian_represented_vortex_strength(
        vpm_strength,
        shape,
        particle_spacing,
        core_radius=core_radius,
    )
    physical_target = represented_vpm + authority[:, None] * (fvm_strength - represented_vpm)

    blended_strength = vpm_strength + authority[:, None] * (fvm_strength - vpm_strength)
    represented_blend = gaussian_represented_vortex_strength(
        blended_strength,
        shape,
        particle_spacing,
        core_radius=core_radius,
    )
    residual = physical_target - represented_blend
    denominator = float(np.linalg.norm(physical_target)) + 1.0e-30
    residual_before = float(np.linalg.norm(residual)) / denominator

    correction_gain = min(cap - 1.0, 1.0)
    corrected_strength = blended_strength + correction_gain * residual
    if weight is not None:
        corrected_strength = corrected_strength * weight[:, None]
    target_maximum = float(np.linalg.norm(physical_target, axis=1).max(initial=0.0)) + 1.0e-30
    maximum_amplification = (
        float(np.linalg.norm(corrected_strength, axis=1).max(initial=0.0)) / target_maximum
    )

    represented_corrected: np.ndarray | None = None
    residual_after: float | None = None
    if compute_final_representation:
        represented_corrected = gaussian_represented_vortex_strength(
            corrected_strength,
            shape,
            particle_spacing,
            core_radius=core_radius,
        )
        residual_after = (
            float(np.linalg.norm(physical_target - represented_corrected)) / denominator
        )
    return RepresentedStateBlend(
        vortex_strength=corrected_strength,
        physical_target=physical_target,
        represented_vortex_strength=represented_corrected,
        residual_before_correction=residual_before,
        residual_after_correction=residual_after,
        maximum_amplification=maximum_amplification,
    )


def soft_prune_vortex_strength(
    vortex_strength: np.ndarray,
    threshold: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the historical continuous non-negative-garrote shrinkage."""
    strength = _vectors("vortex_strength", vortex_strength)
    threshold_array = np.broadcast_to(np.asarray(threshold, dtype=np.float64), (len(strength),))
    if np.any(~np.isfinite(threshold_array)) or np.any(threshold_array < 0.0):
        raise ValueError("prune threshold must be finite and non-negative")
    if not np.any(threshold_array > 0.0):
        return strength.copy(), np.zeros_like(strength)
    magnitude = np.linalg.norm(strength, axis=1)
    scale = np.zeros_like(magnitude)
    active = magnitude > threshold_array
    scale[active] = 1.0 - (threshold_array[active] / magnitude[active]) ** 2
    shrunk = strength * scale[:, None]
    return shrunk, strength - shrunk


def redistribute_pruned_vortex_strength_locally(
    removed_vortex_strength: np.ndarray,
    retained_vortex_strength: np.ndarray,
    shape: tuple[int, int, int],
) -> np.ndarray:
    """Move removed strength to surviving face neighbours without a periodic seam."""
    removed = _vectors("removed_vortex_strength", removed_vortex_strength).reshape(*shape, 3)
    retained = _vectors("retained_vortex_strength", retained_vortex_strength)
    if len(retained) != int(np.prod(shape)):
        raise ValueError("retained_vortex_strength does not match shape")
    if not np.any(removed):
        return retained.copy()

    output = retained.reshape(*shape, 3).copy()
    alive = np.linalg.norm(output, axis=-1) > 0.0
    shifts = ((0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1))
    neighbours = [np.roll(alive, -step, axis=axis) for axis, step in shifts]
    for index, (axis, step) in enumerate(shifts):
        boundary_slice: list[slice | int] = [slice(None)] * 3
        boundary_slice[axis] = -1 if step == 1 else 0
        neighbours[index][tuple(boundary_slice)] = False

    count = np.sum(neighbours, axis=0).astype(np.float64)
    donatable = count > 0.0
    share = np.zeros_like(removed)
    np.divide(removed, count[..., None], out=share, where=donatable[..., None])
    for neighbour, (axis, step) in zip(neighbours, shifts, strict=True):
        contribution = np.where(neighbour[..., None], share, 0.0)
        output += np.roll(contribution, step, axis=axis)
    return output.reshape(-1, 3)


@dataclass(frozen=True)
class StableRenewalResult:
    """Complete particle state produced by one stable whole-belt renewal."""

    position: np.ndarray
    vortex_strength: np.ndarray
    particle_volume: np.ndarray
    core_radius: np.ndarray
    renewed_input_count: int = 0
    renewed_output_count: int = 0
    preserved_outer_count: int = 0
    coalesced_outer_count: int = 0
    coalesced_outer_input_indices: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    excluded_solid_count: int = 0
    pruned_node_count: int = 0
    pruned_vortex_strength_l1: float = 0.0
    pruned_vortex_strength_fraction: float = 0.0
    population_pruned_count: int = 0
    population_pruned_vortex_strength_fraction: float = 0.0
    population_pruned_velocity_bound: float = 0.0
    transfer_cfl: float = 0.0
    conservation_raw_mismatch: dict[str, float] = field(default_factory=dict)
    conservation_applied_correction: dict[str, float] = field(default_factory=dict)
    conservation_residual: dict[str, float] = field(default_factory=dict)
    conservation_applied_particle_strength_fraction: float = 0.0
    conservation_target_invariants: VortexInvariants | None = None
    conservation_raw_invariants: VortexInvariants | None = None
    conservation_reference_strength_l1: float = 0.0
    population_conservation_raw_mismatch: dict[str, float] = field(default_factory=dict)
    population_conservation_applied_correction: dict[str, float] = field(default_factory=dict)
    population_conservation_residual: dict[str, float] = field(default_factory=dict)
    population_conservation_applied_particle_strength_fraction: float = 0.0
    representation_residual_before_prune: float | None = None
    representation_residual_after_prune: float | None = None
    maximum_transfer_amplification: float = 0.0
    excluded_input_vortex_strength_l1: float = 0.0
    excluded_remesh_vortex_strength_l1: float = 0.0
    excluded_target_vortex_strength_l1: float = 0.0

    @property
    def particle_count(self) -> int:
        return len(self.position)


def renew_stable_overlap(
    positions: np.ndarray,
    vortex_strength: np.ndarray,
    lattice: StableRenewalLattice,
    *,
    fvm_vortex_strength_at_node: ArrayFunction,
    particle_fluid_weight: ArrayFunction | None = None,
    particle_in_solid: ArrayFunction | None = None,
    prune_threshold: float = 0.0,
    core_radius_ratio: float = CORE_RADIUS_RATIO,
    amplification_cap: float = DEFAULT_AMPLIFICATION_CAP,
    boundary_prune_multiplier: float = 1.0,
    maximum_particle_count: int | None = None,
    freestream_speed: float = 0.0,
    time_step_size: float = 0.0,
    compute_diagnostics: bool = True,
) -> StableRenewalResult:
    """Renew the whole buffered belt and preserve the outer Lagrangian wake.

    This is the side-effect-free numerical core of the long-running method.
    The caller supplies a synchronized FVM velocity-trace target through
    ``fvm_vortex_strength_at_node`` and may then atomically replace the VPM
    particle arrays with the returned state.
    """
    position = _vectors("positions", positions)
    strength = _vectors("vortex_strength", vortex_strength)
    if len(position) != len(strength):
        raise ValueError("positions and vortex_strength must have the same length")
    spacing = lattice.particle_spacing
    ratio = _positive_finite("core_radius_ratio", core_radius_ratio)
    threshold = _nonnegative_finite("prune_threshold", prune_threshold)
    boundary_multiplier = _positive_finite("boundary_prune_multiplier", boundary_prune_multiplier)
    if maximum_particle_count is not None and maximum_particle_count < 2:
        raise ValueError("maximum_particle_count must be at least two")

    count = len(position)
    input_fluid = _evaluate_weight("particle_fluid_weight", particle_fluid_weight, position)
    excluded_input_l1 = float(np.linalg.norm(strength * (1.0 - input_fluid)[:, None], axis=1).sum())
    tapered_strength = strength * input_fluid[:, None]
    deep_solid = (
        np.zeros(count, dtype=bool)
        if particle_in_solid is None
        else np.asarray(particle_in_solid(position), dtype=bool).reshape(-1)
    )
    if deep_solid.shape != (count,):
        raise ValueError(f"particle_in_solid returned {deep_solid.shape}, expected ({count},)")

    valid = ~deep_solid
    in_renewal_belt = valid & np.all(
        (position >= lattice.renewal_bounds[::2]) & (position <= lattice.renewal_bounds[1::2]),
        axis=1,
    )
    preserved_outer = valid & ~in_renewal_belt

    vpm_lattice_strength = scatter_m4_prime_to_lattice(
        position[in_renewal_belt],
        tapered_strength[in_renewal_belt],
        lattice,
    )
    excluded_remesh_l1 = float(
        np.linalg.norm(vpm_lattice_strength * (1.0 - lattice.fluid_weight)[:, None], axis=1).sum()
    )
    vpm_lattice_strength *= lattice.fluid_weight[:, None]

    raw_target = _vectors(
        "fvm_vortex_strength_at_node",
        fvm_vortex_strength_at_node(lattice.positions),
    )
    if len(raw_target) != len(lattice.positions):
        raise ValueError("fvm_vortex_strength_at_node must return one vector per lattice node")
    excluded_target_l1 = float(
        np.linalg.norm(raw_target * (1.0 - lattice.fluid_weight)[:, None], axis=1).sum()
    )
    fvm_target = raw_target * (lattice.fluid_weight * lattice.mesh_weight)[:, None]

    base_core_radius = ratio * spacing
    blend = blend_represented_state(
        vpm_lattice_strength,
        fvm_target,
        lattice.fvm_authority,
        lattice.shape,
        spacing,
        core_radius=base_core_radius,
        amplification_cap=amplification_cap,
        output_weight=lattice.fluid_weight,
        compute_final_representation=compute_diagnostics,
    )
    comparison_weight = lattice.fluid_weight * lattice.mesh_weight
    residual_before_prune: float | None = None
    if compute_diagnostics:
        if blend.represented_vortex_strength is None:
            raise RuntimeError("represented-state diagnostic was not evaluated")
        denominator = (
            float(np.linalg.norm(blend.physical_target * comparison_weight[:, None])) + 1.0e-30
        )
        residual_before_prune = (
            float(
                np.linalg.norm(
                    (blend.represented_vortex_strength - blend.physical_target)
                    * comparison_weight[:, None]
                )
            )
            / denominator
        )

    pre_prune_invariants = vortex_invariants(lattice.positions, blend.vortex_strength)
    magnitude_before = np.linalg.norm(blend.vortex_strength, axis=1)
    local_threshold = threshold * (
        1.0 + (boundary_multiplier - 1.0) * (1.0 - lattice.fvm_authority)
    )
    shrunk, removed = soft_prune_vortex_strength(blend.vortex_strength, local_threshold)
    redistributed = redistribute_pruned_vortex_strength_locally(
        removed,
        shrunk,
        lattice.shape,
    )
    redistributed[lattice.solid_interior] = 0.0
    keep = np.linalg.norm(redistributed, axis=1) > 0.0
    pruned = (magnitude_before > 0.0) & ~keep
    active_l1 = float(magnitude_before[magnitude_before > 0.0].sum())
    pruned_l1 = float(magnitude_before[pruned].sum())

    renewed_position = lattice.positions[keep]
    raw_renewed_strength = redistributed[keep]
    renewed_strength = raw_renewed_strength
    renewed_volume = np.full(len(renewed_position), spacing**3, dtype=np.float64)
    raw_invariants = vortex_invariants(renewed_position, raw_renewed_strength)
    conservation_raw_mismatch = _invariant_residual(pre_prune_invariants, raw_invariants)
    conservation_target_invariants = pre_prune_invariants
    conservation_raw_invariants = raw_invariants
    conservation_reference_strength_l1 = active_l1
    if len(renewed_position) > 1:
        renewed_strength = recover_vortex_invariants(
            renewed_position,
            raw_renewed_strength,
            pre_prune_invariants,
            volumes=renewed_volume,
        )
    corrected_invariants = vortex_invariants(renewed_position, renewed_strength)
    conservation_applied_correction = _invariant_residual(
        raw_invariants,
        corrected_invariants,
    )
    conservation_residual = _invariant_residual(pre_prune_invariants, corrected_invariants)
    applied_particle_strength_fraction = float(
        np.linalg.norm(renewed_strength - raw_renewed_strength, axis=1).sum(dtype=np.float64)
        / (active_l1 + 1.0e-30)
    )

    residual_after_prune: float | None = None
    if compute_diagnostics:
        final_lattice_strength = np.zeros_like(blend.vortex_strength)
        final_lattice_strength[keep] = renewed_strength
        represented_final = gaussian_represented_vortex_strength(
            final_lattice_strength,
            lattice.shape,
            spacing,
            core_radius=base_core_radius,
        )
        denominator = (
            float(np.linalg.norm(blend.physical_target * comparison_weight[:, None])) + 1.0e-30
        )
        residual_after_prune = (
            float(
                np.linalg.norm(
                    (represented_final - blend.physical_target) * comparison_weight[:, None]
                )
            )
            / denominator
        )

    renewed_radius = np.full(len(renewed_position), base_core_radius, dtype=np.float64)

    # M4' support extends beyond the physical renewal belt.  A support value
    # can therefore land on the same regular node as an existing persistent
    # particle.  Merge only those exact lattice collisions after renewal
    # closure; this keeps the configured persistence boundary unchanged and
    # prevents duplicate co-located particles at the support seam.
    preserved_for_append = preserved_outer.copy()
    coalesced_outer_count = 0
    coalesced_outer_input_indices = np.empty(0, dtype=np.int64)
    if len(renewed_position) and np.any(preserved_outer):
        outer_index = np.flatnonzero(preserved_outer)
        lattice_maximum = lattice.origin + spacing * (np.asarray(lattice.shape) - 1)
        position_tolerance = _ALIGNMENT_TOLERANCE_CELLS * spacing
        inside_support = np.ones(len(outer_index), dtype=bool)
        for axis in range(3):
            coordinate = position[outer_index, axis]
            inside_support &= coordinate >= lattice.origin[axis] - position_tolerance
            inside_support &= coordinate <= lattice_maximum[axis] + position_tolerance
        candidate_outer_index = outer_index[inside_support]
        relative_outer = (position[candidate_outer_index] - lattice.origin) / spacing
        nearest_outer = np.rint(relative_outer).astype(np.int64)
        shape_array = np.asarray(lattice.shape, dtype=np.int64)
        aligned_outer = (
            np.max(np.abs(relative_outer - nearest_outer), axis=1) <= _ALIGNMENT_TOLERANCE_CELLS
        )
        aligned_outer &= np.all(
            (nearest_outer >= 0) & (nearest_outer < shape_array),
            axis=1,
        )
        aligned_candidate = np.flatnonzero(aligned_outer)
        if len(aligned_candidate):
            candidate_lattice_index = nearest_outer[aligned_candidate]
            candidate_flat_index = np.ravel_multi_index(
                candidate_lattice_index.T,
                lattice.shape,
            )
            renewed_flat_index = np.flatnonzero(keep)
            renewed_row = np.searchsorted(renewed_flat_index, candidate_flat_index)
            valid_row = renewed_row < len(renewed_flat_index)
            collision = np.zeros(len(renewed_row), dtype=bool)
            collision[valid_row] = (
                renewed_flat_index[renewed_row[valid_row]] == candidate_flat_index[valid_row]
            )
            if np.any(collision):
                collided_outer_index = candidate_outer_index[aligned_candidate[collision]]
                coalesced_invariants = vortex_invariants(
                    renewed_position[renewed_row[collision]],
                    tapered_strength[collided_outer_index],
                )

                def with_coalesced(base: VortexInvariants) -> VortexInvariants:
                    return VortexInvariants(
                        total_vortex_strength=(
                            base.total_vortex_strength + coalesced_invariants.total_vortex_strength
                        ),
                        linear_impulse=base.linear_impulse + coalesced_invariants.linear_impulse,
                        angular_impulse=(
                            base.angular_impulse + coalesced_invariants.angular_impulse
                        ),
                    )

                conservation_target_invariants = with_coalesced(conservation_target_invariants)
                conservation_raw_invariants = with_coalesced(conservation_raw_invariants)
                conservation_reference_strength_l1 += float(
                    np.linalg.norm(
                        tapered_strength[collided_outer_index],
                        axis=1,
                    ).sum(dtype=np.float64)
                )
                renewed_strength = renewed_strength.copy()
                np.add.at(
                    renewed_strength,
                    renewed_row[collision],
                    tapered_strength[collided_outer_index],
                )
                preserved_for_append[collided_outer_index] = False
                coalesced_outer_count = int(len(collided_outer_index))
                coalesced_outer_input_indices = collided_outer_index

    if np.any(preserved_for_append):
        output_position = np.vstack((renewed_position, position[preserved_for_append]))
        output_strength = np.vstack((renewed_strength, tapered_strength[preserved_for_append]))
        output_volume = np.concatenate(
            (
                renewed_volume,
                np.full(np.count_nonzero(preserved_for_append), spacing**3, dtype=np.float64),
            )
        )
        output_radius = np.concatenate(
            (
                renewed_radius,
                np.full(
                    np.count_nonzero(preserved_for_append),
                    base_core_radius,
                    dtype=np.float64,
                ),
            )
        )
    else:
        output_position = renewed_position
        output_strength = renewed_strength
        output_volume = renewed_volume
        output_radius = renewed_radius

    population_pruned_count = 0
    population_pruned_fraction = 0.0
    population_velocity_bound = 0.0
    population_conservation_raw_mismatch: dict[str, float] = {}
    population_conservation_applied_correction: dict[str, float] = {}
    population_conservation_residual: dict[str, float] = {}
    population_applied_particle_strength_fraction = 0.0
    final_renewed_count = len(renewed_position)
    final_outer_count = int(np.count_nonzero(preserved_for_append))
    if maximum_particle_count is not None and len(output_position) > maximum_particle_count:
        target_count = int(maximum_particle_count)
        combined_invariants = vortex_invariants(output_position, output_strength)
        combined_magnitude = np.linalg.norm(output_strength, axis=1)
        renewed_count = len(renewed_position)
        outer_count = len(output_position) - renewed_count
        if outer_count < target_count:
            renewed_budget = target_count - outer_count
            renewed_keep = np.argpartition(combined_magnitude[:renewed_count], -renewed_budget)[
                -renewed_budget:
            ]
            keep_indices = np.concatenate(
                (
                    renewed_keep,
                    np.arange(renewed_count, len(output_position), dtype=np.int64),
                )
            )
        elif outer_count == target_count:
            keep_indices = np.arange(renewed_count, len(output_position), dtype=np.int64)
        else:
            outer_keep = np.argpartition(combined_magnitude[renewed_count:], -target_count)[
                -target_count:
            ]
            keep_indices = outer_keep + renewed_count
        keep_indices = np.sort(keep_indices)
        population_keep = np.zeros(len(output_position), dtype=bool)
        population_keep[keep_indices] = True
        discarded_l1 = float(combined_magnitude[~population_keep].sum())
        delta = np.maximum(
            np.maximum(
                lattice.transfer_box[::2] - output_position,
                output_position - lattice.transfer_box[1::2],
            ),
            0.0,
        )
        distance_squared = np.einsum("ij,ij->i", delta, delta)
        population_velocity_bound = float(
            np.sum(
                combined_magnitude[~population_keep]
                / (
                    4.0
                    * np.pi
                    * np.maximum(
                        distance_squared[~population_keep] + output_radius[~population_keep] ** 2,
                        1.0e-30,
                    )
                )
            )
        )
        population_pruned_fraction = discarded_l1 / (float(combined_magnitude.sum()) + 1.0e-30)
        population_pruned_count = int(np.count_nonzero(~population_keep))
        final_renewed_count = int(np.count_nonzero(keep_indices < renewed_count))
        final_outer_count = int(len(keep_indices) - final_renewed_count)
        output_position = output_position[keep_indices]
        raw_population_strength = output_strength[keep_indices]
        output_volume = output_volume[keep_indices]
        output_radius = output_radius[keep_indices]
        raw_population_invariants = vortex_invariants(output_position, raw_population_strength)
        population_conservation_raw_mismatch = _invariant_residual(
            combined_invariants,
            raw_population_invariants,
        )
        output_strength = recover_vortex_invariants(
            output_position,
            raw_population_strength,
            combined_invariants,
            volumes=output_volume,
        )
        corrected_population_invariants = vortex_invariants(output_position, output_strength)
        population_conservation_applied_correction = _invariant_residual(
            raw_population_invariants,
            corrected_population_invariants,
        )
        population_conservation_residual = _invariant_residual(
            combined_invariants,
            corrected_population_invariants,
        )
        population_applied_particle_strength_fraction = float(
            np.linalg.norm(output_strength - raw_population_strength, axis=1).sum(dtype=np.float64)
            / (float(combined_magnitude.sum(dtype=np.float64)) + 1.0e-30)
        )

    speed = _nonnegative_finite("freestream_speed", abs(float(freestream_speed)))
    time_step = _nonnegative_finite("time_step_size", abs(float(time_step_size)))
    transfer_cfl = speed * time_step / (lattice.buffer_length + 1.0e-30)
    return StableRenewalResult(
        position=output_position,
        vortex_strength=output_strength,
        particle_volume=output_volume,
        core_radius=output_radius,
        renewed_input_count=int(np.count_nonzero(in_renewal_belt)),
        renewed_output_count=final_renewed_count,
        preserved_outer_count=final_outer_count,
        coalesced_outer_count=coalesced_outer_count,
        coalesced_outer_input_indices=coalesced_outer_input_indices,
        excluded_solid_count=int(np.count_nonzero(deep_solid)),
        pruned_node_count=int(np.count_nonzero(pruned)),
        pruned_vortex_strength_l1=pruned_l1,
        pruned_vortex_strength_fraction=pruned_l1 / (active_l1 + 1.0e-30),
        population_pruned_count=population_pruned_count,
        population_pruned_vortex_strength_fraction=population_pruned_fraction,
        population_pruned_velocity_bound=population_velocity_bound,
        transfer_cfl=float(transfer_cfl),
        conservation_raw_mismatch=conservation_raw_mismatch,
        conservation_applied_correction=conservation_applied_correction,
        conservation_residual=conservation_residual,
        conservation_applied_particle_strength_fraction=applied_particle_strength_fraction,
        conservation_target_invariants=conservation_target_invariants,
        conservation_raw_invariants=conservation_raw_invariants,
        conservation_reference_strength_l1=conservation_reference_strength_l1,
        population_conservation_raw_mismatch=population_conservation_raw_mismatch,
        population_conservation_applied_correction=population_conservation_applied_correction,
        population_conservation_residual=population_conservation_residual,
        population_conservation_applied_particle_strength_fraction=(
            population_applied_particle_strength_fraction
        ),
        representation_residual_before_prune=residual_before_prune,
        representation_residual_after_prune=residual_after_prune,
        maximum_transfer_amplification=blend.maximum_amplification,
        excluded_input_vortex_strength_l1=excluded_input_l1,
        excluded_remesh_vortex_strength_l1=excluded_remesh_l1,
        excluded_target_vortex_strength_l1=excluded_target_l1,
    )


def _regular_grid_positions(
    origin: np.ndarray,
    spacing: float,
    shape: tuple[int, int, int],
) -> np.ndarray:
    axes = [origin[axis] + spacing * np.arange(shape[axis]) for axis in range(3)]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([component.ravel() for component in mesh])


def _evaluate_weight(
    name: str,
    function: ArrayFunction | None,
    positions: np.ndarray,
) -> np.ndarray:
    if function is None or len(positions) == 0:
        return np.ones(len(positions), dtype=np.float64)
    weight = np.asarray(function(positions), dtype=np.float64).reshape(-1)
    if weight.shape != (len(positions),):
        raise ValueError(f"{name} returned {weight.shape}, expected ({len(positions)},)")
    if np.any(~np.isfinite(weight)):
        raise ValueError(f"{name} returned non-finite weights")
    return np.clip(weight, 0.0, 1.0)


def _invariant_residual(
    target: VortexInvariants,
    actual: VortexInvariants,
) -> dict[str, float]:
    return {
        "total_vortex_strength": float(
            np.linalg.norm(target.total_vortex_strength - actual.total_vortex_strength)
        ),
        "linear_impulse": float(np.linalg.norm(target.linear_impulse - actual.linear_impulse)),
        "angular_impulse": float(np.linalg.norm(target.angular_impulse - actual.angular_impulse)),
    }


def _vectors(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1, 3)
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _bounds(values: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    bounds = np.asarray(values, dtype=np.float64).reshape(6)
    if np.any(~np.isfinite(bounds)) or np.any(bounds[1::2] <= bounds[::2]):
        raise ValueError("transfer_box must contain six finite increasing bounds")
    return bounds


def _positive_finite(name: str, value: float) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _nonnegative_finite(name: str, value: float) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


__all__ = [
    "DEFAULT_AMPLIFICATION_CAP",
    "M4_PRIME_SUPPORT_CELLS",
    "RepresentedStateBlend",
    "StableRenewalLattice",
    "StableRenewalResult",
    "VortexInvariants",
    "blend_represented_state",
    "build_stable_renewal_lattice",
    "gaussian_represented_vortex_strength",
    "inward_cosine_authority",
    "m4_prime",
    "maximum_stable_time_step",
    "recover_vortex_invariants",
    "redistribute_pruned_vortex_strength_locally",
    "renew_stable_overlap",
    "required_buffer_length",
    "scatter_m4_prime_to_lattice",
    "soft_prune_vortex_strength",
    "vortex_invariants",
    "vortex_strength_from_velocity_trace",
]
