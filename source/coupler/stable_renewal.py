"""Buffered fixed-lattice FVM-to-VPM renewal compatible with GBD diffusion.

Operators return renewed particle arrays without mutating a solver. Interior
particles receive FVM vorticity while the release buffer preserves the VPM
wake. Renewed particles use the fixed lattice and base core radius.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import convolve1d

from source._numba import cacheable_njit as njit
from source.grid_connectivity import connected_grid_components

from .lattice_transfer import _m4_prime_scalar, m4_prime

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
    """Return the advection buffer plus complete M4' support, in metres."""
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
    """Integrate the velocity circulation over the six faces of a cubic control cell.

    Parameters
    ----------
    positions : array_like, shape (N, 3)
        World-frame control-cell centres in m.
    particle_spacing : float
        Positive cell width h in m.
    velocity_at : callable
        Maps query points (N,3) in m to synchronized fluid velocities (N,3)
        in m/s. The coupler supplies the FVM reconstruction.

    Returns
    -------
    numpy.ndarray, shape (N, 3)
        Independent float64 strengths Gamma = omega*h³ in m³/s.

    Raises
    ------
    ValueError
        If positions/velocities are invalid, lengths differ, or spacing
        is nonpositive or nonfinite. Inputs are not mutated.

    Notes
    -----
    Face midpoint quadrature uses outward normals and n cross u.
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


def inward_cosine_blend_weight(
    positions: np.ndarray,
    transfer_box: np.ndarray | list[float] | tuple[float, ...],
    ramp_width: float,
    vpm_dead_zone: float = 0.0,
    *,
    slip_slab: bool = False,
) -> np.ndarray:
    """Return the dimensionless FVM blending weight at each world-frame point.

    Positions have shape (N,3), and transfer_box contains
    (xmin,xmax,ymin,ymax,zmin,zmax), all in m. The blending weight is zero through
    vpm_dead_zone measured inward from each face, rises with a cosine and
    reaches one at ramp_width (both nonnegative, in m). A zero ramp gives a
    sharp interior mask.
    With slip_slab=True, z faces retain full blending weight through the closed
    physical span but points outside it have zero blending weight.
    The returned independent float64 array has shape (N,) and values in [0,1].
    Invalid bounds, vectors or widths raise ValueError; inputs are unchanged.
    """
    position = _vectors("positions", positions)
    bounds = _bounds(transfer_box)
    width = _nonnegative_finite("ramp_width", ramp_width)
    dead_zone = _nonnegative_finite("vpm_dead_zone", vpm_dead_zone)
    if width > 0.0 and dead_zone >= width:
        raise ValueError("vpm_dead_zone must be smaller than ramp_width")

    skip_z_ramp = slip_slab
    face_distance = np.minimum.reduce(
        [
            position[:, 0] - bounds[0],
            bounds[1] - position[:, 0],
            position[:, 1] - bounds[2],
            bounds[3] - position[:, 1],
            np.full(len(position), np.inf) if skip_z_ramp else position[:, 2] - bounds[4],
            np.full(len(position), np.inf) if skip_z_ramp else bounds[5] - position[:, 2],
        ]
    )
    blend_weight = np.zeros(len(position), dtype=np.float64)
    if width == 0.0:
        blend_weight[face_distance > 0.0] = 1.0
        if slip_slab:
            blend_weight[(position[:, 2] < bounds[4]) | (position[:, 2] > bounds[5])] = 0.0
        return blend_weight

    blend_weight[face_distance >= width] = 1.0
    ramp = (face_distance > dead_zone) & (face_distance < width)
    if np.any(ramp):
        phase = (face_distance[ramp] - dead_zone) / (width - dead_zone)
        blend_weight[ramp] = 0.5 * (1.0 - np.cos(np.pi * phase))
    if slip_slab:
        blend_weight[(position[:, 2] < bounds[4]) | (position[:, 2] > bounds[5])] = 0.0
    return blend_weight


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
    if count == 0:
        if np.any(target_strength != 0.0) or np.any(target_impulse != 0.0):
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
    strength_scale = max(
        float(np.linalg.norm(strength, axis=1).sum()), float(np.linalg.norm(target_strength))
    )
    length_scale = max(float(np.linalg.norm(relative, axis=1).max()), np.finfo(float).tiny)
    roundoff = 4.0 * np.finfo(np.float64).eps * strength_scale
    # Circulation and impulse have different dimensions. An absolute 1e-14
    # skip threshold loses small-but-resolved budgets and can exceed the
    # production wrapper's storage-precision conservation tolerance.
    if (
        np.linalg.norm(residual[:3]) <= roundoff
        and np.linalg.norm(residual[3:]) <= roundoff * length_scale
    ):
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
    corrected = strength.copy()
    # Refinement removes rounding from distributing a correction over many
    # particles; reuse the same six-by-six constraint matrix.
    for _ in range(3):
        multiplier = np.linalg.solve(matrix, residual)
        corrected += volume[:, None] * (multiplier[:3] + 0.5 * np.cross(relative, multiplier[3:]))
        residual = np.concatenate(
            (
                target_strength - corrected.sum(axis=0, dtype=np.float64),
                relative_target_impulse
                - 0.5 * np.cross(relative, corrected).sum(axis=0, dtype=np.float64),
            )
        )
        if (
            np.linalg.norm(residual[:3]) <= roundoff
            and np.linalg.norm(residual[3:]) <= roundoff * length_scale
        ):
            break
    return corrected


@dataclass(frozen=True)
class StableRenewalLattice:
    """Fixed Cartesian renewal geometry and dimensionless transfer weights.

    Lengths, bounds and positions are in m. ``shape`` is (nx,ny,nz), with
    positions flattened in C order. mesh_weight, fluid_weight, fvm_blend_weight
    and solid_interior each have one entry per node. Each particle has volume h³.
    The frozen dataclass prevents attribute reassignment but does not make its
    NumPy arrays immutable; callers must treat these shared arrays as read only.
    """

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
    fvm_blend_weight: np.ndarray
    slip_slab: bool = False
    wall_links: np.ndarray | None = None
    wall_scatter_correction: Callable | None = None

    @property
    def particle_volume(self) -> float:
        """Return the represented volume of each renewal particle in m³.

        The uniform volume is h³, where h is particle_spacing in m.
        """
        return self.particle_spacing**3


def build_stable_renewal_lattice(
    transfer_box: np.ndarray | list[float] | tuple[float, ...],
    particle_spacing: float,
    *,
    buffer_length: float,
    blend_ramp_width: float,
    vpm_dead_zone: float = 0.0,
    lattice_anchor: np.ndarray | None = None,
    mesh_weight_at_node: ArrayFunction | None = None,
    fluid_weight_at_node: ArrayFunction | None = None,
    interior_at_node: ArrayFunction | None = None,
    slip_slab: bool = False,
    solid_boundary=None,
    wall_scatter_correction: Callable | None = None,
) -> StableRenewalLattice:
    """Build fixed renewal nodes and transfer weights around the FVM region.

    Parameters
    ----------
    transfer_box : array_like, shape (6,)
        (xmin,xmax,ymin,ymax,zmin,zmax) in m, with increasing bounds.
    particle_spacing : float
        Positive uniform node spacing h in m.
    buffer_length, blend_ramp_width, vpm_dead_zone : float
        Nonnegative lengths in m. The dead zone must be narrower than a
        nonzero blending ramp.
    lattice_anchor : array_like, shape (3,), optional
        Fixed lattice phase in m. Nodes align to this phase when provided.
    mesh_weight_at_node, fluid_weight_at_node : callable, optional
        Maps world points (N,3) in m to dimensionless scalar weights (N,).
    interior_at_node : callable, optional
        Maps world points (N,3) in m to the solid-interior mask (N,).
    slip_slab : bool, default=False
        Use physical z faces without a spanwise blending ramp; ghost support nodes
        remain on the lattice with zero fluid weight.
    solid_boundary : optional
        Static solid-query interface. Its segment queries cache blocked lattice
        links for component-local pruning and invariant recovery.
    wall_scatter_correction : callable, optional
        Existing GBD visible-support M4 correction for off-lattice 3-D donors.
        Called with positions, strengths, origin, spacing and shape; returns
        sparse node indices, additive strength corrections and diagnostics.

    Returns
    -------
    StableRenewalLattice
        Geometry and masks for repeated renewal. Newly allocated arrays
        remain mutable despite the frozen container and must be treated as read only.

    Raises
    ------
    ValueError
        If lengths, bounds, lattice phase or mask shapes are invalid.

    Notes
    -----
    The physical belt expands transfer_box by buffer_length. Two additional
    nodes on every active axis retain each donor's complete M4-prime support:
    4-by-4-by-4 nodes.
    """
    bounds = _bounds(transfer_box)
    spacing = _positive_finite("particle_spacing", particle_spacing)
    buffer = _nonnegative_finite("buffer_length", buffer_length)
    width = _nonnegative_finite("blend_ramp_width", blend_ramp_width)
    dead_zone = _nonnegative_finite("vpm_dead_zone", vpm_dead_zone)
    if width > 0.0 and dead_zone >= width:
        raise ValueError("vpm_dead_zone must be smaller than blend_ramp_width")

    renewal_bounds = bounds.copy()
    renewal_bounds[::2] -= buffer
    renewal_bounds[1::2] += buffer
    if slip_slab:
        renewal_bounds[4:6] = bounds[4:6]
    lower = renewal_bounds[::2] - M4_PRIME_SUPPORT_CELLS * spacing
    upper = renewal_bounds[1::2] + M4_PRIME_SUPPORT_CELLS * spacing
    anchor_array: np.ndarray | None = None
    if lattice_anchor is not None:
        anchor_array = np.asarray(lattice_anchor, dtype=np.float64).reshape(3)
        if np.any(~np.isfinite(anchor_array)):
            raise ValueError("lattice_anchor must be finite")
        lower = (
            anchor_array + np.floor(_snap_grid_roundoff((lower - anchor_array) / spacing)) * spacing
        )
    shape = tuple(
        (np.ceil(_snap_grid_roundoff((upper - lower) / spacing)).astype(int) + 1).tolist()
    )
    positions = _regular_grid_positions(lower, spacing, shape)
    count = len(positions)

    mesh_weight = _evaluate_weight("mesh_weight_at_node", mesh_weight_at_node, positions)
    fluid_weight = _evaluate_weight("fluid_weight_at_node", fluid_weight_at_node, positions)
    if slip_slab:
        fluid_weight[
            (positions[:, 2] < bounds[4] - 1e-10) | (positions[:, 2] > bounds[5] + 1e-10)
        ] = 0.0
    solid_interior = (
        np.zeros(count, dtype=bool)
        if interior_at_node is None
        else np.asarray(interior_at_node(positions), dtype=bool).reshape(-1)
    )
    if solid_interior.shape != (count,):
        raise ValueError(f"interior_at_node returned {solid_interior.shape}, expected ({count},)")
    blend_weight = (
        inward_cosine_blend_weight(positions, bounds, width, dead_zone, slip_slab=slip_slab)
        * mesh_weight
    )
    wall_links = (
        None
        if solid_boundary is None
        else _blocked_lattice_links(positions, shape, solid_interior, solid_boundary)
    )
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
        fvm_blend_weight=blend_weight,
        slip_slab=slip_slab,
        wall_links=wall_links,
        wall_scatter_correction=wall_scatter_correction,
    )


def _snap_grid_roundoff(coordinates: np.ndarray) -> np.ndarray:
    """Keep floor/ceil invariant under roundoff-sized translations of a grid."""
    nearest = np.rint(coordinates)
    tolerance = 64.0 * np.finfo(float).eps * np.maximum(1.0, np.abs(coordinates))
    return np.where(np.abs(coordinates - nearest) <= tolerance, nearest, coordinates)


def _blocked_lattice_links(positions, shape, solid_interior, boundary) -> np.ndarray:
    """Cache six-neighbour wall visibility using one bit per positive axis."""
    links = np.zeros(shape, dtype=np.uint8)
    flat_links = links.reshape(-1)
    strides = (shape[1] * shape[2], shape[2], 1)
    # Bound coordinate/query temporaries independently of lattice size.
    for start in range(0, len(positions), 65536):
        rows = np.arange(start, min(start + 65536, len(positions)))
        for axis, stride in enumerate(strides):
            lower = rows[(rows // stride) % shape[axis] < shape[axis] - 1]
            upper = lower + stride
            blocked = solid_interior[lower] | solid_interior[upper]
            fluid = ~blocked
            if np.any(fluid):
                blocked[fluid] = boundary.blocks_segments(
                    positions[lower[fluid]], positions[upper[fluid]]
                )
            flat_links[lower[blocked]] |= np.uint8(1 << axis)
    return links


def scatter_m4_prime_to_lattice(
    positions: np.ndarray,
    vortex_strength: np.ndarray,
    lattice: StableRenewalLattice,
    *,
    allow_slab_images: bool = False,
    position_dtype=None,
) -> np.ndarray:
    """Scatter donors with aligned direct insertion and complete M4' support.

    ``position_dtype`` retains source storage precision when a caller has
    already converted positions to double precision for geometric arithmetic.
    """
    storage_dtype = np.dtype(
        np.asarray(positions).dtype if position_dtype is None else position_dtype
    )
    if not np.issubdtype(storage_dtype, np.floating):
        storage_dtype = np.dtype(np.float64)
    position = _vectors("positions", positions)
    strength = _vectors("vortex_strength", vortex_strength)
    if len(position) != len(strength):
        raise ValueError("positions and vortex_strength must have the same length")
    if len(position) == 0:
        return np.zeros((len(lattice.positions), 3), dtype=np.float64)
    if not allow_slab_images and (
        np.any(position < lattice.renewal_bounds[::2])
        or np.any(position > lattice.renewal_bounds[1::2])
    ):
        raise ValueError("M4' donors must lie inside the physical renewal belt")

    if allow_slab_images:
        relative_all = (position - lattice.origin) / lattice.particle_spacing
        lower = np.floor(relative_all).astype(np.int64) - 1
        upper = lower + 3
        shape_array = np.asarray(lattice.shape, dtype=np.int64)
        complete = np.all((lower >= 0) & (upper < shape_array), axis=1)
        position, strength = position[complete], strength[complete]
        if len(position) == 0:
            return np.zeros((len(lattice.positions), 3), dtype=np.float64)

    relative = (position - lattice.origin) / lattice.particle_spacing
    nearest = np.rint(relative).astype(np.int64)
    shape_array = np.asarray(lattice.shape, dtype=np.int64)
    aligned = np.max(np.abs(relative - nearest), axis=1) <= _ALIGNMENT_TOLERANCE_CELLS
    if lattice.wall_links is not None:
        # GBD and renewal share a lattice; finite position storage can shift
        # an exact node by a few ulps. Do not spread those shifts through walls.
        dimensions = 3
        exact = lattice.origin + lattice.particle_spacing * nearest
        storage_roundoff = (
            4.0
            * np.finfo(storage_dtype).eps
            * np.maximum(np.abs(position[:, :dimensions]), np.abs(exact[:, :dimensions]))
        )
        if np.any(storage_roundoff >= 0.01 * lattice.particle_spacing):
            raise ValueError(
                "Wall renewal cannot resolve the particle spacing at "
                f"{storage_dtype.name} position precision"
            )
        tolerance = np.maximum(
            _ALIGNMENT_TOLERANCE_CELLS * lattice.particle_spacing, storage_roundoff
        )
        aligned = np.all(
            np.abs(position[:, :dimensions] - exact[:, :dimensions]) <= tolerance, axis=1
        )
    aligned &= np.all((nearest >= 0) & (nearest < shape_array), axis=1)
    if (
        lattice.wall_links is not None
        and np.any(~aligned)
        and lattice.wall_scatter_correction is None
    ):
        raise ValueError(
            "Wall-aware buffered renewal requires GBD-aligned particles; "
            "off-lattice M4 scatter has no wall-visible support"
        )
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
        if lattice.wall_links is not None:
            indices, corrections, _ = lattice.wall_scatter_correction(
                position[~aligned],
                strength[~aligned],
                lattice.origin,
                lattice.particle_spacing,
                lattice.shape,
            )
            np.add.at(field, tuple(indices.T), corrections)
    if lattice.wall_links is not None:
        field[lattice.solid_interior.reshape(lattice.shape)] = 0.0
    return field.reshape(-1, 3)


def gaussian_represented_vortex_strength(
    lattice_vortex_strength: np.ndarray,
    shape: tuple[int, int, int],
    particle_spacing: float,
    *,
    core_radius: float,
    slip_slab_bounds: tuple[float, float] | None = None,
    lattice_origin_z: float | None = None,
) -> np.ndarray:
    """Sample the represented Gaussian strength on a Cartesian lattice.

    lattice_vortex_strength has shape (prod(shape),3), in m³/s; shape is
    (nx,ny,nz), particle_spacing is h in m, and core_radius is sigma in m.
    The returned independent float64 array has the same shape and units.

    Each source represents Gamma*exp(-r²/sigma²)/(pi**1.5*sigma³).
    The sampled Gaussian weights are not renormalized: doing so would change
    the represented kernel. Truncation occurs beyond six core radii.
    Invalid spacing, radius or lattice length raises ValueError. Inputs are read only.
    """
    spacing = _positive_finite("particle_spacing", particle_spacing)
    radius = _positive_finite("core_radius", core_radius)
    strength = _vectors("lattice_vortex_strength", lattice_vortex_strength)
    expected = int(np.prod(shape))
    if len(strength) != expected:
        raise ValueError(f"lattice_vortex_strength has {len(strength)} rows, expected {expected}")
    half_width = int(np.ceil(6.0 * radius / spacing))
    distance = np.arange(-half_width, half_width + 1, dtype=np.float64) * spacing
    weight = spacing / (np.sqrt(np.pi) * radius) * np.exp(-((distance / radius) ** 2))
    represented = strength.reshape(*shape, 3)
    if slip_slab_bounds is None:
        for axis in range(3):
            represented = convolve1d(represented, weight, axis=axis, mode="constant", cval=0.0)
        return represented.reshape(-1, 3)
    if lattice_origin_z is None or len(shape) != 3:
        raise ValueError("slip-slab representation requires 3D lattice bounds and z origin")
    z_min, z_max = map(float, slip_slab_bounds)
    if not np.isfinite((z_min, z_max, lattice_origin_z)).all() or z_max <= z_min:
        raise ValueError("slip-slab representation needs finite increasing z planes")
    lower_twice = 2.0 * (z_min - lattice_origin_z) / spacing
    upper_twice = 2.0 * (z_max - lattice_origin_z) / spacing
    period = upper_twice - lower_twice
    if (
        abs(lower_twice - round(lower_twice)) > 1e-5
        or abs(upper_twice - round(upper_twice)) > 1e-5
        or abs(period - round(period)) > 1e-5
    ):
        raise ValueError("slip planes must align to lattice nodes or half nodes")
    lower_twice = int(round(lower_twice))
    period = int(round(period))
    if period <= 0:
        raise ValueError("slip-slab image translation must be positive")
    z_nodes = lattice_origin_z + spacing * np.arange(shape[2])
    physical_z = (z_nodes >= z_min - 1e-8 * spacing) & (z_nodes <= z_max + 1e-8 * spacing)
    if not np.any(physical_z):
        raise ValueError("slip-slab lattice has no physical z nodes")

    # Convolution commutes with reflection in z. Smooth only physical sources
    # in x/y, then place their full even/odd image family on a temporary z
    # extension. Adding coincident odd images at node-aligned faces matches the
    # particle induction convention (normal doubles; tangential cancels).
    xy = represented.copy()
    xy[:, :, ~physical_z, :] = 0.0
    for axis in (0, 1):
        xy = convolve1d(xy, weight, axis=axis, mode="constant", cval=0.0)
    pad = half_width
    extended = np.zeros((shape[0], shape[1], shape[2] + 2 * pad, 3), dtype=np.float64)
    lo, hi = -pad, shape[2] - 1 + pad
    for source_z in np.flatnonzero(physical_z):
        source_plane = xy[:, :, source_z, :]
        for base, odd in ((int(source_z), False), (lower_twice - int(source_z), True)):
            first = int(np.ceil((lo - base) / period))
            last = int(np.floor((hi - base) / period))
            for image in range(first, last + 1):
                target_z = base + image * period + pad
                if odd:
                    extended[:, :, target_z, :2] -= source_plane[:, :, :2]
                    extended[:, :, target_z, 2] += source_plane[:, :, 2]
                else:
                    extended[:, :, target_z, :] += source_plane
    represented = convolve1d(extended, weight, axis=2, mode="constant", cval=0.0)
    return represented[:, :, pad : pad + shape[2], :].reshape(-1, 3)


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
    fvm_blend_weight: np.ndarray,
    shape: tuple[int, int, int],
    particle_spacing: float,
    *,
    core_radius: float,
    amplification_cap: float = DEFAULT_AMPLIFICATION_CAP,
    output_weight: np.ndarray | None = None,
    compute_final_representation: bool = True,
    slip_slab_bounds: tuple[float, float] | None = None,
    lattice_origin_z: float | None = None,
) -> RepresentedStateBlend:
    """Correct the represented-field mismatch while preserving VPM coefficients.

    The strength arrays contain volume-integrated Gamma in m³/s.
    The represented field uses three-dimensional Gaussian convolution.
    With a binary output support mask, matching FVM/represented-VPM fields on
    active nodes are a fixed point, including through a spatially varying
    blending ramp. Fractional output weights additionally scale coefficients.
    The approximate inverse acts only on the active physical mismatch.
    Each correction is limited along its update direction so
    no node grows beyond the larger of its previous magnitude and ``cap``
    times the maximum physical target. This prevents repeated inversion of
    unresolved wall structure from accumulating unbounded coefficients.
    """
    cap = float(amplification_cap)
    if not np.isfinite(cap) or cap < 1.0:
        raise ValueError("amplification_cap must be finite and at least one")
    vpm_strength = _vectors("vpm_vortex_strength", vpm_vortex_strength)
    fvm_strength = _vectors("fvm_target_vortex_strength", fvm_target_vortex_strength)
    blend_weight = np.asarray(fvm_blend_weight, dtype=np.float64).reshape(-1)
    if vpm_strength.shape != fvm_strength.shape or blend_weight.shape != (len(vpm_strength),):
        raise ValueError("blend inputs must share one lattice shape")
    weight: np.ndarray | None = None
    if output_weight is not None:
        weight = np.asarray(output_weight, dtype=np.float64).reshape(-1)
        if weight.shape != (len(vpm_strength),):
            raise ValueError("output_weight must share the transfer lattice shape")
    physical = np.ones(len(vpm_strength), dtype=bool)
    if weight is not None:
        physical &= weight > 0.0
    if slip_slab_bounds is not None:
        if lattice_origin_z is None:
            raise ValueError("slip-slab representation requires a z lattice origin")
        z_nodes = lattice_origin_z + particle_spacing * np.arange(shape[2])
        z_min, z_max = slip_slab_bounds
        physical_z = (z_nodes >= z_min - 1e-8 * particle_spacing) & (
            z_nodes <= z_max + 1e-8 * particle_spacing
        )
        physical &= np.broadcast_to(physical_z, shape).reshape(-1)

    represented_vpm = gaussian_represented_vortex_strength(
        vpm_strength,
        shape,
        particle_spacing,
        core_radius=core_radius,
        slip_slab_bounds=slip_slab_bounds,
        lattice_origin_z=lattice_origin_z,
    )
    physical_target = represented_vpm + blend_weight[:, None] * (fvm_strength - represented_vpm)
    physical_target[~physical] = 0.0

    # FVM values are physical vorticity; VPM values are Gaussian coefficients.
    # Blending those two representations directly would damp even F = G(v).
    # Instead invert only r = eta * (F - G(v)), with eta inside the operator:
    # v_new = v + [I + beta * (I - G)] r. This also retains the blending-weight ramp
    # commutator; multiplying a deconvolved difference by eta is not equivalent.
    mismatch = blend_weight[:, None] * (fvm_strength - represented_vpm)
    mismatch[~physical] = 0.0
    represented_mismatch = gaussian_represented_vortex_strength(
        mismatch,
        shape,
        particle_spacing,
        core_radius=core_radius,
        slip_slab_bounds=slip_slab_bounds,
        lattice_origin_z=lattice_origin_z,
    )
    residual = mismatch - represented_mismatch
    denominator = float(np.linalg.norm(physical_target[physical])) + 1.0e-30
    residual_before = float(np.linalg.norm(residual[physical])) / denominator

    correction_gain = min(cap - 1.0, 1.0)
    correction = mismatch + correction_gain * residual
    baseline = vpm_strength.copy()
    if weight is not None:
        baseline *= weight[:, None]
        correction *= weight[:, None]
    baseline[~physical] = 0.0
    correction[~physical] = 0.0
    target_maximum = (
        float(np.linalg.norm(physical_target[physical], axis=1).max(initial=0.0)) + 1.0e-30
    )
    bound_squared = np.maximum(
        np.einsum("ij,ij->i", baseline, baseline), (cap * target_maximum) ** 2
    )
    corrected_strength = baseline + correction
    limited = np.einsum("ij,ij->i", corrected_strength, corrected_strength) > bound_squared
    if np.any(limited):
        previous = baseline[limited]
        delta = correction[limited]
        a = np.einsum("ij,ij->i", delta, delta)
        b = np.einsum("ij,ij->i", previous, delta)
        c = np.minimum(np.einsum("ij,ij->i", previous, previous) - bound_squared[limited], 0.0)
        root = np.sqrt(np.maximum(b * b - a * c, 0.0))
        # Stable positive quadratic root, including an already saturated node.
        fraction = np.zeros_like(a)
        outward = b >= 0.0
        np.divide(-c, root + b, out=fraction, where=outward & (root + b > 0.0))
        np.divide(root - b, a, out=fraction, where=~outward & (a > 0.0))
        corrected_strength[limited] = previous + np.clip(fraction, 0.0, 1.0)[:, None] * delta
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
            slip_slab_bounds=slip_slab_bounds,
            lattice_origin_z=lattice_origin_z,
        )
        residual_after = (
            float(np.linalg.norm((physical_target - represented_corrected)[physical])) / denominator
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
    """Apply continuous non-negative-garrote shrinkage."""
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
    *,
    labels: np.ndarray | None = None,
    wall_links: np.ndarray | None = None,
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
        if labels is not None:
            neighbours[index] &= (labels >= 0) & (labels == np.roll(labels, -step, axis=axis))
        if wall_links is not None:
            edge_links = wall_links if step == 1 else np.roll(wall_links, 1, axis=axis)
            neighbours[index] &= (edge_links & (1 << axis)) == 0
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


def _recover_wall_components(lattice, original, redistributed, labels, recover):
    """Close each fluid component, retaining its original state if rank is lost."""
    raw = redistributed.copy()
    corrected = redistributed.copy()
    flat_labels = labels.reshape(-1)
    rows = np.flatnonzero(flat_labels >= 0)
    if not len(rows):
        return raw, corrected
    if np.any(flat_labels[rows] != flat_labels[rows[0]]):
        rows = rows[np.argsort(flat_labels[rows], kind="stable")]
    boundaries = np.r_[0, np.flatnonzero(np.diff(flat_labels[rows])) + 1, len(rows)]
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        component = rows[start:stop]
        retained = component[np.linalg.norm(raw[component], axis=1) > 0.0]
        target = vortex_invariants(lattice.positions[component], original[component])
        try:
            if len(retained) < 2:
                raise ValueError("insufficient component support")
            result = recover(
                lattice.positions[retained],
                raw[retained],
                target,
                volumes=np.full(len(retained), lattice.particle_volume),
            )
            if not np.all(np.isfinite(result)):
                raise FloatingPointError("nonfinite component correction")
            corrected[retained] = result
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            # A pruning threshold cannot justify moving a component's strength
            # into another fluid region. Keep its original support instead.
            raw[component] = original[component]
            corrected[component] = original[component]
    return raw, corrected


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
        """Return the number of active particles represented by this result."""
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
    release_prune_threshold: float | None = None,
    core_radius_ratio: float = CORE_RADIUS_RATIO,
    amplification_cap: float = DEFAULT_AMPLIFICATION_CAP,
    maximum_particle_count: int | None = None,
    freestream_speed: float = 0.0,
    time_step_size: float = 0.0,
    compute_diagnostics: bool = True,
) -> StableRenewalResult:
    """Renew the whole buffered belt and preserve the outer Lagrangian wake.

    This is the side-effect-free numerical core of the long-running method.
    The caller supplies a synchronized FVM velocity-trace target through
    ``fvm_vortex_strength_at_node`` and may then atomically replace the VPM
    particle arrays with the returned state. ``prune_threshold`` is the
    particle-strength cutoff in the FVM-owned interior. A distinct
    ``release_prune_threshold`` is blended in as FVM blending weight decreases and
    applies at the transfer surface; ``None`` uses a uniform threshold.
    """
    position_dtype = np.asarray(positions).dtype
    position = _vectors("positions", positions)
    strength = _vectors("vortex_strength", vortex_strength)
    if len(position) != len(strength):
        raise ValueError("positions and vortex_strength must have the same length")
    spacing = lattice.particle_spacing
    invariant_recovery = recover_vortex_invariants
    ratio = _positive_finite("core_radius_ratio", core_radius_ratio)
    threshold = _nonnegative_finite("prune_threshold", prune_threshold)
    release_threshold = (
        threshold
        if release_prune_threshold is None
        else _nonnegative_finite("release_prune_threshold", release_prune_threshold)
    )
    if release_threshold > threshold:
        raise ValueError("release_prune_threshold must not exceed prune_threshold")
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
    if lattice.slip_slab and np.any(
        valid
        & (
            (position[:, 2] < lattice.transfer_box[4] - 1e-7)
            | (position[:, 2] > lattice.transfer_box[5] + 1e-7)
        )
    ):
        raise ValueError("physical renewal particle is outside the slip slab")
    in_renewal_belt = valid & np.all(
        (position >= lattice.renewal_bounds[::2]) & (position <= lattice.renewal_bounds[1::2]),
        axis=1,
    )
    preserved_outer = valid & ~in_renewal_belt

    vpm_lattice_strength = scatter_m4_prime_to_lattice(
        position[in_renewal_belt],
        tapered_strength[in_renewal_belt],
        lattice,
        position_dtype=position_dtype,
    )
    if lattice.slip_slab and np.any(in_renewal_belt):
        source = position[in_renewal_belt]
        axial_image = tapered_strength[in_renewal_belt] * np.array([-1.0, -1.0, 1.0])
        for plane in lattice.transfer_box[4:6]:
            image_position = source.copy()
            image_position[:, 2] = 2.0 * plane - image_position[:, 2]
            vpm_lattice_strength += scatter_m4_prime_to_lattice(
                image_position,
                axial_image,
                lattice,
                allow_slab_images=True,
                position_dtype=position_dtype,
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
    # The taper can be positive just inside the wall, but those nodes cannot
    # store particles. Exclude their mismatch before the approximate inverse
    # can spread an impossible solid-interior target into fluid coefficients.
    output_weight = lattice.fluid_weight * ~lattice.solid_interior
    blend = blend_represented_state(
        vpm_lattice_strength,
        fvm_target,
        lattice.fvm_blend_weight,
        lattice.shape,
        spacing,
        core_radius=base_core_radius,
        amplification_cap=amplification_cap,
        output_weight=output_weight,
        compute_final_representation=compute_diagnostics,
        slip_slab_bounds=tuple(lattice.transfer_box[4:6]) if lattice.slip_slab else None,
        lattice_origin_z=float(lattice.origin[2]) if lattice.slip_slab else None,
    )
    comparison_weight = output_weight * lattice.mesh_weight
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

    pre_prune_strength = blend.vortex_strength
    if lattice.wall_links is not None:
        pre_prune_strength = pre_prune_strength.copy()
        pre_prune_strength[lattice.solid_interior] = 0.0
    pre_prune_invariants = vortex_invariants(lattice.positions, pre_prune_strength)
    magnitude_before = np.linalg.norm(pre_prune_strength, axis=1)
    labels = (
        None
        if lattice.wall_links is None
        else connected_grid_components(
            magnitude_before.reshape(lattice.shape),
            np.zeros(lattice.shape, dtype=np.int8),
            lattice.wall_links,
        )
    )
    # Interior state is regenerated from the FVM. At the release surface the
    # VPM is the sole owner, so pruning must fall to its own resolved GBD floor.
    local_threshold = release_threshold + (threshold - release_threshold) * lattice.fvm_blend_weight
    shrunk, removed = soft_prune_vortex_strength(pre_prune_strength, local_threshold)
    redistributed = redistribute_pruned_vortex_strength_locally(
        removed,
        shrunk,
        lattice.shape,
        labels=labels,
        wall_links=lattice.wall_links,
    )
    redistributed[lattice.solid_interior] = 0.0
    wall_corrected = None
    if labels is not None:
        redistributed, wall_corrected = _recover_wall_components(
            lattice, pre_prune_strength, redistributed, labels, invariant_recovery
        )
    keep = np.linalg.norm(redistributed, axis=1) > 0.0
    pruned = (magnitude_before > 0.0) & ~keep
    active_l1 = float(magnitude_before[magnitude_before > 0.0].sum())
    pruned_l1 = float(magnitude_before[pruned].sum())

    renewed_position = lattice.positions[keep]
    raw_renewed_strength = redistributed[keep]
    renewed_strength = raw_renewed_strength if wall_corrected is None else wall_corrected[keep]
    renewed_volume = np.full(len(renewed_position), lattice.particle_volume, dtype=np.float64)
    raw_invariants = vortex_invariants(renewed_position, raw_renewed_strength)
    conservation_raw_mismatch = _invariant_residual(pre_prune_invariants, raw_invariants)
    conservation_target_invariants = pre_prune_invariants
    conservation_raw_invariants = raw_invariants
    conservation_reference_strength_l1 = active_l1
    if wall_corrected is None and len(renewed_position) > 1:
        renewed_strength = invariant_recovery(
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
            slip_slab_bounds=tuple(lattice.transfer_box[4:6]) if lattice.slip_slab else None,
            lattice_origin_z=float(lattice.origin[2]) if lattice.slip_slab else None,
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
                np.full(
                    np.count_nonzero(preserved_for_append),
                    lattice.particle_volume,
                    dtype=np.float64,
                ),
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
        if lattice.wall_links is not None:
            raise RuntimeError(
                "Wall-aware renewal exceeds particle capacity after component-local pruning; "
                "increase capacity rather than applying a global cross-wall correction"
            )
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
        output_strength = invariant_recovery(
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
    "inward_cosine_blend_weight",
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
