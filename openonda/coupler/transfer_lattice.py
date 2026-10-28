"""Vorticity transfer on the fixed transfer lattice, compatible with GBD diffusion.

The particles of the remeshing region are remeshed onto the transfer lattice,
blended with the FVM target strengths, corrected, thresholded and returned as
transfer-lattice particles; particles outside the remeshing region are kept.
The operators return particle arrays and do not modify a solver. Transfer-lattice
particles sit on the lattice nodes and carry the base core radius.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import convolve1d

from openonda.core._numba import cacheable_njit as njit
from openonda.core.grid_connectivity import connected_grid_components
from openonda.core.slip_slab_images import add_slab_images

from .m4_kernel import _m4_prime_scalar, m4_prime

ArrayFunction = Callable[[np.ndarray], np.ndarray]

CORE_RADIUS_RATIO = 1.0
M4_PRIME_SUPPORT_CELLS = 2.0
DEFAULT_MAX_STRENGTH_RATIO = 2.0
_ALIGNMENT_TOLERANCE_CELLS = 1.0e-5


@njit(cache=True, fastmath=False)
def _scatter_unaligned_m4_prime(
    relative_position: np.ndarray,
    lower_stencil_index: np.ndarray,
    vortex_strength: np.ndarray,
    shape: np.ndarray,
) -> np.ndarray:
    result = np.zeros((shape[0], shape[1], shape[2], 3), dtype=np.float64)
    for particle in range(len(relative_position)):
        for offset_x in range(4):
            index_x = lower_stencil_index[particle, 0] + offset_x
            weight_x = _m4_prime_scalar(relative_position[particle, 0] - index_x)
            for offset_y in range(4):
                index_y = lower_stencil_index[particle, 1] + offset_y
                weight_y = _m4_prime_scalar(relative_position[particle, 1] - index_y)
                for offset_z in range(4):
                    index_z = lower_stencil_index[particle, 2] + offset_z
                    weight_z = _m4_prime_scalar(relative_position[particle, 2] - index_z)
                    weight = weight_x * weight_y * weight_z
                    result[index_x, index_y, index_z, 0] += weight * vortex_strength[particle, 0]
                    result[index_x, index_y, index_z, 1] += weight * vortex_strength[particle, 1]
                    result[index_x, index_y, index_z, 2] += weight * vortex_strength[particle, 2]
    return result


def fvm_target_strength(
    positions: np.ndarray,
    particle_spacing: float,
    velocity_at: ArrayFunction,
) -> np.ndarray:
    """Return the FVM target strength: the Stokes integral over the faces of a lattice cube.

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


def blending_weight(
    positions: np.ndarray,
    transfer_region_bounds: np.ndarray | list[float] | tuple[float, ...],
    blend_start_depth: float,
    blend_end_depth: float,
    *,
    slip_slab: bool = False,
) -> np.ndarray:
    """Return the blending weight ``η`` at each world-frame point.

    Positions have shape (N,3), and transfer_region_bounds contains
    (xmin,xmax,ymin,ymax,zmin,zmax), all in m. With ``d`` the depth below the
    transfer surface, the blending weight is zero in the zero-weight layer
    ``d ≤ blend_start_depth``, rises with a cosine and reaches one at
    ``d = blend_end_depth`` (both non-negative, in m). A zero blend_end_depth gives
    a sharp mask, one everywhere inside the transfer surface.
    With slip_slab=True, z faces keep the full blending weight through the closed
    physical span but points outside it have zero blending weight.
    The returned independent float64 array has shape (N,) and values in [0,1].
    Invalid bounds, vectors or depths raise ValueError; inputs are unchanged.
    """
    position = _vectors("positions", positions)
    bounds = _bounds(transfer_region_bounds)
    start_depth = _nonnegative_finite("blend_start_depth", blend_start_depth)
    width = _nonnegative_finite("blend_end_depth", blend_end_depth)
    if width > 0.0 and start_depth >= width:
        raise ValueError("blend_start_depth must be smaller than blend_end_depth")

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
    ramp = (face_distance > start_depth) & (face_distance < width)
    if np.any(ramp):
        phase = (face_distance[ramp] - start_depth) / (width - start_depth)
        blend_weight[ramp] = 0.5 * (1.0 - np.cos(np.pi * phase))
    if slip_slab:
        blend_weight[(position[:, 2] < bounds[4]) | (position[:, 2] > bounds[5])] = 0.0
    return blend_weight


@dataclass(frozen=True)
class VortexMoments:
    """Total vortex strength and the linear and angular impulse per unit density."""

    total_vortex_strength: np.ndarray
    linear_impulse: np.ndarray
    angular_impulse: np.ndarray


def vortex_moments(
    positions: np.ndarray,
    vortex_strength: np.ndarray,
) -> VortexMoments:
    """Return the total vortex strength (m³/s), linear impulse (m⁴/s) and angular impulse (m⁵/s)."""
    position = _vectors("positions", positions)
    strength = _vectors("vortex_strength", vortex_strength)
    if len(position) != len(strength):
        raise ValueError("positions and vortex_strength must have the same length")
    if len(position) == 0:
        zero = np.zeros(3, dtype=np.float64)
        return VortexMoments(zero.copy(), zero.copy(), zero.copy())
    position_cross_strength = np.cross(position, strength)
    return VortexMoments(
        total_vortex_strength=strength.sum(axis=0, dtype=np.float64),
        linear_impulse=0.5 * position_cross_strength.sum(axis=0, dtype=np.float64),
        angular_impulse=(1.0 / 3.0)
        * np.cross(position, position_cross_strength).sum(axis=0, dtype=np.float64),
    )


def correct_vortex_moments(
    positions: np.ndarray,
    vortex_strength: np.ndarray,
    target: VortexMoments,
    *,
    volumes: np.ndarray,
) -> np.ndarray:
    """Apply the moment correction that restores the total strength and linear impulse.

    The correction is the volume-weighted minimum-norm change of the strengths
    that matches both moments of ``target``.
    """
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
            raise ValueError("a non-zero total strength or impulse needs particles to carry it")
        return strength
    if count < 2:
        raise ValueError("the moment correction needs at least two particles")

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
    # Total strength and impulse have different dimensions. An absolute 1e-14
    # skip threshold loses small but resolved moments and can exceed the
    # storage-precision conservation tolerance of the vorticity transfer.
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
            f"moment-correction matrix is ill-conditioned ({condition:.3e})"
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
class TransferLattice:
    """Fixed Cartesian transfer lattice and its dimensionless node weights.

    Lengths, bounds and positions are in m. ``shape`` is (nx,ny,nz), with
    positions flattened in C order. mesh_weight, fluid_weight, blending_weight
    and solid_interior each have one entry per node. Each particle has volume h³.
    The frozen dataclass prevents attribute reassignment but does not make its
    NumPy arrays immutable; callers must treat these shared arrays as read only.
    """

    transfer_region_bounds: np.ndarray
    remeshing_region_bounds: np.ndarray
    origin: np.ndarray
    shape: tuple[int, int, int]
    positions: np.ndarray
    particle_spacing: float
    remeshing_extension_length: float
    lattice_anchor: np.ndarray | None
    mesh_weight: np.ndarray
    fluid_weight: np.ndarray
    solid_interior: np.ndarray
    blending_weight: np.ndarray
    slip_slab: bool = False
    wall_crossing_links: np.ndarray | None = None
    wall_scatter_correction: Callable | None = None

    @property
    def particle_volume(self) -> float:
        """Return the volume of each transfer-lattice particle in m³.

        The uniform volume is h³, where h is particle_spacing in m.
        """
        return self.particle_spacing**3


def build_transfer_lattice(
    transfer_region_bounds: np.ndarray | list[float] | tuple[float, ...],
    particle_spacing: float,
    *,
    remeshing_extension_length: float,
    blend_start_depth: float = 0.0,
    blend_end_depth: float,
    lattice_anchor: np.ndarray | None = None,
    mesh_weight_at_node: ArrayFunction | None = None,
    fluid_weight_at_node: ArrayFunction | None = None,
    interior_at_node: ArrayFunction | None = None,
    slip_slab: bool = False,
    solid_boundary=None,
    wall_scatter_correction: Callable | None = None,
) -> TransferLattice:
    """Build the transfer lattice that covers the remeshing region and its M4' support.

    Parameters
    ----------
    transfer_region_bounds : array_like, shape (6,)
        (xmin,xmax,ymin,ymax,zmin,zmax) in m, with increasing bounds.
    particle_spacing : float
        Positive uniform node spacing h in m.
    remeshing_extension_length, blend_start_depth, blend_end_depth : float
        Non-negative lengths in m: the outward extension ``L_remesh`` of the
        remeshing region and the blending depths ``δ_0 < δ_1`` (``δ_1 = 0``
        disables blending).
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
        Static solid-query interface. Its segment queries mark the lattice links
        that cross the wall, so weak-strength removal and the moment correction
        act on each fluid-connected component separately.
    wall_scatter_correction : callable, optional
        GBD fluid-side M4' correction for off-lattice three-dimensional particles.
        Called with positions, strengths, origin, spacing and shape; returns
        sparse node indices, additive strength corrections and diagnostics.

    Returns
    -------
    TransferLattice
        Geometry and masks for repeated vorticity transfers. Newly allocated arrays
        remain mutable despite the frozen container and must be treated as read only.

    Raises
    ------
    ValueError
        If lengths, bounds, lattice phase or mask shapes are invalid.

    Notes
    -----
    The remeshing region extends transfer_region_bounds outward by
    remeshing_extension_length. Two more nodes on every active axis hold the
    complete M4' support of each particle: 4-by-4-by-4 nodes.
    """
    bounds = _bounds(transfer_region_bounds)
    spacing = _positive_finite("particle_spacing", particle_spacing)
    extension = _nonnegative_finite("remeshing_extension_length", remeshing_extension_length)
    start_depth = _nonnegative_finite("blend_start_depth", blend_start_depth)
    end_depth = _nonnegative_finite("blend_end_depth", blend_end_depth)
    if end_depth > 0.0 and start_depth >= end_depth:
        raise ValueError("blend_start_depth must be smaller than blend_end_depth")

    remeshing_region_bounds = bounds.copy()
    remeshing_region_bounds[::2] -= extension
    remeshing_region_bounds[1::2] += extension
    if slip_slab:
        remeshing_region_bounds[4:6] = bounds[4:6]
    lower = remeshing_region_bounds[::2] - M4_PRIME_SUPPORT_CELLS * spacing
    upper = remeshing_region_bounds[1::2] + M4_PRIME_SUPPORT_CELLS * spacing
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
        blending_weight(positions, bounds, start_depth, end_depth, slip_slab=slip_slab)
        * mesh_weight
    )
    wall_crossing_links = (
        None
        if solid_boundary is None
        else _wall_crossing_links(positions, shape, solid_interior, solid_boundary)
    )
    return TransferLattice(
        transfer_region_bounds=bounds,
        remeshing_region_bounds=remeshing_region_bounds,
        origin=lower,
        shape=shape,
        positions=positions,
        particle_spacing=spacing,
        remeshing_extension_length=extension,
        lattice_anchor=anchor_array,
        mesh_weight=mesh_weight,
        fluid_weight=fluid_weight,
        solid_interior=solid_interior,
        blending_weight=blend_weight,
        slip_slab=slip_slab,
        wall_crossing_links=wall_crossing_links,
        wall_scatter_correction=wall_scatter_correction,
    )


def _snap_grid_roundoff(coordinates: np.ndarray) -> np.ndarray:
    """Keep floor/ceil unchanged under roundoff-sized translations of a grid."""
    nearest = np.rint(coordinates)
    tolerance = 64.0 * np.finfo(float).eps * np.maximum(1.0, np.abs(coordinates))
    return np.where(np.abs(coordinates - nearest) <= tolerance, nearest, coordinates)


def _wall_crossing_links(positions, shape, solid_interior, boundary) -> np.ndarray:
    """Mark the lattice links that cross the wall, one bit per positive axis.

    A set bit closes the link to the diffusion flux, the strength redistribution
    and the fluid-connected components; the diffusion-flux mask is its complement.
    """
    links = np.zeros(shape, dtype=np.uint8)
    flat_links = links.reshape(-1)
    strides = (shape[1] * shape[2], shape[2], 1)
    # Bound coordinate/query temporaries independently of lattice size.
    for start in range(0, len(positions), 65536):
        rows = np.arange(start, min(start + 65536, len(positions)))
        for axis, stride in enumerate(strides):
            lower = rows[(rows // stride) % shape[axis] < shape[axis] - 1]
            upper = lower + stride
            crosses_wall = solid_interior[lower] | solid_interior[upper]
            fluid = ~crosses_wall
            if np.any(fluid):
                crosses_wall[fluid] = boundary.segments_cross_wall(
                    positions[lower[fluid]], positions[upper[fluid]]
                )
            flat_links[lower[crosses_wall]] |= np.uint8(1 << axis)
    return links


def scatter_m4_prime_to_lattice(
    positions: np.ndarray,
    vortex_strength: np.ndarray,
    lattice: TransferLattice,
    *,
    position_dtype=None,
) -> np.ndarray:
    """Remesh particles onto the lattice: node-aligned ones directly, the others with M4'.

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
    if np.any(position < lattice.remeshing_region_bounds[::2]) or np.any(
        position > lattice.remeshing_region_bounds[1::2]
    ):
        raise ValueError("remeshed particles must lie inside the remeshing region")

    relative = (position - lattice.origin) / lattice.particle_spacing
    nearest = np.rint(relative).astype(np.int64)
    shape_array = np.asarray(lattice.shape, dtype=np.int64)
    aligned = np.max(np.abs(relative - nearest), axis=1) <= _ALIGNMENT_TOLERANCE_CELLS
    if lattice.wall_crossing_links is not None:
        # GBD diffusion and the vorticity transfer share a lattice; finite position
        # storage can shift an exact node by a few ulps. Do not spread those shifts
        # through walls.
        dimensions = 3
        exact = lattice.origin + lattice.particle_spacing * nearest
        storage_roundoff = (
            4.0
            * np.finfo(storage_dtype).eps
            * np.maximum(np.abs(position[:, :dimensions]), np.abs(exact[:, :dimensions]))
        )
        if np.any(storage_roundoff >= 0.01 * lattice.particle_spacing):
            raise ValueError(
                "The vorticity transfer next to a wall cannot resolve the particle spacing at "
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
        lattice.wall_crossing_links is not None
        and np.any(~aligned)
        and lattice.wall_scatter_correction is None
    ):
        raise ValueError(
            "The vorticity transfer next to a wall requires GBD-aligned particles; "
            "an off-lattice M4' scatter has no fluid-side remeshing stencil"
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
            raise RuntimeError("the transfer lattice does not contain a complete M4' stencil")
        field += _scatter_unaligned_m4_prime(
            relative_free,
            lower_stencil,
            strength[~aligned],
            shape_array,
        )
        if lattice.wall_crossing_links is not None:
            indices, corrections, _ = lattice.wall_scatter_correction(
                position[~aligned],
                strength[~aligned],
                lattice.origin,
                lattice.particle_spacing,
                lattice.shape,
            )
            np.add.at(field, tuple(indices.T), corrections)
    if lattice.wall_crossing_links is not None:
        field[lattice.solid_interior.reshape(lattice.shape)] = 0.0
    return field.reshape(-1, 3)


def gaussian_representation(
    lattice_vortex_strength: np.ndarray,
    shape: tuple[int, int, int],
    particle_spacing: float,
    *,
    core_radius: float,
    slip_slab_bounds: tuple[float, float] | None = None,
    lattice_origin_z: float | None = None,
) -> np.ndarray:
    """Return the particle-represented vorticity ``ℳ_σ[Γ]`` on a Cartesian lattice, as strengths.

    lattice_vortex_strength has shape (prod(shape),3), in m³/s; shape is
    (nx,ny,nz), particle_spacing is h in m, and core_radius is sigma in m.
    The returned independent float64 array has the same shape and units.

    Each source represents Gamma*exp(-r²/sigma²)/(pi**1.5*sigma³).
    The sampled Gaussian weights are not renormalized: doing so would change
    the Gaussian kernel. Truncation occurs beyond six core radii.
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
class StrengthCorrection:
    """Limited lattice strengths of the strength correction and its residuals.

    ``limited_strength`` holds ``Γ_lim``, ``blended_target`` ``Γ_blend`` and
    ``represented_strength`` the particle-represented vorticity ``ℳ_σ[Γ_lim]``,
    all as strengths in m³/s. The residuals are relative to ``Γ_blend`` and
    ``strength_ratio`` is ``κ``.
    """

    limited_strength: np.ndarray
    blended_target: np.ndarray
    represented_strength: np.ndarray | None
    residual_before_correction: float
    residual_after_correction: float | None
    strength_ratio: float


def correct_lattice_strength(
    vpm_vortex_strength: np.ndarray,
    fvm_target_vortex_strength: np.ndarray,
    blending_weight: np.ndarray,
    shape: tuple[int, int, int],
    particle_spacing: float,
    *,
    core_radius: float,
    max_strength_ratio: float = DEFAULT_MAX_STRENGTH_RATIO,
    output_weight: np.ndarray | None = None,
    compute_final_representation: bool = True,
    slip_slab_bounds: tuple[float, float] | None = None,
    lattice_origin_z: float | None = None,
) -> StrengthCorrection:
    """Correct the remeshed strengths toward the blended target and limit them.

    The strength arrays contain volume-integrated vorticity ``Γ`` in m³/s, and the
    particle-represented vorticity is their three-dimensional Gaussian
    convolution ``ℳ_σ``. The weighted mismatch ``Γ_mis = η (Γ^FVM − ℳ_σ[Γ^VPM])``
    receives one Van Cittert step, ``δΓ_corr = Γ_mis + γ (Γ_mis − ℳ_σ[Γ_mis])``
    with gain ``γ = min(κ_max − 1, 1)``. Where the FVM and particle-represented
    vorticity already agree the strengths are unchanged, also inside the blending
    ramp; fractional output weights scale the strengths. Each corrected strength
    is limited along its update direction so its magnitude does not exceed the
    larger of its remeshed magnitude and ``κ_max`` times the peak target
    strength, which keeps repeated corrections of unresolved wall structure
    bounded.
    """
    max_ratio = float(max_strength_ratio)
    if not np.isfinite(max_ratio) or max_ratio < 1.0:
        raise ValueError("max_strength_ratio must be finite and at least one")
    vpm_strength = _vectors("vpm_vortex_strength", vpm_vortex_strength)
    fvm_strength = _vectors("fvm_target_vortex_strength", fvm_target_vortex_strength)
    blend_weight = np.asarray(blending_weight, dtype=np.float64).reshape(-1)
    if vpm_strength.shape != fvm_strength.shape or blend_weight.shape != (len(vpm_strength),):
        raise ValueError("the remeshed, target and blending arrays must share one lattice shape")
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

    represented_vpm = gaussian_representation(
        vpm_strength,
        shape,
        particle_spacing,
        core_radius=core_radius,
        slip_slab_bounds=slip_slab_bounds,
        lattice_origin_z=lattice_origin_z,
    )
    blended_target = represented_vpm + blend_weight[:, None] * (fvm_strength - represented_vpm)
    blended_target[~physical] = 0.0

    # FVM strengths sample the vorticity; particle strengths are Gaussian
    # coefficients. Blending the two directly would damp even F = M(v). Instead
    # correct only r = eta (F - M(v)), with eta inside the operator:
    # v_new = v + [I + gamma (I - M)] r. This keeps the commutator of the blending
    # ramp; multiplying a deconvolved difference by eta is not equivalent.
    weighted_mismatch = blend_weight[:, None] * (fvm_strength - represented_vpm)
    weighted_mismatch[~physical] = 0.0
    represented_weighted_mismatch = gaussian_representation(
        weighted_mismatch,
        shape,
        particle_spacing,
        core_radius=core_radius,
        slip_slab_bounds=slip_slab_bounds,
        lattice_origin_z=lattice_origin_z,
    )
    residual = weighted_mismatch - represented_weighted_mismatch
    denominator = float(np.linalg.norm(blended_target[physical])) + 1.0e-30
    residual_before = float(np.linalg.norm(residual[physical])) / denominator

    correction_gain = min(max_ratio - 1.0, 1.0)
    strength_correction = weighted_mismatch + correction_gain * residual
    baseline = vpm_strength.copy()
    if weight is not None:
        baseline *= weight[:, None]
        strength_correction *= weight[:, None]
    baseline[~physical] = 0.0
    strength_correction[~physical] = 0.0
    peak_target_strength = (
        float(np.linalg.norm(blended_target[physical], axis=1).max(initial=0.0)) + 1.0e-30
    )
    bound_squared = np.maximum(
        np.einsum("ij,ij->i", baseline, baseline), (max_ratio * peak_target_strength) ** 2
    )
    limited_strength = baseline + strength_correction
    exceeds_limit = np.einsum("ij,ij->i", limited_strength, limited_strength) > bound_squared
    if np.any(exceeds_limit):
        previous = baseline[exceeds_limit]
        delta = strength_correction[exceeds_limit]
        a = np.einsum("ij,ij->i", delta, delta)
        b = np.einsum("ij,ij->i", previous, delta)
        c = np.minimum(
            np.einsum("ij,ij->i", previous, previous) - bound_squared[exceeds_limit], 0.0
        )
        root = np.sqrt(np.maximum(b * b - a * c, 0.0))
        # Stable positive quadratic root, including an already saturated node.
        update_fraction = np.zeros_like(a)
        outward = b >= 0.0
        np.divide(-c, root + b, out=update_fraction, where=outward & (root + b > 0.0))
        np.divide(root - b, a, out=update_fraction, where=~outward & (a > 0.0))
        limited_strength[exceeds_limit] = (
            previous + np.clip(update_fraction, 0.0, 1.0)[:, None] * delta
        )
    strength_ratio = (
        float(np.linalg.norm(limited_strength, axis=1).max(initial=0.0)) / peak_target_strength
    )

    represented_limited: np.ndarray | None = None
    residual_after: float | None = None
    if compute_final_representation:
        represented_limited = gaussian_representation(
            limited_strength,
            shape,
            particle_spacing,
            core_radius=core_radius,
            slip_slab_bounds=slip_slab_bounds,
            lattice_origin_z=lattice_origin_z,
        )
        residual_after = (
            float(np.linalg.norm((blended_target - represented_limited)[physical])) / denominator
        )
    return StrengthCorrection(
        limited_strength=limited_strength,
        blended_target=blended_target,
        represented_strength=represented_limited,
        residual_before_correction=residual_before,
        residual_after_correction=residual_after,
        strength_ratio=strength_ratio,
    )


def remove_weak_strengths(
    vortex_strength: np.ndarray,
    threshold: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove weak strengths by smooth thresholding.

    A strength of magnitude ``|Γ|`` above the strength threshold ``Γ_τ`` is scaled
    by ``1 − (Γ_τ/|Γ|)²``; weaker strengths are removed. Returns the retained
    strengths and the removed part, both in m³/s.
    """
    strength = _vectors("vortex_strength", vortex_strength)
    threshold_array = np.broadcast_to(np.asarray(threshold, dtype=np.float64), (len(strength),))
    if np.any(~np.isfinite(threshold_array)) or np.any(threshold_array < 0.0):
        raise ValueError("the strength threshold must be finite and non-negative")
    if not np.any(threshold_array > 0.0):
        return strength.copy(), np.zeros_like(strength)
    magnitude = np.linalg.norm(strength, axis=1)
    scale = np.zeros_like(magnitude)
    active = magnitude > threshold_array
    scale[active] = 1.0 - (threshold_array[active] / magnitude[active]) ** 2
    retained = strength * scale[:, None]
    return retained, strength - retained


def redistribute_removed_strength(
    removed_vortex_strength: np.ndarray,
    retained_vortex_strength: np.ndarray,
    shape: tuple[int, int, int],
    *,
    labels: np.ndarray | None = None,
    wall_crossing_links: np.ndarray | None = None,
) -> np.ndarray:
    """Move the removed strength to retained face neighbours, never across a wall.

    Neighbours belong to the same fluid-connected component and are not joined by
    a wall-crossing link; the lattice edges are not periodic.
    """
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
        if wall_crossing_links is not None:
            edge_links = (
                wall_crossing_links if step == 1 else np.roll(wall_crossing_links, 1, axis=axis)
            )
            neighbours[index] &= (edge_links & (1 << axis)) == 0
        boundary_slice: list[slice | int] = [slice(None)] * 3
        boundary_slice[axis] = -1 if step == 1 else 0
        neighbours[index][tuple(boundary_slice)] = False

    count = np.sum(neighbours, axis=0).astype(np.float64)
    has_neighbour = count > 0.0
    share = np.zeros_like(removed)
    np.divide(removed, count[..., None], out=share, where=has_neighbour[..., None])
    for neighbour, (axis, step) in zip(neighbours, shifts, strict=True):
        contribution = np.where(neighbour[..., None], share, 0.0)
        output += np.roll(contribution, step, axis=axis)
    return output.reshape(-1, 3)


def _correct_fluid_component_moments(lattice, original, redistributed, labels, correct):
    """Apply the moment correction to each fluid-connected component.

    A component whose retained strengths cannot carry the correction keeps its
    limited strengths unchanged.
    """
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
        target = vortex_moments(lattice.positions[component], original[component])
        try:
            if len(retained) < 2:
                raise ValueError("too few retained nodes for the moment correction")
            result = correct(
                lattice.positions[retained],
                raw[retained],
                target,
                volumes=np.full(len(retained), lattice.particle_volume),
            )
            if not np.all(np.isfinite(result)):
                raise FloatingPointError("nonfinite component correction")
            corrected[retained] = result
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            # A removal threshold cannot justify moving a component's strength
            # into another fluid region. Keep its limited strengths instead.
            raw[component] = original[component]
            corrected[component] = original[component]
    return raw, corrected


@dataclass(frozen=True)
class LatticeTransferResult:
    """Particle state after one vorticity transfer on the lattice, with its diagnostics.

    The particles hold the transfer-lattice particles first, then the kept
    particles outside the remeshing region. The ``particle_set_*`` fields describe
    the removal of the weakest particles when the particle capacity is reached.
    """

    position: np.ndarray
    vortex_strength: np.ndarray
    particle_volume: np.ndarray
    core_radius: np.ndarray
    transfer_input_count: int = 0
    transfer_output_count: int = 0
    preserved_outer_count: int = 0
    coalesced_outer_count: int = 0
    coalesced_outer_input_indices: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    removed_node_count: int = 0
    removed_vortex_strength_l1: float = 0.0
    removed_vortex_strength_fraction: float = 0.0
    particle_set_removed_count: int = 0
    particle_set_removed_vortex_strength_fraction: float = 0.0
    particle_set_removed_velocity_bound: float = 0.0
    transfer_cfl: float = 0.0
    conservation_raw_mismatch: dict[str, float] = field(default_factory=dict)
    conservation_applied_correction: dict[str, float] = field(default_factory=dict)
    conservation_residual: dict[str, float] = field(default_factory=dict)
    conservation_applied_particle_strength_fraction: float = 0.0
    conservation_target_moments: VortexMoments | None = None
    conservation_raw_moments: VortexMoments | None = None
    conservation_reference_strength_l1: float = 0.0
    particle_set_conservation_raw_mismatch: dict[str, float] = field(default_factory=dict)
    particle_set_conservation_applied_correction: dict[str, float] = field(default_factory=dict)
    particle_set_conservation_residual: dict[str, float] = field(default_factory=dict)
    particle_set_conservation_applied_particle_strength_fraction: float = 0.0
    representation_residual_before_removal: float | None = None
    representation_residual_after_removal: float | None = None
    strength_ratio: float = 0.0
    excluded_target_vortex_strength_l1: float = 0.0

    @property
    def particle_count(self) -> int:
        """Return the number of particles of this result."""
        return len(self.position)


def transfer_on_lattice(
    positions: np.ndarray,
    vortex_strength: np.ndarray,
    lattice: TransferLattice,
    *,
    fvm_target_strength_at_node: ArrayFunction,
    particle_fluid_weight: ArrayFunction | None = None,
    particle_in_solid: ArrayFunction | None = None,
    strength_threshold: float = 0.0,
    surface_strength_threshold: float | None = None,
    core_radius_ratio: float = CORE_RADIUS_RATIO,
    max_strength_ratio: float = DEFAULT_MAX_STRENGTH_RATIO,
    maximum_particle_count: int | None = None,
    freestream_speed: float = 0.0,
    time_step_size: float = 0.0,
    compute_diagnostics: bool = True,
) -> LatticeTransferResult:
    """Transfer the FVM vorticity on the lattice and keep the particles outside it.

    The particles of the remeshing region are remeshed onto the lattice, the
    strengths are corrected toward the blended target, weak strengths are
    removed, and the moment correction restores the total strength and linear
    impulse of the limited strengths. Particles outside the remeshing region are
    kept. The function does not modify its inputs; the caller supplies the FVM
    target strengths through ``fvm_target_strength_at_node`` and may then
    replace the particle arrays with the returned state in one step.
    ``strength_threshold`` is the strength threshold ``h³ τ_transfer`` of the
    replacement region. ``surface_strength_threshold`` (``h³ τ_VPM``) applies at
    the transfer surface, and the threshold varies linearly with the blending
    weight between the two; ``None`` uses one threshold everywhere.
    """
    position_dtype = np.asarray(positions).dtype
    position = _vectors("positions", positions)
    strength = _vectors("vortex_strength", vortex_strength)
    if len(position) != len(strength):
        raise ValueError("positions and vortex_strength must have the same length")
    spacing = lattice.particle_spacing
    moment_correction = correct_vortex_moments
    ratio = _positive_finite("core_radius_ratio", core_radius_ratio)
    threshold = _nonnegative_finite("strength_threshold", strength_threshold)
    surface_threshold = (
        threshold
        if surface_strength_threshold is None
        else _nonnegative_finite("surface_strength_threshold", surface_strength_threshold)
    )
    if surface_threshold > threshold:
        raise ValueError("surface_strength_threshold must not exceed strength_threshold")
    if maximum_particle_count is not None and maximum_particle_count < 2:
        raise ValueError("maximum_particle_count must be at least two")

    count = len(position)
    input_fluid = _evaluate_weight("particle_fluid_weight", particle_fluid_weight, position)
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
            (position[:, 2] < lattice.transfer_region_bounds[4] - 1e-7)
            | (position[:, 2] > lattice.transfer_region_bounds[5] + 1e-7)
        )
    ):
        raise ValueError("a particle of the vorticity transfer lies outside the slip slab")
    in_remeshing_region = valid & np.all(
        (position >= lattice.remeshing_region_bounds[::2])
        & (position <= lattice.remeshing_region_bounds[1::2]),
        axis=1,
    )
    preserved_outer = valid & ~in_remeshing_region

    vpm_lattice_strength = scatter_m4_prime_to_lattice(
        position[in_remeshing_region],
        tapered_strength[in_remeshing_region],
        lattice,
        position_dtype=position_dtype,
    )
    if lattice.slip_slab:
        # The lattice holds the complete M4' support of each particle, so the
        # physical deposits fix the field of every slab image.
        vpm_lattice_strength = add_slab_images(
            vpm_lattice_strength.reshape(*lattice.shape, 3),
            lattice.origin[2],
            lattice.particle_spacing,
            lattice.transfer_region_bounds[4:6],
        ).reshape(-1, 3)
    vpm_lattice_strength *= lattice.fluid_weight[:, None]

    raw_target = _vectors(
        "fvm_target_strength_at_node",
        fvm_target_strength_at_node(lattice.positions),
    )
    if len(raw_target) != len(lattice.positions):
        raise ValueError("fvm_target_strength_at_node must return one vector per lattice node")
    excluded_target_l1 = float(
        np.linalg.norm(raw_target * (1.0 - lattice.fluid_weight)[:, None], axis=1).sum()
    )
    fvm_target = raw_target * (lattice.fluid_weight * lattice.mesh_weight)[:, None]

    base_core_radius = ratio * spacing
    # The fluid weight can be positive just inside the wall, but those nodes
    # cannot hold particles. Exclude their mismatch before the strength
    # correction spreads a solid-interior target into fluid strengths.
    output_weight = lattice.fluid_weight * ~lattice.solid_interior
    correction = correct_lattice_strength(
        vpm_lattice_strength,
        fvm_target,
        lattice.blending_weight,
        lattice.shape,
        spacing,
        core_radius=base_core_radius,
        max_strength_ratio=max_strength_ratio,
        output_weight=output_weight,
        compute_final_representation=compute_diagnostics,
        slip_slab_bounds=tuple(lattice.transfer_region_bounds[4:6]) if lattice.slip_slab else None,
        lattice_origin_z=float(lattice.origin[2]) if lattice.slip_slab else None,
    )
    comparison_weight = output_weight * lattice.mesh_weight
    residual_before_removal: float | None = None
    if compute_diagnostics:
        if correction.represented_strength is None:
            raise RuntimeError("the particle-represented vorticity was not evaluated")
        denominator = (
            float(np.linalg.norm(correction.blended_target * comparison_weight[:, None])) + 1.0e-30
        )
        residual_before_removal = (
            float(
                np.linalg.norm(
                    (correction.represented_strength - correction.blended_target)
                    * comparison_weight[:, None]
                )
            )
            / denominator
        )

    limited_strength = correction.limited_strength
    if lattice.wall_crossing_links is not None:
        limited_strength = limited_strength.copy()
        limited_strength[lattice.solid_interior] = 0.0
    limited_moments = vortex_moments(lattice.positions, limited_strength)
    magnitude_before = np.linalg.norm(limited_strength, axis=1)
    labels = (
        None
        if lattice.wall_crossing_links is None
        else connected_grid_components(
            magnitude_before.reshape(lattice.shape),
            np.zeros(lattice.shape, dtype=np.int8),
            lattice.wall_crossing_links,
        )
    )
    # The replacement region is regenerated from the FVM vorticity. At the transfer
    # surface the particles evolve under the VPM alone, so the removal threshold
    # falls to the GBD removal threshold there.
    local_threshold = surface_threshold + (threshold - surface_threshold) * lattice.blending_weight
    retained, removed = remove_weak_strengths(limited_strength, local_threshold)
    redistributed = redistribute_removed_strength(
        removed,
        retained,
        lattice.shape,
        labels=labels,
        wall_crossing_links=lattice.wall_crossing_links,
    )
    redistributed[lattice.solid_interior] = 0.0
    wall_corrected = None
    if labels is not None:
        redistributed, wall_corrected = _correct_fluid_component_moments(
            lattice, limited_strength, redistributed, labels, moment_correction
        )
    keep = np.linalg.norm(redistributed, axis=1) > 0.0
    removed_nodes = (magnitude_before > 0.0) & ~keep
    active_l1 = float(magnitude_before[magnitude_before > 0.0].sum())
    removed_l1 = float(magnitude_before[removed_nodes].sum())

    transfer_position = lattice.positions[keep]
    raw_transfer_strength = redistributed[keep]
    transfer_strength = raw_transfer_strength if wall_corrected is None else wall_corrected[keep]
    transfer_volume = np.full(len(transfer_position), lattice.particle_volume, dtype=np.float64)
    raw_moments = vortex_moments(transfer_position, raw_transfer_strength)
    conservation_raw_mismatch = vortex_moment_residual(limited_moments, raw_moments)
    conservation_target_moments = limited_moments
    conservation_raw_moments = raw_moments
    conservation_reference_strength_l1 = active_l1
    if wall_corrected is None and len(transfer_position) > 1:
        transfer_strength = moment_correction(
            transfer_position,
            raw_transfer_strength,
            limited_moments,
            volumes=transfer_volume,
        )
    corrected_moments = vortex_moments(transfer_position, transfer_strength)
    conservation_applied_correction = vortex_moment_residual(
        raw_moments,
        corrected_moments,
    )
    conservation_residual = vortex_moment_residual(limited_moments, corrected_moments)
    applied_particle_strength_fraction = float(
        np.linalg.norm(transfer_strength - raw_transfer_strength, axis=1).sum(dtype=np.float64)
        / (active_l1 + 1.0e-30)
    )

    residual_after_removal: float | None = None
    if compute_diagnostics:
        transferred_lattice_strength = np.zeros_like(correction.limited_strength)
        transferred_lattice_strength[keep] = transfer_strength
        represented_transfer = gaussian_representation(
            transferred_lattice_strength,
            lattice.shape,
            spacing,
            core_radius=base_core_radius,
            slip_slab_bounds=tuple(lattice.transfer_region_bounds[4:6])
            if lattice.slip_slab
            else None,
            lattice_origin_z=float(lattice.origin[2]) if lattice.slip_slab else None,
        )
        denominator = (
            float(np.linalg.norm(correction.blended_target * comparison_weight[:, None])) + 1.0e-30
        )
        residual_after_removal = (
            float(
                np.linalg.norm(
                    (represented_transfer - correction.blended_target) * comparison_weight[:, None]
                )
            )
            / denominator
        )

    transfer_radius = np.full(len(transfer_position), base_core_radius, dtype=np.float64)

    # The M4' support extends beyond the remeshing region, so a support node can
    # coincide with a kept particle outside it. Merge only those exact lattice
    # coincidences after the moment correction; this keeps the edge of the
    # remeshing region unchanged and avoids two particles at one node.
    preserved_for_append = preserved_outer.copy()
    coalesced_outer_count = 0
    coalesced_outer_input_indices = np.empty(0, dtype=np.int64)
    if len(transfer_position) and np.any(preserved_outer):
        outer_index = np.flatnonzero(preserved_outer)
        lattice_maximum = lattice.origin + spacing * (np.asarray(lattice.shape) - 1)
        position_tolerance = _ALIGNMENT_TOLERANCE_CELLS * spacing
        inside_support = np.ones(len(outer_index), dtype=bool)
        for axis in range(3):
            coordinate = position[outer_index, axis]
            inside_support &= coordinate >= lattice.origin[axis] - position_tolerance
            inside_support &= coordinate <= lattice_maximum[axis] + position_tolerance
        nearby_outer_index = outer_index[inside_support]
        relative_outer = (position[nearby_outer_index] - lattice.origin) / spacing
        nearest_outer = np.rint(relative_outer).astype(np.int64)
        shape_array = np.asarray(lattice.shape, dtype=np.int64)
        aligned_outer = (
            np.max(np.abs(relative_outer - nearest_outer), axis=1) <= _ALIGNMENT_TOLERANCE_CELLS
        )
        aligned_outer &= np.all(
            (nearest_outer >= 0) & (nearest_outer < shape_array),
            axis=1,
        )
        aligned_nearby = np.flatnonzero(aligned_outer)
        if len(aligned_nearby):
            nearby_lattice_index = nearest_outer[aligned_nearby]
            nearby_flat_index = np.ravel_multi_index(
                nearby_lattice_index.T,
                lattice.shape,
            )
            transfer_flat_index = np.flatnonzero(keep)
            transfer_row = np.searchsorted(transfer_flat_index, nearby_flat_index)
            valid_row = transfer_row < len(transfer_flat_index)
            collision = np.zeros(len(transfer_row), dtype=bool)
            collision[valid_row] = (
                transfer_flat_index[transfer_row[valid_row]] == nearby_flat_index[valid_row]
            )
            if np.any(collision):
                collided_outer_index = nearby_outer_index[aligned_nearby[collision]]
                coalesced_moments = vortex_moments(
                    transfer_position[transfer_row[collision]],
                    tapered_strength[collided_outer_index],
                )

                def with_coalesced(base: VortexMoments) -> VortexMoments:
                    return VortexMoments(
                        total_vortex_strength=(
                            base.total_vortex_strength + coalesced_moments.total_vortex_strength
                        ),
                        linear_impulse=base.linear_impulse + coalesced_moments.linear_impulse,
                        angular_impulse=(base.angular_impulse + coalesced_moments.angular_impulse),
                    )

                conservation_target_moments = with_coalesced(conservation_target_moments)
                conservation_raw_moments = with_coalesced(conservation_raw_moments)
                conservation_reference_strength_l1 += float(
                    np.linalg.norm(
                        tapered_strength[collided_outer_index],
                        axis=1,
                    ).sum(dtype=np.float64)
                )
                transfer_strength = transfer_strength.copy()
                np.add.at(
                    transfer_strength,
                    transfer_row[collision],
                    tapered_strength[collided_outer_index],
                )
                preserved_for_append[collided_outer_index] = False
                coalesced_outer_count = int(len(collided_outer_index))
                coalesced_outer_input_indices = collided_outer_index

    if np.any(preserved_for_append):
        output_position = np.vstack((transfer_position, position[preserved_for_append]))
        output_strength = np.vstack((transfer_strength, tapered_strength[preserved_for_append]))
        output_volume = np.concatenate(
            (
                transfer_volume,
                np.full(
                    np.count_nonzero(preserved_for_append),
                    lattice.particle_volume,
                    dtype=np.float64,
                ),
            )
        )
        output_radius = np.concatenate(
            (
                transfer_radius,
                np.full(
                    np.count_nonzero(preserved_for_append),
                    base_core_radius,
                    dtype=np.float64,
                ),
            )
        )
    else:
        output_position = transfer_position
        output_strength = transfer_strength
        output_volume = transfer_volume
        output_radius = transfer_radius

    particle_set_removed_count = 0
    particle_set_removed_fraction = 0.0
    particle_set_velocity_bound = 0.0
    particle_set_conservation_raw_mismatch: dict[str, float] = {}
    particle_set_conservation_applied_correction: dict[str, float] = {}
    particle_set_conservation_residual: dict[str, float] = {}
    particle_set_applied_particle_strength_fraction = 0.0
    final_transfer_count = len(transfer_position)
    final_outer_count = int(np.count_nonzero(preserved_for_append))
    if maximum_particle_count is not None and len(output_position) > maximum_particle_count:
        if lattice.wall_crossing_links is not None:
            raise RuntimeError(
                "The vorticity transfer next to a wall exceeds the particle capacity after "
                "weak-strength removal; increase the capacity rather than correcting the "
                "moments across the wall"
            )
        target_count = int(maximum_particle_count)
        combined_moments = vortex_moments(output_position, output_strength)
        combined_magnitude = np.linalg.norm(output_strength, axis=1)
        transfer_count = len(transfer_position)
        outer_count = len(output_position) - transfer_count
        if outer_count < target_count:
            transfer_budget = target_count - outer_count
            transfer_keep = np.argpartition(combined_magnitude[:transfer_count], -transfer_budget)[
                -transfer_budget:
            ]
            keep_indices = np.concatenate(
                (
                    transfer_keep,
                    np.arange(transfer_count, len(output_position), dtype=np.int64),
                )
            )
        elif outer_count == target_count:
            keep_indices = np.arange(transfer_count, len(output_position), dtype=np.int64)
        else:
            outer_keep = np.argpartition(combined_magnitude[transfer_count:], -target_count)[
                -target_count:
            ]
            keep_indices = outer_keep + transfer_count
        keep_indices = np.sort(keep_indices)
        particle_set_keep = np.zeros(len(output_position), dtype=bool)
        particle_set_keep[keep_indices] = True
        discarded_l1 = float(combined_magnitude[~particle_set_keep].sum())
        delta = np.maximum(
            np.maximum(
                lattice.transfer_region_bounds[::2] - output_position,
                output_position - lattice.transfer_region_bounds[1::2],
            ),
            0.0,
        )
        distance_squared = np.einsum("ij,ij->i", delta, delta)
        particle_set_velocity_bound = float(
            np.sum(
                combined_magnitude[~particle_set_keep]
                / (
                    4.0
                    * np.pi
                    * np.maximum(
                        distance_squared[~particle_set_keep]
                        + output_radius[~particle_set_keep] ** 2,
                        1.0e-30,
                    )
                )
            )
        )
        particle_set_removed_fraction = discarded_l1 / (float(combined_magnitude.sum()) + 1.0e-30)
        particle_set_removed_count = int(np.count_nonzero(~particle_set_keep))
        final_transfer_count = int(np.count_nonzero(keep_indices < transfer_count))
        final_outer_count = int(len(keep_indices) - final_transfer_count)
        output_position = output_position[keep_indices]
        raw_particle_set_strength = output_strength[keep_indices]
        output_volume = output_volume[keep_indices]
        output_radius = output_radius[keep_indices]
        raw_particle_set_moments = vortex_moments(output_position, raw_particle_set_strength)
        particle_set_conservation_raw_mismatch = vortex_moment_residual(
            combined_moments,
            raw_particle_set_moments,
        )
        output_strength = moment_correction(
            output_position,
            raw_particle_set_strength,
            combined_moments,
            volumes=output_volume,
        )
        corrected_particle_set_moments = vortex_moments(output_position, output_strength)
        particle_set_conservation_applied_correction = vortex_moment_residual(
            raw_particle_set_moments,
            corrected_particle_set_moments,
        )
        particle_set_conservation_residual = vortex_moment_residual(
            combined_moments,
            corrected_particle_set_moments,
        )
        particle_set_applied_particle_strength_fraction = float(
            np.linalg.norm(output_strength - raw_particle_set_strength, axis=1).sum(
                dtype=np.float64
            )
            / (float(combined_magnitude.sum(dtype=np.float64)) + 1.0e-30)
        )

    speed = _nonnegative_finite("freestream_speed", abs(float(freestream_speed)))
    time_step = _nonnegative_finite("time_step_size", abs(float(time_step_size)))
    transfer_cfl = speed * time_step / (lattice.remeshing_extension_length + 1.0e-30)
    return LatticeTransferResult(
        position=output_position,
        vortex_strength=output_strength,
        particle_volume=output_volume,
        core_radius=output_radius,
        transfer_input_count=int(np.count_nonzero(in_remeshing_region)),
        transfer_output_count=final_transfer_count,
        preserved_outer_count=final_outer_count,
        coalesced_outer_count=coalesced_outer_count,
        coalesced_outer_input_indices=coalesced_outer_input_indices,
        removed_node_count=int(np.count_nonzero(removed_nodes)),
        removed_vortex_strength_l1=removed_l1,
        removed_vortex_strength_fraction=removed_l1 / (active_l1 + 1.0e-30),
        particle_set_removed_count=particle_set_removed_count,
        particle_set_removed_vortex_strength_fraction=particle_set_removed_fraction,
        particle_set_removed_velocity_bound=particle_set_velocity_bound,
        transfer_cfl=float(transfer_cfl),
        conservation_raw_mismatch=conservation_raw_mismatch,
        conservation_applied_correction=conservation_applied_correction,
        conservation_residual=conservation_residual,
        conservation_applied_particle_strength_fraction=applied_particle_strength_fraction,
        conservation_target_moments=conservation_target_moments,
        conservation_raw_moments=conservation_raw_moments,
        conservation_reference_strength_l1=conservation_reference_strength_l1,
        particle_set_conservation_raw_mismatch=particle_set_conservation_raw_mismatch,
        particle_set_conservation_applied_correction=particle_set_conservation_applied_correction,
        particle_set_conservation_residual=particle_set_conservation_residual,
        particle_set_conservation_applied_particle_strength_fraction=(
            particle_set_applied_particle_strength_fraction
        ),
        representation_residual_before_removal=residual_before_removal,
        representation_residual_after_removal=residual_after_removal,
        strength_ratio=correction.strength_ratio,
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


def vortex_moment_residual(
    target: VortexMoments,
    actual: VortexMoments,
) -> dict[str, float]:
    """Return the Euclidean norms of the differences of the three vortex moments.

    Keys are ``total_vortex_strength`` (m³/s), ``linear_impulse`` (m⁴/s) and
    ``angular_impulse`` (m⁵/s).
    """
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
        raise ValueError("transfer_region_bounds must contain six finite increasing bounds")
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
    "DEFAULT_MAX_STRENGTH_RATIO",
    "LatticeTransferResult",
    "M4_PRIME_SUPPORT_CELLS",
    "StrengthCorrection",
    "TransferLattice",
    "VortexMoments",
    "blending_weight",
    "build_transfer_lattice",
    "correct_lattice_strength",
    "correct_vortex_moments",
    "fvm_target_strength",
    "gaussian_representation",
    "m4_prime",
    "redistribute_removed_strength",
    "remove_weak_strengths",
    "scatter_m4_prime_to_lattice",
    "transfer_on_lattice",
    "vortex_moment_residual",
    "vortex_moments",
]
