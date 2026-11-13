"""Vorticity transfer on the fixed transfer lattice, compatible with GBD diffusion.

The particles of the remeshing region are remeshed onto the transfer lattice,
blended with the FVM circulation of each lattice cube, thresholded and returned
as transfer-lattice particles; particles outside the remeshing region are kept.
The operators return particle arrays and do not modify a solver. Transfer-lattice
particles sit on the lattice nodes and carry the base core radius.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from openonda.core._numba import cacheable_njit as njit
from openonda.core.grid_connectivity import connected_grid_components
from openonda.core.slip_slab_images import add_slab_images
from openonda.vpm.numerics.m4_kernel import m4_prime, m4_prime_scalar

ArrayFunction = Callable[[np.ndarray], np.ndarray]

CORE_RADIUS_RATIO = 1.0
M4_PRIME_SUPPORT_CELLS = 2.0
WALL_BUDGET_SPACINGS = 3.0
"""Depth, in particle spacings, of the near-wall region of the circulation budget."""
_SHELL_SEARCH_CELLS = 2
_SHELL_TIE_TOLERANCE_CELLS = 1.0e-9
_ALIGNMENT_TOLERANCE_CELLS = 1.0e-5
MOMENT_MATRIX_CONDITION_LIMIT = 1.0e12
"""Largest condition number of the six-by-six moment-correction matrix; beyond it the particles
cannot carry the correction of the total strength and the linear impulse."""


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
            weight_x = m4_prime_scalar(relative_position[particle, 0] - index_x)
            for offset_y in range(4):
                index_y = lower_stencil_index[particle, 1] + offset_y
                weight_y = m4_prime_scalar(relative_position[particle, 1] - index_y)
                for offset_z in range(4):
                    index_z = lower_stencil_index[particle, 2] + offset_z
                    weight_z = m4_prime_scalar(relative_position[particle, 2] - index_z)
                    weight = weight_x * weight_y * weight_z
                    result[index_x, index_y, index_z, 0] += weight * vortex_strength[particle, 0]
                    result[index_x, index_y, index_z, 1] += weight * vortex_strength[particle, 1]
                    result[index_x, index_y, index_z, 2] += weight * vortex_strength[particle, 2]
    return result


def blending_weight(
    positions: np.ndarray,
    transfer_region_bounds: np.ndarray | list[float] | tuple[float, ...],
    blend_start_depth: float,
    blend_end_depth: float,
    *,
    slip_slab: bool = False,
    band_weight: float = 0.0,
    band_depth: float = 0.0,
) -> np.ndarray:
    """Return the blending weight ``η`` at each world-frame point.

    Positions have shape (N,3), and transfer_region_bounds contains
    (xmin,xmax,ymin,ymax,zmin,zmax), all in m. With ``d`` the depth below the
    transfer surface, the blending weight is zero in the zero-weight layer
    ``d ≤ δ_0`` (``blend_start_depth``) and rises with a cosine to one at
    ``d = δ_1`` (``blend_end_depth``). A zero ``δ_1`` gives a sharp mask, one
    everywhere inside the transfer surface.

    A positive ``band_weight`` ``w`` ties the particles of an outer band to the FVM
    vorticity: ``η`` rises with a cosine from zero at ``d = δ_0`` to ``w`` at
    ``d = δ_1``, holds ``w`` down to ``d = band_depth`` and rises with a cosine
    from ``w`` to one over the next ``δ_1 − δ_0``.

    With slip_slab=True, z faces keep the full blending weight through the closed
    physical span but points outside it have zero blending weight. The returned
    independent float64 array has shape (N,) and values in [0,1]. Invalid bounds,
    vectors, depths or weights raise ValueError; inputs are unchanged.
    """
    position = _vectors("positions", positions)
    bounds = _bounds(transfer_region_bounds)
    start_depth = _nonnegative_finite("blend_start_depth", blend_start_depth)
    end_depth = _nonnegative_finite("blend_end_depth", blend_end_depth)
    if end_depth > 0.0 and start_depth >= end_depth:
        raise ValueError("blend_start_depth must be smaller than blend_end_depth")
    weight = float(band_weight)
    if not 0.0 <= weight < 1.0:
        raise ValueError("band_weight must lie in [0, 1)")
    plateau_end = _nonnegative_finite("band_depth", band_depth)
    if weight > 0.0 and (end_depth == 0.0 or plateau_end < end_depth):
        raise ValueError("a band weight needs blending and band_depth >= blend_end_depth")

    no_face = np.full(len(position), np.inf)
    face_distance = np.minimum.reduce(
        [
            position[:, 0] - bounds[0],
            bounds[1] - position[:, 0],
            position[:, 1] - bounds[2],
            bounds[3] - position[:, 1],
            no_face if slip_slab else position[:, 2] - bounds[4],
            no_face if slip_slab else bounds[5] - position[:, 2],
        ]
    )
    if end_depth == 0.0:
        blend_weight = (face_distance > 0.0).astype(np.float64)
    else:
        width = end_depth - start_depth

        def cosine_ramp(lower):
            phase = np.clip((face_distance - lower) / width, 0.0, 1.0)
            return 0.5 * (1.0 - np.cos(np.pi * phase))

        if weight == 0.0:
            blend_weight = cosine_ramp(start_depth)
        else:
            blend_weight = weight * cosine_ramp(start_depth) + (1.0 - weight) * cosine_ramp(
                plateau_end
            )
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
    if not np.isfinite(condition) or condition > MOMENT_MATRIX_CONDITION_LIMIT:
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
    With a solid boundary, ``wall_distance`` holds the signed node distance from the
    wall (m, positive in the fluid) and ``shell_destination`` the moves of the
    strength of the shell nodes ``-h < d < 0`` onto fluid nodes, as source rows,
    receiving rows and shares.
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
    wall_distance: np.ndarray | None = None
    shell_destination: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    @property
    def shell(self) -> np.ndarray | None:
        """Return the mask of the shell nodes ``-h < d < 0``, or None without a wall."""
        if self.wall_distance is None:
            return None
        return self.solid_interior & (self.wall_distance > -self.particle_spacing)

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
    band_weight: float = 0.0,
    band_depth: float = 0.0,
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
    band_weight, band_depth : float, optional
        Blending weight of the outer band and the depth in m where it ends; see
        :func:`blending_weight`.
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
            (positions[:, 2] < bounds[4] - 1.0e-8 * spacing)
            | (positions[:, 2] > bounds[5] + 1.0e-8 * spacing)
        ] = 0.0
    solid_interior = (
        np.zeros(count, dtype=bool)
        if interior_at_node is None
        else np.asarray(interior_at_node(positions), dtype=bool).reshape(-1)
    )
    if solid_interior.shape != (count,):
        raise ValueError(f"interior_at_node returned {solid_interior.shape}, expected ({count},)")
    blend_weight = (
        blending_weight(
            positions,
            bounds,
            start_depth,
            end_depth,
            slip_slab=slip_slab,
            band_weight=band_weight,
            band_depth=band_depth,
        )
        * mesh_weight
    )
    wall_crossing_links = None
    wall_distance = None
    shell_destination = None
    if solid_boundary is not None:
        wall_crossing_links = _wall_crossing_links(positions, shape, solid_interior, solid_boundary)
        wall_distance = np.asarray(
            solid_boundary.signed_distance(positions), dtype=np.float64
        ).reshape(-1)
        receives = ~solid_interior & (fluid_weight > 0.0) & (mesh_weight > 0.0)
        shell_destination = _shell_destinations(
            positions,
            shape,
            spacing,
            solid_interior & (wall_distance > -spacing),
            receives,
            solid_boundary,
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
        wall_distance=wall_distance,
        shell_destination=shell_destination,
    )


def _shell_destinations(positions, shape, spacing, shell, receives, boundary):
    """Return the moves of shell strength onto the receiving nodes nearest the wall.

    Each shell node sends its strength to the nearest receiving node, in equal
    shares to nodes at the same distance, so the move keeps the symmetries of the
    wall and the lattice (a closest-wall-point rule picks one face at an edge).
    The segment from the shell node to a receiving node leaves the solid without
    entering it again, so strength never moves around a thin trailing edge into
    another fluid region. Returns the source rows, the
    receiving rows and the shares; a shell node without a receiving node within
    ``_SHELL_SEARCH_CELLS`` cells is absent.
    """
    rows = np.flatnonzero(shell)
    empty = np.empty(0, dtype=np.int64)
    if not len(rows):
        return empty, empty, np.empty(0, dtype=np.float64)
    reach = np.arange(-_SHELL_SEARCH_CELLS, _SHELL_SEARCH_CELLS + 1)
    offsets = np.stack(np.meshgrid(reach, reach, reach, indexing="ij"), axis=-1).reshape(-1, 3)
    index = np.stack(np.unravel_index(rows, shape), axis=1)
    candidate = index[:, None, :] + offsets[None, :, :]
    inside = np.all((candidate >= 0) & (candidate < np.asarray(shape)), axis=2)
    clipped = np.clip(candidate, 0, np.asarray(shape) - 1)
    flat = (clipped[..., 0] * shape[1] + clipped[..., 1]) * shape[2] + clipped[..., 2]
    valid = inside & receives[flat]
    distance = np.linalg.norm(offsets, axis=1)[None, :] * spacing * np.ones((len(rows), 1))
    start = positions[rows]
    sources, receivers, shares = [], [], []
    pending = np.arange(len(rows))
    while len(pending):
        cost = np.where(valid[pending], distance[pending], np.inf)
        nearest = cost.min(axis=1)
        reachable = np.isfinite(nearest)
        pending, cost, nearest = pending[reachable], cost[reachable], nearest[reachable]
        if not len(pending):
            break
        tied = cost <= nearest[:, None] + _SHELL_TIE_TOLERANCE_CELLS * spacing
        row, column = np.nonzero(tied)
        node = pending[row]
        crosses = boundary.segments_cross_wall(start[node], positions[flat[node, column]])
        valid[node[crosses], column[crosses]] = False
        accepted = np.zeros((len(pending), offsets.shape[0]), dtype=bool)
        accepted[row[~crosses], column[~crosses]] = True
        count = accepted.sum(axis=1)
        done = count > 0
        row, column = np.nonzero(accepted[done])
        node = pending[done][row]
        sources.append(rows[node])
        receivers.append(flat[node, column])
        shares.append(1.0 / count[done][row])
        pending = pending[~done]
    if not sources:
        return empty, empty, np.empty(0, dtype=np.float64)
    return np.concatenate(sources), np.concatenate(receivers), np.concatenate(shares)


def _move_shell_strength(lattice, strength):
    """Move the shell strengths onto their receiving nodes and clear the solid nodes.

    Returns the moved strengths and the L1 norms of the moved and the dropped
    shell strength.
    """
    moved = strength.copy()
    source, receiver, share = lattice.shell_destination
    magnitude = np.linalg.norm(strength[lattice.shell], axis=1)
    has_receiver = np.zeros(len(strength), dtype=bool)
    has_receiver[source] = True
    np.add.at(moved, receiver, share[:, None] * strength[source])
    moved[lattice.solid_interior] = 0.0
    kept = has_receiver[lattice.shell]
    return moved, float(magnitude[kept].sum()), float(magnitude[~kept].sum())


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


def blend_lattice_strength(
    vpm_vortex_strength: np.ndarray,
    fvm_target_vortex_strength: np.ndarray,
    blending_weight: np.ndarray,
    *,
    output_weight: np.ndarray | None = None,
    physical: np.ndarray | None = None,
) -> np.ndarray:
    """Return the lattice strengths ``w (Γ^VPM + η (Γ^FVM − Γ^VPM))`` in m³/s.

    ``Γ^FVM`` is the FVM circulation of each lattice cube and ``Γ^VPM`` the
    M4'-remeshed particle strength of the node. Where the blending weight ``η``
    is one a node holds the FVM circulation of its cube, so the transfer keeps
    the FVM circulation of every part of the replacement region, next to a wall
    as well; where ``η`` is zero it keeps its particle strength. ``output_weight``
    ``w`` scales the strengths, and nodes outside ``physical`` (slip-slab images)
    hold none.
    """
    vpm_strength = _vectors("vpm_vortex_strength", vpm_vortex_strength)
    fvm_strength = _vectors("fvm_target_vortex_strength", fvm_target_vortex_strength)
    blend_weight = np.asarray(blending_weight, dtype=np.float64).reshape(-1)
    if vpm_strength.shape != fvm_strength.shape or blend_weight.shape != (len(vpm_strength),):
        raise ValueError("the remeshed, target and blending arrays must share one lattice shape")
    # v + η (F − v) leaves a node whose strength already equals F unchanged; where η = 1
    # the node takes F itself, so it holds the FVM circulation exactly.
    strength = vpm_strength + blend_weight[:, None] * (fvm_strength - vpm_strength)
    replaced = blend_weight >= 1.0
    strength[replaced] = fvm_strength[replaced]
    if output_weight is not None:
        weight = np.asarray(output_weight, dtype=np.float64).reshape(-1)
        if weight.shape != (len(vpm_strength),):
            raise ValueError("output_weight must share the transfer lattice shape")
        strength *= weight[:, None]
    if physical is not None:
        strength[~np.asarray(physical, dtype=bool).reshape(-1)] = 0.0
    return strength


def _slab_physical_nodes(lattice) -> np.ndarray | None:
    """Return the nodes inside the slip slab, or ``None`` without a slab."""
    if not lattice.slip_slab:
        return None
    spacing = lattice.particle_spacing
    z_nodes = lattice.origin[2] + spacing * np.arange(lattice.shape[2])
    z_min, z_max = lattice.transfer_region_bounds[4:6]
    inside = (z_nodes >= z_min - 1e-8 * spacing) & (z_nodes <= z_max + 1e-8 * spacing)
    return np.broadcast_to(inside, lattice.shape).reshape(-1)


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

    A component with no strength above the removal threshold is removed as a whole;
    the returned moments hold its strength, which the removed totals then include.
    A component whose retained strengths cannot carry the correction keeps its
    node strengths from before the removal; its count and strength L1 norm are returned.
    """
    fallback_count, fallback_l1 = 0, 0.0
    dropped = VortexMoments(np.zeros(3), np.zeros(3), np.zeros(3))
    raw = redistributed.copy()
    corrected = redistributed.copy()
    flat_labels = labels.reshape(-1)
    rows = np.flatnonzero(flat_labels >= 0)
    if not len(rows):
        return raw, corrected, fallback_count, fallback_l1, dropped
    if np.any(flat_labels[rows] != flat_labels[rows[0]]):
        rows = rows[np.argsort(flat_labels[rows], kind="stable")]
    boundaries = np.r_[0, np.flatnonzero(np.diff(flat_labels[rows])) + 1, len(rows)]
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        component = rows[start:stop]
        retained = component[np.linalg.norm(raw[component], axis=1) > 0.0]
        target = vortex_moments(lattice.positions[component], original[component])
        if not len(retained):
            raw[component] = 0.0
            corrected[component] = 0.0
            dropped = VortexMoments(
                dropped.total_vortex_strength + target.total_vortex_strength,
                dropped.linear_impulse + target.linear_impulse,
                dropped.angular_impulse + target.angular_impulse,
            )
            continue
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
            # into another fluid region. Keep its strengths from before the removal.
            raw[component] = original[component]
            corrected[component] = original[component]
            fallback_count += 1
            fallback_l1 += float(np.linalg.norm(original[component], axis=1).sum())
    return raw, corrected, fallback_count, fallback_l1, dropped


@dataclass(frozen=True)
class LatticeTransferResult:
    """Particle state after one vorticity transfer on the lattice, with its diagnostics.

    The particles hold the transfer-lattice particles first, then the kept
    particles outside the remeshing region.
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
    preserved_outer_input_indices: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    removed_node_count: int = 0
    removed_vortex_strength_l1: float = 0.0
    removed_vortex_strength_fraction: float = 0.0
    dropped_particle_vortex_strength_fraction: float = 0.0
    transfer_cfl: float = 0.0
    conservation_raw_mismatch: dict[str, float] = field(default_factory=dict)
    conservation_applied_correction: dict[str, float] = field(default_factory=dict)
    conservation_residual: dict[str, float] = field(default_factory=dict)
    conservation_applied_particle_strength_fraction: float = 0.0
    conservation_target_moments: VortexMoments | None = None
    conservation_raw_moments: VortexMoments | None = None
    conservation_reference_strength_l1: float = 0.0
    excluded_target_vortex_strength_l1: float = 0.0
    fvm_budget: FVMBudget | None = None
    solid_particle_vortex_strength_l1: float = 0.0
    tapered_vortex_strength_l1: float = 0.0
    moment_fallback_component_count: int = 0
    moment_fallback_vortex_strength_l1: float = 0.0

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
    freestream_speed: float = 0.0,
    time_step_size: float = 0.0,
) -> LatticeTransferResult:
    """Transfer the FVM vorticity on the lattice and keep the particles outside it.

    The particles of the remeshing region are remeshed onto the lattice and
    blended with the FVM circulation of each lattice cube
    (:func:`blend_lattice_strength`): the replacement region holds the FVM
    circulation exactly. Weak strengths are then removed, and the moment
    correction restores the total strength and linear impulse that the removal
    changed. Particles outside the remeshing region are
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
    shell = lattice.shell
    replacement = lattice.blending_weight >= 1.0
    if shell is None:
        excluded_target_l1 = float(
            np.linalg.norm(raw_target * (1.0 - lattice.fluid_weight)[:, None], axis=1).sum()
        )
        fvm_target = raw_target * (lattice.fluid_weight * lattice.mesh_weight)[:, None]
        # The fluid weight can be positive just inside the wall, but those nodes
        # cannot hold particles.
        output_weight = lattice.fluid_weight * ~lattice.solid_interior
        budget_nodes = replacement & (output_weight > 0.0) & (lattice.mesh_weight > 0.0)
    else:
        # The Stokes integral over a wall-cut lattice cube already holds exactly the
        # vorticity of its fluid part, so the whole target is kept. Part of the wall
        # sheet falls on the shell -h < d < 0: it is moved onto a fluid node below,
        # instead of being lost at every transfer.
        # Nodes outside the FVM mesh have no FVM target and zero blending weight; they
        # keep the strength of the particles that left the mesh.
        output = (lattice.fluid_weight > 0.0) & (~lattice.solid_interior | shell)
        target_nodes = output & (lattice.mesh_weight > 0.0)
        excluded_target_l1 = float(np.linalg.norm(raw_target[~target_nodes], axis=1).sum())
        fvm_target = raw_target * (lattice.mesh_weight * target_nodes)[:, None]
        output_weight = output.astype(np.float64)
        budget_nodes = replacement & target_nodes

    base_core_radius = ratio * spacing
    node_strength = blend_lattice_strength(
        vpm_lattice_strength,
        fvm_target,
        lattice.blending_weight,
        output_weight=output_weight,
        physical=_slab_physical_nodes(lattice),
    )
    kept_l1, dropped_l1 = _zero_weight_strength_l1(lattice, vpm_lattice_strength, node_strength)
    relocated_shell_l1 = 0.0
    dropped_shell_l1 = 0.0
    if shell is not None:
        node_strength, relocated_shell_l1, dropped_shell_l1 = _move_shell_strength(
            lattice, node_strength
        )
    elif lattice.wall_crossing_links is not None:
        node_strength = node_strength.copy()
        node_strength[lattice.solid_interior] = 0.0
    node_moments = vortex_moments(lattice.positions, node_strength)
    magnitude_before = np.linalg.norm(node_strength, axis=1)
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
    retained, removed = remove_weak_strengths(node_strength, local_threshold)
    redistributed = redistribute_removed_strength(
        removed,
        retained,
        lattice.shape,
        labels=labels,
        wall_crossing_links=lattice.wall_crossing_links,
    )
    redistributed[lattice.solid_interior] = 0.0
    wall_corrected = None
    fallback_count, fallback_l1 = 0, 0.0
    # The moments the particles must hold: those of the node strengths, less the
    # components that the removal takes as a whole.
    held_moments = node_moments
    if labels is not None:
        redistributed, wall_corrected, fallback_count, fallback_l1, dropped = (
            _correct_fluid_component_moments(
                lattice, node_strength, redistributed, labels, moment_correction
            )
        )
        held_moments = VortexMoments(
            node_moments.total_vortex_strength - dropped.total_vortex_strength,
            node_moments.linear_impulse - dropped.linear_impulse,
            node_moments.angular_impulse - dropped.angular_impulse,
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
    conservation_raw_mismatch = vortex_moment_residual(held_moments, raw_moments)
    conservation_target_moments = held_moments
    conservation_raw_moments = raw_moments
    conservation_reference_strength_l1 = active_l1
    if wall_corrected is None and len(transfer_position) > 1:
        transfer_strength = moment_correction(
            transfer_position,
            raw_transfer_strength,
            held_moments,
            volumes=transfer_volume,
        )
    corrected_moments = vortex_moments(transfer_position, transfer_strength)
    conservation_applied_correction = vortex_moment_residual(
        raw_moments,
        corrected_moments,
    )
    conservation_residual = vortex_moment_residual(held_moments, corrected_moments)
    applied_particle_strength_fraction = float(
        np.linalg.norm(transfer_strength - raw_transfer_strength, axis=1).sum(dtype=np.float64)
        / (active_l1 + 1.0e-30)
    )

    budget = _fvm_budget(
        lattice,
        fvm_target,
        keep,
        transfer_strength,
        budget_nodes,
        relocated_shell_l1=relocated_shell_l1,
        dropped_shell_l1=dropped_shell_l1,
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

    final_transfer_count = len(transfer_position)
    final_outer_count = int(np.count_nonzero(preserved_for_append))
    outer_input_indices = np.flatnonzero(preserved_for_append)

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
        preserved_outer_input_indices=outer_input_indices,
        removed_node_count=int(np.count_nonzero(removed_nodes)),
        removed_vortex_strength_l1=removed_l1,
        removed_vortex_strength_fraction=removed_l1 / (active_l1 + 1.0e-30),
        dropped_particle_vortex_strength_fraction=dropped_l1 / (kept_l1 + 1.0e-30),
        transfer_cfl=float(transfer_cfl),
        conservation_raw_mismatch=conservation_raw_mismatch,
        conservation_applied_correction=conservation_applied_correction,
        conservation_residual=conservation_residual,
        conservation_applied_particle_strength_fraction=applied_particle_strength_fraction,
        conservation_target_moments=conservation_target_moments,
        conservation_raw_moments=conservation_raw_moments,
        conservation_reference_strength_l1=conservation_reference_strength_l1,
        excluded_target_vortex_strength_l1=excluded_target_l1,
        fvm_budget=budget,
        solid_particle_vortex_strength_l1=float(
            np.linalg.norm(strength[deep_solid], axis=1).sum(dtype=np.float64)
        ),
        tapered_vortex_strength_l1=float(
            np.linalg.norm(strength[valid] - tapered_strength[valid], axis=1).sum(dtype=np.float64)
        ),
        moment_fallback_component_count=fallback_count,
        moment_fallback_vortex_strength_l1=fallback_l1,
    )


@dataclass(frozen=True)
class FVMBudget:
    """Circulation budget of one transfer against the raw FVM target.

    The replacement region holds the lattice nodes with blending weight one. Its
    FVM total strength (m³/s) and linear impulse (m⁴/s) come from the raw Stokes
    target before the strength limit and the removal threshold, with the shell
    strengths moved like the particle strengths; the emitted sums are those of the
    transfer-lattice particles on the same nodes. ``relative_error`` is the norm of
    the total-strength difference over the L1 norm of the FVM target there. The
    near-wall sums cover the nodes within ``WALL_BUDGET_SPACINGS`` spacings of a
    wall; ``wall_circulation_ratio`` projects the emitted total on the FVM total
    (one when there is no wall or no net near-wall circulation), and
    ``wall_fvm_vortex_strength_l1`` is the FVM L1 norm there.
    """

    fvm_total_vortex_strength: np.ndarray
    emitted_total_vortex_strength: np.ndarray
    fvm_linear_impulse: np.ndarray
    emitted_linear_impulse: np.ndarray
    fvm_vortex_strength_l1: float
    relative_error: float
    wall_fvm_total_vortex_strength: np.ndarray
    wall_emitted_total_vortex_strength: np.ndarray
    wall_circulation_ratio: float
    wall_fvm_vortex_strength_l1: float = 0.0
    relocated_shell_vortex_strength_l1: float = 0.0
    dropped_shell_vortex_strength_l1: float = 0.0


WALL_NET_CIRCULATION_FRACTION = 0.05
"""Net near-wall FVM circulation, as a fraction of its L1 norm, above which the wall ratio is checked.

It is 16 % on the NACA 4412 wing, 9.5 % on a thin lifting ellipse and at most about 2 % on the
shedding cylinder. Below it the ratio of two small totals is dominated by the shift of the
weak-strength removal, about 1e-3 of the near-wall L1 norm."""


def check_fvm_budget(
    budget: FVMBudget,
    *,
    relative_limit: float | None,
    wall_ratio_bounds: tuple[float, float] | None,
) -> None:
    """Raise RuntimeError when a transfer loses the FVM circulation.

    ``relative_limit`` bounds the total-strength error of the replacement region,
    as a fraction of the FVM L1 norm there. ``wall_ratio_bounds`` bounds the
    near-wall circulation ratio wherever the net near-wall FVM circulation exceeds
    ``WALL_NET_CIRCULATION_FRACTION`` of its L1 norm: the bound circulation of a
    lifting body is far smaller than the L1 norm of its wall sheets, so the first
    check alone cannot see it lost. ``None`` disables a check.
    """
    if relative_limit is not None and budget.relative_error > relative_limit:
        raise RuntimeError(
            "the vorticity transfer does not hold the FVM circulation of the replacement "
            f"region: total-strength error {budget.relative_error:.3e} of the FVM L1 norm, "
            f"limit {relative_limit:.3e}; near-wall circulation ratio "
            f"{budget.wall_circulation_ratio:.4f}"
        )
    net = float(np.linalg.norm(budget.wall_fvm_total_vortex_strength))
    if (
        wall_ratio_bounds is not None
        and net > WALL_NET_CIRCULATION_FRACTION * budget.wall_fvm_vortex_strength_l1
        and not wall_ratio_bounds[0] <= budget.wall_circulation_ratio <= wall_ratio_bounds[1]
    ):
        raise RuntimeError(
            "the vorticity transfer does not hold the FVM circulation next to the wall: "
            f"near-wall circulation ratio {budget.wall_circulation_ratio:.4f}, bounds "
            f"[{wall_ratio_bounds[0]:g}, {wall_ratio_bounds[1]:g}]; net near-wall FVM "
            f"circulation {net:.3e} of L1 {budget.wall_fvm_vortex_strength_l1:.3e} m^3/s"
        )


DROPPED_PARTICLE_STRENGTH_LIMIT = 1.0e-12
"""Largest L1 fraction of the particle strength of the zero-weight nodes that a transfer may lose.

Where the blending weight is zero, away from walls, a node keeps its M4'-remeshed particle strength
bit for bit, so a correct transfer loses none of it; the limit leaves room for roundoff only. The
check is on the L1 norm because the strength of a wake leaving the FVM mesh sums to about zero: a
transfer that deletes it changes no total strength or impulse."""


def _zero_weight_strength_l1(lattice, vpm_strength, node_strength) -> tuple[float, float]:
    """Return the particle strength L1 of the zero-weight fluid nodes and the L1 the blend lost there."""
    nodes = (lattice.blending_weight == 0.0) & (lattice.fluid_weight >= 1.0)
    nodes &= ~lattice.solid_interior
    physical = _slab_physical_nodes(lattice)
    if physical is not None:
        nodes &= physical
    kept = float(np.linalg.norm(vpm_strength[nodes], axis=1).sum())
    dropped = float(np.linalg.norm(vpm_strength[nodes] - node_strength[nodes], axis=1).sum())
    return kept, dropped


def check_dropped_particle_strength(fraction: float) -> None:
    """Raise RuntimeError when a transfer loses particle strength where it must keep it.

    ``fraction`` is the L1 norm lost on the zero-weight nodes over their particle
    strength L1 norm; above ``DROPPED_PARTICLE_STRENGTH_LIMIT`` the transfer stops
    before it changes the particles.
    """
    if fraction > DROPPED_PARTICLE_STRENGTH_LIMIT:
        raise RuntimeError(
            "the vorticity transfer loses particle strength where the blending weight is zero: "
            f"{fraction:.3e} of its L1 norm there, limit {DROPPED_PARTICLE_STRENGTH_LIMIT:.0e}"
        )


def _fvm_budget(
    lattice,
    fvm_target,
    keep,
    transfer_strength,
    budget_nodes,
    *,
    relocated_shell_l1,
    dropped_shell_l1,
) -> FVMBudget:
    target = fvm_target.copy()
    if lattice.shell is not None:
        target = _move_shell_strength(lattice, target)[0]
    emitted = np.zeros_like(target)
    emitted[keep] = transfer_strength
    region = budget_nodes & ~lattice.solid_interior
    fvm_moments = vortex_moments(lattice.positions[region], target[region])
    emitted_moments = vortex_moments(lattice.positions[region], emitted[region])
    l1 = float(np.linalg.norm(target[region], axis=1).sum())
    error = float(
        np.linalg.norm(emitted_moments.total_vortex_strength - fvm_moments.total_vortex_strength)
        / (l1 + 1.0e-30)
    )
    zero = np.zeros(3, dtype=np.float64)
    wall_fvm, wall_emitted, ratio, wall_l1 = zero, zero, 1.0, 0.0
    if lattice.wall_distance is not None:
        near = region & (lattice.wall_distance < WALL_BUDGET_SPACINGS * lattice.particle_spacing)
        wall_fvm = target[near].sum(axis=0)
        wall_emitted = emitted[near].sum(axis=0)
        squared = float(wall_fvm @ wall_fvm)
        wall_l1 = float(np.linalg.norm(target[near], axis=1).sum())
        if squared > (1.0e-6 * wall_l1) ** 2 and squared > 0.0:
            ratio = float(wall_emitted @ wall_fvm) / squared
    return FVMBudget(
        fvm_total_vortex_strength=fvm_moments.total_vortex_strength,
        emitted_total_vortex_strength=emitted_moments.total_vortex_strength,
        fvm_linear_impulse=fvm_moments.linear_impulse,
        emitted_linear_impulse=emitted_moments.linear_impulse,
        fvm_vortex_strength_l1=l1,
        relative_error=error,
        wall_fvm_total_vortex_strength=wall_fvm,
        wall_emitted_total_vortex_strength=wall_emitted,
        wall_circulation_ratio=ratio,
        wall_fvm_vortex_strength_l1=wall_l1,
        relocated_shell_vortex_strength_l1=relocated_shell_l1,
        dropped_shell_vortex_strength_l1=dropped_shell_l1,
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
    "DROPPED_PARTICLE_STRENGTH_LIMIT",
    "LatticeTransferResult",
    "M4_PRIME_SUPPORT_CELLS",
    "TransferLattice",
    "VortexMoments",
    "blend_lattice_strength",
    "blending_weight",
    "build_transfer_lattice",
    "check_dropped_particle_strength",
    "check_fvm_budget",
    "correct_vortex_moments",
    "m4_prime",
    "redistribute_removed_strength",
    "remove_weak_strengths",
    "scatter_m4_prime_to_lattice",
    "transfer_on_lattice",
    "vortex_moment_residual",
    "vortex_moments",
]
