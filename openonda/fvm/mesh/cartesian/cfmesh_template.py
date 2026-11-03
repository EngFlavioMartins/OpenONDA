# SPDX-License-Identifier: GPL-3.0-or-later
"""cfMesh-compatible Cartesian template generation.

The first cfMesh workflow checkpoint is not body-conformal.  It is extracted
from the octree after every leaf intersected by the input surface has been
classified as a data box and excluded.  Consequently its temporary boundary
lies one leaf away from the real geometry.  Later workflow stages project that
boundary and add wrapper cells.

This module reproduces that deliberately intermediate topology.  It is kept
separate from projection and optimisation so differential tests can stop at
the first stage that disagrees.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence

import numpy as np

from openonda.core._numba import cacheable_njit as njit

from ..data import BoundaryPatchData, MeshData
from ..progress import mesh_stage
from ..surface_classification import (
    SurfaceIndex,
    closest_point_on_triangles,
    triangle_box_overlap,
)
from ..triangulated_surface import SurfaceBounds, TriangulatedSurface
from .cfmesh_automatic_refinement import AutomaticSurface, automatic_refinement
from .cfmesh_octree import (
    LeafLookup,
    _find_leaf,
    balance_leaves,
    object_additional_level,
    refine_near_data,
    refine_objects,
)
from .cfmesh_surface_optimisation import (
    _dot,
    _face_centre,
    _mag_squared,
    inverted_cfmesh_boundary_points,
    smooth_cfmesh_partition_points,
    untangle_cfmesh_surface,
)
from .config import BoxRefinement, PatchRefinement


def _additional_level(max_cell_size: float, requested: float) -> int:
    """Return cfMesh's first dyadic level whose size does not exceed a request."""
    level = 0
    while max_cell_size / (2**level) > requested:
        level += 1
    return level


def _root_cube(
    domain: SurfaceBounds,
    max_cell_size: float,
    lattice_spacing: float | None = None,
    lattice_origin: Sequence[float] | None = None,
) -> tuple[SurfaceBounds, int]:
    """Reproduce ``setRootCubeSizeAndRefParameters`` for the outer surface.

    cfMesh first creates a cube 1.5 times wider than the surface bounds, adds
    half a requested cell to that sizing calculation, and then chooses a
    power-of-two cube whose level cell size is exactly ``max_cell_size``.

    With ``lattice_spacing`` (m), the cube's lower corner is moved down onto the
    lattice ``lattice_origin + k * lattice_spacing`` (origin zero by default), so
    that every leaf face at or below the global level lies on such a plane. The
    cube keeps its size and still contains the domain: the shift is below one
    spacing while the cube exceeds the domain by a quarter of its extent on each
    side.
    """
    lower = np.asarray(domain[::2], dtype=np.float64)
    upper = np.asarray(domain[1::2], dtype=np.float64)
    centre = 0.5 * (lower + upper)
    size = 1.5 * float((upper - lower).max()) + 0.5 * max_cell_size
    global_level = 0
    while size / (2**global_level) >= max_cell_size * (1.0 - 1.0e-15):
        global_level += 1
    root_size = max_cell_size * (2**global_level)
    root_lower = centre - 0.5 * root_size
    root_upper = centre + 0.5 * root_size
    if lattice_spacing is not None:
        origin = np.zeros(3) if lattice_origin is None else np.asarray(lattice_origin, float)
        root_lower = origin + lattice_spacing * np.floor(
            (root_lower - origin) / lattice_spacing + 1.0e-9
        )
        root_upper = root_lower + root_size
    bounds = tuple(
        float(value) for axis in range(3) for value in (root_lower[axis], root_upper[axis])
    )
    return bounds, global_level  # type: ignore[return-value]


def _box_surface_intersects_arrays(
    lower: np.ndarray,
    upper: np.ndarray,
    surface_lower: np.ndarray,
    surface_upper: np.ndarray,
) -> bool:
    """Array form of :func:`_box_surface_intersects` for recursive walks."""
    for axis in range(3):
        touches_plane = (
            lower[axis] <= surface_lower[axis] <= upper[axis]
            or lower[axis] <= surface_upper[axis] <= upper[axis]
        )
        if not touches_plane:
            continue
        tangential = tuple(other for other in range(3) if other != axis)
        if all(
            upper[other] >= surface_lower[other] and lower[other] <= surface_upper[other]
            for other in tangential
        ):
            return True
    return False


def _boxes_touch_domain_sides(
    lower: np.ndarray,
    upper: np.ndarray,
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
) -> np.ndarray:
    """Row form of :func:`_box_surface_intersects_arrays` for boxes ``(n, 3)``."""
    result = np.zeros(len(lower), dtype=bool)
    for axis in range(3):
        touches = (
            (lower[:, axis] <= domain_lower[axis]) & (domain_lower[axis] <= upper[:, axis])
        ) | ((lower[:, axis] <= domain_upper[axis]) & (domain_upper[axis] <= upper[:, axis]))
        for other in range(3):
            if other != axis:
                touches &= (upper[:, other] >= domain_lower[other]) & (
                    lower[:, other] <= domain_upper[other]
                )
        result |= touches
    return result


def _classify_leaves(
    children: np.ndarray,
    *,
    root_lower: np.ndarray,
    finest_size: float,
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    surface_indices: Sequence[SurfaceIndex],
) -> np.ndarray:
    """Append the kind to new leaves ``(x, y, z, width, level)``.

    A leaf touching a domain side or an input surface carries surface data
    (2); otherwise it is a mesh leaf (1) if its centre lies inside the domain
    and outside every surface, and an outside leaf (0) if not. The tests are
    those of one leaf at a time, applied to all leaves at once.
    """
    xyz = children[:, :3]
    lower = root_lower + finest_size * xyz.astype(np.float64)
    upper = root_lower + finest_size * (xyz + children[:, 3:4]).astype(np.float64)
    intersects = _boxes_touch_domain_sides(lower, upper, domain_lower, domain_upper)
    for index in surface_indices:
        pending = np.flatnonzero(~intersects)
        intersects[pending] = index.boxes_intersect_surface(lower[pending], upper[pending])
    centre = 0.5 * (lower + upper)
    in_domain = np.all(centre > domain_lower, axis=1) & np.all(centre < domain_upper, axis=1)
    in_object = np.zeros(len(children), dtype=bool)
    for index in surface_indices:
        pending = np.flatnonzero(~intersects & in_domain & ~in_object)
        parity = index.single_ray_parity(centre[pending])
        in_object[pending[parity == 1]] = True
        # A degenerate first ray needs the full retry over every direction.
        for leaf in pending[parity < 0]:
            in_object[leaf] = bool(index.is_inside(centre[leaf][None, :])[0])
    kind = np.where(intersects, 2, np.where(in_domain & ~in_object, 1, 0))
    return np.column_stack((children, kind)).astype(np.int64)


@njit(cache=True, fastmath=False)
def _write_face_ring(
    rings: np.ndarray,
    row: int,
    axis: int,
    coordinate: int,
    a0: int,
    a1: int,
    b0: int,
    b1: int,
    stride: int,
    positive: bool,
) -> None:
    """Encode one axis-aligned face ring with outward orientation into ``rings[row]``.

    Lattice points are coded ``x + stride * (y + stride * z)``. cfMesh
    reverses a negative face while retaining its first cube vertex. The
    starting vertex matters to its finite-iteration surface optimizer even
    though cyclic rotations are topologically equivalent.
    """
    if axis == 0:
        r0 = coordinate + stride * (a0 + stride * b0)
        r1 = coordinate + stride * (a1 + stride * b0)
        r2 = coordinate + stride * (a1 + stride * b1)
        r3 = coordinate + stride * (a0 + stride * b1)
    elif axis == 1:
        r0 = a0 + stride * (coordinate + stride * b0)
        r1 = a0 + stride * (coordinate + stride * b1)
        r2 = a1 + stride * (coordinate + stride * b1)
        r3 = a1 + stride * (coordinate + stride * b0)
    else:
        r0 = a0 + stride * (b0 + stride * coordinate)
        r1 = a1 + stride * (b0 + stride * coordinate)
        r2 = a1 + stride * (b1 + stride * coordinate)
        r3 = a0 + stride * (b1 + stride * coordinate)
    rings[row, 0] = r0
    rings[row, 2] = r2
    if positive:
        rings[row, 1] = r1
        rings[row, 3] = r3
    else:
        rings[row, 1] = r3
        rings[row, 3] = r1


@njit(cache=True, fastmath=False)
def _face_neighbours(
    lookup: dict,
    axis: int,
    coordinate: int,
    a0: int,
    b0: int,
    width: int,
    max_level: int,
    found: np.ndarray,
) -> int:
    """Distinct leaves across a face, sampled at its quarter points; their count.

    The leaves are stored in ``found`` in sampling order.
    """
    count = 0
    samples = 1 if width == 1 else 2
    for i in range(samples):
        a = a0 if width == 1 else (a0 + width // 4 if i == 0 else a0 + (3 * width) // 4)
        for j in range(samples):
            b = b0 if width == 1 else (b0 + width // 4 if j == 0 else b0 + (3 * width) // 4)
            if axis == 0:
                leaf = _find_leaf(lookup, coordinate, a, b, max_level, max_level)
            elif axis == 1:
                leaf = _find_leaf(lookup, a, coordinate, b, max_level, max_level)
            else:
                leaf = _find_leaf(lookup, a, b, coordinate, max_level, max_level)
            if leaf < 0:
                continue
            duplicate = False
            for k in range(count):
                if found[k] == leaf:
                    duplicate = True
            if not duplicate:
                found[count] = leaf
                count += 1
    return count


@njit(cache=True, fastmath=False)
def _face_frame(
    x0: int, y0: int, z0: int, width: int, axis: int
) -> tuple[int, int, int, int, int, int]:
    """Lower and upper face coordinates along ``axis`` and the tangential extents."""
    if axis == 0:
        return x0, x0 + width, y0, y0 + width, z0, z0 + width
    if axis == 1:
        return y0, y0 + width, x0, x0 + width, z0, z0 + width
    return z0, z0 + width, x0, x0 + width, y0, y0 + width


@njit(cache=True, fastmath=False)
def _grown(array: np.ndarray) -> np.ndarray:
    """A copy of ``array`` with more rows."""
    result = np.empty((2 * len(array) + 16,) + array.shape[1:], dtype=array.dtype)
    result[: len(array)] = array
    return result


@njit(cache=True, fastmath=False)
def _template_faces_kernel(
    leaves: np.ndarray, max_level: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interior and boundary faces of the fluid leaves, in cfMesh's visiting order.

    Each leaf visits its xmin, xmax, ymin, ymax, zmin, zmax sides. A side with
    no fluid leaf across is a boundary face; the upper side of a leaf gives an
    interior face towards every leaf across, in increasing leaf order, each
    clipped to the smaller of the two leaves. The last array names the first
    leaf pair more than one level apart, or holds -1.
    """
    n_cells = len(leaves)
    lookup = {(leaves[i, 4], leaves[i, 0], leaves[i, 1], leaves[i, 2]): i for i in range(n_cells)}
    stride = (1 << max_level) + 1
    interior = np.empty((4 * n_cells + 16, 4), dtype=np.int64)
    owners = np.empty(len(interior), dtype=np.int64)
    neighbours = np.empty(len(interior), dtype=np.int64)
    boundary = np.empty((n_cells + 16, 4), dtype=np.int64)
    boundary_owners = np.empty(len(boundary), dtype=np.int64)
    unbalanced = np.full(2, -1, dtype=np.int64)
    found = np.empty(4, dtype=np.int64)
    n_interior = 0
    n_boundary = 0
    for owner in range(n_cells):
        x0, y0, z0, width, level = leaves[owner, :5]
        for axis in range(3):
            lower, upper, a0, a1, b0, b1 = _face_frame(x0, y0, z0, width, axis)
            if _face_neighbours(lookup, axis, lower - 1, a0, b0, width, max_level, found) == 0:
                if n_boundary == len(boundary):
                    boundary = _grown(boundary)
                    boundary_owners = _grown(boundary_owners)
                _write_face_ring(boundary, n_boundary, axis, lower, a0, a1, b0, b1, stride, False)
                boundary_owners[n_boundary] = owner
                n_boundary += 1
            count = _face_neighbours(lookup, axis, upper, a0, b0, width, max_level, found)
            if count == 0:
                if n_boundary == len(boundary):
                    boundary = _grown(boundary)
                    boundary_owners = _grown(boundary_owners)
                _write_face_ring(boundary, n_boundary, axis, upper, a0, a1, b0, b1, stride, True)
                boundary_owners[n_boundary] = owner
                n_boundary += 1
                continue
            ordered = np.sort(found[:count])
            for neighbour in ordered:
                nx0, ny0, nz0, n_width, n_level = leaves[neighbour, :5]
                if abs(level - n_level) > 1:
                    unbalanced[0] = owner
                    unbalanced[1] = neighbour
                    return (
                        interior[:n_interior],
                        owners[:n_interior],
                        neighbours[:n_interior],
                        boundary[:n_boundary],
                        boundary_owners[:n_boundary],
                        unbalanced,
                    )
                _n_lower, _n_upper, c0, c1, d0, d1 = _face_frame(nx0, ny0, nz0, n_width, axis)
                if n_interior == len(interior):
                    interior = _grown(interior)
                    owners = _grown(owners)
                    neighbours = _grown(neighbours)
                _write_face_ring(
                    interior,
                    n_interior,
                    axis,
                    upper,
                    max(a0, c0),
                    min(a1, c1),
                    max(b0, d0),
                    min(b1, d1),
                    stride,
                    True,
                )
                owners[n_interior] = owner
                neighbours[n_interior] = neighbour
                n_interior += 1
    return (
        interior[:n_interior],
        owners[:n_interior],
        neighbours[:n_interior],
        boundary[:n_boundary],
        boundary_owners[:n_boundary],
        unbalanced,
    )


def _extract_topology(
    leaves: np.ndarray, max_level: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[BoundaryPatchData], np.ndarray]:
    """Native faces of the already-classified fluid leaves.

    cfMesh extracts the template in its root cube, not the requested fluid
    box: its outer sides and the faces towards excluded surface leaves all
    form the one temporary ``defaultFaces`` patch.
    """
    limit = 2**max_level
    max_point_code = (limit + 1) ** 3 - 1
    code_dtype = np.int32 if max_point_code <= np.iinfo(np.int32).max else np.int64
    records = np.ascontiguousarray(leaves, dtype=np.int64)
    interior, owners, neighbours, boundary_faces, boundary_owners, unbalanced = (
        _template_faces_kernel(records, max_level)
    )
    if unbalanced[0] >= 0:
        owner, neighbour = (int(value) for value in unbalanced)
        raise RuntimeError(
            "Refinement transition is not 2:1 balanced between cells "
            f"{owner} (level {int(records[owner, 4])}) and {neighbour} "
            f"(level {int(records[neighbour, 4])})"
        )
    boundary: list[BoundaryPatchData] = [
        {
            "name": "defaultFaces",
            "start_face": len(interior),
            "n_faces": len(boundary_faces),
            "type": "patch",
        }
    ]
    return (
        np.ascontiguousarray(np.vstack((interior, boundary_faces)), dtype=code_dtype),
        np.ascontiguousarray(np.concatenate((owners, boundary_owners)), dtype=np.int32),
        np.ascontiguousarray(neighbours, dtype=np.int32),
        boundary,
        np.asarray(records[:, 4], dtype=np.int8),
    )


@njit(cache=True, fastmath=False)
def _octree_face_order_kernel(
    octree_leaves: np.ndarray, max_level: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Faces in the order cfMesh creates them while visiting every octree leaf.

    A face is created across a side with exactly one leaf of the same or a
    coarser level: between two fluid leaves once, and between a fluid leaf
    and a finer or equal surface-data leaf as a boundary face. Returns the
    sorted vertex codes of each face, whether it is internal, its one or two
    fluid cells (else -1), and the leaves that own octree vertices: every
    fluid leaf and every leaf across one of its sides.
    """
    n_leaves = len(octree_leaves)
    lookup = {
        (octree_leaves[i, 4], octree_leaves[i, 0], octree_leaves[i, 1], octree_leaves[i, 2]): i
        for i in range(n_leaves)
    }
    cell_ids = np.full(n_leaves, -1, dtype=np.int64)
    next_cell = 0
    for leaf_id in range(n_leaves):
        if octree_leaves[leaf_id, 5] == 1:
            cell_ids[leaf_id] = next_cell
            next_cell += 1
    participating = octree_leaves[:, 5] == 1
    stride = (1 << max_level) + 1
    signatures = np.empty((3 * n_leaves + 16, 4), dtype=np.int64)
    internal = np.empty(len(signatures), dtype=np.bool_)
    attached = np.empty((len(signatures), 2), dtype=np.int64)
    ring = np.empty((1, 4), dtype=np.int64)
    found = np.empty(4, dtype=np.int64)
    n_faces = 0
    for leaf_id in range(n_leaves):
        x0, y0, z0, width, level, kind = octree_leaves[leaf_id]
        for axis in range(3):
            lower, upper, a0, a1, b0, b1 = _face_frame(x0, y0, z0, width, axis)
            for positive in (False, True):
                query = upper if positive else lower - 1
                count = _face_neighbours(lookup, axis, query, a0, b0, width, max_level, found)
                if kind == 1:
                    for k in range(count):
                        participating[found[k]] = True
                if count != 1:
                    continue
                neighbour = found[0]
                neighbour_level = octree_leaves[neighbour, 4]
                neighbour_kind = octree_leaves[neighbour, 5]
                emitted_boundary = (kind == 1 and neighbour_kind == 2) or (
                    kind == 2 and neighbour_kind == 1 and neighbour_level < level
                )
                emitted_internal = (
                    kind == 1
                    and neighbour_kind == 1
                    and (neighbour > leaf_id or neighbour_level < level)
                )
                if not emitted_boundary and not emitted_internal:
                    continue
                if n_faces == len(signatures):
                    signatures = _grown(signatures)
                    internal = _grown(internal)
                    attached = _grown(attached)
                coordinate = upper if positive else lower
                _write_face_ring(ring, 0, axis, coordinate, a0, a1, b0, b1, stride, positive)
                signatures[n_faces] = np.sort(ring[0])
                internal[n_faces] = emitted_internal
                if emitted_internal:
                    attached[n_faces, 0] = cell_ids[leaf_id]
                    attached[n_faces, 1] = cell_ids[neighbour]
                elif kind == 1:
                    attached[n_faces, 0] = cell_ids[leaf_id]
                    attached[n_faces, 1] = -1
                else:
                    attached[n_faces, 0] = cell_ids[neighbour]
                    attached[n_faces, 1] = -1
                n_faces += 1
    return signatures[:n_faces], internal[:n_faces], attached[:n_faces], participating


_QUAD_KEY = np.dtype([(f"vertex_{index}", np.int64) for index in range(4)])


def _quad_keys(faces: np.ndarray) -> np.ndarray:
    """One sortable key per quad, its sorted vertex codes."""
    return (
        np.ascontiguousarray(np.sort(np.asarray(faces, dtype=np.int64), axis=1))
        .view(_QUAD_KEY)
        .reshape(-1)
    )


def _last_matches(keys: np.ndarray, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index of the last equal key for each query, and whether there is one."""
    order = np.argsort(keys, kind="stable")
    ordered = keys[order]
    positions = np.searchsorted(ordered, queries, side="right") - 1
    if not len(ordered):
        return np.zeros(len(queries), dtype=np.int64), np.zeros(len(queries), dtype=bool)
    clipped = np.maximum(positions, 0)
    return order[clipped], (positions >= 0) & (ordered[clipped] == queries)


@njit(cache=True)
def _expand_lattice_faces(
    encoded_faces: np.ndarray, sx: int, sy: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Split lattice edges at existing nodes without Python polygon scans."""
    available = set(encoded_faces.reshape(-1))
    flat = []
    offsets = np.empty(len(encoded_faces) + 1, dtype=np.int64)
    offsets[0] = 0
    expanded_count = 0
    for face_id in range(len(encoded_faces)):
        face = encoded_faces[face_id]
        for position in range(len(face)):
            first = face[position]
            second = face[(position + 1) % len(face)]
            flat.append(first)
            x, y, z = first % sx, (first // sx) % sy, first // (sx * sy)
            dx = second % sx - x
            dy = (second // sx) % sy - y
            dz = second // (sx * sy) - z
            length = max(abs(dx), abs(dy), abs(dz))
            if length <= 1:
                continue
            dx, dy, dz = dx // length, dy // length, dz // length
            for step in range(1, length):
                candidate = x + step * dx + sx * (y + step * dy + sy * (z + step * dz))
                if candidate in available:
                    flat.append(candidate)
        offsets[face_id + 1] = len(flat)
        if offsets[face_id + 1] - offsets[face_id] > len(face):
            expanded_count += 1
    return np.asarray(flat, dtype=np.int64), offsets, expanded_count


def _extract_mesh(
    root_bounds: SurfaceBounds,
    root_size: float,
    leaves: np.ndarray,
    octree_leaves: np.ndarray,
    max_level: int,
    global_level: int,
    boundary_level: int,
) -> MeshData:
    limits = (2**max_level, 2**max_level, 2**max_level)
    encoded_faces, owners, neighbours, boundary, levels = _extract_topology(leaves, max_level)

    # cfMesh creates octree faces while visiting every Morton-ordered leaf,
    # including excluded surface-data leaves.  A fine excluded leaf therefore
    # emits the boundary face towards a coarser fluid leaf at its own position
    # in that traversal.  Reproduce that ordering: its finite-iteration
    # surface optimiser constructs local triangle simplexes in face order.
    signatures, internal, attached, participating = _octree_face_order_kernel(
        np.ascontiguousarray(octree_leaves, dtype=np.int64), max_level
    )

    # ``reorderBoundaryFaces`` swaps misplaced boundary and internal faces in
    # pairs; it is intentionally not a stable partition.  Simulating those
    # swaps recovers the boundary order seen by all later cfMesh stages.
    n_internal = len(neighbours)
    internal_to_change = np.flatnonzero(~internal[:n_internal])
    boundary_to_change = n_internal + np.flatnonzero(internal[n_internal:])
    if len(internal_to_change) != len(boundary_to_change):
        raise RuntimeError("cfMesh octree face-order reconstruction is inconsistent")
    reordered = np.arange(len(signatures))
    reordered[internal_to_change] = boundary_to_change
    reordered[boundary_to_change] = internal_to_change
    final_boundary = reordered[n_internal:]
    final_boundary = final_boundary[~internal[final_boundary]]
    # Each boundary signature takes the position of its last occurrence there.
    face_order_keys = _quad_keys(signatures[final_boundary])
    face_order_entries = len(np.unique(face_order_keys))
    matched_boundary_order = 0
    if len(encoded_faces) > n_internal:
        position, matched = _last_matches(face_order_keys, _quad_keys(encoded_faces[n_internal:]))
        matched_boundary_order = int(np.count_nonzero(matched))
        boundary_order = np.where(
            matched, position, face_order_entries + np.arange(len(matched), dtype=np.int64)
        )
        permutation = np.argsort(boundary_order, kind="stable")
        encoded_faces[n_internal:] = encoded_faces[n_internal:][permutation]
        owners[n_internal:] = owners[n_internal:][permutation]

    # Each cell lists its octree faces in creation order; a signature repeated
    # by several faces names the last of them.
    face_ids, matched = _last_matches(_quad_keys(encoded_faces), _quad_keys(signatures))
    if not np.all(matched):
        raise RuntimeError("cfMesh cell-face ordering references an absent octree face")
    cells = attached.reshape(-1)
    cell_faces = np.repeat(face_ids, 2)[cells >= 0]
    cells = cells[cells >= 0]
    counts = np.bincount(cells, minlength=len(leaves))
    if np.any(counts == 0):
        raise RuntimeError("cfMesh cell-face ordering left an empty mesh cell")
    flat_cell_faces = cell_faces[np.argsort(cells, kind="stable")].tolist()
    offsets = np.concatenate(([0], np.cumsum(counts))).tolist()
    cfmesh_cell_face_order = [
        flat_cell_faces[start:stop] for start, stop in zip(offsets[:-1], offsets[1:], strict=True)
    ]

    sx = limits[0] + 1
    sy = limits[1] + 1
    # Retain hanging edge vertices with compiled integer lattice arithmetic.
    flat_face_codes, face_offsets, expanded_face_count = _expand_lattice_faces(
        encoded_faces, sx, sy
    )
    point_codes = np.unique(flat_face_codes)
    flat_face_indices = np.searchsorted(point_codes, flat_face_codes).astype(np.int32)
    face_widths = np.diff(face_offsets)
    faces: list[np.ndarray] | np.ndarray = (
        np.ascontiguousarray(flat_face_indices.reshape(-1, int(face_widths[0])), dtype=np.int32)
        if len(face_widths) and np.all(face_widths == face_widths[0])
        else [
            flat_face_indices[start:stop]
            for start, stop in zip(face_offsets[:-1], face_offsets[1:], strict=True)
        ]
    )
    px = point_codes % sx
    yz = point_codes // sx
    py = yz % sy
    pz = yz // sy
    points = np.column_stack((px, py, pz)).astype(np.float64)
    finest_size = root_size / (2**max_level)
    points *= finest_size
    points += np.asarray(root_bounds[::2], dtype=np.float64)
    # meshOctreeCubeCoordinates::vertices expands EACH leaf by SMALL*span.
    # createOctreePoints writes shared nodes in leaf order, so the last
    # participating leaf determines each vertex's signed perturbation. It is
    # not a root-box expansion or a uniformly shifted Cartesian lattice.
    root_lower = np.asarray(root_bounds[::2], dtype=np.float64)
    root_span = np.asarray(root_bounds[1::2], dtype=np.float64) - root_lower
    vertex_tolerance = 1.0e-15 * root_span
    records = np.asarray(octree_leaves, dtype=np.int64)[participating]
    lower_lattice = records[:, :3]
    widths = records[:, 3]
    corners = ((np.arange(8)[:, None] >> np.arange(3)) & 1).astype(np.int64)
    lattice = lower_lattice[:, None, :] + widths[:, None, None] * corners[None, :, :]
    codes = lattice[:, :, 0] + sx * (lattice[:, :, 1] + sy * lattice[:, :, 2])
    flat_codes = codes.ravel()

    # ``dict.setdefault`` above encoded first occurrence order while the
    # point coordinates were overwritten by the last participating leaf.
    # Recover both orders with vectorized unique operations; this removes an
    # 8-corner Python loop over every reference-grid leaf.
    _unique_codes, first_positions = np.unique(flat_codes, return_index=True)
    first_order_codes = _unique_codes[np.argsort(first_positions, kind="stable")]
    ordered_labels = np.searchsorted(point_codes, first_order_codes)
    ordered_valid = (ordered_labels < len(point_codes)) & (
        point_codes[np.minimum(ordered_labels, len(point_codes) - 1)] == first_order_codes
    )
    point_order = ordered_labels[ordered_valid].astype(np.int32)
    reversed_codes, reverse_positions = np.unique(flat_codes[::-1], return_index=True)
    last_positions = len(flat_codes) - 1 - reverse_positions
    last_labels = np.searchsorted(point_codes, reversed_codes)
    last_valid = (last_labels < len(point_codes)) & (
        point_codes[np.minimum(last_labels, len(point_codes) - 1)] == reversed_codes
    )
    last_labels = last_labels[last_valid].astype(np.int32)
    last_positions = last_positions[last_valid]
    last_leaf = last_positions // 8
    last_corner = last_positions % 8
    last_records = records[last_leaf]
    last_lower = root_lower + (
        root_span / np.power(2.0, last_records[:, 4].astype(np.int64))[:, None]
    ) * (last_records[:, :3] // last_records[:, 3, None])
    last_upper = (
        last_lower + root_span / np.power(2.0, last_records[:, 4].astype(np.int64))[:, None]
    )
    last_lower -= vertex_tolerance
    last_upper += vertex_tolerance
    points[last_labels] = np.where(corners[last_corner].astype(bool), last_upper, last_lower)
    cell_vertex_indices = np.empty((len(leaves), 8), dtype=np.int32)
    lower_x, lower_y, lower_z, leaf_width = (
        leaves[:, index].astype(np.int64) for index in range(4)
    )
    upper_x, upper_y, upper_z = lower_x + leaf_width, lower_y + leaf_width, lower_z + leaf_width

    def encode(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        return x + sx * (y + sy * z)

    cell_codes = np.column_stack(
        (
            encode(lower_x, lower_y, lower_z),
            encode(upper_x, lower_y, lower_z),
            encode(upper_x, upper_y, lower_z),
            encode(lower_x, upper_y, lower_z),
            encode(lower_x, lower_y, upper_z),
            encode(upper_x, lower_y, upper_z),
            encode(upper_x, upper_y, upper_z),
            encode(lower_x, upper_y, upper_z),
        )
    )
    indices = np.searchsorted(point_codes, cell_codes)
    if np.any(indices >= len(point_codes)) or not np.array_equal(point_codes[indices], cell_codes):
        raise RuntimeError("A cfMesh-template cell corner is absent from the face points")
    cell_vertex_indices[:] = indices.astype(np.int32)
    # Native createNodeLabels visits participating leaves and their corners
    # in order, then unused vertices are compacted without reordering. Later
    # boundary-edge/corner traversal uses these labels, so retain that order.
    if len(point_order) != len(points):
        raise RuntimeError("A template vertex has no participating octree corner")
    old_to_new = np.empty(len(points), dtype=np.int32)
    old_to_new[point_order] = np.arange(len(points), dtype=np.int32)
    points = points[point_order]
    faces = (
        old_to_new[faces] if isinstance(faces, np.ndarray) else [old_to_new[face] for face in faces]
    )
    cell_vertex_indices = old_to_new[cell_vertex_indices]
    return {
        "vertex_position": np.ascontiguousarray(points),
        "faces": faces,
        "owners": owners,
        "neighbours": neighbours,
        "boundary": boundary,
        "n_cells": len(leaves),
        "n_faces": len(faces),
        "n_interior_faces": len(neighbours),
        "n_points": len(points),
        "cell_levels": levels,
        "cell_sizes": np.asarray(root_size / np.power(2.0, levels), dtype=np.float32),
        "cell_vertex_indices": cell_vertex_indices,
        "cell_type_code": np.full(len(leaves), 5, dtype=np.int32),
        "_cfmesh_cell_face_order": cfmesh_cell_face_order,
        "mesh_generation": {
            "method": "cfmesh_template_octree",
            "root_box": root_bounds,
            "global_refinement_level": global_level,
            "boundary_refinement_level": boundary_level,
            "surface_patch_refinement_levels": {},
            "finest_cell_size": finest_size,
            "ordered_boundary_faces": matched_boundary_order,
            "octree_boundary_face_order_entries": face_order_entries,
            "coarse_faces_with_hanging_perimeter_points": expanded_face_count,
            "workflow_checkpoint": "templateGeneration",
        },
    }


def build_cfmesh_template(
    *,
    domain: SurfaceBounds,
    surfaces: Sequence[TriangulatedSurface],
    max_cell_size: float,
    boundary_cell_size: float,
    min_cell_size: float | None = None,
    box_refinements: Sequence[BoxRefinement] = (),
    patch_refinements: Sequence[PatchRefinement] = (),
    domain_patch_names: Sequence[str] = (),
    surface_patch_names: Sequence[str] = (),
    exact_cell_sizes: bool = False,
    root_lattice_spacing: float | None = None,
    root_lattice_origin: Sequence[float] | None = None,
) -> MeshData:
    """Build the cfMesh template, optionally preserving an exact size lattice.

    ``root_lattice_spacing`` and ``root_lattice_origin`` align the octree with the
    planes ``origin + k * spacing`` (see :func:`_root_cube`); exact Cartesian
    coupling sides require it.
    """
    root_bounds, global_level = _root_cube(
        domain, max_cell_size, root_lattice_spacing, root_lattice_origin
    )
    boundary_level = global_level + _additional_level(max_cell_size, boundary_cell_size)
    automatic_level = (
        global_level + _additional_level(max_cell_size, min_cell_size * (1.0 + 1.0e-15))
        if min_cell_size is not None
        else 0
    )
    needs_automatic = automatic_level > boundary_level
    patch_levels = {
        request.patch: global_level + _additional_level(max_cell_size, request.cell_size)
        for request in patch_refinements
    }
    object_level = _additional_level if exact_cell_sizes else object_additional_level
    max_level = max(
        [
            boundary_level,
            *patch_levels.values(),
            *(
                global_level + object_level(max_cell_size, request.cell_size)
                for request in box_refinements
            ),
        ]
    )
    if needs_automatic:
        # Automatic refinement adds a full neighbour layer, including finer
        # leaves, and can therefore exceed its selected-leaf level by one.
        max_level = max(max_level, automatic_level) + 1
    root_lower = np.asarray(root_bounds[::2], dtype=np.float64)
    root_upper = np.asarray(root_bounds[1::2], dtype=np.float64)
    root_size = float(root_upper[0] - root_lower[0])
    finest_size = root_size / (2**max_level)
    lattice_width = 2**max_level
    domain_lower = np.asarray(domain[::2], dtype=np.float64)
    domain_upper = np.asarray(domain[1::2], dtype=np.float64)
    surface_indices = tuple(SurfaceIndex.build(surface.triangles) for surface in surfaces)
    local_indices: list[tuple[SurfaceIndex, int]] = []
    if patch_levels:
        domain_triangles = _box_triangles(domain)
        for side, name in enumerate(domain_patch_names):
            if name in patch_levels:
                local_indices.append(
                    (
                        SurfaceIndex.build(domain_triangles[2 * side : 2 * side + 2]),
                        patch_levels[name],
                    )
                )
        for index, name in zip(surface_indices, surface_patch_names, strict=True):
            if name in patch_levels:
                local_indices.append((index, patch_levels[name]))

    def intersects_input_surface(lower: np.ndarray, upper: np.ndarray) -> bool:
        if _box_surface_intersects_arrays(lower, upper, domain_lower, domain_upper):
            return True
        return any(index.box_intersects_surface(lower, upper) for index in surface_indices)

    octree_leaves: list[tuple[int, int, int, int, int, int]] = []

    mesh_cell = 1
    surface_data_cell = 2
    other_cell = 0

    def visit(x0: int, y0: int, z0: int, width: int, level: int) -> None:
        lower = root_lower + finest_size * np.asarray((x0, y0, z0), dtype=np.float64)
        upper = root_lower + finest_size * np.asarray(
            (x0 + width, y0 + width, z0 + width), dtype=np.float64
        )
        overlaps_domain = bool(np.all(upper > domain_lower) and np.all(lower < domain_upper))
        if not overlaps_domain:
            # The cfMesh root cube is intentionally larger than the requested
            # fluid box.  Prune exterior branches before descending to the
            # global level; otherwise a sparse octree degenerates into every
            # one of the root cube's (often millions of) outside cells.
            octree_leaves.append((x0, y0, z0, width, level, other_cell))
            return
        intersects = intersects_input_surface(lower, upper)
        target_level = global_level
        if level >= global_level and intersects:
            target_level = boundary_level
            for index, local_level in local_indices:
                if local_level > target_level and index.box_intersects_surface(lower, upper):
                    target_level = local_level
        if level < target_level:
            child = width // 2
            for dz in (0, child):
                for dy in (0, child):
                    for dx in (0, child):
                        visit(x0 + dx, y0 + dy, z0 + dz, child, level + 1)
            return

        # cfMesh excludes every final leaf carrying surface data.  The
        # resulting temporary boundary is intentionally offset from the STL.
        if intersects:
            octree_leaves.append((x0, y0, z0, width, level, surface_data_cell))
            return
        centre = 0.5 * (lower + upper)
        if not bool(np.all(centre > domain_lower) and np.all(centre < domain_upper)):
            octree_leaves.append((x0, y0, z0, width, level, other_cell))
            return
        if any(bool(index.is_inside(centre[None, :])[0]) for index in surface_indices):
            octree_leaves.append((x0, y0, z0, width, level, other_cell))
            return
        octree_leaves.append((x0, y0, z0, width, level, mesh_cell))

    with mesh_stage("octree seed refinement") as progress:
        visit(0, 0, 0, lattice_width, 0)
        progress.details(leaves=len(octree_leaves), maximum_level=max_level)
    if box_refinements or patch_refinements or needs_automatic:

        def classify(children: np.ndarray) -> np.ndarray:
            return _classify_leaves(
                children,
                root_lower=root_lower,
                finest_size=finest_size,
                domain_lower=domain_lower,
                domain_upper=domain_upper,
                surface_indices=surface_indices,
            )

        with mesh_stage("octree balancing") as progress:
            records = balance_leaves(octree_leaves, max_level, classify)
            progress.details(leaves=len(records))
        automatic_diagnostics: dict[str, list[int]] = {}
        if needs_automatic:
            groups: dict[str, list[np.ndarray]] = defaultdict(list)
            triangles = _box_triangles(domain)
            names = tuple(domain_patch_names) or tuple(f"domain_{i}" for i in range(6))
            for side, name in enumerate(names):
                groups[name].extend(triangles[2 * side : 2 * side + 2])
            names = tuple(surface_patch_names) or tuple(
                f"surface_{i}" for i in range(len(surfaces))
            )
            for surface, name in zip(surfaces, names, strict=True):
                groups[name].extend(surface.triangles)
            automatic_surface = AutomaticSurface(
                {name: np.asarray(triangles) for name, triangles in groups.items()},
                _cfmesh_nearest_points_on_triangles,
            )
            with mesh_stage("automatic surface refinement") as progress:
                records, automatic_diagnostics = automatic_refinement(
                    records,
                    automatic_surface,
                    root_lower=root_lower,
                    root_size=root_size,
                    max_level=max_level,
                    automatic_level=automatic_level,
                    classify=classify,
                )
                progress.details(leaves=len(records))
        with mesh_stage("object refinement") as progress:
            records = refine_objects(
                records,
                box_refinements,
                root_lower=root_lower,
                root_size=root_size,
                max_cell_size=max_cell_size,
                global_level=global_level,
                max_level=max_level,
                exact_cell_sizes=exact_cell_sizes,
                classify=classify,
            )
            progress.details(leaves=len(records))
        with mesh_stage("near-surface refinement") as progress:
            records, near_data_refined = refine_near_data(records, max_level, classify)
            progress.details(leaves=len(records), refined=near_data_refined)
        leaves = records[records[:, 5] == mesh_cell, :5]
        if not len(leaves):
            raise ValueError("cfMesh template classification removed every fluid leaf")
        with mesh_stage("template topology extraction") as progress:
            mesh_data = _extract_mesh(
                root_bounds,
                root_size,
                np.ascontiguousarray(leaves, dtype=np.int32),
                np.ascontiguousarray(records, dtype=np.int32),
                max_level,
                global_level,
                boundary_level,
            )
            progress.details(
                cells=mesh_data.get("n_cells"),
                faces=mesh_data.get("n_faces"),
                points=mesh_data.get("n_points"),
            )
        mesh_data["_cfmesh_octree_leaves"] = np.ascontiguousarray(records, dtype=np.int32)
        mesh_data["mesh_generation"]["near_data_coarse_leaves_refined"] = near_data_refined
        mesh_data["mesh_generation"]["automatic_refinement"] = automatic_diagnostics
        mesh_data["mesh_generation"]["surface_patch_refinement_levels"] = dict(patch_levels)
        return mesh_data
    # ``refineBoxesNearDataBoxes(1)`` refines every non-outside coarse leaf
    # touching a surface-data leaf through a face, edge, or vertex. This final
    # regularity shell is material for oblique geometry: refining only boxes
    # intersected by triangles leaves too many coarse transition cells.
    data_boxes = np.asarray(
        [record[:4] for record in octree_leaves if record[5] == surface_data_cell],
        dtype=np.int32,
    ).reshape(-1, 4)
    regularised: list[tuple[int, int, int, int, int, int]] = []
    near_data_refined = 0
    for record in octree_leaves:
        x0, y0, z0, width, level, kind = record
        touches_data = False
        if kind == mesh_cell and level < max_level and len(data_boxes):
            lower = np.asarray((x0, y0, z0), dtype=np.int32)
            upper = lower + width
            data_lower = data_boxes[:, :3]
            data_upper = data_lower + data_boxes[:, 3, None]
            touches = np.all((data_lower <= upper) & (data_upper >= lower), axis=1)
            touches_data = bool(np.any(touches))
        if not touches_data:
            regularised.append(record)
            continue
        child = width // 2
        if child < 1:
            raise RuntimeError("cfMesh near-data refinement exceeded the finest lattice")
        near_data_refined += 1
        for dz in (0, child):
            for dy in (0, child):
                for dx in (0, child):
                    regularised.append((x0 + dx, y0 + dy, z0 + dz, child, level + 1, mesh_cell))
    octree_leaves = regularised
    leaves = [record[:5] for record in octree_leaves if record[5] == mesh_cell]
    if not leaves:
        raise ValueError("cfMesh template classification removed every fluid leaf")
    leaf_array = np.ascontiguousarray(leaves, dtype=np.int32)
    mesh_data = _extract_mesh(
        root_bounds,
        root_size,
        leaf_array,
        np.ascontiguousarray(octree_leaves, dtype=np.int32),
        max_level,
        global_level,
        boundary_level,
    )
    mesh_data["_cfmesh_octree_leaves"] = np.ascontiguousarray(octree_leaves, dtype=np.int32)
    mesh_data["mesh_generation"]["near_data_coarse_leaves_refined"] = near_data_refined
    mesh_data["mesh_generation"]["surface_patch_refinement_levels"] = dict(patch_levels)
    return mesh_data


def _box_triangles(bounds: SurfaceBounds) -> np.ndarray:
    """Return the twelve outward-oriented triangles of an axis-aligned box."""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    corners = np.asarray(
        (
            (xmin, ymin, zmin),
            (xmin, ymin, zmax),
            (xmin, ymax, zmin),
            (xmin, ymax, zmax),
            (xmax, ymin, zmin),
            (xmax, ymin, zmax),
            (xmax, ymax, zmin),
            (xmax, ymax, zmax),
        ),
        dtype=np.float64,
    )
    quads = (
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 5, 7, 3),
    )
    return np.ascontiguousarray(
        [
            (corners[a], corners[b], corners[c])
            for a, b, c, d in quads
            for a, b, c in ((a, b, c), (a, c, d))
        ],
        dtype=np.float64,
    )


def _foam_face_centre(coordinates: np.ndarray) -> np.ndarray:
    """Area-weighted polygon centre used by OpenFOAM's ``face::centre``."""
    return _face_centre(coordinates)


def _face_area_vector_with_centre(coordinates: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """OpenFOAM polygon area vector about an explicitly supplied centre."""
    area = np.zeros(3, dtype=np.float64)
    for i, current in enumerate(coordinates):
        following = coordinates[(i + 1) % len(coordinates)]
        area += 0.5 * np.cross(following - current, centre - current)
    return area


def _cfmesh_nearest_points_on_triangles(point: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Vectorise cfMesh's barycentric nearest-point arithmetic."""

    def row_dot(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        # Match Foam::vector::operator& component order.  BLAS/einsum may use
        # a different reduction tree (or contraction), which is visible at
        # iterative patch intersections on highly symmetric surfaces.
        return first[:, 0] * second[:, 0] + first[:, 1] * second[:, 1] + first[:, 2] * second[:, 2]

    a = triangles[:, 0]
    vector_0 = triangles[:, 1] - a
    vector_1 = triangles[:, 2] - a
    vector_2 = point - a
    dot_00 = row_dot(vector_0, vector_0)
    dot_01 = row_dot(vector_0, vector_1)
    dot_02 = row_dot(vector_0, vector_2)
    dot_11 = row_dot(vector_1, vector_1)
    dot_12 = row_dot(vector_1, vector_2)
    determinant = dot_00 * dot_11 - dot_01 * dot_01
    degenerate = np.abs(determinant) < 1.0e-300
    safe_determinant = np.where(degenerate, 1.0, determinant)
    u = (dot_11 * dot_02 - dot_01 * dot_12) / safe_determinant
    v = (dot_00 * dot_12 - dot_01 * dot_02) / safe_determinant
    projected = a + u[:, None] * vector_0 + v[:, None] * vector_1
    result = projected.copy()

    inside = (u >= -1.0e-15) & (v >= -1.0e-15) & (u + v <= 1.0 + 1.0e-15)
    outside = ~inside & ~degenerate
    before_u = outside & (u < -1.0e-15)
    before_v = outside & ~before_u & (v < -1.0e-15)
    opposite = outside & ~before_u & ~before_v

    if np.any(before_u):
        direction = vector_1[before_u]
        fraction = row_dot(projected[before_u] - a[before_u], direction) / (
            row_dot(direction, direction) + 1.0e-300
        )
        mapped = a[before_u] + np.clip(fraction, 0.0, 1.0)[:, None] * direction
        mapped[fraction < 0.0] = a[before_u][fraction < 0.0]
        mapped[fraction > 1.0] = triangles[before_u, 2][fraction > 1.0]
        result[before_u] = mapped
    if np.any(before_v):
        direction = vector_0[before_v]
        fraction = row_dot(projected[before_v] - a[before_v], direction) / (
            row_dot(direction, direction) + 1.0e-300
        )
        mapped = a[before_v] + np.clip(fraction, 0.0, 1.0)[:, None] * direction
        mapped[fraction < 0.0] = a[before_v][fraction < 0.0]
        mapped[fraction > 1.0] = triangles[before_v, 1][fraction > 1.0]
        result[before_v] = mapped
    if np.any(opposite):
        c = triangles[opposite, 2]
        direction = triangles[opposite, 1] - c
        fraction = row_dot(projected[opposite] - c, direction) / (
            row_dot(direction, direction) + 1.0e-300
        )
        mapped = c + np.clip(fraction, 0.0, 1.0)[:, None] * direction
        mapped[fraction < 0.0] = c[fraction < 0.0]
        mapped[fraction > 1.0] = triangles[opposite, 1][fraction > 1.0]
        result[opposite] = mapped
    if np.any(degenerate):
        result[degenerate] = closest_point_on_triangles(
            point,
            triangles[degenerate, 0],
            triangles[degenerate, 1],
            triangles[degenerate, 2],
        )
    return result


class _OctreeSurfaceLocator:
    """Cached replica of cfMesh's deliberately local surface search."""

    def __init__(
        self,
        index: SurfaceIndex,
        *,
        root_bounds: SurfaceBounds,
        finest_cell_size: float,
        leaves: np.ndarray,
    ) -> None:
        self.index = index
        self.root_lower = np.asarray(root_bounds[::2], dtype=np.float64)
        self.finest_cell_size = finest_cell_size
        self.leaves = leaves
        self.lower_lattice = leaves[:, :3].astype(np.int64)
        self.widths = leaves[:, 3].astype(np.int64)
        self.leaf_lower = self.root_lower + finest_cell_size * self.lower_lattice
        self.leaf_upper = self.leaf_lower + finest_cell_size * self.widths[:, None]
        lattice_width = int(round((root_bounds[1] - root_bounds[0]) / finest_cell_size))
        self.leaf_lookup = LeafLookup(leaves, lattice_width.bit_length() - 1)
        self.tolerance = 1.0e-15 * float(root_bounds[1] - root_bounds[0])
        self._leaf_triangles: dict[int, np.ndarray] = {}
        self._nearest_cache: dict[tuple[int, bytes], tuple[np.ndarray, float, int]] = {}

    def leaves_in_box(self, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        """Return octree leaves with positive-volume overlap with a box."""
        lattice_limit = self.leaf_lookup.limit
        first = np.floor((lower - self.root_lower) / self.finest_cell_size).astype(np.int64)
        last = np.ceil((upper - self.root_lower) / self.finest_cell_size).astype(np.int64) - 1
        first = np.clip(first, 0, lattice_limit - 1)
        last = np.clip(last, 0, lattice_limit - 1)
        return self.leaf_lookup.in_box(first, last)

    def _triangles_in_leaf(self, leaf_id: int) -> np.ndarray:
        cached = self._leaf_triangles.get(leaf_id)
        if cached is not None:
            return cached
        lower = self.leaf_lower[leaf_id] - self.tolerance
        upper = self.leaf_upper[leaf_id] + self.tolerance
        broad = self.index.candidate_triangles(lower, upper)
        if len(broad):
            triangles = self.index.triangles[broad]
            overlap = triangle_box_overlap(
                0.5 * (lower + upper),
                0.5 * (upper - lower),
                triangles[:, 0],
                triangles[:, 1],
                triangles[:, 2],
            )
            broad = broad[overlap]
        self._leaf_triangles[leaf_id] = broad
        return broad

    def nearest_triangle(
        self,
        point: np.ndarray,
        *,
        max_search_iterations: int,
    ) -> tuple[np.ndarray, float, int]:
        """Stop at the first octree neighborhood containing any triangles."""
        value = np.asarray(point, dtype=np.float64)
        cache_key = (max_search_iterations, value.tobytes())
        cached = self._nearest_cache.get(cache_key)
        if cached is not None:
            return cached
        # Query with the actual vertex, including any leaf-corner perturbation.
        # Native findNearestSurfacePointInRegion does not shift the search box.
        search_value = value
        lattice = np.floor((search_value - self.root_lower) / self.finest_cell_size).astype(
            np.int64
        )
        lattice = np.clip(lattice, 0, self.leaf_lookup.limit - 1)
        containing = self.leaf_lookup.find(*map(int, lattice))
        search_size = (
            0.75 * float(self.widths[containing]) * self.finest_cell_size
            if containing >= 0
            else self.finest_cell_size
        )
        for _iteration in range(max_search_iterations + 1):
            selected = self.leaves_in_box(search_value - search_size, search_value + search_size)
            candidate_ids: list[int] = []
            seen_candidates: set[int] = set()
            for leaf_id_value in selected[selected >= 0]:
                for triangle_id_value in self._triangles_in_leaf(int(leaf_id_value)):
                    triangle_id = int(triangle_id_value)
                    if triangle_id in seen_candidates:
                        continue
                    seen_candidates.add(triangle_id)
                    candidate_ids.append(triangle_id)
            if candidate_ids:
                ids = np.asarray(candidate_ids, dtype=np.int64)
                triangles = self.index.triangles[ids]
                candidates = _cfmesh_nearest_points_on_triangles(value, triangles)
                offsets = candidates - value
                distance_squared = (
                    offsets[:, 0] * offsets[:, 0]
                    + offsets[:, 1] * offsets[:, 1]
                    + offsets[:, 2] * offsets[:, 2]
                )
                best = int(np.argmin(distance_squared))
                result = (
                    candidates[best],
                    float(np.sqrt(distance_squared[best])),
                    int(ids[best]),
                )
                self._nearest_cache[cache_key] = result
                return result
            search_size *= 2.0
        triangles = self.index.triangles
        candidates = _cfmesh_nearest_points_on_triangles(value, triangles)
        offsets = candidates - value
        distance_squared = (
            offsets[:, 0] * offsets[:, 0]
            + offsets[:, 1] * offsets[:, 1]
            + offsets[:, 2] * offsets[:, 2]
        )
        best = int(np.argmin(distance_squared))
        result = (candidates[best], float(np.sqrt(distance_squared[best])), best)
        self._nearest_cache[cache_key] = result
        return result

    def nearest(self, point: np.ndarray, *, max_search_iterations: int) -> tuple[np.ndarray, float]:
        """Return the local nearest point and distance without its triangle id."""
        nearest, distance, _triangle = self.nearest_triangle(
            point, max_search_iterations=max_search_iterations
        )
        return nearest, distance


class _SurfaceFeatureLocator:
    """Native patch-intersection edges and corners, without box-specific snaps."""

    def __init__(self, patch_locators: Sequence[_OctreeSurfaceLocator]) -> None:
        self.locator = patch_locators[0]
        point_ids: dict[tuple[float, ...], int] = {}
        vertices: list[np.ndarray] = []
        self.triangles: list[tuple[int, ...]] = []
        self.triangle_patches: list[int] = []
        self.point_patches: dict[int, set[int]] = defaultdict(set)
        edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
        for patch_id, locator in enumerate(patch_locators):
            for triangle in locator.index.triangles:
                ids: list[int] = []
                for point in triangle:
                    key = tuple(map(float, point))
                    if key not in point_ids:
                        point_ids[key] = len(vertices)
                        vertices.append(point)
                    point_id = point_ids[key]
                    ids.append(point_id)
                    self.point_patches[point_id].add(patch_id)
                triangle_id = len(self.triangles)
                self.triangles.append(tuple(ids))
                self.triangle_patches.append(patch_id)
                for i in range(3):
                    first, second = ids[i], ids[(i + 1) % 3]
                    edge_faces[(min(first, second), max(first, second))].append(triangle_id)
        self.points = np.asarray(vertices)
        self.feature_patches = {
            edge: {self.triangle_patches[face] for face in faces}
            for edge, faces in edge_faces.items()
            if len(faces) == 2
            and self.triangle_patches[faces[0]] != self.triangle_patches[faces[1]]
        }
        feature_degree: dict[int, int] = defaultdict(int)
        for first, second in self.feature_patches:
            feature_degree[first] += 1
            feature_degree[second] += 1
        self.corners = tuple(
            point_id for point_id in range(len(vertices)) if feature_degree[point_id] >= 3
        )
        self.patch_locators = patch_locators
        self._leaf_edges: dict[int, list[tuple[int, int]]] = {}

    def _intersects_leaf(self, edge: tuple[int, int], leaf_id: int) -> bool:
        locator = self.locator
        tol = locator.tolerance
        lower = locator.leaf_lower[leaf_id] - tol
        upper = locator.leaf_upper[leaf_id] + tol
        start, end = self.points[list(edge)]
        direction = end - start
        for axis in range(3):
            if abs(direction[axis]) <= tol:
                continue
            other = [i for i in range(3) if i != axis]
            for plane in (lower[axis], upper[axis]):
                fraction = (plane - start[axis]) / direction[axis]
                intersection = start + fraction * direction
                if (
                    -tol < fraction < 1.0 + tol
                    and np.all(intersection[other] - lower[other] > -tol)
                    and np.all(intersection[other] - upper[other] < tol)
                ):
                    return True
        return bool(np.all(start >= lower) and np.all(start <= upper))

    def _edges_in_leaf(self, leaf_id: int) -> list[tuple[int, int]]:
        if leaf_id not in self._leaf_edges:
            result: list[tuple[int, int]] = []
            seen: set[tuple[int, int]] = set()
            offset = 0
            # Native contained edges follow the contained-triangle and cyclic
            # facet-edge order, not a global nearest-edge sort.
            for locator in self.patch_locators:
                for local_id in locator._triangles_in_leaf(leaf_id):
                    triangle = self.triangles[offset + int(local_id)]
                    for i in range(3):
                        edge = tuple(sorted((triangle[i], triangle[(i + 1) % 3])))
                        if edge in seen or edge not in self.feature_patches:
                            continue
                        if self._intersects_leaf(edge, leaf_id):
                            seen.add(edge)
                            result.append(edge)
                offset += len(locator.index.triangles)
            self._leaf_edges[leaf_id] = result
        return self._leaf_edges[leaf_id]

    def nearest_edge(self, point: np.ndarray, patches: Sequence[int]) -> tuple[np.ndarray, float]:
        locator = self.locator
        lattice = np.floor((point - locator.root_lower) / locator.finest_cell_size).astype(int)
        lattice = np.clip(lattice, 0, locator.leaf_lookup.limit - 1)
        containing = locator.leaf_lookup.find(*map(int, lattice))
        size = 0.75 * locator.widths[containing] * locator.finest_cell_size
        best = point.copy()
        distance_squared = 1.0e300
        selected_patches = set(patches)
        for _iteration in range(3):
            for leaf_id in locator.leaves_in_box(point - size, point + size):
                if leaf_id < 0:
                    continue
                for edge in self._edges_in_leaf(int(leaf_id)):
                    if not self.feature_patches[edge].issubset(selected_patches):
                        continue
                    start, end = self.points[list(edge)]
                    direction = end - start
                    length_squared = _mag_squared(direction)
                    fraction = _dot(direction, point - start) / (length_squared + 1.0e-300)
                    candidate = (
                        start
                        if fraction < 0.0 or length_squared < 1.0e-300
                        else end
                        if fraction > 1.0
                        else start + direction * fraction
                    )
                    candidate_distance = _mag_squared(candidate - point)
                    if candidate_distance < distance_squared:
                        best, distance_squared = candidate, candidate_distance
            if distance_squared < 1.0e300:
                break
            size *= 2.0
        return best, distance_squared

    def nearest_corner(
        self, point: np.ndarray, patches: Sequence[int], maximum_distance_squared: float
    ) -> tuple[np.ndarray, float]:
        best = point.copy()
        distance_squared = maximum_distance_squared
        for point_id in self.corners:
            candidate = self.points[point_id]
            candidate_distance = _mag_squared(candidate - point)
            if candidate_distance < distance_squared and set(patches).issubset(
                self.point_patches[point_id]
            ):
                best, distance_squared = candidate, candidate_distance
        return best, distance_squared


def _map_assigned_patch_points(
    mesh_data: MeshData,
    face_patch_ids: np.ndarray,
    patch_locators: Sequence[_OctreeSurfaceLocator],
    *,
    selected_points: set[int] | None = None,
    feature_locator: _SurfaceFeatureLocator | None = None,
) -> dict[int, set[int]]:
    """Map partition, edge, and corner points to their assigned patches."""
    points = np.asarray(mesh_data["vertex_position"], dtype=np.float64)
    faces = [np.asarray(face, dtype=np.int32) for face in mesh_data["faces"]]
    boundary_start = int(mesh_data["n_interior_faces"])
    point_faces: dict[int, list[int]] = defaultdict(list)
    point_patches: dict[int, set[int]] = defaultdict(set)
    for local_face_id, face_id in enumerate(range(boundary_start, len(faces))):
        patch_id = int(face_patch_ids[local_face_id])
        for point_id_value in faces[face_id]:
            point_id = int(point_id_value)
            point_faces[point_id].append(face_id)
            point_patches[point_id].add(patch_id)

    partition_updates: dict[int, np.ndarray] = {}
    for point_id, patches in point_patches.items():
        if selected_points is not None and point_id not in selected_points:
            continue
        ordered_patches = tuple(sorted(patches))
        if len(ordered_patches) == 1:
            partition_updates[point_id] = patch_locators[ordered_patches[0]].nearest(
                points[point_id], max_search_iterations=5
            )[0]
    for point_id, value in partition_updates.items():
        points[point_id] = value

    # cfMesh updates face geometry after mapping partition points and only
    # then computes the mapping range used by edge/corner convergence.
    face_centres = {
        face_id: _foam_face_centre(points[faces[face_id]])
        for face_id in range(boundary_start, len(faces))
    }
    if feature_locator is None:
        feature_locator = _SurfaceFeatureLocator(patch_locators)
    feature_updates: dict[int, np.ndarray] = {}
    for point_id, patches in point_patches.items():
        if selected_points is not None and point_id not in selected_points:
            continue
        ordered_patches = tuple(sorted(patches))
        if len(ordered_patches) == 1:
            continue

        original = points[point_id]
        approximate = original.copy()
        maximum_distance_squared = 4.0 * max(
            float(_dot(face_centres[face_id] - original, face_centres[face_id] - original))
            for face_id in point_faces[point_id]
        )
        for _iteration in range(20):
            mapped = np.asarray(
                [
                    patch_locators[patch_id].nearest(approximate, max_search_iterations=5)[0]
                    for patch_id in ordered_patches
                ]
            ).mean(axis=0)
            if float(_dot(mapped - approximate, mapped - approximate)) < (
                1.0e-8 * maximum_distance_squared
            ):
                break
            approximate = mapped
        approximate_distance = _mag_squared(approximate - original)
        mapped, distance_squared = (
            feature_locator.nearest_edge(original, ordered_patches)
            if len(ordered_patches) == 2
            else feature_locator.nearest_corner(original, ordered_patches, maximum_distance_squared)
        )
        if distance_squared > 1.2 * approximate_distance:
            mapped, distance_squared = approximate, approximate_distance
        displacement = mapped - original
        if len(ordered_patches) == 2 and distance_squared > maximum_distance_squared:
            mapped = displacement * np.sqrt(maximum_distance_squared / distance_squared) + original
        feature_updates[point_id] = mapped
    for point_id, value in feature_updates.items():
        points[point_id] = value
    return point_patches


def _untangle_assigned_patch_surface(
    mesh_data: MeshData,
    face_patch_ids: np.ndarray,
    patch_locators: Sequence[_OctreeSurfaceLocator],
    global_locator: _OctreeSurfaceLocator,
    initial_inverted: set[int],
    *,
    neighbour_layers: int = 1,
) -> list[int]:
    """Untangle the active patch-assignment region with cfMesh's smoothers."""
    points = np.asarray(mesh_data["vertex_position"], dtype=np.float64)
    faces = [np.asarray(face, dtype=np.int32) for face in mesh_data["faces"]]
    boundary_start = int(mesh_data["n_interior_faces"])
    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    point_neighbours: dict[int, set[int]] = defaultdict(set)
    point_patches: dict[int, set[int]] = defaultdict(set)
    for local_face_id, face_id in enumerate(range(boundary_start, len(faces))):
        face = faces[face_id]
        patch_id = int(face_patch_ids[local_face_id])
        for point_id_value in face:
            point_patches[int(point_id_value)].add(patch_id)
        for first_value, second_value in zip(face, np.roll(face, -1), strict=True):
            first = int(first_value)
            second = int(second_value)
            edge = (min(first, second), max(first, second))
            edge_faces[edge].append(local_face_id)
            point_neighbours[first].add(second)
            point_neighbours[second].add(first)

    feature_neighbours: dict[int, list[int]] = defaultdict(list)
    for edge, incident_faces in edge_faces.items():
        patches = {int(face_patch_ids[face_id]) for face_id in incident_faces}
        if len(patches) < 2:
            continue
        first, second = edge
        feature_neighbours[first].append(second)
        feature_neighbours[second].append(first)

    feature_locator = _SurfaceFeatureLocator(patch_locators)

    # meshSurfaceOptimizer owns one partTriMesh throughout the untangling
    # loop, including restores to the least-inverted vertex positions.
    auxiliary_state: dict[str, np.ndarray] = {}

    def constrained_smooth(selected: set[int], *, remap: bool) -> None:
        edge_points = tuple(
            sorted(point_id for point_id in selected if len(point_patches[point_id]) == 2)
        )
        partition_points = tuple(
            sorted(point_id for point_id in selected if len(point_patches[point_id]) == 1)
        )
        edge_updates = {
            point_id: points[np.asarray(feature_neighbours[point_id], dtype=np.int64)].mean(axis=0)
            for point_id in edge_points
            if len(feature_neighbours[point_id]) == 2
        }
        for point_id, value in edge_updates.items():
            points[point_id] = value
        if remap and edge_points:
            _map_assigned_patch_points(
                mesh_data,
                face_patch_ids,
                patch_locators,
                selected_points=set(edge_points),
                feature_locator=feature_locator,
            )
        smooth_cfmesh_partition_points(mesh_data, partition_points, auxiliary_state=auxiliary_state)
        if remap and partition_points:
            point_indices = np.asarray(partition_points, dtype=np.int64)
            points[point_indices] = np.asarray(
                [
                    global_locator.nearest(points[point_id], max_search_iterations=100)[0]
                    for point_id in partition_points
                ]
            )

    def unconstrained_face_centre_smooth(selected: set[int], *, remap: bool) -> None:
        face_centres = {
            face_id: _foam_face_centre(points[faces[face_id]])
            for face_id in range(boundary_start, len(faces))
        }
        point_faces: dict[int, list[int]] = defaultdict(list)
        for face_id in range(boundary_start, len(faces)):
            for point_id_value in faces[face_id]:
                point_faces[int(point_id_value)].append(face_id)
        updates = {
            point_id: np.asarray([face_centres[face_id] for face_id in point_faces[point_id]]).mean(
                axis=0
            )
            for point_id in selected
        }
        for point_id, value in updates.items():
            points[point_id] = value
        if not remap:
            return
        point_indices = np.asarray(tuple(sorted(selected)), dtype=np.int64)
        points[point_indices] = np.asarray(
            [
                global_locator.nearest(points[point_id], max_search_iterations=100)[0]
                for point_id in point_indices
            ]
        )

    counts: list[int] = []
    active = set(initial_inverted)
    boundary_point_ids = np.asarray(tuple(sorted(point_patches)), dtype=np.int64)
    minimum_count = len(boundary_point_ids)
    minimum_positions = points[boundary_point_ids].copy()
    remap_vertices = True
    inverted_count = len(initial_inverted)
    for global_iteration in range(10):
        history: list[int] = []
        iterations_after_minimum = 0
        for _iteration in range(20):
            inverted = inverted_cfmesh_boundary_points(
                mesh_data, face_patch_ids, active_points=active
            )
            inverted_count = len(inverted)
            counts.append(inverted_count)
            if not inverted:
                return counts
            selected = set(inverted)
            for _layer in range(neighbour_layers):
                selected.update(
                    neighbour
                    for point_id in tuple(selected)
                    for neighbour in point_neighbours[point_id]
                )
            active = selected
            if inverted_count < minimum_count:
                minimum_count = inverted_count
                iterations_after_minimum = 0
                minimum_positions = points[boundary_point_ids].copy()
            iterations_after_minimum += 1
            history.append(inverted_count)
            history = history[-2:]
            if minimum_count not in history or iterations_after_minimum > 2:
                break
            constrained_smooth(selected, remap=remap_vertices)

        points[boundary_point_ids] = minimum_positions
        if inverted_count:
            unconstrained_face_centre_smooth(active, remap=remap_vertices)
            if global_iteration > 5:
                remap_vertices = False
    return counts


def project_cfmesh_template(
    mesh_data: MeshData,
    *,
    domain: SurfaceBounds,
    domain_patch_names: Sequence[str],
    surfaces: Sequence[TriangulatedSurface],
    surface_patch_names: Sequence[str],
) -> None:
    """Apply cfMesh's ``surfaceProjection`` stage to a template mesh in place."""
    if len(domain_patch_names) != 6:
        raise ValueError("domain_patch_names must follow xmin, xmax, ymin, ymax, zmin, zmax")
    if len(surface_patch_names) != len(surfaces):
        raise ValueError("surface_patch_names must correspond one-to-one with surfaces")
    groups: dict[str, list[np.ndarray]] = {}
    domain_triangles = _box_triangles(domain)
    for side, patch_name in enumerate(domain_patch_names):
        groups.setdefault(patch_name, []).extend(domain_triangles[2 * side : 2 * side + 2])
    for patch_name, surface in zip(surface_patch_names, surfaces, strict=True):
        groups.setdefault(patch_name, []).extend(surface.triangles)
    patch_names = tuple(sorted(groups))
    triangles = np.ascontiguousarray(np.concatenate(tuple(groups[name] for name in patch_names)))
    global_index = SurfaceIndex.build(triangles)

    points = np.asarray(mesh_data["vertex_position"], dtype=np.float64)
    faces = [np.asarray(face, dtype=np.int32) for face in mesh_data["faces"]]
    boundary_start = int(mesh_data["n_interior_faces"])
    boundary_face_ids = range(boundary_start, len(faces))
    boundary_point_ids = np.unique(
        np.concatenate([faces[face_id] for face_id in boundary_face_ids])
    )
    # The native stage performs a local nearest-surface search for each
    # boundary point.  The same geometric predicate is evaluated here in one
    # compiled VTK/index batch; retaining the standard combined surface avoids
    # a second patch-specific reference geometry and keeps this stage bounded
    # for the production reference grid.
    mapped_points, _distances, _triangle_ids = global_index.nearest_points(
        points[boundary_point_ids]
    )
    points[boundary_point_ids] = mapped_points
    untangling = untangle_cfmesh_surface(
        mesh_data,
        map_to_surface=lambda point: global_index.nearest_point(point)[0],
    )
    mesh_data["mesh_generation"]["surface_projection"] = {
        "pre_map_iterations": 1,
        "attempted_points": int(len(boundary_point_ids)),
        "accepted_points": int(len(boundary_point_ids)),
        "patch_names": patch_names,
        "mapping_method": "batched_surface_index_nearest_points",
        "untangling": untangling,
    }
    mesh_data["mesh_generation"]["workflow_checkpoint"] = "surfaceProjection"


def _patch_alignment_weight(
    max_distance_squared: float, distance_squared: float, normal_alignment: float
) -> float:
    """Evaluate cfMesh's distance/normal score without an overflowing ratio.

    Projected face centres can have zero distance to a candidate surface.
    Dividing squared distances by the tiny distance floor can then overflow,
    even though the square root is representable. Take both square roots
    before dividing; retain the floor and the common numerator so all-zero
    distances still produce zero weights and preserve candidate-order ties.
    """
    distance = np.sqrt(max(distance_squared, np.finfo(np.float64).tiny))
    return float(np.sqrt(max_distance_squared) / distance * normal_alignment)


def assign_cfmesh_patches(
    mesh_data: MeshData,
    *,
    domain: SurfaceBounds,
    domain_patch_names: Sequence[str],
    surfaces: Sequence[TriangulatedSurface],
    surface_patch_names: Sequence[str],
) -> None:
    """Assign projected boundary faces to cfMesh input-surface regions."""
    if len(domain_patch_names) != 6:
        raise ValueError("domain_patch_names must follow xmin, xmax, ymin, ymax, zmin, zmax")
    if len(surface_patch_names) != len(surfaces):
        raise ValueError("surface_patch_names must correspond one-to-one with surfaces")
    groups: dict[str, list[np.ndarray]] = {}
    domain_triangles = _box_triangles(domain)
    for side, patch_name in enumerate(domain_patch_names):
        groups.setdefault(patch_name, []).extend(domain_triangles[2 * side : 2 * side + 2])
    for patch_name, surface in zip(surface_patch_names, surfaces, strict=True):
        groups.setdefault(patch_name, []).extend(surface.triangles)
    patch_names = tuple(sorted(groups))
    triangle_groups = tuple(np.ascontiguousarray(groups[name]) for name in patch_names)
    global_index = SurfaceIndex.build(np.ascontiguousarray(np.concatenate(triangle_groups)))
    triangle_patch_ids = np.concatenate(
        [
            np.full(len(group), patch_id, dtype=np.int32)
            for patch_id, group in enumerate(triangle_groups)
        ]
    )
    octree_leaves = np.asarray(mesh_data["_cfmesh_octree_leaves"], dtype=np.int32)
    root_bounds = tuple(mesh_data["mesh_generation"]["root_box"])
    finest_cell_size = float(mesh_data["mesh_generation"]["finest_cell_size"])
    global_locator = _OctreeSurfaceLocator(
        global_index,
        root_bounds=root_bounds,  # type: ignore[arg-type]
        finest_cell_size=finest_cell_size,
        leaves=octree_leaves,
    )
    patch_locators = tuple(
        _OctreeSurfaceLocator(
            SurfaceIndex.build(group),
            root_bounds=root_bounds,  # type: ignore[arg-type]
            finest_cell_size=finest_cell_size,
            leaves=octree_leaves,
        )
        for group in triangle_groups
    )

    n_internal = int(mesh_data["n_interior_faces"])
    points = np.asarray(mesh_data["vertex_position"], dtype=np.float64)
    source_faces = [np.asarray(face, dtype=np.int32) for face in mesh_data["faces"]]
    source_owners = np.asarray(mesh_data["owners"], dtype=np.int32)
    boundary_face_ids = tuple(range(n_internal, len(source_faces)))
    face_centres = np.asarray(
        [_foam_face_centre(points[source_faces[face_id]]) for face_id in boundary_face_ids],
        dtype=np.float64,
    )
    if len(boundary_face_ids) > 10_000:
        _nearest, _distance, triangle_ids = global_index.nearest_points(face_centres)
        face_patch_ids = np.asarray(triangle_patch_ids[triangle_ids], dtype=np.int32)
    else:
        face_patch_ids = np.empty(len(boundary_face_ids), dtype=np.int32)
        for local_face_id, centre in enumerate(face_centres):
            _nearest_point, _nearest_distance, triangle_id = global_locator.nearest_triangle(
                centre, max_search_iterations=100
            )
            face_patch_ids[local_face_id] = int(triangle_patch_ids[triangle_id])

    initial_patch_counts = {
        patch_name: int(np.count_nonzero(face_patch_ids == patch_id))
        for patch_id, patch_name in enumerate(patch_names)
    }

    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for local_face_id, face_id in enumerate(boundary_face_ids):
        face = source_faces[face_id]
        for first_value, second_value in zip(face, np.roll(face, -1), strict=True):
            first_id = int(first_value)
            second_id = int(second_value)
            edge = (min(first_id, second_id), max(first_id, second_id))
            edge_faces[edge].append(local_face_id)

    normal_alignment_changes = 0
    normal_alignment_history: list[list[int]] = []
    for _iteration in range(5):
        updated = face_patch_ids.copy()
        changed = 0
        changed_faces: list[int] = []
        for local_face_id, face_id in enumerate(boundary_face_ids):
            face = source_faces[face_id]
            neighbour_patches: list[int] = []
            for first_value, second_value in zip(face, np.roll(face, -1), strict=True):
                first_id = int(first_value)
                second_id = int(second_value)
                edge = (min(first_id, second_id), max(first_id, second_id))
                for neighbour_face in edge_faces[edge]:
                    if neighbour_face == local_face_id:
                        continue
                    patch_id = int(face_patch_ids[neighbour_face])
                    if patch_id not in neighbour_patches:
                        neighbour_patches.append(patch_id)
            if len(neighbour_patches) <= 1:
                continue
            centre = _foam_face_centre(points[face])
            face_normal = _face_area_vector_with_centre(points[face], centre)
            face_normal /= max(float(np.linalg.norm(face_normal)), np.finfo(np.float64).tiny)
            candidates: list[tuple[int, float, float]] = []
            for patch_id in neighbour_patches:
                _mapped, distance, triangle_id = patch_locators[patch_id].nearest_triangle(
                    centre,
                    max_search_iterations=5,
                )
                triangle = triangle_groups[patch_id][triangle_id]
                triangle_normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
                triangle_normal /= max(
                    float(np.linalg.norm(triangle_normal)), np.finfo(np.float64).tiny
                )
                candidates.append(
                    (
                        patch_id,
                        distance * distance,
                        abs(float(_dot(triangle_normal, face_normal))),
                    )
                )
            if not candidates:
                continue
            max_distance_squared = max(item[1] for item in candidates)
            best_patch = max(
                candidates,
                key=lambda item: _patch_alignment_weight(max_distance_squared, item[1], item[2]),
            )[0]
            if best_patch != int(face_patch_ids[local_face_id]):
                updated[local_face_id] = best_patch
                changed += 1
                changed_faces.append(local_face_id)
        face_patch_ids = updated
        normal_alignment_changes += changed
        normal_alignment_history.append(changed_faces)
        if changed == 0:
            break

    normal_alignment_patch_counts = {
        patch_name: int(np.count_nonzero(face_patch_ids == patch_id))
        for patch_id, patch_name in enumerate(patch_names)
    }

    geometry_patch_changes = 0
    geometry_patch_history: list[list[int]] = []
    inverted_point_counts: list[int] = []
    untangling_iteration_counts: list[list[int]] = []

    def nearest_to_two_patches(value: np.ndarray, patches: tuple[int, int]) -> np.ndarray:
        current = value.copy()
        for _iteration in range(40):
            mapped = np.asarray(
                [
                    patch_locators[patch_id].nearest(current, max_search_iterations=5)[0]
                    for patch_id in patches
                ]
            ).mean(axis=0)
            distance_squared = float(_dot(mapped - value, mapped - value))
            if float(_dot(mapped - current, mapped - current)) < 1.0e-4 * distance_squared:
                return mapped
            current = mapped
        return current

    active_geometry_points: set[int] | None = None
    for _geometry_iteration in range(3):
        _map_assigned_patch_points(
            mesh_data,
            face_patch_ids,
            patch_locators,
            selected_points=active_geometry_points,
        )
        inverted_points = inverted_cfmesh_boundary_points(
            mesh_data,
            face_patch_ids,
            active_points=active_geometry_points,
        )
        inverted_point_counts.append(len(inverted_points))
        if not inverted_points:
            break
        active_geometry_points = set(inverted_points)
        untangling_iteration_counts.append(
            _untangle_assigned_patch_surface(
                mesh_data,
                face_patch_ids,
                patch_locators,
                global_locator,
                inverted_points,
            )
        )
        updated = face_patch_ids.copy()
        changed = 0
        changed_faces = []
        edge_metric_cache: dict[tuple[tuple[int, int], int, int], float] = {}

        def edge_metric(
            edge: tuple[int, int],
            first_patch: int,
            second_patch: int,
            cache: dict[tuple[tuple[int, int], int, int], float] = edge_metric_cache,
        ) -> float:
            key = (edge, min(first_patch, second_patch), max(first_patch, second_patch))
            cached = cache.get(key)
            if cached is not None:
                return cached
            start = points[edge[0]]
            end = points[edge[1]]
            mapped_start = nearest_to_two_patches(start, key[1:])
            mapped_end = nearest_to_two_patches(end, key[1:])
            edge_vector = end - start
            edge_length = float(np.linalg.norm(edge_vector))
            mapped_vector = mapped_end - mapped_start
            mapped_length = float(np.linalg.norm(mapped_vector))
            cosine = float(
                _dot(edge_vector, mapped_vector)
                / max(edge_length * mapped_length, np.finfo(np.float64).tiny)
            )
            angle = float(np.arccos(np.clip(cosine, -1.0, 1.0)))
            value = (
                0.5
                * (
                    float(np.linalg.norm(mapped_start - start))
                    + float(np.linalg.norm(mapped_end - end))
                )
                + edge_length * angle
            )
            cache[key] = value
            return value

        for local_face_id, face_id in enumerate(boundary_face_ids):
            face = source_faces[face_id]
            if not inverted_points.intersection(map(int, face)):
                continue
            neighbour_patches = []
            face_edges: list[tuple[int, int]] = []
            for first_value, second_value in zip(face, np.roll(face, -1), strict=True):
                first_id = int(first_value)
                second_id = int(second_value)
                edge = (min(first_id, second_id), max(first_id, second_id))
                face_edges.append(edge)
                neighbours = [
                    candidate for candidate in edge_faces[edge] if candidate != local_face_id
                ]
                neighbour_patches.append(
                    int(face_patch_ids[neighbours[0]])
                    if neighbours
                    else int(face_patch_ids[local_face_id])
                )
            candidate_patches = tuple(dict.fromkeys(neighbour_patches))
            if not candidate_patches:
                continue
            energies = []
            for candidate_patch in candidate_patches:
                energy = sum(
                    edge_metric(edge, candidate_patch, neighbour_patch)
                    for edge, neighbour_patch in zip(face_edges, neighbour_patches, strict=True)
                    if neighbour_patch != candidate_patch
                )
                energies.append(energy)
            best_patch = candidate_patches[int(np.argmin(energies))]
            if best_patch != int(face_patch_ids[local_face_id]):
                updated[local_face_id] = best_patch
                changed += 1
                changed_faces.append(local_face_id)
        face_patch_ids = updated
        geometry_patch_changes += changed
        geometry_patch_history.append(changed_faces)
        if changed == 0:
            break

    grouped_faces: list[list[np.ndarray]] = [[] for _name in patch_names]
    grouped_owners: list[list[int]] = [[] for _name in patch_names]
    for local_face_id, face_id in enumerate(boundary_face_ids):
        patch_id = int(face_patch_ids[local_face_id])
        grouped_faces[patch_id].append(source_faces[face_id])
        grouped_owners[patch_id].append(int(source_owners[face_id]))

    faces = source_faces[:n_internal]
    owners = list(map(int, source_owners[:n_internal]))
    boundary: list[BoundaryPatchData] = []
    start_face = n_internal
    for patch_name, patch_faces, patch_owners in zip(
        patch_names, grouped_faces, grouped_owners, strict=True
    ):
        faces.extend(patch_faces)
        owners.extend(patch_owners)
        boundary.append(
            {
                "name": patch_name,
                "start_face": start_face,
                "n_faces": len(patch_faces),
                "type": "empty",
            }
        )
        start_face += len(patch_faces)
    face_widths = {len(face) for face in faces}
    mesh_data["faces"] = (
        np.ascontiguousarray(faces, dtype=np.int32)
        if len(face_widths) == 1
        else [np.ascontiguousarray(face, dtype=np.int32) for face in faces]
    )
    mesh_data["owners"] = np.ascontiguousarray(owners, dtype=np.int32)
    mesh_data["boundary"] = boundary
    if "_cfmesh_cell_face_order" in mesh_data:
        face_by_signature = {
            tuple(sorted(map(int, face))): face_id for face_id, face in enumerate(faces)
        }
        mesh_data["_cfmesh_cell_face_order"] = [
            [face_by_signature[tuple(sorted(map(int, source_faces[face_id])))] for face_id in cell]
            for cell in mesh_data["_cfmesh_cell_face_order"]
        ]
    mesh_data["mesh_generation"]["workflow_checkpoint"] = "patchAssignment"
    mesh_data["mesh_generation"]["patch_assignment"] = {
        "patch_names": patch_names,
        "normal_alignment_changes": normal_alignment_changes,
        "normal_alignment_history": normal_alignment_history,
        "initial_patch_counts": initial_patch_counts,
        "normal_alignment_patch_counts": normal_alignment_patch_counts,
        "geometry_patch_changes": geometry_patch_changes,
        "geometry_patch_history": geometry_patch_history,
        "inverted_point_counts": inverted_point_counts,
        "untangling_iteration_counts": untangling_iteration_counts,
    }


def remap_cfmesh_patch_points(
    mesh_data: MeshData,
    *,
    domain: SurfaceBounds,
    domain_patch_names: Sequence[str],
    surfaces: Sequence[TriangulatedSurface],
    surface_patch_names: Sequence[str],
) -> tuple[Callable[[Sequence[int]], None], Callable[[], list[int]]]:
    """Map all boundary points and return edge-map and untangle callbacks."""
    if len(domain_patch_names) != 6:
        raise ValueError("domain_patch_names must follow xmin, xmax, ymin, ymax, zmin, zmax")
    if len(surface_patch_names) != len(surfaces):
        raise ValueError("surface_patch_names must correspond one-to-one with surfaces")
    groups: dict[str, list[np.ndarray]] = {}
    domain_triangles = _box_triangles(domain)
    for side, patch_name in enumerate(domain_patch_names):
        groups.setdefault(patch_name, []).extend(domain_triangles[2 * side : 2 * side + 2])
    for patch_name, surface in zip(surface_patch_names, surfaces, strict=True):
        groups.setdefault(patch_name, []).extend(surface.triangles)
    patch_names = tuple(sorted(groups))
    mesh_patch_names = tuple(str(patch["name"]) for patch in mesh_data["boundary"])
    if mesh_patch_names != patch_names:
        raise ValueError("cfMesh edge remapping requires boundary patches in assigned-region order")
    root_bounds = tuple(mesh_data["mesh_generation"]["root_box"])
    finest_cell_size = float(mesh_data["mesh_generation"]["finest_cell_size"])
    leaves = np.asarray(mesh_data["_cfmesh_octree_leaves"], dtype=np.int32)
    patch_locators = tuple(
        _OctreeSurfaceLocator(
            SurfaceIndex.build(np.ascontiguousarray(groups[name])),
            root_bounds=root_bounds,  # type: ignore[arg-type]
            finest_cell_size=finest_cell_size,
            leaves=leaves,
        )
        for name in patch_names
    )
    global_locator = _OctreeSurfaceLocator(
        SurfaceIndex.build(
            np.ascontiguousarray(
                np.concatenate(tuple(np.ascontiguousarray(groups[name]) for name in patch_names))
            )
        ),
        root_bounds=root_bounds,  # type: ignore[arg-type]
        finest_cell_size=finest_cell_size,
        leaves=leaves,
    )

    def current_face_patch_ids() -> np.ndarray:
        return np.concatenate(
            [
                np.full(int(patch["n_faces"]), patch_id, dtype=np.int32)
                for patch_id, patch in enumerate(mesh_data["boundary"])
            ]
        )

    face_patch_ids = current_face_patch_ids()
    point_patches = _map_assigned_patch_points(mesh_data, face_patch_ids, patch_locators)

    def remap_selected(point_ids: Sequence[int]) -> None:
        _map_assigned_patch_points(
            mesh_data,
            current_face_patch_ids(),
            patch_locators,
            selected_points=set(map(int, point_ids)),
        )

    def untangle_surface() -> list[int]:
        active_face_patch_ids = current_face_patch_ids()
        initial = inverted_cfmesh_boundary_points(mesh_data, active_face_patch_ids)
        return _untangle_assigned_patch_surface(
            mesh_data,
            active_face_patch_ids,
            patch_locators,
            global_locator,
            initial,
            neighbour_layers=0,
        )

    mesh_data["mesh_generation"]["cfmesh_patch_remapping"] = {
        "mapped_points": len(point_patches),
        "partition_points": sum(len(patches) == 1 for patches in point_patches.values()),
        "edge_points": sum(len(patches) == 2 for patches in point_patches.values()),
        "corner_points": sum(len(patches) > 2 for patches in point_patches.values()),
    }
    return remap_selected, untangle_surface


__all__ = [
    "assign_cfmesh_patches",
    "build_cfmesh_template",
    "project_cfmesh_template",
    "remap_cfmesh_patch_points",
]
