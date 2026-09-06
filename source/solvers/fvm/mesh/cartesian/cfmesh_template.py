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
from typing import Any, cast

import numpy as np

from ..surface_classification import (
    SurfaceIndex,
    closest_point_on_triangles,
    triangle_box_overlap,
)
from ..triangulated_surface import SurfaceBounds, TriangulatedSurface
from .cfmesh_automatic_refinement import AutomaticSurface, automatic_refinement
from .cfmesh_octree import (
    LeafLookup,
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
from .octree import CartesianOctree


def _additional_level(max_cell_size: float, requested: float) -> int:
    """Return cfMesh's first dyadic level whose size does not exceed a request."""
    level = 0
    while max_cell_size / (2**level) > requested:
        level += 1
    return level


def _quad_signature(values: Sequence[int] | np.ndarray) -> tuple[int, int, int, int]:
    """Return the sorted fixed-width key for one Cartesian quad."""
    ordered = sorted(map(int, values))
    if len(ordered) != 4:
        raise ValueError(f"Expected four Cartesian face vertices, received {len(ordered)}")
    return ordered[0], ordered[1], ordered[2], ordered[3]


def _root_cube(domain: SurfaceBounds, max_cell_size: float) -> tuple[SurfaceBounds, int]:
    """Reproduce ``setRootCubeSizeAndRefParameters`` for the outer surface.

    cfMesh first creates a cube 1.5 times wider than the surface bounds, adds
    half a requested cell to that sizing calculation, and then chooses a
    power-of-two cube whose level cell size is exactly ``max_cell_size``.
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
    bounds = tuple(
        float(value) for axis in range(3) for value in (root_lower[axis], root_upper[axis])
    )
    return bounds, global_level  # type: ignore[return-value]


def _box_surface_intersects(
    lower: np.ndarray,
    upper: np.ndarray,
    bounds: SurfaceBounds,
) -> bool:
    """Return whether a closed AABB touches any triangle-bearing box plane."""
    surface_lower = np.asarray(bounds[::2], dtype=np.float64)
    surface_upper = np.asarray(bounds[1::2], dtype=np.float64)
    return _box_surface_intersects_arrays(lower, upper, surface_lower, surface_upper)


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


class _TemplateTopologyExtractor(CartesianOctree):
    """Use the proven native face extractor on already-classified octree leaves."""

    def __init__(self, root_bounds: SurfaceBounds) -> None:
        # This object needs only the attributes consumed by
        # ``CartesianOctree._extract_topology``.  Skipping the normal
        # constructor is intentional: that constructor fits the requested
        # fluid box, whereas this stage operates in cfMesh's larger root cube.
        self.domain = root_bounds
        self.merge_outer_patch = "defaultFaces"
        self.preserve_outer_patches: tuple[str, ...] = ()
        self.surface = cast(TriangulatedSurface, object())
        self.wall_patch_name = "defaultFaces"
        self._wall_patch_type = "patch"


def _extract_mesh(
    root_bounds: SurfaceBounds,
    root_size: float,
    leaves: np.ndarray,
    octree_leaves: np.ndarray,
    max_level: int,
    global_level: int,
    boundary_level: int,
) -> dict[str, Any]:
    limits = (2**max_level, 2**max_level, 2**max_level)
    extractor = _TemplateTopologyExtractor(root_bounds)
    encoded_faces, owners, neighbours, boundary, levels = extractor._extract_topology(
        leaves, max_level, limits
    )

    # cfMesh creates octree faces while visiting every Morton-ordered leaf,
    # including excluded surface-data leaves.  A fine excluded leaf therefore
    # emits the boundary face towards a coarser fluid leaf at its own position
    # in that traversal.  Reproduce that ordering: its finite-iteration
    # surface optimiser constructs local triangle simplexes in face order.
    level_maps: list[dict[int, int]] = [{} for _ in range(max_level + 1)]
    cell_stride_x = limits[0]
    cell_stride_y = limits[1]
    for leaf_id, (x0, y0, z0, _width, level, _kind) in enumerate(octree_leaves):
        key = int(x0) + cell_stride_x * (int(y0) + cell_stride_y * int(z0))
        level_maps[int(level)][key] = leaf_id

    def find_leaf(x: int, y: int, z: int) -> int:
        if x < 0 or y < 0 or z < 0 or x >= limits[0] or y >= limits[1] or z >= limits[2]:
            return -1
        for level in range(max_level, -1, -1):
            width = 2 ** (max_level - level)
            origin_x = (x // width) * width
            origin_y = (y // width) * width
            origin_z = (z // width) * width
            key = origin_x + cell_stride_x * (origin_y + cell_stride_y * origin_z)
            found = level_maps[level].get(key)
            if found is not None:
                return found
        return -1

    def samples(lower: int, width: int) -> tuple[int, ...]:
        if width == 1:
            return (lower,)
        return (lower + width // 4, lower + (3 * width) // 4)

    mesh_cell = 1
    surface_data_cell = 2
    leaf_cell_ids = np.full(len(octree_leaves), -1, dtype=np.int32)
    next_cell_id = 0
    for leaf_id, record in enumerate(octree_leaves):
        if int(record[5]) == mesh_cell:
            leaf_cell_ids[leaf_id] = next_cell_id
            next_cell_id += 1
    ordered_octree_faces: list[tuple[tuple[int, int, int, int], bool, tuple[int, ...]]] = []
    node_leaves = set(map(int, np.flatnonzero(leaf_cell_ids >= 0)))
    for leaf_id, record in enumerate(octree_leaves):
        x0, y0, z0, width, level, kind = map(int, record)
        x1, y1, z1 = x0 + width, y0 + width, z0 + width
        origins = (x0, y0, z0)
        ends = (x1, y1, z1)
        tangential = ((y0, y1, z0, z1), (x0, x1, z0, z1), (x0, x1, y0, y1))
        for axis in range(3):
            a0, a1, b0, b1 = tangential[axis]
            for positive in (False, True):
                query = [x0, y0, z0]
                query[axis] = ends[axis] if positive else origins[axis] - 1
                neighbour_ids: set[int] = set()
                for a in samples(a0, width):
                    for b in samples(b0, width):
                        if axis == 0:
                            candidate = find_leaf(query[0], a, b)
                        elif axis == 1:
                            candidate = find_leaf(a, query[1], b)
                        else:
                            candidate = find_leaf(a, b, query[2])
                        if candidate >= 0:
                            neighbour_ids.add(candidate)
                if kind == mesh_cell:
                    # findUsedBoxes marks face-neighbours BOUNDARY; diagonal
                    # surface-data neighbours alone have no nodeLabels row.
                    node_leaves.update(neighbour_ids)
                if len(neighbour_ids) != 1:
                    continue
                neighbour_id = next(iter(neighbour_ids))
                neighbour_level = int(octree_leaves[neighbour_id, 4])
                neighbour_kind = int(octree_leaves[neighbour_id, 5])
                emitted_boundary = kind == mesh_cell and neighbour_kind == surface_data_cell
                emitted_boundary = emitted_boundary or (
                    kind == surface_data_cell
                    and neighbour_kind == mesh_cell
                    and neighbour_level < level
                )
                emitted_internal = (
                    kind == mesh_cell
                    and neighbour_kind == mesh_cell
                    and (neighbour_id > leaf_id or neighbour_level < level)
                )
                if not emitted_boundary and not emitted_internal:
                    continue
                coordinate = ends[axis] if positive else origins[axis]
                codes = extractor._face_codes(
                    axis,
                    coordinate,
                    a0,
                    a1,
                    b0,
                    b1,
                    (limits[0] + 1, limits[1] + 1),
                    positive,
                )
                attached_cells: tuple[int, ...]
                if emitted_internal:
                    attached_cells = (
                        int(leaf_cell_ids[leaf_id]),
                        int(leaf_cell_ids[neighbour_id]),
                    )
                elif kind == mesh_cell:
                    attached_cells = (int(leaf_cell_ids[leaf_id]),)
                else:
                    attached_cells = (int(leaf_cell_ids[neighbour_id]),)
                ordered_codes = sorted(codes)
                signature = (
                    ordered_codes[0],
                    ordered_codes[1],
                    ordered_codes[2],
                    ordered_codes[3],
                )
                ordered_octree_faces.append((signature, emitted_internal, attached_cells))

    # ``reorderBoundaryFaces`` swaps misplaced boundary and internal faces in
    # pairs; it is intentionally not a stable partition.  Simulating those
    # swaps recovers the boundary order seen by all later cfMesh stages.
    expected_internal = len(neighbours)
    reordered_octree_faces = ordered_octree_faces.copy()
    internal_to_change = [
        face_id
        for face_id, (_key, internal, _cells) in enumerate(
            reordered_octree_faces[:expected_internal]
        )
        if not internal
    ]
    boundary_to_change = [
        face_id
        for face_id, (_key, internal, _cells) in enumerate(
            reordered_octree_faces[expected_internal:], start=expected_internal
        )
        if internal
    ]
    if len(internal_to_change) != len(boundary_to_change):
        raise RuntimeError("cfMesh octree face-order reconstruction is inconsistent")
    for internal_face, boundary_face in zip(internal_to_change, boundary_to_change, strict=True):
        reordered_octree_faces[internal_face], reordered_octree_faces[boundary_face] = (
            reordered_octree_faces[boundary_face],
            reordered_octree_faces[internal_face],
        )
    final_boundary_keys = [
        key for key, internal, _cells in reordered_octree_faces[expected_internal:] if not internal
    ]
    face_order = {key: order for order, key in enumerate(final_boundary_keys)}

    n_internal = len(neighbours)
    matched_boundary_order = 0
    if len(encoded_faces) > n_internal:
        matched_boundary_order = sum(
            _quad_signature(face) in face_order for face in encoded_faces[n_internal:]
        )
        boundary_order = np.asarray(
            [
                face_order.get(_quad_signature(face), len(face_order) + face_id)
                for face_id, face in enumerate(encoded_faces[n_internal:])
            ],
            dtype=np.int64,
        )
        permutation = np.argsort(boundary_order, kind="stable")
        encoded_faces[n_internal:] = encoded_faces[n_internal:][permutation]
        owners[n_internal:] = owners[n_internal:][permutation]

    face_by_octree_signature = {
        _quad_signature(face): face_id for face_id, face in enumerate(encoded_faces)
    }
    cfmesh_cell_face_order: list[list[int]] = [[] for _cell in leaves]
    for signature, _internal, attached_cells in ordered_octree_faces:
        face_id = face_by_octree_signature.get(signature)
        if face_id is None:
            raise RuntimeError("cfMesh cell-face ordering references an absent octree face")
        for cell_id in attached_cells:
            cfmesh_cell_face_order[cell_id].append(face_id)
    if any(not face_ids for face_ids in cfmesh_cell_face_order):
        raise RuntimeError("cfMesh cell-face ordering left an empty mesh cell")

    sx = limits[0] + 1
    sy = limits[1] + 1
    available_codes = set(map(int, np.unique(encoded_faces)))

    def decode(code: int) -> tuple[int, int, int]:
        """Decode one lattice code without allocating a temporary array."""
        x = int(code % sx)
        yz = int(code // sx)
        return x, int(yz % sy), int(yz // sy)

    # The extractor retains existing hanging points around a coarse polygon's
    # perimeter. Most coarse/fine interfaces are split into fine quads, but a
    # face shared by two coarse cells can therefore become an 8-node polygon.
    # Its adjacency stays one face; dropping those collinear nodes changes the
    # mandatory face-valence invariant.
    expanded_encoded_faces: list[np.ndarray] = []
    expanded_face_count = 0
    available_code_set = set(map(int, available_codes))
    for encoded_face in encoded_faces:
        expanded: list[int] = []
        for first_value, second_value in zip(encoded_face, np.roll(encoded_face, -1), strict=True):
            first = int(first_value)
            second = int(second_value)
            expanded.append(first)
            first_x, first_y, first_z = decode(first)
            second_x, second_y, second_z = decode(second)
            delta_x = second_x - first_x
            delta_y = second_y - first_y
            delta_z = second_z - first_z
            length = max(abs(delta_x), abs(delta_y), abs(delta_z))
            if length <= 1:
                continue
            step_x = delta_x // length
            step_y = delta_y // length
            step_z = delta_z // length
            for offset in range(1, length):
                candidate = int(
                    (first_x + offset * step_x)
                    + sx * (first_y + offset * step_y + sy * (first_z + offset * step_z))
                )
                if candidate in available_code_set:
                    expanded.append(candidate)
        if len(expanded) > len(encoded_face):
            expanded_face_count += 1
        expanded_encoded_faces.append(np.asarray(expanded, dtype=np.int64))

    point_codes = np.unique(np.concatenate(expanded_encoded_faces))
    flat_face_codes = np.concatenate(expanded_encoded_faces)
    flat_face_indices = np.searchsorted(point_codes, flat_face_codes).astype(np.int32)
    face_offsets = np.cumsum(
        np.asarray([0, *(len(face) for face in expanded_encoded_faces)], dtype=np.int64)
    )
    indexed_faces = [
        flat_face_indices[start:stop]
        for start, stop in zip(face_offsets[:-1], face_offsets[1:], strict=True)
    ]
    widths = {len(face) for face in indexed_faces}
    faces: list[np.ndarray] | np.ndarray = (
        np.ascontiguousarray(indexed_faces, dtype=np.int32) if len(widths) == 1 else indexed_faces
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
    participating = np.zeros(len(octree_leaves), dtype=bool)
    participating[np.fromiter(node_leaves, dtype=np.int64)] = True
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
    ordered_codes = _unique_codes[np.argsort(first_positions, kind="stable")]
    ordered_labels = np.searchsorted(point_codes, ordered_codes)
    ordered_valid = (ordered_labels < len(point_codes)) & (
        point_codes[np.minimum(ordered_labels, len(point_codes) - 1)] == ordered_codes
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
    x0, y0, z0, width = (leaves[:, index].astype(np.int64) for index in range(4))
    x1, y1, z1 = x0 + width, y0 + width, z0 + width

    def encode(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        return x + sx * (y + sy * z)

    cell_codes = np.column_stack(
        (
            encode(x0, y0, z0),
            encode(x1, y0, z0),
            encode(x1, y1, z0),
            encode(x0, y1, z0),
            encode(x0, y0, z1),
            encode(x1, y0, z1),
            encode(x1, y1, z1),
            encode(x0, y1, z1),
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
            "octree_boundary_face_order_entries": len(face_order),
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
) -> dict[str, Any]:
    """Build the native equivalent of cfMesh's ``templateGeneration`` stage."""
    root_bounds, global_level = _root_cube(domain, max_cell_size)
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
    max_level = max(
        [
            boundary_level,
            *patch_levels.values(),
            *(
                global_level + object_additional_level(max_cell_size, request.cell_size)
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

    visit(0, 0, 0, lattice_width, 0)
    if box_refinements or patch_refinements or needs_automatic:

        def classify(x: int, y: int, z: int, width: int, level: int):
            lower = root_lower + finest_size * np.asarray((x, y, z), dtype=np.float64)
            upper = root_lower + finest_size * np.asarray(
                (x + width, y + width, z + width), dtype=np.float64
            )
            if intersects_input_surface(lower, upper):
                kind = surface_data_cell
            else:
                centre = 0.5 * (lower + upper)
                in_domain = bool(np.all(centre > domain_lower) and np.all(centre < domain_upper))
                in_object = in_domain and any(
                    bool(index.is_inside(centre[None, :])[0]) for index in surface_indices
                )
                kind = mesh_cell if in_domain and not in_object else other_cell
            return (x, y, z, width, level, kind)

        octree_leaves = balance_leaves(octree_leaves, max_level, classify)
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
            octree_leaves, automatic_diagnostics = automatic_refinement(
                octree_leaves,
                automatic_surface,
                root_lower=root_lower,
                root_size=root_size,
                max_level=max_level,
                automatic_level=automatic_level,
                classify=classify,
            )
        octree_leaves = refine_objects(
            octree_leaves,
            box_refinements,
            root_lower=root_lower,
            root_size=root_size,
            max_cell_size=max_cell_size,
            global_level=global_level,
            max_level=max_level,
            classify=classify,
        )
        octree_leaves, near_data_refined = refine_near_data(octree_leaves, max_level, classify)
        leaves = [record[:5] for record in octree_leaves if record[5] == mesh_cell]
        if not leaves:
            raise ValueError("cfMesh template classification removed every fluid leaf")
        mesh_data = _extract_mesh(
            root_bounds,
            root_size,
            np.ascontiguousarray(leaves, dtype=np.int32),
            np.ascontiguousarray(octree_leaves, dtype=np.int32),
            max_level,
            global_level,
            boundary_level,
        )
        mesh_data["_cfmesh_octree_leaves"] = np.ascontiguousarray(octree_leaves, dtype=np.int32)
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
        self._nearest_cache: dict[tuple[int, bool, bytes], tuple[np.ndarray, float, int]] = {}

    def leaves_in_box(self, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        """Return octree leaves with positive-volume overlap with a box."""
        lattice_limit = self.leaf_lookup.limit
        first = np.floor((lower - self.root_lower) / self.finest_cell_size).astype(np.int64)
        last = np.ceil((upper - self.root_lower) / self.finest_cell_size).astype(np.int64) - 1
        first = np.clip(first, 0, lattice_limit - 1)
        last = np.clip(last, 0, lattice_limit - 1)
        return self.leaf_lookup.in_box(first, last)

    def has_triangles_in_leaves(self, leaf_ids: np.ndarray) -> bool:
        """Return whether any selected leaf contains this locator's surface."""
        return any(
            len(self._triangles_in_leaf(int(leaf_id))) for leaf_id in leaf_ids[leaf_ids >= 0]
        )

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
        prefer_last_tie: bool = False,
    ) -> tuple[np.ndarray, float, int]:
        """Stop at the first octree neighborhood containing any triangles."""
        value = np.asarray(point, dtype=np.float64)
        cache_key = (max_search_iterations, prefer_last_tie, value.tobytes())
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
                if prefer_last_tie:
                    tied = np.flatnonzero(distance_squared == distance_squared[best])
                    best = int(tied[-1])
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
        if prefer_last_tie:
            tied = np.flatnonzero(distance_squared == distance_squared[best])
            best = int(tied[-1])
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
    mesh_data: dict[str, Any],
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
    mesh_data: dict[str, Any],
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
    mesh_data: dict[str, Any],
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
    # compiled VTK/index batch; retaining the canonical combined surface avoids
    # a second patch-specific geometry authority and keeps this stage bounded
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


def assign_cfmesh_patches(
    mesh_data: dict[str, Any],
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
            _nearest, _distance, triangle_id = global_locator.nearest_triangle(
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
                key=lambda item: (
                    np.sqrt(max_distance_squared / max(item[1], np.finfo(np.float64).tiny))
                    * item[2]
                ),
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
        changed_faces: list[int] = []
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
            neighbour_patches: list[int] = []
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
    boundary: list[dict[str, Any]] = []
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
    mesh_data: dict[str, Any],
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
