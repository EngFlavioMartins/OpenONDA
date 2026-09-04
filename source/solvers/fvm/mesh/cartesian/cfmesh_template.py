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
from .cfmesh_surface_optimisation import (
    inverted_cfmesh_boundary_points,
    smooth_cfmesh_partition_points,
    untangle_cfmesh_surface,
)
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
    while size / (2**global_level) >= max_cell_size * (1.0 - 1.0e-14):
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

    def decode(code: int) -> np.ndarray:
        x = code % sx
        yz = code // sx
        return np.asarray((x, yz % sy, yz // sy), dtype=np.int64)

    # The extractor retains existing hanging points around a coarse polygon's
    # perimeter. Most coarse/fine interfaces are split into fine quads, but a
    # face shared by two coarse cells can therefore become an 8-node polygon.
    # Its adjacency stays one face; dropping those collinear nodes changes the
    # mandatory face-valence invariant.
    expanded_encoded_faces: list[np.ndarray] = []
    expanded_face_count = 0
    for encoded_face in encoded_faces:
        expanded: list[int] = []
        for first_value, second_value in zip(encoded_face, np.roll(encoded_face, -1), strict=True):
            first = int(first_value)
            second = int(second_value)
            expanded.append(first)
            first_coordinate = decode(first)
            delta = decode(second) - first_coordinate
            length = int(np.max(np.abs(delta)))
            if length <= 1:
                continue
            step = delta // length
            for offset in range(1, length):
                coordinate = first_coordinate + offset * step
                candidate = int(coordinate[0] + sx * (coordinate[1] + sy * coordinate[2]))
                if candidate in available_codes:
                    expanded.append(candidate)
        if len(expanded) > len(encoded_face):
            expanded_face_count += 1
        expanded_encoded_faces.append(np.asarray(expanded, dtype=np.int64))

    point_codes = np.unique(np.concatenate(expanded_encoded_faces))
    indexed_faces = [
        np.searchsorted(point_codes, face).astype(np.int32) for face in expanded_encoded_faces
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
) -> dict[str, Any]:
    """Build the native equivalent of cfMesh's ``templateGeneration`` stage."""
    root_bounds, global_level = _root_cube(domain, max_cell_size)
    boundary_level = global_level + _additional_level(max_cell_size, boundary_cell_size)
    max_level = boundary_level
    root_lower = np.asarray(root_bounds[::2], dtype=np.float64)
    root_upper = np.asarray(root_bounds[1::2], dtype=np.float64)
    root_size = float(root_upper[0] - root_lower[0])
    finest_size = root_size / (2**max_level)
    lattice_width = 2**max_level
    domain_lower = np.asarray(domain[::2], dtype=np.float64)
    domain_upper = np.asarray(domain[1::2], dtype=np.float64)
    surface_indices = tuple(SurfaceIndex.build(surface.triangles) for surface in surfaces)

    def intersects_input_surface(lower: np.ndarray, upper: np.ndarray) -> bool:
        if _box_surface_intersects(lower, upper, domain):
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
        intersects = intersects_input_surface(lower, upper)
        target_level = global_level
        if level >= global_level and intersects:
            target_level = boundary_level
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
        refine_near_data = False
        if kind == mesh_cell and level < max_level and len(data_boxes):
            lower = np.asarray((x0, y0, z0), dtype=np.int32)
            upper = lower + width
            data_lower = data_boxes[:, :3]
            data_upper = data_lower + data_boxes[:, 3, None]
            touches = np.all((data_lower <= upper) & (data_upper >= lower), axis=1)
            refine_near_data = bool(np.any(touches))
        if not refine_near_data:
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
    if len(coordinates) == 3:
        return (coordinates[0] + coordinates[1] + coordinates[2]) / 3.0
    centre = np.zeros(3, dtype=np.float64)
    for coordinate in coordinates:
        centre += coordinate
    centre /= len(coordinates)
    area_sum = 0.0
    weighted = np.zeros(3, dtype=np.float64)
    for position, coordinate in enumerate(coordinates):
        following = coordinates[(position + 1) % len(coordinates)]
        cross = np.cross(coordinate - centre, following - centre)
        twice_area = float(np.sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]))
        area_sum += twice_area
        weighted += twice_area * (coordinate + following + centre)
    if area_sum <= np.finfo(np.float64).tiny:
        return centre
    return weighted / (3.0 * area_sum)


def _face_area_vector_with_centre(coordinates: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """OpenFOAM polygon area vector about an explicitly supplied centre."""
    return 0.5 * np.cross(coordinates - centre, np.roll(coordinates, -1, axis=0) - centre).sum(
        axis=0
    )


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
        fraction = np.clip(fraction, 0.0, 1.0)
        result[before_u] = a[before_u] + fraction[:, None] * direction
    if np.any(before_v):
        direction = vector_0[before_v]
        fraction = row_dot(projected[before_v] - a[before_v], direction) / (
            row_dot(direction, direction) + 1.0e-300
        )
        fraction = np.clip(fraction, 0.0, 1.0)
        result[before_v] = a[before_v] + fraction[:, None] * direction
    if np.any(opposite):
        c = triangles[opposite, 2]
        direction = triangles[opposite, 1] - c
        fraction = row_dot(projected[opposite] - c, direction) / (
            row_dot(direction, direction) + 1.0e-300
        )
        fraction = np.clip(fraction, 0.0, 1.0)
        result[opposite] = c + fraction[:, None] * direction
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
        self.leaf_at_lattice = np.full(
            (lattice_width, lattice_width, lattice_width), -1, dtype=np.int32
        )
        for leaf_id, (lower, width) in enumerate(zip(self.lower_lattice, self.widths, strict=True)):
            x0, y0, z0 = map(int, lower)
            stop = lower + width
            self.leaf_at_lattice[x0 : int(stop[0]), y0 : int(stop[1]), z0 : int(stop[2])] = leaf_id
        self.tolerance = 1.0e-15 * float(root_bounds[1] - root_bounds[0])
        self._leaf_triangles: dict[int, np.ndarray] = {}
        self._nearest_cache: dict[tuple[int, bool, bytes], tuple[np.ndarray, float, int]] = {}

    def leaves_in_box(self, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        """Return octree leaves with positive-volume overlap with a box."""
        lattice_limit = np.asarray(self.leaf_at_lattice.shape, dtype=np.int64)
        first = np.floor((lower - self.root_lower) / self.finest_cell_size).astype(np.int64)
        last = np.ceil((upper - self.root_lower) / self.finest_cell_size).astype(np.int64) - 1
        first = np.clip(first, 0, lattice_limit - 1)
        last = np.clip(last, 0, lattice_limit - 1)
        return np.unique(
            self.leaf_at_lattice[
                first[0] : last[0] + 1,
                first[1] : last[1] + 1,
                first[2] : last[2] + 1,
            ]
        )

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
        # cfMesh expands its root cube by ``SMALL * span``.  The resulting
        # uniformly negative lattice offset decides which side of an exact
        # search-box/grid tie is visited on curved surfaces.
        search_value = value - self.tolerance
        lattice = np.floor((search_value - self.root_lower) / self.finest_cell_size).astype(
            np.int64
        )
        lattice = np.clip(lattice, 0, np.asarray(self.leaf_at_lattice.shape) - 1)
        containing = int(self.leaf_at_lattice[tuple(lattice)])
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


def _map_assigned_patch_points(
    mesh_data: dict[str, Any],
    face_patch_ids: np.ndarray,
    patch_locators: Sequence[_OctreeSurfaceLocator],
    *,
    selected_points: set[int] | None = None,
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
            float(np.dot(face_centres[face_id] - original, face_centres[face_id] - original))
            for face_id in point_faces[point_id]
        )
        for _iteration in range(20):
            mapped = np.asarray(
                [
                    patch_locators[patch_id].nearest(approximate, max_search_iterations=5)[0]
                    for patch_id in ordered_patches
                ]
            ).mean(axis=0)
            if float(np.dot(mapped - approximate, mapped - approximate)) < (
                1.0e-8 * maximum_distance_squared
            ):
                break
            approximate = mapped
        displacement = approximate - original
        distance_squared = float(np.dot(displacement, displacement))
        if len(ordered_patches) == 2 and distance_squared > maximum_distance_squared:
            approximate = original + displacement * np.sqrt(
                maximum_distance_squared / distance_squared
            )
        feature_updates[point_id] = approximate
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
            if not remap:
                points[point_id] = value
                continue
            patches = tuple(sorted(point_patches[point_id]))
            approximate = value
            for _mapping_iteration in range(20):
                mapped = np.asarray(
                    [
                        patch_locators[patch_id].nearest(approximate, max_search_iterations=5)[0]
                        for patch_id in patches
                    ]
                ).mean(axis=0)
                if float(np.dot(mapped - approximate, mapped - approximate)) < 1.0e-10:
                    break
                approximate = mapped
            points[point_id] = approximate
        smooth_cfmesh_partition_points(mesh_data, partition_points)
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
    patch_indices = tuple(
        SurfaceIndex.build(np.ascontiguousarray(groups[name])) for name in patch_names
    )
    triangles = np.ascontiguousarray(np.concatenate(tuple(groups[name] for name in patch_names)))
    global_index = SurfaceIndex.build(triangles)
    octree_leaves = np.asarray(mesh_data["_cfmesh_octree_leaves"], dtype=np.int32)
    root_bounds = tuple(mesh_data["mesh_generation"]["root_box"])
    finest_cell_size = float(mesh_data["mesh_generation"]["finest_cell_size"])
    patch_locators = tuple(
        _OctreeSurfaceLocator(
            index,
            root_bounds=root_bounds,  # type: ignore[arg-type]
            finest_cell_size=finest_cell_size,
            leaves=octree_leaves,
        )
        for index in patch_indices
    )
    global_locator = _OctreeSurfaceLocator(
        global_index,
        root_bounds=root_bounds,  # type: ignore[arg-type]
        finest_cell_size=finest_cell_size,
        leaves=octree_leaves,
    )

    def nearest_in_region(patch_id: int, value: np.ndarray) -> tuple[np.ndarray, float]:
        return patch_locators[patch_id].nearest(value, max_search_iterations=5)

    def nearest_global(value: np.ndarray) -> tuple[np.ndarray, float]:
        return global_locator.nearest(value, max_search_iterations=100)

    points = np.asarray(mesh_data["vertex_position"], dtype=np.float64)
    faces = [np.asarray(face, dtype=np.int32) for face in mesh_data["faces"]]
    boundary_start = int(mesh_data["n_interior_faces"])
    boundary_face_ids = range(boundary_start, len(faces))
    boundary_point_ids = np.unique(
        np.concatenate([faces[face_id] for face_id in boundary_face_ids])
    )
    point_boundary_faces: dict[int, list[tuple[int, int]]] = {
        int(point_id): [] for point_id in boundary_point_ids
    }
    initial_face_distances: dict[tuple[int, int], float] = {}
    for face_id in boundary_face_ids:
        face = faces[face_id]
        face_centre = _foam_face_centre(points[face])
        for position, point_id_value in enumerate(face):
            point_id = int(point_id_value)
            point_boundary_faces[point_id].append((face_id, position))
            offset = points[point_id] - face_centre
            initial_face_distances[(face_id, position)] = max(
                float(np.dot(offset, offset)), np.finfo(np.float64).tiny
            )

    def nearest_to_patches(value: np.ndarray, patch_ids: Sequence[int]) -> np.ndarray:
        if len(patch_ids) == 1:
            return nearest_in_region(patch_ids[0], value)[0]
        nearest = value.copy()
        for _iteration in range(40):
            projected = np.asarray(
                [nearest_in_region(patch_id, nearest)[0] for patch_id in patch_ids]
            )
            updated = projected.mean(axis=0)
            distance_squared = float(np.dot(updated - value, updated - value))
            change = updated - nearest
            if float(np.dot(change, change)) < 1.0e-4 * distance_squared:
                break
            nearest = updated
        return nearest

    for _iteration in range(3):
        face_centres = {
            face_id: _foam_face_centre(points[faces[face_id]]) for face_id in boundary_face_ids
        }
        point_patches: dict[int, list[int]] = {int(point_id): [] for point_id in boundary_point_ids}
        for face_id in boundary_face_ids:
            centre = face_centres[face_id]
            face = faces[face_id]
            face_area = _face_area_vector_with_centre(points[face], centre)
            box_size = float(np.linalg.norm(points[face] - centre, axis=1).max(initial=0.0))
            nearby_leaves = global_locator.leaves_in_box(centre - box_size, centre + box_size)
            nearby_patches = [
                patch_id
                for patch_id, locator in enumerate(patch_locators)
                if locator.has_triangles_in_leaves(nearby_leaves)
            ]
            if not nearby_patches:
                nearby_patches = list(range(len(patch_indices)))
            metrics: list[float] = []
            for patch_id in nearby_patches:
                projected_centre, centre_distance = nearest_in_region(patch_id, centre)
                projected_points = np.asarray(
                    [nearest_in_region(patch_id, points[int(point_id)])[0] for point_id in face],
                    dtype=np.float64,
                )
                projected_area = _face_area_vector_with_centre(projected_points, projected_centre)
                metrics.append(
                    centre_distance * centre_distance
                    + abs(float(np.linalg.norm(projected_area) - np.linalg.norm(face_area)))
                )
            best_patch = nearby_patches[int(np.argmin(metrics))]
            for point_id_value in faces[face_id]:
                point_id = int(point_id_value)
                if best_patch not in point_patches[point_id]:
                    point_patches[point_id].append(best_patch)

        weighted_centres = np.empty((len(boundary_point_ids), 3), dtype=np.float64)
        for local_index, point_id_value in enumerate(boundary_point_ids):
            point_id = int(point_id_value)
            point = points[point_id]
            weighted_centre = np.zeros(3, dtype=np.float64)
            weight_sum = 0.0
            for face_id, position in point_boundary_faces[point_id]:
                face_centre = face_centres[face_id]
                offset = point - face_centre
                weight = max(
                    float(np.dot(offset, offset)) / initial_face_distances[(face_id, position)],
                    np.finfo(np.float64).tiny,
                )
                weighted_centre += weight * face_centre
                weight_sum += weight
            weighted_centres[local_index] = weighted_centre / weight_sum

        updates: dict[int, np.ndarray] = {}
        for local_index, point_id_value in enumerate(boundary_point_ids):
            point_id = int(point_id_value)
            mapped = nearest_to_patches(weighted_centres[local_index], point_patches[point_id])
            updates[point_id] = points[point_id] + 0.5 * (mapped - points[point_id])
        for point_id, value in updates.items():
            points[point_id] = value

    domain_lower = np.asarray(domain[::2], dtype=np.float64)
    domain_upper = np.asarray(domain[1::2], dtype=np.float64)
    domain_names = set(domain_patch_names)
    tie_tolerance = 1.0e-12 * max(float(np.max(domain_upper - domain_lower)), 1.0)
    mapped_points = np.empty((len(boundary_point_ids), 3), dtype=np.float64)
    for local_index, point_id_value in enumerate(boundary_point_ids):
        point_id = int(point_id_value)
        point = points[point_id]
        mapped = nearest_global(point)[0]
        selected_names = {patch_names[index] for index in point_patches[point_id]}
        if len(selected_names) > 1 and selected_names <= domain_names:
            lower_distances = np.abs(point - domain_lower)
            upper_distances = np.abs(domain_upper - point)
            distances = np.minimum(lower_distances, upper_distances)
            minimum = float(distances.min())
            tied_axes = tuple(
                int(axis) for axis in np.flatnonzero(np.abs(distances - minimum) <= tie_tolerance)
            )
            if len(tied_axes) > 1:
                upper_side = upper_distances < lower_distances
                if 2 in tied_axes:
                    axis = (
                        2
                        if not upper_side[2]
                        else min(candidate for candidate in tied_axes if candidate != 2)
                    )
                else:
                    axis = 1 if upper_side[0] and not upper_side[1] else 0
                mapped = point.copy()
                mapped[axis] = domain_upper[axis] if upper_side[axis] else domain_lower[axis]
        mapped_points[local_index] = mapped
    points[boundary_point_ids] = mapped_points
    untangling = untangle_cfmesh_surface(
        mesh_data,
        map_to_surface=lambda point: nearest_global(point)[0],
    )
    mesh_data["mesh_generation"]["surface_projection"] = {
        "pre_map_iterations": 3,
        "attempted_points": int(len(boundary_point_ids)),
        "accepted_points": int(len(boundary_point_ids)),
        "patch_names": patch_names,
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
    boundary_point_ids = np.unique(np.concatenate(source_faces[n_internal:]))
    domain_lower = np.asarray(domain[::2], dtype=np.float64)
    domain_upper = np.asarray(domain[1::2], dtype=np.float64)
    edge_distance = 0.5 * float(mesh_data["mesh_generation"]["finest_cell_size"])
    edge_and_corner_points = 0
    for point_id_value in boundary_point_ids:
        point_id = int(point_id_value)
        point = points[point_id]
        lower_distances = np.abs(point - domain_lower)
        upper_distances = np.abs(domain_upper - point)
        distances = np.minimum(lower_distances, upper_distances)
        near_axes = np.flatnonzero(distances <= edge_distance)
        if len(near_axes) < 2:
            continue
        edge_and_corner_points += 1
        upper_side = upper_distances < lower_distances
        for axis_value in near_axes:
            axis = int(axis_value)
            point[axis] = domain_upper[axis] if upper_side[axis] else domain_lower[axis]

    boundary_face_ids = tuple(range(n_internal, len(source_faces)))
    face_patch_ids = np.empty(len(boundary_face_ids), dtype=np.int32)
    for local_face_id, face_id in enumerate(boundary_face_ids):
        face = source_faces[face_id]
        centre = _foam_face_centre(points[face])
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
                        abs(float(np.dot(triangle_normal, face_normal))),
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
            distance_squared = float(np.dot(mapped - value, mapped - value))
            if float(np.dot(mapped - current, mapped - current)) < 1.0e-4 * distance_squared:
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
                np.dot(edge_vector, mapped_vector)
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
        "edge_and_corner_points": edge_and_corner_points,
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
