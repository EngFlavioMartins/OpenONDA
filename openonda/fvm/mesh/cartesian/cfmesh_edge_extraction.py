# SPDX-License-Identifier: GPL-3.0-or-later
"""Topological edge correction used by cfMesh's ``edgeExtraction`` stage.

cfMesh triangulates internal faces beside concave patch junctions, splits
boundary quads which carry two feature edges at one vertex, and then replaces
every affected polyhedron by one pyramid per face.  The operation is entirely
topological apart from the inserted face and cell centres.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import chain

import numpy as np

from ..data import BoundaryPatchData, MeshData
from ..topology import last_key_matches, pack_face_nodes, sorted_vertex_keys
from .cfmesh_surface_optimisation import (
    _face_centre,
    inverted_cfmesh_boundary_points,
)

_VSMALL = 1.0e-300


def _reverse_face(face: np.ndarray) -> np.ndarray:
    """OpenFOAM reverseFace retains the first vertex."""
    return np.concatenate((face[:1], face[:0:-1]))


def _boundary_addressing(
    mesh_data: MeshData, faces: list[np.ndarray]
) -> tuple[dict[int, int], dict[tuple[int, int], list[int]], list[int]]:
    face_patch: dict[int, int] = {}
    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    face_patch_ids: list[int] = []
    for patch_id, patch in enumerate(mesh_data["boundary"]):
        start = int(patch["start_face"])
        stop = start + int(patch["n_faces"])
        for face_id in range(start, stop):
            face_patch[face_id] = patch_id
            face_patch_ids.append(patch_id)
            face = faces[face_id]
            for first, second in zip(face, np.roll(face, -1), strict=True):
                first_id = int(first)
                second_id = int(second)
                edge_faces[(min(first_id, second_id), max(first_id, second_id))].append(face_id)
    return face_patch, edge_faces, face_patch_ids


def _ordered_edge_faces(
    edge: tuple[int, int], attached: list[int], faces: list[np.ndarray]
) -> tuple[int, int]:
    """Reproduce meshSurfaceEngine's owner-face-first edge ordering."""
    first, second = sorted(attached)
    start, end = edge
    face = faces[second]
    position = int(np.flatnonzero(face == start)[0])
    if int(face[(position + 1) % len(face)]) == end:
        first, second = second, first
    return first, second


def _faces_touching(
    faces: list[np.ndarray] | np.ndarray, points: set[int], n_points: int
) -> np.ndarray:
    """Whether each face has a vertex among ``points``."""
    mask = np.zeros(n_points, dtype=bool)
    mask[np.fromiter(points, dtype=np.int64, count=len(points))] = True
    nodes, offsets = pack_face_nodes(faces)
    face_ids = np.repeat(np.arange(len(faces)), np.diff(offsets))
    return np.bincount(face_ids, weights=mask[nodes], minlength=len(faces)) > 0


def _topology_plan(mesh_data: MeshData) -> tuple[set[int], dict[int, int], set[int]]:
    points = np.asarray(mesh_data["vertex_position"], dtype=np.float64)
    faces = mesh_data["faces"]
    owners = np.asarray(mesh_data["owners"], dtype=np.int32)
    neighbours = np.asarray(mesh_data["neighbours"], dtype=np.int32)
    n_internal = int(mesh_data["n_interior_faces"])
    face_patch, edge_faces, face_patch_ids = _boundary_addressing(mesh_data, faces)

    problematic = inverted_cfmesh_boundary_points(mesh_data, face_patch_ids)
    edge_patch: dict[tuple[int, int], int] = {}
    feature_edges: set[tuple[int, int]] = set()
    concave_points: set[int] = set()
    feature_count: dict[int, int] = defaultdict(int)
    for edge, attached in edge_faces.items():
        if len(attached) != 2:
            continue
        patch0 = face_patch[attached[0]]
        patch1 = face_patch[attached[1]]
        if patch0 == patch1:
            edge_patch[edge] = patch0
            continue
        edge_patch[edge] = -1
        feature_edges.add(edge)
        feature_count[edge[0]] += 1
        feature_count[edge[1]] += 1
        edge_face0, edge_face1 = _ordered_edge_faces(edge, attached, faces)
        start, end = edge
        face_centre0 = _face_centre(points[faces[edge_face0]])
        face_centre1 = _face_centre(points[faces[edge_face1]])
        signed_tet_volume = float(
            np.dot(
                np.cross(points[end] - points[start], face_centre0 - points[start]),
                face_centre1 - points[start],
            )
            / 6.0
        )
        if start in problematic or end in problematic or signed_tet_volume > -_VSMALL:
            concave_points.update(edge)

    split_internal: set[int] = set()
    marked_cells: set[int] = set()
    concave_faces = np.flatnonzero(_faces_touching(faces, concave_points, len(points)))
    for face_value in concave_faces[concave_faces < n_internal]:
        face_id = int(face_value)
        face = faces[face_id]
        boundary_edge_patches: list[int] = []
        for first, second in zip(face, np.roll(face, -1), strict=True):
            first_id = int(first)
            second_id = int(second)
            edge = (min(first_id, second_id), max(first_id, second_id))
            if edge in edge_faces:
                patch_id = edge_patch[edge]
                if patch_id not in boundary_edge_patches:
                    boundary_edge_patches.append(patch_id)
        if len(boundary_edge_patches) <= 1:
            continue
        split_internal.add(face_id)
        marked_cells.update((int(owners[face_id]), int(neighbours[face_id])))

    split_boundary: dict[int, int] = {}
    edge_points = {point_id for point_id, count in feature_count.items() if count == 2}
    edge_faces_found = np.flatnonzero(_faces_touching(faces, edge_points, len(points)))
    for face_value in edge_faces_found[edge_faces_found >= n_internal]:
        face_id = int(face_value)
        face = faces[face_id]
        for position, point_value in enumerate(face):
            point_id = int(point_value)
            if point_id not in edge_points:
                continue
            following = int(face[(position + 1) % len(face)])
            previous = int(face[(position - 1) % len(face)])
            following_edge = (min(point_id, following), max(point_id, following))
            previous_edge = (min(previous, point_id), max(previous, point_id))
            if (
                following_edge in feature_edges
                and previous_edge in feature_edges
                and len(face) >= 4
            ):
                split_boundary[face_id] = position
                marked_cells.add(int(owners[face_id]))
                break
    return split_internal, split_boundary, marked_cells


def _split_faces(
    mesh_data: MeshData,
    split_internal: set[int],
    split_boundary: dict[int, int],
) -> tuple[
    np.ndarray,
    list[np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[list[int]],
]:
    source_points = np.asarray(mesh_data["vertex_position"], dtype=np.float64)
    source_faces = mesh_data["faces"]
    source_owners = np.asarray(mesh_data["owners"], dtype=np.int32)
    source_neighbours = np.asarray(mesh_data["neighbours"], dtype=np.int32)
    n_internal = int(mesh_data["n_interior_faces"])
    point_values = [source_points]
    n_points = len(source_points)
    faces: list[np.ndarray] = []
    owners: list[int] = []
    neighbours: list[int] = []
    boundary_patch_ids: list[int] = []
    face_replacements: list[list[int]] = [[] for _face in source_faces]

    face_patch: dict[int, int] = {}
    for patch_id, patch in enumerate(mesh_data["boundary"]):
        start = int(patch["start_face"])
        stop = start + int(patch["n_faces"])
        for face_id in range(start, stop):
            face_patch[face_id] = patch_id

    for face_id in range(n_internal):
        face = source_faces[face_id]
        if face_id not in split_internal:
            face_replacements[face_id].append(len(faces))
            faces.append(face.copy())
            owners.append(int(source_owners[face_id]))
            neighbours.append(int(source_neighbours[face_id]))
            continue
        centre_id = n_points
        n_points += 1
        point_values.append(_face_centre(source_points[face])[None, :])
        for first, second in zip(face, np.roll(face, -1), strict=True):
            face_replacements[face_id].append(len(faces))
            faces.append(np.asarray((first, second, centre_id), dtype=np.int32))
            owners.append(int(source_owners[face_id]))
            neighbours.append(int(source_neighbours[face_id]))

    for face_id in range(n_internal, len(source_faces)):
        face = source_faces[face_id]
        patch_id = face_patch[face_id]
        if face_id in split_boundary and len(face) > 4:
            centre_id = n_points
            n_points += 1
            point_values.append(_face_centre(source_points[face])[None, :])
            new_faces = tuple(
                np.asarray((first, second, centre_id), dtype=np.int32)
                for first, second in zip(face, np.roll(face, -1), strict=True)
            )
        elif face_id in split_boundary:
            position = split_boundary[face_id]
            first = np.asarray(
                (
                    face[position],
                    face[(position + 1) % 4],
                    face[(position + 2) % 4],
                ),
                dtype=np.int32,
            )
            second = np.asarray(
                (
                    face[position],
                    face[(position + 2) % 4],
                    face[(position - 1) % 4],
                ),
                dtype=np.int32,
            )
            new_faces = (first, second)
        else:
            new_faces = (face.copy(),)
        for new_face in new_faces:
            face_replacements[face_id].append(len(faces))
            faces.append(new_face)
            owners.append(int(source_owners[face_id]))
            boundary_patch_ids.append(patch_id)

    return (
        np.ascontiguousarray(np.concatenate(point_values), dtype=np.float64),
        faces,
        np.ascontiguousarray(owners, dtype=np.int32),
        np.ascontiguousarray(neighbours, dtype=np.int32),
        np.asarray(boundary_patch_ids, dtype=np.int64),
        face_replacements,
    )


def _cell_face_entries(
    n_cells: int,
    owners: np.ndarray,
    neighbours: np.ndarray,
    n_faces: int,
    *,
    source_order: list[list[int]] | None,
    face_replacements: list[list[int]],
    source_n_internal: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Each cell's faces in :func:`_cell_faces` order, as parallel cell and face arrays."""
    n_internal = len(neighbours)
    boundary_faces = np.arange(n_internal, n_faces)
    if source_order is not None:
        counts = [len(row) for row in source_order]
        old = np.fromiter(chain.from_iterable(source_order), dtype=np.int64, count=sum(counts))
        cells = np.repeat(np.arange(n_cells), counts)
        keep = old < source_n_internal
        old, cells = old[keep], cells[keep]
        replacement_counts = [len(row) for row in face_replacements]
        replacements = np.fromiter(
            chain.from_iterable(face_replacements), dtype=np.int64, count=sum(replacement_counts)
        )
        starts = np.concatenate(([0], np.cumsum(replacement_counts, dtype=np.int64)))
        expanded = starts[old + 1] - starts[old]
        within = np.arange(int(expanded.sum())) - np.repeat(
            np.cumsum(expanded) - expanded, expanded
        )
        internal_faces = replacements[np.repeat(starts[old], expanded) + within]
        entry_cells = np.concatenate((np.repeat(cells, expanded), owners[boundary_faces]))
        entry_faces = np.concatenate((internal_faces, boundary_faces))
    else:
        interior = np.arange(n_internal)
        entry_cells = np.concatenate(
            (np.column_stack((owners[:n_internal], neighbours)).reshape(-1), owners[boundary_faces])
        )
        entry_faces = np.concatenate((np.repeat(interior, 2), boundary_faces))
    order = np.argsort(entry_cells, kind="stable")
    return entry_cells[order], entry_faces[order]


def _rebuild_from_cells(
    mesh_data: MeshData,
    points: np.ndarray,
    split_faces: list[np.ndarray],
    split_owners: np.ndarray,
    split_neighbours: np.ndarray,
    boundary_patch_ids: np.ndarray,
    marked_cells: set[int],
    face_replacements: list[list[int]],
) -> None:
    """Rebuild the mesh with every marked cell replaced by one pyramid per face.

    Unmarked cells keep their faces, outward and in stored cell order; the
    pyramids follow, by marked cell. A vertex set met by two of these faces
    becomes an internal face oriented as its first one, met by one a
    boundary face of the split face it came from.
    """
    n_source_cells = int(mesh_data["n_cells"])
    n_split_internal = len(split_neighbours)
    entry_cells, entry_faces = _cell_face_entries(
        n_source_cells,
        split_owners,
        split_neighbours,
        len(split_faces),
        source_order=mesh_data.get("_cfmesh_cell_face_order"),
        face_replacements=face_replacements,
        source_n_internal=int(mesh_data["n_interior_faces"]),
    )
    entry_owner = split_owners[entry_faces] == entry_cells
    nodes, offsets = pack_face_nodes(split_faces)
    marked = np.zeros(n_source_cells, dtype=bool)
    marked[np.fromiter(marked_cells, dtype=np.int64, count=len(marked_cells))] = True

    # Unmarked cells: their faces, reversed (first vertex kept) where they are the neighbour
    kept = ~marked[entry_cells]
    kept_faces, kept_owner = entry_faces[kept], entry_owner[kept]
    lengths = offsets[kept_faces + 1] - offsets[kept_faces]
    kept_offsets = np.concatenate(([0], np.cumsum(lengths)))
    position = np.arange(int(lengths.sum())) - np.repeat(kept_offsets[:-1], lengths)
    length = np.repeat(lengths, lengths)
    position = np.where(np.repeat(kept_owner, lengths), position, (length - position) % length)
    kept_nodes = nodes[np.repeat(offsets[kept_faces], lengths) + position]
    kept_cell_ids = np.cumsum(~marked) - 1
    kept_face_cells = kept_cell_ids[entry_cells[kept]]

    # Marked cells: one pyramid per face, apex at the mean of the cell's vertices
    n_kept_cells = int(np.count_nonzero(~marked))
    point_values = [points]
    pyramid_nodes: list[np.ndarray] = []
    pyramid_lengths: list[int] = []
    pyramid_cells: list[int] = []
    source_cell_ids = [np.flatnonzero(~marked)]
    output_cell = n_kept_cells
    top_vertex = len(points)
    entry_starts = np.searchsorted(entry_cells, np.arange(n_source_cells + 1))
    for cell_id in sorted(marked_cells):
        cell_entries = range(entry_starts[cell_id], entry_starts[cell_id + 1])
        # cell::labels preserves first occurrence in cell-face/cyclic-vertex
        # order. Sorting the labels changes the floating-point summation.
        unique_points = tuple(
            dict.fromkeys(
                int(point_id)
                for entry in cell_entries
                for point_id in split_faces[entry_faces[entry]]
            )
        )
        centre = np.zeros(3, dtype=np.float64)
        for point_id in unique_points:
            centre += points[point_id]
        point_values.append((centre / len(unique_points))[None, :])
        for entry in cell_entries:
            face = split_faces[entry_faces[entry]]
            is_owner = bool(entry_owner[entry])
            pyramid_nodes.append(face.copy() if is_owner else _reverse_face(face))
            pyramid_lengths.append(len(face))
            for local in range(len(face)):
                following = int(face[(local + 1) % len(face)])
                current = int(face[local])
                side = (
                    (following, current, top_vertex)
                    if is_owner
                    else (following, top_vertex, current)
                )
                pyramid_nodes.append(np.asarray(side, dtype=np.int32))
                pyramid_lengths.append(3)
            pyramid_cells.extend([output_cell] * (len(face) + 1))
            source_cell_ids.append(np.asarray([cell_id]))
            output_cell += 1
        top_vertex += 1

    all_nodes = np.concatenate([kept_nodes.astype(np.int32), *pyramid_nodes]).astype(np.int64)
    all_offsets = np.concatenate(
        (kept_offsets, kept_offsets[-1] + np.cumsum(np.asarray(pyramid_lengths, dtype=np.int64)))
    )
    face_cells = np.concatenate((kept_face_cells, np.asarray(pyramid_cells, dtype=np.int64)))

    # Group equal vertex sets in order of first appearance
    boundary_nodes, boundary_offsets = pack_face_nodes(split_faces[n_split_internal:])
    width = int(max(np.diff(all_offsets).max(initial=0), np.diff(boundary_offsets).max(initial=0)))
    keys = sorted_vertex_keys(all_nodes, all_offsets, width)
    unique_keys, first, inverse, counts = np.unique(
        keys, return_index=True, return_inverse=True, return_counts=True
    )
    group_order = np.argsort(first, kind="stable")
    by_group = np.argsort(inverse, kind="stable")
    group_starts = np.concatenate(([0], np.cumsum(counts)))
    boundary_keys = sorted_vertex_keys(boundary_nodes, boundary_offsets, width)
    split_position, from_split = last_key_matches(boundary_keys, unique_keys)
    faulty = (counts > 2) | ((counts == 1) & ~from_split)
    if np.any(faulty):
        group = group_order[np.flatnonzero(faulty[group_order])[0]]
        face = int(first[group])
        signature = tuple(sorted(map(int, all_nodes[all_offsets[face] : all_offsets[face + 1]])))
        if counts[group] == 1:
            raise ValueError(f"cfMesh edge correction left an unmatched internal face: {signature}")
        raise ValueError(f"cfMesh edge correction produced a non-manifold face: {signature}")
    internal_groups = group_order[counts[group_order] == 2]
    internal_first = first[internal_groups]
    internal_second = by_group[group_starts[internal_groups] + 1]

    # Boundary faces by patch, each patch in the order of the faces it came from
    boundary_groups = group_order[counts[group_order] == 1]
    patches = boundary_patch_ids[split_position[boundary_groups]]
    order = np.lexsort((split_position[boundary_groups], patches))
    boundary_groups = boundary_groups[order]
    patches = patches[order]

    output_faces = np.concatenate((internal_first, first[boundary_groups]))
    n_internal = len(internal_first)
    combined_owners = face_cells[output_faces]
    combined_neighbours = face_cells[internal_second]
    face_lengths = all_offsets[output_faces + 1] - all_offsets[output_faces]
    combined_nodes = all_nodes[
        np.repeat(all_offsets[output_faces], face_lengths)
        + np.arange(int(face_lengths.sum()))
        - np.repeat(np.cumsum(face_lengths) - face_lengths, face_lengths)
    ].astype(np.int32)
    face_offsets = np.concatenate(([0], np.cumsum(face_lengths))).tolist()
    combined_faces = [
        combined_nodes[start:stop]
        for start, stop in zip(face_offsets[:-1], face_offsets[1:], strict=True)
    ]
    patch_counts = np.bincount(patches, minlength=len(mesh_data["boundary"]))
    boundary: list[BoundaryPatchData] = []
    start_face = n_internal
    for patch_id, patch in enumerate(mesh_data["boundary"]):
        boundary.append(
            {
                "name": str(patch["name"]),
                "start_face": start_face,
                "n_faces": int(patch_counts[patch_id]),
                "type": str(patch.get("type", "patch")),
            }
        )
        start_face += int(patch_counts[patch_id])

    # Each cell lists its internal faces in its own face order, then its
    # boundary faces: replaceBoundary removes old boundary references before
    # appending the replacements, even when the face geometry did not change.
    combined_ids = np.empty(len(unique_keys), dtype=np.int64)
    combined_ids[np.concatenate((internal_groups, boundary_groups))] = np.arange(len(output_faces))
    traversal_ids = combined_ids[inverse]
    internal_entry = traversal_ids < n_internal
    order_cells = np.concatenate((face_cells[internal_entry], combined_owners[n_internal:]))
    order_faces = np.concatenate(
        (traversal_ids[internal_entry], np.arange(n_internal, len(output_faces)))
    )
    order = np.argsort(order_cells, kind="stable")
    n_output_cells = output_cell
    order_counts = np.bincount(order_cells, minlength=n_output_cells)
    flat_order = order_faces[order].tolist()
    order_offsets = np.concatenate(([0], np.cumsum(order_counts))).tolist()
    cfmesh_cell_face_order = [
        flat_order[start:stop]
        for start, stop in zip(order_offsets[:-1], order_offsets[1:], strict=True)
    ]

    old_levels = np.asarray(mesh_data.get("cell_levels", np.zeros(n_source_cells)), dtype=np.int8)
    old_sizes = np.asarray(mesh_data.get("cell_sizes", np.ones(n_source_cells)), dtype=np.float32)
    source_ids = np.concatenate(source_cell_ids).astype(np.int32)
    vertex_position = np.ascontiguousarray(np.concatenate(point_values), dtype=np.float64)
    mesh_data.update(
        {
            "vertex_position": vertex_position,
            "faces": combined_faces,
            "owners": np.ascontiguousarray(combined_owners, dtype=np.int32),
            "neighbours": np.ascontiguousarray(combined_neighbours, dtype=np.int32),
            "boundary": boundary,
            "n_cells": n_output_cells,
            "n_faces": len(combined_faces),
            "n_interior_faces": n_internal,
            "n_points": len(vertex_position),
            "cell_levels": old_levels[source_ids],
            "cell_sizes": old_sizes[source_ids],
            "cell_type_code": np.full(n_output_cells, 5, dtype=np.int32),
            "_cfmesh_cell_face_order": cfmesh_cell_face_order,
        }
    )
    mesh_data.pop("cell_vertex_indices", None)
    mesh_data.pop("cell_face_indices", None)
    mesh_data.pop("cell_face_offset", None)
    mesh_data.pop("global_cell_id", None)
    mesh_data.pop("global_face_id", None)
    mesh_data.pop("boundary_layer_index", None)


def extract_cfmesh_edges(mesh_data: MeshData) -> None:
    """Apply cfMesh's concave-edge correction and pyramid decomposition."""
    split_internal, split_boundary, marked_cells = _topology_plan(mesh_data)
    points, faces, owners, neighbours, boundary_patches, face_replacements = _split_faces(
        mesh_data, split_internal, split_boundary
    )
    source_cells = int(mesh_data["n_cells"])
    _rebuild_from_cells(
        mesh_data,
        points,
        faces,
        owners,
        neighbours,
        boundary_patches,
        marked_cells,
        face_replacements,
    )
    mesh_data["mesh_generation"]["cfmesh_edge_extraction"] = {
        "split_internal_faces": len(split_internal),
        "split_boundary_faces": len(split_boundary),
        "decomposed_cells": len(marked_cells),
        "replacement_pyramids": int(mesh_data["n_cells"]) - source_cells + len(marked_cells),
    }


__all__ = ["extract_cfmesh_edges"]
