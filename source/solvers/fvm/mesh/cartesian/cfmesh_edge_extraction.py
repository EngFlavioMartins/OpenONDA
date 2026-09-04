# SPDX-License-Identifier: GPL-3.0-or-later
"""Topological edge correction used by cfMesh's ``edgeExtraction`` stage.

cfMesh triangulates internal faces beside concave patch junctions, splits
boundary quads which carry two feature edges at one vertex, and then replaces
every affected polyhedron by one pyramid per face.  The operation is entirely
topological apart from the inserted face and cell centres.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .cfmesh_surface_optimisation import (
    _face_centre,
    inverted_cfmesh_boundary_points,
)

_VSMALL = 1.0e-300


def _boundary_addressing(
    mesh_data: dict[str, Any], faces: list[np.ndarray]
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


def _topology_plan(mesh_data: dict[str, Any]) -> tuple[set[int], dict[int, int], set[int]]:
    points = np.asarray(mesh_data["vertex_position"], dtype=np.float64)
    faces = [np.asarray(face, dtype=np.int32) for face in mesh_data["faces"]]
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
    for face_id in range(n_internal):
        face = faces[face_id]
        if not concave_points.intersection(map(int, face)):
            continue
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
    for face_id in range(n_internal, len(faces)):
        face = faces[face_id]
        for position, point_value in enumerate(face):
            point_id = int(point_value)
            if point_id not in edge_points:
                continue
            following = int(face[(position + 1) % len(face)])
            previous = int(face[(position - 1) % len(face)])
            following_edge = (min(point_id, following), max(point_id, following))
            previous_edge = (min(previous, point_id), max(previous, point_id))
            if following_edge in feature_edges and previous_edge in feature_edges:
                if len(face) != 4:
                    raise ValueError(
                        "cfMesh edge correction currently supports feature-corner quads only"
                    )
                split_boundary[face_id] = position
                marked_cells.add(int(owners[face_id]))
                break
    return split_internal, split_boundary, marked_cells


def _split_faces(
    mesh_data: dict[str, Any],
    split_internal: set[int],
    split_boundary: dict[int, int],
) -> tuple[
    np.ndarray,
    list[np.ndarray],
    np.ndarray,
    np.ndarray,
    dict[tuple[int, ...], int],
    list[list[int]],
]:
    source_points = np.asarray(mesh_data["vertex_position"], dtype=np.float64)
    source_faces = [np.asarray(face, dtype=np.int32) for face in mesh_data["faces"]]
    source_owners = np.asarray(mesh_data["owners"], dtype=np.int32)
    source_neighbours = np.asarray(mesh_data["neighbours"], dtype=np.int32)
    n_internal = int(mesh_data["n_interior_faces"])
    point_values = [point.copy() for point in source_points]
    faces: list[np.ndarray] = []
    owners: list[int] = []
    neighbours: list[int] = []
    boundary_patch_by_signature: dict[tuple[int, ...], int] = {}
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
        centre_id = len(point_values)
        point_values.append(_face_centre(source_points[face]))
        for first, second in zip(face, np.roll(face, -1), strict=True):
            face_replacements[face_id].append(len(faces))
            faces.append(np.asarray((first, second, centre_id), dtype=np.int32))
            owners.append(int(source_owners[face_id]))
            neighbours.append(int(source_neighbours[face_id]))

    for face_id in range(n_internal, len(source_faces)):
        face = source_faces[face_id]
        patch_id = face_patch[face_id]
        if face_id in split_boundary:
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
            boundary_patch_by_signature[tuple(sorted(map(int, new_face)))] = patch_id

    return (
        np.ascontiguousarray(point_values, dtype=np.float64),
        faces,
        np.ascontiguousarray(owners, dtype=np.int32),
        np.ascontiguousarray(neighbours, dtype=np.int32),
        boundary_patch_by_signature,
        face_replacements,
    )


def _cell_faces(
    n_cells: int,
    faces: list[np.ndarray],
    owners: np.ndarray,
    neighbours: np.ndarray,
    *,
    source_order: list[list[int]] | None = None,
    face_replacements: list[list[int]] | None = None,
    source_n_internal: int | None = None,
) -> list[list[tuple[np.ndarray, bool]]]:
    result: list[list[tuple[np.ndarray, bool]]] = [[] for _cell in range(n_cells)]
    n_internal = len(neighbours)
    if source_order is not None and face_replacements is not None:
        if source_n_internal is None:
            raise ValueError("source_n_internal is required with cfMesh cell order")
        for cell_id, old_face_ids in enumerate(source_order):
            for old_face_id in old_face_ids:
                if old_face_id >= source_n_internal:
                    continue
                for face_id in face_replacements[old_face_id]:
                    result[cell_id].append((faces[face_id], int(owners[face_id]) == cell_id))
        # meshSurfaceEdgeExtractor removes every old boundary face and its
        # boundaryFacesGenerator appends the replacements after the retained
        # internal faces.  Preserve that per-cell insertion order.
        for face_id in range(n_internal, len(faces)):
            result[int(owners[face_id])].append((faces[face_id], True))
        return result
    for face_id, face in enumerate(faces):
        result[int(owners[face_id])].append((face, True))
        if face_id < n_internal:
            result[int(neighbours[face_id])].append((face, False))
    return result


def _rebuild_from_cells(
    mesh_data: dict[str, Any],
    points: np.ndarray,
    split_faces: list[np.ndarray],
    split_owners: np.ndarray,
    split_neighbours: np.ndarray,
    boundary_patch_by_signature: dict[tuple[int, ...], int],
    marked_cells: set[int],
    face_replacements: list[list[int]],
) -> None:
    n_source_cells = int(mesh_data["n_cells"])
    source_order = mesh_data.get("_cfmesh_cell_face_order")
    cell_faces = _cell_faces(
        n_source_cells,
        split_faces,
        split_owners,
        split_neighbours,
        source_order=source_order,
        face_replacements=face_replacements,
        source_n_internal=int(mesh_data["n_interior_faces"]),
    )
    point_values = [point.copy() for point in points]
    output_cells: list[list[np.ndarray]] = []
    source_cell_ids: list[int] = []

    for cell_id, entries in enumerate(cell_faces):
        if cell_id in marked_cells:
            continue
        output_cells.append(
            [face.copy() if is_owner else face[::-1].copy() for face, is_owner in entries]
        )
        source_cell_ids.append(cell_id)

    for cell_id in sorted(marked_cells):
        entries = cell_faces[cell_id]
        unique_points = np.unique(np.concatenate([face for face, _is_owner in entries]))
        top_vertex = len(point_values)
        point_values.append(points[unique_points].mean(axis=0))
        for face, is_owner in entries:
            outward_base = face.copy() if is_owner else face[::-1].copy()
            pyramid_faces = [outward_base]
            for position in range(len(face)):
                following = int(face[(position + 1) % len(face)])
                current = int(face[position])
                side = (
                    np.asarray((following, current, top_vertex), dtype=np.int32)
                    if is_owner
                    else np.asarray((following, top_vertex, current), dtype=np.int32)
                )
                pyramid_faces.append(side)
            output_cells.append(pyramid_faces)
            source_cell_ids.append(cell_id)

    records: dict[tuple[int, ...], list[tuple[np.ndarray, int]]] = defaultdict(list)
    for cell_id, entries in enumerate(output_cells):
        for face in entries:
            records[tuple(sorted(map(int, face)))].append((face, cell_id))

    internal_faces: list[np.ndarray] = []
    internal_owners: list[int] = []
    internal_neighbours: list[int] = []
    patch_count = len(mesh_data["boundary"])
    boundary_faces: list[list[np.ndarray]] = [[] for _patch in range(patch_count)]
    boundary_owners: list[list[int]] = [[] for _patch in range(patch_count)]
    for signature, entries in records.items():
        if len(entries) == 2:
            first, second = entries
            internal_faces.append(first[0])
            internal_owners.append(first[1])
            internal_neighbours.append(second[1])
        elif len(entries) == 1:
            patch_id = boundary_patch_by_signature.get(signature)
            if patch_id is None:
                raise ValueError(
                    f"cfMesh edge correction left an unmatched internal face: {signature}"
                )
            face, owner = entries[0]
            boundary_faces[patch_id].append(face)
            boundary_owners[patch_id].append(owner)
        else:
            raise ValueError(f"cfMesh edge correction produced a non-manifold face: {signature}")

    combined_faces = internal_faces.copy()
    combined_owners = internal_owners.copy()
    combined_neighbours = internal_neighbours.copy()
    boundary: list[dict[str, Any]] = []
    start_face = len(internal_faces)
    for patch_id, patch in enumerate(mesh_data["boundary"]):
        combined_faces.extend(boundary_faces[patch_id])
        combined_owners.extend(boundary_owners[patch_id])
        boundary.append(
            {
                "name": str(patch["name"]),
                "start_face": start_face,
                "n_faces": len(boundary_faces[patch_id]),
                "type": str(patch.get("type", "patch")),
            }
        )
        start_face += len(boundary_faces[patch_id])

    old_levels = np.asarray(mesh_data.get("cell_levels", np.zeros(n_source_cells)), dtype=np.int8)
    old_sizes = np.asarray(mesh_data.get("cell_sizes", np.ones(n_source_cells)), dtype=np.float32)
    source_ids = np.asarray(source_cell_ids, dtype=np.int32)
    face_by_signature = {
        tuple(sorted(map(int, face))): face_id for face_id, face in enumerate(combined_faces)
    }
    cfmesh_cell_face_order = [
        [face_by_signature[tuple(sorted(map(int, face)))] for face in entries]
        for entries in output_cells
    ]
    mesh_data.update(
        {
            "vertex_position": np.ascontiguousarray(point_values, dtype=np.float64),
            "faces": combined_faces,
            "owners": np.ascontiguousarray(combined_owners, dtype=np.int32),
            "neighbours": np.ascontiguousarray(combined_neighbours, dtype=np.int32),
            "boundary": boundary,
            "n_cells": len(output_cells),
            "n_faces": len(combined_faces),
            "n_interior_faces": len(internal_faces),
            "n_points": len(point_values),
            "cell_levels": old_levels[source_ids],
            "cell_sizes": old_sizes[source_ids],
            "cell_type_code": np.full(len(output_cells), 5, dtype=np.int32),
            "_cfmesh_cell_face_order": cfmesh_cell_face_order,
        }
    )
    for stale in (
        "cell_vertex_indices",
        "cell_face_indices",
        "cell_face_offset",
        "global_cell_id",
        "global_face_id",
        "boundary_layer_index",
    ):
        mesh_data.pop(stale, None)


def extract_cfmesh_edges(mesh_data: dict[str, Any]) -> None:
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
