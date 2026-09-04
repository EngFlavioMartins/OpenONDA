# SPDX-License-Identifier: GPL-3.0-or-later
"""cfMesh's mandatory all-patch wrapper-layer topology.

``cartesianMeshGenerator::generateBoundaryLayers`` invokes
``boundaryLayers::addLayerForAllPatches`` even without user layer controls.
The operation is topological: it creates one cell per boundary face, one cell
per edge between distinct patch keys, and one cell at every three-patch
corner.  Boundary vertices are expanded into the subset lattice of their
incident patch-normal displacements.  This module reproduces that serial
patch-wise construction; it does not substitute vertex smoothing for cells.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from itertools import combinations
from typing import Any

import numpy as np

from .cfmesh_surface_optimisation import inverted_cfmesh_boundary_points

_VSMALL = 1.0e-300


def _area_vector(coordinates: np.ndarray) -> np.ndarray:
    centre = coordinates.mean(axis=0)
    return 0.5 * np.cross(
        coordinates - centre,
        np.roll(coordinates, -1, axis=0) - centre,
    ).sum(axis=0)


def _distance_to_segment(point: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
    edge = second - first
    denominator = float(np.dot(edge, edge))
    if denominator <= _VSMALL:
        return float(np.linalg.norm(point - first))
    parameter = float(np.dot(point - first, edge) / denominator)
    parameter = min(1.0, max(0.0, parameter))
    return float(np.linalg.norm(point - (first + parameter * edge)))


def _normalised(vector: np.ndarray, *, context: str) -> np.ndarray:
    magnitude = float(np.linalg.norm(vector))
    if magnitude <= _VSMALL:
        raise ValueError(f"cfMesh wrapper has a zero normal at {context}")
    return vector / magnitude


def _proper_subsets(values: tuple[int, ...]) -> Iterable[frozenset[int]]:
    for size in range(len(values)):
        for subset in combinations(values, size):
            yield frozenset(subset)


def _signed_tetrahedron_volume(
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
    fourth: np.ndarray,
) -> float:
    return float(np.dot(np.cross(second - first, third - first), fourth - first) / 6.0)


def _shared_edge_is_concave(
    points: np.ndarray, first_face: np.ndarray, second_face: np.ndarray
) -> bool:
    """Match cfMesh's shared-edge convexity and 135-degree tests."""
    angles: list[float] = []
    second_positions = {int(point): position for position, point in enumerate(second_face)}
    for first_position, point_value in enumerate(first_face):
        point_id = int(point_value)
        second_position = second_positions.get(point_id)
        if second_position is None:
            continue
        own = (
            point_id,
            int(first_face[(first_position + 1) % len(first_face)]),
            int(first_face[(first_position - 1) % len(first_face)]),
        )
        neighbour = (
            point_id,
            int(second_face[(second_position + 1) % len(second_face)]),
            int(second_face[(second_position - 1) % len(second_face)]),
        )
        unique_own = next((point for point in own if point not in neighbour), None)
        if unique_own is None:
            continue
        volume = _signed_tetrahedron_volume(
            points[neighbour[0]],
            points[neighbour[1]],
            points[neighbour[2]],
            points[unique_own],
        )
        own_normal = np.cross(
            points[own[1]] - points[own[0]],
            points[own[2]] - points[own[0]],
        )
        neighbour_normal = np.cross(
            points[neighbour[1]] - points[neighbour[0]],
            points[neighbour[2]] - points[neighbour[0]],
        )
        own_normal = _normalised(own_normal, context="shared-edge own face")
        neighbour_normal = _normalised(neighbour_normal, context="shared-edge neighbour face")
        dot = float(np.clip(np.dot(own_normal, neighbour_normal), -1.0, 1.0))
        if volume > -_VSMALL:
            return True
        angles.append(float(np.arccos(-dot)))
    if not angles:
        raise ValueError("cfMesh wrapper found boundary faces without a shared edge")
    return float(np.mean(angles)) > 0.75 * np.pi


def _cfmesh_patch_keys(
    mesh_data: dict[str, Any],
    points: np.ndarray,
    faces: list[np.ndarray],
    face_patch: dict[int, int],
    point_patches: dict[int, set[int]],
    edge_faces: dict[tuple[int, int], list[int]],
) -> tuple[dict[int, int], list[tuple[int, int]]]:
    """Build cfMesh's transitive patch groups for concave junctions."""
    patch_count = len(mesh_data["boundary"])
    parent = list(range(patch_count))

    def find(patch_id: int) -> int:
        while parent[patch_id] != patch_id:
            parent[patch_id] = parent[parent[patch_id]]
            patch_id = parent[patch_id]
        return patch_id

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        parent[max(first_root, second_root)] = min(first_root, second_root)

    # A corner carrying more than three raw patches is handled as one group.
    for raw_patches in point_patches.values():
        if len(raw_patches) <= 3:
            continue
        first = min(raw_patches)
        for other in raw_patches:
            union(first, other)

    face_patch_ids = [
        face_patch[face_id] for face_id in range(int(mesh_data["n_interior_faces"]), len(faces))
    ]
    inverted = inverted_cfmesh_boundary_points(mesh_data, face_patch_ids)
    concave_pairs: set[tuple[int, int]] = set()
    for edge, attached in edge_faces.items():
        if len(attached) != 2 or edge[0] in inverted or edge[1] in inverted:
            continue
        patch_i = face_patch[attached[0]]
        patch_j = face_patch[attached[1]]
        if patch_i == patch_j:
            continue
        pair = (min(patch_i, patch_j), max(patch_i, patch_j))
        if _shared_edge_is_concave(points, faces[attached[0]], faces[attached[1]]):
            concave_pairs.add(pair)
            union(*pair)

    roots = {patch_id: find(patch_id) for patch_id in range(patch_count)}
    root_keys = {root: key for key, root in enumerate(sorted(set(roots.values())))}
    return (
        {patch_id: root_keys[root] for patch_id, root in roots.items()},
        sorted(concave_pairs),
    )


def _ordered_point_patch_ids(
    point_id: int,
    point_face_ids: list[int],
    faces: list[np.ndarray],
    face_patch: dict[int, int],
) -> tuple[int, ...]:
    """Reproduce cfMesh's cyclic face walk around a boundary corner."""
    ordered_faces = point_face_ids.copy()
    for index in range(len(ordered_faces) - 1):
        face = faces[ordered_faces[index]]
        position = int(np.flatnonzero(face == point_id)[0])
        edge = {
            point_id,
            int(face[(position + 1) % len(face)]),
        }
        for candidate in range(index + 1, len(ordered_faces)):
            if edge.issubset(set(map(int, faces[ordered_faces[candidate]]))):
                ordered_faces[index + 1], ordered_faces[candidate] = (
                    ordered_faces[candidate],
                    ordered_faces[index + 1],
                )
                break
    patch_ids: list[int] = []
    for face_id in ordered_faces:
        patch_id = face_patch[face_id]
        if patch_id not in patch_ids:
            patch_ids.append(patch_id)
    return tuple(patch_ids)


def add_cfmesh_wrapper_layer(mesh_data: dict[str, Any]) -> None:
    """Add cfMesh's default patch-wise boundary wrapper in place.

    Patch keys merge transitively at concave multi-patch junctions, matching
    ``boundaryLayers::findPatchesToBeTreatedTogether``.
    """
    source_points = np.asarray(mesh_data["vertex_position"], dtype=np.float64)
    source_faces = [np.asarray(face, dtype=np.int32) for face in mesh_data["faces"]]
    source_owners = np.asarray(mesh_data["owners"], dtype=np.int32)
    source_neighbours = np.asarray(mesh_data["neighbours"], dtype=np.int32)
    n_internal = int(mesh_data["n_interior_faces"])
    n_core_cells = int(mesh_data["n_cells"])
    patches = tuple(mesh_data["boundary"])

    face_patch: dict[int, int] = {}
    point_faces: dict[int, list[int]] = defaultdict(list)
    point_patches: dict[int, set[int]] = defaultdict(set)
    point_neighbours: dict[int, set[int]] = defaultdict(set)
    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for patch_id, patch in enumerate(patches):
        start = int(patch["start_face"])
        stop = start + int(patch["n_faces"])
        for face_id in range(start, stop):
            face_patch[face_id] = patch_id
            face = source_faces[face_id]
            for position, point_value in enumerate(face):
                point_id = int(point_value)
                previous = int(face[(position - 1) % len(face)])
                following = int(face[(position + 1) % len(face)])
                point_faces[point_id].append(face_id)
                point_patches[point_id].add(patch_id)
                point_neighbours[point_id].update((previous, following))
                edge = (min(point_id, following), max(point_id, following))
                if face_id not in edge_faces[edge]:
                    edge_faces[edge].append(face_id)

    open_edges = [edge for edge, attached in edge_faces.items() if len(attached) != 2]
    if open_edges:
        raise ValueError(
            f"cfMesh wrapper requires a closed manifold boundary; edge {open_edges[0]}"
        )

    patch_key, concave_patch_pairs = _cfmesh_patch_keys(
        mesh_data,
        source_points,
        source_faces,
        face_patch,
        point_patches,
        edge_faces,
    )
    point_keys = {
        point_id: {patch_key[patch_id] for patch_id in raw_patches}
        for point_id, raw_patches in point_patches.items()
    }

    face_normals: dict[int, np.ndarray] = {
        face_id: _area_vector(source_points[source_faces[face_id]])
        for face_id in range(n_internal, len(source_faces))
    }
    penetration: dict[tuple[int, int], np.ndarray] = {}
    for point_id in sorted(point_patches):
        keys = tuple(sorted(point_keys[point_id]))
        raw_patches = point_patches[point_id]
        point = source_points[point_id]
        for key in keys:
            group_patches = {patch_id for patch_id in raw_patches if patch_key[patch_id] == key}
            other_patches = tuple(sorted(raw_patches - group_patches))
            if not other_patches:
                normal = sum(
                    (face_normals[face_id] for face_id in point_faces[point_id]),
                    np.zeros(3, dtype=np.float64),
                )
                normal = _normalised(normal, context=f"point {point_id}, key {key}")
                distance = 0.5 * min(
                    float(np.linalg.norm(source_points[neighbour] - point))
                    for neighbour in point_neighbours[point_id]
                )
            elif len(other_patches) == 1:
                normal = sum(
                    (
                        face_normals[face_id]
                        for face_id in point_faces[point_id]
                        if patch_key[face_patch[face_id]] == key
                    ),
                    np.zeros(3, dtype=np.float64),
                )
                other_normal = sum(
                    (
                        face_normals[face_id]
                        for face_id in point_faces[point_id]
                        if face_patch[face_id] in other_patches
                    ),
                    np.zeros(3, dtype=np.float64),
                )
                other_normal = _normalised(other_normal, context=f"point {point_id}, other patch")
                normal -= float(np.dot(normal, other_normal)) * other_normal
                normal = _normalised(normal, context=f"edge point {point_id}, key {key}")
                candidates = [
                    0.5 * abs(float(np.dot(source_points[neighbour] - point, normal)))
                    for neighbour in point_neighbours[point_id]
                    if key not in point_keys[neighbour]
                ]
                if not candidates:
                    raise ValueError(
                        f"cfMesh wrapper cannot find an off-patch neighbour for point {point_id}"
                    )
                distance = min(candidates)
                for face_id in point_faces[point_id]:
                    if face_patch[face_id] not in other_patches:
                        continue
                    face = source_faces[face_id]
                    position = int(np.flatnonzero(face == point_id)[0])
                    limit = _distance_to_segment(
                        point,
                        source_points[int(face[(position - 1) % len(face)])],
                        source_points[int(face[(position + 1) % len(face)])],
                    )
                    distance = min(distance, 0.9 * limit)
            else:
                other_vertex = next(
                    (
                        neighbour
                        for neighbour in point_neighbours[point_id]
                        if set(other_patches).issubset(point_patches[neighbour])
                    ),
                    None,
                )
                if other_vertex is None:
                    raise ValueError(
                        f"cfMesh wrapper cannot find a corner edge for point {point_id}"
                    )
                direction = point - source_points[other_vertex]
                distance = 0.5 * float(np.linalg.norm(direction)) + _VSMALL
                normal = direction / (2.0 * distance)
            penetration[(point_id, key)] = -max(distance, _VSMALL) * normal

    # cfMesh stores the fully displaced position at the original point label;
    # every proper subset, including the unchanged surface position, gets a
    # new label. This is the 2^k vertex lattice at k-patch edges and corners.
    all_points = [point.copy() for point in source_points]
    state_id: dict[tuple[int, frozenset[int]], int] = {}
    for point_id in sorted(point_patches):
        keys = tuple(sorted(point_keys[point_id]))
        full = frozenset(keys)
        base = source_points[point_id]
        all_points[point_id] = base + sum(
            (penetration[(point_id, key)] for key in keys),
            np.zeros(3, dtype=np.float64),
        )
        state_id[(point_id, full)] = point_id
        for subset in _proper_subsets(keys):
            coordinate = base + sum(
                (penetration[(point_id, key)] for key in subset),
                np.zeros(3, dtype=np.float64),
            )
            state_id[(point_id, subset)] = len(all_points)
            all_points.append(coordinate)
    points = np.ascontiguousarray(all_points, dtype=np.float64)

    def state(point_id: int, retained: Iterable[int]) -> int:
        return state_id[(point_id, frozenset(retained))]

    # Each record is (face vertices, output boundary patch or None). Faces
    # marked None must pair with another newly created cell face.
    new_cells: list[list[tuple[np.ndarray, int | None]]] = []
    interface_cells: list[tuple[int, int]] = []
    source_cell_for_new: list[int] = []
    layer_outer_signatures: set[tuple[int, ...]] = set()
    for face_id in range(n_internal, len(source_faces)):
        face = source_faces[face_id]
        patch_id = face_patch[face_id]
        key = patch_key[patch_id]
        cell_id = n_core_cells + len(new_cells)
        interface_cells.append((face_id, cell_id))
        full_states = [state(int(point), point_keys[int(point)]) for point in face]
        outer_states = [state(int(point), point_keys[int(point)] - {key}) for point in face]
        entries: list[tuple[np.ndarray, int | None]] = [
            (np.asarray(outer_states, dtype=np.int32), patch_id)
        ]
        layer_outer_signatures.add(tuple(sorted(outer_states)))
        for position in range(len(face)):
            following = (position + 1) % len(face)
            entries.append(
                (
                    np.asarray(
                        (
                            full_states[position],
                            full_states[following],
                            outer_states[following],
                            outer_states[position],
                        ),
                        dtype=np.int32,
                    ),
                    None,
                )
            )
        new_cells.append(entries)
        source_cell_for_new.append(int(source_owners[face_id]))

    feature_edges = 0
    for (start_point, end_point), attached_faces in edge_faces.items():
        patch_i = face_patch[attached_faces[0]]
        patch_j = face_patch[attached_faces[1]]
        key_i = patch_key[patch_i]
        key_j = patch_key[patch_j]
        if key_i == key_j:
            continue
        feature_edges += 1

        def edge_states(
            point_id: int, first_patch: int = patch_i, second_patch: int = patch_j
        ) -> tuple[int, int, int, int]:
            keys = point_keys[point_id]
            return (
                state(point_id, keys),
                state(point_id, keys - {patch_key[second_patch]}),
                state(point_id, keys - {patch_key[first_patch]}),
                state(
                    point_id,
                    keys - {patch_key[first_patch], patch_key[second_patch]},
                ),
            )

        sf, p0s, p1s, ns = edge_states(start_point)
        ef, p0e, p1e, ne = edge_states(end_point)
        new_cells.append(
            [
                (np.asarray((ef, sf, p1s, p1e), dtype=np.int32), None),
                (np.asarray((p0e, ne, ns, p0s), dtype=np.int32), patch_j),
                (np.asarray((sf, ef, p0e, p0s), dtype=np.int32), None),
                (np.asarray((p1s, ns, ne, p1e), dtype=np.int32), patch_i),
                (np.asarray((ef, p1e, ne, p0e), dtype=np.int32), None),
                (np.asarray((sf, p0s, ns, p1s), dtype=np.int32), None),
            ]
        )
        source_cell_for_new.append(int(source_owners[attached_faces[0]]))

    corner_points = 0
    for point_id in sorted(point_patches):
        patch_ids = _ordered_point_patch_ids(
            point_id, point_faces[point_id], source_faces, face_patch
        )
        keys = tuple(dict.fromkeys(patch_key[patch_id] for patch_id in patch_ids))
        if len(keys) != 3:
            continue
        corner_points += 1
        first, second, third = keys
        full = state(point_id, keys)
        empty = state(point_id, ())
        p00 = state(point_id, (first,))
        p11 = state(point_id, (second,))
        p22 = state(point_id, (third,))
        p01 = state(point_id, (first, second))
        p02 = state(point_id, (first, third))
        p12 = state(point_id, (second, third))
        new_cells.append(
            [
                (np.asarray((full, p02, p00, p01), dtype=np.int32), None),
                (np.asarray((p12, p11, empty, p22), dtype=np.int32), patch_ids[0]),
                (np.asarray((full, p01, p11, p12), dtype=np.int32), None),
                (np.asarray((p02, p22, empty, p00), dtype=np.int32), patch_ids[1]),
                (np.asarray((full, p12, p22, p02), dtype=np.int32), None),
                (np.asarray((p01, p00, empty, p11), dtype=np.int32), patch_ids[2]),
            ]
        )
        source_cell_for_new.append(int(source_owners[point_faces[point_id][0]]))

    records: dict[tuple[int, ...], list[tuple[np.ndarray, int, int | None]]] = defaultdict(list)
    for local_cell, cell_entries in enumerate(new_cells):
        cell_id = n_core_cells + local_cell
        for face, boundary_patch in cell_entries:
            signature = tuple(sorted(map(int, face)))
            records[signature].append((face, cell_id, boundary_patch))

    cell_references = np.zeros((n_core_cells + len(new_cells), 3), dtype=np.float64)
    reference_counts = np.zeros(n_core_cells + len(new_cells), dtype=np.int32)
    for face_id, face in enumerate(source_faces):
        centre = points[face].mean(axis=0)
        owner = int(source_owners[face_id])
        cell_references[owner] += centre
        reference_counts[owner] += 1
        if face_id < n_internal:
            neighbour = int(source_neighbours[face_id])
            cell_references[neighbour] += centre
            reference_counts[neighbour] += 1
    for local_cell, entries in enumerate(new_cells):
        unique_points = np.unique(np.concatenate([entry[0] for entry in entries]))
        cell_id = n_core_cells + local_cell
        cell_references[cell_id] = points[unique_points].mean(axis=0)
        reference_counts[cell_id] = 1
    if np.any(reference_counts == 0):
        raise ValueError("cfMesh wrapper encountered a cell without a geometric reference")
    cell_references /= reference_counts[:, None]

    def orient(face: np.ndarray, owner: int, neighbour: int | None = None) -> np.ndarray:
        centre = points[face].mean(axis=0)
        direction = (
            cell_references[neighbour] - cell_references[owner]
            if neighbour is not None
            else centre - cell_references[owner]
        )
        if float(np.dot(_area_vector(points[face]), direction)) < 0.0:
            return face[::-1].copy()
        return face.copy()

    # Existing core faces already obey OpenFOAM's owner-to-neighbour
    # orientation. Re-estimating that orientation from cell references is
    # unstable for the very small pyramids created during edge extraction.
    internal_faces = [face.copy() for face in source_faces[:n_internal]]
    internal_owners = list(map(int, source_owners[:n_internal]))
    internal_neighbours = list(map(int, source_neighbours))
    for face_id, layer_cell in interface_cells:
        internal_faces.append(source_faces[face_id].copy())
        internal_owners.append(int(source_owners[face_id]))
        internal_neighbours.append(layer_cell)
    boundary_faces: list[list[np.ndarray]] = [[] for _patch in patches]
    boundary_owners: list[list[int]] = [[] for _patch in patches]
    for signature, face_records in records.items():
        if len(face_records) == 2:
            first, second = face_records
            if first[2] is not None or second[2] is not None:
                raise ValueError(f"cfMesh wrapper boundary face was duplicated: {signature}")
            internal_faces.append(orient(first[0], first[1], second[1]))
            internal_owners.append(first[1])
            internal_neighbours.append(second[1])
        elif len(face_records) == 1:
            face, owner, boundary_patch = face_records[0]
            if boundary_patch is None:
                raise ValueError(f"cfMesh wrapper has an unmatched internal face: {signature}")
            boundary_faces[boundary_patch].append(
                face.copy() if signature in layer_outer_signatures else orient(face, owner)
            )
            boundary_owners[boundary_patch].append(owner)
        else:
            raise ValueError(f"cfMesh wrapper produced a non-manifold face: {signature}")

    combined_faces = internal_faces.copy()
    combined_owners = internal_owners.copy()
    combined_neighbours = internal_neighbours.copy()
    boundary: list[dict[str, Any]] = []
    start_face = len(internal_faces)
    for patch_id, patch in enumerate(patches):
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

    widths = {len(face) for face in combined_faces}
    n_new_cells = len(new_cells)
    old_levels = np.asarray(mesh_data.get("cell_levels", np.zeros(n_core_cells)), dtype=np.int8)
    old_sizes = np.asarray(mesh_data.get("cell_sizes", np.ones(n_core_cells)), dtype=np.float32)
    source_ids = np.asarray(source_cell_for_new, dtype=np.int32)
    face_by_signature = {
        tuple(sorted(map(int, face))): face_id for face_id, face in enumerate(combined_faces)
    }
    source_order = mesh_data.get("_cfmesh_cell_face_order")
    if source_order is None:
        source_order = [[] for _cell in range(n_core_cells)]
        for face_id, owner_value in enumerate(source_owners):
            source_order[int(owner_value)].append(face_id)
            if face_id < n_internal:
                source_order[int(source_neighbours[face_id])].append(face_id)
    cfmesh_cell_face_order = [
        [
            face_by_signature[tuple(sorted(map(int, source_faces[source_face_id])))]
            for source_face_id in cell
        ]
        for cell in source_order
    ]
    for local_cell, entries in enumerate(new_cells):
        ordered_faces: list[np.ndarray] = []
        if local_cell < len(interface_cells):
            ordered_faces.append(source_faces[interface_cells[local_cell][0]])
        ordered_faces.extend(face for face, _boundary_patch in entries)
        cfmesh_cell_face_order.append(
            [face_by_signature[tuple(sorted(map(int, face)))] for face in ordered_faces]
        )
    mesh_data.update(
        {
            "vertex_position": points,
            "faces": (
                np.ascontiguousarray(combined_faces, dtype=np.int32)
                if len(widths) == 1
                else combined_faces
            ),
            "owners": np.ascontiguousarray(combined_owners, dtype=np.int32),
            "neighbours": np.ascontiguousarray(combined_neighbours, dtype=np.int32),
            "boundary": boundary,
            "n_cells": n_core_cells + n_new_cells,
            "n_faces": len(combined_faces),
            "n_interior_faces": len(internal_faces),
            "n_points": len(points),
            "cell_levels": np.concatenate((old_levels, old_levels[source_ids])),
            "cell_sizes": np.concatenate((old_sizes, old_sizes[source_ids])),
            "cell_type_code": np.full(n_core_cells + n_new_cells, 5, dtype=np.int32),
            "boundary_layer_index": np.concatenate(
                (
                    np.full(n_core_cells, -1, dtype=np.int16),
                    np.zeros(n_new_cells, dtype=np.int16),
                )
            ),
            "_cfmesh_cell_face_order": cfmesh_cell_face_order,
        }
    )
    for stale in (
        "cell_vertex_indices",
        "cell_face_indices",
        "cell_face_offset",
        "global_cell_id",
        "global_face_id",
    ):
        mesh_data.pop(stale, None)
    mesh_data["mesh_generation"]["workflow_checkpoint"] = "boundaryLayerGeneration"
    mesh_data["mesh_generation"]["cfmesh_wrapper_layer"] = {
        "face_cells": len(interface_cells),
        "edge_cells": feature_edges,
        "corner_cells": corner_points,
        "new_cells": n_new_cells,
        "new_points": len(points) - len(source_points),
        "core_points": len(source_points),
        "patch_keys": {str(patches[patch_id]["name"]): key for patch_id, key in patch_key.items()},
        "concave_patch_pairs": [
            [str(patches[first]["name"]), str(patches[second]["name"])]
            for first, second in concave_patch_pairs
        ],
    }


__all__ = ["add_cfmesh_wrapper_layer"]
