# SPDX-License-Identifier: GPL-3.0-or-later
"""Finite-volume smoothing used by cfMesh's ``meshOptimisation`` stage."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .cfmesh_surface_optimisation import (
    _face_area_vector,
    _face_centre,
    optimise_cfmesh_surface,
)

_SMALL = 1.0e-15
_VSMALL = 1.0e-300
_ROOT_VSMALL = 1.0e-150


def _mesh_addressing(
    faces: list[np.ndarray],
    owners: np.ndarray,
    neighbours: np.ndarray,
    n_cells: int,
    n_points: int,
    *,
    cell_face_order: Sequence[Sequence[int]] | None = None,
) -> tuple[list[list[int]], list[set[int]]]:
    cell_faces = (
        [[] for _cell in range(n_cells)]
        if cell_face_order is None
        else [list(map(int, face_ids)) for face_ids in cell_face_order]
    )
    if len(cell_faces) != n_cells:
        raise ValueError("cfMesh cell-face order does not match the cell count")
    expected_faces: list[set[int]] | None = (
        None if cell_face_order is None else [set() for _cell in range(n_cells)]
    )
    point_cells: list[set[int]] = [set() for _point in range(n_points)]
    for face_id, face in enumerate(faces):
        owner = int(owners[face_id])
        if expected_faces is None:
            cell_faces[owner].append(face_id)
        else:
            expected_faces[owner].add(face_id)
        neighbour = int(neighbours[face_id]) if face_id < len(neighbours) else None
        if neighbour is not None:
            if expected_faces is None:
                cell_faces[neighbour].append(face_id)
            else:
                expected_faces[neighbour].add(face_id)
        for point_value in face:
            point_id = int(point_value)
            point_cells[point_id].add(owner)
            if neighbour is not None:
                point_cells[point_id].add(neighbour)
    if expected_faces is not None and any(
        len(face_ids) != len(set(face_ids)) or set(face_ids) != expected_faces[cell_id]
        for cell_id, face_ids in enumerate(cell_faces)
    ):
        raise ValueError("cfMesh cell-face order is inconsistent with mesh topology")
    return cell_faces, point_cells


def _face_geometry(points: np.ndarray, faces: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([_face_centre(points[face]) for face in faces]),
        np.asarray([_face_area_vector(points[face]) for face in faces]),
    )


def _cfmesh_cell_centres(
    points: np.ndarray,
    faces: list[np.ndarray],
    owners: np.ndarray,
    neighbours: np.ndarray,
    n_cells: int,
    *,
    cell_face_order: Sequence[Sequence[int]] | None = None,
) -> np.ndarray:
    """Match ``polyMeshGenAddressing::makeCellCentresAndVols``."""
    face_centres, face_areas = _face_geometry(points, faces)
    cell_faces, _point_cells = _mesh_addressing(
        faces,
        owners,
        neighbours,
        n_cells,
        len(points),
        cell_face_order=cell_face_order,
    )
    centres = np.zeros((n_cells, 3), dtype=np.float64)
    for cell_id, face_ids in enumerate(cell_faces):
        selected = np.asarray(face_ids, dtype=np.int32)
        estimate = face_centres[selected].mean(axis=0)
        volumes3 = np.einsum(
            "ij,ij->i",
            face_areas[selected],
            face_centres[selected] - estimate,
        )
        signs = np.where(owners[selected] == cell_id, 1.0, -1.0)
        volumes3 = np.maximum(signs * volumes3, _VSMALL)
        pyramid_centres = 0.75 * face_centres[selected] + 0.25 * estimate
        centres[cell_id] = np.einsum("i,ij->j", volumes3, pyramid_centres) / volumes3.sum()
    return centres


def _cfmesh_bad_faces(
    points: np.ndarray,
    faces: list[np.ndarray],
    owners: np.ndarray,
    neighbours: np.ndarray,
    n_cells: int,
    *,
    active_faces: np.ndarray | None = None,
    cell_face_order: Sequence[Sequence[int]] | None = None,
) -> set[int]:
    """Return cfMesh's default invalid-face set.

    With no ``meshQualitySettings`` dictionary, ``findBadFaces`` combines
    pyramid orientation, face flatness, part-cell tetrahedra, and face area.
    The practical failures in the Cartesian workflow are the signed
    part-tetrahedron checks, but all four defaults are retained here.
    """
    face_centres, face_areas = _face_geometry(points, faces)
    cell_centres = _cfmesh_cell_centres(
        points,
        faces,
        owners,
        neighbours,
        n_cells,
        cell_face_order=cell_face_order,
    )
    selected = (
        np.arange(len(faces), dtype=np.int32)
        if active_faces is None
        else np.flatnonzero(active_faces)
    )
    bad: set[int] = set()
    for face_value in selected:
        face_id = int(face_value)
        face = faces[face_id]
        centre = face_centres[face_id]
        area = face_areas[face_id]
        owner = int(owners[face_id])

        # pyramidPointFaceRef has the opposite sign to the stored face area
        # for the owner side.  Testing the equivalent dot products avoids a
        # second polygon triangulation and matches the VSMALL=1e-300 gate.
        if float(np.dot(area, centre - cell_centres[owner])) <= 0.0:
            bad.add(face_id)
        if face_id < len(neighbours):
            neighbour = int(neighbours[face_id])
            if float(np.dot(area, centre - cell_centres[neighbour])) >= 0.0:
                bad.add(face_id)

        coordinates = points[face]
        area_magnitude = float(np.linalg.norm(area))
        if area_magnitude < _VSMALL:
            bad.add(face_id)
        if len(face) > 3 and area_magnitude > _VSMALL:
            following = np.roll(coordinates, -1, axis=0)
            triangle_areas = 0.5 * np.linalg.norm(
                np.cross(following - coordinates, centre - coordinates), axis=1
            )
            if area_magnitude / (float(triangle_areas.sum()) + _VSMALL) < 0.8:
                bad.add(face_id)

        for edge_index in range(len(face)):
            current = points[int(face[edge_index])]
            following = points[int(face[(edge_index + 1) % len(face)])]
            owner_volume = float(
                np.dot(
                    np.cross(following - centre, current - centre),
                    cell_centres[owner] - centre,
                )
                / 6.0
            )
            if owner_volume < _VSMALL:
                bad.add(face_id)
            if face_id < len(neighbours):
                neighbour_volume = float(
                    np.dot(
                        np.cross(current - centre, following - centre),
                        cell_centres[int(neighbours[face_id])] - centre,
                    )
                    / 6.0
                )
                if neighbour_volume < _VSMALL:
                    bad.add(face_id)
    return bad


def _cfmesh_low_quality_faces(
    points: np.ndarray,
    faces: list[np.ndarray],
    owners: np.ndarray,
    neighbours: np.ndarray,
    n_cells: int,
    *,
    active_faces: np.ndarray | None = None,
    cell_face_order: Sequence[Sequence[int]] | None = None,
) -> set[int]:
    """Return faces exceeding cfMesh's 70-degree or 2.0 skew gates."""
    face_centres, face_areas = _face_geometry(points, faces)
    cell_centres = _cfmesh_cell_centres(
        points,
        faces,
        owners,
        neighbours,
        n_cells,
        cell_face_order=cell_face_order,
    )
    selected = (
        np.arange(len(faces), dtype=np.int32)
        if active_faces is None
        else np.flatnonzero(active_faces)
    )
    non_orthogonal_limit = float(np.cos(np.deg2rad(70.0)))
    bad: set[int] = set()
    for face_value in selected:
        face_id = int(face_value)
        owner = int(owners[face_id])
        face_centre = face_centres[face_id]
        if face_id < len(neighbours):
            neighbour = int(neighbours[face_id])
            delta = cell_centres[neighbour] - cell_centres[owner]
            area = face_areas[face_id]
            dot_product = float(
                np.dot(delta, area) / (np.linalg.norm(delta) * np.linalg.norm(area) + _VSMALL)
            )
            if dot_product < non_orthogonal_limit:
                bad.add(face_id)
            owner_distance = float(np.linalg.norm(face_centre - cell_centres[owner]))
            neighbour_distance = float(np.linalg.norm(face_centre - cell_centres[neighbour]))
            intersection = (
                cell_centres[owner] * neighbour_distance + cell_centres[neighbour] * owner_distance
            ) / (owner_distance + neighbour_distance)
            skewness = float(
                np.linalg.norm(face_centre - intersection) / (np.linalg.norm(delta) + _VSMALL)
            )
        else:
            delta = face_centre - cell_centres[owner]
            area = face_areas[face_id]
            magnitude = float(np.linalg.norm(area))
            if magnitude <= _VSMALL:
                continue
            normal = area / magnitude
            normal_delta = float(np.dot(normal, delta)) * normal
            skewness = float(
                np.linalg.norm(delta - normal_delta) / (np.linalg.norm(delta) + _VSMALL)
            )
        if skewness > 2.0:
            bad.add(face_id)
    return bad


@dataclass(slots=True)
class _PartTetMesh:
    """The local tetrahedral decomposition used by cfMesh's smoothers."""

    points: np.ndarray
    tets: np.ndarray
    smooth_nodes: np.ndarray
    boundary_nodes: np.ndarray
    node_to_original: np.ndarray
    face_centre_nodes: dict[int, int]
    cell_centre_nodes: dict[int, int]
    point_tets: list[list[int]]


def _tet_vertices_for_cell(
    cell_id: int,
    face_ids: list[int],
    faces: list[np.ndarray],
    owners: np.ndarray,
) -> tuple[int, int, int, int] | None:
    if len(face_ids) != 4 or any(len(faces[face_id]) != 3 for face_id in face_ids):
        return None
    first_face_id = face_ids[0]
    first = faces[first_face_id]
    if int(owners[first_face_id]) == cell_id:
        base = (int(first[0]), int(first[2]), int(first[1]))
    else:
        base = (int(first[0]), int(first[1]), int(first[2]))
    vertices = set(map(int, np.concatenate([faces[face_id] for face_id in face_ids])))
    remaining = vertices.difference(base)
    if len(vertices) != 4 or len(remaining) != 1:
        return None
    return (*base, remaining.pop())


def _selected_cells(
    bad_faces: set[int],
    faces: list[np.ndarray],
    point_cells: list[set[int]],
    cell_faces: list[list[int]],
    additional_layers: int,
) -> np.ndarray:
    levels = np.zeros(len(cell_faces), dtype=np.uint8)
    for face_id in bad_faces:
        for point_value in faces[face_id]:
            for cell_id in point_cells[int(point_value)]:
                levels[cell_id] = 1
    for layer in range(1, additional_layers + 1):
        for cell_id in np.flatnonzero(levels == layer):
            for face_id in cell_faces[int(cell_id)]:
                for point_value in faces[face_id]:
                    for neighbour_cell in point_cells[int(point_value)]:
                        if levels[neighbour_cell] == 0:
                            levels[neighbour_cell] = layer + 1
    return levels != 0


def _build_part_tet_mesh(
    points: np.ndarray,
    faces: list[np.ndarray],
    owners: np.ndarray,
    neighbours: np.ndarray,
    n_cells: int,
    bad_faces: set[int],
    additional_layers: int,
    *,
    cell_face_order: Sequence[Sequence[int]] | None = None,
) -> _PartTetMesh:
    cell_faces, point_cells = _mesh_addressing(
        faces,
        owners,
        neighbours,
        n_cells,
        len(points),
        cell_face_order=cell_face_order,
    )
    use_cell = _selected_cells(bad_faces, faces, point_cells, cell_faces, additional_layers)
    used_faces = np.zeros(len(faces), dtype=np.uint8)
    for face_id, owner_value in enumerate(owners):
        used_faces[face_id] += int(use_cell[int(owner_value)])
        if face_id < len(neighbours):
            used_faces[face_id] += int(use_cell[int(neighbours[face_id])])

    local_points: list[np.ndarray] = []
    node_to_original: list[int] = []
    node_for_point: dict[int, int] = {}
    face_centre_nodes: dict[int, int] = {}
    cell_centre_nodes: dict[int, int] = {}
    node_types: list[int] = []
    face_centres, _face_areas = _face_geometry(points, faces)
    cell_centres = _cfmesh_cell_centres(
        points,
        faces,
        owners,
        neighbours,
        n_cells,
        cell_face_order=cell_face_order,
    )
    boundary_start = len(neighbours)

    def add_face(face_id: int, point_type: int) -> None:
        if used_faces[face_id] == 0:
            return
        face = faces[face_id]
        if len(face) > 3:
            face_centre_nodes[face_id] = len(local_points)
            local_points.append(face_centres[face_id].copy())
            node_to_original.append(-1)
            node_types.append(4)  # FACECENTRE
        for point_value in face:
            point_id = int(point_value)
            if point_id not in node_for_point:
                node_for_point[point_id] = len(local_points)
                local_points.append(points[point_id].copy())
                node_to_original.append(point_id)
                node_types.append(point_type)
            elif point_type == 0:
                node_types[node_for_point[point_id]] = 0

    for face_id in range(boundary_start, len(faces)):
        add_face(face_id, 2)  # BOUNDARY
    for face_id in range(boundary_start):
        add_face(face_id, 1 if used_faces[face_id] == 2 else 0)  # SMOOTH/NONE

    tets: list[tuple[int, int, int, int]] = []
    for cell_value in np.flatnonzero(use_cell):
        cell_id = int(cell_value)
        tet_vertices = _tet_vertices_for_cell(cell_id, cell_faces[cell_id], faces, owners)
        if tet_vertices is not None:
            tets.append(
                (
                    node_for_point[tet_vertices[0]],
                    node_for_point[tet_vertices[1]],
                    node_for_point[tet_vertices[2]],
                    node_for_point[tet_vertices[3]],
                )
            )
            continue
        centre_node = len(local_points)
        cell_centre_nodes[cell_id] = centre_node
        local_points.append(cell_centres[cell_id].copy())
        node_to_original.append(-1)
        node_types.append(8)  # CELLCENTRE
        for face_id in cell_faces[cell_id]:
            face = faces[face_id]
            if len(face) == 3:
                if int(owners[face_id]) == cell_id:
                    order = (int(face[0]), int(face[2]), int(face[1]))
                else:
                    order = (int(face[0]), int(face[1]), int(face[2]))
                tets.append(
                    (
                        node_for_point[order[0]],
                        node_for_point[order[1]],
                        node_for_point[order[2]],
                        centre_node,
                    )
                )
                continue
            face_centre_node = face_centre_nodes[face_id]
            for index, point_value in enumerate(face):
                if int(owners[face_id]) == cell_id:
                    other_value = face[(index - 1) % len(face)]
                else:
                    other_value = face[(index + 1) % len(face)]
                tets.append(
                    (
                        node_for_point[int(point_value)],
                        node_for_point[int(other_value)],
                        face_centre_node,
                        centre_node,
                    )
                )

    tet_array = np.asarray(tets, dtype=np.int32)
    point_tets: list[list[int]] = [[] for _point in local_points]
    for tet_id, tet in enumerate(tet_array):
        for node_value in tet:
            point_tets[int(node_value)].append(tet_id)
    types = np.asarray(node_types, dtype=np.uint8)
    return _PartTetMesh(
        points=np.asarray(local_points, dtype=np.float64),
        tets=tet_array,
        smooth_nodes=np.flatnonzero(types == 1).astype(np.int32),
        boundary_nodes=np.flatnonzero(types == 2).astype(np.int32),
        node_to_original=np.asarray(node_to_original, dtype=np.int32),
        face_centre_nodes=face_centre_nodes,
        cell_centre_nodes=cell_centre_nodes,
        point_tets=point_tets,
    )


def _signed_tet_volumes(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    coordinates = points[tets]
    return (
        np.einsum(
            "ij,ij->i",
            np.cross(coordinates[:, 1] - coordinates[:, 0], coordinates[:, 2] - coordinates[:, 0]),
            coordinates[:, 3] - coordinates[:, 0],
        )
        / 6.0
    )


def _simplex_triangles(part: _PartTetMesh, node_id: int) -> np.ndarray:
    triangles: list[tuple[int, int, int]] = []
    for tet_id in part.point_tets[node_id]:
        tet = tuple(map(int, part.tets[tet_id]))
        position = tet.index(node_id)
        if position == 0:
            triangles.append((tet[1], tet[3], tet[2]))
        elif position == 1:
            triangles.append((tet[0], tet[2], tet[3]))
        elif position == 2:
            triangles.append((tet[0], tet[3], tet[1]))
        else:
            triangles.append((tet[0], tet[1], tet[2]))
    return np.asarray(triangles, dtype=np.int32)


def _volume_stabilisation(points: np.ndarray, triangles: np.ndarray, point: np.ndarray) -> float:
    values = points[triangles]
    volumes = (
        np.einsum(
            "ij,ij->i",
            np.cross(values[:, 1] - values[:, 0], values[:, 2] - values[:, 0]),
            point - values[:, 0],
        )
        / 6.0
    )
    lengths = (
        np.sum((point - values[:, 0]) ** 2, axis=1)
        + np.sum((point - values[:, 1]) ** 2, axis=1)
        + np.sum((point - values[:, 2]) ** 2, axis=1)
    )
    if float(volumes.min()) < _SMALL * float(lengths.max()):
        return _SMALL * float(lengths.max())
    return 0.0


def _volume_objective(points: np.ndarray, triangles: np.ndarray, point: np.ndarray) -> float:
    values = points[triangles]
    volumes = (
        np.einsum(
            "ij,ij->i",
            np.cross(values[:, 1] - values[:, 0], values[:, 2] - values[:, 0]),
            point - values[:, 0],
        )
        / 6.0
    )
    lengths = (
        np.sum((point - values[:, 0]) ** 2, axis=1)
        + np.sum((point - values[:, 1]) ** 2, axis=1)
        + np.sum((point - values[:, 2]) ** 2, axis=1)
    )
    stabilisation = _volume_stabilisation(points, triangles, point)
    stable_volumes = 0.5 * (volumes + np.sqrt(volumes * volumes + stabilisation))
    stable_volumes = np.maximum(stable_volumes, _ROOT_VSMALL)
    return float(np.sum(lengths / np.power(stable_volumes, 2.0 / 3.0)))


def _volume_gradients(
    points: np.ndarray, triangles: np.ndarray, point: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    gradient = np.zeros(3, dtype=np.float64)
    hessian = np.zeros((3, 3), dtype=np.float64)
    values = points[triangles]
    stabilisation = _volume_stabilisation(points, triangles, point)
    constant = (2.0 / 3.0) * np.power(0.5, 2.0 / 3.0)
    for triangle in values:
        a, b, c = triangle
        volume_gradient = np.cross(b - a, c - a) / 6.0
        volume = float(np.dot(volume_gradient, point - a))
        length_squared = float(
            np.dot(point - a, point - a)
            + np.dot(point - b, point - b)
            + np.dot(point - c, point - c)
        )
        stable = float(np.sqrt(volume * volume + stabilisation))
        stable_volume = max(_ROOT_VSMALL, 0.5 * (volume + stable))
        stable_gradient = 0.5 * (volume_gradient + volume * volume_gradient / stable)
        length_gradient = 2.0 * (3.0 * point - a - b - c)
        root_volume = np.power(2.0 * stable_volume, 1.0 / 3.0)
        volume_power = np.power(stable_volume, 2.0 / 3.0)
        volume_power_squared = volume_power * volume_power
        power_gradient = constant * (2.0 * stable_gradient) / root_volume
        gradient += (
            length_gradient / volume_power - length_squared * power_gradient / volume_power_squared
        )
        stable_hessian = (
            np.outer(volume_gradient, volume_gradient) / stable
            - volume * volume * np.outer(volume_gradient, volume_gradient) / stable**3
        )
        power_hessian = (
            constant * stable_hessian / root_volume
            - (constant / 3.0) * 4.0 * np.outer(stable_gradient, stable_gradient) / root_volume**4
        )
        hessian += (
            6.0 * np.eye(3) / volume_power
            - (
                np.outer(length_gradient, power_gradient)
                + np.outer(power_gradient, length_gradient)
            )
            / volume_power_squared
            - length_squared * power_hessian / volume_power_squared
            + 2.0
            * length_squared
            * np.outer(power_gradient, power_gradient)
            / (volume_power_squared * volume_power)
        )
    return gradient, hessian


def _optimise_volume_point(
    points: np.ndarray, triangles: np.ndarray, point: np.ndarray, tolerance: float = 1.0e-5
) -> np.ndarray:
    neighbours = points[triangles.ravel()]
    lower = neighbours.min(axis=0)
    upper = neighbours.max(axis=0)
    scale = float(np.linalg.norm(upper - lower))
    if scale <= _VSMALL:
        return point.copy()
    values = points / scale
    lower = lower / scale
    upper = upper / scale
    candidate = point.copy() / scale
    if np.any(candidate < lower) or np.any(candidate > upper):
        candidate = 0.5 * (lower + upper)
    candidate = 0.5 * (lower + upper)
    current = candidate.copy()
    half_range = 0.5 * (upper - lower)
    before = _volume_objective(values, triangles, candidate)
    after = before
    directions = np.asarray(
        [
            (-1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (-1.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
        ]
    )
    for _iteration in range(100):
        before = after
        best_value = np.inf
        best_point = current.copy()
        for direction in directions:
            trial = current + 0.5 * direction * half_range
            value = _volume_objective(values, triangles, trial)
            if value < best_value:
                best_value = value
                best_point = trial
        current = best_point
        candidate = best_point.copy()
        half_range *= 0.5
        after = best_value
        if abs(after - before) / after < tolerance:
            break
    divide_point = candidate.copy()
    divide_value = after

    after = _volume_objective(values, triangles, candidate)
    for _iteration in range(100):
        original = candidate.copy()
        before = after
        gradient, hessian = _volume_gradients(values, triangles, candidate)
        determinant = float(np.linalg.det(hessian))
        finished = False
        if determinant > _SMALL:
            displacement = np.linalg.solve(hessian, gradient)
            candidate -= displacement
            after = _volume_objective(values, triangles, candidate)
            relaxation = 0.8
            loops = 0
            while after > before:
                candidate = original - relaxation * displacement
                relaxation *= 0.5
                after = _volume_objective(values, triangles, candidate)
                if after < before:
                    continue
                loops += 1
                if loops == 5:
                    candidate = original
                    displacement = np.zeros(3)
                    after = before
                    finished = True
            if abs(before - after) / before < tolerance:
                finished = True
        else:
            displacement = np.zeros(3, dtype=np.float64)
            triangle_values = values[triangles]
            volumes = (
                np.einsum(
                    "ij,ij->i",
                    np.cross(
                        triangle_values[:, 1] - triangle_values[:, 0],
                        triangle_values[:, 2] - triangle_values[:, 0],
                    ),
                    candidate - triangle_values[:, 0],
                )
                / 6.0
            )
            for triangle, volume in zip(triangle_values, volumes, strict=True):
                if volume < _SMALL:
                    normal = 0.5 * np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
                    magnitude = float(np.linalg.norm(normal))
                    if magnitude > _VSMALL:
                        displacement += 0.01 * normal / magnitude
            candidate += displacement
            after = _volume_objective(values, triangles, candidate)
        if finished:
            break
    if after > divide_value:
        candidate = divide_point
    return candidate * scale


def _refresh_part_centres(part: _PartTetMesh, changed_nodes: Iterable[int]) -> None:
    """Update only auxiliary centres attached to changed native vertices."""
    face_centre_ids = set(part.face_centre_nodes.values())
    cell_centre_ids = set(part.cell_centre_nodes.values())
    update_faces: set[int] = set()
    update_cells: set[int] = set()
    for changed_value in changed_nodes:
        for tet_id in part.point_tets[int(changed_value)]:
            tet = part.tets[tet_id]
            face_centre = int(tet[2])
            cell_centre = int(tet[3])
            if cell_centre in cell_centre_ids:
                update_cells.add(cell_centre)
            if face_centre in face_centre_ids:
                update_faces.add(face_centre)

    # updateVerticesSMP traverses its point array in label order.  In a
    # partTetMesh face-centre labels precede cell-centre labels, so the latter
    # observe the newly refreshed face centres.
    for node_id in sorted(update_faces):
        tet_ids = np.asarray(part.point_tets[node_id], dtype=np.int32)
        values = part.points[part.tets[tet_ids]]
        # updateVerticesSMP intentionally uses tet[0:3], which includes the
        # old face-centre at tet[2].  This is one relaxation step, not a fresh
        # polygon-centroid evaluation.
        centroids = values[:, :3].mean(axis=1)
        areas = (
            0.5
            * np.linalg.norm(
                np.cross(values[:, 2] - values[:, 0], values[:, 1] - values[:, 0]),
                axis=1,
            )
            + _VSMALL
        )
        part.points[node_id] = np.einsum("i,ij->j", areas, centroids) / areas.sum()
    for node_id in sorted(update_cells):
        tet_ids = np.asarray(part.point_tets[node_id], dtype=np.int32)
        tets = part.tets[tet_ids]
        values = part.points[tets]
        centroids = values.mean(axis=1)
        volumes = np.abs(_signed_tet_volumes(part.points, tets)) + _VSMALL
        part.points[node_id] = np.einsum("i,ij->j", volumes, centroids) / volumes.sum()


def _knupp_objective(
    point: np.ndarray,
    normals: np.ndarray,
    centres: np.ndarray,
    beta: float,
) -> float:
    values = (normals @ point) - np.einsum("ij,ij->i", normals, centres) - beta
    return float(np.sum((np.abs(values) - values) ** 2))


def _knupp_point(points: np.ndarray, triangles: np.ndarray, point: np.ndarray) -> np.ndarray:
    values = points[triangles]
    raw_normals = 0.5 * np.cross(values[:, 1] - values[:, 0], values[:, 2] - values[:, 0])
    magnitudes = np.linalg.norm(raw_normals, axis=1)
    valid = magnitudes > _VSMALL
    if not np.any(valid):
        return point.copy()
    normals = raw_normals[valid] / magnitudes[valid, None]
    centres = values[valid].mean(axis=1)
    lower = values.reshape((-1, 3)).min(axis=0)
    upper = values.reshape((-1, 3)).max(axis=0)
    candidate = point.copy()
    if np.any(candidate < lower) or np.any(candidate > upper):
        candidate = 0.5 * (lower + upper)
    beta = 0.01 * float(np.linalg.norm(upper - lower))
    tolerance = (2.0 * _SMALL) ** 2 * float(np.dot(upper - lower, upper - lower))
    for _outer in range(5):
        previous = _knupp_objective(candidate, normals, centres, beta)
        displacement = np.zeros(3, dtype=np.float64)
        for _iteration in range(10):
            original = candidate.copy()
            offsets = (normals @ candidate) - np.einsum("ij,ij->i", normals, centres) - beta
            metric_gradients = (np.sign(offsets) - 1.0)[:, None] * normals
            gradient = np.einsum("i,ij->j", np.abs(offsets) - offsets, metric_gradients)
            hessian = np.einsum("ij,ik->jk", metric_gradients, metric_gradients)
            determinant = float(np.linalg.det(hessian))
            if determinant > _SMALL:
                displacement = np.linalg.solve(hessian, gradient)
                if not np.isfinite(displacement).all():
                    displacement = np.zeros(3, dtype=np.float64)
                candidate -= displacement
                current = _knupp_objective(candidate, normals, centres, beta)
                relaxation = 0.8
                loops = 0
                while current > previous:
                    candidate = original - relaxation * displacement
                    relaxation *= 0.5
                    current = _knupp_objective(candidate, normals, centres, beta)
                    if current < previous:
                        continue
                    loops += 1
                    if loops == 5:
                        candidate = original
                        displacement = np.zeros(3, dtype=np.float64)
                        current = 0.0
                previous = current
            else:
                displacement = np.zeros(3, dtype=np.float64)
            if float(np.dot(displacement, displacement)) <= tolerance:
                break
        no_beta = _knupp_objective(candidate, normals, centres, 0.0)
        if not (previous < _VSMALL and no_beta > _VSMALL):
            break
        beta *= 0.5
    return candidate


def _optimise_part_knupp(part: _PartTetMesh) -> None:
    inverted = _signed_tet_volumes(part.points, part.tets) < _VSMALL
    if not np.any(inverted):
        return
    negative_nodes = set(map(int, part.tets[inverted].ravel()))
    updates = {
        int(node_value): _knupp_point(
            part.points,
            _simplex_triangles(part, int(node_value)),
            part.points[int(node_value)],
        )
        for node_value in part.smooth_nodes
        if int(node_value) in negative_nodes
    }
    for node_id, position in updates.items():
        part.points[node_id] = position
    _refresh_part_centres(part, updates)


@dataclass(slots=True)
class _CutRegion:
    """Topology-preserving port of ``meshUntangler::cutRegion``."""

    points: list[np.ndarray]
    edges: list[tuple[int, int]]
    faces: list[list[int]]
    tolerance: float
    valid: bool = True
    new_vertex_labels: list[int] | None = None
    vertex_distances: list[float] | None = None
    vertex_types: list[int] | None = None
    new_edge_labels: list[int] | None = None
    candidate_points: list[np.ndarray] | None = None
    candidate_edges: list[tuple[int, int]] | None = None

    @classmethod
    def from_bounds(cls, lower: np.ndarray, upper: np.ndarray) -> _CutRegion:
        centre = 0.5 * (lower + upper)
        half = 0.5 * (upper - lower)
        points = [
            centre + half * np.asarray(direction, dtype=np.float64)
            for direction in (
                (-1.0, -1.0, -1.0),
                (1.0, -1.0, -1.0),
                (1.0, 1.0, -1.0),
                (-1.0, 1.0, -1.0),
                (-1.0, -1.0, 1.0),
                (1.0, -1.0, 1.0),
                (1.0, 1.0, 1.0),
                (-1.0, 1.0, 1.0),
            )
        ]
        edges = [
            (0, 1),
            (3, 2),
            (7, 6),
            (4, 5),
            (1, 2),
            (0, 3),
            (4, 7),
            (5, 6),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]
        faces = [
            [5, 11, 6, 8],
            [4, 10, 7, 9],
            [0, 8, 3, 9],
            [1, 11, 2, 10],
            [0, 4, 1, 5],
            [3, 7, 2, 6],
        ]
        return cls(points, edges, faces, _SMALL * float(np.linalg.norm(upper - lower)))

    def _find_new_vertices(self, reference: np.ndarray, normal: np.ndarray) -> bool:
        self.new_vertex_labels = [-1] * len(self.points)
        self.vertex_distances = []
        self.vertex_types = [0] * len(self.points)
        self.candidate_points = []
        for point_id, point in enumerate(self.points):
            distance = float(np.dot(point - reference, normal))
            if distance > self.tolerance:
                self.new_vertex_labels[point_id] = len(self.candidate_points)
                self.candidate_points.append(point.copy())
                self.vertex_types[point_id] = 1  # KEEP
            elif distance >= -self.tolerance:
                self.new_vertex_labels[point_id] = len(self.candidate_points)
                self.candidate_points.append(point.copy())
                self.vertex_types[point_id] = 2  # INPLANE
                distance = 0.0
            self.vertex_distances.append(distance)
        if len(self.candidate_points) < len(self.points):
            return True
        self.candidate_points = None
        return False

    def _find_new_edges(self) -> None:
        assert self.new_vertex_labels is not None
        assert self.vertex_distances is not None
        assert self.vertex_types is not None
        assert self.candidate_points is not None
        self.candidate_edges = []
        self.new_edge_labels = [-1] * len(self.edges)
        for edge_id, (start, end) in enumerate(self.edges):
            new_start = self.new_vertex_labels[start]
            new_end = self.new_vertex_labels[end]
            if new_start != -1 and new_end != -1:
                self.new_edge_labels[edge_id] = len(self.candidate_edges)
                self.candidate_edges.append((new_start, new_end))
            elif new_end != -1 and self.vertex_types[end] & 1:
                self.new_edge_labels[edge_id] = len(self.candidate_edges)
                self.candidate_edges.append((new_end, len(self.candidate_points)))
                parameter = -self.vertex_distances[start] / (
                    self.vertex_distances[end] - self.vertex_distances[start]
                )
                self.candidate_points.append(
                    (1.0 - parameter) * self.points[start] + parameter * self.points[end]
                )
            elif new_start != -1 and self.vertex_types[start] & 1:
                self.new_edge_labels[edge_id] = len(self.candidate_edges)
                self.candidate_edges.append((new_start, len(self.candidate_points)))
                parameter = -self.vertex_distances[end] / (
                    self.vertex_distances[start] - self.vertex_distances[end]
                )
                self.candidate_points.append(
                    (1.0 - parameter) * self.points[end] + parameter * self.points[start]
                )

    @staticmethod
    def _edge_chain(
        face_edges: list[tuple[int, int]],
    ) -> list[int] | None:
        if not face_edges:
            return None
        unused = list(range(len(face_edges)))
        first = unused.pop(0)
        start, end = face_edges[first]
        chain = [start, end]
        while unused:
            found = None
            for position, edge_id in enumerate(unused):
                edge = face_edges[edge_id]
                if edge[0] == chain[-1]:
                    found = (position, edge[1])
                    break
                if edge[1] == chain[-1]:
                    found = (position, edge[0])
                    break
            if found is None:
                return None
            position, next_point = found
            unused.pop(position)
            chain.append(next_point)
        if chain[-1] != chain[0]:
            return None
        return chain[:-1]

    def _tie_break(self, face: list[int]) -> None:
        assert self.vertex_types is not None
        assert self.vertex_distances is not None
        face_vertices = self._edge_chain([self.edges[edge_id] for edge_id in face])
        if face_vertices is None:
            self.valid = False
            return
        regions = [0] * len(face_vertices)
        region = 1
        for index, point_id in enumerate(face_vertices):
            if self.vertex_types[point_id] or regions[index]:
                continue
            regions[index] = region
            forward = (index + 1) % len(face_vertices)
            reverse = (index - 1) % len(face_vertices)
            found = True
            while found:
                found = False
                if not self.vertex_types[face_vertices[forward]]:
                    regions[forward] = region
                    forward = (forward + 1) % len(face_vertices)
                    found = True
                if not self.vertex_types[face_vertices[reverse]]:
                    regions[reverse] = region
                    reverse = (reverse - 1) % len(face_vertices)
                    found = True
            region += 1
        if region > 2:
            minimum_distance = np.inf
            minimum_region = -1
            for index, point_id in enumerate(face_vertices):
                if regions[index] and self.vertex_distances[point_id] < minimum_distance:
                    minimum_distance = self.vertex_distances[point_id]
                    minimum_region = regions[index]
            for index, point_id in enumerate(face_vertices):
                if regions[index] and regions[index] != minimum_region:
                    self.vertex_types[point_id] |= 2
        else:
            for index, point_id in enumerate(face_vertices):
                if (
                    self.vertex_types[point_id] & 2
                    and not regions[(index - 1) % len(face_vertices)]
                    and not regions[(index + 1) % len(face_vertices)]
                ):
                    self.vertex_types[point_id] ^= 2
                    self.vertex_types[point_id] |= 1
        self.candidate_points = []
        self.new_vertex_labels = [-1] * len(self.points)
        for point_id, point_type in enumerate(self.vertex_types):
            if point_type:
                self.new_vertex_labels[point_id] = len(self.candidate_points)
                self.candidate_points.append(self.points[point_id].copy())
        self._find_new_edges()

    def _find_new_faces(self) -> list[list[int]] | None:
        while True:
            assert self.candidate_points is not None
            assert self.candidate_edges is not None
            assert self.new_edge_labels is not None
            candidate_faces: list[list[int]] = []
            restart = False
            for face in self.faces:
                usage = [0] * len(self.candidate_points)
                new_face: list[int] = []
                for old_edge_id in face:
                    edge_id = self.new_edge_labels[old_edge_id]
                    if edge_id == -1:
                        continue
                    start, end = self.candidate_edges[edge_id]
                    usage[start] += 1
                    usage[end] += 1
                    new_face.append(edge_id)
                if len(new_face) <= 1:
                    continue
                loose_points = [point_id for point_id, count in enumerate(usage) if count == 1]
                if len(loose_points) == 2:
                    new_face.append(len(self.candidate_edges))
                    self.candidate_edges.append((loose_points[0], loose_points[1]))
                elif len(loose_points) > 2:
                    self._tie_break(face)
                    if not self.valid:
                        return None
                    restart = True
                    break
                candidate_faces.append(new_face)
            if restart:
                continue
            edge_usage = [0] * len(self.candidate_edges)
            for face in candidate_faces:
                for edge_id in face:
                    edge_usage[edge_id] += 1
            plane_face = [edge_id for edge_id, count in enumerate(edge_usage) if count == 1]
            if len(plane_face) > 2:
                candidate_faces.append(plane_face)
            return candidate_faces

    def cut(self, reference: np.ndarray, normal: np.ndarray) -> None:
        if not self.valid:
            return
        magnitude = float(np.linalg.norm(normal))
        if magnitude <= _VSMALL:
            return
        normal = normal / magnitude
        if not self._find_new_vertices(reference, normal):
            return
        self._find_new_edges()
        candidate_faces = self._find_new_faces()
        if candidate_faces is None:
            return
        assert self.candidate_points is not None
        assert self.candidate_edges is not None
        self.points = self.candidate_points
        self.edges = self.candidate_edges
        self.faces = candidate_faces
        self.candidate_points = None
        self.candidate_edges = None


def _untangle_point(points: np.ndarray, triangles: np.ndarray) -> np.ndarray | None:
    """Return the vertex centroid of cfMesh's feasible cut region."""
    values = points[triangles]
    lower = values.reshape((-1, 3)).min(axis=0)
    upper = values.reshape((-1, 3)).max(axis=0)
    region = _CutRegion.from_bounds(lower, upper)
    for triangle in values:
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        if float(np.linalg.norm(normal)) > _VSMALL:
            region.cut(triangle[0], normal)
    if not region.points:
        return None
    return np.asarray(region.points).mean(axis=0)


def _optimise_part_untangler(part: _PartTetMesh, iterations: int = 5) -> None:
    del iterations  # Native implementation clears its active set after one pass.
    inverted = _signed_tet_volumes(part.points, part.tets) < _VSMALL
    if not np.any(inverted):
        return
    negative_nodes = set(map(int, part.tets[inverted].ravel()))
    updates: dict[int, np.ndarray] = {}
    for node_value in part.smooth_nodes:
        node_id = int(node_value)
        if node_id not in negative_nodes:
            continue
        position = _untangle_point(part.points, _simplex_triangles(part, node_id))
        if position is not None and np.isfinite(position).all():
            updates[node_id] = position
    for node_id, position in updates.items():
        part.points[node_id] = position
    _refresh_part_centres(part, updates)


def _optimise_part_volume(part: _PartTetMesh, iterations: int = 10) -> None:
    for _iteration in range(iterations):
        updates = {
            int(node_value): _optimise_volume_point(
                part.points,
                _simplex_triangles(part, int(node_value)),
                part.points[int(node_value)],
            )
            for node_value in part.smooth_nodes
        }
        for node_id, position in updates.items():
            part.points[node_id] = position
        _refresh_part_centres(part, updates)


def _optimise_part_boundary_volume(
    part: _PartTetMesh, *, iterations: int = 3, non_shrinking: bool = True
) -> None:
    for _iteration in range(iterations):
        updates: dict[int, np.ndarray] = {}
        for node_value in part.boundary_nodes:
            node_id = int(node_value)
            triangles = _simplex_triangles(part, node_id)
            candidate = _optimise_volume_point(part.points, triangles, part.points[node_id])
            if not non_shrinking:
                updates[node_id] = candidate
                continue
            edge_counts: dict[tuple[int, int], int] = {}
            for triangle in triangles:
                for index in range(3):
                    start = int(triangle[index])
                    end = int(triangle[(index + 1) % 3])
                    edge = (min(start, end), max(start, end))
                    edge_counts[edge] = edge_counts.get(edge, 0) + 1
            normal_tensor = np.zeros((3, 3), dtype=np.float64)
            for (start, end), count in edge_counts.items():
                if count != 1:
                    continue
                normal = np.cross(
                    part.points[end] - part.points[start],
                    part.points[node_id] - part.points[start],
                )
                magnitude = float(np.linalg.norm(normal))
                if magnitude > _VSMALL:
                    normal /= magnitude
                    normal_tensor += np.outer(normal, normal)
            eigenvalues, eigenvectors = np.linalg.eigh(normal_tensor)
            displacement = candidate - part.points[node_id]
            if abs(float(eigenvalues[2])) > abs(float(eigenvalues[1])) + abs(float(eigenvalues[0])):
                normal = eigenvectors[:, 2]
                displacement -= float(np.dot(displacement, normal)) * normal
            elif abs(float(eigenvalues[1])) > 0.5 * (
                abs(float(eigenvalues[2])) + abs(float(eigenvalues[0]))
            ):
                edge_direction = np.cross(eigenvectors[:, 1], eigenvectors[:, 2])
                edge_direction /= np.linalg.norm(edge_direction) + _VSMALL
                displacement = float(np.dot(displacement, edge_direction)) * edge_direction
            else:
                continue
            updates[node_id] = part.points[node_id] + displacement
        for node_id, position in updates.items():
            part.points[node_id] = position
        _refresh_part_centres(part, updates)


def _update_mesh_from_part(
    points: np.ndarray,
    part: _PartTetMesh,
    faces: list[np.ndarray],
    point_cells: list[set[int]],
    cell_faces: list[list[int]],
) -> np.ndarray:
    changed_points: set[int] = set()
    for node_id, original_value in enumerate(part.node_to_original):
        original_id = int(original_value)
        if original_id >= 0:
            points[original_id] = part.points[node_id]
            changed_points.add(original_id)
    changed_faces = np.zeros(len(faces), dtype=np.bool_)
    for point_id in changed_points:
        for cell_id in point_cells[point_id]:
            changed_faces[np.asarray(cell_faces[cell_id], dtype=np.int32)] = True
    return changed_faces


def _optimise_part_boundary_laplace(part: _PartTetMesh, *, iterations: int = 3) -> None:
    for _iteration in range(iterations):
        updates: dict[int, np.ndarray] = {}
        for node_value in part.boundary_nodes:
            node_id = int(node_value)
            triangles = _simplex_triangles(part, node_id)
            edge_counts: dict[tuple[int, int], int] = {}
            for triangle in triangles:
                for index in range(3):
                    start = int(triangle[index])
                    end = int(triangle[(index + 1) % 3])
                    edge = (min(start, end), max(start, end))
                    edge_counts[edge] = edge_counts.get(edge, 0) + 1
            triangle_centres = [
                (part.points[start] + part.points[end] + part.points[node_id]) / 3.0
                for (start, end), count in edge_counts.items()
                if count == 1
            ]
            if triangle_centres:
                updates[node_id] = np.asarray(triangle_centres).mean(axis=0)
        for node_id, position in updates.items():
            part.points[node_id] = position
        _refresh_part_centres(part, updates)


def _run_cfmesh_untangle(
    points: np.ndarray,
    faces: list[np.ndarray],
    owners: np.ndarray,
    neighbours: np.ndarray,
    n_cells: int,
    point_cells: list[set[int]],
    cell_faces: list[list[int]],
    *,
    cell_face_order: Sequence[Sequence[int]] | None = None,
    max_global_iterations: int = 10,
    max_internal_iterations: int = 50,
    max_surface_iterations: int = 2,
) -> tuple[list[int], list[int]]:
    """Port ``meshOptimizer::untangleMeshFV`` and return count traces."""
    changed_faces = np.ones(len(faces), dtype=np.bool_)
    internal_trace: list[int] = []
    boundary_trace: list[int] = []
    global_iteration = 0
    n_bad_faces = 0

    while True:
        iteration = 0
        minimum_bad_faces = 10 * len(faces)
        minimum_iteration = -1
        while True:
            bad_faces = _cfmesh_bad_faces(
                points,
                faces,
                owners,
                neighbours,
                n_cells,
                active_faces=changed_faces,
                cell_face_order=cell_face_order,
            )
            n_bad_faces = len(bad_faces)
            internal_trace.append(n_bad_faces)
            if n_bad_faces == 0:
                break
            if n_bad_faces < minimum_bad_faces:
                minimum_bad_faces = n_bad_faces
                minimum_iteration = iteration

            part = _build_part_tet_mesh(
                points,
                faces,
                owners,
                neighbours,
                n_cells,
                bad_faces,
                (global_iteration // 2) + 1,
                cell_face_order=cell_face_order,
            )
            _optimise_part_knupp(part)
            _optimise_part_untangler(part)
            _optimise_part_volume(part)
            changed_faces = _update_mesh_from_part(points, part, faces, point_cells, cell_faces)

            if not (iteration < minimum_iteration + 5 and iteration + 1 < max_internal_iterations):
                break
            iteration += 1

        if n_bad_faces == 0:
            break
        global_iteration += 1
        if global_iteration >= max_global_iterations:
            break

        for _surface_iteration in range(max_surface_iterations):
            bad_faces = _cfmesh_bad_faces(
                points,
                faces,
                owners,
                neighbours,
                n_cells,
                active_faces=changed_faces,
                cell_face_order=cell_face_order,
            )
            n_bad_faces = len(bad_faces)
            boundary_trace.append(n_bad_faces)
            if n_bad_faces == 0:
                break
            part = _build_part_tet_mesh(
                points,
                faces,
                owners,
                neighbours,
                n_cells,
                bad_faces,
                0,
                cell_face_order=cell_face_order,
            )
            if global_iteration < 2:
                _optimise_part_boundary_volume(part, non_shrinking=True)
            elif global_iteration < 5:
                _optimise_part_boundary_laplace(part)
            else:
                _optimise_part_boundary_volume(part, non_shrinking=False)
            changed_faces = _update_mesh_from_part(points, part, faces, point_cells, cell_faces)

        if n_bad_faces == 0:
            break

    return internal_trace, boundary_trace


def _run_cfmesh_low_quality(
    points: np.ndarray,
    faces: list[np.ndarray],
    owners: np.ndarray,
    neighbours: np.ndarray,
    n_cells: int,
    point_cells: list[set[int]],
    cell_faces: list[list[int]],
    *,
    cell_face_order: Sequence[Sequence[int]] | None = None,
    max_iterations: int = 10,
) -> list[int]:
    """Port ``meshOptimizer::optimizeLowQualityFaces``."""
    trace: list[int] = []
    for _iteration in range(max_iterations):
        low_quality_faces = _cfmesh_low_quality_faces(
            points,
            faces,
            owners,
            neighbours,
            n_cells,
            cell_face_order=cell_face_order,
        )
        trace.append(len(low_quality_faces))
        if not low_quality_faces:
            break
        part = _build_part_tet_mesh(
            points,
            faces,
            owners,
            neighbours,
            n_cells,
            low_quality_faces,
            2,
            cell_face_order=cell_face_order,
        )
        _optimise_part_volume(part)
        _update_mesh_from_part(points, part, faces, point_cells, cell_faces)
    return trace


def optimise_cfmesh_mesh(
    mesh_data: dict[str, Any],
    *,
    iterations: int = 5,
    map_edge_points: Callable[[Sequence[int]], None] | None = None,
    untangle_surface: Callable[[], list[int]] | None = None,
) -> None:
    """Reproduce cfMesh's active final finite-volume optimization path.

    This follows ``cartesianMeshGenerator``: surface optimization, point-to-
    cell-centre Laplacian smoothing, bad-face untangling, low-quality-face
    optimization, and the final bad-face untangling pass.  The boundary-layer
    optimizer is intentionally absent because the parity dictionary contains
    no ``boundaryLayers`` sub-dictionary, matching cfMesh's own early return.
    """
    optimise_cfmesh_surface(
        mesh_data,
        map_edge_points=map_edge_points,
        untangle_surface=untangle_surface,
    )
    points = np.asarray(mesh_data["vertex_position"], dtype=np.float64)
    faces = [np.asarray(face, dtype=np.int32) for face in mesh_data["faces"]]
    owners = np.asarray(mesh_data["owners"], dtype=np.int32)
    neighbours = np.asarray(mesh_data["neighbours"], dtype=np.int32)
    n_internal = int(mesh_data["n_interior_faces"])
    n_cells = int(mesh_data["n_cells"])
    cell_face_order = mesh_data.get("_cfmesh_cell_face_order")

    boundary_points = set(map(int, np.unique(np.concatenate(faces[n_internal:]))))
    smooth_points = np.asarray(
        [point_id for point_id in range(len(points)) if point_id not in boundary_points],
        dtype=np.int32,
    )
    cell_faces, point_cells = _mesh_addressing(
        faces,
        owners,
        neighbours,
        n_cells,
        len(points),
        cell_face_order=cell_face_order,
    )
    if any(not point_cells[int(point_id)] for point_id in smooth_points):
        raise ValueError("cfMesh mesh optimization found an inside point without incident cells")

    for _iteration in range(iterations):
        cell_centres = _cfmesh_cell_centres(
            points,
            faces,
            owners,
            neighbours,
            n_cells,
            cell_face_order=cell_face_order,
        )
        updates = np.asarray(
            [
                cell_centres[np.asarray(sorted(point_cells[int(point_id)]), dtype=np.int32)].mean(
                    axis=0
                )
                for point_id in smooth_points
            ],
            dtype=np.float64,
        )
        points[smooth_points] = updates

    bad_after_laplacian = _cfmesh_bad_faces(
        points,
        faces,
        owners,
        neighbours,
        n_cells,
        cell_face_order=cell_face_order,
    )
    first_internal_trace, first_boundary_trace = _run_cfmesh_untangle(
        points,
        faces,
        owners,
        neighbours,
        n_cells,
        point_cells,
        cell_faces,
        cell_face_order=cell_face_order,
    )
    bad_after_optimize_fv = _cfmesh_bad_faces(
        points,
        faces,
        owners,
        neighbours,
        n_cells,
        cell_face_order=cell_face_order,
    )

    low_quality_trace = _run_cfmesh_low_quality(
        points,
        faces,
        owners,
        neighbours,
        n_cells,
        point_cells,
        cell_faces,
        cell_face_order=cell_face_order,
    )
    low_quality_after_optimisation = _cfmesh_low_quality_faces(
        points,
        faces,
        owners,
        neighbours,
        n_cells,
        cell_face_order=cell_face_order,
    )

    final_internal_trace, final_boundary_trace = _run_cfmesh_untangle(
        points,
        faces,
        owners,
        neighbours,
        n_cells,
        point_cells,
        cell_faces,
        cell_face_order=cell_face_order,
    )
    final_bad_faces = _cfmesh_bad_faces(
        points,
        faces,
        owners,
        neighbours,
        n_cells,
        cell_face_order=cell_face_order,
    )
    final_low_quality_faces = _cfmesh_low_quality_faces(
        points,
        faces,
        owners,
        neighbours,
        n_cells,
        cell_face_order=cell_face_order,
    )

    mesh_data.pop("cell_face_indices", None)
    mesh_data.pop("cell_face_offset", None)
    mesh_data["mesh_generation"]["workflow_checkpoint"] = "meshOptimisation"
    mesh_data["mesh_generation"]["cfmesh_mesh_optimisation"] = {
        "laplacian_pc_iterations": iterations,
        "smooth_inside_points": len(smooth_points),
        "bad_faces_after_laplacian": len(bad_after_laplacian),
        "bad_face_ids_after_laplacian": sorted(bad_after_laplacian),
        "optimize_fv_internal_bad_face_trace": first_internal_trace,
        "optimize_fv_boundary_bad_face_trace": first_boundary_trace,
        "bad_faces_after_optimize_fv": len(bad_after_optimize_fv),
        "low_quality_face_trace": low_quality_trace,
        "low_quality_faces_after_optimisation": len(low_quality_after_optimisation),
        "final_internal_bad_face_trace": final_internal_trace,
        "final_boundary_bad_face_trace": final_boundary_trace,
        "final_bad_faces": len(final_bad_faces),
        "final_bad_face_ids": sorted(final_bad_faces),
        "final_low_quality_faces": len(final_low_quality_faces),
        "final_low_quality_face_ids": sorted(final_low_quality_faces),
        "boundary_layer_optimisation": "skipped_no_boundary_layers_dictionary",
    }


__all__ = ["optimise_cfmesh_mesh"]
