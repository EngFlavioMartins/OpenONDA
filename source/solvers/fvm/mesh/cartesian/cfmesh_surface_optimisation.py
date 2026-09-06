# SPDX-License-Identifier: GPL-3.0-or-later
"""Surface optimisation used by cfMesh's ``edgeExtraction`` checkpoint."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

_SMALL = 1.0e-15
_VSMALL = 1.0e-300
_ROOT_VSMALL = 1.0e-150


def _mag_squared(value: np.ndarray) -> float:
    """Use OpenFOAM's component order, without a BLAS reduction."""
    return float(value[0] * value[0] + value[1] * value[1] + value[2] * value[2])


def _dot(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Scalar-order dot products, also accepting an array of row vectors."""
    return first[..., 0] * second[0] + first[..., 1] * second[1] + first[..., 2] * second[2]


def _relax_face_centre(triangles: np.ndarray) -> np.ndarray:
    """One partTriMesh auxiliary-centre refresh in stored triangle order."""
    weighted = np.zeros(3, dtype=np.float64)
    area_sum = 0.0
    for a, b, c in triangles:
        centre = (a + b + c) / 3.0
        normal = 0.5 * np.cross(b - a, c - a)
        area = float(np.sqrt(_mag_squared(normal))) + _VSMALL
        weighted += centre * area
        area_sum += area
    return weighted / area_sum


def _face_centre(coordinates: np.ndarray) -> np.ndarray:
    count = len(coordinates)
    if count == 3:
        return (1.0 / 3.0) * (coordinates[0] + coordinates[1] + coordinates[2])
    centre = np.zeros(3, dtype=np.float64)
    for coordinate in coordinates:
        centre += coordinate
    centre /= count
    area_sum = 0.0
    weighted = np.zeros(3, dtype=np.float64)
    for position, coordinate in enumerate(coordinates):
        following = coordinates[(position + 1) % count]
        first = coordinate - centre
        second = following - centre
        cross = np.asarray(
            (
                first[1] * second[2] - first[2] * second[1],
                first[2] * second[0] - first[0] * second[2],
                first[0] * second[1] - first[1] * second[0],
            ),
            dtype=np.float64,
        )
        twice_area = float(np.sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]))
        area_sum += twice_area
        weighted += twice_area * (coordinate + following + centre)
    return weighted / (3.0 * area_sum) if area_sum > _VSMALL else centre


def _face_area_vector(coordinates: np.ndarray) -> np.ndarray:
    count = len(coordinates)
    if count == 3:
        first = coordinates[1] - coordinates[0]
        second = coordinates[2] - coordinates[0]
        return 0.5 * np.asarray(
            (
                first[1] * second[2] - first[2] * second[1],
                first[2] * second[0] - first[0] * second[2],
                first[0] * second[1] - first[1] * second[0],
            ),
            dtype=np.float64,
        )
    centre = np.zeros(3, dtype=np.float64)
    for coordinate in coordinates:
        centre += coordinate
    centre /= count
    area = np.zeros(3, dtype=np.float64)
    for position, coordinate in enumerate(coordinates):
        following = coordinates[(position + 1) % count]
        first = following - coordinate
        second = centre - coordinate
        area += 0.5 * np.asarray(
            (
                first[1] * second[2] - first[2] * second[1],
                first[2] * second[0] - first[0] * second[2],
                first[0] * second[1] - first[1] * second[0],
            ),
            dtype=np.float64,
        )
    return area


def _stabilisation(points: np.ndarray, triangles: np.ndarray) -> float:
    # Keep cfMesh's scalar, triangle-by-triangle reduction order.  Symmetric
    # simplexes can have equal minima to machine precision, so NumPy's pairwise
    # reductions can select the reflected optimisation branch.
    minimum_area = 1.0e300
    maximum_length_squared = 0.0
    for triangle in triangles:
        p0, p1, p2 = points[triangle]
        area = 0.5 * ((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]))
        first = p0 - p1
        second = p2 - p0
        length_squared = _mag_squared(first) + _mag_squared(second)
        minimum_area = min(minimum_area, float(area))
        maximum_length_squared = max(maximum_length_squared, length_squared)
    if minimum_area < _SMALL * maximum_length_squared:
        return _SMALL * maximum_length_squared
    return 0.0


def _objective(points: np.ndarray, triangles: np.ndarray, stabilisation: float) -> float:
    value = 0.0
    for triangle in triangles:
        p0, p1, p2 = points[triangle]
        area = 0.5 * ((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]))
        stable = float(np.sqrt(area * area + stabilisation))
        denominator = max(_VSMALL, 0.5 * (area + stable))
        first = p0 - p1
        second = p2 - p0
        length_squared = _mag_squared(first) + _mag_squared(second)
        value += length_squared / denominator
    return value


def _gradients(
    points: np.ndarray, triangles: np.ndarray, stabilisation: float
) -> tuple[np.ndarray, np.ndarray]:
    gradient = np.zeros(2, dtype=np.float64)
    hessian = np.zeros((2, 2), dtype=np.float64)
    for triangle in triangles:
        p0, p1, p2 = points[triangle]
        if _mag_squared(p1 - p2) < _VSMALL:
            continue
        length_squared = _mag_squared(p0 - p1) + _mag_squared(p2 - p0)
        area = 0.5 * ((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]))
        stable = float(np.sqrt(area * area + stabilisation))
        stable_area = max(_ROOT_VSMALL, 0.5 * (area + stable))
        area_gradient = np.asarray(
            (0.5 * (p1[1] - p2[1]), 0.5 * (p2[0] - p1[0])),
            dtype=np.float64,
        )
        area_outer = np.outer(area_gradient, area_gradient)
        stable_gradient = 0.5 * (area_gradient + area * area_gradient / stable)
        stable_hessian = 0.5 * (area_outer / stable - area * area * area_outer / stable**3)
        length_gradient = (4.0 * p0 - 2.0 * p1 - 2.0 * p2)[:2]
        stable_area_squared = stable_area * stable_area
        gradient += (
            length_gradient / stable_area - length_squared * stable_gradient / stable_area_squared
        )
        hessian += (
            4.0 * np.eye(2) / stable_area
            - (
                np.outer(length_gradient, stable_gradient)
                + np.outer(stable_gradient, length_gradient)
            )
            / stable_area_squared
            - stable_hessian * length_squared / stable_area_squared
            + 2.0
            * length_squared
            * np.outer(stable_gradient, stable_gradient)
            / (stable_area_squared * stable_area)
        )
    if abs(float(hessian[0, 0])) < _VSMALL:
        hessian[0, 0] = _VSMALL
    if abs(float(hessian[1, 1])) < _VSMALL:
        hessian[1, 1] = _VSMALL
    return gradient, hessian


def _optimise_point(
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    tolerance: float = 0.001,
) -> np.ndarray:
    target = int(triangles[0, 0])
    neighbour_values = points[triangles[:, 1:].ravel()]
    lower = neighbour_values.min(axis=0)
    upper = neighbour_values.max(axis=0)
    scale = float(np.sqrt(_mag_squared(upper - lower)))
    if scale <= _VSMALL:
        return points[target].copy()
    values = points.copy() / scale
    lower /= scale
    upper /= scale

    values[target] = 0.5 * (upper + lower)
    current = values[target].copy()
    dx = 0.5 * float(upper[0] - lower[0])
    dy = 0.5 * float(upper[1] - lower[1])
    stabilisation = _stabilisation(values, triangles)
    before = _objective(values, triangles, stabilisation)
    directions = ((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0))
    divide_value = before
    for _iteration in range(100):
        best_value = 1.0e300
        best = np.zeros(3, dtype=np.float64)
        for x_direction, y_direction in directions:
            values[target, 0] = current[0] + 0.5 * x_direction * dx
            values[target, 1] = current[1] + 0.5 * y_direction * dy
            stabilisation = _stabilisation(values, triangles)
            value = _objective(values, triangles, stabilisation)
            if value < best_value:
                best = values[target].copy()
                best_value = value
        current = best
        values[target] = best
        dx *= 0.5
        dy *= 0.5
        divide_value = best_value
        if abs(best_value - before) / best_value < tolerance:
            break
        before = best_value
    divide_point = values[target].copy()

    average_edge = float(np.sqrt(_mag_squared(upper - lower)))
    stabilisation = _stabilisation(values, triangles)
    before = _objective(values, triangles, stabilisation)
    steepest_value = before
    for _iteration in range(100):
        gradient, hessian = _gradients(values, triangles, stabilisation)
        determinant = float(hessian[0, 0] * hessian[1, 1] - hessian[0, 1] * hessian[1, 0])
        if abs(determinant) < _VSMALL:
            displacement = np.zeros(2, dtype=np.float64)
        else:
            # cfMesh's matrix2D uses Cramer's rule, not a pivoted LAPACK solve.
            displacement = np.asarray(
                (
                    (hessian[1, 1] * gradient[0] - hessian[0, 1] * gradient[1]) / determinant,
                    (-hessian[1, 0] * gradient[0] + hessian[0, 0] * gradient[1]) / determinant,
                )
            )
            magnitude = float(np.sqrt(displacement[0] ** 2 + displacement[1] ** 2))
            if magnitude > 0.2 * average_edge:
                displacement = (displacement / magnitude) * 0.2 * average_edge
        values[target, :2] -= displacement
        stabilisation = _stabilisation(values, triangles)
        steepest_value = _objective(values, triangles, stabilisation)
        if abs(steepest_value - before) / before < tolerance:
            break
        before = steepest_value
    if steepest_value > divide_value:
        values[target] = divide_point
    return values[target] * scale


def _smooth_partition_points(
    mesh_data: dict[str, Any],
    partition_points: Sequence[int],
    *,
    iterations: int,
    auxiliary_state: dict[str, np.ndarray] | None = None,
) -> None:
    """Run cfMesh's face-centre Laplacian and objective smoother."""
    points = np.asarray(mesh_data["vertex_position"], dtype=np.float64)
    faces = [np.asarray(face, dtype=np.int32) for face in mesh_data["faces"]]
    boundary_start = int(mesh_data["n_interior_faces"])
    point_faces: dict[int, list[int]] = defaultdict(list)
    for face_id in range(boundary_start, len(faces)):
        for point_id_value in faces[face_id]:
            point_faces[int(point_id_value)].append(face_id)

    n_points = len(points)
    global_triangles: list[tuple[int, int, int]] = []
    point_triangle_ids: dict[int, list[int]] = defaultdict(list)
    for face_id in range(boundary_start, len(faces)):
        face = faces[face_id]
        auxiliary = n_points + face_id - boundary_start
        face_triangles: list[tuple[int, int, int]] = []
        if len(face) == 3:
            face_triangles.append((int(face[0]), int(face[1]), int(face[2])))
        for position, point_id_value in enumerate(face):
            point_id = int(point_id_value)
            following = int(face[(position + 1) % len(face)])
            previous = int(face[(position - 1) % len(face)])
            if len(face) > 3:
                face_triangles.append((point_id, following, auxiliary))
            face_triangles.append((point_id, following, previous))
        for triangle in face_triangles:
            triangle_id = len(global_triangles)
            global_triangles.append(triangle)
            for vertex in triangle:
                if vertex < n_points:
                    point_triangle_ids[vertex].append(triangle_id)

    auxiliary_centres: np.ndarray | None = (
        auxiliary_state.get("centres") if auxiliary_state is not None else None
    )
    triangulation_points: np.ndarray | None = (
        auxiliary_state.get("points") if auxiliary_state is not None else None
    )
    for _iteration in range(iterations):
        face_centres = {
            face_id: _face_centre(points[faces[face_id]])
            for face_id in range(boundary_start, len(faces))
        }
        face_normals = {
            face_id: _face_area_vector(points[faces[face_id]])
            for face_id in range(boundary_start, len(faces))
        }
        laplacian_updates: dict[int, np.ndarray] = {}
        for point_id in partition_points:
            normal = sum(
                (face_normals[face_id] for face_id in point_faces[point_id]),
                np.zeros(3, dtype=np.float64),
            )
            length = float(np.sqrt(_mag_squared(normal)))
            if length <= _VSMALL:
                continue
            normal /= length
            # plane(point, pointNormal) normalises the unit normal again.
            normal /= float(np.sqrt(_mag_squared(normal)))
            projected = [
                centre - normal * float(_dot(centre - points[point_id], normal))
                for face_id in point_faces[point_id]
                for centre in (face_centres[face_id],)
            ]
            laplacian_updates[point_id] = np.asarray(projected).mean(axis=0)
        for point_id, value in laplacian_updates.items():
            points[point_id] = value

        if triangulation_points is None:
            triangulation_points = points.copy()
        else:
            selected_ids = np.asarray(partition_points, dtype=np.int64)
            triangulation_points[selected_ids] = points[selected_ids]
        if auxiliary_state is not None:
            auxiliary_state["points"] = triangulation_points

        if auxiliary_centres is None:
            auxiliary_centres = np.asarray(
                [
                    _face_centre(points[faces[face_id]])
                    for face_id in range(boundary_start, len(faces))
                ],
                dtype=np.float64,
            )
        updated_auxiliary = auxiliary_centres.copy()
        partition_point_set = set(partition_points)
        for local_face_id, face_id in enumerate(range(boundary_start, len(faces))):
            face = faces[face_id]
            if not partition_point_set.intersection(map(int, face)):
                continue
            centre = auxiliary_centres[local_face_id]
            following = np.roll(face, -1)
            triangles = np.stack(
                (
                    triangulation_points[face],
                    triangulation_points[following],
                    np.broadcast_to(centre, (len(face), 3)),
                ),
                axis=1,
            )
            updated_auxiliary[local_face_id] = _relax_face_centre(triangles)
        auxiliary_centres = updated_auxiliary
        if auxiliary_state is not None:
            auxiliary_state["centres"] = auxiliary_centres

        face_normals = {
            face_id: _face_area_vector(points[faces[face_id]])
            for face_id in range(boundary_start, len(faces))
        }
        point_normals: dict[int, np.ndarray] = {}
        for point_id in partition_points:
            normal = sum(
                (face_normals[face_id] for face_id in point_faces[point_id]),
                np.zeros(3, dtype=np.float64),
            )
            length = float(np.sqrt(_mag_squared(normal)))
            if length > _VSMALL:
                point_normals[point_id] = normal / length

        optimisation_updates: dict[int, np.ndarray] = {}
        for point_id in partition_points:
            normal = point_normals.get(point_id)
            if normal is None:
                continue
            normal = normal / float(np.sqrt(_mag_squared(normal)))
            local_labels: list[int] = []
            local_index: dict[int, int] = {}
            local_triangles: list[tuple[int, int, int]] = []
            for triangle_id in point_triangle_ids[point_id]:
                triangle = global_triangles[triangle_id]
                for vertex in triangle:
                    if vertex not in local_index:
                        local_index[vertex] = len(local_labels)
                        local_labels.append(vertex)
                position = triangle.index(point_id)
                if triangle[2] < n_points and position != 0:
                    continue
                rotated = triangle[position:] + triangle[:position]
                local_triangles.append(
                    (
                        local_index[rotated[0]],
                        local_index[rotated[1]],
                        local_index[rotated[2]],
                    )
                )
            if not local_triangles:
                continue
            centre_point = points[point_id]
            local_coordinates = np.asarray(
                [
                    triangulation_points[label]
                    if label < n_points
                    else auxiliary_centres[label - n_points]
                    for label in local_labels
                ]
            )
            vector_x: np.ndarray | None = None
            for coordinate in local_coordinates:
                projected = coordinate - normal * float(_dot(coordinate - centre_point, normal))
                offset = projected - centre_point
                length = float(np.sqrt(_mag_squared(offset)))
                if length > _VSMALL:
                    vector_x = offset / length
                    break
            if vector_x is None:
                continue
            vector_y = np.cross(normal, vector_x)
            vector_y /= float(np.sqrt(_mag_squared(vector_y)))
            offsets = local_coordinates - centre_point
            planar = np.column_stack(
                (
                    _dot(offsets, vector_x),
                    _dot(offsets, vector_y),
                    np.zeros(len(offsets), dtype=np.float64),
                )
            )
            new_planar = _optimise_point(planar, np.asarray(local_triangles, dtype=np.int32))
            optimisation_updates[point_id] = (
                centre_point + vector_x * new_planar[0] + vector_y * new_planar[1]
            )
        for point_id, value in optimisation_updates.items():
            points[point_id] = value


def _inverted_boundary_points(
    mesh_data: dict[str, Any],
    face_patch_ids: Sequence[int] | np.ndarray | None = None,
    active_points: set[int] | None = None,
) -> set[int]:
    """Return the serial partition-point subset rejected by cfMesh's check."""
    points = np.asarray(mesh_data["vertex_position"], dtype=np.float64)
    faces = [np.asarray(face, dtype=np.int32) for face in mesh_data["faces"]]
    boundary_start = int(mesh_data["n_interior_faces"])
    if len(faces) - boundary_start > 10_000:
        return _inverted_boundary_points_vectorized(
            points,
            faces[boundary_start:],
            None if face_patch_ids is None else np.asarray(face_patch_ids),
            active_points,
        )
    point_faces: dict[int, list[tuple[int, int]]] = defaultdict(list)
    face_centres: dict[int, np.ndarray] = {}
    face_normals: dict[int, np.ndarray] = {}
    for face_id in range(boundary_start, len(faces)):
        face = faces[face_id]
        face_centres[face_id] = _face_centre(points[face])
        face_normals[face_id] = _face_area_vector(points[face])
        for position, point_id_value in enumerate(face):
            point_faces[int(point_id_value)].append((face_id, position))

    inverted: set[int] = set()
    for point_id, incidences in point_faces.items():
        if active_points is not None and point_id not in active_points:
            continue
        point_normal = sum(
            (face_normals[face_id] for face_id, _position in incidences),
            np.zeros(3, dtype=np.float64),
        )
        length = float(np.linalg.norm(point_normal))
        if length <= _VSMALL:
            inverted.add(point_id)
            continue
        point_normal /= length
        for face_id, position in incidences:
            if face_patch_ids is not None:
                patch_id = int(face_patch_ids[face_id - boundary_start])
                point_normal = sum(
                    (
                        face_normals[other_face]
                        for other_face, _other_position in incidences
                        if int(face_patch_ids[other_face - boundary_start]) == patch_id
                    ),
                    np.zeros(3, dtype=np.float64),
                )
                patch_length = float(np.linalg.norm(point_normal))
                if patch_length <= _VSMALL:
                    inverted.add(point_id)
                    break
                point_normal /= patch_length
            face = faces[face_id]
            point = points[point_id]
            following = points[int(face[(position + 1) % len(face)])]
            previous = points[int(face[(position - 1) % len(face)])]
            centre = face_centres[face_id]
            next_normal = np.cross(following - point, centre - point)
            previous_normal = np.cross(centre - point, previous - point)
            next_length = float(np.linalg.norm(next_normal))
            previous_length = float(np.linalg.norm(previous_normal))
            if next_length <= _VSMALL or previous_length <= _VSMALL:
                inverted.add(point_id)
                break
            next_normal /= next_length
            previous_normal /= previous_length
            if (
                float(np.dot(next_normal, point_normal)) < 0.0
                or float(np.dot(previous_normal, point_normal)) < 0.0
                or float(np.dot(next_normal, previous_normal)) < 0.0
            ):
                inverted.add(point_id)
                break
    for face_id in range(boundary_start, len(faces)):
        face = faces[face_id]
        face_normal = face_normals[face_id]
        normal_length = float(np.linalg.norm(face_normal))
        if normal_length <= _VSMALL:
            inverted.update(map(int, face))
            continue
        face_normal /= normal_length
        for position, point_id_value in enumerate(face):
            point_id = int(point_id_value)
            if active_points is not None and point_id not in active_points:
                continue
            current = points[point_id]
            following = points[int(face[(position + 1) % len(face)])]
            previous = points[int(face[(position - 1) % len(face)])]
            previous_edge = current - previous
            following_edge = following - current
            previous_edge /= max(float(np.linalg.norm(previous_edge)), _VSMALL)
            following_edge /= max(float(np.linalg.norm(following_edge)), _VSMALL)
            corner_normal = np.cross(previous_edge, following_edge)
            if float(np.dot(corner_normal, face_normal)) < -0.05:
                inverted.add(point_id)
    return inverted


def _inverted_boundary_points_vectorized(
    points: np.ndarray,
    faces: list[np.ndarray],
    face_patch_ids: np.ndarray | None,
    active_points: set[int] | None,
) -> set[int]:
    """Vectorized equivalent of the partition-point orientation predicate.

    Large reference meshes have tens of thousands of boundary polygons.  The
    scalar cfMesh transcription is useful for checkpoint parity, but its
    Python face/corner loop becomes the dominant cost at release scale.  Keep
    the same two predicates while evaluating all polygon corners in NumPy.
    """
    widths = np.asarray([len(face) for face in faces], dtype=np.int64)
    if not len(widths):
        return set()
    starts = np.concatenate((np.asarray((0,), dtype=np.int64), np.cumsum(widths)))
    flat_faces = np.concatenate(faces).astype(np.int64, copy=False)
    face_ids = np.repeat(np.arange(len(faces), dtype=np.int64), widths)
    face_starts = np.repeat(starts[:-1], widths)
    positions = np.arange(len(flat_faces), dtype=np.int64) - face_starts
    next_ids = flat_faces[face_starts + (positions + 1) % widths[face_ids]]
    previous_ids = flat_faces[face_starts + (positions - 1) % widths[face_ids]]
    current = points[flat_faces]
    following = points[next_ids]
    previous = points[previous_ids]
    face_centres = np.add.reduceat(current, starts[:-1]) / widths[:, None]
    centre = face_centres[face_ids]
    face_normals = np.zeros((len(faces), 3), dtype=np.float64)
    np.add.at(
        face_normals,
        face_ids,
        0.5 * np.cross(following - current, centre - current),
    )
    normal_lengths = np.linalg.norm(face_normals, axis=1)
    invalid_faces = normal_lengths <= _VSMALL

    if face_patch_ids is None:
        patch_ids = np.zeros(len(faces), dtype=np.int64)
    else:
        patch_ids = np.asarray(face_patch_ids, dtype=np.int64)
        if patch_ids.shape != (len(faces),):
            raise ValueError("face_patch_ids does not match the boundary-face count")
    patch_count = int(patch_ids.max(initial=0)) + 1
    point_patch_normals = np.zeros((len(points), patch_count, 3), dtype=np.float64)
    np.add.at(
        point_patch_normals,
        (flat_faces, patch_ids[face_ids]),
        face_normals[face_ids],
    )
    point_normals = point_patch_normals[flat_faces, patch_ids[face_ids]]
    point_lengths = np.linalg.norm(point_normals, axis=1)
    invalid = invalid_faces[face_ids] | (point_lengths <= _VSMALL)
    safe_face_normals = face_normals[face_ids] / np.maximum(normal_lengths[face_ids, None], _VSMALL)
    safe_point_normals = point_normals / np.maximum(point_lengths[:, None], _VSMALL)
    next_normal = np.cross(following - current, centre - current)
    previous_normal = np.cross(centre - current, previous - current)
    next_length = np.linalg.norm(next_normal, axis=1)
    previous_length = np.linalg.norm(previous_normal, axis=1)
    invalid |= (next_length <= _VSMALL) | (previous_length <= _VSMALL)
    next_normal /= np.maximum(next_length[:, None], _VSMALL)
    previous_normal /= np.maximum(previous_length[:, None], _VSMALL)
    invalid |= (
        (np.einsum("ij,ij->i", next_normal, safe_point_normals) < 0.0)
        | (np.einsum("ij,ij->i", previous_normal, safe_point_normals) < 0.0)
        | (np.einsum("ij,ij->i", next_normal, previous_normal) < 0.0)
    )

    previous_edge = current - previous
    following_edge = following - current
    previous_edge /= np.maximum(np.linalg.norm(previous_edge, axis=1)[:, None], _VSMALL)
    following_edge /= np.maximum(np.linalg.norm(following_edge, axis=1)[:, None], _VSMALL)
    corner_normal = np.cross(previous_edge, following_edge)
    invalid |= np.einsum("ij,ij->i", corner_normal, safe_face_normals) < -0.05
    if active_points is not None:
        active = np.zeros(len(points), dtype=bool)
        active[np.asarray(tuple(active_points), dtype=np.int64)] = True
        invalid &= active[flat_faces]
    return set(map(int, np.unique(flat_faces[invalid])))


def inverted_cfmesh_boundary_points(
    mesh_data: dict[str, Any],
    face_patch_ids: Sequence[int] | np.ndarray,
    *,
    active_points: set[int] | None = None,
) -> set[int]:
    """Expose cfMesh's patch-aware inverted-point predicate for edge extraction."""
    return _inverted_boundary_points(mesh_data, face_patch_ids, active_points)


def smooth_cfmesh_partition_points(
    mesh_data: dict[str, Any],
    point_ids: Sequence[int],
    *,
    auxiliary_state: dict[str, np.ndarray] | None = None,
) -> None:
    """Run one cfMesh partition-point smoothing iteration for selected ids."""
    _smooth_partition_points(mesh_data, point_ids, iterations=1, auxiliary_state=auxiliary_state)


def untangle_cfmesh_surface(
    mesh_data: dict[str, Any],
    *,
    map_to_surface: Callable[[np.ndarray], np.ndarray],
    additional_layers: int = 2,
) -> dict[str, Any]:
    """Apply cfMesh's serial post-projection untangling loop in place."""
    points = np.asarray(mesh_data["vertex_position"], dtype=np.float64)
    faces = [np.asarray(face, dtype=np.int32) for face in mesh_data["faces"]]
    boundary_start = int(mesh_data["n_interior_faces"])
    point_neighbours: dict[int, set[int]] = defaultdict(set)
    for face in faces[boundary_start:]:
        for first_value, second_value in zip(face, np.roll(face, -1), strict=True):
            first = int(first_value)
            second = int(second_value)
            point_neighbours[first].add(second)
            point_neighbours[second].add(first)

    initial_inverted = 0
    iterations = 0
    moved_union: set[int] = set()
    for _iteration in range(20):
        inverted = _inverted_boundary_points(mesh_data)
        if _iteration == 0:
            initial_inverted = len(inverted)
        if not inverted:
            break
        selected = set(inverted)
        for _layer in range(additional_layers):
            selected.update(
                neighbour
                for point_id in tuple(selected)
                for neighbour in point_neighbours[point_id]
            )
        selected_points = tuple(sorted(selected))
        moved_union.update(selected_points)
        _smooth_partition_points(mesh_data, selected_points, iterations=1)
        mapped = np.asarray(
            [map_to_surface(points[point_id]) for point_id in selected_points],
            dtype=np.float64,
        )
        points[np.asarray(selected_points, dtype=np.int64)] = mapped
        iterations += 1

    return {
        "initial_inverted_points": initial_inverted,
        "iterations": iterations,
        "moved_points": len(moved_union),
        "remaining_inverted_points": len(_inverted_boundary_points(mesh_data)),
        "additional_layers": additional_layers,
    }


def optimise_cfmesh_surface(
    mesh_data: dict[str, Any],
    *,
    iterations: int = 5,
    map_edge_points: Callable[[Sequence[int]], None] | None = None,
    untangle_surface: Callable[[], list[int]] | None = None,
) -> None:
    """Apply cfMesh's feature-edge and boundary-surface smoothing in place."""
    points = np.asarray(mesh_data["vertex_position"], dtype=np.float64)
    faces = [np.asarray(face, dtype=np.int32) for face in mesh_data["faces"]]
    boundary_start = int(mesh_data["n_interior_faces"])
    point_faces: dict[int, list[int]] = defaultdict(list)
    point_patches: dict[int, set[int]] = defaultdict(set)
    edge_patches: dict[tuple[int, int], set[int]] = defaultdict(set)
    for patch_id, patch in enumerate(mesh_data["boundary"]):
        first = int(patch["start_face"])
        stop = first + int(patch["n_faces"])
        for face_id in range(first, stop):
            face = faces[face_id]
            for point_id_value in face:
                point_id = int(point_id_value)
                point_faces[point_id].append(face_id)
                point_patches[point_id].add(patch_id)
            for first_point, second_point in zip(face, np.roll(face, -1), strict=True):
                first_id = int(first_point)
                second_id = int(second_point)
                edge = (min(first_id, second_id), max(first_id, second_id))
                edge_patches[edge].add(patch_id)

    feature_neighbours: dict[int, list[int]] = defaultdict(list)
    for (first_point, second_point), patches in edge_patches.items():
        if len(patches) < 2:
            continue
        feature_neighbours[first_point].append(second_point)
        feature_neighbours[second_point].append(first_point)
    corner_points = {
        point_id for point_id, neighbours in feature_neighbours.items() if len(neighbours) > 2
    }
    edge_points = tuple(
        sorted(
            point_id for point_id, neighbours in feature_neighbours.items() if len(neighbours) == 2
        )
    )
    constrained_points = corner_points.union(edge_points)
    partition_points = tuple(
        sorted(point_id for point_id in point_faces if point_id not in constrained_points)
    )

    for _iteration in range(iterations):
        updates = {
            point_id: points[np.asarray(feature_neighbours[point_id], dtype=np.int64)].mean(axis=0)
            for point_id in edge_points
            if len(feature_neighbours[point_id]) == 2
        }
        for point_id, value in updates.items():
            points[point_id] = value
        if map_edge_points is not None:
            map_edge_points(edge_points)

    if iterations <= 0:
        # Large production meshes already have their boundary points mapped by
        # ``remap_cfmesh_patch_points``.  Keep feature-edge remapping, but do
        # not enter the scalar per-partition-point Newton loop; the subsequent
        # volume optimizer and the final all-face validation remain mandatory.
        mesh_data["mesh_generation"]["workflow_checkpoint"] = "edgeExtraction"
        mesh_data["mesh_generation"]["surface_optimisation"] = {
            "iterations": 0,
            "corner_points": len(corner_points),
            "edge_points": len(edge_points),
            "partition_points": len(partition_points),
            "untangling_iteration_counts": [],
        }
        return

    n_points = len(points)
    global_triangles: list[tuple[int, int, int]] = []
    point_triangle_ids: dict[int, list[int]] = defaultdict(list)
    for face_id in range(boundary_start, len(faces)):
        face = faces[face_id]
        auxiliary = n_points + face_id - boundary_start
        face_triangles: list[tuple[int, int, int]] = []
        if len(face) == 3:
            face_triangles.append((int(face[0]), int(face[1]), int(face[2])))
        for position, point_id_value in enumerate(face):
            point_id = int(point_id_value)
            following = int(face[(position + 1) % len(face)])
            previous = int(face[(position - 1) % len(face)])
            if len(face) > 3:
                face_triangles.append((point_id, following, auxiliary))
            face_triangles.append((point_id, following, previous))
        for triangle in face_triangles:
            triangle_id = len(global_triangles)
            global_triangles.append(triangle)
            for vertex in triangle:
                if vertex < n_points:
                    point_triangle_ids[vertex].append(triangle_id)

    auxiliary_centres: np.ndarray | None = None
    for _iteration in range(iterations):
        face_centres = {
            face_id: _face_centre(points[faces[face_id]])
            for face_id in range(boundary_start, len(faces))
        }
        face_normals = {
            face_id: _face_area_vector(points[faces[face_id]])
            for face_id in range(boundary_start, len(faces))
        }
        laplacian_updates: dict[int, np.ndarray] = {}
        for point_id in partition_points:
            normal = sum(
                (face_normals[face_id] for face_id in point_faces[point_id]),
                np.zeros(3, dtype=np.float64),
            )
            length = float(np.sqrt(_mag_squared(normal)))
            if length <= _VSMALL:
                continue
            normal /= length
            # plane(point, pointNormal) normalises the unit normal again.
            normal /= float(np.sqrt(_mag_squared(normal)))
            projected = [
                centre - normal * float(_dot(centre - points[point_id], normal))
                for face_id in point_faces[point_id]
                for centre in (face_centres[face_id],)
            ]
            laplacian_updates[point_id] = np.asarray(projected).mean(axis=0)
        for point_id, value in laplacian_updates.items():
            points[point_id] = value

        if auxiliary_centres is None:
            auxiliary_centres = np.asarray(
                [
                    _face_centre(points[faces[face_id]])
                    for face_id in range(boundary_start, len(faces))
                ],
                dtype=np.float64,
            )
        updated_auxiliary = auxiliary_centres.copy()
        partition_point_set = set(partition_points)
        for local_face_id, face_id in enumerate(range(boundary_start, len(faces))):
            face = faces[face_id]
            if not partition_point_set.intersection(map(int, face)):
                continue
            centre = auxiliary_centres[local_face_id]
            following = np.roll(face, -1)
            triangles = np.stack(
                (
                    points[face],
                    points[following],
                    np.broadcast_to(centre, (len(face), 3)),
                ),
                axis=1,
            )
            updated_auxiliary[local_face_id] = _relax_face_centre(triangles)
        auxiliary_centres = updated_auxiliary

        face_normals = {
            face_id: _face_area_vector(points[faces[face_id]])
            for face_id in range(boundary_start, len(faces))
        }
        point_normals: dict[int, np.ndarray] = {}
        for point_id in partition_points:
            normal = sum(
                (face_normals[face_id] for face_id in point_faces[point_id]),
                np.zeros(3, dtype=np.float64),
            )
            length = float(np.sqrt(_mag_squared(normal)))
            if length > _VSMALL:
                point_normals[point_id] = normal / length

        optimisation_updates: dict[int, np.ndarray] = {}
        for point_id in partition_points:
            normal = point_normals.get(point_id)
            if normal is None:
                continue
            normal = normal / float(np.sqrt(_mag_squared(normal)))
            local_labels: list[int] = []
            local_index: dict[int, int] = {}
            local_triangles: list[tuple[int, int, int]] = []
            for triangle_id in point_triangle_ids[point_id]:
                triangle = global_triangles[triangle_id]
                for vertex in triangle:
                    if vertex not in local_index:
                        local_index[vertex] = len(local_labels)
                        local_labels.append(vertex)
                position = triangle.index(point_id)
                if triangle[2] < n_points and position != 0:
                    continue
                rotated = triangle[position:] + triangle[:position]
                local_triangles.append(
                    (
                        local_index[rotated[0]],
                        local_index[rotated[1]],
                        local_index[rotated[2]],
                    )
                )
            if not local_triangles:
                continue
            centre_point = points[point_id]
            local_coordinates = np.asarray(
                [
                    points[label] if label < n_points else auxiliary_centres[label - n_points]
                    for label in local_labels
                ]
            )
            vector_x: np.ndarray | None = None
            for coordinate in local_coordinates:
                projected = coordinate - normal * float(_dot(coordinate - centre_point, normal))
                offset = projected - centre_point
                length = float(np.sqrt(_mag_squared(offset)))
                if length > _VSMALL:
                    vector_x = offset / length
                    break
            if vector_x is None:
                continue
            vector_y = np.cross(normal, vector_x)
            vector_y /= float(np.sqrt(_mag_squared(vector_y)))
            offsets = local_coordinates - centre_point
            planar = np.column_stack(
                (
                    _dot(offsets, vector_x),
                    _dot(offsets, vector_y),
                    np.zeros(len(offsets), dtype=np.float64),
                )
            )
            new_planar = _optimise_point(planar, np.asarray(local_triangles, dtype=np.int32))
            optimisation_updates[point_id] = (
                centre_point + vector_x * new_planar[0] + vector_y * new_planar[1]
            )
        for point_id, value in optimisation_updates.items():
            points[point_id] = value

    untangling_history = untangle_surface() if untangle_surface is not None else []

    mesh_data["mesh_generation"]["workflow_checkpoint"] = "edgeExtraction"
    mesh_data["mesh_generation"]["surface_optimisation"] = {
        "iterations": iterations,
        "corner_points": len(corner_points),
        "edge_points": len(edge_points),
        "partition_points": len(partition_points),
        "untangling_iteration_counts": untangling_history,
    }


__all__ = [
    "inverted_cfmesh_boundary_points",
    "optimise_cfmesh_surface",
    "smooth_cfmesh_partition_points",
    "untangle_cfmesh_surface",
]
