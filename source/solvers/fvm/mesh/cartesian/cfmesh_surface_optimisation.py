# SPDX-License-Identifier: GPL-3.0-or-later
"""Surface optimisation used by cfMesh's ``edgeExtraction`` checkpoint."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

from numba import prange
from numba.extending import register_jitable
import numpy as np

from source._numba import cacheable_njit as njit

_SMALL = 1.0e-15
_VSMALL = 1.0e-300
_ROOT_VSMALL = 1.0e-150


@register_jitable
def _mag_squared(value: np.ndarray) -> float:
    """Use OpenFOAM's component order, without a BLAS reduction."""
    return float(value[0] * value[0] + value[1] * value[1] + value[2] * value[2])


@register_jitable
def _dot(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Scalar-order dot products, also accepting an array of row vectors."""
    return first[..., 0] * second[0] + first[..., 1] * second[1] + first[..., 2] * second[2]


@njit(cache=True, fastmath=False)
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


@njit(cache=True, fastmath=False)
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


@njit(cache=True, fastmath=False)
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


@register_jitable
def _distance_squared(first, second):
    x = first[0] - second[0]
    y = first[1] - second[1]
    z = first[2] - second[2]
    return x * x + y * y + z * z


@register_jitable
def _stabilisation(points: np.ndarray, triangles: np.ndarray) -> float:
    # Keep cfMesh's scalar, triangle-by-triangle reduction order.  Symmetric
    # simplexes can have equal minima to machine precision, so NumPy's pairwise
    # reductions can select the reflected optimisation branch.
    minimum_area = 1.0e300
    maximum_length_squared = 0.0
    for triangle in triangles:
        p0 = points[triangle[0]]
        p1 = points[triangle[1]]
        p2 = points[triangle[2]]
        area = 0.5 * ((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]))
        length_squared = _distance_squared(p0, p1) + _distance_squared(p2, p0)
        minimum_area = min(minimum_area, float(area))
        maximum_length_squared = max(maximum_length_squared, length_squared)
    if minimum_area < _SMALL * maximum_length_squared:
        return _SMALL * maximum_length_squared
    return 0.0


@register_jitable
def _objective(points: np.ndarray, triangles: np.ndarray, stabilisation: float) -> float:
    value = 0.0
    for triangle in triangles:
        p0 = points[triangle[0]]
        p1 = points[triangle[1]]
        p2 = points[triangle[2]]
        area = 0.5 * ((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]))
        stable = float(np.sqrt(area * area + stabilisation))
        denominator = max(_VSMALL, 0.5 * (area + stable))
        length_squared = _distance_squared(p0, p1) + _distance_squared(p2, p0)
        value += length_squared / denominator
    return value


@register_jitable
def _gradients(
    points: np.ndarray, triangles: np.ndarray, stabilisation: float
) -> tuple[np.ndarray, np.ndarray]:
    gradient = np.zeros(2, dtype=np.float64)
    hessian = np.zeros((2, 2), dtype=np.float64)
    for triangle in triangles:
        p0 = points[triangle[0]]
        p1 = points[triangle[1]]
        p2 = points[triangle[2]]
        if _distance_squared(p1, p2) < _VSMALL:
            continue
        length_squared = _distance_squared(p0, p1) + _distance_squared(p2, p0)
        area = 0.5 * ((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]))
        stable = float(np.sqrt(area * area + stabilisation))
        stable_area = max(_ROOT_VSMALL, 0.5 * (area + stable))
        area_gradient = (0.5 * (p1[1] - p2[1]), 0.5 * (p2[0] - p1[0]))
        stable_gradient = (
            0.5 * (area_gradient[0] + area * area_gradient[0] / stable),
            0.5 * (area_gradient[1] + area * area_gradient[1] / stable),
        )
        length_gradient = (
            4.0 * p0[0] - 2.0 * p1[0] - 2.0 * p2[0],
            4.0 * p0[1] - 2.0 * p1[1] - 2.0 * p2[1],
        )
        stable_area_squared = stable_area * stable_area
        # Retain libm pow and each expression's operation order. Thousands of
        # tiny outer-product temporaries per point otherwise dominate Newton.
        stable_cubed = stable**3.0
        for row in range(2):
            gradient[row] += (
                length_gradient[row] / stable_area
                - length_squared * stable_gradient[row] / stable_area_squared
            )
            for column in range(2):
                area_outer = area_gradient[row] * area_gradient[column]
                stable_hessian = 0.5 * (
                    area_outer / stable - area * area * area_outer / stable_cubed
                )
                hessian[row, column] += (
                    (4.0 if row == column else 0.0) / stable_area
                    - (
                        length_gradient[row] * stable_gradient[column]
                        + stable_gradient[row] * length_gradient[column]
                    )
                    / stable_area_squared
                    - stable_hessian * length_squared / stable_area_squared
                    + 2.0
                    * length_squared
                    * (stable_gradient[row] * stable_gradient[column])
                    / (stable_area_squared * stable_area)
                )
    if abs(float(hessian[0, 0])) < _VSMALL:
        hessian[0, 0] = _VSMALL
    if abs(float(hessian[1, 1])) < _VSMALL:
        hessian[1, 1] = _VSMALL
    return gradient, hessian


@register_jitable
def _optimise_point(
    points: np.ndarray,
    triangles: np.ndarray,
    tolerance: float = 0.001,
) -> np.ndarray:
    target = int(triangles[0, 0])
    neighbour_values = points[triangles[:, 1:].ravel()]
    # Axis-wise reductions are explicit because Numba does not support the
    # axis argument of ndarray.min/max.  This does not change the extrema.
    lower = np.empty(3, dtype=np.float64)
    upper = np.empty(3, dtype=np.float64)
    for axis in range(3):
        lower[axis] = neighbour_values[:, axis].min()
        upper[axis] = neighbour_values[:, axis].max()
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


@njit(cache=True, fastmath=False)
def _optimise_point_kernel(
    points: np.ndarray, triangles: np.ndarray, tolerance: float = 0.001
) -> np.ndarray:
    """Compile the reference arithmetic, including its branch-sensitive order.

    Keeping one implementation avoids changes to normalization, stabilization,
    and symmetric-simplex tie breaking in a separately transcribed fast path.
    """
    return _optimise_point(points, triangles, tolerance=tolerance)


def _pack_rows(rows):
    """Pack small topology rows once, preserving their stored order."""
    offsets = np.empty(len(rows) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum([len(row) for row in rows], out=offsets[1:])
    values = np.asarray([value for row in rows for value in row], dtype=np.int32)
    return values, offsets


@njit(cache=True, fastmath=False)
def _surface_face_geometry(points, vertices, offsets):
    centres = np.empty((len(offsets) - 1, 3), dtype=np.float64)
    normals = np.empty_like(centres)
    for face in range(len(centres)):
        coordinates = points[vertices[offsets[face] : offsets[face + 1]]]
        centres[face] = _face_centre(coordinates)
        normals[face] = _face_area_vector(coordinates)
    return centres, normals


@njit(cache=True, fastmath=False)
def _surface_laplacian(points, selected, face_ids, offsets, centres, normals):
    updates = points[selected].copy()
    for local in range(len(selected)):
        point = points[selected[local]]
        normal = np.zeros(3, dtype=np.float64)
        first, stop = offsets[local], offsets[local + 1]
        for entry in range(first, stop):
            normal += normals[face_ids[entry]]
        length = np.sqrt(_mag_squared(normal))
        if length <= _VSMALL:
            continue
        normal /= length
        normal /= np.sqrt(_mag_squared(normal))
        value = np.zeros(3, dtype=np.float64)
        for entry in range(first, stop):
            centre = centres[face_ids[entry]]
            value += centre - normal * _dot(centre - point, normal)
        updates[local] = value / (stop - first)
    points[selected] = updates


@njit(cache=True, fastmath=False)
def _surface_auxiliary(points, vertices, offsets, active, centres):
    updated = centres.copy()
    for face in range(len(centres)):
        first, stop = offsets[face], offsets[face + 1]
        if not np.any(active[vertices[first:stop]]):
            continue
        triangles = np.empty((stop - first, 3, 3), dtype=np.float64)
        for local in range(stop - first):
            triangles[local, 0] = points[vertices[first + local]]
            triangles[local, 1] = points[vertices[first + (local + 1) % (stop - first)]]
            triangles[local, 2] = centres[face]
        updated[face] = _relax_face_centre(triangles)
    return updated


def _surface_optimisation_updates(
    points,
    triangulation_points,
    auxiliary,
    selected,
    face_ids,
    face_offsets,
    face_normals,
    labels,
    label_offsets,
    triangles,
    triangle_offsets,
):
    """Independent Jacobi updates; Newton arithmetic stays in stored order."""
    updates = points[selected].copy()
    for local in prange(len(selected)):
        normal = np.zeros(3, dtype=np.float64)
        for entry in range(face_offsets[local], face_offsets[local + 1]):
            normal += face_normals[face_ids[entry]]
        length = np.sqrt(_mag_squared(normal))
        if length <= _VSMALL:
            continue
        normal /= length
        normal /= np.sqrt(_mag_squared(normal))
        local_triangles = triangles[triangle_offsets[local] : triangle_offsets[local + 1]]
        if len(local_triangles) == 0:
            continue
        local_labels = labels[label_offsets[local] : label_offsets[local + 1]]
        coordinates = np.empty((len(local_labels), 3), dtype=np.float64)
        for i in range(len(local_labels)):
            label = local_labels[i]
            coordinates[i] = (
                triangulation_points[label]
                if label < len(points)
                else auxiliary[label - len(points)]
            )
        point = points[selected[local]]
        vector_x = np.zeros(3, dtype=np.float64)
        found = False
        for coordinate in coordinates:
            projected = coordinate - normal * _dot(coordinate - point, normal)
            offset = projected - point
            length = np.sqrt(_mag_squared(offset))
            if length > _VSMALL:
                vector_x = offset / length
                found = True
                break
        if not found:
            continue
        vector_y = np.cross(normal, vector_x)
        vector_y /= np.sqrt(_mag_squared(vector_y))
        planar = np.zeros((len(coordinates), 3), dtype=np.float64)
        for i in range(len(coordinates)):
            offset = coordinates[i] - point
            planar[i, 0] = _dot(offset, vector_x)
            planar[i, 1] = _dot(offset, vector_y)
        new_planar = _optimise_point_kernel(planar, local_triangles)
        updates[local] = point + vector_x * new_planar[0] + vector_y * new_planar[1]
    return updates


_surface_optimisation_updates_serial = njit(cache=True, fastmath=False)(
    _surface_optimisation_updates
)
_surface_optimisation_updates = njit(cache=True, fastmath=False, parallel=True)(
    _surface_optimisation_updates
)


def _smooth_partition_points(
    mesh_data: dict[str, Any],
    partition_points: Sequence[int],
    *,
    iterations: int,
    auxiliary_state: dict[str, np.ndarray] | None = None,
) -> None:
    """Run all surface passes using reusable stencils and compiled arithmetic."""
    if iterations <= 0:
        return
    points = np.asarray(mesh_data["vertex_position"], dtype=np.float64)
    boundary_start = int(mesh_data["n_interior_faces"])
    faces = mesh_data["faces"][boundary_start:]
    selected = np.asarray(partition_points, dtype=np.int32)
    vertices, face_offsets = _pack_rows(faces)
    point_faces: dict[int, list[int]] = defaultdict(list)
    point_triangles: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    n_points = len(points)
    for face_id, face in enumerate(faces):
        for point in face:
            point_faces[int(point)].append(face_id)
        auxiliary = n_points + face_id
        face_triangles = []
        if len(face) == 3:
            face_triangles.append(tuple(map(int, face)))
        for position, point in enumerate(face):
            current = int(point)
            following = int(face[(position + 1) % len(face)])
            previous = int(face[(position - 1) % len(face)])
            if len(face) > 3:
                face_triangles.append((current, following, auxiliary))
            face_triangles.append((current, following, previous))
        for triangle in face_triangles:
            for vertex in triangle:
                if vertex < n_points:
                    point_triangles[vertex].append(triangle)

    label_rows = []
    triangle_rows = []
    for point in selected:
        local_index = {}
        local_labels = []
        local_triangles = []
        for triangle in point_triangles[int(point)]:
            for vertex in triangle:
                if vertex not in local_index:
                    local_index[vertex] = len(local_labels)
                    local_labels.append(vertex)
            position = triangle.index(point)
            if triangle[2] < n_points and position != 0:
                continue
            rotated = triangle[position:] + triangle[:position]
            local_triangles.append(tuple(local_index[v] for v in rotated))
        label_rows.append(local_labels)
        triangle_rows.append(local_triangles)
    labels, label_offsets = _pack_rows(label_rows)
    triangles, triangle_offsets = _pack_rows(triangle_rows)
    triangles = triangles.reshape((-1, 3))
    face_ids, point_face_offsets = _pack_rows([point_faces[int(p)] for p in selected])
    active = np.zeros(n_points, dtype=np.bool_)
    active[selected] = True
    optimise = (
        _surface_optimisation_updates
        if len(selected) >= 1000
        else _surface_optimisation_updates_serial
    )
    auxiliary_centres = auxiliary_state.get("centres") if auxiliary_state is not None else None
    triangulation_points = auxiliary_state.get("points") if auxiliary_state is not None else None
    for _iteration in range(iterations):
        centres, normals = _surface_face_geometry(points, vertices, face_offsets)
        _surface_laplacian(points, selected, face_ids, point_face_offsets, centres, normals)
        if triangulation_points is None:
            triangulation_points = points.copy()
        else:
            triangulation_points[selected] = points[selected]
        centres, normals = _surface_face_geometry(points, vertices, face_offsets)
        if auxiliary_centres is None:
            auxiliary_centres = centres
        auxiliary_centres = _surface_auxiliary(
            triangulation_points, vertices, face_offsets, active, auxiliary_centres
        )
        points[selected] = optimise(
            points,
            triangulation_points,
            auxiliary_centres,
            selected,
            face_ids,
            point_face_offsets,
            normals,
            labels,
            label_offsets,
            triangles,
            triangle_offsets,
        )
    if auxiliary_state is not None:
        auxiliary_state["points"] = triangulation_points
        auxiliary_state["centres"] = auxiliary_centres


def _inverted_boundary_points(
    mesh_data: dict[str, Any],
    face_patch_ids: Sequence[int] | np.ndarray | None = None,
    active_points: set[int] | None = None,
) -> set[int]:
    """Return the serial partition-point subset rejected by cfMesh's check."""
    points = np.asarray(mesh_data["vertex_position"], dtype=np.float64)
    faces = [np.asarray(face, dtype=np.int32) for face in mesh_data["faces"]]
    boundary_start = int(mesh_data["n_interior_faces"])
    if len(faces) - boundary_start >= 128:
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

    Repeated Python face/corner scans become costly with even a few hundred
    boundary polygons.  Keep the scalar cfMesh transcription for small parity
    cases and evaluate larger boundary sets in NumPy with the same predicates.
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
    arithmetic_centres = np.add.reduceat(current, starts[:-1]) / widths[:, None]
    arithmetic_centre = arithmetic_centres[face_ids]
    # cfMesh's face centre is area weighted for polygons, while its area
    # vector uses the arithmetic centre.  The two coincide only for planar,
    # symmetric faces; wrapper smoothing produces non-planar quads.
    twice_areas = np.linalg.norm(
        np.cross(current - arithmetic_centre, following - arithmetic_centre), axis=1
    )
    area_sums = np.add.reduceat(twice_areas, starts[:-1])
    weighted_centres = np.add.reduceat(
        twice_areas[:, None] * (current + following + arithmetic_centre), starts[:-1]
    )
    face_centres = arithmetic_centres.copy()
    weighted_faces = (widths > 3) & (area_sums > _VSMALL)
    face_centres[weighted_faces] = weighted_centres[weighted_faces] / (
        3.0 * area_sums[weighted_faces, None]
    )
    centre = face_centres[face_ids]
    face_normals = np.zeros((len(faces), 3), dtype=np.float64)
    np.add.at(
        face_normals,
        face_ids,
        0.5 * np.cross(following - current, arithmetic_centre - current),
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

    _smooth_partition_points(mesh_data, partition_points, iterations=iterations)

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
