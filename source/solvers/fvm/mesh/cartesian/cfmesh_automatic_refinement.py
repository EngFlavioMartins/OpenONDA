# SPDX-License-Identifier: GPL-3.0-or-later
"""cfMesh curvature and surface-proximity octree refinement.

The criteria and ordering follow meshOctreeAutomaticRefinement, with the
quadric fit from triSurfaceCurvatureEstimator. This module has no knowledge
of geometric primitive names: all decisions use the supplied triangles.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from itertools import combinations

import numpy as np

from source._numba import cacheable_njit as njit

from ..surface_classification import SurfaceIndex, triangle_box_overlap
from .cfmesh_octree import Leaf, LeafLookup, refine_selected_leaves


@njit(cache=True)
def _foam_lu_solve(matrix: np.ndarray, source: np.ndarray) -> np.ndarray:
    """OpenFOAM scaled partial pivoting, including its exact-zero pivot rule."""
    size = len(source)
    values = matrix.copy()
    solution = source.copy()
    scales = np.empty(size)
    pivots = np.empty(size, dtype=np.int64)
    for i in range(size):
        largest = 0.0
        for j in range(size):
            largest = max(largest, abs(values[i, j]))
        if largest == 0.0:
            raise ValueError("Singular cfMesh surface-curvature fit: zero matrix row")
        scales[i] = 1.0 / largest
    for j in range(size):
        for i in range(j):
            total = values[i, j]
            for k in range(i):
                total -= values[i, k] * values[k, j]
            values[i, j] = total
        best, largest = 0, 0.0
        for i in range(j, size):
            total = values[i, j]
            for k in range(j):
                total -= values[i, k] * values[k, j]
            values[i, j] = total
            score = scales[i] * abs(total)
            if score >= largest:
                best, largest = i, score
        pivots[j] = best
        if j != best:
            for k in range(size):
                values[j, k], values[best, k] = values[best, k], values[j, k]
            scales[best] = scales[j]
        if values[j, j] == 0.0:
            values[j, j] = 1.0e-15
        if j != size - 1:
            reciprocal = 1.0 / values[j, j]
            for i in range(j + 1, size):
                values[i, j] *= reciprocal
    first_nonzero = 0
    for i in range(size):
        pivot = pivots[i]
        total = solution[pivot]
        solution[pivot] = solution[i]
        if first_nonzero != 0:
            for j in range(first_nonzero - 1, i):
                total -= values[i, j] * solution[j]
        elif total != 0.0:
            first_nonzero = i + 1
        solution[i] = total
    for i in range(size - 1, -1, -1):
        total = solution[i]
        for j in range(i + 1, size):
            total -= values[i, j] * solution[j]
        solution[i] = total / values[i, i]
    return solution


@njit(cache=True)
def _quadric_coefficients(design: np.ndarray, z: np.ndarray) -> np.ndarray:
    matrix = np.zeros((5, 5))
    source = np.zeros(5)
    for i in range(len(z)):
        for row in range(5):
            for column in range(row, 5):
                matrix[row, column] += design[i, row] * design[i, column]
            source[row] += design[i, row] * z[i]
    for row in range(1, 5):
        for column in range(row):
            matrix[row, column] = matrix[column, row]
        if abs(matrix[row, row]) < 1.0e-15:
            matrix[row, row] = 1.0e-15
    return _foam_lu_solve(matrix, source)


def _edge_distance_squared(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    *,
    segment: bool = True,
) -> np.ndarray:
    direction = end - start
    denominator = np.sum(direction * direction, axis=-1)
    fraction = np.sum(direction * (point - start), axis=-1) / (denominator + 1.0e-300)
    if segment:
        fraction = np.clip(fraction, 0.0, 1.0)
    offset = start + fraction[..., None] * direction - point
    return np.sum(offset * offset, axis=-1)


def _quadric_curvature(origin: np.ndarray, normal: np.ndarray, points: np.ndarray) -> float:
    """Mean of the fitted principal curvatures, with ten native normal updates."""
    magnitude = float(np.linalg.norm(normal))
    if magnitude < 1.0e-300:
        raise ValueError("Cannot estimate cfMesh curvature at a zero-normal surface point")
    normal = normal / magnitude
    coefficients = np.zeros(5)
    offsets = points - origin
    for _iteration in range(10):
        plane_normal = normal / np.linalg.norm(normal)
        tangent = offsets - np.sum(offsets * plane_normal, axis=1)[:, None] * plane_normal
        candidates = np.flatnonzero(np.sum(tangent * tangent, axis=1) >= 1.0e-300)
        if not len(candidates):
            return 0.0
        x_axis = tangent[int(candidates[0])]
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(normal, x_axis)
        x, y, z = offsets @ x_axis, offsets @ y_axis, offsets @ normal
        design = np.column_stack((x * x, y * y, x * y, x, y))
        coefficients = _quadric_coefficients(design, z)
        a, b = coefficients[3:]
        if abs(a) <= 1.0e-15 and abs(b) <= 1.0e-15:
            break
        divisor = np.sqrt(1.0 + a * a + b * b)
        normal = normal / divisor - a * x_axis / divisor - b * y_axis / divisor
    return float(coefficients[0] + coefficients[1])


class AutomaticSurface:
    """Indexed triangulated surface used by automatic octree refinement.

    Parameters
    ----------
    groups : mapping[str, ndarray]
        Named surface groups.  Each value has shape ``(T, 3, 3)`` and contains
        triangle vertices in metres.  Groups are assigned deterministic integer
        region ids in sorted-name order.
    nearest_triangles : callable
        Callback ``nearest_triangles(point, triangles)`` returning the closest
        point on each candidate triangle; used by proximity tests.

    Attributes
    ----------
    points, triangles, coordinates : ndarray
        Deduplicated vertices ``(P, 3)``, triangle connectivity ``(T, 3)``,
        and materialized triangle coordinates ``(T, 3, 3)``.
    regions : ndarray, shape (T,)
        Integer group id for each triangle.
    feature_ids, corners : ndarray
        Surface feature-edge ids and vertices where at least three feature
        edges meet.

    Notes
    -----
    Construction computes connectivity and smoothed curvature eagerly.  The
    object is read-only during refinement; its query methods return ids into
    the stored arrays rather than new geometry.
    """

    def __init__(
        self,
        groups: Mapping[str, np.ndarray],
        nearest_triangles: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> None:
        """Build connectivity, feature groups, and curvature metadata."""
        self.nearest_triangles = nearest_triangles
        vertices: list[np.ndarray] = []
        ids: dict[tuple[float, ...], int] = {}
        triangles: list[list[int]] = []
        regions: list[int] = []
        for region, name in enumerate(sorted(groups)):
            for triangle in groups[name]:
                row = []
                for point in triangle:
                    key = tuple(map(float, point))
                    if key not in ids:
                        ids[key] = len(vertices)
                        vertices.append(point)
                    row.append(ids[key])
                triangles.append(row)
                regions.append(region)
        self.points = np.asarray(vertices)
        self.triangles = np.asarray(triangles, dtype=np.int64)
        self.regions = np.asarray(regions, dtype=np.int64)
        self.coordinates = self.points[self.triangles]
        self.index = SurfaceIndex.build(self.coordinates)
        self.point_faces: list[list[int]] = [[] for _ in vertices]
        self.point_edges: list[list[int]] = [[] for _ in vertices]
        edge_ids: dict[tuple[int, int], int] = {}
        edges: list[tuple[int, int]] = []
        self.edge_faces: list[list[int]] = []
        self.face_edges: list[list[int]] = []
        for face_id, triangle in enumerate(self.triangles):
            face_edges = []
            for i, first_value in enumerate(triangle):
                first, second = int(first_value), int(triangle[(i + 1) % 3])
                self.point_faces[first].append(face_id)
                key = (min(first, second), max(first, second))
                if key not in edge_ids:
                    edge_ids[key] = len(edges)
                    edges.append((first, second))
                    self.edge_faces.append([])
                    self.point_edges[first].append(edge_ids[key])
                    self.point_edges[second].append(edge_ids[key])
                edge_id = edge_ids[key]
                self.edge_faces[edge_id].append(face_id)
                face_edges.append(edge_id)
            self.face_edges.append(face_edges)
        self.edges = np.asarray(edges, dtype=np.int64)
        self.feature = np.asarray(
            [
                len(faces) == 2 and self.regions[faces[0]] != self.regions[faces[1]]
                for faces in self.edge_faces
            ]
        )
        self.feature_ids = np.flatnonzero(self.feature)
        self.corners = np.asarray(
            [
                point_id
                for point_id, row in enumerate(self.point_edges)
                if np.count_nonzero(self.feature[row]) >= 3
            ],
            dtype=np.int64,
        )
        self.patch_neighbours: dict[int, set[int]] = defaultdict(set)
        for edge_id in self.feature_ids:
            first, second = (int(self.regions[face]) for face in self.edge_faces[edge_id])
            self.patch_neighbours[first].add(second)
            self.patch_neighbours[second].add(first)
        self.edge_groups = np.full(len(edges), -1, dtype=np.int64)
        corner_set = set(map(int, self.corners))
        group = 0
        for edge_id_value in self.feature_ids:
            edge_id = int(edge_id_value)
            if self.edge_groups[edge_id] >= 0:
                continue
            self.edge_groups[edge_id] = group
            front = [edge_id]
            while front:
                current = front.pop()
                for point_id in self.edges[current]:
                    if int(point_id) in corner_set:
                        continue
                    for neighbour in self.point_edges[point_id]:
                        if self.feature[neighbour] and self.edge_groups[neighbour] < 0:
                            self.edge_groups[neighbour] = group
                            front.append(neighbour)
            group += 1
        self.group_neighbours: dict[int, set[int]] = defaultdict(set)
        for point_id in self.corners:
            groups_at_point = set(map(int, self.edge_groups[self.point_edges[point_id]])) - {-1}
            for first, second in combinations(groups_at_point, 2):
                self.group_neighbours[first].add(second)
                self.group_neighbours[second].add(first)
        self.triangle_curvature = self._surface_curvature()
        self.edge_curvature = self._feature_curvature()

    def _surface_curvature(self) -> np.ndarray:
        area = 0.5 * np.cross(
            self.coordinates[:, 1] - self.coordinates[:, 0],
            self.coordinates[:, 2] - self.coordinates[:, 0],
        )
        curvature: dict[tuple[int, int], float] = {}
        neighbours: dict[tuple[int, int], list[int]] = {}
        for point_id, faces in enumerate(self.point_faces):
            for region in sorted(set(map(int, self.regions[faces]))):
                region_faces = [face for face in faces if self.regions[face] == region]
                adjacent = list(
                    dict.fromkeys(
                        int(vertex)
                        for face in region_faces
                        for vertex in self.triangles[face]
                        if vertex != point_id
                    )
                )
                neighbours[point_id, region] = adjacent
                fit_points = set(adjacent)
                if len(fit_points) <= 5:
                    fit_points.update(
                        int(vertex)
                        for point in adjacent
                        for face in self.point_faces[point]
                        if self.regions[face] == region
                        for vertex in self.triangles[face]
                    )
                normal = np.zeros(3)
                for face in region_faces:
                    normal += area[face]
                    normal += area[face]
                curvature[point_id, region] = _quadric_curvature(
                    self.points[point_id], normal, self.points[sorted(fit_points)]
                )
        # Smoothing each principal curvature and then averaging is the same
        # linear operation as smoothing their mean, with identical neighbours.
        for _iteration in range(2):
            curvature = {
                key: 0.5
                * (
                    value
                    + sum(curvature[point, key[1]] for point in neighbours[key])
                    / len(neighbours[key])
                )
                for key, value in curvature.items()
            }
        return np.asarray(
            [
                sum(curvature[int(point), int(region)] for point in triangle) / 3.0
                for triangle, region in zip(self.triangles, self.regions, strict=True)
            ]
        )

    def _feature_curvature(self) -> np.ndarray:
        values = np.zeros(len(self.points))
        for point_id, row in enumerate(self.point_edges):
            selected = [edge for edge in row if self.feature[edge]]
            if len(selected) != 2:
                continue
            first, second = self.points[self.edges[selected]]
            a, b = first[1] - first[0], second[1] - second[0]
            da, db = np.linalg.norm(a) + 1.0e-300, np.linalg.norm(b) + 1.0e-300
            cosine = np.clip(np.dot(a / da, b / db), -1.0, 1.0)
            values[point_id] = abs(np.arccos(cosine) / (0.5 * (da + db + 1.0e-300)))
        return np.mean(values[self.edges], axis=1)

    def triangles_in_box(self, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        """Return surface-triangle ids intersecting an axis-aligned box.

        Parameters
        ----------
        lower, upper : ndarray, shape (3,)
            Box bounds in metres.

        Returns
        -------
        ndarray, shape (K,)
            Integer ids of triangles that geometrically overlap the box.
        """
        ids = self.index.candidate_triangles(lower, upper)
        if not len(ids):
            return ids
        coordinates = self.coordinates[ids]
        return ids[
            triangle_box_overlap(
                0.5 * (lower + upper),
                0.5 * (upper - lower),
                coordinates[:, 0],
                coordinates[:, 1],
                coordinates[:, 2],
            )
        ]

    def edges_in_box(self, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        """Return feature-edge ids intersecting an axis-aligned box.

        Parameters
        ----------
        lower, upper : ndarray, shape (3,)
            Box bounds in metres.

        Returns
        -------
        ndarray, shape (K,)
            Integer ids from :attr:`feature_ids` whose line segments intersect
            the box.
        """
        coordinates = self.points[self.edges[self.feature_ids]]
        start, end = coordinates[:, 0], coordinates[:, 1]
        direction = end - start
        moving = direction != 0.0
        divisor = np.where(moving, direction, 1.0)
        first, last = (lower - start) / divisor, (upper - start) / divisor
        low, high = np.minimum(first, last), np.maximum(first, last)
        low = np.where(moving, low, -np.inf)
        high = np.where(moving, high, np.inf)
        valid = np.all(moving | ((start >= lower) & (start <= upper)), axis=1)
        valid &= np.maximum(0.0, low.max(axis=1)) <= np.minimum(1.0, high.min(axis=1))
        return self.feature_ids[valid]

    def in_range(self, centre: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
        """Find nearby triangles and feature edges around a refinement cell.

        Parameters
        ----------
        centre : ndarray, shape (3,)
            Cell centre in metres.
        radius : float
            Search radius in metres.

        Returns
        -------
        triangles, edges : ndarray
            Integer ids of nearby triangles and feature edges.
        """
        ids = self.index.candidate_triangles(centre - radius, centre + radius)
        nearest = self.nearest_triangles(centre, self.coordinates[ids])
        triangles = ids[np.sum((nearest - centre) ** 2, axis=1) < radius * radius]
        edges = self.edges[self.feature_ids]
        inside = (
            _edge_distance_squared(centre, self.points[edges[:, 0]], self.points[edges[:, 1]])
            < radius * radius
        )
        return triangles, self.feature_ids[inside]

    def partitions_need_refinement(self, centre: np.ndarray, radius: float) -> bool:
        """Return whether nearby patches/features are topologically separated."""
        triangles, edges = self.in_range(centre, radius)
        regions = set(map(int, self.regions[triangles]))
        groups = set(map(int, self.edge_groups[edges]))
        return any(
            second not in self.patch_neighbours[first] for first, second in combinations(regions, 2)
        ) or any(
            second not in self.group_neighbours[first] for first, second in combinations(groups, 2)
        )

    def proximity_needs_refinement(self, centre: np.ndarray, radius: float) -> bool:
        """Return whether a refinement ball contains disconnected surface pieces."""
        triangles, edges = self.in_range(centre, radius)
        remaining = set(map(int, triangles))
        groups = 0
        while remaining:
            groups += 1
            if groups > 1:
                return True
            front = [remaining.pop()]
            while front:
                current = front.pop()
                for edge_id in self.face_edges[current]:
                    start, end = self.points[self.edges[edge_id]]
                    # Native face-group connectivity tests an infinite line.
                    if _edge_distance_squared(centre, start, end, segment=False) >= radius**2:
                        continue
                    for neighbour in self.edge_faces[edge_id]:
                        if neighbour in remaining:
                            remaining.remove(neighbour)
                            front.append(neighbour)
        remaining = set(map(int, edges))
        groups = 0
        while remaining:
            groups += 1
            if groups > 1:
                return True
            front = [remaining.pop()]
            while front:
                current = front.pop()
                for point_id in self.edges[current]:
                    if np.sum((self.points[point_id] - centre) ** 2) >= radius**2:
                        continue
                    for neighbour in self.point_edges[point_id]:
                        if neighbour in remaining:
                            remaining.remove(neighbour)
                            front.append(neighbour)
        return False


def automatic_refinement(
    leaves: list[Leaf],
    surface: AutomaticSurface,
    *,
    root_lower: np.ndarray,
    root_size: float,
    max_level: int,
    automatic_level: int,
    classify: Callable[[int, int, int, int, int], Leaf],
) -> tuple[list[Leaf], dict[str, list[int]]]:
    """Apply curvature, corners, partitions, proximity, each with one extra layer."""
    finest = root_size / 2**max_level
    tolerance = 1.0e-15 * root_size
    diagnostics: dict[str, list[int]] = {}
    for criterion in ("curvature", "corners", "partitions", "proximity"):
        history: list[int] = []
        diagnostics[criterion] = history
        while True:
            selected: set[int] = set()
            lookup = LeafLookup(leaves, max_level)
            corner_in_leaf: dict[int, int] = {}
            if criterion == "corners":
                for point_id_value in surface.corners:
                    point_id = int(point_id_value)
                    coordinates = np.floor((surface.points[point_id] - root_lower) / finest).astype(
                        int
                    )
                    leaf_id = lookup.find(*map(int, coordinates))
                    if leaf_id < 0:
                        continue
                    if leaf_id in corner_in_leaf:
                        selected.add(leaf_id)
                    else:
                        corner_in_leaf[leaf_id] = point_id
            for leaf_id, leaf in enumerate(leaves):
                x, y, z, width, level, kind = leaf
                if level >= automatic_level or kind != 2:
                    continue
                lower = root_lower + finest * np.asarray((x, y, z))
                size = width * finest
                upper = lower + size
                centre = 0.5 * (lower + upper)
                refine = False
                if criterion == "curvature":
                    triangles = surface.triangles_in_box(lower - tolerance, upper + tolerance)
                    edges = surface.edges_in_box(lower - tolerance, upper + tolerance)
                    curvature = max(
                        float(np.max(np.abs(surface.triangle_curvature[triangles]), initial=0)),
                        float(np.max(np.abs(surface.edge_curvature[edges]), initial=0)),
                    )
                    refine = size > 0.2835 / (curvature + 1.0e-15)
                elif criterion == "corners":
                    if leaf_id not in corner_in_leaf:
                        continue
                    radius = 1.732 * size
                    for neighbour, point_id in corner_in_leaf.items():
                        if neighbour == leaf_id:
                            continue
                        if np.linalg.norm(surface.points[point_id] - centre) < radius:
                            selected.add(neighbour)
                            refine = True
                            break
                elif criterion == "partitions":
                    refine = surface.partitions_need_refinement(centre, 1.733 * size)
                else:
                    refine = surface.proximity_needs_refinement(centre, 1.732 * size)
                if refine:
                    selected.add(leaf_id)
            history.append(len(selected))
            if not selected:
                break
            # markAdditionalLayers(1) includes finer neighbours as well as
            # same-level/coarser ones. Query the closed leaf boundary on the
            # finest integer lattice to include every touching leaf.
            additional: set[int] = set()
            for leaf_id in selected:
                leaf = leaves[leaf_id]
                origin = np.asarray(leaf[:3])
                additional.update(map(int, lookup.in_box(origin - 1, origin + leaf[3])))
            selected.update(additional)
            leaves = refine_selected_leaves(leaves, selected, max_level, classify)
    return leaves, diagnostics
