# SPDX-License-Identifier: GPL-3.0-or-later
"""Transactional STL cut-cell recovery for the native Cartesian mesh.

The octree extractor deliberately produces Cartesian cells and split faces.
This stage changes topology only in the narrow band intersected by the input
surface. Intersected STL triangles become exact wall polygons; Cartesian
faces are clipped to their fluid-side polygon. Untouched cells therefore
remain hexahedra and 2:1 transitions remain split-face polyhedra.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import Delaunay

from ..surface_classification import SurfaceIndex, triangle_box_overlap


@dataclass(frozen=True, slots=True)
class RecoveryDiagnostics:
    """Summary of a surface-recovery transaction."""

    attempted: int
    accepted: int
    rejected: int
    partial_accepted: int = 0

    @classmethod
    def from_mesh(cls, mesh_data: dict) -> RecoveryDiagnostics:
        """Extract recovery counts emitted by the native recovery stage."""
        projection = mesh_data.get("mesh_generation", {}).get("surface_projection", {})
        attempted = int(projection.get("attempted_points", 0))
        accepted = int(projection.get("accepted_points", 0))
        partial = int(projection.get("partial_accepted_points", 0))
        rejected = max(0, attempted - accepted)
        return cls(attempted, accepted, rejected, partial)

    def as_dict(self) -> dict[str, int]:
        """Return a serialisable diagnostics snapshot."""
        return {
            "attempted": self.attempted,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "partial_accepted": self.partial_accepted,
        }


def _polygon_area_vector(points: np.ndarray) -> np.ndarray:
    centre = points.mean(axis=0)
    area = np.zeros(3, dtype=np.float64)
    for index in range(len(points)):
        area += 0.5 * np.cross(
            points[index] - centre,
            points[(index + 1) % len(points)] - centre,
        )
    return area


def _clip_polygon_axis(
    polygon: np.ndarray,
    axis: int,
    value: float,
    keep_greater: bool,
    tolerance: float,
) -> np.ndarray:
    """Clip one planar polygon against an axis-aligned half-space."""
    if len(polygon) == 0:
        return polygon
    result: list[np.ndarray] = []

    def distance(point: np.ndarray) -> float:
        return float(point[axis] - value) if keep_greater else float(value - point[axis])

    previous = polygon[-1]
    previous_distance = distance(previous)
    previous_inside = previous_distance >= -tolerance
    for current in polygon:
        current_distance = distance(current)
        current_inside = current_distance >= -tolerance
        if current_inside != previous_inside:
            denominator = previous_distance - current_distance
            fraction = previous_distance / denominator if denominator != 0.0 else 0.5
            intersection = previous + fraction * (current - previous)
            intersection[axis] = value
            result.append(intersection)
        if current_inside:
            result.append(current.copy())
        previous = current
        previous_distance = current_distance
        previous_inside = current_inside
    if not result:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(result, dtype=np.float64)


def _clip_triangle_to_box(
    triangle: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    polygon = np.asarray(triangle, dtype=np.float64).copy()
    for axis in range(3):
        polygon = _clip_polygon_axis(polygon, axis, float(lower[axis]), True, tolerance)
        polygon = _clip_polygon_axis(polygon, axis, float(upper[axis]), False, tolerance)
        if len(polygon) < 3:
            return np.empty((0, 3), dtype=np.float64)
    return polygon


def _deduplicate_coordinates(points: np.ndarray, tolerance: float) -> np.ndarray:
    if not len(points):
        return np.empty((0, 3), dtype=np.float64)
    keys = np.rint(np.asarray(points, dtype=np.float64) / tolerance).astype(np.int64)
    _unique, first = np.unique(keys, axis=0, return_index=True)
    return np.asarray(points, dtype=np.float64)[np.sort(first)]


def _convex_hull_2d(points: np.ndarray, axes: tuple[int, int], tolerance: float) -> np.ndarray:
    """Return deterministic counter-clockwise indices for a local convex polygon."""
    coordinates = points[:, axes]
    order = np.lexsort((coordinates[:, 1], coordinates[:, 0]))
    ordered = coordinates[order]

    def cross(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        ab = b - a
        ac = c - a
        return float(ab[0] * ac[1] - ab[1] * ac[0])

    lower: list[int] = []
    for local_id in range(len(ordered)):
        while len(lower) >= 2 and cross(
            ordered[lower[-2]], ordered[lower[-1]], ordered[local_id]
        ) <= tolerance:
            lower.pop()
        lower.append(local_id)
    upper: list[int] = []
    for local_id in range(len(ordered) - 1, -1, -1):
        while len(upper) >= 2 and cross(
            ordered[upper[-2]], ordered[upper[-1]], ordered[local_id]
        ) <= tolerance:
            upper.pop()
        upper.append(local_id)
    hull = lower[:-1] + upper[:-1]
    return order[np.asarray(hull, dtype=np.int64)]


def _surface_fragments(
    index: SurfaceIndex,
    lower: np.ndarray,
    upper: np.ndarray,
    tolerance: float,
    reverse_for_fluid: bool,
) -> list[np.ndarray]:
    candidate_ids = index.candidate_triangles(lower - tolerance, upper + tolerance)
    if not len(candidate_ids):
        return []
    centre = 0.5 * (lower + upper)
    half = 0.5 * (upper - lower) + tolerance
    triangles = index.triangles[candidate_ids]
    overlaps = triangle_box_overlap(
        centre,
        half,
        triangles[:, 0],
        triangles[:, 1],
        triangles[:, 2],
    )
    result: list[np.ndarray] = []
    for triangle in triangles[overlaps]:
        polygon = _clip_triangle_to_box(triangle, lower, upper, tolerance)
        polygon = _deduplicate_coordinates(polygon, tolerance)
        if len(polygon) < 3:
            continue
        source_normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        if float(np.dot(_polygon_area_vector(polygon), source_normal)) < 0.0:
            polygon = polygon[::-1].copy()
        if reverse_for_fluid:
            polygon = polygon[::-1].copy()
        if float(np.linalg.norm(_polygon_area_vector(polygon))) > tolerance**2:
            result.append(polygon)
    return result


def _merge_surface_fragments(
    fragments: list[np.ndarray], tolerance: float
) -> list[np.ndarray]:
    """Remove internal STL edges and return manifold boundary loops per cut cell."""
    if len(fragments) < 2:
        return fragments
    coordinates: list[np.ndarray] = []
    point_ids: dict[tuple[int, int, int], int] = {}

    def point_id(point: np.ndarray) -> int:
        key = tuple(np.rint(point / tolerance).astype(np.int64))
        found = point_ids.get(key)
        if found is None:
            found = len(coordinates)
            point_ids[key] = found
            coordinates.append(np.asarray(point, dtype=np.float64).copy())
        return found

    edge_entries: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for polygon in fragments:
        ids = [point_id(point) for point in polygon]
        for index, first in enumerate(ids):
            second = ids[(index + 1) % len(ids)]
            key = (min(first, second), max(first, second))
            edge_entries.setdefault(key, []).append((first, second))
    boundary_edges = [entries[0] for entries in edge_entries.values() if len(entries) == 1]
    if not boundary_edges:
        return fragments
    if any(len(entries) > 2 for entries in edge_entries.values()):
        raise ValueError("Recovered STL fragments form a non-manifold cut-cell wall")
    outgoing: dict[int, list[int]] = {}
    incoming: dict[int, int] = {}
    for first, second in boundary_edges:
        outgoing.setdefault(first, []).append(second)
        incoming[second] = incoming.get(second, 0) + 1
    if any(len(values) != 1 for values in outgoing.values()) or any(
        count != 1 for count in incoming.values()
    ):
        # A surface can legitimately create multiple local sheets in a cell;
        # preserve exact fragments and let the strict topology gate decide.
        return fragments
    remaining = set(boundary_edges)
    loops: list[np.ndarray] = []
    while remaining:
        first_edge = min(remaining)
        start, current = first_edge
        loop = [start]
        remaining.remove(first_edge)
        while current != start:
            loop.append(current)
            following = outgoing[current][0]
            edge = (current, following)
            if edge not in remaining:
                return fragments
            remaining.remove(edge)
            current = following
        polygon = np.asarray([coordinates[index] for index in loop], dtype=np.float64)
        if len(polygon) >= 3 and np.linalg.norm(_polygon_area_vector(polygon)) > tolerance**2:
            loops.append(polygon)
    return loops or fragments


def _merge_coplanar_surface_fragments(
    fragments: list[np.ndarray], tolerance: float
) -> list[np.ndarray]:
    """Remove artificial STL diagonals without flattening curved geometry."""
    if len(fragments) < 2:
        return fragments
    groups: list[tuple[np.ndarray, float, list[np.ndarray]]] = []
    for polygon in fragments:
        area = _polygon_area_vector(polygon)
        normal = area / np.linalg.norm(area)
        plane = float(np.dot(normal, polygon[0]))
        for group_normal, group_plane, members in groups:
            if (
                float(np.dot(normal, group_normal)) >= 1.0 - 1.0e-10
                and abs(plane - group_plane) <= 10.0 * tolerance
            ):
                members.append(polygon)
                break
        else:
            groups.append((normal, plane, [polygon]))
    merged: list[np.ndarray] = []
    for _normal, _plane, members in groups:
        merged.extend(_merge_surface_fragments(members, tolerance))
    return merged


def _face_fluid_polygons(
    original: np.ndarray,
    fragment_points: list[np.ndarray],
    surface_indices: tuple[SurfaceIndex, ...],
    tolerance: float,
) -> list[np.ndarray]:
    """Clip one Cartesian face into one or more fluid-side polygons."""
    span = np.ptp(original, axis=0)
    axis = int(np.argmin(span))
    tangential = tuple(value for value in range(3) if value != axis)
    plane = float(original[:, axis].mean())
    outside = np.ones(len(original), dtype=bool)
    for index in surface_indices:
        outside &= ~index.is_inside(original)
    candidates = [point for point, keep in zip(original, outside, strict=True) if keep]
    for polygon in fragment_points:
        for point in polygon:
            if abs(float(point[axis] - plane)) <= tolerance:
                candidates.append(point)
    if not candidates:
        return []
    points = _deduplicate_coordinates(np.asarray(candidates), tolerance)
    if len(points) < 3:
        return []
    if fragment_points:
        # First recover an exact planar boundary graph. Most cuts produce one
        # concave fluid polygon; preserving that loop avoids the near-zero
        # simplices generated when several intersection points are almost
        # collinear.
        point_lookup = {
            tuple(np.rint(point / tolerance).astype(np.int64)): point_id
            for point_id, point in enumerate(points)
        }

        def local_id(point: np.ndarray) -> int | None:
            return point_lookup.get(tuple(np.rint(point / tolerance).astype(np.int64)))

        segments: set[tuple[int, int]] = set()
        for edge, first_point in enumerate(original):
            second_point = original[(edge + 1) % len(original)]
            direction = second_point - first_point
            length_squared = float(np.dot(direction, direction))
            parameter = (points - first_point) @ direction / length_squared
            projection = first_point + parameter[:, None] * direction
            on_edge = np.flatnonzero(
                (parameter >= -1.0e-10)
                & (parameter <= 1.0 + 1.0e-10)
                & (np.linalg.norm(points - projection, axis=1) <= 10.0 * tolerance)
            )
            ordered = on_edge[np.argsort(parameter[on_edge])]
            for first_id, second_id in zip(ordered[:-1], ordered[1:], strict=True):
                midpoint = 0.5 * (points[first_id] + points[second_id])
                if any(
                    bool(index.is_inside(midpoint[None, :])[0])
                    for index in surface_indices
                ):
                    continue
                segments.add(tuple(sorted((int(first_id), int(second_id)))))
        for fragment in fragment_points:
            for edge, first_point in enumerate(fragment):
                second_point = fragment[(edge + 1) % len(fragment)]
                if (
                    abs(float(first_point[axis] - plane)) > tolerance
                    or abs(float(second_point[axis] - plane)) > tolerance
                ):
                    continue
                first_id = local_id(first_point)
                second_id = local_id(second_point)
                if first_id is not None and second_id is not None and first_id != second_id:
                    segments.add(tuple(sorted((first_id, second_id))))
        adjacency: dict[int, list[int]] = {}
        for first_id, second_id in segments:
            adjacency.setdefault(first_id, []).append(second_id)
            adjacency.setdefault(second_id, []).append(first_id)
        if adjacency and all(len(neighbours) == 2 for neighbours in adjacency.values()):
            remaining = set(segments)
            loops: list[np.ndarray] = []
            while remaining:
                start, current = min(remaining)
                previous = start
                loop = [start]
                remaining.remove((min(start, current), max(start, current)))
                while current != start:
                    loop.append(current)
                    choices = [value for value in adjacency[current] if value != previous]
                    if len(choices) != 1:
                        loops = []
                        break
                    following = choices[0]
                    segment = (min(current, following), max(current, following))
                    if following != start and segment not in remaining:
                        loops = []
                        break
                    remaining.discard(segment)
                    previous, current = current, following
                if not loops and current != start:
                    break
                loops.append(points[np.asarray(loop, dtype=np.int64)])
            if len(loops) == 1 and len(loops[0]) >= 3:
                polygon = loops[0]
                original_area = _polygon_area_vector(original)
                if float(np.dot(_polygon_area_vector(polygon), original_area)) < 0.0:
                    polygon = polygon[::-1].copy()
                return [polygon]

        # Surface cuts make the fluid portion concave or, when every original
        # corner remains fluid, perforated. A convex hull would bridge across
        # the solid and leave overlapping cell faces. Triangulate the planar
        # arrangement and retain only fluid-side simplices.
        coordinates_2d = points[:, tangential]
        triangulation = Delaunay(coordinates_2d)
        original_area = _polygon_area_vector(original)
        result: list[np.ndarray] = []
        for simplex in triangulation.simplices:
            polygon = points[np.asarray(simplex, dtype=np.int64)]
            triangle_centre = polygon.mean(axis=0, keepdims=True)
            if any(
                bool(index.is_inside(triangle_centre)[0])
                for index in surface_indices
            ):
                continue
            if float(np.dot(_polygon_area_vector(polygon), original_area)) < 0.0:
                polygon = polygon[::-1].copy()
            if np.linalg.norm(_polygon_area_vector(polygon)) > tolerance**2:
                result.append(polygon)
        return result
    hull = _convex_hull_2d(points, tangential, tolerance**2)
    polygon = points[hull]
    if len(polygon) < 3:
        return []
    original_area = _polygon_area_vector(original)
    if float(np.dot(_polygon_area_vector(polygon), original_area)) < 0.0:
        polygon = polygon[::-1].copy()
    return [polygon]


class _PointRegistry:
    def __init__(self, points: np.ndarray, tolerance: float) -> None:
        self.tolerance = tolerance
        self.points = [np.asarray(point, dtype=np.float64).copy() for point in points]
        self.lookup = {
            tuple(np.rint(point / tolerance).astype(np.int64)): point_id
            for point_id, point in enumerate(points)
        }

    def ids(self, coordinates: np.ndarray) -> np.ndarray:
        result = np.empty(len(coordinates), dtype=np.int32)
        for index, point in enumerate(coordinates):
            key = tuple(np.rint(point / self.tolerance).astype(np.int64))
            point_id = self.lookup.get(key)
            if point_id is None:
                point_id = len(self.points)
                self.lookup[key] = point_id
                self.points.append(np.asarray(point, dtype=np.float64).copy())
            result[index] = point_id
        return result


def recover_cut_cells(
    mesh_data: dict[str, Any],
    surface_indices: tuple[SurfaceIndex, ...],
    wall_patch_name: str,
) -> dict[str, Any]:
    """Replace the Cartesian staircase by exact STL cut-cell topology.

    Memory is proportional to the narrow intersected band. The packed core
    arrays are retained, and fragments are discarded after face assembly.
    """
    if not surface_indices:
        return mesh_data
    if "cell_vertex_indices" not in mesh_data:
        raise ValueError("Cut-cell recovery requires Cartesian cell corner metadata")

    points = np.asarray(mesh_data["vertex_position"], dtype=np.float64)
    cell_vertices = np.asarray(mesh_data["cell_vertex_indices"], dtype=np.int64)
    cell_points = points[cell_vertices]
    lower = cell_points.min(axis=1)
    upper = cell_points.max(axis=1)
    scale = max(float(np.ptp(points, axis=0).max()), 1.0)
    tolerance = max(1.0e-11 * scale, np.finfo(np.float64).eps * scale * 64.0)

    # STL orientation may be consistently inward or outward.  Positive signed
    # volume is outward, so the external-fluid wall uses the reversed order;
    # negative signed volume already points into the solid and is retained.
    reverse_for_fluid = tuple(
        float(
            np.einsum(
                "ij,ij->i",
                index.triangles[:, 0],
                np.cross(index.triangles[:, 1], index.triangles[:, 2]),
            ).sum()
            / 6.0
        )
        > 0.0
        for index in surface_indices
    )
    cut_fragments: dict[int, list[np.ndarray]] = {}
    for cell_id in range(len(cell_vertices)):
        fragments: list[np.ndarray] = []
        for index, reverse in zip(surface_indices, reverse_for_fluid, strict=True):
            if index.box_intersects_surface(lower[cell_id], upper[cell_id]):
                fragments.extend(
                    _surface_fragments(
                        index,
                        lower[cell_id],
                        upper[cell_id],
                        tolerance,
                        reverse,
                    )
                )
        if fragments:
            cut_fragments[cell_id] = _merge_coplanar_surface_fragments(
                fragments, tolerance
            )
    if not cut_fragments:
        raise ValueError("Surface recovery found no intersected fluid-side Cartesian cells")

    n_internal = int(mesh_data["n_interior_faces"])
    owners = np.asarray(mesh_data["owners"], dtype=np.int64)
    neighbours = np.asarray(mesh_data["neighbours"], dtype=np.int64)
    face_patch: list[str | None] = [None] * n_internal
    for patch in mesh_data["boundary"]:
        start = int(patch["start_face"])
        stop = start + int(patch["n_faces"])
        face_patch.extend([str(patch["name"])] * (stop - start))
    if len(face_patch) != int(mesh_data["n_faces"]):
        raise ValueError("Boundary patch ranges do not cover all extracted faces")

    registry = _PointRegistry(points, tolerance)
    internal_faces: list[np.ndarray] = []
    internal_owners: list[int] = []
    internal_neighbours: list[int] = []
    patch_order = [str(patch["name"]) for patch in mesh_data["boundary"]]
    patch_types = {
        str(patch["name"]): str(patch.get("type", "patch"))
        for patch in mesh_data["boundary"]
    }
    patch_faces: dict[str, list[np.ndarray]] = {name: [] for name in patch_order}
    patch_owners: dict[str, list[int]] = {name: [] for name in patch_order}

    for face_id, source_face in enumerate(mesh_data["faces"]):
        patch_name = face_patch[face_id]
        if patch_name == wall_patch_name:
            continue
        owner = int(owners[face_id])
        neighbour = int(neighbours[face_id]) if face_id < n_internal else -1
        incident = [owner] + ([neighbour] if neighbour >= 0 else [])
        fragments = [
            polygon
            for cell_id in incident
            for polygon in cut_fragments.get(cell_id, ())
        ]
        original = points[np.asarray(source_face, dtype=np.int64)]
        polygons = (
            _face_fluid_polygons(original, fragments, surface_indices, tolerance)
            if fragments
            else [original]
        )
        for polygon in polygons:
            if len(polygon) < 3:
                continue
            face = registry.ids(polygon)
            if neighbour >= 0:
                internal_faces.append(face)
                internal_owners.append(owner)
                internal_neighbours.append(neighbour)
            else:
                if patch_name is None:
                    raise ValueError("Extracted boundary face has no patch")
                patch_faces[patch_name].append(face)
                patch_owners[patch_name].append(owner)

    for cell_id, fragments in cut_fragments.items():
        for source_polygon in fragments:
            # Fragment order was normalised to the external-fluid wall normal
            # while clipping the source surface.
            polygon = source_polygon
            patch_faces[wall_patch_name].append(registry.ids(polygon))
            patch_owners[wall_patch_name].append(cell_id)

    face_blocks: list[np.ndarray] = list(internal_faces)
    owner_blocks = list(internal_owners)
    boundary: list[dict[str, Any]] = []
    start_face = len(face_blocks)
    for patch_name in patch_order:
        faces = patch_faces[patch_name]
        boundary.append(
            {
                "name": patch_name,
                "start_face": start_face,
                "n_faces": len(faces),
                "type": patch_types[patch_name],
            }
        )
        face_blocks.extend(faces)
        owner_blocks.extend(patch_owners[patch_name])
        start_face += len(faces)

    if not patch_faces[wall_patch_name]:
        raise ValueError("Surface recovery produced no conformal wall polygons")
    used = np.unique(np.concatenate(face_blocks))
    point_lookup = np.full(len(registry.points), -1, dtype=np.int64)
    point_lookup[used] = np.arange(len(used), dtype=np.int64)
    rebuilt_faces = [
        np.asarray(point_lookup[np.asarray(face, dtype=np.int64)], dtype=np.int32)
        for face in face_blocks
    ]
    rebuilt_points = np.ascontiguousarray(
        np.asarray(registry.points, dtype=np.float64)[used]
    )
    # Recovered cell centroids must be based on the actual cut topology, not
    # on the original Cartesian corners.  Orient only the narrow cut band;
    # untouched packed-core faces retain the extractor's proven orientation.
    cut_ids = set(cut_fragments)
    cut_point_ids: dict[int, set[int]] = {cell_id: set() for cell_id in cut_ids}
    rebuilt_owners = np.asarray(owner_blocks, dtype=np.int32)
    rebuilt_neighbours = np.asarray(internal_neighbours, dtype=np.int32)
    for face_id, face in enumerate(rebuilt_faces):
        owner = int(rebuilt_owners[face_id])
        if owner in cut_point_ids:
            cut_point_ids[owner].update(map(int, face))
        if face_id < len(rebuilt_neighbours):
            neighbour = int(rebuilt_neighbours[face_id])
            if neighbour in cut_point_ids:
                cut_point_ids[neighbour].update(map(int, face))
    cut_centres = {
        cell_id: rebuilt_points[np.asarray(sorted(point_ids), dtype=np.int64)].mean(axis=0)
        for cell_id, point_ids in cut_point_ids.items()
        if point_ids
    }
    original_centres = cell_points.mean(axis=1)
    wall_patch = next(patch for patch in boundary if patch["name"] == wall_patch_name)
    wall_start = int(wall_patch["start_face"])
    wall_stop = wall_start + int(wall_patch["n_faces"])
    for face_id, face in enumerate(rebuilt_faces):
        if wall_start <= face_id < wall_stop:
            continue
        owner = int(rebuilt_owners[face_id])
        neighbour = (
            int(rebuilt_neighbours[face_id]) if face_id < len(rebuilt_neighbours) else -1
        )
        if owner not in cut_ids and neighbour not in cut_ids:
            continue
        coordinates = rebuilt_points[np.asarray(face, dtype=np.int64)]
        face_centre = coordinates.mean(axis=0)
        owner_centre = cut_centres.get(owner, original_centres[owner])
        direction = (
            cut_centres.get(neighbour, original_centres[neighbour]) - owner_centre
            if neighbour >= 0
            else face_centre - owner_centre
        )
        if float(np.dot(_polygon_area_vector(coordinates), direction)) < 0.0:
            rebuilt_faces[face_id] = face[::-1].copy()
    widths = {len(face) for face in rebuilt_faces}
    mesh_data["faces"] = (
        np.ascontiguousarray(rebuilt_faces, dtype=np.int32)
        if len(widths) == 1
        else rebuilt_faces
    )
    mesh_data["vertex_position"] = rebuilt_points
    mesh_data["owners"] = np.ascontiguousarray(rebuilt_owners, dtype=np.int32)
    mesh_data["neighbours"] = np.ascontiguousarray(rebuilt_neighbours, dtype=np.int32)
    mesh_data["boundary"] = boundary
    mesh_data["n_faces"] = len(rebuilt_faces)
    mesh_data["n_interior_faces"] = len(internal_faces)
    mesh_data["n_points"] = len(used)
    mesh_data.pop("cell_vertex_indices", None)
    mesh_data.pop("cell_type_code", None)
    recovered_points = int(
        sum(len(polygon) for value in cut_fragments.values() for polygon in value)
    )
    generation = mesh_data.setdefault("mesh_generation", {})
    generation["surface_projection"] = {
        "method": "stl_cut_cell",
        "attempted_points": recovered_points,
        "accepted_points": recovered_points,
        "partial_accepted_points": 0,
        "cut_cells": len(cut_fragments),
        "wall_fragments": int(sum(len(value) for value in cut_fragments.values())),
    }
    # Native validation defines orientation against face-pyramid cell
    # centroids.  Concave cut cells (for example an inner toroidal wall) can
    # place that centroid on a different side than a vertex-average estimate.
    # Use the same bounded face geometry here and converge the orientation
    # transaction before the mesh is allowed to leave this stage.
    from ..geometry import compute_mesh_geometry

    for _iteration in range(4):
        geometry = compute_mesh_geometry(mesh_data, compute_lsq=False)
        area = np.asarray(geometry["face_area_vector"], dtype=np.float64)
        face_centre = np.asarray(geometry["face_centre"], dtype=np.float64)
        cell_centre = np.asarray(geometry["cell_centre"], dtype=np.float64)
        direction = np.empty_like(area)
        n_internal_faces = int(mesh_data["n_interior_faces"])
        direction[:n_internal_faces] = (
            cell_centre[mesh_data["neighbours"]]
            - cell_centre[mesh_data["owners"][:n_internal_faces]]
        )
        direction[n_internal_faces:] = (
            face_centre[n_internal_faces:]
            - cell_centre[mesh_data["owners"][n_internal_faces:]]
        )
        reversed_ids = np.flatnonzero(np.einsum("ij,ij->i", area, direction) < 0.0)
        if not len(reversed_ids):
            break
        for face_id in reversed_ids:
            rebuilt_faces[int(face_id)] = rebuilt_faces[int(face_id)][::-1].copy()
        widths = {len(face) for face in rebuilt_faces}
        mesh_data["faces"] = (
            np.ascontiguousarray(rebuilt_faces, dtype=np.int32)
            if len(widths) == 1
            else rebuilt_faces
        )
    else:
        raise ValueError("Cut-cell face orientation did not converge")
    return mesh_data


__all__ = ["RecoveryDiagnostics", "recover_cut_cells"]
