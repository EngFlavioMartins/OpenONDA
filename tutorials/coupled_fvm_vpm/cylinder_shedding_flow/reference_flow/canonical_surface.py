"""Build the frozen fluid-boundary representation for the cylinder case.

The source cylinder is deliberately longer than the reference domain.  This
module clips only its side wall at the two span planes and triangulates the
remaining box-plane annuli, leaving the cylinder holes open.  The generated
ASCII STL is used by the native oracle; the Python mesher uses the clipped
wall triangles and its declarative outer box patches.
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import tempfile
from typing import Iterable

import numpy as np
import vtk

from source.solvers.fvm.mesh.triangulated_surface import TriangulatedSurface


try:
    from .case_definition import DOMAIN
except ImportError:
    from case_definition import DOMAIN  # pyrefly: ignore [missing-import]


def _clip_polygon_z(polygon: np.ndarray, z_min: float, z_max: float) -> np.ndarray:
    """Clip one triangle polygon against the two span planes in float64."""

    def clip(values: np.ndarray, bound: float, keep_above: bool) -> np.ndarray:
        output: list[np.ndarray] = []
        for first, second in zip(values, np.roll(values, -1, axis=0), strict=True):
            first_inside = bool(first[2] >= bound if keep_above else first[2] <= bound)
            second_inside = bool(second[2] >= bound if keep_above else second[2] <= bound)
            if first_inside != second_inside:
                fraction = (bound - first[2]) / (second[2] - first[2])
                output.append(first + fraction * (second - first))
            if second_inside:
                output.append(second.copy())
        return np.asarray(output, dtype=np.float64)

    result = clip(polygon, z_min, True)
    if len(result) >= 3:
        result = clip(result, z_max, False)
    return result


def _triangulate_polygon(values: np.ndarray) -> np.ndarray:
    """Fan-triangulate a convex clipped polygon without changing precision."""
    return np.asarray(
        [(values[0], values[index], values[index + 1]) for index in range(1, len(values) - 1)],
        dtype=np.float64,
    )


def clipped_wall_triangles(source: Path, *, z_min: float, z_max: float) -> np.ndarray:
    """Return source side-wall triangles clipped to the reference span."""
    surface = TriangulatedSurface.from_stl(source)
    triangles = np.asarray(surface.triangles, dtype=np.float64)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    side = np.abs(normals[:, 2]) <= 1.0e-12 * np.maximum(lengths, 1.0)
    clipped: list[np.ndarray] = []
    for triangle in triangles[side]:
        polygon = _clip_polygon_z(triangle, z_min, z_max)
        if len(polygon) >= 3:
            clipped.extend(_triangulate_polygon(polygon))
    if not clipped:
        raise ValueError("The cylinder source contains no side-wall triangles in the span")
    return np.ascontiguousarray(clipped, dtype=np.float64)


def _boundary_loop(points: np.ndarray) -> list[tuple[float, float, float]]:
    """Return the ordered, de-duplicated cylinder perimeter at one span plane."""
    points = np.asarray(points, dtype=np.float64)
    unique = np.unique(points, axis=0)
    centre = unique[:, :2].mean(axis=0)
    angles = np.arctan2(unique[:, 1] - centre[1], unique[:, 0] - centre[0])
    order = np.argsort(angles, kind="mergesort")
    return [
        (float(unique[index, 0]), float(unique[index, 1]), float(unique[index, 2]))
        for index in order
    ]


def _boundary_perimeter(wall: np.ndarray, z: float) -> list[tuple[float, float, float]]:
    """Recover the actual clipped-wall edge loop, including triangulation turns."""
    edges: dict[tuple[tuple[float, float, float], tuple[float, float, float]], int] = {}

    def key(point: np.ndarray) -> tuple[float, float, float]:
        return tuple(round(float(value), 12) for value in point)  # type: ignore[return-value]

    for triangle in wall:
        for first, second in zip(triangle, np.roll(triangle, -1, axis=0), strict=True):
            left = key(first)
            right = key(second)
            edge_key = (left, right) if left < right else (right, left)
            edges[edge_key] = edges.get(edge_key, 0) + 1
    perimeter_edges = [
        edge
        for edge, count in edges.items()
        if count == 1
        and np.isclose(edge[0][2], z, rtol=0.0, atol=1.0e-12)
        and np.isclose(edge[1][2], z, rtol=0.0, atol=1.0e-12)
    ]
    if not perimeter_edges:
        raise ValueError(f"No clipped-wall perimeter edges found at z={z}")
    neighbours: dict[tuple[float, float, float], list[tuple[float, float, float]]] = {}
    for first, second in perimeter_edges:
        neighbours.setdefault(first, []).append(second)
        neighbours.setdefault(second, []).append(first)
    if any(len(values) != 2 for values in neighbours.values()):
        raise ValueError(f"Clipped-wall perimeter at z={z} is not a single edge loop")
    start = min(neighbours)
    result = [start]
    previous: tuple[float, float, float] | None = None
    current = start
    while True:
        candidates = neighbours[current]
        following = candidates[0] if candidates[0] != previous else candidates[1]
        if following == start:
            break
        result.append(following)
        previous, current = current, following
        if len(result) > len(perimeter_edges):
            raise ValueError(f"Clipped-wall perimeter at z={z} did not close")
    return result


def _plane_annulus(
    z: float,
    perimeter: Iterable[tuple[float, float, float]],
    *,
    outward_sign: float,
    domain=DOMAIN,
) -> np.ndarray:
    """Triangulate a rectangular plane with the cylinder hole preserved."""
    xmin, xmax, ymin, ymax, _zmin, _zmax = domain
    inner = list(perimeter)
    outer = [(xmin, ymin, z), (xmax, ymin, z), (xmax, ymax, z), (xmin, ymax, z)]
    points = vtk.vtkPoints()
    points.SetDataTypeToDouble()
    for point in outer + inner:
        points.InsertNextPoint(*point)
    lines = vtk.vtkCellArray()
    for sequence in (list(range(4)), list(range(4, 4 + len(inner)))):
        polyline = vtk.vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(len(sequence) + 1)
        for position, point_id in enumerate(sequence + [sequence[0]]):
            polyline.GetPointIds().SetId(position, point_id)
        lines.InsertNextCell(polyline)
    source = vtk.vtkPolyData()
    source.SetPoints(points)
    source.SetLines(lines)
    triangulator = vtk.vtkContourTriangulator()
    triangulator.SetInputData(source)
    triangulator.Update()
    result = triangulator.GetOutput()
    triangles: list[np.ndarray] = []
    for cell_id in range(result.GetNumberOfCells()):
        cell = result.GetCell(cell_id)
        values = np.asarray(
            [result.GetPoint(cell.GetPointId(index)) for index in range(3)], dtype=np.float64
        )
        normal = np.cross(values[1] - values[0], values[2] - values[0])
        if float(np.linalg.norm(normal)) <= 1.0e-12:
            continue
        if normal[2] * outward_sign < 0.0:
            values[[1, 2]] = values[[2, 1]]
        triangles.append(values)
    if not triangles:
        raise ValueError(f"Could not triangulate the span-plane annulus at z={z}")
    return np.asarray(triangles, dtype=np.float64)


def fluid_wall_triangles(source: Path) -> np.ndarray:
    """Return clipped wall plus the two open span-plane annuli."""
    wall = clipped_wall_triangles(source, z_min=DOMAIN[4], z_max=DOMAIN[5])
    return np.concatenate(
        (
            wall,
            _plane_annulus(DOMAIN[4], _boundary_perimeter(wall, DOMAIN[4]), outward_sign=-1.0),
            _plane_annulus(DOMAIN[5], _boundary_perimeter(wall, DOMAIN[5]), outward_sign=1.0),
        ),
    )


def _native_cylinder_triangles(source: Path) -> np.ndarray:
    """Return the cylinder side plus the two span annuli for native input."""
    return fluid_wall_triangles(source)


def _box_triangles(domain=DOMAIN) -> dict[str, np.ndarray]:
    """Return outward-oriented outer box triangles grouped by patch."""
    xmin, xmax, ymin, ymax, zmin, zmax = domain
    c = np.asarray(
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
    faces = {
        "inlet": (0, 1, 3, 2),
        "outlet": (4, 6, 7, 5),
        "ymin": (0, 4, 5, 1),
        "ymax": (2, 3, 7, 6),
        "zmin": (0, 2, 6, 4),
        "zmax": (1, 5, 7, 3),
    }
    return {
        name: np.asarray(
            (c[ids[0]], c[ids[1]], c[ids[2]], c[ids[0]], c[ids[2]], c[ids[3]])
        ).reshape(2, 3, 3)
        for name, ids in faces.items()
    }


def _orient_closed_surface(triangles: np.ndarray) -> np.ndarray:
    """Orient a connected closed triangle soup consistently and outward."""
    values = np.asarray(triangles, dtype=np.float64).copy()
    scale = max(float(np.ptp(values, axis=(0, 1)).max()), 1.0)
    tolerance = scale * 1.0e-9
    quantized = np.rint(values.reshape(-1, 3) / tolerance).astype(np.int64)
    _, point_ids = np.unique(quantized, axis=0, return_inverse=True)
    point_ids = point_ids.reshape(-1, 3)
    edges: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for triangle_id, triangle in enumerate(point_ids):
        for position, start in enumerate(triangle):
            end = int(triangle[(position + 1) % 3])
            start = int(start)
            edge = (min(start, end), max(start, end))
            edges.setdefault(edge, []).append((triangle_id, start, end))
    if any(len(items) != 2 for items in edges.values()):
        raise ValueError("Canonical native fluid surface is not closed after clipping")
    adjacency: dict[int, list[tuple[int, bool]]] = {}
    for items in edges.values():
        first, second = items
        same_direction = first[1] == second[1] and first[2] == second[2]
        adjacency.setdefault(first[0], []).append((second[0], same_direction))
        adjacency.setdefault(second[0], []).append((first[0], same_direction))
    orientation: dict[int, bool] = {0: False}
    queue = [0]
    while queue:
        current = queue.pop(0)
        for neighbour, same_direction in adjacency[current]:
            required = orientation[current] ^ same_direction
            previous = orientation.get(neighbour)
            if previous is None:
                orientation[neighbour] = required
                queue.append(neighbour)
            elif previous != required:
                raise ValueError("Canonical native surface has an orientation conflict")
    if len(orientation) != len(values):
        raise ValueError("Canonical native fluid surface is disconnected")
    for triangle_id, reverse in orientation.items():
        if reverse:
            values[triangle_id, [1, 2]] = values[triangle_id, [2, 1]]
    volume = float(
        np.einsum("ij,ij->i", values[:, 0], np.cross(values[:, 1], values[:, 2])).sum() / 6.0
    )
    if volume < 0.0:
        values[:, [1, 2]] = values[:, [2, 1]]
    return values


def _write_ascii_stl(groups: dict[str, np.ndarray], path: Path) -> None:
    lines: list[str] = []
    for name in sorted(groups):
        lines.append(f"solid {name}")
        for triangle in np.asarray(groups[name], dtype=np.float64):
            normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            length = float(np.linalg.norm(normal))
            if length:
                normal /= length
            lines.append(f"  facet normal {normal[0]:.17g} {normal[1]:.17g} {normal[2]:.17g}")
            lines.append("    outer loop")
            lines.extend(
                f"      vertex {point[0]:.17g} {point[1]:.17g} {point[2]:.17g}"
                for point in triangle
            )
            lines.extend(("    endloop", "  endfacet"))
        lines.append(f"endsolid {name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            stream.write("\n".join(lines) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def prepare_canonical_surfaces(
    source: Path, directory: Path, *, domain=DOMAIN
) -> dict[str, Path | str | int]:
    """Materialize and hash the shared Python/native fluid-boundary inputs."""
    wall = clipped_wall_triangles(source, z_min=domain[4], z_max=domain[5])
    wall_path = directory / "cylinder_fluid_wall.stl"
    native_path = directory / "native_geometry.stl"
    _write_ascii_stl({"cylinder": wall}, wall_path)
    groups = _box_triangles(domain)
    perimeter_lower = _boundary_perimeter(wall, domain[4])
    perimeter_upper = _boundary_perimeter(wall, domain[5])
    groups["zmin"] = _plane_annulus(domain[4], perimeter_lower, outward_sign=-1.0, domain=domain)
    groups["zmax"] = _plane_annulus(domain[5], perimeter_upper, outward_sign=1.0, domain=domain)
    # The span annuli belong to the outer z patches below; the cylinder solid
    # contributes only its clipped side wall, so no disk is introduced.
    groups["cylinder"] = wall
    names = tuple(groups)
    counts = [len(groups[name]) for name in names]
    oriented = _orient_closed_surface(np.concatenate([groups[name] for name in names]))
    offset = 0
    groups = {}
    for name, count in zip(names, counts, strict=True):
        groups[name] = oriented[offset : offset + count]
        offset += count
    _write_ascii_stl(groups, native_path)
    return {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "wall_path": wall_path,
        "native_path": native_path,
        "wall_sha256": hashlib.sha256(wall_path.read_bytes()).hexdigest(),
        "native_sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
        "wall_triangles": int(len(wall)),
    }


__all__ = ["DOMAIN", "fluid_wall_triangles", "prepare_canonical_surfaces"]
