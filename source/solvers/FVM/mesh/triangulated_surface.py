"""Read and validate triangulated surfaces used by the native Cartesian mesher."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import struct
from typing import TypeAlias, cast

import numpy as np

SurfaceBounds: TypeAlias = tuple[float, float, float, float, float, float]


def _ascii_triangles(data: bytes, path: Path) -> np.ndarray:
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Cannot decode STL surface as ASCII: {path}") from exc

    vertices: list[tuple[float, float, float]] = []
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0].lower() == "vertex":
            if len(fields) != 4:
                raise ValueError(f"Malformed STL vertex in {path}: {line.strip()}")
            try:
                vertices.append((float(fields[1]), float(fields[2]), float(fields[3])))
            except ValueError as exc:
                raise ValueError(f"Malformed STL vertex in {path}: {line.strip()}") from exc

    if not vertices or len(vertices) % 3:
        raise ValueError(f"STL surface contains no complete triangles: {path}")
    return np.asarray(vertices, dtype=np.float64).reshape(-1, 3, 3)


def _binary_triangles(data: bytes, count: int) -> np.ndarray:
    record = np.dtype(
        [
            ("normal", "<f4", (3,)),
            ("vertices", "<f4", (3, 3)),
            ("attribute", "<u2"),
        ]
    )
    facets = np.frombuffer(data, dtype=record, count=count, offset=84)
    return np.asarray(facets["vertices"], dtype=np.float64)


def _read_stl(path: Path) -> tuple[np.ndarray, bytes]:
    if path.suffix.lower() != ".stl":
        raise ValueError(f"surface_file must be an STL file, got: {path}")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise FileNotFoundError(f"Cannot read STL surface: {path}") from exc

    binary_count = struct.unpack_from("<I", data, 80)[0] if len(data) >= 84 else -1
    binary_size = 84 + 50 * binary_count
    if binary_count >= 0 and binary_size == len(data):
        triangles = _binary_triangles(data, binary_count)
    else:
        triangles = _ascii_triangles(data, path)
    return np.ascontiguousarray(triangles), data


def _validate_triangles(triangles: np.ndarray, path: Path) -> None:
    if len(triangles) < 4:
        raise ValueError(f"STL surface must contain at least four triangles: {path}")
    if not np.all(np.isfinite(triangles)):
        raise ValueError(f"STL surface contains non-finite coordinates: {path}")

    edges_a = triangles[:, 1] - triangles[:, 0]
    edges_b = triangles[:, 2] - triangles[:, 0]
    doubled_areas = np.linalg.norm(np.cross(edges_a, edges_b), axis=1)
    scale = max(float(np.ptp(triangles, axis=(0, 1)).max()), 1.0)
    if np.any(doubled_areas <= np.finfo(np.float64).eps * scale * scale * 32.0):
        raise ValueError(f"STL surface contains degenerate triangles: {path}")

    tolerance = scale * 1.0e-9
    quantized = np.rint(triangles.reshape(-1, 3) / tolerance).astype(np.int64)
    _, vertex_ids = np.unique(quantized, axis=0, return_inverse=True)
    triangle_ids = vertex_ids.reshape(-1, 3)
    edges = np.concatenate(
        (
            triangle_ids[:, (0, 1)],
            triangle_ids[:, (1, 2)],
            triangle_ids[:, (2, 0)],
        )
    )
    edges.sort(axis=1)
    _, edge_counts = np.unique(edges, axis=0, return_counts=True)
    if np.any(edge_counts != 2):
        raise ValueError(f"STL surface is not watertight: {path}")


def _axis_aligned_box_bounds(triangles: np.ndarray, path: Path) -> SurfaceBounds:
    lower = triangles.min(axis=(0, 1))
    upper = triangles.max(axis=(0, 1))
    extent = upper - lower
    if np.any(extent <= 0.0):
        raise ValueError(f"STL surface has zero extent: {path}")

    tolerance = max(float(extent.max()), 1.0) * 1.0e-8
    covered_planes: set[tuple[int, int]] = set()
    for triangle in triangles:
        plane = None
        for axis in range(3):
            coordinate = float(triangle[0, axis])
            if not np.allclose(triangle[:, axis], coordinate, rtol=0.0, atol=tolerance):
                continue
            if _coordinates_close(coordinate, float(lower[axis]), tolerance):
                plane = (axis, 0)
            elif _coordinates_close(coordinate, float(upper[axis]), tolerance):
                plane = (axis, 1)
        if plane is None:
            raise ValueError(
                "The native Cartesian mesher currently requires every STL triangle "
                f"to lie on an axis-aligned bounding-box plane: {path}"
            )
        covered_planes.add(plane)

    expected_planes = {(axis, side) for axis in range(3) for side in range(2)}
    if covered_planes != expected_planes:
        raise ValueError(f"STL surface does not cover all six sides of a closed box: {path}")

    doubled_areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    expected_area = 2.0 * (extent[0] * extent[1] + extent[1] * extent[2] + extent[0] * extent[2])
    if not np.isclose(0.5 * doubled_areas.sum(), expected_area, rtol=1.0e-8, atol=tolerance**2):
        raise ValueError(f"STL triangles do not tessellate the complete box surface: {path}")

    return cast(
        SurfaceBounds,
        tuple(float(value) for axis in range(3) for value in (lower[axis], upper[axis])),
    )


def _coordinates_close(left: float, right: float, tolerance: float) -> bool:
    """Absolute comparison used for geometric plane classification."""
    return abs(left - right) <= tolerance


@dataclass(frozen=True)
class TriangulatedSurface:
    """Validated STL surface and the Cartesian solid bounds it defines."""

    path: Path
    triangles: np.ndarray
    bounds: SurfaceBounds
    sha256: str

    @classmethod
    def from_stl(cls, surface_file: str | Path) -> TriangulatedSurface:
        """Load a watertight, axis-aligned STL surface from ``surface_file``."""
        path = Path(surface_file).expanduser().resolve()
        triangles, data = _read_stl(path)
        _validate_triangles(triangles, path)
        bounds = _axis_aligned_box_bounds(triangles, path)
        triangles.setflags(write=False)
        return cls(
            path=path,
            triangles=triangles,
            bounds=bounds,
            sha256=hashlib.sha256(data).hexdigest(),
        )


__all__ = ["SurfaceBounds", "TriangulatedSurface"]
