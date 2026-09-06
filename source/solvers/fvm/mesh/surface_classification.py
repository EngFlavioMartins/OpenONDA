"""Cell/point classification and surface projection against a general,
watertight triangulated surface.

This is the piece of cfMesh's ``cartesianMesh`` workflow that the axis-aligned
box path in the octree stage never needed: real triangle-geometry
tests against a curved surface, used to classify octree leaves as solid,
fluid, or boundary, and to project boundary-cell corners onto the surface so
the extracted wall patch conforms to the true geometry rather than a
staircase of unmodified Cartesian corners.

The core primitives (triangle/box overlap, closest point on a triangle, ray/
triangle intersection) are the standard computational-geometry formulas
(Akenine-Moller separating-axis triangle/box test; Ericson's closest-point-
on-triangle regions; Moller-Trumbore ray/triangle intersection), not a copy
of any external mesher's source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from numba import njit
import numpy as np

_DEFAULT_RAY_DIRECTIONS = (
    np.array([1.0, 0.0, 0.0]),
    np.array([0.371390, 0.915670, 0.151340]),
    np.array([-0.591100, 0.283900, 0.755100]),
)


def _normalize(direction: np.ndarray) -> np.ndarray:
    return direction / np.linalg.norm(direction)


def closest_point_on_triangles(
    point: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> np.ndarray:
    """Closest point on each triangle ``(a[i], b[i], c[i])`` to ``point``.

    Vectorised region test (Ericson, *Real-Time Collision Detection*, 5.1.5).
    ``a``, ``b``, ``c`` have shape ``(n, 3)``; the result has shape ``(n, 3)``.
    """
    ab = b - a
    ac = c - a
    ap = point - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    mask_a = (d1 <= 0.0) & (d2 <= 0.0)

    bp = point - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    mask_b = (~mask_a) & (d3 >= 0.0) & (d4 <= d3)

    cp = point - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    mask_c = (~mask_a) & (~mask_b) & (d6 >= 0.0) & (d5 <= d6)

    vc = d1 * d4 - d3 * d2
    mask_ab = (~mask_a) & (~mask_b) & (~mask_c) & (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)

    vb = d5 * d2 - d1 * d6
    mask_ac = (
        (~mask_a) & (~mask_b) & (~mask_c) & (~mask_ab) & (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    )

    va = d3 * d6 - d5 * d4
    mask_bc = (
        (~mask_a)
        & (~mask_b)
        & (~mask_c)
        & (~mask_ab)
        & (~mask_ac)
        & (va <= 0.0)
        & ((d4 - d3) >= 0.0)
        & ((d5 - d6) >= 0.0)
    )

    mask_face = ~(mask_a | mask_b | mask_c | mask_ab | mask_ac | mask_bc)

    result = np.empty_like(a)
    result[mask_a] = a[mask_a]
    result[mask_b] = b[mask_b]
    result[mask_c] = c[mask_c]

    if np.any(mask_ab):
        denom = d1[mask_ab] - d3[mask_ab]
        v = d1[mask_ab] / np.where(denom != 0.0, denom, 1.0)
        result[mask_ab] = a[mask_ab] + v[:, None] * ab[mask_ab]
    if np.any(mask_ac):
        denom = d2[mask_ac] - d6[mask_ac]
        w = d2[mask_ac] / np.where(denom != 0.0, denom, 1.0)
        result[mask_ac] = a[mask_ac] + w[:, None] * ac[mask_ac]
    if np.any(mask_bc):
        denom = (d4[mask_bc] - d3[mask_bc]) + (d5[mask_bc] - d6[mask_bc])
        w = (d4[mask_bc] - d3[mask_bc]) / np.where(denom != 0.0, denom, 1.0)
        result[mask_bc] = b[mask_bc] + w[:, None] * (c[mask_bc] - b[mask_bc])
    if np.any(mask_face):
        denom = va[mask_face] + vb[mask_face] + vc[mask_face]
        denom = np.where(denom != 0.0, denom, 1.0)
        v = vb[mask_face] / denom
        w = vc[mask_face] / denom
        result[mask_face] = a[mask_face] + v[:, None] * ab[mask_face] + w[:, None] * ac[mask_face]
    return result


def triangle_box_overlap(
    box_centre: np.ndarray, box_half: np.ndarray, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray
) -> np.ndarray:
    """Separating-axis triangle/AABB overlap test (Akenine-Moller).

    ``v0``, ``v1``, ``v2`` have shape ``(n, 3)``; returns a boolean mask of
    shape ``(n,)``.
    """
    t0 = v0 - box_centre
    t1 = v1 - box_centre
    t2 = v2 - box_centre

    tri_min = np.minimum(np.minimum(t0, t1), t2)
    tri_max = np.maximum(np.maximum(t0, t1), t2)
    overlap = np.all((tri_min <= box_half) & (tri_max >= -box_half), axis=1)

    e0 = t1 - t0
    e1 = t2 - t1
    e2 = t0 - t2
    normal = np.cross(e0, e1)
    dist = np.einsum("ij,ij->i", normal, t0)
    radius = (
        box_half[0] * np.abs(normal[:, 0])
        + box_half[1] * np.abs(normal[:, 1])
        + box_half[2] * np.abs(normal[:, 2])
    )
    overlap &= np.abs(dist) <= np.where(radius > 0.0, radius, np.finfo(np.float64).eps)

    axes_x = np.array([1.0, 0.0, 0.0])
    axes_y = np.array([0.0, 1.0, 0.0])
    axes_z = np.array([0.0, 0.0, 1.0])
    for edge in (e0, e1, e2):
        for base in (axes_x, axes_y, axes_z):
            axis = np.cross(np.broadcast_to(base, edge.shape), edge)
            axis_norm = np.linalg.norm(axis, axis=1)
            valid = axis_norm > 1.0e-14
            p0 = np.einsum("ij,ij->i", t0, axis)
            p1 = np.einsum("ij,ij->i", t1, axis)
            p2 = np.einsum("ij,ij->i", t2, axis)
            proj_min = np.minimum(np.minimum(p0, p1), p2)
            proj_max = np.maximum(np.maximum(p0, p1), p2)
            r = (
                box_half[0] * np.abs(axis[:, 0])
                + box_half[1] * np.abs(axis[:, 1])
                + box_half[2] * np.abs(axis[:, 2])
            )
            separated = valid & ((proj_min > r) | (proj_max < -r))
            overlap &= ~separated
    return overlap


@njit(cache=True, fastmath=False)
def _triangle_box_overlap_any_kernel(
    box_centre: np.ndarray,
    box_half: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
) -> bool:
    """Compiled any-hit form of the triangle/AABB separating-axis test."""
    eps = np.finfo(np.float64).eps
    for index in range(len(v0)):
        t0x = v0[index, 0] - box_centre[0]
        t0y = v0[index, 1] - box_centre[1]
        t0z = v0[index, 2] - box_centre[2]
        t1x = v1[index, 0] - box_centre[0]
        t1y = v1[index, 1] - box_centre[1]
        t1z = v1[index, 2] - box_centre[2]
        t2x = v2[index, 0] - box_centre[0]
        t2y = v2[index, 1] - box_centre[1]
        t2z = v2[index, 2] - box_centre[2]
        if (
            max(t0x, t1x, t2x) < -box_half[0]
            or min(t0x, t1x, t2x) > box_half[0]
            or max(t0y, t1y, t2y) < -box_half[1]
            or min(t0y, t1y, t2y) > box_half[1]
            or max(t0z, t1z, t2z) < -box_half[2]
            or min(t0z, t1z, t2z) > box_half[2]
        ):
            continue
        e0x = t1x - t0x
        e0y = t1y - t0y
        e0z = t1z - t0z
        e1x = t2x - t1x
        e1y = t2y - t1y
        e1z = t2z - t1z
        e2x = t0x - t2x
        e2y = t0y - t2y
        e2z = t0z - t2z
        nx = e0y * e1z - e0z * e1y
        ny = e0z * e1x - e0x * e1z
        nz = e0x * e1y - e0y * e1x
        if abs(nx * t0x + ny * t0y + nz * t0z) > (
            box_half[0] * abs(nx) + box_half[1] * abs(ny) + box_half[2] * abs(nz)
        ):
            continue
        for edge_id in range(3):
            if edge_id == 0:
                ex, ey, ez = e0x, e0y, e0z
            elif edge_id == 1:
                ex, ey, ez = e1x, e1y, e1z
            else:
                ex, ey, ez = e2x, e2y, e2z
            # The three cross products with the Cartesian box axes are
            # (0,-ez,ey), (ez,0,-ex), and (-ey,ex,0).
            for axis_id in range(3):
                if axis_id == 0:
                    ax, ay, az = 0.0, -ez, ey
                elif axis_id == 1:
                    ax, ay, az = ez, 0.0, -ex
                else:
                    ax, ay, az = -ey, ex, 0.0
                if ax * ax + ay * ay + az * az <= 1.0e-28:
                    continue
                p0 = t0x * ax + t0y * ay + t0z * az
                p1 = t1x * ax + t1y * ay + t1z * az
                p2 = t2x * ax + t2y * ay + t2z * az
                radius = box_half[0] * abs(ax) + box_half[1] * abs(ay) + box_half[2] * abs(az)
                if max(p0, p1, p2) < -radius - eps or min(p0, p1, p2) > radius + eps:
                    break
            else:
                continue
            break
        else:
            return True
    return False


def _ray_triangle_intersections(
    origins: np.ndarray, direction: np.ndarray, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Moller-Trumbore ray/triangle intersections for many origins at once.

    ``origins`` has shape ``(m, 3)``, ``v0``/``v1``/``v2`` have shape
    ``(n, 3)``.  Returns ``(hit, degenerate)`` boolean masks of shape
    ``(m, n)``: ``hit`` marks a forward intersection with ``t > eps``, and
    ``degenerate`` marks a hit so close to a triangle edge/vertex that ray
    parity is not trustworthy (the caller should retry with another ray).
    """
    eps = 1.0e-9
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(direction, edge2)
    a = np.einsum("ij,ij->i", edge1, h)
    parallel = np.abs(a) < eps
    f = np.where(parallel, 0.0, 1.0 / np.where(parallel, 1.0, a))

    s = origins[:, None, :] - v0[None, :, :]
    u = f[None, :] * np.einsum("mnj,nj->mn", s, h)
    q = np.cross(s, edge1[None, :, :])
    v = f[None, :] * np.einsum("j,mnj->mn", direction, q)
    t = f[None, :] * np.einsum("nj,mnj->mn", edge2, q)

    valid = (
        (~parallel)[None, :]
        & (u >= -eps)
        & (u <= 1.0 + eps)
        & (v >= -eps)
        & (u + v <= 1.0 + eps)
        & (t > eps)
    )
    margin = 1.0e-6
    degenerate = valid & (
        (np.abs(u) < margin) | (np.abs(v) < margin) | (np.abs(u + v - 1.0) < margin)
    )
    return valid, degenerate


@dataclass
class SurfaceIndex:
    """Broad-phase triangle grid plus exact geometric tests for one surface."""

    triangles: np.ndarray
    cell_size: float
    grid_origin: np.ndarray
    grid: dict[tuple[int, int, int], np.ndarray] = field(repr=False)
    triangle_min: np.ndarray = field(init=False, repr=False)
    triangle_max: np.ndarray = field(init=False, repr=False)
    surface_lower: np.ndarray = field(init=False, repr=False)
    surface_upper: np.ndarray = field(init=False, repr=False)

    @classmethod
    def build(cls, triangles: np.ndarray) -> SurfaceIndex:
        triangles = np.ascontiguousarray(triangles, dtype=np.float64)
        n = len(triangles)
        v0, v1, v2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        edge_lengths = np.column_stack(
            (
                np.linalg.norm(v1 - v0, axis=1),
                np.linalg.norm(v2 - v1, axis=1),
                np.linalg.norm(v0 - v2, axis=1),
            )
        )
        # Use a local surface scale, not the median of all edge lengths.  A
        # long, thin STL triangle (for example a spanwise cylinder facet) can
        # have one edge hundreds of times longer than its circumferential
        # edges; using that long edge makes every point query scan most of the
        # surface and defeats the spatial index.
        cell_size = float(np.median(np.min(edge_lengths, axis=1))) * 2.0
        if not np.isfinite(cell_size) or cell_size <= 0.0:
            span = float(np.max(triangles.max(axis=(0, 1)) - triangles.min(axis=(0, 1))))
            cell_size = max(span / 8.0, 1.0e-9)

        tri_min = triangles.min(axis=1)
        tri_max = triangles.max(axis=1)
        grid_origin = tri_min.min(axis=0)
        lo = np.floor((tri_min - grid_origin) / cell_size).astype(np.int64)
        hi = np.floor((tri_max - grid_origin) / cell_size).astype(np.int64)

        buckets: dict[tuple[int, int, int], list[int]] = {}
        for i in range(n):
            for ix in range(int(lo[i, 0]), int(hi[i, 0]) + 1):
                for iy in range(int(lo[i, 1]), int(hi[i, 1]) + 1):
                    for iz in range(int(lo[i, 2]), int(hi[i, 2]) + 1):
                        buckets.setdefault((ix, iy, iz), []).append(i)
        grid = {key: np.asarray(value, dtype=np.int64) for key, value in buckets.items()}
        result = cls(triangles=triangles, cell_size=cell_size, grid_origin=grid_origin, grid=grid)
        result.triangle_min = tri_min
        result.triangle_max = tri_max
        result.surface_lower = tri_min.min(axis=0)
        result.surface_upper = tri_max.max(axis=0)
        return result

    def candidate_triangles(self, box_min: np.ndarray, box_max: np.ndarray) -> np.ndarray:
        """Triangle indices whose grid cells overlap ``[box_min, box_max]``."""
        lo = np.floor((box_min - self.grid_origin) / self.cell_size).astype(np.int64)
        hi = np.floor((box_max - self.grid_origin) / self.cell_size).astype(np.int64)
        cell_count = int(np.prod(hi - lo + 1, dtype=np.int64))
        # Large ancestor boxes can span a huge number of empty spatial-grid
        # buckets.  Scanning those bucket coordinates is slower than testing
        # the compact triangle AABB array directly; both paths are broad
        # phases and the exact triangle test remains the caller's job.
        if cell_count > max(4096, 4 * len(self.triangles)):
            return np.arange(len(self.triangles), dtype=np.int64)
        found: set[int] = set()
        for ix in range(int(lo[0]), int(hi[0]) + 1):
            for iy in range(int(lo[1]), int(hi[1]) + 1):
                for iz in range(int(lo[2]), int(hi[2]) + 1):
                    bucket = self.grid.get((ix, iy, iz))
                    if bucket is not None:
                        found.update(bucket.tolist())
        if not found:
            return np.empty(0, dtype=np.int64)
        return np.asarray(sorted(found), dtype=np.int64)

    def box_intersects_surface(self, box_min: np.ndarray, box_max: np.ndarray) -> bool:
        """True if any triangle has positive-area overlap with the box."""
        if np.any(box_max < self.surface_lower) or np.any(box_min > self.surface_upper):
            return False
        candidates = self.candidate_triangles(box_min, box_max)
        if candidates.size == 0:
            return False
        # The spatial grid is deliberately coarse enough to keep indexing
        # cheap.  Reject candidate triangles whose exact AABBs do not touch
        # the query before allocating the separating-axis arrays.  This
        # broad-phase filter is especially important during the production
        # cylinder template walk, which visits every octree ancestor.
        candidates = candidates[
            np.all(self.triangle_min[candidates] <= box_max, axis=1)
            & np.all(self.triangle_max[candidates] >= box_min, axis=1)
        ]
        if candidates.size == 0:
            return False
        centre = 0.5 * (box_min + box_max)
        half = 0.5 * (box_max - box_min)
        v0 = self.triangles[candidates, 0]
        v1 = self.triangles[candidates, 1]
        v2 = self.triangles[candidates, 2]
        return _triangle_box_overlap_any_kernel(centre, half, v0, v1, v2)

    def is_inside(self, points: np.ndarray, *, chunk_size: int = 500) -> np.ndarray:
        """Point-in-closed-manifold test by ray-parity, with degenerate retry."""
        points = np.atleast_2d(points)
        n = len(points)
        result = np.zeros(n, dtype=bool)
        resolved = np.zeros(n, dtype=bool)
        v0, v1, v2 = self.triangles[:, 0], self.triangles[:, 1], self.triangles[:, 2]
        if n == 1:
            # Template extraction asks about one cell centre at a time.  The
            # default +x ray only needs triangles whose y/z AABBs contain the
            # point and whose x extent is ahead of it; testing all triangles
            # for every scalar centre otherwise dominates large cylinder
            # builds.  Degenerate hits fall through to the general retry loop.
            point = points[0]
            candidate = np.flatnonzero(
                (self.triangle_min[:, 1] <= point[1])
                & (self.triangle_max[:, 1] >= point[1])
                & (self.triangle_min[:, 2] <= point[2])
                & (self.triangle_max[:, 2] >= point[2])
                & (self.triangle_max[:, 0] >= point[0])
            )
            direction = _normalize(_DEFAULT_RAY_DIRECTIONS[0])
            if not candidate.size:
                return result
            if candidate.size:
                valid, degenerate = _ray_triangle_intersections(
                    points, direction, v0[candidate], v1[candidate], v2[candidate]
                )
                if not np.any(degenerate):
                    result[0] = bool(np.count_nonzero(valid[0]) % 2)
                    return result
        # Ray testing materialises several ``(points, triangles)`` arrays.
        # Keep that product bounded when layer preparation has subdivided long
        # source facets; otherwise a benign surface refinement can exhaust RAM.
        chunk_size = min(chunk_size, max(16, 2_000_000 // max(len(self.triangles), 1)))
        for raw_direction in _DEFAULT_RAY_DIRECTIONS:
            direction = _normalize(raw_direction)
            pending = np.flatnonzero(~resolved)
            if pending.size == 0:
                break
            for start in range(0, pending.size, chunk_size):
                idx = pending[start : start + chunk_size]
                valid, degenerate = _ray_triangle_intersections(points[idx], direction, v0, v1, v2)
                clean = ~np.any(degenerate, axis=1)
                if not np.any(clean):
                    continue
                counts = np.count_nonzero(valid[clean], axis=1)
                target = idx[clean]
                result[target] = (counts % 2) == 1
                resolved[target] = True
        if not np.all(resolved):
            unresolved = int(np.count_nonzero(~resolved))
            raise RuntimeError(
                f"{unresolved} point(s) hit a degenerate ray/triangle intersection on every "
                "trial direction; the surface tessellation may be malformed near those points"
            )
        return result

    def nearest_point(self, point: np.ndarray) -> tuple[np.ndarray, float]:
        """Closest point on the surface using expanding indexed candidates."""
        point = np.asarray(point, dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("point must contain three finite coordinates")
        v0, v1, v2 = self.triangles[:, 0], self.triangles[:, 1], self.triangles[:, 2]
        span = float(np.max(self.triangles.max(axis=(0, 1)) - self.triangles.min(axis=(0, 1))))
        radius = max(self.cell_size, np.finfo(np.float64).eps * max(span, 1.0))
        for _ in range(32):
            candidate_ids = self.candidate_triangles(point - radius, point + radius)
            if candidate_ids.size:
                candidates = closest_point_on_triangles(
                    point, v0[candidate_ids], v1[candidate_ids], v2[candidate_ids]
                )
                distances = np.linalg.norm(candidates - point, axis=1)
                best = int(np.argmin(distances))
                distance = float(distances[best])
                if radius >= distance:
                    return candidates[best], distance
            radius *= 2.0
            if radius > 4.0 * max(span, self.cell_size):
                break

        candidates = closest_point_on_triangles(point, v0, v1, v2)
        distances = np.linalg.norm(candidates - point, axis=1)
        best = int(np.argmin(distances))
        return candidates[best], float(distances[best])

    def nearest_points(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Closest surface points, distances, and triangle ids for a batch.

        VTK's static cell locator performs the same triangle projection in
        compiled code and is orders of magnitude faster for the thousands of
        wall vertices in a production mesh.  Keep the scalar indexed routine
        as a fallback for minimal installations.
        """
        values = np.atleast_2d(np.asarray(points, dtype=np.float64))
        if values.shape[1] != 3 or not np.all(np.isfinite(values)):
            raise ValueError("points must have shape (n, 3) and contain finite coordinates")
        try:
            import pyvista as pv
        except ImportError:  # pragma: no cover - VTK is part of the FVM extra.
            projected = [self.nearest_point(point) for point in values]
            nearest = np.asarray([item[0] for item in projected], dtype=np.float64)
            distances = np.asarray([item[1] for item in projected], dtype=np.float64)
            return nearest, distances, np.full(len(values), -1, dtype=np.int64)

        locator_mesh = getattr(self, "_locator_mesh", None)
        if locator_mesh is None:
            vertices = np.ascontiguousarray(self.triangles.reshape(-1, 3))
            connectivity = np.column_stack(
                (
                    np.full(len(self.triangles), 3, dtype=np.int64),
                    np.arange(len(vertices), dtype=np.int64).reshape(-1, 3),
                )
            ).ravel()
            locator_mesh = pv.PolyData(
                vertices,
                connectivity,  # pyrefly: ignore[bad-argument-type]
            )
            self._locator_mesh = locator_mesh
        located: Any = locator_mesh.find_closest_cell(values, return_closest_point=True)
        cell_ids, nearest = located
        nearest = np.asarray(nearest, dtype=np.float64)
        distances = np.linalg.norm(nearest - values, axis=1)
        return nearest, distances, np.asarray(cell_ids, dtype=np.int64)

    def classify(self, points: np.ndarray, *, tolerance: float | None = None) -> np.ndarray:
        """Classify points as outside ``0``, inside ``1``, or on surface ``-1``."""
        values = np.atleast_2d(np.asarray(points, dtype=np.float64))
        if values.shape[1] != 3 or not np.all(np.isfinite(values)):
            raise ValueError("points must have shape (n, 3) and contain finite coordinates")
        scale = max(float(np.ptp(self.triangles, axis=(0, 1)).max()), 1.0)
        surface_tolerance = tolerance if tolerance is not None else 1.0e-10 * scale
        if not np.isfinite(surface_tolerance) or surface_tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive")
        result = np.where(self.is_inside(values), 1, 0).astype(np.int8)
        for index, point in enumerate(values):
            _nearest, distance = self.nearest_point(point)
            if distance <= surface_tolerance:
                result[index] = -1
        return result


__all__ = [
    "SurfaceIndex",
    "closest_point_on_triangles",
    "triangle_box_overlap",
]
