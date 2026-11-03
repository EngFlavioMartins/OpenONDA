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

import numpy as np

from openonda.core._numba import cacheable_njit as njit

# Larger triangles are tested by every query instead of being bucketed
_MAX_TRIANGLE_BUCKETS = 4096

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
    return _triangle_box_overlap_mask_kernel(box_centre, box_half, v0, v1, v2)


@njit(cache=True, fastmath=False)
def _triangle_box_overlap_mask_kernel(
    box_centre: np.ndarray,
    box_half: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
) -> np.ndarray:
    """Compiled mask form retaining the public separating-axis tolerances."""
    result = np.zeros(len(v0), dtype=np.bool_)
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
        radius = box_half[0] * abs(nx) + box_half[1] * abs(ny) + box_half[2] * abs(nz)
        if abs(nx * t0x + ny * t0y + nz * t0z) > (
            radius if radius > 0.0 else np.finfo(np.float64).eps
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
                if np.sqrt(ax * ax + ay * ay + az * az) <= 1.0e-14:
                    continue
                p0 = t0x * ax + t0y * ay + t0z * az
                p1 = t1x * ax + t1y * ay + t1z * az
                p2 = t2x * ax + t2y * ay + t2z * az
                radius = box_half[0] * abs(ax) + box_half[1] * abs(ay) + box_half[2] * abs(az)
                if max(p0, p1, p2) < -radius or min(p0, p1, p2) > radius:
                    break
            else:
                continue
            break
        else:
            result[index] = True
    return result


@njit(cache=True, fastmath=False)
def _triangle_overlaps_box(
    box_centre: np.ndarray,
    box_half: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
) -> bool:
    """Separating-axis test of one triangle against a box, with any-hit tolerances."""
    eps = np.finfo(np.float64).eps
    t0x = v0[0] - box_centre[0]
    t0y = v0[1] - box_centre[1]
    t0z = v0[2] - box_centre[2]
    t1x = v1[0] - box_centre[0]
    t1y = v1[1] - box_centre[1]
    t1z = v1[2] - box_centre[2]
    t2x = v2[0] - box_centre[0]
    t2y = v2[1] - box_centre[1]
    t2z = v2[2] - box_centre[2]
    if (
        max(t0x, t1x, t2x) < -box_half[0]
        or min(t0x, t1x, t2x) > box_half[0]
        or max(t0y, t1y, t2y) < -box_half[1]
        or min(t0y, t1y, t2y) > box_half[1]
        or max(t0z, t1z, t2z) < -box_half[2]
        or min(t0z, t1z, t2z) > box_half[2]
    ):
        return False
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
        return False
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
                return False
    return True


@njit(cache=True, fastmath=False)
def _triangle_meets_box(
    triangles: np.ndarray,
    triangle_min: np.ndarray,
    triangle_max: np.ndarray,
    triangle: int,
    box_min: np.ndarray,
    box_max: np.ndarray,
    box_centre: np.ndarray,
    box_half: np.ndarray,
) -> bool:
    """Exact triangle bounds filter, then the separating-axis test."""
    for axis in range(3):
        if (
            triangle_min[triangle, axis] > box_max[axis]
            or triangle_max[triangle, axis] < box_min[axis]
        ):
            return False
    return _triangle_overlaps_box(
        box_centre, box_half, triangles[triangle, 0], triangles[triangle, 1], triangles[triangle, 2]
    )


@njit(cache=True, fastmath=False)
def _boxes_meet_surface_kernel(
    box_min: np.ndarray,
    box_max: np.ndarray,
    triangles: np.ndarray,
    triangle_min: np.ndarray,
    triangle_max: np.ndarray,
    surface_lower: np.ndarray,
    surface_upper: np.ndarray,
    grid_origin: np.ndarray,
    cell_size: float,
    grid_shape: np.ndarray,
    grid_keys: np.ndarray,
    grid_offsets: np.ndarray,
    grid_triangles: np.ndarray,
    oversized: np.ndarray,
) -> np.ndarray:
    """Positive-area overlap of each box with any triangle.

    The bucket grid only prunes: every triangle whose bounds overlap a box lies
    in one of its buckets (bucket indices are monotone in the coordinates), so
    the result equals testing every triangle whose bounds touch the box.
    """
    result = np.zeros(len(box_min), dtype=np.bool_)
    lower_bucket = np.empty(3, dtype=np.int64)
    upper_bucket = np.empty(3, dtype=np.int64)
    for box in range(len(box_min)):
        lower = box_min[box]
        upper = box_max[box]
        outside = False
        for axis in range(3):
            if upper[axis] < surface_lower[axis] or lower[axis] > surface_upper[axis]:
                outside = True
        if outside:
            continue
        centre = 0.5 * (lower + upper)
        half = 0.5 * (upper - lower)
        count = 1
        for axis in range(3):
            lower_bucket[axis] = np.int64(np.floor((lower[axis] - grid_origin[axis]) / cell_size))
            upper_bucket[axis] = np.int64(np.floor((upper[axis] - grid_origin[axis]) / cell_size))
            count *= upper_bucket[axis] - lower_bucket[axis] + 1
        if count > max(4096, 4 * len(triangles)):
            for triangle in range(len(triangles)):
                if _triangle_meets_box(
                    triangles, triangle_min, triangle_max, triangle, lower, upper, centre, half
                ):
                    result[box] = True
                    break
            continue
        for triangle in oversized:
            if _triangle_meets_box(
                triangles, triangle_min, triangle_max, triangle, lower, upper, centre, half
            ):
                result[box] = True
                break
        if result[box]:
            continue
        for iz in range(max(lower_bucket[2], 0), min(upper_bucket[2], grid_shape[2] - 1) + 1):
            for iy in range(max(lower_bucket[1], 0), min(upper_bucket[1], grid_shape[1] - 1) + 1):
                for ix in range(
                    max(lower_bucket[0], 0), min(upper_bucket[0], grid_shape[0] - 1) + 1
                ):
                    key = ix + grid_shape[0] * (iy + grid_shape[1] * iz)
                    position = np.searchsorted(grid_keys, key)
                    if position == len(grid_keys) or grid_keys[position] != key:
                        continue
                    for entry in range(grid_offsets[position], grid_offsets[position + 1]):
                        if _triangle_meets_box(
                            triangles,
                            triangle_min,
                            triangle_max,
                            grid_triangles[entry],
                            lower,
                            upper,
                            centre,
                            half,
                        ):
                            result[box] = True
                            break
                    if result[box]:
                        break
                if result[box]:
                    break
            if result[box]:
                break
    return result


@njit(cache=True, fastmath=False)
def _ray_triangle_data(
    triangles: np.ndarray, directions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-triangle Moller-Trumbore terms that do not depend on the ray origin.

    Returns the edges ``v1 - v0`` and ``v2 - v0``, and per direction the
    vector ``direction x edge2``, the reciprocal of ``edge1 . h`` and whether
    the ray is parallel to the triangle. Three-term dot products are summed as
    NumPy's ``einsum`` sums them, (0 + 2) + 1, and cross products follow
    ``numpy.cross``, so every hit test repeats the former vectorised arithmetic.
    """
    n_triangles = len(triangles)
    edge1 = np.empty((n_triangles, 3), dtype=np.float64)
    edge2 = np.empty((n_triangles, 3), dtype=np.float64)
    h = np.empty((len(directions), n_triangles, 3), dtype=np.float64)
    inverse = np.zeros((len(directions), n_triangles), dtype=np.float64)
    parallel = np.zeros((len(directions), n_triangles), dtype=np.bool_)
    for triangle in range(n_triangles):
        for axis in range(3):
            edge1[triangle, axis] = triangles[triangle, 1, axis] - triangles[triangle, 0, axis]
            edge2[triangle, axis] = triangles[triangle, 2, axis] - triangles[triangle, 0, axis]
        e1 = edge1[triangle]
        e2 = edge2[triangle]
        for direction_id in range(len(directions)):
            d = directions[direction_id]
            h0 = d[1] * e2[2] - d[2] * e2[1]
            h1 = d[2] * e2[0] - d[0] * e2[2]
            h2 = d[0] * e2[1] - d[1] * e2[0]
            h[direction_id, triangle, 0] = h0
            h[direction_id, triangle, 1] = h1
            h[direction_id, triangle, 2] = h2
            a = (e1[0] * h0 + e1[2] * h2) + e1[1] * h1
            if abs(a) < 1.0e-9:
                parallel[direction_id, triangle] = True
            else:
                inverse[direction_id, triangle] = 1.0 / a
    return edge1, edge2, h, inverse, parallel


@njit(cache=True, fastmath=False)
def _ray_triangle_hit(
    point: np.ndarray,
    direction: np.ndarray,
    v0: np.ndarray,
    edge1: np.ndarray,
    edge2: np.ndarray,
    h: np.ndarray,
    f: float,
) -> int:
    """Moller-Trumbore hit of one ray: 0 miss, 1 hit, 2 hit too near an edge or vertex.

    A degenerate hit makes ray parity untrustworthy; the caller retries with
    another ray. ``f`` is the reciprocal from :func:`_ray_triangle_data` of a
    triangle the ray is not parallel to.
    """
    eps = 1.0e-9
    sx = point[0] - v0[0]
    sy = point[1] - v0[1]
    sz = point[2] - v0[2]
    u = f * ((sx * h[0] + sz * h[2]) + sy * h[1])
    if not (u >= -eps and u <= 1.0 + eps):
        return 0
    q0 = sy * edge1[2] - sz * edge1[1]
    q1 = sz * edge1[0] - sx * edge1[2]
    q2 = sx * edge1[1] - sy * edge1[0]
    v = f * ((direction[0] * q0 + direction[2] * q2) + direction[1] * q1)
    if not (v >= -eps and u + v <= 1.0 + eps):
        return 0
    t = f * ((edge2[0] * q0 + edge2[2] * q2) + edge2[1] * q1)
    if not t > eps:
        return 0
    margin = 1.0e-6
    if abs(u) < margin or abs(v) < margin or abs(u + v - 1.0) < margin:
        return 2
    return 1


@njit(cache=True, fastmath=False)
def _single_point_parity_kernel(
    points: np.ndarray,
    triangles: np.ndarray,
    triangle_min: np.ndarray,
    triangle_max: np.ndarray,
    surface_lower: np.ndarray,
    surface_upper: np.ndarray,
    direction: np.ndarray,
    edge1: np.ndarray,
    edge2: np.ndarray,
    h: np.ndarray,
    inverse: np.ndarray,
    parallel: np.ndarray,
) -> np.ndarray:
    """Inside ``1``, outside ``0``, or undecided ``-1`` by one ray per point.

    Outside the surface bounds a point is outside. Otherwise only triangles
    whose bounds contain the point's y-z position ahead of it are tested, the
    triangles a +x ray can reach.
    """
    result = np.zeros(len(points), dtype=np.int8)
    for index in range(len(points)):
        point = points[index]
        outside = False
        for axis in range(3):
            if point[axis] < surface_lower[axis] or point[axis] > surface_upper[axis]:
                outside = True
        if outside:
            continue
        hits = 0
        for triangle in range(len(triangles)):
            if not (
                triangle_min[triangle, 1] <= point[1]
                and triangle_max[triangle, 1] >= point[1]
                and triangle_min[triangle, 2] <= point[2]
                and triangle_max[triangle, 2] >= point[2]
                and triangle_max[triangle, 0] >= point[0]
            ):
                continue
            if parallel[triangle]:
                continue
            hit = _ray_triangle_hit(
                point,
                direction,
                triangles[triangle, 0],
                edge1[triangle],
                edge2[triangle],
                h[triangle],
                inverse[triangle],
            )
            if hit == 2:
                hits = -1
                break
            hits += hit
        result[index] = -1 if hits < 0 else hits % 2
    return result


@njit(cache=True, fastmath=False)
def _ray_parity_kernel(
    points: np.ndarray,
    directions: np.ndarray,
    triangles: np.ndarray,
    edge1: np.ndarray,
    edge2: np.ndarray,
    h: np.ndarray,
    inverse: np.ndarray,
    parallel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Ray parity against every triangle, retrying the next direction on a degenerate hit."""
    inside = np.zeros(len(points), dtype=np.bool_)
    resolved = np.zeros(len(points), dtype=np.bool_)
    for index in range(len(points)):
        point = points[index]
        for direction_id in range(len(directions)):
            direction = directions[direction_id]
            hits = 0
            degenerate = False
            for triangle in range(len(triangles)):
                if parallel[direction_id, triangle]:
                    continue
                hit = _ray_triangle_hit(
                    point,
                    direction,
                    triangles[triangle, 0],
                    edge1[triangle],
                    edge2[triangle],
                    h[direction_id, triangle],
                    inverse[direction_id, triangle],
                )
                if hit == 2:
                    degenerate = True
                    break
                hits += hit
            if not degenerate:
                inside[index] = hits % 2 == 1
                resolved[index] = True
                break
    return inside, resolved


@dataclass
class SurfaceIndex:
    """Broad-phase triangle grid plus exact geometric tests for one surface."""

    triangles: np.ndarray
    cell_size: float
    grid_origin: np.ndarray
    grid: dict[tuple[int, int, int], np.ndarray] = field(repr=False)
    oversized_triangles: np.ndarray = field(init=False, repr=False)
    triangle_min: np.ndarray = field(init=False, repr=False)
    triangle_max: np.ndarray = field(init=False, repr=False)
    surface_lower: np.ndarray = field(init=False, repr=False)
    surface_upper: np.ndarray = field(init=False, repr=False)
    # The bucket grid packed for compiled queries: sorted linear bucket keys
    # and, per key, an offset range into the bucket triangle ids.
    grid_shape: np.ndarray = field(init=False, repr=False)
    grid_keys: np.ndarray = field(init=False, repr=False)
    grid_offsets: np.ndarray = field(init=False, repr=False)
    grid_triangles: np.ndarray = field(init=False, repr=False)
    # Origin-independent Moller-Trumbore terms per ray direction and triangle
    ray_directions: np.ndarray = field(init=False, repr=False)
    ray_terms: tuple[np.ndarray, ...] = field(init=False, repr=False)
    _locator_mesh: object = field(default=None, init=False, repr=False)

    @classmethod
    def build(cls, triangles: np.ndarray) -> SurfaceIndex:
        """Build broad-phase lookup data for exact surface queries.

        Parameters
        ----------
        triangles : ndarray, shape (T, 3, 3)
            Cartesian triangle vertices in metres. Vertex order is retained
            because it defines surface winding and normals.

        Returns
        -------
        SurfaceIndex
            Per-triangle bounds and a spatial grid used by overlap,
            closest-point, and inside/outside operations. Triangle data are
            stored as a contiguous float64 copy.

        Raises
        ------
        ValueError
            If the input is empty or cannot be represented as triangular
            ``(T, 3, 3)`` geometry.

        Notes
        -----
        The index accelerates queries; it does not repair winding, degenerate
        triangles, or non-manifold topology.
        """
        triangles = np.ascontiguousarray(triangles, dtype=np.float64)
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

        # A triangle far larger than the local scale, such as a domain face
        # next to a finely faceted body, would fill millions of buckets. Every
        # query tests those few triangles directly instead.
        oversized = np.prod(hi - lo + 1, axis=1) > _MAX_TRIANGLE_BUCKETS
        buckets: dict[tuple[int, int, int], list[int]] = {}
        for i in np.flatnonzero(~oversized):
            for ix in range(int(lo[i, 0]), int(hi[i, 0]) + 1):
                for iy in range(int(lo[i, 1]), int(hi[i, 1]) + 1):
                    for iz in range(int(lo[i, 2]), int(hi[i, 2]) + 1):
                        buckets.setdefault((ix, iy, iz), []).append(int(i))
        grid = {key: np.asarray(value, dtype=np.int64) for key, value in buckets.items()}
        result = cls(triangles=triangles, cell_size=cell_size, grid_origin=grid_origin, grid=grid)
        result.oversized_triangles = np.flatnonzero(oversized)
        result.triangle_min = tri_min
        result.triangle_max = tri_max
        result.surface_lower = tri_min.min(axis=0)
        result.surface_upper = tri_max.max(axis=0)
        shape = hi[~oversized].max(axis=0, initial=0) + 1
        keys = np.asarray(
            [ix + shape[0] * (iy + shape[1] * iz) for ix, iy, iz in grid], dtype=np.int64
        )
        order = np.argsort(keys)
        members = [grid[key] for key in grid]
        result.grid_shape = shape.astype(np.int64)
        result.grid_keys = keys[order]
        result.grid_offsets = np.concatenate(
            ([0], np.cumsum([len(members[i]) for i in order], dtype=np.int64))
        ).astype(np.int64)
        result.grid_triangles = (
            np.concatenate([members[i] for i in order]).astype(np.int64)
            if len(members)
            else np.empty(0, dtype=np.int64)
        )
        result.oversized_triangles = result.oversized_triangles.astype(np.int64)
        result.ray_directions = np.asarray(
            [_normalize(direction) for direction in _DEFAULT_RAY_DIRECTIONS]
        )
        result.ray_terms = _ray_triangle_data(triangles, result.ray_directions)
        return result

    def boxes_intersect_surface(self, box_min: np.ndarray, box_max: np.ndarray) -> np.ndarray:
        """Whether each box ``[box_min[i], box_max[i]]`` has positive-area overlap with a triangle."""
        return _boxes_meet_surface_kernel(
            np.ascontiguousarray(box_min, dtype=np.float64).reshape(-1, 3),
            np.ascontiguousarray(box_max, dtype=np.float64).reshape(-1, 3),
            self.triangles,
            self.triangle_min,
            self.triangle_max,
            self.surface_lower,
            self.surface_upper,
            self.grid_origin,
            float(self.cell_size),
            self.grid_shape,
            self.grid_keys,
            self.grid_offsets,
            self.grid_triangles,
            self.oversized_triangles,
        )

    def single_ray_parity(self, points: np.ndarray) -> np.ndarray:
        """Inside ``1``, outside ``0`` or undecided ``-1`` from one +x ray per point.

        This is the first trial of :meth:`is_inside` for one point; an
        undecided point needs its full multi-direction retry.
        """
        edge1, edge2, h, inverse, parallel = self.ray_terms
        return _single_point_parity_kernel(
            np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3),
            self.triangles,
            self.triangle_min,
            self.triangle_max,
            self.surface_lower,
            self.surface_upper,
            self.ray_directions[0],
            edge1,
            edge2,
            h[0],
            inverse[0],
            parallel[0],
        )

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
        found: set[int] = set(self.oversized_triangles.tolist())
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
        for axis in range(3):
            if box_max[axis] < self.surface_lower[axis] or box_min[axis] > self.surface_upper[axis]:
                return False
        return bool(self.boxes_intersect_surface(box_min, box_max)[0])

    def is_inside(self, points: np.ndarray) -> np.ndarray:
        """Point-in-closed-manifold test by ray-parity, with degenerate retry."""
        points = np.ascontiguousarray(np.atleast_2d(points), dtype=np.float64)
        if len(points) == 1:
            # Template extraction asks about one cell centre at a time.  The
            # default +x ray only needs triangles whose y/z AABBs contain the
            # point and whose x extent is ahead of it.  Degenerate hits fall
            # through to the general retry over every triangle.
            parity = int(self.single_ray_parity(points)[0])
            if parity >= 0:
                return np.asarray([parity == 1])
        result, resolved = _ray_parity_kernel(
            points, self.ray_directions, self.triangles, *self.ray_terms
        )
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
        span = float(np.max(self.surface_upper - self.surface_lower))
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

        locator_mesh = self._locator_mesh
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
