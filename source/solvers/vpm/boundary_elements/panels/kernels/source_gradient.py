"""Double-precision derivative of a constant triangular source velocity.

This CPU operator differentiates the solid-angle and edge-log terms
analytically. Targets must be off the source surface. The derivative follows
J[i,j] = d velocity[i] / d target[j], before any stretching convention.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
import os

from numba import njit
import numpy as np


def triangle_geometry(triangles):
    triangles = np.ascontiguousarray(triangles, dtype=float)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not np.all(np.isfinite(triangles)):
        raise ValueError("Finite three-dimensional triangles are required")
    edges = np.roll(triangles, -1, axis=1) - triangles
    lengths = np.linalg.norm(edges, axis=2)
    vector = np.cross(edges[:, 0], -edges[:, 2])
    double_area = np.linalg.norm(vector, axis=1)
    if np.any(double_area <= 0) or np.any(lengths <= 0):
        raise ValueError("Degenerate triangles are unsupported")
    normals = vector / double_area[:, None]
    tangents = edges / lengths[:, :, None]
    outward = np.cross(tangents, normals[:, None])
    return triangles, normals, tangents, outward, lengths


@njit(cache=True, nogil=True)
def _gradient(points, vertices, normals, outward, lengths, strength):
    result = np.zeros((len(points), 3, 3))
    for query in range(len(points)):
        for panel in range(len(vertices)):
            r = vertices[panel] - points[query]
            distance = np.sqrt(np.sum(r * r, axis=1))
            if np.min(distance) == 0:
                raise ValueError("Source-vertex target is singular")
            norm_gradient = -r / distance[:, None]
            products = np.array([np.dot(r[0], r[1]), np.dot(r[1], r[2]), np.dot(r[2], r[0])])
            determinant = np.dot(r[0], np.cross(r[1], r[2]))
            denominator = (
                distance[0] * distance[1] * distance[2]
                + products[0] * distance[2]
                + products[1] * distance[0]
                + products[2] * distance[1]
            )
            determinant_gradient = -(
                np.cross(r[1], r[2]) + np.cross(r[2], r[0]) + np.cross(r[0], r[1])
            )
            denominator_gradient = np.zeros(3)
            for edge in range(3):
                a, b, c = edge, (edge + 1) % 3, (edge + 2) % 3
                denominator_gradient += (
                    norm_gradient[a] * distance[b] * distance[c]
                    - (r[a] + r[b]) * distance[c]
                    + products[edge] * norm_gradient[c]
                )
            angle_denominator = denominator**2 + determinant**2
            if angle_denominator == 0:
                raise ValueError("Source-edge target is singular")
            angle_gradient = (
                2
                * (denominator * determinant_gradient - determinant * denominator_gradient)
                / angle_denominator
            )
            jacobian = -np.outer(normals[panel], angle_gradient)
            for edge in range(3):
                a, b = edge, (edge + 1) % 3
                radius_sum, length = distance[a] + distance[b], lengths[panel, edge]
                log_denominator = (radius_sum - length) * (radius_sum + length)
                if log_denominator <= 0:
                    raise ValueError("Source-edge target is singular or numerically unresolved")
                log_gradient = -2 * length * (norm_gradient[a] + norm_gradient[b]) / log_denominator
                jacobian += np.outer(outward[panel, edge], log_gradient)
            result[query] += strength[panel] * jacobian / (4 * math.pi)
    return result


def source_panel_gradient(points, vertices, strengths):
    """Return the Jacobian of the given saved source field, without resolving it."""
    points = np.ascontiguousarray(points, dtype=float).reshape(-1, 3)
    strengths = np.ascontiguousarray(strengths, dtype=float)
    vertices, normals, _, outward, lengths = triangle_geometry(vertices)
    if (
        strengths.shape != (len(vertices),)
        or not np.all(np.isfinite(strengths))
        or not np.all(np.isfinite(points))
    ):
        raise ValueError("Finite targets and one strength per source triangle are required")
    workers = min(
        max(1, int(os.environ.get("TI_CPU_MAX_NUM_THREADS", "1"))), max(1, len(points) // 2048)
    )
    if workers == 1:
        result = _gradient(points, vertices, normals, outward, lengths, strengths)
    else:
        # Compile before workers enter Numba. Each target keeps the same panel
        # summation order, so concurrency introduces no floating-point reduction.
        _gradient(points[:0], vertices, normals, outward, lengths, strengths)

        def evaluate(chunk):
            return _gradient(chunk, vertices, normals, outward, lengths, strengths)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            result = np.concatenate(list(pool.map(evaluate, np.array_split(points, workers))))
    if not np.all(np.isfinite(result)):
        raise ValueError("Nonfinite analytical source gradient")
    return result
