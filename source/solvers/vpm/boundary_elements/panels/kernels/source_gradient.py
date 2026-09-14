"""Double-precision derivative of a constant triangular source velocity.

This CPU operator differentiates the solid-angle and edge-log terms
analytically. Targets must be off the source surface. The derivative follows
J[i,j] = d velocity[i] / d target[j], before any stretching convention.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math

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
    r = np.empty((3, 3))
    distance = np.empty(3)
    products = np.empty(3)
    log_factor = np.empty(3)
    for query in range(len(points)):
        for panel in range(len(vertices)):
            for vertex in range(3):
                squared = 0.0
                for axis in range(3):
                    value = vertices[panel, vertex, axis] - points[query, axis]
                    r[vertex, axis] = value
                    squared += value * value
                distance[vertex] = math.sqrt(squared)
                if distance[vertex] == 0:
                    raise ValueError("Source-vertex target is singular")
            determinant = 0.0
            for axis in range(3):
                a, b = (axis + 1) % 3, (axis + 2) % 3
                determinant += r[0, axis] * (r[1, a] * r[2, b] - r[1, b] * r[2, a])
            for edge in range(3):
                following = (edge + 1) % 3
                products[edge] = 0.0
                for axis in range(3):
                    products[edge] += r[edge, axis] * r[following, axis]
                radius_sum = distance[edge] + distance[following]
                length = lengths[panel, edge]
                log_denominator = (radius_sum - length) * (radius_sum + length)
                if log_denominator <= 0:
                    raise ValueError("Source-edge target is singular or numerically unresolved")
                log_factor[edge] = -2 * length / log_denominator
            denominator = (
                distance[0] * distance[1] * distance[2]
                + products[0] * distance[2]
                + products[1] * distance[0]
                + products[2] * distance[1]
            )
            angle_denominator = denominator**2 + determinant**2
            if angle_denominator == 0:
                raise ValueError("Source-edge target is singular")
            for axis in range(3):
                a, b = (axis + 1) % 3, (axis + 2) % 3
                determinant_gradient = 0.0
                denominator_gradient = 0.0
                for edge in range(3):
                    v, w, u = edge, (edge + 1) % 3, (edge + 2) % 3
                    determinant_gradient -= r[v, a] * r[w, b] - r[v, b] * r[w, a]
                    denominator_gradient += (
                        -r[v, axis] / distance[v] * distance[w] * distance[u]
                        - (r[v, axis] + r[w, axis]) * distance[u]
                        - products[edge] * r[u, axis] / distance[u]
                    )
                angle_gradient = (
                    2
                    * (denominator * determinant_gradient - determinant * denominator_gradient)
                    / angle_denominator
                )
                for component in range(3):
                    jacobian = -normals[panel, component] * angle_gradient
                    for edge in range(3):
                        following = (edge + 1) % 3
                        log_gradient = log_factor[edge] * (
                            -r[edge, axis] / distance[edge]
                            - r[following, axis] / distance[following]
                        )
                        jacobian += outward[panel, edge, component] * log_gradient
                    result[query, component, axis] += strength[panel] * jacobian / (4 * math.pi)
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
    from openonda.runtime import worker_thread_count

    workers = min(worker_thread_count(), max(1, len(points) // 2048))
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
