"""Independent quadrature of a source-panel field and its full 3D Jacobian.

This diagnostic integrates the smooth area kernel at off-surface targets.
Triangle subdivision uses geometry alone; quadrature-order refinement must
be checked by the caller. It is not a production particle-stage operator.
"""

from __future__ import annotations

import numpy as np


def separated_triangles(vertices, strengths, point):
    """Subdivide until each target is at least three enclosing radii away."""
    pending = np.asarray(vertices, dtype=float)
    amplitudes = np.asarray(strengths, dtype=float)
    leaves, values = [], []
    for _ in range(14):
        center = pending.mean(axis=1)
        radius = np.max(np.linalg.norm(pending - center[:, None], axis=2), axis=1)
        distant = np.linalg.norm(center - point, axis=1) >= 3 * radius
        leaves.append(pending[distant])
        values.append(amplitudes[distant])
        if np.all(distant):
            return np.concatenate(leaves), np.concatenate(values)
        triangle, strength = pending[~distant], amplitudes[~distant]
        a, b, c = triangle[:, 0], triangle[:, 1], triangle[:, 2]
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        pending = np.concatenate([np.stack(corners, axis=1) for corners in
                                  ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))])
        amplitudes = np.tile(strength, 4)
    raise ValueError("Target is on or too close to the panel surface for this reference")


def surface_velocity_gradient(vertices, strengths, points, order):
    """Integrate R/r^3 and I/r^3 - 3 RR/r^5 with Duffy area quadrature."""
    vertices, strengths, points = (np.asarray(value, dtype=float) for value in
                                  (vertices, strengths, points))
    assert vertices.shape == (len(strengths), 3, 3) and points.shape[1:] == (3,)
    assert all(np.all(np.isfinite(value)) for value in (vertices, strengths, points))
    nodes, weights = np.polynomial.legendre.leggauss(order)
    a, b = np.meshgrid((nodes + 1) / 2, (nodes + 1) / 2, indexing="ij")
    weight = (np.outer(weights / 2, weights / 2) * (1 - a)).ravel()
    velocity, gradient = np.zeros((len(points), 3)), np.zeros((len(points), 3, 3))
    counts = []
    for index, point in enumerate(points):
        triangles, strength = separated_triangles(vertices, strengths, point)
        counts.append(len(triangles))
        e1, e2 = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        jacobian = np.linalg.norm(np.cross(e1, e2), axis=1)
        source = (triangles[:, None, 0] + a.ravel()[None, :, None] * e1[:, None]
                  + ((1 - a) * b).ravel()[None, :, None] * e2[:, None]).reshape(-1, 3)
        amplitude = (strength[:, None] * jacobian[:, None] * weight[None]).ravel() / (4 * np.pi)
        r = point - source
        radius2 = np.sum(r * r, axis=1)
        assert radius2.min() > 0
        inverse3 = radius2 ** -1.5
        velocity[index] = np.einsum("si,s->i", r, amplitude * inverse3)
        gradient[index] = (np.eye(3) * np.sum(amplitude * inverse3)
                           - np.einsum("si,sj,s->ij", r, r, 3 * amplitude * inverse3 / radius2))
    return velocity, gradient, np.asarray(counts, dtype=int)
