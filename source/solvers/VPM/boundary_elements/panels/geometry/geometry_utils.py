"""
Geometry helpers for the panel solver: point-in-STL-body tests (ray casting)
and filtering of vortex particles that fall inside solid bodies.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from dataclasses import dataclass
import logging

import numpy as np

logger = logging.getLogger("vpm")

@dataclass
class ParticleFilterResult:
    positions: np.ndarray
    velocities: np.ndarray
    strengths: np.ndarray
    radii: np.ndarray
    volumes: np.ndarray
    viscosities: np.ndarray
    outside_mask: np.ndarray
    outside_indices: np.ndarray

def _ray_intersects_triangle(
    origin: np.ndarray, direction: np.ndarray, tri: np.ndarray, eps: float
) -> bool:
    v0, v1, v2 = tri[0], tri[1], tri[2]
    edge1 = v1 - v0
    edge2 = v2 - v0
    pvec = np.cross(direction, edge2)
    det = np.dot(edge1, pvec)
    if abs(det) < eps:
        return False

    inv_det = 1.0 / det
    tvec = origin - v0
    u = np.dot(tvec, pvec) * inv_det
    if u < 0.0 or u > 1.0:
        return False

    qvec = np.cross(tvec, edge1)
    v = np.dot(direction, qvec) * inv_det
    if v < 0.0 or (u + v) > 1.0:
        return False

    t = np.dot(edge2, qvec) * inv_det
    return t > eps

def point_inside_stl_body(
    points: np.ndarray, triangles: np.ndarray, tolerance: float = 1e-9
) -> tuple[np.ndarray, np.ndarray]:
    """
    Determine if points are inside a closed STL body using ray casting.

    Ray casting algorithm:
    - Cast a ray from each point in a random direction
    - Count intersections with surface
    - Odd number of intersections = inside
    - Even number = outside

    Parameters
    ----------
    points : np.ndarray
        Points to test (N, 3)
    triangles : np.ndarray
        Triangle vertices (M, 3, 3)
    tolerance : float
        Tolerance for intersection tests

    Returns
    -------
    inside : np.ndarray
        Boolean array (N,) where True = inside body
    inside_indices : np.ndarray
        Indices of points inside body
    """
    N = points.shape[0]
    inside = np.zeros(N, dtype=bool)
    ray_direction = np.array([1.0, 0.1234567, 0.017], dtype=np.float64)
    ray_direction /= np.linalg.norm(ray_direction)

    for i in range(N):
        intersections = 0
        origin = points[i]
        for tri in triangles:
            if _ray_intersects_triangle(origin, ray_direction, tri, tolerance):
                intersections += 1
        inside[i] = (intersections % 2) == 1

    inside_indices = np.nonzero(inside)[0]
    return inside, inside_indices

def filter_particles_outside_body(
    positions: np.ndarray,
    velocities: np.ndarray,
    strengths: np.ndarray,
    radii: np.ndarray,
    volumes: np.ndarray,
    viscosities: np.ndarray,
    body_triangles: np.ndarray,
) -> ParticleFilterResult:
    """
    Filter particle arrays to exclude particles inside body.

    Parameters
    ----------
    positions : np.ndarray
        Particle positions (N, 3)
    velocities : np.ndarray
        Particle velocities (N, 3)
    strengths : np.ndarray
        Particle vortex strengths (N, 3)
    radii : np.ndarray
        Particle radii (N,)
    volumes : np.ndarray
        Particle volumes (N,)
    viscosities : np.ndarray
        Particle viscosities (N,)
    body_triangles : np.ndarray
        Body triangle vertices (M, 3, 3)

    Returns
    -------
    ParticleFilterResult
        Typed container with filtered arrays and mask/index metadata.
    """
    # Determine which particles are outside
    inside, _inside_indices = point_inside_stl_body(positions, body_triangles)
    outside = ~inside

    n_inside = np.sum(inside)
    n_outside = np.sum(outside)

    logger.debug(
        f"Filtering particles: {n_inside} inside body (removed), {n_outside} outside (kept)"
    )

    # Filter arrays
    return ParticleFilterResult(
        positions=positions[outside],
        velocities=velocities[outside],
        strengths=strengths[outside],
        radii=radii[outside],
        volumes=volumes[outside],
        viscosities=viscosities[outside],
        outside_mask=outside,
        outside_indices=np.nonzero(outside)[0],
    )
