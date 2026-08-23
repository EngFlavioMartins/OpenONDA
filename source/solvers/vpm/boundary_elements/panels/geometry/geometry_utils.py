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
    """Result of filtering particles against a closed STL body.

    Contains the filtered particle arrays (only particles outside the body)
    together with the boolean mask and integer indices of the outside subset.

    Parameters
    ----------
    position : np.ndarray
        Filtered particle position with shape ``(M, 3)``.
    velocity : np.ndarray
        Filtered particle velocity with shape ``(M, 3)``.
    doublet_strength : np.ndarray
        Filtered particle vortex doublet_strength with shape ``(M, 3)``.
    core_radius : np.ndarray
        Filtered particle core_radius with shape ``(M,)``.
    particle_volume : np.ndarray
        Filtered particle volume with shape ``(M,)``.
    kinematic_viscosity : np.ndarray
        Filtered particle kinematic_viscosity with shape ``(M,)``.
    outside_mask : np.ndarray
        Boolean mask of shape ``(N,)`` where ``True`` indicates particles
        outside the body.
    outside_indices : np.ndarray
        Integer indices of the outside particles with shape ``(M,)``.
    """

    position: np.ndarray
    velocity: np.ndarray
    doublet_strength: np.ndarray
    core_radius: np.ndarray
    particle_volume: np.ndarray
    kinematic_viscosity: np.ndarray
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
        Triangle vertex_position (M, 3, 3)
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
    position: np.ndarray,
    velocity: np.ndarray,
    doublet_strength: np.ndarray,
    core_radius: np.ndarray,
    particle_volume: np.ndarray,
    kinematic_viscosity: np.ndarray,
    body_triangles: np.ndarray,
) -> ParticleFilterResult:
    """
    Filter particle arrays to exclude particles inside body.

    Parameters
    ----------
    position : np.ndarray
        Particle position (N, 3)
    velocity : np.ndarray
        Particle velocity (N, 3)
    doublet_strength : np.ndarray
        Particle vortex doublet_strength (N, 3)
    core_radius : np.ndarray
        Particle core_radius (N,)
    particle_volume : np.ndarray
        Particle particle_volume (N,)
    kinematic_viscosity : np.ndarray
        Particle kinematic_viscosity (N,)
    body_triangles : np.ndarray
        Body triangle vertex_position (M, 3, 3)

    Returns
    -------
    ParticleFilterResult
        Typed container with filtered arrays and mask/index metadata.
    """
    # Determine which particles are outside
    inside, _inside_indices = point_inside_stl_body(position, body_triangles)
    outside = ~inside

    n_inside = np.sum(inside)
    n_outside = np.sum(outside)

    logger.debug(
        f"Filtering particles: {n_inside} inside body (removed), {n_outside} outside (kept)"
    )

    # Filter arrays
    return ParticleFilterResult(
        position=position[outside],
        velocity=velocity[outside],
        doublet_strength=doublet_strength[outside],
        core_radius=core_radius[outside],
        particle_volume=particle_volume[outside],
        kinematic_viscosity=kinematic_viscosity[outside],
        outside_mask=outside,
        outside_indices=np.nonzero(outside)[0],
    )
