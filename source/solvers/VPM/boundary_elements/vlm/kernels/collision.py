"""
Particle-surface collision detection for the VLM (point-in-quad tests).

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import taichi as ti


@ti.func
def is_point_in_quad(
    p: ti.types.vector(3, ti.f32),
    a: ti.types.vector(3, ti.f32),
    b: ti.types.vector(3, ti.f32),
    c: ti.types.vector(3, ti.f32),
    d: ti.types.vector(3, ti.f32),
    normal: ti.types.vector(3, ti.f32),
) -> bool:
    """
    Check if point p (projected onto plane) is inside quad ABCD.

    Uses the cross product method: if p is inside the quad, all edge-to-point
    cross products should align with the normal (same sign).

    Args:
        p: Point to test
        a, b, c, d: Quad vertices in order (CCW or CW)
        normal: Quad normal vector (unit)

    Returns:
        True if point is inside quad
    """
    # Edge vectors
    ab = b - a
    bc = c - b
    cd = d - c
    da = a - d

    # Vectors to point
    ap = p - a
    bp = p - b
    cp = p - c
    dp = p - d

    # Cross products dotted with normal (to check "sidedness")
    # If p is inside, all cross products should align with normal
    c1 = ab.cross(ap).dot(normal)
    c2 = bc.cross(bp).dot(normal)
    c3 = cd.cross(cp).dot(normal)
    c4 = da.cross(dp).dot(normal)

    return c1 >= 0.0 and c2 >= 0.0 and c3 >= 0.0 and c4 >= 0.0


@ti.kernel
def detect_surface_collisions_kernel(
    particle_pos: ti.template(),
    particle_tags: ti.template(),
    panel_corners: ti.template(),  # (N_panels, 4, 3)
    panel_normals: ti.template(),
    n_particles: int,
    n_panels: int,
    tolerance: float,
):
    """
    Tag particles that impinge on VLM panels.

    A particle is considered to collide if:
    1. It is within `tolerance` distance of the panel plane
    2. Its projection onto the plane falls inside the panel quad

    Args:
        particle_pos: Particle positions (N_particles, 3)
        particle_tags: Output tags (N_particles,) - 0=safe, 1=collision
        panel_corners: Panel corner points (N_panels, 4, 3)
        panel_normals: Panel normal vectors (N_panels, 3)
        n_particles: Number of active particles
        n_panels: Number of active panels
        tolerance: Collision distance threshold [m]
    """
    for i in range(n_particles):
        pos = particle_pos[i]

        # Skip if already tagged (optimization)
        if particle_tags[i] == 0:
            for j in range(n_panels):
                # 1. Plane Distance Check
                # Use first corner 'A' as reference point on plane
                a = panel_corners[j, 0]
                n = panel_normals[j]

                vec = pos - a
                dist_perp = ti.abs(vec.dot(n))

                # Check 1: Is particle within 'thickness' of the plate?
                if dist_perp < tolerance:
                    # 2. Boundary Check (Point in Quad)
                    # Project point onto plane to handle slight offsets
                    pos_proj = pos - dist_perp * n

                    b = panel_corners[j, 1]
                    c = panel_corners[j, 2]
                    d = panel_corners[j, 3]

                    if is_point_in_quad(pos_proj, a, b, c, d, n):
                        particle_tags[i] = 1
                        # Break inner loop (particle can only be tagged once)
                        break
