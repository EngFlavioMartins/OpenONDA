"""
Particle-surface collision detection for the VLM (point-in-quad tests).

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import taichi as ti

SURFACE_COLLISION_EVENT_NONE = 0
SURFACE_COLLISION_EVENT_INTERSECTION = 1
SURFACE_COLLISION_EVENT_SIDE_BYPASS = 2
SURFACE_COLLISION_EVENT_CORE_OVERLAP = 3


@ti.func
def _safe_normalized_normal(normal: ti.math.vec3) -> ti.math.vec3:
    """Return a normalized surface normal, handling degenerate zero normals."""
    normal_mag = normal.norm()
    result = ti.Vector([0.0, 0.0, 0.0])
    if normal_mag > 0.0:
        result = normal / normal_mag
    return result


@ti.func
def _is_point_in_triangle(
    point: ti.math.vec3,
    a: ti.math.vec3,
    b: ti.math.vec3,
    c: ti.math.vec3,
    normal_unit: ti.math.vec3,
) -> bool:
    """Return true if point is inside triangle ABC (including edges)."""
    tol = 1.0e-14
    ab = b - a
    bc = c - b
    ca = a - c

    ap = point - a
    bp = point - b
    cp = point - c

    c1 = ab.cross(ap).dot(normal_unit)
    c2 = bc.cross(bp).dot(normal_unit)
    c3 = ca.cross(cp).dot(normal_unit)

    all_nonneg = c1 >= -tol and c2 >= -tol and c3 >= -tol
    all_nonpos = c1 <= tol and c2 <= tol and c3 <= tol
    return all_nonneg or all_nonpos


@ti.func
def is_point_in_quad(
    p: ti.math.vec3,
    a: ti.math.vec3,
    b: ti.math.vec3,
    c: ti.math.vec3,
    d: ti.math.vec3,
    normal: ti.math.vec3,
) -> bool:
    """Check whether point ``p`` lies in finite quad ABCD.

    The quad may be oriented with either normal winding; triangulate once and
    accept both diagonals if one triangle is degenerate.
    """
    normal_unit = _safe_normalized_normal(normal)
    inside = False
    if normal_unit.norm() > 0.0:
        # Primary split on diagonal AC
        if _is_point_in_triangle(p, a, b, c, normal_unit) or _is_point_in_triangle(
            p, a, c, d, normal_unit
        ):
            inside = True
        else:
            # Degenerate AC diagonal or explicit fallback for warped quads
            abcd = (b - a).cross(c - a).norm()
            if abcd <= 0.0 and (
                _is_point_in_triangle(p, a, b, d, normal_unit)
                or _is_point_in_triangle(p, b, c, d, normal_unit)
            ):
                inside = True
    return inside


@ti.kernel
def detect_surface_collision_events_kernel(
    particle_start_position: ti.template(),
    particle_end_position: ti.template(),
    particle_core_radius: ti.template(),
    particle_event: ti.template(),
    particle_panel: ti.template(),
    panel_corners: ti.template(),
    panel_normals: ti.template(),
    n_particles_total: int,
    n_panels: int,
    tolerance: float,
    core_overlap_scale: float,
):
    """Classify finite-surface trajectory events without mutating particles.

    event codes:
        0 = no event
        1 = full-panel intersection / interior endpoint
        2 = signed-side bypass around a finite edge
        3 = core-overlap outside finite polygon (closest approach)
    """
    for i in range(n_particles_total):
        particle_event[i] = SURFACE_COLLISION_EVENT_NONE
        particle_panel[i] = -1

        start = particle_start_position[i]
        end = particle_end_position[i]
        radius = particle_core_radius[i]
        delta = end - start

        for j in range(n_panels):
            if particle_event[i] != SURFACE_COLLISION_EVENT_NONE:
                break

            a = panel_corners[j, 0]
            b = panel_corners[j, 1]
            c = panel_corners[j, 2]
            d = panel_corners[j, 3]
            unit_normal = _safe_normalized_normal(panel_normals[j])
            if unit_normal.norm() == 0.0:
                continue

            signed_start = (start - a).dot(unit_normal)
            signed_end = (end - a).dot(unit_normal)
            abs_start = ti.abs(signed_start)
            abs_end = ti.abs(signed_end)

            if abs_start <= tolerance and is_point_in_quad(
                start - signed_start * unit_normal, a, b, c, d, unit_normal
            ):
                particle_event[i] = SURFACE_COLLISION_EVENT_INTERSECTION
                particle_panel[i] = j
                break

            if abs_end <= tolerance and is_point_in_quad(
                end - signed_end * unit_normal, a, b, c, d, unit_normal
            ):
                particle_event[i] = SURFACE_COLLISION_EVENT_INTERSECTION
                particle_panel[i] = j
                break

            # Signed crossing of the panel plane with finite-quad check.
            if signed_start * signed_end < 0.0:
                t = signed_start / (signed_start - signed_end)
                if 0.0 <= t <= 1.0:
                    intersection = start + t * delta
                    if is_point_in_quad(intersection, a, b, c, d, unit_normal):
                        particle_event[i] = SURFACE_COLLISION_EVENT_INTERSECTION
                        particle_panel[i] = j
                        break
                    particle_event[i] = SURFACE_COLLISION_EVENT_SIDE_BYPASS
                    particle_panel[i] = j

            # Core overlap: minimum normal distance is small while the closest
            # projected position remains outside the finite panel.
            denom = signed_end - signed_start
            t_clamp = 0.0
            if ti.abs(denom) > 0.0:
                t_clamp = -signed_start / denom
                t_clamp = ti.max(0.0, ti.min(1.0, t_clamp))
            closest = start + t_clamp * delta
            closest_signed_distance = (start - a).dot(unit_normal) + t_clamp * denom
            closest_projection = closest - closest_signed_distance * unit_normal
            if (
                particle_event[i] == SURFACE_COLLISION_EVENT_NONE
                and not is_point_in_quad(closest_projection, a, b, c, d, unit_normal)
                and ti.abs(closest_signed_distance) <= tolerance + core_overlap_scale * radius
            ):
                particle_event[i] = SURFACE_COLLISION_EVENT_CORE_OVERLAP
                particle_panel[i] = j


@ti.kernel
def detect_surface_collisions_kernel(
    particle_pos: ti.template(),
    particle_tags: ti.template(),
    panel_corners: ti.template(),  # (N_panels, 4, 3)
    panel_normals: ti.template(),
    n_particles_total: int,
    n_panels: int,
    tolerance: float,
):
    """
    Tag particles that impinge on VLM panels.

    A particle is considered to collide if:
    1. It is within `tolerance` distance of the panel plane
    2. Its projection onto the plane falls inside the panel quad

    Args:
        particle_pos: Particle position (N_particles, 3)
        particle_tags: Output tags (N_particles,) - 0=safe, 1=collision
        panel_corners: Panel corner points (N_panels, 4, 3)
        panel_normals: Panel normal vectors (N_panels, 3)
        n_particles_total: Number of active particles
        n_panels: Number of active panels
        tolerance: Collision distance threshold [m]
    """
    for i in range(n_particles_total):
        pos = particle_pos[i]

        # Skip if already tagged (optimization)
        if particle_tags[i] == 0:
            for j in range(n_panels):
                # 1. Plane Distance Check
                # Use first corner 'A' as reference point on plane
                a = panel_corners[j, 0]
                n = panel_normals[j]
                n_unit = _safe_normalized_normal(n)
                if n_unit.norm() == 0.0:
                    continue

                vec = pos - a
                dist_signed = vec.dot(n_unit)
                dist_perp = ti.abs(dist_signed)

                # Check 1: Is particle within 'thickness' of the plate?
                if dist_perp < tolerance:
                    # 2. Boundary Check (Point in Quad)
                    # Project point onto plane to handle slight offsets
                    pos_proj = pos - dist_signed * n_unit

                    b = panel_corners[j, 1]
                    c = panel_corners[j, 2]
                    d = panel_corners[j, 3]

                    if is_point_in_quad(pos_proj, a, b, c, d, n):
                        particle_tags[i] = 1
                        # Break inner loop (particle can only be tagged once)
                        break
