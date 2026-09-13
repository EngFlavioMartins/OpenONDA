"""
Particle-surface collision detection for the VLM (point-in-quad tests).

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import numpy as np
import taichi as ti

SURFACE_COLLISION_EVENT_NONE = 0
SURFACE_COLLISION_EVENT_INTERSECTION = 1
SURFACE_COLLISION_EVENT_SIDE_BYPASS = 2
SURFACE_COLLISION_EVENT_CORE_OVERLAP = 3


def swept_panel_candidates(start, end, radius, panel_min, panel_max, tolerance):
    """Yield conservative ``(particle_index, panel_indices)`` swept AABB pairs.

    Positions have shape (N, 3), radii (N,), and panel bounds (M, 3), in m.
    Both trajectories and panel corners are linearly interpolated by the
    narrow phase. The factor two preserves its margin on *both* boxes.
    Batches bound temporary memory; distant particles never enter the costly
    substep/finite-surface classifier. Candidates are yielded in index order.
    """
    for first in range(0, len(start), 1024):
        last = min(first + 1024, len(start))
        margin = 2.0 * (radius[first:last, None] + tolerance)
        lower = np.minimum(start[first:last], end[first:last]) - margin
        upper = np.maximum(start[first:last], end[first:last]) + margin
        overlap = np.all(
            (upper[:, None, :] >= panel_min[None, :, :])
            & (lower[:, None, :] <= panel_max[None, :, :]),
            axis=2,
        )
        for local in np.flatnonzero(np.any(overlap, axis=1)):
            yield first + int(local), np.flatnonzero(overlap[local])


def _host_point_in_triangle(
    point: np.ndarray, triangle: np.ndarray, normal: np.ndarray, tol: float
) -> bool:
    """Scale-aware point-in-triangle test for the observer path."""
    signs = []
    for index in range(3):
        edge = triangle[(index + 1) % 3] - triangle[index]
        offset = point - triangle[index]
        signs.append(float(np.dot(np.cross(edge, offset), normal)))
    return bool(max(signs) <= tol or min(signs) >= -tol)


def _host_point_in_surface(
    point: np.ndarray, corners: np.ndarray, normal: np.ndarray, tol: float
) -> bool:
    """Check a planar quad, or an explicitly triangular apex, on the host."""
    a, b, c, d = corners
    scale = max(float(np.linalg.norm(corners - corners.mean(axis=0), axis=1).max()), 1.0)

    def duplicate(p, q):
        """Test whether two corners represent the same apex at the panel tolerance."""
        return np.linalg.norm(p - q) <= tol * scale

    triangles = []
    if duplicate(d, c) or duplicate(d, a):
        triangles.append(np.array([a, b, c]))
    elif duplicate(c, b):
        triangles.append(np.array([a, b, d]))
    else:
        first = np.array([a, b, c])
        second = np.array([a, c, d])
        alternate = np.array([a, b, d])
        alternate_second = np.array([b, c, d])
        primary_area = np.linalg.norm(np.cross(first[1] - first[0], first[2] - first[0]))
        alternate_area = np.linalg.norm(
            np.cross(alternate[1] - alternate[0], alternate[2] - alternate[0])
        )
        triangles = (
            [first, second] if primary_area >= alternate_area else [alternate, alternate_second]
        )
    return any(
        _host_point_in_triangle(point, triangle, normal, tol * scale) for triangle in triangles
    )


def _host_point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    """Return Euclidean point-to-segment distance."""
    delta = end - start
    denominator = float(np.dot(delta, delta))
    if denominator <= 0.0:
        return float(np.linalg.norm(point - start))
    parameter = np.clip(float(np.dot(point - start, delta) / denominator), 0.0, 1.0)
    return float(np.linalg.norm(point - (start + parameter * delta)))


def _host_segment_segment_distance(
    p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray
) -> float:
    """Return the closest distance between two finite segments."""
    u = p1 - p0
    v = q1 - q0
    w = p0 - q0
    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))
    denominator = a * c - b * b
    epsilon = np.finfo(np.float64).eps
    if a <= epsilon and c <= epsilon:
        return float(np.linalg.norm(p0 - q0))
    if a <= epsilon:
        return _host_point_segment_distance(p0, q0, q1)
    if c <= epsilon:
        return _host_point_segment_distance(q0, p0, p1)
    s = np.clip((b * e - c * d) / denominator, 0.0, 1.0) if denominator > epsilon else 0.0
    t = (b * s + e) / c
    if t < 0.0:
        t = 0.0
        s = np.clip(-d / a, 0.0, 1.0)
    elif t > 1.0:
        t = 1.0
        s = np.clip((b - d) / a, 0.0, 1.0)
    closest_p = p0 + s * u
    closest_q = q0 + t * v
    return float(np.linalg.norm(closest_p - closest_q))


def _host_segment_intersection_parameter_2d(
    p0: np.ndarray,
    p1: np.ndarray,
    q0: np.ndarray,
    q1: np.ndarray,
    tol: float,
) -> float | None:
    """Return the segment parameter for a 2-D segment intersection."""
    r = p1 - p0
    s = q1 - q0
    cross = float(r[0] * s[1] - r[1] * s[0])
    offset = q0 - p0
    if abs(cross) <= tol:
        # Collinear projections are handled by their nearest endpoint.  This
        # is sufficient for the distance minimum and avoids a false gap at a
        # shared panel edge.
        rr = float(np.dot(r, r))
        if rr <= tol * tol:
            return 0.0 if np.linalg.norm(offset) <= tol else None
        candidates = [float(np.dot(q0 - p0, r) / rr), float(np.dot(q1 - p0, r) / rr)]
        for candidate in candidates:
            if -tol <= candidate <= 1.0 + tol:
                projected = p0 + np.clip(candidate, 0.0, 1.0) * r
                if _host_point_segment_distance(projected, q0, q1) <= tol:
                    return float(np.clip(candidate, 0.0, 1.0))
        return None
    parameter = float((offset[0] * s[1] - offset[1] * s[0]) / cross)
    segment_parameter = float((offset[0] * r[1] - offset[1] * r[0]) / cross)
    if -tol <= parameter <= 1.0 + tol and -tol <= segment_parameter <= 1.0 + tol:
        return float(np.clip(parameter, 0.0, 1.0))
    return None


def _host_surface_frame(corners: np.ndarray, normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build an orthonormal in-plane frame for frame-independent distances."""
    normal = normal / np.linalg.norm(normal)
    tangent = corners[1] - corners[0]
    tangent -= np.dot(tangent, normal) * normal
    if np.linalg.norm(tangent) <= np.finfo(float).eps:
        tangent = corners[2] - corners[0]
        tangent -= np.dot(tangent, normal) * normal
    tangent /= np.linalg.norm(tangent)
    return tangent, np.cross(normal, tangent)


def _host_segment_surface_distance(
    start: np.ndarray, end: np.ndarray, corners: np.ndarray, normal: np.ndarray, tol: float
) -> tuple[float, np.ndarray]:
    """Return distance/representative point to a finite planar surface.

    The face interior is a valid closest set.  In particular, a stationary
    particle above the middle of a panel must report its normal separation even
    when every panel edge is far away; edge-only checks classify that case as
    a miss.
    """
    a = corners[0]
    signed_start = float(np.dot(start - a, normal))
    signed_end = float(np.dot(end - a, normal))
    candidates: list[tuple[float, np.ndarray]] = []
    if signed_start * signed_end <= 0.0:
        denominator = signed_start - signed_end
        parameter = 0.0 if abs(denominator) <= np.finfo(float).eps else signed_start / denominator
        parameter = float(np.clip(parameter, 0.0, 1.0))
        point = start + parameter * (end - start)
        projection = point - float(np.dot(point - a, normal)) * normal
        if _host_point_in_surface(projection, corners, normal, tol):
            candidates.append((abs(float(np.dot(point - a, normal))), point))
    tangent, bitangent = _host_surface_frame(corners, normal)
    origin = a

    def frame(point: np.ndarray) -> np.ndarray:
        """Project a world coordinate into the orthonormal in-plane distance frame."""
        return np.array(
            [float(np.dot(point - origin, tangent)), float(np.dot(point - origin, bitangent))]
        )

    projected_start = start - signed_start * normal
    projected_end = end - signed_end * normal
    if _host_point_in_surface(projected_start, corners, normal, tol):
        candidates.append((abs(signed_start), projected_start))
    if _host_point_in_surface(projected_end, corners, normal, tol):
        candidates.append((abs(signed_end), projected_end))
    start_2d, end_2d = frame(projected_start), frame(projected_end)
    polygon_2d = [frame(point) for point in corners]
    for edge_start, edge_end in zip(polygon_2d, polygon_2d[1:] + polygon_2d[:1], strict=True):
        parameter = _host_segment_intersection_parameter_2d(
            start_2d, end_2d, edge_start, edge_end, tol
        )
        if parameter is not None:
            point = start + parameter * (end - start)
            candidates.append((abs(signed_start + parameter * (signed_end - signed_start)), point))
    edges = list(zip(corners, np.roll(corners, -1, axis=0), strict=True))
    for edge_start, edge_end in edges:
        distance = _host_segment_segment_distance(start, end, edge_start, edge_end)
        candidates.append((distance, 0.5 * (edge_start + edge_end)))
    candidates.extend(
        (
            _host_point_segment_distance(corner, start, end),
            corner,
        )
        for corner in corners
    )
    return min(candidates, key=lambda item: item[0])


def _host_rigid_transform(
    start_corners: np.ndarray, end_corners: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the proper rigid transform mapping one panel pose to another."""
    start_centre = np.mean(start_corners, axis=0)
    end_centre = np.mean(end_corners, axis=0)
    source = start_corners - start_centre
    target = end_corners - end_centre
    covariance = source.T @ target
    left, _, right_transpose = np.linalg.svd(covariance)
    rotation = right_transpose.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_transpose[-1] *= -1.0
        rotation = right_transpose.T @ left.T
    translation = end_centre - rotation @ start_centre
    return rotation, translation


def classify_moving_finite_surface_segment(
    start: np.ndarray,
    end: np.ndarray,
    core_radius: float,
    start_panel_corners: np.ndarray,
    end_panel_corners: np.ndarray,
    start_panel_normal: np.ndarray,
    end_panel_normal: np.ndarray,
    *,
    tolerance: float,
    core_overlap_scale: float = 1.0,
) -> dict[str, object]:
    """Classify a particle segment in the relative frame of a moving panel.

    The end point is mapped back through the measured rigid transform from the
    start panel pose.  This is exact for translating or rigidly rotating
    panels over a subsegment and avoids freezing the panel at its midpoint.
    The endpoint normal is accepted to keep the moving-pose contract explicit;
    the fitted corner transform supplies the relative frame.  For a rigid
    pose this is exact.  Warped/non-rigid endpoint corners remain a diagnostic
    approximation and are handled by the fixed-start finite-face classifier.
    """
    del end_panel_normal
    start_corners = np.asarray(start_panel_corners, dtype=np.float64)
    end_corners = np.asarray(end_panel_corners, dtype=np.float64)
    rotation, translation = _host_rigid_transform(start_corners, end_corners)
    relative_end = rotation.T @ (np.asarray(end, dtype=np.float64) - translation)
    result = classify_finite_surface_segment(
        np.asarray(start, dtype=np.float64),
        relative_end,
        core_radius,
        start_corners,
        np.asarray(start_panel_normal, dtype=np.float64),
        tolerance=tolerance,
        core_overlap_scale=core_overlap_scale,
    )
    parameter = float(result.get("parameter", 0.5))
    result["position"] = np.asarray(start, dtype=np.float64) + parameter * (
        np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    )
    result["moving_surface"] = True
    return result


def classify_finite_surface_segment(
    start: np.ndarray,
    end: np.ndarray,
    core_radius: float,
    panel_corners: np.ndarray,
    panel_normal: np.ndarray,
    *,
    tolerance: float,
    core_overlap_scale: float = 1.0,
) -> dict[str, object]:
    """Classify one accepted transport segment against a finite surface.

    The operation is observer-only and never changes particle or surface
    state.  A signed plane crossing is classified as an interior intersection
    only when its finite projected position lies inside the panel.  Closest
    distance to the polygon edges is used for core overlap, so a trajectory
    that passes around a distant edge is not mislabeled as overlap.

    ``tolerance`` is a geometric measurement tolerance, not a collision
    thickness.  ``core_radius`` is the particle-centre filter scale and is
    reported separately by callers.
    """
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    corners = np.asarray(panel_corners, dtype=np.float64)
    normal = np.asarray(panel_normal, dtype=np.float64)
    if start.shape != (3,) or end.shape != (3,) or corners.shape != (4, 3):
        raise ValueError("finite-surface event geometry has invalid shape")
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= np.finfo(float).eps:
        raise ValueError("finite-surface event geometry has a zero normal")
    normal = normal / normal_norm
    scale = max(float(np.linalg.norm(corners - corners.mean(axis=0), axis=1).max()), 1.0)
    area = max(
        float(np.linalg.norm(np.cross(corners[1] - corners[0], corners[2] - corners[0]))),
        float(np.linalg.norm(np.cross(corners[2] - corners[0], corners[3] - corners[0]))),
    )
    if area <= 64.0 * np.finfo(float).eps * scale * scale:
        raise ValueError("finite-surface event geometry contains a degenerate panel")
    tolerance = max(float(tolerance), 64.0 * np.finfo(float).eps * scale)
    signed_start = float(np.dot(start - corners[0], normal))
    signed_end = float(np.dot(end - corners[0], normal))
    crossing = signed_start * signed_end <= 0.0
    intersection = None
    if crossing:
        denominator = signed_start - signed_end
        parameter = 0.0 if abs(denominator) <= np.finfo(float).eps else signed_start / denominator
        parameter = float(np.clip(parameter, 0.0, 1.0))
        candidate = start + parameter * (end - start)
        projection = candidate - float(np.dot(candidate - corners[0], normal)) * normal
        if _host_point_in_surface(projection, corners, normal, tolerance / scale):
            intersection = candidate
    if intersection is not None:
        denominator = signed_start - signed_end
        parameter = (
            0.5
            if abs(denominator) <= np.finfo(float).eps
            else float(np.clip(signed_start / denominator, 0.0, 1.0))
        )
        return {
            "event": SURFACE_COLLISION_EVENT_INTERSECTION,
            "position": intersection,
            "distance": 0.0,
            "signed_start": signed_start,
            "signed_end": signed_end,
            "parameter": parameter,
        }

    distance, closest = _host_segment_surface_distance(
        start, end, corners, normal, tolerance / scale
    )
    overlap_distance = tolerance + max(float(core_radius), 0.0) * float(core_overlap_scale)
    if distance <= overlap_distance:
        segment_delta = end - start
        denominator = float(np.dot(segment_delta, segment_delta))
        closest_parameter = (
            0.5
            if denominator <= np.finfo(float).eps
            else float(np.clip(np.dot(closest - start, segment_delta) / denominator, 0.0, 1.0))
        )
        return {
            "event": SURFACE_COLLISION_EVENT_CORE_OVERLAP,
            "position": closest,
            "distance": distance,
            "signed_start": signed_start,
            "signed_end": signed_end,
            "parameter": closest_parameter,
        }
    if signed_start * signed_end < 0.0:
        denominator = signed_start - signed_end
        parameter = float(np.clip(signed_start / denominator, 0.0, 1.0))
        return {
            "event": SURFACE_COLLISION_EVENT_SIDE_BYPASS,
            "position": start + (signed_start / (signed_start - signed_end)) * (end - start),
            "distance": distance,
            "signed_start": signed_start,
            "signed_end": signed_end,
            "parameter": parameter,
        }
    return {
        "event": SURFACE_COLLISION_EVENT_NONE,
        "position": closest,
        "distance": distance,
        "signed_start": signed_start,
        "signed_end": signed_end,
        "parameter": 0.5,
    }


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
