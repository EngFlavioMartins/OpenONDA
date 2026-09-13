"""Compiled, double-precision moving-panel observation without state mutation.

The scalar NumPy implementation in ``collision`` is the independent oracle.
Numba is used here because it preserves f64 geometry on every VPM backend,
including Metal, and is already a required dependency. No fast-math is enabled.
Rigid fits are batched once per panel/subinterval, independent of particle count.
"""

from numba import njit, prange
import numpy as np

from .collision import swept_panel_candidates

EPS = np.finfo(np.float64).eps


@njit(cache=True)
def _clip(value):
    """Clamp a segment coordinate to [0, 1]."""
    return min(max(value, 0.0), 1.0)


@njit(cache=True)
def _point_segment(point, start, end):
    """Euclidean distance to a finite segment, including zero length."""
    delta = end - start
    denominator = np.dot(delta, delta)
    parameter = _clip(np.dot(point - start, delta) / denominator) if denominator > 0 else 0.0
    return np.linalg.norm(point - (start + parameter * delta))


@njit(cache=True)
def _segment_segment(p0, p1, q0, q1):
    """Closest distance between finite segments, matching the host oracle."""
    u, v, w = p1 - p0, q1 - q0, p0 - q0
    a, b, c, d, e = np.dot(u, u), np.dot(u, v), np.dot(v, v), np.dot(u, w), np.dot(v, w)
    denominator = a * c - b * b
    if a <= EPS and c <= EPS:
        return np.linalg.norm(p0 - q0)
    if a <= EPS:
        return _point_segment(p0, q0, q1)
    if c <= EPS:
        return _point_segment(q0, p0, p1)
    s = _clip((b * e - c * d) / denominator) if denominator > EPS else 0.0
    t = (b * s + e) / c
    if t < 0.0:
        t, s = 0.0, _clip(-d / a)
    elif t > 1.0:
        t, s = 1.0, _clip((b - d) / a)
    return np.linalg.norm(p0 + s * u - q0 - t * v)


@njit(cache=True)
def _triangle(point, a, b, c, normal, tol):
    """Winding-independent signed edge test."""
    s0 = np.dot(np.cross(b - a, point - a), normal)
    s1 = np.dot(np.cross(c - b, point - b), normal)
    s2 = np.dot(np.cross(a - c, point - c), normal)
    return max(s0, s1, s2) <= tol or min(s0, s1, s2) >= -tol


@njit(cache=True)
def _inside(point, corners, normal, tol, scale):
    """Choose the same quad diagonal/duplicate-apex triangle as the oracle."""
    a, b, c, d = corners[0], corners[1], corners[2], corners[3]
    eps = tol * scale
    if np.linalg.norm(d - c) <= eps or np.linalg.norm(d - a) <= eps:
        return _triangle(point, a, b, c, normal, eps)
    if np.linalg.norm(c - b) <= eps:
        return _triangle(point, a, b, d, normal, eps)
    primary = np.linalg.norm(np.cross(b - a, c - a))
    alternate = np.linalg.norm(np.cross(b - a, d - a))
    if primary >= alternate:
        return _triangle(point, a, b, c, normal, eps) or _triangle(point, a, c, d, normal, eps)
    return _triangle(point, a, b, d, normal, eps) or _triangle(point, b, c, d, normal, eps)


@njit(cache=True)
def _intersection_2d(p0, p1, q0, q1, tol):
    """Return a projected intersection coordinate, or -1 for no intersection."""
    r, s, offset = p1 - p0, q1 - q0, q0 - p0
    cross = r[0] * s[1] - r[1] * s[0]
    if abs(cross) <= tol:
        rr = np.dot(r, r)
        if rr <= tol * tol:
            return 0.0 if np.linalg.norm(offset) <= tol else -1.0
        for candidate in (np.dot(q0 - p0, r) / rr, np.dot(q1 - p0, r) / rr):
            if (
                -tol <= candidate <= 1.0 + tol
                and _point_segment(p0 + _clip(candidate) * r, q0, q1) <= tol
            ):
                return _clip(candidate)
        return -1.0
    parameter = (offset[0] * s[1] - offset[1] * s[0]) / cross
    other = (offset[0] * r[1] - offset[1] * r[0]) / cross
    return (
        _clip(parameter) if -tol <= parameter <= 1.0 + tol and -tol <= other <= 1.0 + tol else -1.0
    )


@njit(cache=True)
def classify_relative_segment(start, end, radius, corners, normal, tolerance):
    """Return (event, distance, parameter, signed_start, signed_end).

    Inputs are f64, in the panel's start pose. The caller maps the parameter back
    to the world-space particle segment. Event codes match ``collision.py``.
    """
    normal_norm = np.linalg.norm(normal)
    if normal_norm <= EPS:
        raise ValueError("finite-surface event geometry has a zero normal")
    normal = normal / normal_norm
    centre = (corners[0] + corners[1] + corners[2] + corners[3]) * 0.25
    scale = 1.0
    for j in range(4):
        scale = max(scale, np.linalg.norm(corners[j] - centre))
    area = max(
        np.linalg.norm(np.cross(corners[1] - corners[0], corners[2] - corners[0])),
        np.linalg.norm(np.cross(corners[2] - corners[0], corners[3] - corners[0])),
    )
    if area <= 64 * EPS * scale * scale:
        raise ValueError("finite-surface event geometry contains a degenerate panel")
    tolerance = max(tolerance, 64 * EPS * scale)
    tol = tolerance / scale
    a = corners[0]
    ds, de = np.dot(start - a, normal), np.dot(end - a, normal)
    delta = end - start
    if ds * de <= 0:
        parameter = _clip(ds / (ds - de)) if abs(ds - de) > EPS else 0.0
        point = start + parameter * delta
        projection = point - np.dot(point - a, normal) * normal
        if _inside(projection, corners, normal, tol, scale):
            return 1, 0.0, parameter if abs(ds - de) > EPS else 0.5, ds, de

    tangent = corners[1] - a
    tangent -= np.dot(tangent, normal) * normal
    if np.linalg.norm(tangent) <= EPS:
        tangent = corners[2] - a
        tangent -= np.dot(tangent, normal) * normal
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)
    ps, pe = start - ds * normal, end - de * normal
    best, closest = np.inf, a.copy()
    # Maintain the host's candidate ordering for equal-distance ties.
    if _inside(ps, corners, normal, tol, scale):
        best, closest = abs(ds), ps
    if _inside(pe, corners, normal, tol, scale) and abs(de) < best:
        best, closest = abs(de), pe
    p0 = np.array([np.dot(ps - a, tangent), np.dot(ps - a, bitangent)])
    p1 = np.array([np.dot(pe - a, tangent), np.dot(pe - a, bitangent)])
    for j in range(4):
        q = corners[j] - a
        r = corners[(j + 1) % 4] - a
        q0 = np.array([np.dot(q, tangent), np.dot(q, bitangent)])
        q1 = np.array([np.dot(r, tangent), np.dot(r, bitangent)])
        parameter = _intersection_2d(p0, p1, q0, q1, tol)
        if parameter >= 0:
            distance = abs(ds + parameter * (de - ds))
            if distance < best:
                best, closest = distance, start + parameter * delta
    for j in range(4):
        q0, q1 = corners[j], corners[(j + 1) % 4]
        distance = _segment_segment(start, end, q0, q1)
        if distance < best:
            best, closest = distance, 0.5 * (q0 + q1)
    for j in range(4):
        distance = _point_segment(corners[j], start, end)
        if distance < best:
            best, closest = distance, corners[j]
    if best <= tolerance + max(radius, 0.0):
        denominator = np.dot(delta, delta)
        parameter = (
            _clip(np.dot(closest - start, delta) / denominator) if denominator > EPS else 0.5
        )
        return 3, best, parameter, ds, de
    if ds * de < 0:
        return 2, best, _clip(ds / (ds - de)), ds, de
    return 0, best, 0.5, ds, de


def moving_panel_geometry(start_corners, end_corners, start_normals, end_normals, n_substeps):
    """Batch proper SVD rigid fits for linearly interpolated panel poses."""
    alphas = np.arange(n_substeps + 1, dtype=np.float64) / n_substeps
    corners = (
        start_corners[None] + alphas[:, None, None, None] * (end_corners - start_corners)[None]
    )
    normals = start_normals[None] + alphas[:, None, None] * (end_normals - start_normals)[None]
    centres = corners.mean(axis=2)
    centred = corners - centres[:, :, None, :]
    covariance = np.swapaxes(centred[:-1], -1, -2) @ centred[1:]
    left, _, right = np.linalg.svd(covariance)
    rotation = np.swapaxes(right, -1, -2) @ np.swapaxes(left, -1, -2)
    negative = np.linalg.det(rotation) < 0
    right[negative, -1] *= -1
    rotation = np.swapaxes(right, -1, -2) @ np.swapaxes(left, -1, -2)
    translation = centres[1:] - np.einsum("spij,spj->spi", rotation, centres[:-1])
    return (
        np.ascontiguousarray(corners[:-1]),
        np.ascontiguousarray(normals[:-1]),
        rotation,
        translation,
    )


@njit(cache=True, parallel=True)
def _observe(
    start,
    end,
    radii,
    corners,
    normals,
    rotation,
    translation,
    panel_min,
    panel_max,
    surface_ids,
    n_surfaces,
    tolerance,
):
    """Sweep candidates and retain one event per particle/surface in oracle order."""
    count, n_substeps, n_panels = len(start), len(corners), len(panel_min)
    # Columns: event, panel, substep, distance, segment parameter.
    output = np.zeros((count, n_surfaces, 5), dtype=np.float64)
    priority = np.array([0, 3, 1, 2])
    for i in prange(count):
        margin = 2.0 * (radii[i] + tolerance)
        candidates = np.empty(n_panels, dtype=np.int64)
        n_candidates = 0
        for panel in range(n_panels):
            if np.all(np.maximum(start[i], end[i]) + margin >= panel_min[panel]) and np.all(
                np.minimum(start[i], end[i]) - margin <= panel_max[panel]
            ):
                candidates[n_candidates] = panel
                n_candidates += 1
        if n_candidates == 0:
            continue
        delta = end[i] - start[i]
        for sub in range(n_substeps):
            p0, p1 = (
                start[i] + (sub / n_substeps) * delta,
                start[i] + ((sub + 1) / n_substeps) * delta,
            )
            for candidate in range(n_candidates):
                panel = candidates[candidate]
                if np.any(np.maximum(p0, p1) + margin < panel_min[panel]) or np.any(
                    np.minimum(p0, p1) - margin > panel_max[panel]
                ):
                    continue
                relative_end = rotation[sub, panel].T @ (p1 - translation[sub, panel])
                event, distance, parameter, _, _ = classify_relative_segment(
                    p0, relative_end, radii[i], corners[sub, panel], normals[sub, panel], tolerance
                )
                if event == 0:
                    continue
                surface = surface_ids[panel]
                old_event = int(output[i, surface, 0])
                if priority[event] > priority[old_event] or (
                    priority[event] == priority[old_event] and distance < output[i, surface, 3]
                ):
                    output[i, surface, 0] = event
                    output[i, surface, 1] = panel
                    output[i, surface, 2] = sub
                    output[i, surface, 3] = distance
                    output[i, surface, 4] = parameter
    return output


def observe_moving_surfaces(
    start,
    end,
    radii,
    start_corners,
    end_corners,
    start_normals,
    end_normals,
    panel_surfaces,
    n_substeps,
    tolerance,
):
    """Yield winning (particle, panel, substep, event, distance, world position).

    All geometry is evaluated in f64. Output storage is O(particles * surfaces);
    the pair scan has no particle-by-panel tensor. Empty surface names retain
    the oracle's per-panel fallback identity.
    """
    keys = [str(name) or f"panel_{i}" for i, name in enumerate(panel_surfaces)]
    names = sorted(set(keys))
    mapping = {name: i for i, name in enumerate(names)}
    ids = np.array([mapping[name] for name in keys], dtype=np.int64)
    geometry = moving_panel_geometry(
        start_corners, end_corners, start_normals, end_normals, n_substeps
    )
    lower = np.minimum(start_corners.min(axis=1), end_corners.min(axis=1))
    upper = np.maximum(start_corners.max(axis=1), end_corners.max(axis=1))
    start, end, radii = (np.ascontiguousarray(a, dtype=np.float64) for a in (start, end, radii))
    # Cull far-wake particles in bounded vectorized batches. A compiled scalar
    # all-panel scan still allocates many tiny arrays for a mature wake; only
    # particles with a conservative candidate enter the detailed compiled walk.
    selected = np.fromiter(
        (i for i, _ in swept_panel_candidates(start, end, radii, lower, upper, tolerance)),
        dtype=np.int64,
    )
    if not len(selected):
        return
    result = _observe(
        start[selected],
        end[selected],
        radii[selected],
        *geometry,
        lower,
        upper,
        ids,
        len(names),
        tolerance,
    )
    for i, surface in np.argwhere(result[:, :, 0] != 0):
        event, panel, sub, distance, parameter = result[i, surface]
        i = selected[i]
        position = start[i] + ((sub + parameter) / n_substeps) * (end[i] - start[i])
        yield int(i), int(panel), int(sub), int(event), float(distance), position
