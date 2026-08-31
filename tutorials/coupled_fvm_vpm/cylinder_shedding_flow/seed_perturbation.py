"""Deterministic initial states for the cylinder reference calculation."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _normalise(velocity: np.ndarray, amplitude: float) -> np.ndarray:
    peak = float(np.linalg.norm(velocity, axis=1).max())
    if not np.isfinite(peak) or peak <= 0.0:
        raise RuntimeError("Cylinder seed is zero or non-finite")
    return velocity * (amplitude / peak)


def _affine_interpolate_2d(
    source_xy: np.ndarray,
    source_values: np.ndarray,
    target_xy: np.ndarray,
    *,
    k: int = 12,
) -> np.ndarray:
    """Interpolate a smooth saved x-y field with affine-exact local weights."""
    from scipy.spatial import cKDTree

    source_xy = np.asarray(source_xy, dtype=np.float64)
    target_xy = np.asarray(target_xy, dtype=np.float64)
    values = np.asarray(source_values, dtype=np.float64)
    neighbours = min(int(k), len(source_xy))
    distance, indices = cKDTree(source_xy).query(target_xy, k=neighbours)
    if neighbours == 1:
        distance = distance[:, np.newaxis]
        indices = indices[:, np.newaxis]
    inverse_distance = 1.0 / (distance + 1.0e-12) ** 2
    inverse_distance /= inverse_distance.sum(axis=1, keepdims=True)
    offsets = source_xy[indices] - target_xy[:, np.newaxis, :]
    scale = np.max(np.abs(offsets), axis=1)
    scale[scale < 1.0e-12] = 1.0
    design = np.concatenate(
        (np.ones((*offsets.shape[:2], 1)), offsets / scale[:, np.newaxis, :]), axis=2
    )
    normal = np.einsum("nki,nk,nkj->nij", design, inverse_distance, design)
    intercept = np.linalg.pinv(normal, rcond=1.0e-12)[:, :, 0]
    weights = inverse_distance * np.einsum("nki,ni->nk", design, intercept)
    weight_sum = weights.sum(axis=1, keepdims=True)
    valid = (
        np.all(np.isfinite(weights), axis=1)
        & (np.abs(weight_sum[:, 0]) > 1.0e-12)
        & (np.sum(np.abs(weights), axis=1) < 10.0)
    )
    weights[valid] /= weight_sum[valid]
    weights[~valid] = inverse_distance[~valid]
    exact = np.any(distance <= 1.0e-12, axis=1)
    if np.any(exact):
        weights[exact] = 0.0
        weights[np.flatnonzero(exact), np.argmin(distance[exact], axis=1)] = 1.0
    sampled = values[indices]
    if values.ndim == 1:
        return np.sum(weights * sampled, axis=1)
    return np.sum(weights[:, :, np.newaxis] * sampled, axis=1)


def build_cylinder_initial_state(
    cell_centre: np.ndarray,
    *,
    freestream_velocity,
    diameter: float,
    seed_amplitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a no-slip-compatible analytic start with an antisymmetric seed."""
    points = np.asarray(cell_centre, dtype=np.float64).reshape(-1, 3)
    freestream = np.asarray(freestream_velocity, dtype=np.float64)
    if freestream.shape != (3,) or not np.allclose(freestream[1:], 0.0):
        raise ValueError("Cylinder reference requires an x-directed freestream")

    radius = 0.5 * float(diameter)
    x = points[:, 0]
    y = points[:, 1]
    radius_squared = x * x + y * y
    if np.any(radius_squared <= radius * radius):
        raise ValueError("Cylinder initial state contains solid cell centres")
    radial = np.sqrt(radius_squared)
    ratio = radius * radius / radius_squared
    wall_shape = (1.0 - ratio) ** 2
    wall_shape_derivative = 4.0 * radius * radius * (1.0 - ratio) / radial**3

    velocity = np.zeros_like(points)
    velocity[:, 0] = freestream[0] * (wall_shape + y * y * wall_shape_derivative / radial)
    velocity[:, 1] = -freestream[0] * x * y * wall_shape_derivative / radial

    if seed_amplitude > 0.0:
        centre_x = 0.75 * float(diameter)
        half_width = 0.4 * float(diameter)
        envelope = np.exp(-((x - centre_x) ** 2 + y**2) / half_width**2)
        envelope_x = -2.0 * (x - centre_x) * envelope / half_width**2
        envelope_y = -2.0 * y * envelope / half_width**2
        radial_factor = (1.0 - ratio) ** 2
        radial_factor_x = 4.0 * radius * radius * (1.0 - ratio) * x / radius_squared**2
        radial_factor_y = 4.0 * radius * radius * (1.0 - ratio) * y / radius_squared**2
        seed = np.zeros_like(points)
        seed[:, 0] = envelope_y * radial_factor + envelope * radial_factor_y
        seed[:, 1] = -(envelope_x * radial_factor + envelope * radial_factor_x)
        velocity += _normalise(seed, float(seed_amplitude) * abs(float(freestream[0])))

    pressure = 0.5 * (float(np.dot(freestream, freestream)) - np.einsum("ij,ij->i", velocity, velocity))
    return velocity, pressure


def build_transferred_initial_state(
    cell_centre: np.ndarray,
    source_field: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Transfer a saved field by spanwise averaging and nearest interpolation.

    This is kept only for an explicit user-requested restart/transfer.  The
    ordinary grid study starts each case from the analytic state above.
    """
    import pyvista as pv

    source_path = Path(source_field).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    grid = pv.read(source_path)
    centres = np.asarray(grid.cell_centers().points, dtype=np.float64)
    velocity = np.asarray(grid.cell_data["velocity"], dtype=np.float64)
    pressure = np.asarray(grid.cell_data["kinematic_pressure"], dtype=np.float64)
    rounded_xy = np.round(centres[:, :2], decimals=12)
    source_xy, inverse, counts = np.unique(rounded_xy, axis=0, return_inverse=True, return_counts=True)
    mean_velocity = np.zeros((len(source_xy), 3), dtype=np.float64)
    mean_pressure = np.zeros(len(source_xy), dtype=np.float64)
    np.add.at(mean_velocity, inverse, velocity)
    np.add.at(mean_pressure, inverse, pressure)
    mean_velocity /= counts[:, np.newaxis]
    mean_pressure /= counts
    mean_velocity[:, 2] = 0.0
    target_xy = np.round(np.asarray(cell_centre)[:, :2], decimals=12)
    return (
        _affine_interpolate_2d(source_xy, mean_velocity, target_xy),
        _affine_interpolate_2d(source_xy, mean_pressure, target_xy),
    )
