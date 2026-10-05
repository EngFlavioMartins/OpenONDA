"""Compact divergence-free velocity disturbance for the cylinder flow."""

import numpy as np


def cylinder_initial_velocity(
    positions: np.ndarray,
    *,
    amplitude: float,
    radius: float,
    centre: float,
    freestream_velocity,
) -> np.ndarray:
    """In-plane curl of a compact streamfunction, uniform along the periodic span."""
    position = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    x, y = position[:, :2].T
    radius_squared = radius**2
    support = np.maximum(1 - ((x - centre) ** 2 + y**2) / radius_squared, 0)
    velocity = np.tile(np.asarray(freestream_velocity, dtype=np.float64), (len(position), 1))
    velocity[:, 0] -= 8 * amplitude * y / radius_squared * support**3
    velocity[:, 1] += 8 * amplitude * (x - centre) / radius_squared * support**3
    return velocity
