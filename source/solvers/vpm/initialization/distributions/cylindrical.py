"""Exact-spacing cylindrical particle distributions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..data import ParticleDistribution
from .rectangular import create_rectangular_distribution


def create_cylindrical_distribution(
    *,
    radius: float,
    length: float,
    spacing: float,
    core_radius_ratio: float,
    centre: Sequence[float] = (0.0, 0.0, 0.0),
    axis: str = "z",
) -> ParticleDistribution:
    """Create a Cartesian lattice clipped to a solid circular cylinder."""
    radius = float(radius)
    length = float(length)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and positive")
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("length must be finite and positive")
    if axis not in {"x", "y", "z"}:
        raise ValueError("axis must be 'x', 'y', or 'z'")
    centre_array = np.asarray(centre, dtype=float)
    if centre_array.shape != (3,) or not np.all(np.isfinite(centre_array)):
        raise ValueError("centre must contain three finite coordinates")
    axial_index = {"x": 0, "y": 1, "z": 2}[axis]
    bounds = [(centre_array[i] - radius, centre_array[i] + radius) for i in range(3)]
    bounds[axial_index] = (
        centre_array[axial_index] - 0.5 * length,
        centre_array[axial_index] + 0.5 * length,
    )
    box = create_rectangular_distribution(
        bounds=bounds,
        spacing=spacing,
        core_radius_ratio=core_radius_ratio,
    )
    transverse = [index for index in range(3) if index != axial_index]
    relative = box.position[:, transverse] - centre_array[transverse]
    return box.select(np.sum(relative**2, axis=1) <= radius**2 + np.finfo(float).eps)
