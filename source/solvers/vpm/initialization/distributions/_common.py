"""Shared validation and exact-spacing lattice helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

Bounds3D = Sequence[Sequence[float]]


def validate_spacing(spacing: float, core_radius_ratio: float) -> tuple[float, float]:
    spacing = float(spacing)
    core_radius_ratio = float(core_radius_ratio)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("spacing must be finite and positive")
    if not np.isfinite(core_radius_ratio) or core_radius_ratio <= 0.0:
        raise ValueError("core_radius_ratio (sigma/h) must be finite and positive")
    return spacing, core_radius_ratio


def validate_bounds(bounds: Bounds3D) -> np.ndarray:
    values = np.asarray(bounds, dtype=float)
    if values.shape != (3, 2):
        raise ValueError("bounds must contain ((xmin, xmax), (ymin, ymax), (zmin, zmax))")
    if not np.all(np.isfinite(values)):
        raise ValueError("bounds must contain only finite values")
    if np.any(values[:, 1] < values[:, 0]):
        raise ValueError("each upper bound must be greater than or equal to its lower bound")
    return values


def centred_coordinates(lower: float, upper: float, spacing: float) -> np.ndarray:
    """Return points at exact spacing, centered within inclusive limits."""
    width = upper - lower
    count = max(1, int(np.floor(width / spacing + 1.0e-12)) + 1)
    occupied_width = (count - 1) * spacing
    start = 0.5 * (lower + upper - occupied_width)
    return start + spacing * np.arange(count, dtype=float)


def axis_vector(axis: str) -> np.ndarray:
    if axis not in {"x", "y", "z"}:
        raise ValueError("axis must be 'x', 'y', or 'z'")
    result = np.zeros(3)
    result[{"x": 0, "y": 1, "z": 2}[axis]] = 1.0
    return result
