"""Shared validation and exact-spacing lattice helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

Bounds3D = Sequence[Sequence[float]]


def validate_spacing(spacing: float, core_radius_ratio: float) -> tuple[float, float]:
    """Validate particle spacing and the dimensionless ``sigma/h`` ratio.

    Parameters
    ----------
    spacing : float
        Particle spacing ``h`` in metres.
    core_radius_ratio : float
        Core radius divided by spacing, ``sigma/h``.

    Returns
    -------
    spacing, core_radius_ratio : float
        Validated floating-point values.

    Raises
    ------
    ValueError
        If either value is non-finite or not strictly positive.
    """
    spacing = float(spacing)
    core_radius_ratio = float(core_radius_ratio)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("spacing must be finite and positive")
    if not np.isfinite(core_radius_ratio) or core_radius_ratio <= 0.0:
        raise ValueError("core_radius_ratio (sigma/h) must be finite and positive")
    return spacing, core_radius_ratio


def validate_bounds(bounds: Bounds3D) -> np.ndarray:
    """Validate and copy Cartesian bounds.

    Parameters
    ----------
    bounds : sequence, shape (3, 2)
        ``((xmin, xmax), (ymin, ymax), (zmin, zmax))`` in metres.

    Returns
    -------
    ndarray, shape (3, 2)
        Float bounds copy.

    Raises
    ------
    ValueError
        If the shape, finiteness, or lower/upper ordering is invalid.
    """
    values = np.asarray(bounds, dtype=float)
    if values.shape != (3, 2):
        raise ValueError("bounds must contain ((xmin, xmax), (ymin, ymax), (zmin, zmax))")
    if not np.all(np.isfinite(values)):
        raise ValueError("bounds must contain only finite values")
    if np.any(values[:, 1] < values[:, 0]):
        raise ValueError("each upper bound must be greater than or equal to its lower bound")
    return values


def centred_coordinates(lower: float, upper: float, spacing: float) -> np.ndarray:
    """Return an exactly spaced one-dimensional lattice centred in bounds.

    Parameters
    ----------
    lower, upper : float
        Inclusive coordinate limits in metres.
    spacing : float
        Requested lattice spacing in metres.

    Returns
    -------
    ndarray, shape (N,)
        Coordinates with spacing exactly equal to ``spacing``; the sequence is
        centred in ``[lower, upper]``.
    """
    width = upper - lower
    count = max(1, int(np.floor(width / spacing + 1.0e-12)) + 1)
    occupied_width = (count - 1) * spacing
    start = 0.5 * (lower + upper - occupied_width)
    return start + spacing * np.arange(count, dtype=float)
