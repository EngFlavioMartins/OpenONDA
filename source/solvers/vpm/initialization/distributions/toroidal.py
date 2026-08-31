"""Purely geometric toroidal particle distributions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..data import ParticleDistribution
from ._common import validate_spacing


def create_toroidal_distribution(
    *,
    ring_radius: float,
    tube_radius: float,
    spacing: float,
    core_radius_ratio: float,
    centre: Sequence[float] = (0.0, 0.0, 0.0),
    axis: str = "x",
) -> ParticleDistribution:
    """Create a circular toroidal cloud with no attributed flow disturbance."""
    spacing, core_radius_ratio = validate_spacing(spacing, core_radius_ratio)
    ring_radius = float(ring_radius)
    tube_radius = float(tube_radius)
    if not np.isfinite(ring_radius) or ring_radius <= 0.0:
        raise ValueError("ring_radius must be finite and positive")
    if not np.isfinite(tube_radius) or tube_radius <= 0.0:
        raise ValueError("tube_radius must be finite and positive")
    if tube_radius >= ring_radius:
        raise ValueError("tube_radius must be smaller than ring_radius")
    if axis not in {"x", "y", "z"}:
        raise ValueError("axis must be 'x', 'y', or 'z'")
    centre_array = np.asarray(centre, dtype=float)
    if centre_array.shape != (3,) or not np.all(np.isfinite(centre_array)):
        raise ValueError("centre must contain three finite coordinates")

    row_spacing = np.sqrt(3.0) * spacing / 2.0
    max_row = int(np.ceil(tube_radius / row_spacing))
    max_column = int(np.ceil(tube_radius / spacing)) + max_row
    cross_section: list[tuple[float, float]] = []
    cutoff_squared = (tube_radius + 16.0 * np.finfo(float).eps) ** 2
    for row in range(-max_row, max_row + 1):
        radial_offset = row * row_spacing
        for column in range(-max_column, max_column + 1):
            axial_offset = spacing * (column + 0.5 * row)
            if axial_offset**2 + radial_offset**2 <= cutoff_squared:
                cross_section.append((axial_offset, radial_offset))

    offsets = np.asarray(cross_section, dtype=float)
    outer_circumference = 2.0 * np.pi * (ring_radius + tube_radius)
    azimuth_count = max(8, int(np.ceil(outer_circumference / spacing)))
    azimuth_count += (-azimuth_count) % 4
    azimuth = 2.0 * np.pi * np.arange(azimuth_count) / azimuth_count
    cosine = np.tile(np.cos(azimuth), len(offsets))
    sine = np.tile(np.sin(azimuth), len(offsets))
    axial = np.repeat(offsets[:, 0], azimuth_count)
    radial = np.repeat(ring_radius + offsets[:, 1], azimuth_count)

    local = np.empty((len(axial), 3), dtype=float)
    if axis == "x":
        local[:, 0] = axial
        local[:, 1] = radial * cosine
        local[:, 2] = radial * sine
    elif axis == "y":
        local[:, 0] = radial * sine
        local[:, 1] = axial
        local[:, 2] = radial * cosine
    else:
        local[:, 0] = radial * cosine
        local[:, 1] = radial * sine
        local[:, 2] = axial

    cell_area = np.sqrt(3.0) * spacing**2 / 2.0
    particle_volume = cell_area * radial * (2.0 * np.pi / azimuth_count)
    return ParticleDistribution(
        position=local + centre_array,
        core_radius=np.full(len(local), core_radius_ratio * spacing),
        particle_volume=particle_volume,
        spacing=spacing,
    )
