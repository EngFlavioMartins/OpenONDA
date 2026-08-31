"""Toroidal particle-distribution construction object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..data import ParticleDistribution
from ._common import validate_spacing

Axis = Literal["x", "y", "z"]


@dataclass(frozen=True, slots=True)
class ToroidalDistribution:
    """Hexagonal-cross-section particle cloud around a circular centreline.

    ``ring_radius`` exceeds positive ``tube_radius``. ``spacing`` is the target
    transverse particle spacing; returned volume weights are cylindrical-cell
    quadrature weights in cubic length units.
    """

    ring_radius: float
    tube_radius: float
    spacing: float
    core_radius_ratio: float
    centre: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis: Axis = "x"

    def build(self) -> ParticleDistribution:
        """Build immutable toroidal geometry and curved-cell quadrature."""
        spacing, ratio = validate_spacing(self.spacing, self.core_radius_ratio)
        if not np.isfinite(self.ring_radius) or self.ring_radius <= 0.0:
            raise ValueError("ring_radius must be finite and positive")
        if not np.isfinite(self.tube_radius) or self.tube_radius <= 0.0:
            raise ValueError("tube_radius must be finite and positive")
        if self.tube_radius >= self.ring_radius:
            raise ValueError("tube_radius must be smaller than ring_radius")
        if self.axis not in {"x", "y", "z"}:
            raise ValueError("axis must be 'x', 'y', or 'z'")
        centre = np.asarray(self.centre, dtype=float)
        if centre.shape != (3,) or not np.all(np.isfinite(centre)):
            raise ValueError("centre must contain three finite coordinates")
        row_spacing = np.sqrt(3.0) * spacing / 2.0
        cross_section: list[tuple[float, float]] = []
        for row in range(
            -int(np.ceil(self.tube_radius / row_spacing)),
            int(np.ceil(self.tube_radius / row_spacing)) + 1,
        ):
            radial_offset = row * row_spacing
            max_column = int(np.ceil(self.tube_radius / spacing)) + abs(row)
            for column in range(-max_column, max_column + 1):
                axial_offset = spacing * (column + 0.5 * row)
                if (
                    axial_offset**2 + radial_offset**2
                    <= (self.tube_radius + 16 * np.finfo(float).eps) ** 2
                ):
                    cross_section.append((axial_offset, radial_offset))
        offsets = np.asarray(cross_section, dtype=float)
        azimuth_count = max(
            8, int(np.ceil(2.0 * np.pi * (self.ring_radius + self.tube_radius) / spacing))
        )
        azimuth_count += (-azimuth_count) % 4
        azimuth = 2.0 * np.pi * np.arange(azimuth_count) / azimuth_count
        cosine, sine = (
            np.tile(np.cos(azimuth), len(offsets)),
            np.tile(np.sin(azimuth), len(offsets)),
        )
        axial, radial = (
            np.repeat(offsets[:, 0], azimuth_count),
            np.repeat(self.ring_radius + offsets[:, 1], azimuth_count),
        )
        local = np.empty((len(axial), 3), dtype=float)
        if self.axis == "x":
            local[:, 0], local[:, 1], local[:, 2] = axial, radial * cosine, radial * sine
        elif self.axis == "y":
            local[:, 0], local[:, 1], local[:, 2] = radial * sine, axial, radial * cosine
        else:
            local[:, 0], local[:, 1], local[:, 2] = radial * cosine, radial * sine, axial
        weights = np.sqrt(3.0) * spacing**2 / 2.0 * radial * (2.0 * np.pi / azimuth_count)
        return ParticleDistribution(
            position=local + centre,
            core_radius=np.full(len(local), ratio * spacing),
            particle_volume=weights,
            spacing=spacing,
        )
