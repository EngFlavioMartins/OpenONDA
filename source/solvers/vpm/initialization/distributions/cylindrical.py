"""Cylindrical particle-distribution construction object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..data import ParticleDistribution
from ._common import validate_spacing
from .rectangular import RectangularDistribution

Axis = Literal["x", "y", "z"]


@dataclass(frozen=True, slots=True)
class CylindricalDistribution:
    """Cartesian midpoint cells clipped to a solid cylinder.

    ``radius`` and ``length`` are positive length units. Boundary-cell volumes
    are normalized to the cylinder's analytic volume, preserving integrated
    quadrature rather than retaining an unqualified ``h³`` after clipping.
    """

    radius: float
    length: float
    spacing: float
    core_radius_ratio: float
    centre: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis: Axis = "z"

    def build(self) -> ParticleDistribution:
        """Build immutable cylinder geometry with volume-conserving weights."""
        spacing, ratio = validate_spacing(self.spacing, self.core_radius_ratio)
        if not np.isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError("radius must be finite and positive")
        if not np.isfinite(self.length) or self.length <= 0.0:
            raise ValueError("length must be finite and positive")
        if self.axis not in {"x", "y", "z"}:
            raise ValueError("axis must be 'x', 'y', or 'z'")
        centre = np.asarray(self.centre, dtype=float)
        if centre.shape != (3,) or not np.all(np.isfinite(centre)):
            raise ValueError("centre must contain three finite coordinates")
        axial = {"x": 0, "y": 1, "z": 2}[self.axis]
        bounds = [(centre[i] - self.radius, centre[i] + self.radius) for i in range(3)]
        bounds[axial] = (centre[axial] - 0.5 * self.length, centre[axial] + 0.5 * self.length)
        box = RectangularDistribution(
            bounds=bounds, spacing=spacing, core_radius_ratio=ratio
        ).build()
        transverse = [i for i in range(3) if i != axial]
        relative = box.position[:, transverse] - centre[transverse]
        result = box.select(np.sum(relative**2, axis=1) <= self.radius**2 + np.finfo(float).eps)
        analytic_volume = np.pi * self.radius**2 * self.length
        weights = np.full(len(result), analytic_volume / len(result))
        return ParticleDistribution(
            position=result.position,
            core_radius=result.core_radius,
            particle_volume=weights,
            spacing=result.spacing,
        )
