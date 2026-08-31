"""Triangular-prism particle-distribution construction object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..data import ParticleDistribution
from ._common import Bounds3D, centred_coordinates, validate_bounds, validate_spacing

Axis = Literal["x", "y", "z"]


@dataclass(frozen=True, slots=True)
class TriangularPrismDistribution:
    """Triangular transverse lattice extruded along ``axis``.

    Positive ``spacing`` controls transverse nodes; ``axial_spacing`` defaults
    to it. Bounds are inclusive and finite. Particle weights equal the regular
    triangular cell area times axial spacing.
    """

    bounds: Bounds3D
    spacing: float
    core_radius_ratio: float
    axis: Axis = "z"
    axial_spacing: float | None = None

    def build(self) -> ParticleDistribution:
        """Build immutable triangular-prism geometry and quadrature."""
        spacing, ratio = validate_spacing(self.spacing, self.core_radius_ratio)
        axial_spacing = spacing if self.axial_spacing is None else float(self.axial_spacing)
        if not np.isfinite(axial_spacing) or axial_spacing <= 0.0:
            raise ValueError("axial_spacing must be finite and positive")
        if self.axis not in {"x", "y", "z"}:
            raise ValueError("axis must be 'x', 'y', or 'z'")
        limits = validate_bounds(self.bounds)
        axial_index = {"x": 0, "y": 1, "z": 2}[self.axis]
        transverse = [i for i in range(3) if i != axial_index]
        first = centred_coordinates(*limits[transverse[0]], spacing)
        second = centred_coordinates(*limits[transverse[1]], np.sqrt(3.0) * spacing / 2.0)
        plane: list[tuple[float, float]] = []
        for row, second_value in enumerate(second):
            shifted = first + (0.5 * spacing if row % 2 else 0.0)
            shifted = shifted[
                (shifted >= limits[transverse[0], 0] - np.finfo(float).eps)
                & (shifted <= limits[transverse[0], 1] + np.finfo(float).eps)
            ]
            plane.extend((first_value, second_value) for first_value in shifted)
        axial = centred_coordinates(*limits[axial_index], axial_spacing)
        position = np.empty((len(plane) * len(axial), 3), dtype=float)
        for orbit, (first_value, second_value) in enumerate(plane):
            part = slice(orbit * len(axial), (orbit + 1) * len(axial))
            (
                position[part, axial_index],
                position[part, transverse[0]],
                position[part, transverse[1]],
            ) = axial, first_value, second_value
        weights = np.full(len(position), np.sqrt(3.0) * spacing**2 / 2.0 * axial_spacing)
        return ParticleDistribution(
            position=position,
            core_radius=np.full(len(position), ratio * spacing),
            particle_volume=weights,
            spacing=spacing,
        )
