"""Rectangular particle-distribution construction objects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data import ParticleDistribution
from ._common import Bounds3D, centred_coordinates, validate_bounds, validate_spacing


@dataclass(frozen=True, slots=True)
class RectangularDistribution:
    """Cartesian particle lattice within finite ``((min, max),) * 3`` bounds.

    ``spacing`` is positive and in length units. ``core_radius_ratio`` is the
    positive ``sigma / h`` ratio. The build result has nominal ``h³`` cell
    volumes and immutable arrays.
    """

    bounds: Bounds3D
    spacing: float
    core_radius_ratio: float

    def build(self) -> ParticleDistribution:
        """Build immutable lattice geometry and midpoint quadrature."""
        spacing, ratio = validate_spacing(self.spacing, self.core_radius_ratio)
        limits = validate_bounds(self.bounds)
        coordinates = [centred_coordinates(*limits[axis], spacing) for axis in range(3)]
        position = np.stack(np.meshgrid(*coordinates, indexing="ij"), axis=-1).reshape(-1, 3)
        return ParticleDistribution(
            position=position,
            core_radius=np.full(len(position), ratio * spacing),
            particle_volume=np.full(len(position), spacing**3),
            spacing=spacing,
        )


@dataclass(frozen=True, slots=True)
class NoisyRectangularDistribution:
    """Bounded jitter of a rectangular lattice with a reproducible random seed.

    ``noise_fraction`` lies in ``[0, 1]`` and controls jitter relative to one
    nominal cell. Jitter is reflected at boundaries to avoid piled-up points.
    """

    bounds: Bounds3D
    spacing: float
    core_radius_ratio: float
    noise_fraction: float = 0.3
    seed: int | None = None

    def build(self) -> ParticleDistribution:
        """Build bounded, reproducibly jittered immutable lattice geometry."""
        if not np.isfinite(self.noise_fraction) or not 0.0 <= self.noise_fraction <= 1.0:
            raise ValueError("noise_fraction must be finite and between zero and one")
        base = RectangularDistribution(
            bounds=self.bounds, spacing=self.spacing, core_radius_ratio=self.core_radius_ratio
        ).build()
        limits = validate_bounds(self.bounds)
        position = np.array(base.position, copy=True)
        position += np.random.default_rng(self.seed).uniform(
            -0.5 * self.noise_fraction * base.spacing,
            0.5 * self.noise_fraction * base.spacing,
            size=position.shape,
        )
        lower, upper = limits[:, 0], limits[:, 1]
        position = np.where(position < lower, 2.0 * lower - position, position)
        position = np.where(position > upper, 2.0 * upper - position, position)
        return ParticleDistribution(
            position=position,
            core_radius=base.core_radius,
            particle_volume=base.particle_volume,
            spacing=base.spacing,
        )
