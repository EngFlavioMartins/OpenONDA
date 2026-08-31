"""Taylor--Green vortex construction object."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data import ParticleDistribution, VortexParticleSet, attributed_particle_set
from ._common import validate_viscosity
from ._shared import DistributionSource, InitialVelocity, resolve_distribution


@dataclass(frozen=True, slots=True)
class TaylorGreenVortex:
    """Periodic Taylor--Green velocity and vorticity field.

    ``box_size`` is positive. ``ANALYTICAL`` velocity is the default because
    the exact velocity is known; choose ``ZERO`` to defer all field evaluation.
    """

    box_size: float
    kinematic_viscosity: float
    distribution: DistributionSource = None
    time: float = 0.0
    initial_velocity: InitialVelocity = InitialVelocity.ANALYTICAL

    def build(self, distribution: ParticleDistribution | None = None) -> VortexParticleSet:
        """Attribute exact Taylor--Green fields to immutable geometry."""
        geometry = resolve_distribution(distribution, self.distribution)
        box_size, viscosity, time = (
            float(self.box_size),
            validate_viscosity(self.kinematic_viscosity),
            float(self.time),
        )
        if not np.isfinite(box_size) or box_size <= 0.0:
            raise ValueError("box_size must be finite and positive")
        if not np.isfinite(time) or time < 0.0:
            raise ValueError("time must be finite and non-negative")
        wave_number = 2.0 * np.pi / box_size
        x, y, z = (geometry.position * wave_number).T
        decay = np.exp(-2.0 * viscosity * time * wave_number**2)
        velocity = np.zeros_like(geometry.position)
        if self.initial_velocity is InitialVelocity.ANALYTICAL:
            velocity[:, 0], velocity[:, 1] = (
                decay * np.sin(x) * np.cos(y) * np.cos(z),
                -decay * np.cos(x) * np.sin(y) * np.cos(z),
            )
        elif self.initial_velocity is not InitialVelocity.ZERO:
            raise ValueError(f"unsupported initial velocity: {self.initial_velocity}")
        vorticity = np.column_stack(
            (
                -decay * wave_number * np.cos(x) * np.sin(y) * np.sin(z),
                -decay * wave_number * np.sin(x) * np.cos(y) * np.sin(z),
                2.0 * decay * wave_number * np.sin(x) * np.sin(y) * np.cos(z),
            )
        )
        return attributed_particle_set(
            geometry,
            velocity=velocity,
            vortex_strength=vorticity * geometry.particle_volume[:, None],
            kinematic_viscosity=viscosity,
        )
