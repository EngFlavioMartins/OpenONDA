"""Taylor--Green vortex particle initialization."""

from __future__ import annotations

import numpy as np

from ..data import ParticleDistribution, VortexParticleDistribution, _attributed_distribution
from ._common import validate_viscosity


def initialize_taylor_green_vortex(
    distribution: ParticleDistribution,
    *,
    box_size: float,
    kinematic_viscosity: float,
    time: float = 0.0,
) -> VortexParticleDistribution:
    """Attribute the canonical periodic Taylor--Green velocity/vorticity field."""
    box_size = float(box_size)
    viscosity = validate_viscosity(kinematic_viscosity)
    if not np.isfinite(box_size) or box_size <= 0.0:
        raise ValueError("box_size must be finite and positive")
    if not np.isfinite(time) or time < 0.0:
        raise ValueError("time must be finite and non-negative")
    wave_number = 2.0 * np.pi / box_size
    phase = distribution.position * wave_number
    decay = np.exp(-2.0 * viscosity * time * wave_number**2)
    x, y, z = phase.T
    velocity = np.zeros_like(distribution.position)
    velocity[:, 0] = decay * np.sin(x) * np.cos(y) * np.cos(z)
    velocity[:, 1] = -decay * np.cos(x) * np.sin(y) * np.cos(z)
    vorticity = np.empty_like(distribution.position)
    vorticity[:, 0] = -decay * wave_number * np.cos(x) * np.sin(y) * np.sin(z)
    vorticity[:, 1] = -decay * wave_number * np.sin(x) * np.cos(y) * np.sin(z)
    vorticity[:, 2] = 2.0 * decay * wave_number * np.sin(x) * np.sin(y) * np.cos(z)
    return _attributed_distribution(
        distribution,
        velocity=velocity,
        vortex_strength=vorticity * distribution.particle_volume[:, None],
        kinematic_viscosity=viscosity,
    )
