"""Vortex-doublet particle initialization."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..data import ParticleDistribution, VortexParticleDistribution, _attributed_distribution
from ._common import unit_vector, validate_viscosity, vector3


def initialize_vortex_doublet(
    distribution: ParticleDistribution,
    *,
    centre: Sequence[float],
    direction: Sequence[float],
    strength: float,
    kinematic_viscosity: float,
) -> VortexParticleDistribution:
    """Attribute the canonical three-dimensional vortex-doublet field."""
    centre_array = vector3(centre, "centre")
    direction_array = unit_vector(direction, "direction")
    strength = float(strength)
    viscosity = validate_viscosity(kinematic_viscosity)
    if not np.isfinite(strength):
        raise ValueError("strength must be finite")
    relative = distribution.position - centre_array
    distance_squared = np.einsum("ij,ij->i", relative, relative)
    safe_distance = np.maximum(distance_squared, np.finfo(float).eps) ** 2.5
    projection = relative @ direction_array
    vorticity = (-strength / (4.0 * np.pi * safe_distance))[:, None] * (
        distance_squared[:, None] * direction_array - 3.0 * relative * projection[:, None]
    )
    return _attributed_distribution(
        distribution,
        velocity=np.zeros_like(distribution.position),
        vortex_strength=vorticity * distribution.particle_volume[:, None],
        kinematic_viscosity=viscosity,
    )
