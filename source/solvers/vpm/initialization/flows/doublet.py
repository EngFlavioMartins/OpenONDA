"""Vortex-doublet construction object."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..data import ParticleDistribution, VortexParticleSet, attributed_particle_set
from ._common import unit_vector, validate_viscosity, vector3
from ._shared import DistributionSource, resolve_distribution


@dataclass(frozen=True, slots=True)
class VortexDoublet:
    """Canonical three-dimensional vortex doublet.

    ``strength`` is finite, ``direction`` is nonzero, and initial velocity uses
    explicitly zero because the solver owns induced-field evaluation.
    """

    centre: Sequence[float]
    direction: Sequence[float]
    strength: float
    kinematic_viscosity: float
    distribution: DistributionSource = None

    def build(self, distribution: ParticleDistribution | None = None) -> VortexParticleSet:
        """Attribute this doublet to geometry and return immutable fields."""
        geometry = resolve_distribution(distribution, self.distribution)
        centre, direction = vector3(self.centre, "centre"), unit_vector(self.direction, "direction")
        strength, viscosity = float(self.strength), validate_viscosity(self.kinematic_viscosity)
        if not np.isfinite(strength):
            raise ValueError("strength must be finite")
        relative = geometry.position - centre
        distance_squared = np.einsum("ij,ij->i", relative, relative)
        safe_distance = np.maximum(distance_squared, np.finfo(float).eps) ** 2.5
        projection = relative @ direction
        vorticity = (-strength / (4.0 * np.pi * safe_distance))[:, None] * (
            distance_squared[:, None] * direction - 3.0 * relative * projection[:, None]
        )
        return attributed_particle_set(
            geometry,
            velocity=np.zeros_like(geometry.position),
            vortex_strength=vorticity * geometry.particle_volume[:, None],
            kinematic_viscosity=viscosity,
        )
