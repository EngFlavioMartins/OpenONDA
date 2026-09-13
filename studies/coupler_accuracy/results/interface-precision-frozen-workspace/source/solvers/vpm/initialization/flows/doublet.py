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
    """Configure a canonical three-dimensional vortex-doublet field.

    Parameters
    ----------
    centre : sequence[float], shape (3,)
        Cartesian doublet centre in m.
    direction : sequence[float], shape (3,)
        Non-zero orientation vector, normalized internally.
    strength : float
        Finite doublet amplitude in m³/s. Division by ``r**3`` gives the
        analytic vorticity scale in 1/s.
    kinematic_viscosity : float
        Non-negative molecular viscosity in m²/s.
    distribution : ParticleDistribution, distribution builder, or None
        Geometry/quadrature used to discretize the field. ``None`` requires an
        explicit geometry in :meth:`build`.

    Notes
    -----
    Pointwise vorticity is sampled analytically and integrated as
    ``Gamma_i=omega_i*V_i`` in m³/s. The singular centre is bounded at machine
    epsilon for finite construction. Initial velocity is zero and is refreshed
    by the solver's induction backend.
    """

    centre: Sequence[float]
    direction: Sequence[float]
    strength: float
    kinematic_viscosity: float
    distribution: DistributionSource = None

    def build(self, distribution: ParticleDistribution | None = None) -> VortexParticleSet:
        """Attribute the doublet field to immutable particle geometry.

        Parameters
        ----------
        distribution : ParticleDistribution or None, default=None
            Explicit geometry overriding the configured distribution.

        Returns
        -------
        VortexParticleSet
            Immutable solver-ready particle fields using SI array conventions.

        Raises
        ------
        ValueError
            If geometry is absent, centre/direction is invalid, strength is
            non-finite, or viscosity is negative.
        """
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
