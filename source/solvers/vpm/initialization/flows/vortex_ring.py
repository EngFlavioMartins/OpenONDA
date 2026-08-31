"""Gaussian vortex-ring construction object."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..data import ParticleDistribution, VortexParticleSet, attributed_particle_set
from ..disturbances import WidnallDisturbance
from ._common import (
    represented_core_radius_squared,
    transverse_basis,
    unit_vector,
    validate_viscosity,
    vector3,
)
from ._shared import (
    DistributionSource,
    ParticleCoreCompensation,
    constant_group_id,
    resolve_distribution,
)


@dataclass(frozen=True, slots=True)
class VortexRing:
    """Gaussian vortex ring with optional Widnall centreline disturbance.

    ``radius`` and ``vortex_core_radius`` are positive lengths; ``circulation``
    is nonzero. Its initial velocity is zero, so induced velocity remains
    refreshed by the solver; analytical velocity is not presently available.
    """

    radius: float
    vortex_core_radius: float
    circulation: float
    kinematic_viscosity: float
    distribution: DistributionSource = None
    centre: Sequence[float] = (0.0, 0.0, 0.0)
    axis: Sequence[float] = (1.0, 0.0, 0.0)
    disturbance: WidnallDisturbance | None = None
    core_compensation: ParticleCoreCompensation | None = None
    group_id: int | None = None

    def build(self, distribution: ParticleDistribution | None = None) -> VortexParticleSet:
        """Attribute this ring to geometry and return immutable particle fields."""
        geometry = resolve_distribution(distribution, self.distribution)
        centre, axis = vector3(self.centre, "centre"), unit_vector(self.axis, "axis")
        first, second = transverse_basis(axis)
        radius, circulation, viscosity = (
            float(self.radius),
            float(self.circulation),
            validate_viscosity(self.kinematic_viscosity),
        )
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius must be finite and positive")
        if not np.isfinite(circulation) or circulation == 0.0:
            raise ValueError("circulation must be finite and non-zero")
        core_squared = represented_core_radius_squared(
            self.vortex_core_radius, geometry.core_radius, compensation=self.core_compensation
        )
        relative = geometry.position - centre
        axial, first_position, second_position = (
            relative @ axis,
            relative @ first,
            relative @ second,
        )
        radial, azimuth = (
            np.hypot(first_position, second_position),
            np.arctan2(second_position, first_position),
        )
        if self.disturbance is None:
            centreline_radius, slope = np.full(len(geometry), radius), np.zeros(len(geometry))
        else:
            centreline_radius, slope = self.disturbance.centreline(azimuth, radius)
        magnitude = (
            circulation
            / (np.pi * core_squared)
            * np.exp(-((radial - centreline_radius) ** 2 + axial**2) / core_squared)
        )
        cosine, sine = np.cos(azimuth), np.sin(azimuth)
        tangent, radial_direction = (
            -sine[:, None] * first + cosine[:, None] * second,
            cosine[:, None] * first + sine[:, None] * second,
        )
        away = radial > np.finfo(float).eps
        radial_vorticity = np.zeros(len(geometry))
        radial_vorticity[away] = magnitude[away] * slope[away] / radial[away]
        strength = (
            magnitude[:, None] * tangent + radial_vorticity[:, None] * radial_direction
        ) * geometry.particle_volume[:, None]
        represented = np.sum(
            np.einsum("ij,ij->i", strength[away], tangent[away]) / radial[away]
        ) / (2.0 * np.pi)
        if abs(represented) <= np.finfo(float).tiny:
            raise ValueError("particle distribution represents zero vortex-ring circulation")
        return attributed_particle_set(
            geometry,
            velocity=np.zeros_like(geometry.position),
            vortex_strength=strength * circulation / represented,
            kinematic_viscosity=viscosity,
            group_id=constant_group_id(self.group_id, len(geometry)),
        )
