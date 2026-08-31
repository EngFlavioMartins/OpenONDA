"""Straight Gaussian/Lamb--Oseen vortex-filament construction object."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..data import ParticleDistribution, VortexParticleSet, attributed_particle_set
from ..disturbances import FilamentDisturbance
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
from .filament_tail import filter_tail


@dataclass(frozen=True, slots=True)
class VortexFilament:
    """Straight or sinusoidally displaced Gaussian filament.

    ``circulation`` is nonzero; lengths are positive. Its initial particle
    velocity is explicitly zero because induced velocity is solver-evaluated.
    """

    vortex_core_radius: float
    circulation: float
    kinematic_viscosity: float
    distribution: DistributionSource = None
    centre: Sequence[float] = (0.0, 0.0, 0.0)
    direction: Sequence[float] = (0.0, 0.0, 1.0)
    disturbance: FilamentDisturbance | None = None
    core_compensation: ParticleCoreCompensation | None = None
    group_id: int | None = None
    tail_minimum_relative_strength: float | None = None
    tail_circulation_per_length: float | None = None
    tail_represented_length: float | None = None
    tail_direction: Sequence[float] | None = None

    def build(self, distribution: ParticleDistribution | None = None) -> VortexParticleSet:
        """Attribute this filament to geometry and return immutable fields."""
        geometry = resolve_distribution(distribution, self.distribution)
        centre, direction = vector3(self.centre, "centre"), unit_vector(self.direction, "direction")
        circulation, viscosity = (
            float(self.circulation),
            validate_viscosity(self.kinematic_viscosity),
        )
        if not np.isfinite(circulation) or circulation == 0.0:
            raise ValueError("circulation must be finite and non-zero")
        core_squared = represented_core_radius_squared(
            self.vortex_core_radius, geometry.core_radius, compensation=self.core_compensation
        )
        relative = geometry.position - centre
        axial = relative @ direction
        transverse = relative - axial[:, None] * direction
        tangent = np.broadcast_to(direction, geometry.position.shape).copy()
        if self.disturbance is not None:
            first, second = transverse_basis(direction)
            polarization = (
                np.cos(self.disturbance.polarization_angle) * first
                + np.sin(self.disturbance.polarization_angle) * second
            )
            argument = 2.0 * np.pi / self.disturbance.wavelength * axial + self.disturbance.phase
            transverse -= (self.disturbance.amplitude * np.sin(argument))[:, None] * polarization
            tangent += (
                self.disturbance.amplitude
                * 2.0
                * np.pi
                / self.disturbance.wavelength
                * np.cos(argument)
            )[:, None] * polarization
        radial_squared = np.einsum("ij,ij->i", transverse, transverse)
        magnitude = circulation / (np.pi * core_squared) * np.exp(-radial_squared / core_squared)
        particles = attributed_particle_set(
            geometry,
            velocity=np.zeros_like(geometry.position),
            vortex_strength=magnitude[:, None] * tangent * geometry.particle_volume[:, None],
            kinematic_viscosity=viscosity,
            group_id=constant_group_id(self.group_id, len(geometry)),
        )
        if self.tail_minimum_relative_strength is None:
            if (
                self.tail_circulation_per_length is not None
                or self.tail_represented_length is not None
            ):
                raise ValueError("tail fields require tail_minimum_relative_strength")
            return particles
        return filter_tail(
            particles,
            minimum_relative_strength=self.tail_minimum_relative_strength,
            circulation_per_length=self.tail_circulation_per_length,
            represented_length=self.tail_represented_length,
            direction=direction if self.tail_direction is None else self.tail_direction,
        )
