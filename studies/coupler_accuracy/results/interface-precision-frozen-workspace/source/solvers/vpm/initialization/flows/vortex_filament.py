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
    """Configure a straight or sinusoidally displaced Gaussian filament.

    Parameters
    ----------
    vortex_core_radius : float
        Positive physical Gaussian core radius in m, distinct from particle
        regularization radii ``sigma``.
    circulation : float
        Non-zero signed filament circulation in m²/s.
    kinematic_viscosity : float
        Non-negative molecular viscosity in m²/s.
    distribution : ParticleDistribution, distribution builder, or None
        Geometry/quadrature used by :meth:`build`; ``None`` requires an
        explicit build argument.
    centre : sequence[float], shape (3,), default=(0, 0, 0)
        Cartesian point on the undisturbed filament axis in m.
    direction : sequence[float], shape (3,), default=(0, 0, 1)
        Non-zero global filament direction, normalized internally.
    disturbance : FilamentDisturbance or None, default=None
        Optional sinusoidal transverse displacement and tangent correction.
    core_compensation : ParticleCoreCompensation or None, default=None
        Optional correction for particle-core regularization.
    group_id : int or None, default=None
        Optional uniform int32 particle-group label.
    tail_minimum_relative_strength : float or None, default=None
        If set, retain particles whose strength magnitude is at least this
        fraction of the peak; valid range is ``[0, 1)``.
    tail_circulation_per_length : float or None, default=None
        Optional signed target circulation per represented length in m²/s.
        Requires tail filtering and ``tail_represented_length``.
    tail_represented_length : float or None, default=None
        Positive axial length in m used to restore circulation after filtering.
    tail_direction : sequence[float] or None, default=None
        Direction used by tail circulation recovery; ``None`` uses
        ``direction``.

    Notes
    -----
    The Gaussian vorticity is integrated as particle-strength vectors
    ``Gamma=omega*V`` in m³/s. Initial velocity is explicitly zero because the
    solver owns induced-field evaluation.
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
        """Attribute filament vorticity to immutable particle geometry.

        Parameters
        ----------
        distribution : ParticleDistribution or None, default=None
            Explicit geometry overriding the configured distribution.

        Returns
        -------
        VortexParticleSet
            Solver-ready immutable arrays with ``(N, 3)`` vectors and ``(N,)``
            scalar fields in the units documented by :class:`VortexParticleSet`.

        Raises
        ------
        ValueError
            If geometry/parameters are invalid, numerical cores cannot
            represent the physical core, or tail filtering/recovery is
            inconsistent or removes every particle.
        """
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
