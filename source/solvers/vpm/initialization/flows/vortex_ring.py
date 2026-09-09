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
    """Configure a Gaussian vortex ring on supplied particle geometry.

    Parameters
    ----------
    radius : float
        Positive centreline radius ``R`` in m.
    vortex_core_radius : float
        Positive physical Gaussian core radius in m. This is distinct from
        each numerical particle core ``sigma``.
    circulation : float
        Signed scalar filament circulation in m²/s. Its sign sets the
        tangential vorticity orientation around ``axis``.
    kinematic_viscosity : float
        Non-negative molecular viscosity in m²/s assigned to every particle.
    distribution : ParticleDistribution, distribution builder, or None
        Geometry/quadrature used by :meth:`build`. ``None`` requires an
        explicit distribution argument at build time.
    centre : sequence[float], shape (3,), default=(0, 0, 0)
        Cartesian ring centre in m.
    axis : sequence[float], shape (3,), default=(1, 0, 0)
        Non-zero normal to the ring plane; it is normalized internally.
    disturbance : WidnallDisturbance or None, default=None
        Optional radial/axial azimuthal centreline perturbation.
    core_compensation : ParticleCoreCompensation or None, default=None
        Optional correction separating the requested physical core from the
        regularization already supplied by particle cores.
    group_id : int or None, default=None
        Optional int32 label assigned uniformly to the generated particles.

    Notes
    -----
    The attributed continuum vorticity is Gaussian in the tube cross-section
    and is integrated as ``Gamma_i = omega_i * V_i`` in m³/s. Strengths are
    rescaled so their discrete circulation represents the requested scalar
    ``circulation``. Initial particle velocity is zero because induced velocity
    is refreshed by :class:`VPMSolver`.

    Examples
    --------
    >>> ring = VortexRing(
    ...     radius=1.0, vortex_core_radius=0.15, circulation=1.0,
    ...     kinematic_viscosity=1e-4, distribution=distribution,
    ... )
    >>> particles = ring.build()
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
        """Attribute ring vorticity to geometry and return solver-ready fields.

        Parameters
        ----------
        distribution : ParticleDistribution or None, default=None
            Explicit immutable geometry. When supplied it takes precedence over
            the constructor's ``distribution``.

        Returns
        -------
        VortexParticleSet
            Immutable arrays: position/velocity/strength have shape ``(N, 3)``
            in m, m/s, and m³/s; radius, volume, and viscosity have shape
            ``(N,)`` in m, m³, and m²/s.

        Raises
        ------
        ValueError
            If geometry is missing, a physical parameter/vector is invalid,
            particle cores cannot represent the requested core, or the finite
            geometry represents zero discrete ring circulation.
        """
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
            axial_shift, axial_slope = np.zeros(len(geometry)), np.zeros(len(geometry))
        else:
            centreline_radius, slope = self.disturbance.centreline(azimuth, radius)
            axial_shift, axial_slope = self.disturbance.axial_centreline(azimuth, radius)
        magnitude = (
            circulation
            / (np.pi * core_squared)
            * np.exp(
                -((radial - centreline_radius) ** 2 + (axial - axial_shift) ** 2) / core_squared
            )
        )
        cosine, sine = np.cos(azimuth), np.sin(azimuth)
        tangent, radial_direction = (
            -sine[:, None] * first + cosine[:, None] * second,
            cosine[:, None] * first + sine[:, None] * second,
        )
        away = radial > np.finfo(float).eps
        radial_vorticity = np.zeros(len(geometry))
        radial_vorticity[away] = magnitude[away] * slope[away] / radial[away]
        axial_vorticity = np.zeros(len(geometry))
        axial_vorticity[away] = magnitude[away] * axial_slope[away] / radial[away]
        strength = (
            magnitude[:, None] * tangent
            + radial_vorticity[:, None] * radial_direction
            + axial_vorticity[:, None] * axis
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
