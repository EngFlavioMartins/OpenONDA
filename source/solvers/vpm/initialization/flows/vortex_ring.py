"""Gaussian vortex-ring particle initialization."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..data import ParticleDistribution, VortexParticleDistribution, attributed_distribution
from ..disturbances import WidnallDisturbance
from ._common import (
    represented_core_radius_squared,
    transverse_basis,
    unit_vector,
    validate_viscosity,
    vector3,
)


def initialize_vortex_ring(
    distribution: ParticleDistribution,
    *,
    centre: Sequence[float],
    axis: Sequence[float] = (1.0, 0.0, 0.0),
    radius: float,
    vortex_core_radius: float,
    circulation: float,
    kinematic_viscosity: float,
    disturbance: WidnallDisturbance | None = None,
    compensate_particle_core: bool = False,
    kernel_diffusivity: float = 4.0,
) -> VortexParticleDistribution:
    """Attribute a divergence-free Gaussian vortex ring to a particle distribution.

    The input particle geometry is never displaced. A Widnall disturbance changes
    only the centreline used to evaluate vorticity and its solenoidal direction.
    Initial velocity is zero; the VPM evaluates it from vortex strength.
    """
    centre_array = vector3(centre, "centre")
    axis_array = unit_vector(axis, "axis")
    first, second = transverse_basis(axis_array)
    radius = float(radius)
    circulation = float(circulation)
    viscosity = validate_viscosity(kinematic_viscosity)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and positive")
    if not np.isfinite(circulation) or circulation == 0.0:
        raise ValueError("circulation must be finite and non-zero")
    represented_core_squared = represented_core_radius_squared(
        vortex_core_radius,
        distribution.core_radius,
        compensate_particle_core=compensate_particle_core,
        kernel_diffusivity=kernel_diffusivity,
    )

    relative = distribution.position - centre_array
    axial_position = relative @ axis_array
    first_position = relative @ first
    second_position = relative @ second
    radial_position = np.hypot(first_position, second_position)
    azimuth = np.arctan2(second_position, first_position)
    if disturbance is None:
        centreline_radius = np.full(len(distribution), radius)
        centreline_slope = np.zeros(len(distribution))
    else:
        centreline_radius, centreline_slope = disturbance.centreline(azimuth, radius)

    core_distance_squared = (radial_position - centreline_radius) ** 2 + axial_position**2
    vorticity_magnitude = (
        circulation
        / (np.pi * represented_core_squared)
        * np.exp(-core_distance_squared / represented_core_squared)
    )
    cosine = np.cos(azimuth)
    sine = np.sin(azimuth)
    tangent = -sine[:, None] * first + cosine[:, None] * second
    radial_direction = cosine[:, None] * first + sine[:, None] * second
    radial_vorticity = np.zeros(len(distribution))
    away_from_axis = radial_position > np.finfo(float).eps
    radial_vorticity[away_from_axis] = (
        vorticity_magnitude[away_from_axis]
        * centreline_slope[away_from_axis]
        / radial_position[away_from_axis]
    )
    vorticity = (
        vorticity_magnitude[:, None] * tangent + radial_vorticity[:, None] * radial_direction
    )
    vortex_strength = vorticity * distribution.particle_volume[:, None]

    represented_circulation = np.sum(
        np.einsum("ij,ij->i", vortex_strength[away_from_axis], tangent[away_from_axis])
        / radial_position[away_from_axis]
    ) / (2.0 * np.pi)
    if abs(represented_circulation) <= np.finfo(float).tiny:
        raise ValueError("particle distribution represents zero vortex-ring circulation")
    vortex_strength *= circulation / represented_circulation
    return attributed_distribution(
        distribution,
        velocity=np.zeros_like(distribution.position),
        vortex_strength=vortex_strength,
        kinematic_viscosity=viscosity,
    )
