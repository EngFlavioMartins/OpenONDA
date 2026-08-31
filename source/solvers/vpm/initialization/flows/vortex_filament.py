"""Straight Gaussian/Lamb--Oseen vortex-filament initialization."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..data import ParticleDistribution, VortexParticleDistribution, attributed_distribution
from ..disturbances import FilamentDisturbance
from ._common import (
    represented_core_radius_squared,
    transverse_basis,
    unit_vector,
    validate_viscosity,
    vector3,
)


def initialize_vortex_filament(
    distribution: ParticleDistribution,
    *,
    centre: Sequence[float],
    direction: Sequence[float] = (0.0, 0.0, 1.0),
    vortex_core_radius: float,
    circulation: float,
    kinematic_viscosity: float,
    disturbance: FilamentDisturbance | None = None,
    compensate_particle_core: bool = False,
    kernel_diffusivity: float = 4.0,
) -> VortexParticleDistribution:
    """Attribute a straight or sinusoidally disturbed Gaussian filament.

    Initial velocity is deliberately zero and is refreshed by the VPM.
    """
    centre_array = vector3(centre, "centre")
    direction_array = unit_vector(direction, "direction")
    circulation = float(circulation)
    viscosity = validate_viscosity(kinematic_viscosity)
    if not np.isfinite(circulation) or circulation == 0.0:
        raise ValueError("circulation must be finite and non-zero")
    represented_core_squared = represented_core_radius_squared(
        vortex_core_radius,
        distribution.core_radius,
        compensate_particle_core=compensate_particle_core,
        kernel_diffusivity=kernel_diffusivity,
    )

    relative = distribution.position - centre_array
    axial_position = relative @ direction_array
    transverse = relative - axial_position[:, None] * direction_array
    tangent = np.broadcast_to(direction_array, distribution.position.shape).copy()
    if disturbance is not None:
        first, second = transverse_basis(direction_array)
        polarization = (
            np.cos(disturbance.polarization_angle) * first
            + np.sin(disturbance.polarization_angle) * second
        )
        wave_number = 2.0 * np.pi / disturbance.wavelength
        argument = wave_number * axial_position + disturbance.phase
        displacement = disturbance.amplitude * np.sin(argument)
        slope = disturbance.amplitude * wave_number * np.cos(argument)
        transverse -= displacement[:, None] * polarization
        tangent += slope[:, None] * polarization

    radial_distance_squared = np.einsum("ij,ij->i", transverse, transverse)
    vorticity_magnitude = (
        circulation
        / (np.pi * represented_core_squared)
        * np.exp(-radial_distance_squared / represented_core_squared)
    )
    vortex_strength = vorticity_magnitude[:, None] * tangent * distribution.particle_volume[:, None]
    return attributed_distribution(
        distribution,
        velocity=np.zeros_like(distribution.position),
        vortex_strength=vortex_strength,
        kinematic_viscosity=viscosity,
    )
