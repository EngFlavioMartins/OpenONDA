"""Synthetic homogeneous-isotropic turbulence initialization."""

from __future__ import annotations

import numpy as np

from ..data import ParticleDistribution, VortexParticleDistribution, attributed_distribution
from ._common import validate_viscosity


def initialize_isotropic_turbulence(
    distribution: ParticleDistribution,
    *,
    box_size: float,
    spectrum_peak_wave_number: float,
    turbulent_intensity: float,
    kinematic_viscosity: float,
    number_of_modes: int = 96,
    seed: int = 42,
) -> VortexParticleDistribution:
    """Synthesize a reproducible divergence-free periodic random field."""
    box_size = float(box_size)
    peak = float(spectrum_peak_wave_number)
    intensity = float(turbulent_intensity)
    viscosity = validate_viscosity(kinematic_viscosity)
    if not np.isfinite(box_size) or box_size <= 0.0:
        raise ValueError("box_size must be finite and positive")
    if not np.isfinite(peak) or peak <= 0.0:
        raise ValueError("spectrum_peak_wave_number must be finite and positive")
    if not np.isfinite(intensity) or intensity <= 0.0:
        raise ValueError("turbulent_intensity must be finite and positive")
    if number_of_modes < 1:
        raise ValueError("number_of_modes must be positive")

    rng = np.random.default_rng(seed)
    integer_wave_vectors = rng.integers(
        -number_of_modes, number_of_modes + 1, size=(number_of_modes, 3)
    )
    zero = np.all(integer_wave_vectors == 0, axis=1)
    integer_wave_vectors[zero, 0] = 1
    wave_vectors = 2.0 * np.pi / box_size * integer_wave_vectors
    velocity = np.zeros_like(distribution.position)
    vorticity = np.zeros_like(distribution.position)
    for wave_vector in wave_vectors:
        magnitude = float(np.linalg.norm(wave_vector))
        random_direction = rng.normal(size=3)
        transverse = (
            random_direction - np.dot(random_direction, wave_vector) * wave_vector / magnitude**2
        )
        transverse_norm = np.linalg.norm(transverse)
        if transverse_norm <= np.finfo(float).eps:
            continue
        transverse /= transverse_norm
        energy = (magnitude / peak) ** 4 / (1.0 + (magnitude / peak) ** 2) ** (17.0 / 6.0)
        amplitude = np.sqrt(energy) / magnitude
        phase = distribution.position @ wave_vector + rng.uniform(0.0, 2.0 * np.pi)
        velocity += amplitude * np.cos(phase)[:, None] * transverse
        vorticity -= amplitude * np.sin(phase)[:, None] * np.cross(wave_vector, transverse)
    rms = float(np.sqrt(np.mean(np.sum(velocity**2, axis=1))))
    if rms <= np.finfo(float).tiny:
        raise ValueError("synthetic turbulence realization has zero kinetic energy")
    scale = intensity / rms
    velocity *= scale
    vorticity *= scale
    return attributed_distribution(
        distribution,
        velocity=velocity,
        vortex_strength=vorticity * distribution.particle_volume[:, None],
        kinematic_viscosity=viscosity,
    )
