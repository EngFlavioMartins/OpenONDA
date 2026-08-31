"""Synthetic homogeneous-isotropic turbulence construction object."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data import ParticleDistribution, VortexParticleSet, attributed_particle_set
from ._common import validate_viscosity
from ._shared import DistributionSource, InitialVelocity, resolve_distribution


@dataclass(frozen=True, slots=True)
class IsotropicTurbulence:
    """Reproducible solenoidal periodic random velocity field.

    ``box_size``, peak wave number and intensity are positive. ``seed`` fixes
    the realization. ``ANALYTICAL`` means synthesized spectral velocity;
    ``ZERO`` retains only the attributed vorticity.
    """

    box_size: float
    spectrum_peak_wave_number: float
    turbulent_intensity: float
    kinematic_viscosity: float
    distribution: DistributionSource = None
    number_of_modes: int = 96
    seed: int = 42
    initial_velocity: InitialVelocity = InitialVelocity.ANALYTICAL

    def build(self, distribution: ParticleDistribution | None = None) -> VortexParticleSet:
        """Build a deterministic divergence-free spectral realization."""
        geometry = resolve_distribution(distribution, self.distribution)
        box, peak, intensity = (
            float(self.box_size),
            float(self.spectrum_peak_wave_number),
            float(self.turbulent_intensity),
        )
        viscosity = validate_viscosity(self.kinematic_viscosity)
        if not np.isfinite(box) or box <= 0.0:
            raise ValueError("box_size must be finite and positive")
        if not np.isfinite(peak) or peak <= 0.0:
            raise ValueError("spectrum_peak_wave_number must be finite and positive")
        if not np.isfinite(intensity) or intensity <= 0.0:
            raise ValueError("turbulent_intensity must be finite and positive")
        if self.number_of_modes < 1:
            raise ValueError("number_of_modes must be positive")
        rng = np.random.default_rng(self.seed)
        integer_wave_vectors = rng.integers(
            -self.number_of_modes, self.number_of_modes + 1, size=(self.number_of_modes, 3)
        )
        integer_wave_vectors[np.all(integer_wave_vectors == 0, axis=1), 0] = 1
        velocity, vorticity = np.zeros_like(geometry.position), np.zeros_like(geometry.position)
        for wave in 2.0 * np.pi / box * integer_wave_vectors:
            magnitude = float(np.linalg.norm(wave))
            direction = rng.normal(size=3)
            transverse = direction - np.dot(direction, wave) * wave / magnitude**2
            norm = np.linalg.norm(transverse)
            if norm <= np.finfo(float).eps:
                continue
            transverse /= norm
            energy = (magnitude / peak) ** 4 / (1.0 + (magnitude / peak) ** 2) ** (17.0 / 6.0)
            amplitude, phase = (
                np.sqrt(energy) / magnitude,
                geometry.position @ wave + rng.uniform(0.0, 2.0 * np.pi),
            )
            velocity += amplitude * np.cos(phase)[:, None] * transverse
            vorticity -= amplitude * np.sin(phase)[:, None] * np.cross(wave, transverse)
        rms = float(np.sqrt(np.mean(np.sum(velocity**2, axis=1))))
        if rms <= np.finfo(float).tiny:
            raise ValueError("synthetic turbulence realization has zero kinetic energy")
        velocity, vorticity = velocity * intensity / rms, vorticity * intensity / rms
        if self.initial_velocity is InitialVelocity.ZERO:
            velocity.fill(0.0)
        elif self.initial_velocity is not InitialVelocity.ANALYTICAL:
            raise ValueError(f"unsupported initial velocity: {self.initial_velocity}")
        return attributed_particle_set(
            geometry,
            velocity=velocity,
            vortex_strength=vorticity * geometry.particle_volume[:, None],
            kinematic_viscosity=viscosity,
        )
