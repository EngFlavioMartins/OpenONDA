"""Exact-spacing rectangular particle distributions."""

from __future__ import annotations

import numpy as np

from ..data import ParticleDistribution
from ._common import Bounds3D, centred_coordinates, validate_bounds, validate_spacing


def create_rectangular_distribution(
    *,
    bounds: Bounds3D,
    spacing: float,
    core_radius_ratio: float,
) -> ParticleDistribution:
    """Create a Cartesian lattice without altering the requested spacing.

    When an extent is not divisible by ``spacing``, the lattice is centered
    inside that extent. Particle volumes are the nominal cell volume ``h^3``.
    """
    spacing, core_radius_ratio = validate_spacing(spacing, core_radius_ratio)
    limits = validate_bounds(bounds)
    coordinates = [centred_coordinates(*limits[axis], spacing) for axis in range(3)]
    position = np.stack(np.meshgrid(*coordinates, indexing="ij"), axis=-1).reshape(-1, 3)
    count = len(position)
    return ParticleDistribution(
        position=position,
        core_radius=np.full(count, core_radius_ratio * spacing),
        particle_volume=np.full(count, spacing**3),
        spacing=spacing,
    )


def create_noisy_rectangular_distribution(
    *,
    bounds: Bounds3D,
    spacing: float,
    core_radius_ratio: float,
    noise_fraction: float = 0.3,
    seed: int | None = None,
) -> ParticleDistribution:
    """Create a Cartesian lattice with bounded, reproducible point jitter."""
    if not np.isfinite(noise_fraction) or not 0.0 <= noise_fraction <= 1.0:
        raise ValueError("noise_fraction must be finite and between zero and one")
    base = create_rectangular_distribution(
        bounds=bounds,
        spacing=spacing,
        core_radius_ratio=core_radius_ratio,
    )
    limits = validate_bounds(bounds)
    rng = np.random.default_rng(seed)
    position = base.position.copy()
    position += rng.uniform(
        -0.5 * noise_fraction * base.spacing,
        0.5 * noise_fraction * base.spacing,
        size=position.shape,
    )
    position = np.clip(position, limits[:, 0], limits[:, 1])
    return ParticleDistribution(
        position=position,
        core_radius=base.core_radius,
        particle_volume=base.particle_volume,
        spacing=base.spacing,
    )
