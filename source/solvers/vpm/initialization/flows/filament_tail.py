"""Straight-filament tail filtering helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..data import VortexParticleSet
from ._common import unit_vector


def filter_tail(
    particles: VortexParticleSet,
    *,
    minimum_relative_strength: float,
    circulation_per_length: float | None,
    represented_length: float | None,
    direction: Sequence[float],
) -> VortexParticleSet:
    """Drop weak filament particles and optionally restore axial circulation."""
    if not np.isfinite(minimum_relative_strength) or not 0.0 <= minimum_relative_strength < 1.0:
        raise ValueError("tail_minimum_relative_strength must be finite and in [0, 1)")
    magnitude = np.linalg.norm(particles.vortex_strength, axis=1)
    peak = float(np.max(magnitude, initial=0.0))
    if peak <= np.finfo(float).tiny:
        raise ValueError("cannot filter a zero-strength particle set")
    result = particles.select(magnitude >= minimum_relative_strength * peak)
    if len(result) == 0:
        raise ValueError("tail filtering removed every particle")
    if circulation_per_length is None:
        return result
    if not np.isfinite(circulation_per_length) or circulation_per_length == 0.0:
        raise ValueError("tail_circulation_per_length must be finite and non-zero")
    if (
        represented_length is None
        or not np.isfinite(represented_length)
        or represented_length <= 0.0
    ):
        raise ValueError("tail_represented_length must be finite and positive")
    axis = unit_vector(direction, "tail_direction")
    current = float(np.sum(result.vortex_strength @ axis) / represented_length)
    if abs(current) <= np.finfo(float).tiny:
        raise ValueError("cannot restore zero axial circulation")
    return VortexParticleSet(
        position=result.position,
        velocity=result.velocity,
        vortex_strength=result.vortex_strength * circulation_per_length / current,
        core_radius=result.core_radius,
        particle_volume=result.particle_volume,
        kinematic_viscosity=result.kinematic_viscosity,
        spacing=result.spacing,
        group_id=result.group_id,
        zone_id=result.zone_id,
    )
