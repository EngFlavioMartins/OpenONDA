"""Straight Gaussian vortex filaments along z on a triangular particle lattice.

The physical core radius ``a`` is corrected for the particle cores ``sigma``
already smoothing the field: the attributed Gaussian has squared width
``a**2 - mean(sigma)**2``.
"""

from __future__ import annotations

import numpy as np

import openonda.vpm as vpm


def centred_coordinates(lower, upper, spacing):
    """Exactly spaced lattice coordinates centred in ``[lower, upper]``."""
    count = max(1, int(np.floor((upper - lower) / spacing + 1.0e-12)) + 1)
    start = 0.5 * (lower + upper - (count - 1) * spacing)
    return start + spacing * np.arange(count, dtype=float)


class TriangularPrismDistribution:
    """Triangular lattice in the x-y plane, extruded along z with the same spacing.

    Each particle volume is the triangular cell area ``sqrt(3) * h**2 / 2``
    times the axial spacing ``h``.
    """

    def __init__(self, bounds, spacing, core_radius_ratio):
        self.bounds = bounds
        self.spacing = spacing
        self.core_radius_ratio = core_radius_ratio

    def build(self):
        h = self.spacing
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = self.bounds
        x = centred_coordinates(x_min, x_max, h)
        y = centred_coordinates(y_min, y_max, np.sqrt(3.0) * h / 2.0)
        plane = []
        for row, y_value in enumerate(y):
            shifted = x + (0.5 * h if row % 2 else 0.0)
            eps = np.finfo(float).eps
            shifted = shifted[(shifted >= x_min - eps) & (shifted <= x_max + eps)]
            plane.extend((x_value, y_value) for x_value in shifted)
        z = centred_coordinates(z_min, z_max, h)

        position = np.empty((len(plane) * len(z), 3), dtype=float)
        for orbit, (x_value, y_value) in enumerate(plane):
            column = slice(orbit * len(z), (orbit + 1) * len(z))
            position[column, 2], position[column, 0], position[column, 1] = z, x_value, y_value
        return vpm.ParticleDistribution(
            position=position,
            core_radius=np.full(len(position), self.core_radius_ratio * h),
            particle_volume=np.full(len(position), np.sqrt(3.0) * h**2 / 2.0 * h),
            spacing=h,
        )


class VortexFilament:
    """Gaussian vortex filament along z through ``centre``.

    Particles weaker than ``tail_minimum_relative_strength`` times the peak are
    dropped; the remaining strengths are rescaled so the column of length
    ``represented_length`` carries ``circulation`` per unit length.
    """

    def __init__(
        self,
        vortex_core_radius,
        circulation,
        kinematic_viscosity,
        distribution,
        represented_length,
        tail_minimum_relative_strength,
        *,
        centre=(0.0, 0.0, 0.0),
        group_id=0,
    ):
        self.vortex_core_radius = vortex_core_radius
        self.circulation = circulation
        self.kinematic_viscosity = kinematic_viscosity
        self.distribution = distribution
        self.represented_length = represented_length
        self.tail_minimum_relative_strength = tail_minimum_relative_strength
        self.centre = centre
        self.group_id = group_id

    def build(self):
        geometry = self.distribution.build()
        core_squared = self.vortex_core_radius**2 - np.mean(geometry.core_radius) ** 2

        # Gaussian vorticity about the filament axis
        direction = np.array([0.0, 0.0, 1.0])
        relative = geometry.position - np.asarray(self.centre, dtype=float)
        transverse = relative - (relative @ direction)[:, None] * direction
        radial_squared = np.einsum("ij,ij->i", transverse, transverse)
        magnitude = (
            self.circulation / (np.pi * core_squared) * np.exp(-radial_squared / core_squared)
        )
        strength = magnitude[:, None] * direction * geometry.particle_volume[:, None]

        # Drop the weak tail, then restore the circulation per unit length
        strength_magnitude = np.linalg.norm(strength, axis=1)
        kept = strength_magnitude >= self.tail_minimum_relative_strength * strength_magnitude.max()
        strength = strength[kept]
        represented = np.sum(strength @ direction) / self.represented_length
        count = int(kept.sum())
        return vpm.VortexParticleSet(
            position=geometry.position[kept],
            velocity=np.zeros((count, 3)),
            vortex_strength=strength * self.circulation / represented,
            core_radius=geometry.core_radius[kept],
            particle_volume=geometry.particle_volume[kept],
            kinematic_viscosity=np.full(count, float(self.kinematic_viscosity)),
            spacing=geometry.spacing,
            group_id=np.full(count, self.group_id, dtype=np.int32),
        )
