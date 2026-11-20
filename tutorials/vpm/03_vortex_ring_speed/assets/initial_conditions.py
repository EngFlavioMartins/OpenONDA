"""Gaussian vortex ring on a toroidal particle lattice, normal to the x axis.

The physical core radius ``a`` is corrected for the particle cores ``sigma``
already smoothing the field: the attributed Gaussian has squared width
``a**2 - mean(sigma)**2``.
"""

from __future__ import annotations

import numpy as np

import openonda.vpm as vpm


class WidnallDisturbance:
    """Azimuthal displacement of the ring centreline, relative to its radius.

    A single mode gives ``amplitude * R * sin(mode * theta + phase)``; without
    ``mode`` the displacement sums ``number_of_modes`` equal-weight modes with
    phases drawn from ``seed``. ``direction`` is ``"radial"`` or ``"axial"``.
    """

    def __init__(
        self, amplitude, *, mode=None, phase=0.0, number_of_modes=24, seed=42, direction="radial"
    ):
        self.amplitude = amplitude
        self.mode = mode
        self.phase = phase
        self.number_of_modes = number_of_modes
        self.seed = seed
        self.direction = direction

    def displacement(self, azimuth, ring_radius):
        """Return the displacement and its azimuthal derivative, both in m."""
        if self.mode is not None:
            argument = self.mode * azimuth + self.phase
            radius = ring_radius * self.amplitude * np.sin(argument)
            slope = ring_radius * self.amplitude * self.mode * np.cos(argument)
            return radius, slope
        phases = 2.0 * np.pi * np.random.default_rng(self.seed).random(self.number_of_modes)
        shape = np.zeros_like(azimuth, dtype=float)
        slope = np.zeros_like(azimuth, dtype=float)
        for mode in range(1, self.number_of_modes + 1):
            argument = mode * azimuth + phases[mode - 1]
            shape += np.cos(argument)
            slope -= mode * np.sin(argument)
        normalization = np.sqrt(self.number_of_modes)
        return (
            ring_radius * self.amplitude * shape / normalization,
            ring_radius * self.amplitude * slope / normalization,
        )

    def centreline(self, azimuth, ring_radius):
        """Return the radial centreline position and its azimuthal derivative."""
        if self.direction == "axial":
            return np.full_like(azimuth, ring_radius, dtype=float), np.zeros_like(azimuth)
        displacement, slope = self.displacement(azimuth, ring_radius)
        return ring_radius + displacement, slope

    def axial_centreline(self, azimuth, ring_radius):
        """Return the axial centreline shift and its azimuthal derivative."""
        if self.direction == "radial":
            return np.zeros_like(azimuth, dtype=float), np.zeros_like(azimuth, dtype=float)
        return self.displacement(azimuth, ring_radius)


class ToroidalDistribution:
    """Hexagonal tube cross-section revolved about the x axis.

    The azimuthal particle count is rounded up to a multiple of four, and each
    particle volume includes the cylindrical Jacobian. A disturbance moves the
    tube with the ring centreline.
    """

    def __init__(
        self,
        ring_radius,
        tube_radius,
        spacing,
        core_radius_ratio,
        *,
        centre=(0.0, 0.0, 0.0),
        disturbance=None,
    ):
        self.ring_radius = ring_radius
        self.tube_radius = tube_radius
        self.spacing = spacing
        self.core_radius_ratio = core_radius_ratio
        self.centre = centre
        self.disturbance = disturbance

    def build(self):
        spacing = self.spacing
        row_spacing = np.sqrt(3.0) * spacing / 2.0
        rows = int(np.ceil(self.tube_radius / row_spacing))
        cross_section = []
        for row in range(-rows, rows + 1):
            radial_offset = row * row_spacing
            columns = int(np.ceil(self.tube_radius / spacing)) + abs(row)
            for column in range(-columns, columns + 1):
                axial_offset = spacing * (column + 0.5 * row)
                inside = (self.tube_radius + 16 * np.finfo(float).eps) ** 2
                if axial_offset**2 + radial_offset**2 <= inside:
                    cross_section.append((axial_offset, radial_offset))
        offsets = np.asarray(cross_section, dtype=float)

        azimuth_count = max(
            8, int(np.ceil(2.0 * np.pi * (self.ring_radius + self.tube_radius) / spacing))
        )
        azimuth_count += (-azimuth_count) % 4
        azimuth = 2.0 * np.pi * np.arange(azimuth_count) / azimuth_count
        cosine = np.tile(np.cos(azimuth), len(offsets))
        sine = np.tile(np.sin(azimuth), len(offsets))
        axial = np.repeat(offsets[:, 0], azimuth_count)
        radial = np.repeat(self.ring_radius + offsets[:, 1], azimuth_count)

        if self.disturbance is not None:
            # Same azimuth origin as VortexRing's transverse basis about x
            field_azimuth = np.tile(azimuth - np.pi / 2, len(offsets))
            shifted_radius, _ = self.disturbance.centreline(field_azimuth, self.ring_radius)
            radial = radial + (shifted_radius - self.ring_radius)
            axial_shift, _ = self.disturbance.axial_centreline(field_azimuth, self.ring_radius)
            axial = axial + axial_shift

        position = np.column_stack((axial, radial * cosine, radial * sine))
        volume = np.sqrt(3.0) * spacing**2 / 2.0 * radial * (2.0 * np.pi / azimuth_count)
        return vpm.ParticleDistribution(
            position=position + np.asarray(self.centre, dtype=float),
            core_radius=np.full(len(position), self.core_radius_ratio * spacing),
            particle_volume=volume,
            spacing=spacing,
        )


class VortexRing:
    """Gaussian vortex ring of circulation ``circulation`` normal to the x axis.

    Strengths ``Gamma_i = omega_i * V_i`` are rescaled so the discrete ring
    carries exactly ``circulation``. Particle velocity starts at zero; the
    solver evaluates the induced velocity.
    """

    def __init__(
        self,
        radius,
        vortex_core_radius,
        circulation,
        kinematic_viscosity,
        distribution,
        *,
        centre=(0.0, 0.0, 0.0),
        disturbance=None,
        group_id=0,
    ):
        self.radius = radius
        self.vortex_core_radius = vortex_core_radius
        self.circulation = circulation
        self.kinematic_viscosity = kinematic_viscosity
        self.distribution = distribution
        self.centre = centre
        self.disturbance = disturbance
        self.group_id = group_id

    def build(self):
        geometry = self.distribution.build()
        count = len(geometry)
        core_squared = self.vortex_core_radius**2 - np.mean(geometry.core_radius) ** 2

        # Ring frame: x is the ring axis, (first, second) span the ring plane
        axis = np.array([1.0, 0.0, 0.0])
        first = np.array([0.0, 0.0, 1.0])
        second = np.array([0.0, -1.0, 0.0])
        relative = geometry.position - np.asarray(self.centre, dtype=float)
        axial = relative @ axis
        first_position, second_position = relative @ first, relative @ second
        radial = np.hypot(first_position, second_position)
        azimuth = np.arctan2(second_position, first_position)

        if self.disturbance is None:
            centreline_radius, slope = np.full(count, self.radius), np.zeros(count)
            axial_shift, axial_slope = np.zeros(count), np.zeros(count)
        else:
            centreline_radius, slope = self.disturbance.centreline(azimuth, self.radius)
            axial_shift, axial_slope = self.disturbance.axial_centreline(azimuth, self.radius)

        # Gaussian vorticity about the (displaced) centreline
        distance_squared = (radial - centreline_radius) ** 2 + (axial - axial_shift) ** 2
        magnitude = (
            self.circulation / (np.pi * core_squared) * np.exp(-distance_squared / core_squared)
        )
        cosine, sine = np.cos(azimuth), np.sin(azimuth)
        tangent = -sine[:, None] * first + cosine[:, None] * second
        radial_direction = cosine[:, None] * first + sine[:, None] * second
        away = radial > np.finfo(float).eps
        radial_vorticity = np.zeros(count)
        radial_vorticity[away] = magnitude[away] * slope[away] / radial[away]
        axial_vorticity = np.zeros(count)
        axial_vorticity[away] = magnitude[away] * axial_slope[away] / radial[away]
        strength = (
            magnitude[:, None] * tangent
            + radial_vorticity[:, None] * radial_direction
            + axial_vorticity[:, None] * axis
        ) * geometry.particle_volume[:, None]

        # Discrete circulation around the ring axis
        represented = np.sum(
            np.einsum("ij,ij->i", strength[away], tangent[away]) / radial[away]
        ) / (2.0 * np.pi)
        return vpm.VortexParticleSet(
            position=geometry.position,
            velocity=np.zeros_like(geometry.position),
            vortex_strength=strength * self.circulation / represented,
            core_radius=geometry.core_radius,
            particle_volume=geometry.particle_volume,
            kinematic_viscosity=np.full(count, float(self.kinematic_viscosity)),
            spacing=geometry.spacing,
            group_id=np.full(count, self.group_id, dtype=np.int32),
        )
