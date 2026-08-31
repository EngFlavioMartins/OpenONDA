"""Particle geometry and canonical VPM flow attribution."""

import numpy as np
import pytest

from source.solvers.vpm.initialization import (
    CylindricalDistribution,
    FilamentDisturbance,
    IsotropicTurbulence,
    NoisyRectangularDistribution,
    ParticleCoreCompensation,
    RectangularDistribution,
    TaylorGreenVortex,
    ToroidalDistribution,
    TriangularPrismDistribution,
    VortexDoublet,
    VortexFilament,
    VortexRing,
    WidnallDisturbance,
)


def test_rectangular_distribution_preserves_spacing_and_sigma_over_h():
    distribution = RectangularDistribution(
        bounds=((0.0, 1.0), (-0.6, 0.6), (2.0, 2.37)),
        spacing=0.2,
        core_radius_ratio=1.5,
    ).build()

    for axis in range(3):
        coordinates = np.unique(distribution.position[:, axis])
        if len(coordinates) > 1:
            np.testing.assert_allclose(np.diff(coordinates), 0.2)
    np.testing.assert_allclose(distribution.core_radius, 0.3)
    np.testing.assert_allclose(distribution.particle_volume, 0.2**3)
    assert distribution.core_radius_ratio == pytest.approx(1.5)
    assert distribution.position[:, 2].min() >= 2.0
    assert distribution.position[:, 2].max() <= 2.37


def test_single_widnall_mode_has_the_requested_centreline_and_slope():
    azimuth = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    disturbance = WidnallDisturbance.single_mode(amplitude=0.05, mode=8, phase=0.3)

    radius, slope = disturbance.centreline(azimuth, ring_radius=2.0)

    argument = 8 * azimuth + 0.3
    np.testing.assert_allclose(radius, 2.0 * (1.0 + 0.05 * np.sin(argument)))
    np.testing.assert_allclose(slope, 2.0 * 0.05 * 8 * np.cos(argument))


def test_widnall_disturbance_rejects_invalid_mode():
    with pytest.raises(ValueError, match="mode must be positive"):
        WidnallDisturbance.single_mode(amplitude=0.05, mode=0)


def test_every_distribution_uses_the_requested_sigma_over_h():
    distributions = (
        NoisyRectangularDistribution(
            bounds=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2)),
            spacing=0.1,
            core_radius_ratio=1.25,
            seed=3,
        ).build(),
        TriangularPrismDistribution(
            bounds=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2)),
            spacing=0.1,
            core_radius_ratio=1.25,
        ).build(),
        ToroidalDistribution(
            ring_radius=1.0,
            tube_radius=0.2,
            spacing=0.1,
            core_radius_ratio=1.25,
        ).build(),
        CylindricalDistribution(
            radius=0.3,
            length=0.5,
            spacing=0.1,
            core_radius_ratio=1.25,
        ).build(),
    )

    for distribution in distributions:
        assert len(distribution) > 0
        np.testing.assert_allclose(distribution.core_radius, 0.125)
        assert np.all(distribution.particle_volume > 0.0)


def test_vortex_ring_attributes_flow_without_moving_particles():
    distribution = RectangularDistribution(
        bounds=((-0.2, 0.2), (-1.3, 1.3), (-1.3, 1.3)),
        spacing=0.1,
        core_radius_ratio=0.5,
    ).build()
    original_position = distribution.position.copy()
    particles = VortexRing(
        centre=(0.0, 0.0, 0.0),
        radius=1.0,
        vortex_core_radius=0.2,
        circulation=1.0,
        kinematic_viscosity=1.0e-3,
        disturbance=WidnallDisturbance.single_mode(amplitude=0.02, mode=4),
        core_compensation=ParticleCoreCompensation(),
    ).build(distribution)

    np.testing.assert_array_equal(particles.position, original_position)
    np.testing.assert_array_equal(distribution.position, original_position)
    np.testing.assert_array_equal(particles.velocity, np.zeros_like(particles.position))
    assert np.linalg.norm(particles.vortex_strength) > 0.0
    assert particles.position.flags.writeable is False
    assert particles.vortex_strength.flags.writeable is False


def test_vortex_filament_supports_arbitrary_direction_and_zero_velocity():
    distribution = RectangularDistribution(
        bounds=((-0.4, 0.4), (-0.4, 0.4), (-1.0, 1.0)),
        spacing=0.2,
        core_radius_ratio=0.5,
    ).build()
    particles = VortexFilament(
        centre=(0.0, 0.0, 0.0),
        direction=(0.0, 0.0, 2.0),
        vortex_core_radius=0.3,
        circulation=1.0,
        kinematic_viscosity=1.0e-3,
        disturbance=FilamentDisturbance(amplitude=0.02, wavelength=1.0),
        core_compensation=ParticleCoreCompensation(),
    ).build(distribution)

    np.testing.assert_array_equal(particles.position, distribution.position)
    np.testing.assert_array_equal(particles.velocity, np.zeros_like(particles.position))
    assert np.linalg.norm(particles.vortex_strength) > 0.0


def test_filament_tail_settings_normalize_circulation_and_assign_group_id():
    distribution = RectangularDistribution(
        bounds=((-0.4, 0.4), (-0.4, 0.4), (-1.0, 1.0)),
        spacing=0.2,
        core_radius_ratio=0.5,
    ).build()
    particles = VortexFilament(
        vortex_core_radius=0.3,
        circulation=1.0,
        kinematic_viscosity=1.0e-3,
        group_id=7,
        tail_minimum_relative_strength=0.05,
        tail_circulation_per_length=1.0,
        tail_represented_length=2.0,
    ).build(distribution)

    assert np.all(particles.group_id == 7)
    assert len(particles) < len(distribution)
    assert particles.vortex_strength[:, 2].sum() / 2.0 == pytest.approx(1.0)


def test_remaining_fundamental_flows_return_solver_ready_particles():
    distribution = RectangularDistribution(
        bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        spacing=0.5,
        core_radius_ratio=0.5,
    ).build()
    initializations = (
        VortexDoublet(
            centre=(0.25, 0.25, 0.25),
            direction=(1.0, 1.0, 0.0),
            strength=1.0,
            kinematic_viscosity=1.0e-3,
        ).build(distribution),
        TaylorGreenVortex(
            box_size=1.0,
            kinematic_viscosity=1.0e-3,
        ).build(distribution),
        IsotropicTurbulence(
            box_size=1.0,
            spectrum_peak_wave_number=2.0 * np.pi,
            turbulent_intensity=0.1,
            kinematic_viscosity=1.0e-3,
            number_of_modes=8,
            seed=7,
        ).build(distribution),
    )

    for particles in initializations:
        assert len(particles) == len(distribution)
        assert particles.position.flags.writeable is False
        assert np.all(np.isfinite(particles.velocity))
        assert np.all(np.isfinite(particles.vortex_strength))
