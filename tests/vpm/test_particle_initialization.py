"""Contracts for particle geometry and canonical VPM flow attribution."""

import numpy as np
import pytest

from source.solvers.vpm.initialization import (
    FilamentDisturbance,
    WidnallDisturbance,
    create_cylindrical_distribution,
    create_noisy_rectangular_distribution,
    create_rectangular_distribution,
    create_toroidal_distribution,
    create_triangular_prism_distribution,
    initialize_isotropic_turbulence,
    initialize_taylor_green_vortex,
    initialize_vortex_doublet,
    initialize_vortex_filament,
    initialize_vortex_ring,
)


def test_rectangular_distribution_preserves_spacing_and_sigma_over_h():
    distribution = create_rectangular_distribution(
        bounds=((0.0, 1.0), (-0.6, 0.6), (2.0, 2.37)),
        spacing=0.2,
        core_radius_ratio=1.5,
    )

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
        create_noisy_rectangular_distribution(
            bounds=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2)),
            spacing=0.1,
            core_radius_ratio=1.25,
            seed=3,
        ),
        create_triangular_prism_distribution(
            bounds=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2)),
            spacing=0.1,
            core_radius_ratio=1.25,
        ),
        create_toroidal_distribution(
            ring_radius=1.0,
            tube_radius=0.2,
            spacing=0.1,
            core_radius_ratio=1.25,
        ),
        create_cylindrical_distribution(
            radius=0.3,
            length=0.5,
            spacing=0.1,
            core_radius_ratio=1.25,
        ),
    )

    for distribution in distributions:
        assert len(distribution) > 0
        np.testing.assert_allclose(distribution.core_radius, 0.125)
        assert np.all(distribution.particle_volume > 0.0)


def test_vortex_ring_attributes_flow_without_moving_particles():
    distribution = create_rectangular_distribution(
        bounds=((-0.2, 0.2), (-1.3, 1.3), (-1.3, 1.3)),
        spacing=0.1,
        core_radius_ratio=0.5,
    )
    original_position = distribution.position.copy()
    particles = initialize_vortex_ring(
        distribution,
        centre=(0.0, 0.0, 0.0),
        radius=1.0,
        vortex_core_radius=0.2,
        circulation=1.0,
        kinematic_viscosity=1.0e-3,
        disturbance=WidnallDisturbance.single_mode(amplitude=0.02, mode=4),
        compensate_particle_core=True,
    )

    np.testing.assert_array_equal(particles.position, original_position)
    np.testing.assert_array_equal(distribution.position, original_position)
    np.testing.assert_array_equal(particles.velocity, np.zeros_like(particles.position))
    assert np.linalg.norm(particles.vortex_strength) > 0.0
    assert set(particles.solver_kwargs()) == {
        "position",
        "velocity",
        "vortex_strength",
        "core_radius",
        "particle_volume",
        "kinematic_viscosity",
    }


def test_vortex_filament_supports_arbitrary_direction_and_zero_velocity():
    distribution = create_rectangular_distribution(
        bounds=((-0.4, 0.4), (-0.4, 0.4), (-1.0, 1.0)),
        spacing=0.2,
        core_radius_ratio=0.5,
    )
    particles = initialize_vortex_filament(
        distribution,
        centre=(0.0, 0.0, 0.0),
        direction=(0.0, 0.0, 2.0),
        vortex_core_radius=0.3,
        circulation=1.0,
        kinematic_viscosity=1.0e-3,
        disturbance=FilamentDisturbance(amplitude=0.02, wavelength=1.0),
        compensate_particle_core=True,
    )

    np.testing.assert_array_equal(particles.position, distribution.position)
    np.testing.assert_array_equal(particles.velocity, np.zeros_like(particles.position))
    assert np.linalg.norm(particles.vortex_strength) > 0.0


def test_remaining_fundamental_flows_return_solver_ready_particles():
    distribution = create_rectangular_distribution(
        bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        spacing=0.5,
        core_radius_ratio=0.5,
    )
    initializations = (
        initialize_vortex_doublet(
            distribution,
            centre=(0.25, 0.25, 0.25),
            direction=(1.0, 1.0, 0.0),
            strength=1.0,
            kinematic_viscosity=1.0e-3,
        ),
        initialize_taylor_green_vortex(
            distribution,
            box_size=1.0,
            kinematic_viscosity=1.0e-3,
        ),
        initialize_isotropic_turbulence(
            distribution,
            box_size=1.0,
            spectrum_peak_wave_number=2.0 * np.pi,
            turbulent_intensity=0.1,
            kinematic_viscosity=1.0e-3,
            number_of_modes=8,
            seed=7,
        ),
    )

    for particles in initializations:
        assert len(particles) == len(distribution)
        assert set(particles.solver_kwargs()) >= {"position", "vortex_strength"}
        assert np.all(np.isfinite(particles.velocity))
        assert np.all(np.isfinite(particles.vortex_strength))
