"""Tests for symmetry-preserving vortex-ring particle seeding."""

import numpy as np
import pytest

from source.solvers.vpm.initial_conditions import vortex_ring_vpm
from source.solvers.vpm.particles.distribution import ParticleDistributor


@pytest.mark.unit
def test_toroidal_distribution_has_periodic_symmetry_and_correct_measure():
    ring_radius = 1.0
    tube_radius = 0.18
    spacing = 0.04
    position, particle_volume, core_radius = ParticleDistributor.toroidal_distribution(
        ring_radius, tube_radius, spacing
    )

    rho = np.linalg.norm(position[:, 1:], axis=1)
    tangent = np.column_stack(
        (np.zeros(len(position)), -position[:, 2] / rho, position[:, 1] / rho)
    )
    exact_torus_volume = 2.0 * np.pi**2 * ring_radius * tube_radius**2

    assert len(position) > 0
    assert np.all(particle_volume > 0.0)
    assert np.all(core_radius == 2.0 * spacing)
    assert np.linalg.norm((tangent * particle_volume[:, None]).sum(axis=0)) < 1.0e-13
    assert particle_volume.sum() == pytest.approx(exact_torus_volume, rel=0.08)
    assert np.max(np.sqrt(position[:, 0] ** 2 + (rho - ring_radius) ** 2)) <= tube_radius


@pytest.mark.unit
def test_toroidal_distribution_returns_complete_contiguous_orbits():
    position, _, _, orbit_id = ParticleDistributor.toroidal_distribution(
        1.0,
        0.18,
        0.04,
        return_orbit_ids=True,
    )
    counts = np.bincount(orbit_id)

    assert np.all(counts == counts[0])
    assert np.array_equal(orbit_id, np.repeat(np.arange(len(counts)), counts[0]))
    for orbit in range(len(counts)):
        selected = orbit_id == orbit
        assert np.ptp(position[selected, 0]) < 1.0e-14
        assert np.ptp(np.linalg.norm(position[selected, 1:], axis=1)) < 1.0e-14


@pytest.mark.unit
@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_toroidal_distribution_honors_axis_and_centre(axis):
    centre = np.array([0.2, -0.3, 0.4])
    position, _, _ = ParticleDistributor.toroidal_distribution(
        0.8, 0.12, 0.05, centre_position=centre, axis=axis
    )

    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    assert position[:, axis_index].mean() == pytest.approx(centre[axis_index], abs=1.0e-14)


@pytest.mark.unit
def test_perturbed_ring_seeding_preserves_zero_vector_circulation():
    spacing = 0.04
    kinematic_viscosity = np.pi / 3000.0
    core_radius = 0.1
    particle_core_radius = 2.0 * spacing
    represented_core = np.sqrt(core_radius**2 - particle_core_radius**2)
    tube_radius = represented_core * np.sqrt(-np.log(0.05))
    position, particle_volume, core_radius = ParticleDistributor.toroidal_distribution(
        1.0,
        tube_radius,
        spacing,
        widnall_amplitude=0.025,
        seed=7,
        n_widnall_modes=12,
    )
    _, _, vortex_strength = vortex_ring_vpm(
        kinematic_viscosity=kinematic_viscosity,
        ring_centre=np.zeros(3),
        tube_circulation=np.pi,
        ring_radius=1.0,
        ring_core_radius=core_radius,
        mean_core_radius=float(core_radius.mean()),
        position=position,
        particle_volume=particle_volume,
        widnall_amplitude=0.025,
        seed=7,
        n_widnall_modes=12,
        is_anti_diffusion_enabled=True,
        is_circulation_normalization_enabled=True,
    )

    rho = np.linalg.norm(position[:, 1:], axis=1)
    tangent = np.column_stack(
        (np.zeros(len(position)), -position[:, 2] / rho, position[:, 1] / rho)
    )
    represented_circulation = np.sum(np.einsum("ij,ij->i", vortex_strength, tangent) / rho) / (
        2.0 * np.pi
    )
    assert (
        np.linalg.norm(vortex_strength.sum(axis=0)) / np.linalg.norm(vortex_strength, axis=1).sum()
        < 1.0e-14
    )
    assert represented_circulation == pytest.approx(np.pi, rel=1.0e-14)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("ring_radius", "tube_radius", "spacing", "message"),
    [
        (0.0, 0.1, 0.02, "ring_radius"),
        (1.0, 0.0, 0.02, "tube_radius"),
        (1.0, 0.1, 0.0, "spacing"),
        (1.0, 1.0, 0.02, "smaller"),
    ],
)
def test_toroidal_distribution_rejects_invalid_geometry(ring_radius, tube_radius, spacing, message):
    with pytest.raises(ValueError, match=message):
        ParticleDistributor.toroidal_distribution(ring_radius, tube_radius, spacing)
