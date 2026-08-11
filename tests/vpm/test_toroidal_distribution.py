"""Tests for symmetry-preserving vortex-ring particle seeding."""

import numpy as np
import pytest

from source.solvers.VPM.particles.distribution import ParticleDistributor
from source.solvers.VPM.utils.flow_models import VortexRingVPM


@pytest.mark.unit
def test_toroidal_distribution_has_periodic_symmetry_and_correct_measure():
    ring_radius = 1.0
    tube_radius = 0.18
    spacing = 0.04
    positions, volumes, radii = ParticleDistributor.toroidal_distribution(
        ring_radius, tube_radius, spacing
    )

    rho = np.linalg.norm(positions[:, 1:], axis=1)
    tangent = np.column_stack(
        (np.zeros(len(positions)), -positions[:, 2] / rho, positions[:, 1] / rho)
    )
    exact_torus_volume = 2.0 * np.pi**2 * ring_radius * tube_radius**2

    assert len(positions) > 0
    assert np.all(volumes > 0.0)
    assert np.all(radii == 2.0 * spacing)
    assert np.linalg.norm((tangent * volumes[:, None]).sum(axis=0)) < 1.0e-13
    assert volumes.sum() == pytest.approx(exact_torus_volume, rel=0.08)
    assert np.max(np.sqrt(positions[:, 0] ** 2 + (rho - ring_radius) ** 2)) <= tube_radius


@pytest.mark.unit
def test_toroidal_distribution_returns_complete_contiguous_orbits():
    positions, _, _, orbit_id = ParticleDistributor.toroidal_distribution(
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
        assert np.ptp(positions[selected, 0]) < 1.0e-14
        assert np.ptp(np.linalg.norm(positions[selected, 1:], axis=1)) < 1.0e-14


@pytest.mark.unit
@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_toroidal_distribution_honors_axis_and_center(axis):
    center = np.array([0.2, -0.3, 0.4])
    positions, _, _ = ParticleDistributor.toroidal_distribution(
        0.8, 0.12, 0.05, center=center, axis=axis
    )

    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    assert positions[:, axis_index].mean() == pytest.approx(center[axis_index], abs=1.0e-14)


@pytest.mark.unit
def test_perturbed_ring_seeding_preserves_zero_vector_circulation():
    spacing = 0.04
    viscosity = np.pi / 3000.0
    core_radius = 0.1
    particle_radius = 2.0 * spacing
    represented_core = np.sqrt(core_radius**2 - particle_radius**2)
    tube_radius = represented_core * np.sqrt(-np.log(0.05))
    positions, volumes, radii = ParticleDistributor.toroidal_distribution(
        1.0,
        tube_radius,
        spacing,
        epsilon_w=0.025,
        seed=7,
        max_modes=12,
    )
    _, _, strengths = VortexRingVPM(
        viscosity=viscosity,
        ring_center=np.zeros(3),
        ring_strength=np.pi,
        ring_radius=1.0,
        ring_thickness=core_radius,
        avg_particle_radius=float(radii.mean()),
        positions=positions,
        volumes=volumes,
        epsilon_W=0.025,
        seed=7,
        max_modes=12,
        anti_diffuse_flag=True,
        normalize_circulation=True,
    )

    rho = np.linalg.norm(positions[:, 1:], axis=1)
    tangent = np.column_stack(
        (np.zeros(len(positions)), -positions[:, 2] / rho, positions[:, 1] / rho)
    )
    represented_circulation = np.sum(np.einsum("ij,ij->i", strengths, tangent) / rho) / (
        2.0 * np.pi
    )
    assert np.linalg.norm(strengths.sum(axis=0)) / np.linalg.norm(strengths, axis=1).sum() < 1.0e-14
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
