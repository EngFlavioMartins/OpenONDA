from __future__ import annotations

import numpy as np
import pytest

from source.solvers.VPM import FilamentRefinementConfig, StabilizationConfig, VPMSetup
from source.solvers.VPM.stabilization.filament_refinement import (
    FilamentRefinementError,
    gaussian_particle_moments,
    gaussian_refinement_integral_transfer,
    split_stretched_filaments,
)


def _particles() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(81)
    position = rng.normal(size=(32, 3))
    circulation = rng.normal(size=(32, 3))
    circulation[:12] *= 4.0
    radius = rng.uniform(0.08, 0.13, size=32)
    volume = rng.uniform(1e-5, 4e-5, size=32)
    return position, circulation, radius, volume


def _reference_state(
    circulation: np.ndarray,
    volume: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    reference_strength = np.linalg.norm(circulation, axis=1).copy()
    reference_strength[:12] *= 0.2
    return reference_strength, np.cbrt(volume)


def _direct_gaussian_integrals(
    position: np.ndarray,
    circulation: np.ndarray,
    radius: np.ndarray,
) -> np.ndarray:
    from scipy.special import erf

    values = np.zeros(3)
    for i in range(len(position)):
        for j in range(len(position)):
            displacement = position[i] - position[j]
            distance = np.linalg.norm(displacement)
            sigma = np.sqrt(radius[i] ** 2 + radius[j] ** 2)
            density = distance / sigma
            dot_product = circulation[i] @ circulation[j]
            if density < 1e-12:
                energy_kernel = 1.0 / (2.0 * np.pi**1.5)
            else:
                energy_kernel = erf(density) / (4.0 * np.pi * density)
            values[0] += 0.5 * dot_product * energy_kernel / sigma
            values[1] += dot_product * np.exp(-(density**2)) / (np.pi**1.5 * sigma**3)
            if distance >= 1e-12:
                q_value = (
                    erf(density) - 2.0 / np.sqrt(np.pi) * density * np.exp(-(density**2))
                ) / (4.0 * np.pi)
                values[2] += (
                    q_value
                    * displacement.dot(np.cross(circulation[i], circulation[j]))
                    / distance**3
                )
    return values


def test_axial_split_preserves_gaussian_blob_moments_and_volume():
    position, circulation, radius, volume = _particles()
    reference_strength, reference_length = _reference_state(circulation, volume)
    before = gaussian_particle_moments(position, circulation, radius)
    result = split_stretched_filaments(
        position,
        circulation,
        radius,
        volume,
        reference_vortex_strength=reference_strength,
        reference_length=reference_length,
        max_stretch_factor=2.0,
    )
    after = gaussian_particle_moments(result.position, result.vortex_strength, result.radius)

    assert result.refined_particles > 0
    np.testing.assert_allclose(after[0], before[0], rtol=0.0, atol=2e-14)
    np.testing.assert_allclose(after[1], before[1], rtol=0.0, atol=2e-14)
    np.testing.assert_allclose(after[2], before[2], rtol=0.0, atol=2e-14)
    np.testing.assert_allclose(after[3], before[3], rtol=0.0, atol=3e-14)
    np.testing.assert_allclose(result.volume.sum(), volume.sum(), rtol=0.0, atol=2e-19)


def test_children_are_symmetric_and_parallel_to_parent_circulation():
    position = np.array([[0.4, -0.2, 0.1]])
    circulation = np.array([[2.0, -3.0, 6.0]])
    radius = np.array([0.12])
    volume = np.array([8e-6])
    reference_strength = np.array([2.0])
    reference_length = np.array([0.04])
    result = split_stretched_filaments(
        position,
        circulation,
        radius,
        volume,
        reference_vortex_strength=reference_strength,
        reference_length=reference_length,
        max_stretch_factor=2.0,
    )

    offsets = result.position - position[0]
    np.testing.assert_allclose(offsets[0], -offsets[1], rtol=0.0, atol=1e-16)
    np.testing.assert_allclose(np.cross(offsets[0], circulation[0]), 0.0, atol=2e-16)
    np.testing.assert_allclose(
        result.vortex_strength,
        np.repeat(0.5 * circulation, 2, axis=0),
    )
    np.testing.assert_allclose(result.radius, radius[0])
    np.testing.assert_allclose(result.volume, 0.5 * volume[0])
    expected_current_length = reference_length[0] * np.linalg.norm(circulation[0]) / 2.0
    np.testing.assert_allclose(np.linalg.norm(offsets, axis=1), 0.25 * expected_current_length)
    np.testing.assert_allclose(
        result.reference_vortex_strength,
        0.5 * np.linalg.norm(circulation[0]),
    )
    np.testing.assert_allclose(result.reference_length, 0.5 * expected_current_length)


def test_isolated_split_energy_is_strictly_non_increasing():
    position, circulation, radius, volume = _particles()
    reference_strength, reference_length = _reference_state(circulation, volume)
    result = split_stretched_filaments(
        position,
        circulation,
        radius,
        volume,
        reference_vortex_strength=reference_strength,
        reference_length=reference_length,
        max_stretch_factor=2.0,
    )
    assert result.isolated_energy_change < 0.0

    colocated = split_stretched_filaments(
        position,
        circulation,
        radius,
        volume,
        reference_vortex_strength=reference_strength,
        reference_length=reference_length,
        max_stretch_factor=2.0,
        offset_fraction=0.0,
    )
    assert colocated.isolated_energy_change == pytest.approx(0.0, abs=2e-15)


def test_transfer_audit_matches_full_gaussian_pair_integrals():
    position, circulation, radius, volume = _particles()
    position = position[:9]
    circulation = circulation[:9]
    radius = radius[:9]
    volume = volume[:9]
    reference_strength, reference_length = _reference_state(circulation, volume)
    result = split_stretched_filaments(
        position,
        circulation,
        radius,
        volume,
        reference_vortex_strength=reference_strength,
        reference_length=reference_length,
        max_stretch_factor=2.0,
    )

    transfer = gaussian_refinement_integral_transfer(
        position,
        circulation,
        radius,
        result,
    )
    expected = _direct_gaussian_integrals(
        result.position,
        result.vortex_strength,
        result.radius,
    ) - _direct_gaussian_integrals(position, circulation, radius)

    np.testing.assert_allclose(
        [
            transfer.energy_change,
            transfer.enstrophy_change,
            transfer.helicity_change,
        ],
        expected,
        rtol=2e-13,
        atol=2e-13,
    )


def test_noop_transfer_has_exactly_zero_integral_jump():
    position, circulation, radius, volume = _particles()
    result = split_stretched_filaments(
        position,
        circulation,
        radius,
        volume,
        reference_vortex_strength=np.linalg.norm(circulation, axis=1),
        reference_length=np.cbrt(volume),
        max_stretch_factor=2.0,
    )
    transfer = gaussian_refinement_integral_transfer(
        position,
        circulation,
        radius,
        result,
    )
    assert transfer.energy_change == 0.0
    assert transfer.enstrophy_change == 0.0
    assert transfer.helicity_change == 0.0


def test_refinement_rejects_an_insufficient_particle_budget():
    position, circulation, radius, volume = _particles()
    reference_strength, reference_length = _reference_state(circulation, volume)
    with pytest.raises(FilamentRefinementError, match="declared budget"):
        split_stretched_filaments(
            position,
            circulation,
            radius,
            volume,
            reference_vortex_strength=reference_strength,
            reference_length=reference_length,
            max_stretch_factor=2.0,
            max_particles=len(position),
        )


def test_particles_below_threshold_are_an_exact_noop():
    position, circulation, radius, volume = _particles()
    reference_strength = np.linalg.norm(circulation, axis=1)
    reference_length = np.cbrt(volume)
    result = split_stretched_filaments(
        position,
        circulation,
        radius,
        volume,
        reference_vortex_strength=reference_strength,
        reference_length=reference_length,
        max_stretch_factor=2.0,
    )
    assert result.refined_particles == 0
    np.testing.assert_array_equal(result.position, position)
    np.testing.assert_array_equal(result.vortex_strength, circulation)
    np.testing.assert_array_equal(result.radius, radius)
    np.testing.assert_array_equal(result.volume, volume)
    np.testing.assert_array_equal(result.reference_vortex_strength, reference_strength)
    np.testing.assert_array_equal(result.reference_length, reference_length)


def test_children_reset_their_own_stretch_reference():
    position = np.zeros((1, 3))
    circulation = np.array([[0.0, 0.0, 4.2]])
    radius = np.array([0.1])
    volume = np.array([1e-3])
    result = split_stretched_filaments(
        position,
        circulation,
        radius,
        volume,
        reference_vortex_strength=np.array([2.0]),
        reference_length=np.array([0.1]),
        max_stretch_factor=2.0,
    )

    second = split_stretched_filaments(
        result.position,
        result.vortex_strength,
        result.radius,
        result.volume,
        reference_vortex_strength=result.reference_vortex_strength,
        reference_length=result.reference_length,
        max_stretch_factor=2.0,
    )
    assert second.refined_particles == 0


def test_axial_split_preserves_strength_weighted_centroid():
    position, circulation, radius, volume = _particles()
    reference_strength, reference_length = _reference_state(circulation, volume)
    weights = np.linalg.norm(circulation, axis=1)
    centroid_before = (weights[:, None] * position).sum(axis=0) / weights.sum()
    result = split_stretched_filaments(
        position,
        circulation,
        radius,
        volume,
        reference_vortex_strength=reference_strength,
        reference_length=reference_length,
        max_stretch_factor=2.0,
    )
    weights_after = np.linalg.norm(result.vortex_strength, axis=1)
    centroid_after = (weights_after[:, None] * result.position).sum(axis=0) / weights_after.sum()
    np.testing.assert_allclose(centroid_after, centroid_before, rtol=0.0, atol=2e-15)


def test_filament_refinement_configuration_round_trip():
    setup = VPMSetup(
        stabilization=StabilizationConfig(
            filament_refinement=FilamentRefinementConfig.adaptive(
                interval_steps=10,
                max_vortex_strength_factor=2.5,
                offset_fraction=0.4,
                max_particles=120_000,
            )
        )
    )
    restored = VPMSetup.from_dict(setup.to_dict())
    assert restored.stabilization == setup.stabilization


def test_filament_refinement_rejects_a_non_gaussian_kernel():
    with pytest.raises(ValueError, match="requires GAUSSIAN particles"):
        VPMSetup(
            particle_kernel="WINCKELMANS",
            stabilization=StabilizationConfig(
                filament_refinement=FilamentRefinementConfig.adaptive(interval_steps=1)
            ),
        )


def test_filament_refinement_rejects_children_outside_the_parent_segment():
    with pytest.raises(ValueError, match=r"offset_fraction must be in \[0, 0.5\]"):
        FilamentRefinementConfig.adaptive(
            interval_steps=1,
            offset_fraction=0.6,
        )


def test_refinement_budget_must_fit_the_fixed_particle_allocation():
    with pytest.raises(ValueError, match="cannot exceed VPMSetup.max_particles"):
        VPMSetup(
            max_particles=100,
            stabilization=StabilizationConfig(
                filament_refinement=FilamentRefinementConfig.adaptive(
                    interval_steps=1,
                    max_particles=101,
                )
            ),
        )
