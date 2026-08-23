from __future__ import annotations

import numpy as np
import pytest

from source.solvers.vpm import (
    DivergenceRelaxationConfig,
    FilamentRefinementConfig,
    StabilizationConfig,
    VPMSetup,
)
from source.solvers.vpm.numerics.fourier_integrals import (
    gaussian_fourier_integrals,
)
from source.solvers.vpm.stabilization.divergence_relaxation import (
    DivergenceRelaxationError,
    GaussianParticleGridOperator,
    constrained_divergence_relaxation,
)
from source.solvers.vpm.stabilization.filament_refinement import (
    gaussian_particle_moments,
)


def _cloud() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    coordinates = np.linspace(-0.2, 0.2, 5)
    position = (
        np.array(np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")).reshape(3, -1).T
    )
    radius_squared = np.sum(position * position, axis=1)
    vorticity = np.column_stack((-position[:, 1], position[:, 0], np.zeros(len(position))))
    vorticity *= np.exp(-radius_squared / 0.05)[:, None]
    # Add a localized gradient component so that the field is deliberately
    # non-solenoidal and gives the relaxation something measurable to remove.
    vorticity += 0.15 * position * np.exp(-radius_squared / 0.04)[:, None]
    spacing = float(coordinates[1] - coordinates[0])
    particle_volume = np.full(len(position), spacing**3)
    circulation = vorticity * particle_volume[:, None]
    radius = np.full(len(position), 1.5 * spacing)
    return position, circulation, radius, particle_volume, spacing


def test_particle_grid_operator_is_symmetric():
    position, circulation, radius, _, spacing = _cloud()
    operator = GaussianParticleGridOperator(
        position,
        radius,
        np.linalg.norm(circulation, axis=1),
        spacing=spacing,
    )
    rng = np.random.default_rng(192)
    left = rng.normal(size=circulation.shape)
    right = rng.normal(size=circulation.shape)

    np.testing.assert_allclose(
        np.vdot(left, operator.apply(right)),
        np.vdot(operator.apply(left), right),
        rtol=2e-14,
        atol=2e-14,
    )


def test_helmholtz_projection_is_solenoidal_and_preserves_the_mean():
    position, circulation, radius, _, spacing = _cloud()
    operator = GaussianParticleGridOperator(
        position,
        radius,
        np.linalg.norm(circulation, axis=1),
        spacing=spacing,
    )
    rng = np.random.default_rng(193)
    field = rng.normal(size=(*operator.shape, 3))

    projected, divergence_before, divergence_after = operator.helmholtz_project(field)

    assert divergence_before > 0.1
    assert divergence_after < 2e-16
    np.testing.assert_allclose(
        projected.mean(axis=(0, 1, 2)),
        field.mean(axis=(0, 1, 2)),
        rtol=0.0,
        atol=2e-15,
    )


def test_relaxation_reduces_actual_divergence_and_preserves_invariants():
    position, circulation, radius, particle_volume, spacing = _cloud()
    before = gaussian_particle_moments(position, circulation, radius)
    result = constrained_divergence_relaxation(
        position,
        circulation,
        radius,
        particle_volume,
        grid_spacing=spacing,
        regularization=0.1,
        max_correction_norm=0.2,
        max_residual_ratio=0.9,
        total_kinetic_energy_tolerance=1e-12,
        total_enstrophy_tolerance=0.03,
        total_helicity_tolerance=1e-12,
        variation_tolerance=0.02,
    )
    after = gaussian_particle_moments(
        position,
        result.vortex_strength,
        radius,
    )

    assert result.final_residual_ratio < 0.3
    assert result.grid_divergence_after < 0.5 * result.grid_divergence_before
    assert result.correction_norm_relative < 0.14
    assert abs(result.total_kinetic_energy_change_relative) < 1e-12
    assert abs(result.total_enstrophy_change_relative) < 0.03
    assert abs(result.total_helicity_change_relative) < 1e-12
    assert abs(result.total_variation_change_relative) < 0.02
    np.testing.assert_allclose(after[0], before[0], rtol=0.0, atol=2e-16)
    np.testing.assert_allclose(after[2], before[2], rtol=0.0, atol=2e-17)
    np.testing.assert_allclose(after[3], before[3], rtol=0.0, atol=2e-18)


def test_relaxation_restores_reference_moments_without_energy_transfer():
    position, circulation, radius, particle_volume, spacing = _cloud()
    reference = gaussian_particle_moments(position, circulation, radius)
    drifted = circulation.copy()
    drifted[0] += np.array([2.0e-8, -1.0e-8, 1.5e-8])
    drifted_moments = gaussian_particle_moments(position, drifted, radius)

    result = constrained_divergence_relaxation(
        position,
        drifted,
        radius,
        particle_volume,
        grid_spacing=spacing,
        regularization=0.1,
        max_correction_norm=0.2,
        max_residual_ratio=0.9,
        total_kinetic_energy_tolerance=1e-12,
        total_enstrophy_tolerance=0.03,
        total_helicity_tolerance=1e-5,
        variation_tolerance=0.02,
        target_moments=(reference[0], reference[2], reference[3]),
    )
    restored = gaussian_particle_moments(
        position,
        result.vortex_strength,
        radius,
    )

    assert result.vortex_strength_restored == pytest.approx(
        np.linalg.norm(reference[0] - drifted_moments[0])
    )
    assert result.linear_impulse_restored == pytest.approx(
        np.linalg.norm(reference[2] - drifted_moments[2])
    )
    assert result.angular_impulse_restored == pytest.approx(
        np.linalg.norm(reference[3] - drifted_moments[3])
    )
    np.testing.assert_allclose(restored[0], reference[0], rtol=0.0, atol=2e-16)
    np.testing.assert_allclose(restored[2], reference[2], rtol=0.0, atol=2e-17)
    np.testing.assert_allclose(restored[3], reference[3], rtol=0.0, atol=2e-18)
    assert abs(result.total_kinetic_energy_change_relative) < 1e-12
    assert result.final_residual_ratio < 0.9


def test_physical_reference_mode_restores_energy_and_enstrophy_together():
    position, circulation, radius, particle_volume, spacing = _cloud()
    reference = gaussian_particle_moments(position, circulation, radius)
    drifted = circulation.copy()
    drifted[0] += np.array([2.0e-8, -1.0e-8, 1.5e-8])

    result = constrained_divergence_relaxation(
        position,
        drifted,
        radius,
        particle_volume,
        grid_spacing=spacing,
        regularization=0.1,
        max_correction_norm=0.2,
        max_residual_ratio=0.95,
        total_kinetic_energy_tolerance=1e-12,
        total_enstrophy_tolerance=1e-12,
        total_helicity_tolerance=1e-5,
        variation_tolerance=0.02,
        reference_scales=(1.0, 1.0, 1.0),
        target_moments=(reference[0], reference[2], reference[3]),
    )

    assert abs(result.total_kinetic_energy_change_relative) < 1e-12
    assert abs(result.total_enstrophy_change_relative) < 1e-12
    assert result.final_residual_ratio < 0.95


def test_restoration_and_divergence_amplitudes_backtrack_independently():
    position, circulation, radius, particle_volume, spacing = _cloud()
    reference = gaussian_particle_moments(position, circulation, radius)
    drifted = circulation.copy()
    drifted[0] += np.array([2.0e-6, -1.0e-6, 1.5e-6])
    before = gaussian_particle_moments(position, drifted, radius)

    result = constrained_divergence_relaxation(
        position,
        drifted,
        radius,
        particle_volume,
        grid_spacing=spacing,
        regularization=0.1,
        max_correction_norm=0.2,
        max_residual_ratio=0.9,
        total_kinetic_energy_tolerance=1e-12,
        total_enstrophy_tolerance=0.03,
        total_helicity_tolerance=1e-5,
        variation_tolerance=0.02,
        target_moments=(reference[0], reference[2], reference[3]),
    )
    after = gaussian_particle_moments(position, result.vortex_strength, radius)

    assert result.reference_restoration_scale == pytest.approx(0.0625)
    assert result.trust_region_scale < 1.0
    assert result.final_residual_ratio < 0.9
    assert abs(result.total_helicity_change_relative) < 1e-5
    assert np.linalg.norm(after[0] - reference[0]) < np.linalg.norm(before[0] - reference[0])
    assert np.linalg.norm(after[2] - reference[2]) < np.linalg.norm(before[2] - reference[2])
    assert np.linalg.norm(after[3] - reference[3]) < np.linalg.norm(before[3] - reference[3])


def test_reference_gate_rejects_a_locally_admissible_but_globally_weak_repair():
    position, circulation, radius, particle_volume, spacing = _cloud()
    reference = gaussian_particle_moments(position, circulation, radius)
    drifted = circulation.copy()
    drifted[0] += np.array([2.0e-8, -1.0e-8, 1.5e-8])
    before = gaussian_particle_moments(position, drifted, radius)
    drift_scales = tuple(
        float(np.linalg.norm(current - target))
        for current, target in zip(
            (before[0], before[2], before[3]),
            (reference[0], reference[2], reference[3]),
            strict=True,
        )
    )

    with pytest.raises(DivergenceRelaxationError, match="reference error"):
        constrained_divergence_relaxation(
            position,
            drifted,
            radius,
            particle_volume,
            grid_spacing=spacing,
            regularization=0.1,
            max_correction_norm=0.2,
            max_residual_ratio=0.9,
            total_kinetic_energy_tolerance=1e-12,
            total_enstrophy_tolerance=0.03,
            total_helicity_tolerance=1e-5,
            variation_tolerance=0.02,
            reference_scales=drift_scales,
            reference_tolerances=(1e-12, 1e-12, 1e-12),
            target_moments=(reference[0], reference[2], reference[3]),
        )


def test_iterated_projection_rejects_when_global_correction_budget_is_exhausted():
    position, circulation, radius, particle_volume, spacing = _cloud()
    with pytest.raises(DivergenceRelaxationError, match="correction norm"):
        constrained_divergence_relaxation(
            position,
            circulation,
            radius,
            particle_volume,
            grid_spacing=spacing,
            regularization=0.1,
            max_correction_norm=0.01,
            max_residual_ratio=0.9,
            total_kinetic_energy_tolerance=1.0,
            total_enstrophy_tolerance=1.0,
            total_helicity_tolerance=1.0,
            variation_tolerance=1.0,
        )


def test_iterated_projection_uses_multiple_sweeps_as_one_physics_transaction():
    position, circulation, radius, particle_volume, spacing = _cloud()
    before = gaussian_particle_moments(position, circulation, radius)

    result = constrained_divergence_relaxation(
        position,
        circulation,
        radius,
        particle_volume,
        grid_spacing=spacing,
        regularization=5.0,
        max_correction_norm=0.2,
        max_residual_ratio=0.9,
        total_kinetic_energy_tolerance=1e-12,
        total_enstrophy_tolerance=1.0,
        total_helicity_tolerance=1e-12,
        variation_tolerance=1.0,
    )
    after = gaussian_particle_moments(position, result.vortex_strength, radius)

    assert result.projection_sweeps == 2
    assert result.final_residual_ratio < 0.9
    assert result.correction_norm_relative < 0.2
    assert abs(result.total_kinetic_energy_change_relative) < 1e-12
    assert abs(result.total_helicity_change_relative) < 1e-12
    np.testing.assert_allclose(after[0], before[0], rtol=0.0, atol=2e-16)
    np.testing.assert_allclose(after[2], before[2], rtol=0.0, atol=2e-17)
    np.testing.assert_allclose(after[3], before[3], rtol=0.0, atol=2e-18)


def test_line_search_reduces_transfer_without_sacrificing_residual_gate():
    position, circulation, radius, particle_volume, spacing = _cloud()
    result = constrained_divergence_relaxation(
        position,
        circulation,
        radius,
        particle_volume,
        grid_spacing=spacing,
        regularization=0.1,
        max_correction_norm=0.2,
        max_residual_ratio=0.9,
        total_kinetic_energy_tolerance=1e-12,
        total_enstrophy_tolerance=0.01,
        total_helicity_tolerance=1e-12,
        variation_tolerance=0.02,
    )

    assert result.trust_region_scale == pytest.approx(0.25)
    assert abs(result.total_enstrophy_change_relative) < 0.01
    assert result.final_residual_ratio < 0.9


def test_relaxation_config_round_trip_and_combined_method_contract():
    refinement = FilamentRefinementConfig.adaptive(interval_steps=1)
    relaxation = DivergenceRelaxationConfig.constrained(
        interval_steps=10,
        grid_spacing=0.03,
        vortex_strength_reference_scale=1.0,
        linear_impulse_reference_scale=2.0,
        angular_impulse_reference_scale=3.0,
    )
    original = VPMSetup(
        stabilization=StabilizationConfig(
            filament_refinement=refinement,
            divergence_relaxation=relaxation,
        )
    )

    restored = VPMSetup.from_dict(original.to_dict())

    assert restored.stabilization.filament_refinement == refinement
    assert restored.stabilization.divergence_relaxation == relaxation


def test_relaxation_requires_gaussian_particles_and_filament_refinement():
    relaxation = DivergenceRelaxationConfig.constrained(
        interval_steps=10,
        grid_spacing=0.03,
    )
    with pytest.raises(ValueError, match="requires filament refinement"):
        VPMSetup(stabilization=StabilizationConfig(divergence_relaxation=relaxation))
    with pytest.raises(ValueError, match="requires GAUSSIAN particles"):
        VPMSetup(
            particle_kernel="WINCKELMANS",
            stabilization=StabilizationConfig(
                filament_refinement=FilamentRefinementConfig.adaptive(interval_steps=1),
                divergence_relaxation=relaxation,
            ),
        )


def test_particle_grid_operator_remains_symmetric_with_mixed_core_radii():
    position, circulation, radius, _, spacing = _cloud()
    radius[0] *= 1.01
    operator = GaussianParticleGridOperator(
        position,
        radius,
        np.linalg.norm(circulation, axis=1),
        spacing=spacing,
    )
    rng = np.random.default_rng(194)
    left = rng.normal(size=circulation.shape)
    right = rng.normal(size=circulation.shape)

    assert operator.radius_spread > 0.0
    np.testing.assert_allclose(
        np.vdot(left, operator.apply(right)),
        np.vdot(operator.apply(left), right),
        rtol=2e-14,
        atol=2e-14,
    )


def test_variable_core_fourier_energy_is_quadratic_and_order_audited():
    position, circulation, radius, particle_volume, spacing = _cloud()
    radius *= np.linspace(0.95, 1.05, len(radius))
    rng = np.random.default_rng(195)
    perturbation = 0.02 * rng.normal(size=circulation.shape) * np.linalg.norm(circulation)

    base = gaussian_fourier_integrals(
        position,
        circulation,
        radius,
        particle_volume,
        spacing=spacing,
    )
    plus = gaussian_fourier_integrals(
        position,
        circulation + perturbation,
        radius,
        particle_volume,
        spacing=spacing,
    )
    minus = gaussian_fourier_integrals(
        position,
        circulation - perturbation,
        radius,
        particle_volume,
        spacing=spacing,
    )
    perturbation_only = gaussian_fourier_integrals(
        position,
        perturbation,
        radius,
        particle_volume,
        spacing=spacing,
    )
    explicit_previous = gaussian_fourier_integrals(
        position,
        circulation,
        radius,
        particle_volume,
        spacing=spacing,
        radius_expansion_order=2,
    )

    np.testing.assert_allclose(
        plus.total_kinetic_energy + minus.total_kinetic_energy - 2.0 * base.total_kinetic_energy,
        2.0 * perturbation_only.total_kinetic_energy,
        rtol=2e-13,
        atol=2e-16,
    )
    assert base.radius_expansion_order == 3
    assert base.previous_order_total_kinetic_energy == pytest.approx(
        explicit_previous.total_kinetic_energy,
        rel=2e-14,
    )
    assert base.previous_order_total_enstrophy == pytest.approx(
        explicit_previous.total_enstrophy,
        rel=2e-14,
    )
    assert base.previous_order_total_helicity == pytest.approx(
        explicit_previous.total_helicity,
        rel=2e-14,
    )
