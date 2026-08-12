from __future__ import annotations

import numpy as np
import pytest

from source.solvers.VPM import Solver, VPMSetup
from source.solvers.VPM.config.backend import reset_taichi_backend
from source.solvers.VPM.config.types import (
    AdvectionConfig,
    DivergenceRelaxationConfig,
    FilamentRefinementConfig,
    StretchingConfig,
    ViscousConfig,
)
from source.solvers.VPM.io.backup import BackupSystem
from source.solvers.VPM.numerics.filament_refinement import (
    FilamentRefinementError,
)


def _solver(tmp_path, *, refinement: bool, permissive_transfer: bool = False) -> Solver:
    return Solver(
        VPMSetup(
            processing_unit="CPU",
            stretching=StretchingConfig.disabled(),
            viscous=ViscousConfig(scheme="NONE"),
            advection=AdvectionConfig(scheme="NONE"),
            filament_refinement=(
                FilamentRefinementConfig.adaptive(
                    frequency=1,
                    max_particles=32,
                    energy_injection_tolerance=(1.0 if permissive_transfer else 1e-4),
                    energy_dissipation_tolerance=(1.0 if permissive_transfer else 1e-4),
                    enstrophy_transfer_tolerance=(1.0 if permissive_transfer else 1e-4),
                    helicity_transfer_tolerance=(1.0 if permissive_transfer else 1e-4),
                    cumulative_energy_tolerance=(1.0 if permissive_transfer else 1e-3),
                    cumulative_enstrophy_tolerance=(1.0 if permissive_transfer else 2e-2),
                    cumulative_helicity_tolerance=(1.0 if permissive_transfer else 2e-2),
                )
                if refinement
                else FilamentRefinementConfig.disabled()
            ),
            backup_frequency=0,
            logging_frequency=0,
            backup_directory=str(tmp_path),
            max_particles=32,
        )
    )


def _add_cloud(solver: Solver, count: int) -> None:
    coordinate = np.arange(count, dtype=np.float32)
    solver.add_vortex_particles(
        position=np.column_stack((0.1 * coordinate, np.zeros(count), np.zeros(count))).astype(
            np.float32
        ),
        velocity=np.zeros((count, 3), dtype=np.float32),
        circulation=np.column_stack((np.zeros(count), np.zeros(count), 0.1 + coordinate)).astype(
            np.float32
        ),
        radius=np.full(count, 0.1, dtype=np.float32),
        volume=np.full(count, 1e-3, dtype=np.float32),
        viscosity=np.zeros(count, dtype=np.float32),
    )


def test_checkpoint_round_trip_preserves_material_lineage_and_transfer_audit(
    tmp_path,
):
    reset_taichi_backend()
    try:
        source = _solver(tmp_path, refinement=True)
        _add_cloud(source, 4)
        source._capture_filament_refinement_reference()
        expected_strength = np.array([0.04, 0.2, 0.7, 1.3])
        expected_length = np.array([0.03, 0.04, 0.05, 0.06])
        source._filament_reference_strengths = expected_strength.copy()
        source._filament_reference_lengths = expected_length.copy()
        source._filament_refinement_diagnostics["refinement_events"] = 7
        source._divergence_relaxation_diagnostics["relaxation_events"] = 3
        source._divergence_relaxation_diagnostics["relaxation_last_residual_ratio"] = 0.42
        source._regularization_diagnostics["regularization_events"] = 2
        source._regularization_diagnostics["regularization_cumulative_energy_transfer"] = -0.125
        source._regularization_diagnostics["regularization_last_step"] = 75
        source._core_spreading_diagnostics["core_spreading_moment_projection_events"] = 11
        source._core_spreading_diagnostics["core_spreading_max_angular_impulse_error_relative"] = (
            2.5e-13
        )
        source._filament_refinement_cumulative_energy_transfer = -2.5e-6
        source._filament_refinement_energy_reference = 1.75
        expected_moments = tuple(
            value.copy() for value in source._divergence_relaxation_reference_moments
        )
        checkpoint = tmp_path / "lineage"
        BackupSystem.backup_solver(
            source,
            str(checkpoint),
            append_step=False,
            verbose=False,
        )

        restored = _solver(tmp_path, refinement=True)
        _add_cloud(restored, 4)
        restored._capture_filament_refinement_reference()
        BackupSystem.load_numerical_state(restored, checkpoint.with_suffix(".h5"))

        np.testing.assert_array_equal(
            restored._filament_reference_strengths,
            expected_strength,
        )
        np.testing.assert_array_equal(
            restored._filament_reference_lengths,
            expected_length,
        )
        assert restored._filament_refinement_diagnostics["refinement_events"] == 7
        assert restored._divergence_relaxation_diagnostics["relaxation_events"] == 3
        assert restored._divergence_relaxation_diagnostics[
            "relaxation_last_residual_ratio"
        ] == pytest.approx(0.42)
        assert restored._regularization_diagnostics["regularization_events"] == 2
        assert restored._regularization_diagnostics[
            "regularization_cumulative_energy_transfer"
        ] == pytest.approx(-0.125)
        assert restored._regularization_diagnostics["regularization_last_step"] == 75
        assert restored._core_spreading_diagnostics["core_spreading_moment_projection_events"] == 11
        assert restored._core_spreading_diagnostics[
            "core_spreading_max_angular_impulse_error_relative"
        ] == pytest.approx(2.5e-13)
        assert restored._filament_refinement_cumulative_energy_transfer == pytest.approx(-2.5e-6)
        assert restored._filament_refinement_energy_reference == pytest.approx(1.75)
        assert restored._filament_refinement_enstrophy_reference == pytest.approx(
            source._filament_refinement_enstrophy_reference
        )
        for restored_value, expected_value in zip(
            restored._divergence_relaxation_reference_moments,
            expected_moments,
            strict=True,
        ):
            np.testing.assert_array_equal(restored_value, expected_value)
    finally:
        reset_taichi_backend()


def test_solver_uploads_relaxation_and_preserves_lineage_stretch_ratio(
    tmp_path,
):
    reset_taichi_backend()
    try:
        spacing = 0.1
        solver = Solver(
            VPMSetup(
                processing_unit="CPU",
                stretching=StretchingConfig.disabled(),
                viscous=ViscousConfig(scheme="NONE"),
                advection=AdvectionConfig(scheme="NONE"),
                filament_refinement=FilamentRefinementConfig.adaptive(
                    frequency=100,
                    max_particles=256,
                ),
                divergence_relaxation=DivergenceRelaxationConfig.constrained(
                    frequency=1,
                    grid_spacing=spacing,
                    max_correction_norm=0.2,
                    max_residual_ratio=0.9,
                    energy_tolerance=1e-6,
                    enstrophy_tolerance=0.03,
                    helicity_tolerance=1e-5,
                    variation_tolerance=0.02,
                    cumulative_enstrophy_tolerance=0.03,
                ),
                backup_frequency=0,
                logging_frequency=0,
                backup_directory=str(tmp_path),
                max_particles=256,
            )
        )
        coordinates = np.linspace(-0.2, 0.2, 5, dtype=np.float32)
        position = (
            np.array(
                np.meshgrid(
                    coordinates,
                    coordinates,
                    coordinates,
                    indexing="ij",
                )
            )
            .reshape(3, -1)
            .T
        )
        radius_squared = np.sum(position * position, axis=1)
        vorticity = np.column_stack(
            (
                -position[:, 1],
                position[:, 0],
                np.zeros(len(position), dtype=np.float32),
            )
        )
        vorticity *= np.exp(-radius_squared / 0.05)[:, None]
        vorticity += 0.15 * position * np.exp(-radius_squared / 0.04)[:, None]
        volume = np.full(len(position), spacing**3, dtype=np.float32)
        circulation = vorticity * volume[:, None]
        solver.add_vortex_particles(
            position=position,
            velocity=np.zeros_like(position),
            circulation=circulation,
            radius=(1.5 * spacing * np.linspace(0.995, 1.005, len(position))).astype(np.float32),
            volume=volume,
            viscosity=np.zeros(len(position), dtype=np.float32),
        )
        solver._capture_filament_refinement_reference()
        target_moments = tuple(
            value.copy() for value in solver._divergence_relaxation_reference_moments
        )
        drifted_circulation = solver.particles.circulation_cpu()
        drifted_circulation[0] += np.array(
            [2.0e-8, -1.0e-8, 1.5e-8],
            dtype=np.float32,
        )
        solver.set_particles_properties(strengths=drifted_circulation)
        old_circulation = solver.particles.circulation_cpu()
        old_ratio = np.linalg.norm(old_circulation, axis=1) / (solver._filament_reference_strengths)
        solver.time_step = 1

        solver._apply_divergence_relaxation()

        new_circulation = solver.particles.circulation_cpu()
        new_ratio = np.linalg.norm(new_circulation, axis=1) / (solver._filament_reference_strengths)
        nonzero = np.linalg.norm(old_circulation, axis=1) > 1e-12
        np.testing.assert_allclose(
            new_ratio[nonzero],
            old_ratio[nonzero],
            rtol=2e-6,
            atol=2e-6,
        )
        diagnostics = solver._divergence_relaxation_diagnostics
        assert diagnostics["relaxation_events"] == 1
        assert diagnostics["relaxation_last_residual_ratio"] < 0.3
        assert (
            diagnostics["relaxation_grid_divergence_after"]
            < 0.5 * diagnostics["relaxation_grid_divergence_before"]
        )
        assert diagnostics["relaxation_last_energy_spectral_error"] < 1e-7
        assert diagnostics["relaxation_last_enstrophy_spectral_error"] < 3e-3
        assert diagnostics["relaxation_last_helicity_spectral_error"] < 1e-6
        restored_moments = (
            solver.particles.circulation_cpu().astype(np.float64).sum(axis=0),
            0.5
            * np.cross(
                solver.particles.position_cpu().astype(np.float64),
                solver.particles.circulation_cpu().astype(np.float64),
            ).sum(axis=0),
        )
        np.testing.assert_allclose(
            restored_moments[0],
            target_moments[0],
            rtol=0.0,
            atol=2e-8,
        )
        np.testing.assert_allclose(
            restored_moments[1],
            target_moments[1],
            rtol=0.0,
            atol=2e-8,
        )
    finally:
        reset_taichi_backend()


def test_legacy_checkpoint_cannot_silently_reset_an_already_refined_lineage(
    tmp_path,
):
    reset_taichi_backend()
    try:
        legacy = _solver(tmp_path, refinement=False)
        _add_cloud(legacy, 5)
        checkpoint = tmp_path / "legacy"
        BackupSystem.backup_solver(
            legacy,
            str(checkpoint),
            append_step=False,
            verbose=False,
        )

        restored = _solver(tmp_path, refinement=True)
        _add_cloud(restored, 4)
        restored._capture_filament_refinement_reference()
        with pytest.raises(ValueError, match="reset its material stretch history"):
            BackupSystem.load_numerical_state(
                restored,
                checkpoint.with_suffix(".h5"),
            )
    finally:
        reset_taichi_backend()


def test_solver_uploads_refinement_without_reporting_particle_deletion(tmp_path):
    reset_taichi_backend()
    try:
        solver = _solver(
            tmp_path,
            refinement=True,
            permissive_transfer=True,
        )
        _add_cloud(solver, 4)
        solver._capture_filament_refinement_reference()
        solver._filament_reference_strengths /= 2.1
        solver.time_step = 1

        solver._apply_filament_refinement()

        assert solver.particles.number_of_particles == 8
        assert solver._filament_refinement_diagnostics["refinement_events"] == 1
        assert solver._filament_refinement_diagnostics["refinement_parents_total"] == 4
        assert solver._particles_removed_this_step == 0
        np.testing.assert_array_equal(
            solver._circulation_removed_this_step,
            np.zeros(3),
        )
        for name in (
            "refinement_max_circulation_error_relative",
            "refinement_max_strength_variation_error_relative",
            "refinement_max_linear_impulse_error_relative",
            "refinement_max_angular_impulse_error_relative",
        ):
            assert solver._filament_refinement_diagnostics[name] <= (
                512.0 * np.finfo(np.float32).eps
            )
        assert len(solver._filament_reference_strengths) == 8
        assert len(solver._filament_reference_lengths) == 8
    finally:
        reset_taichi_backend()


def test_solver_rejects_cumulative_refinement_transfer_before_upload(
    tmp_path,
):
    reset_taichi_backend()
    try:
        solver = _solver(
            tmp_path,
            refinement=True,
            permissive_transfer=True,
        )
        _add_cloud(solver, 4)
        solver._capture_filament_refinement_reference()
        solver._filament_reference_strengths /= 2.1
        solver.time_step = 1
        solver._filament_refinement_diagnostics["refinement_cumulative_abs_energy_change"] = (
            solver.filament_refinement_config.cumulative_energy_tolerance
        )

        with pytest.raises(
            FilamentRefinementError,
            match="cumulative absolute energy transfer",
        ):
            solver._apply_filament_refinement()

        assert solver.particles.number_of_particles == 4
    finally:
        reset_taichi_backend()
