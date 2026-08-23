from __future__ import annotations

import numpy as np
import pytest

from source.solvers.vpm import VPMSetup, VPMSolver
from source.solvers.vpm.config.types import (
    AdvectionConfig,
    DivergenceRelaxationConfig,
    FilamentRefinementConfig,
    StabilizationConfig,
    StretchingConfig,
    ViscousConfig,
)
from source.solvers.vpm.io.checkpoint import CheckpointManager
from source.solvers.vpm.runtime.backend import reset_taichi_backend


def _solver(tmp_path, *, refinement: bool) -> VPMSolver:
    return VPMSolver(
        VPMSetup(
            compute_device="CPU",
            stretching=StretchingConfig.disabled(),
            viscous=ViscousConfig(scheme="NONE"),
            advection=AdvectionConfig(scheme="NONE"),
            stabilization=StabilizationConfig(
                filament_refinement=(
                    FilamentRefinementConfig.adaptive(interval_steps=1, max_n_particles=32)
                    if refinement
                    else FilamentRefinementConfig.disabled()
                )
            ),
            checkpoint_interval_steps=0,
            logging_interval_steps=0,
            checkpoint_directory=str(tmp_path),
            max_n_particles=32,
        )
    )


def _add_cloud(solver: VPMSolver, count: int) -> None:
    coordinate = np.arange(count, dtype=np.float32)
    solver.add_vortex_particles(
        position=np.column_stack((0.1 * coordinate, np.zeros(count), np.zeros(count))).astype(
            np.float32
        ),
        velocity=np.zeros((count, 3), dtype=np.float32),
        vortex_strength=np.column_stack(
            (np.zeros(count), np.zeros(count), 0.1 + coordinate)
        ).astype(np.float32),
        core_radius=np.full(count, 0.1, dtype=np.float32),
        particle_volume=np.full(count, 1e-3, dtype=np.float32),
        kinematic_viscosity=np.zeros(count, dtype=np.float32),
    )


def test_checkpoint_round_trip_preserves_material_lineage_and_transfer_audit(
    tmp_path,
):
    reset_taichi_backend()
    try:
        source = _solver(tmp_path, refinement=True)
        _add_cloud(source, 4)
        source.stabilization.capture_reference_state()
        expected_strength = np.array([0.04, 0.2, 0.7, 1.3])
        expected_length = np.array([0.03, 0.04, 0.05, 0.06])
        source.stabilization.reference_vortex_strength = expected_strength.copy()
        source.stabilization.reference_lengths = expected_length.copy()
        source.stabilization.events = 7
        source.stabilization.last_mechanism = "filament refinement"
        source.stabilization.last_vortex_strength_error = 4.2e-9
        source.stabilization.max_vorticity_growth = 1.5e-3
        source._is_particle_regeneration_pending = True
        expected_moments = tuple(value.copy() for value in source.stabilization.reference_moments)
        checkpoint = tmp_path / "lineage"
        CheckpointManager.write_checkpoint(
            source,
            str(checkpoint),
            append_step=False,
            verbose=False,
        )

        restored = _solver(tmp_path, refinement=True)
        _add_cloud(restored, 4)
        restored.stabilization.capture_reference_state()
        CheckpointManager.load_numerical_state(restored, checkpoint.with_suffix(".h5"))

        np.testing.assert_array_equal(
            restored.stabilization.reference_vortex_strength,
            expected_strength,
        )
        np.testing.assert_array_equal(
            restored.stabilization.reference_lengths,
            expected_length,
        )
        assert restored.stabilization.events == 7
        assert restored.stabilization.last_mechanism == "filament refinement"
        assert restored.stabilization.last_vortex_strength_error == pytest.approx(4.2e-9)
        assert restored.stabilization.max_vorticity_growth == pytest.approx(1.5e-3)
        assert restored._is_particle_regeneration_pending is True
        for restored_value, expected_value in zip(
            restored.stabilization.reference_moments,
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
        solver = VPMSolver(
            VPMSetup(
                compute_device="CPU",
                stretching=StretchingConfig.disabled(),
                viscous=ViscousConfig(scheme="NONE"),
                advection=AdvectionConfig(scheme="NONE"),
                stabilization=StabilizationConfig(
                    filament_refinement=FilamentRefinementConfig.adaptive(
                        interval_steps=100,
                        max_n_particles=256,
                    ),
                    divergence_relaxation=DivergenceRelaxationConfig.constrained(
                        interval_steps=1,
                        grid_spacing=spacing,
                        max_correction_norm=0.2,
                        max_residual_ratio=0.9,
                        total_kinetic_energy_tolerance=1e-6,
                        total_enstrophy_tolerance=0.03,
                        total_helicity_tolerance=1e-5,
                        variation_tolerance=0.02,
                    ),
                ),
                checkpoint_interval_steps=0,
                logging_interval_steps=0,
                checkpoint_directory=str(tmp_path),
                max_n_particles=256,
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
        particle_volume = np.full(len(position), spacing**3, dtype=np.float32)
        circulation = vorticity * particle_volume[:, None]
        solver.add_vortex_particles(
            position=position,
            velocity=np.zeros_like(position),
            vortex_strength=circulation,
            core_radius=(1.5 * spacing * np.linspace(0.995, 1.005, len(position))).astype(
                np.float32
            ),
            particle_volume=particle_volume,
            kinematic_viscosity=np.zeros(len(position), dtype=np.float32),
        )
        solver.stabilization.capture_reference_state()
        target_moments = tuple(value.copy() for value in solver.stabilization.reference_moments)
        drifted_circulation = solver.particles.vortex_strength_cpu()
        drifted_circulation[0] += np.array(
            [2.0e-8, -1.0e-8, 1.5e-8],
            dtype=np.float32,
        )
        solver.set_particles_properties(vortex_strength=drifted_circulation)
        old_circulation = solver.particles.vortex_strength_cpu()
        old_ratio = np.linalg.norm(old_circulation, axis=1) / (
            solver.stabilization.reference_vortex_strength
        )
        solver.step = 1

        solver.stabilization.apply_divergence_relaxation()

        new_circulation = solver.particles.vortex_strength_cpu()
        new_ratio = np.linalg.norm(new_circulation, axis=1) / (
            solver.stabilization.reference_vortex_strength
        )
        nonzero = np.linalg.norm(old_circulation, axis=1) > 1e-12
        np.testing.assert_allclose(
            new_ratio[nonzero],
            old_ratio[nonzero],
            rtol=2e-6,
            atol=2e-6,
        )
        assert solver.stabilization.events == 1
        assert solver.stabilization.last_mechanism == "divergence relaxation"
        assert solver.stabilization.last_vortex_strength_error < 1e-5
        assert solver.stabilization.last_vorticity_growth <= 0.05
        restored_moments = (
            solver.particles.vortex_strength_cpu().astype(np.float64).sum(axis=0),
            0.5
            * np.cross(
                solver.particles.position_cpu().astype(np.float64),
                solver.particles.vortex_strength_cpu().astype(np.float64),
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
        CheckpointManager.write_checkpoint(
            legacy,
            str(checkpoint),
            append_step=False,
            verbose=False,
        )

        restored = _solver(tmp_path, refinement=True)
        _add_cloud(restored, 4)
        restored.stabilization.capture_reference_state()
        with pytest.raises(
            ValueError, match="no filament-lineage state compatible with this refined cloud"
        ):
            CheckpointManager.load_numerical_state(
                restored,
                checkpoint.with_suffix(".h5"),
            )
    finally:
        reset_taichi_backend()


def test_solver_uploads_refinement_without_reporting_particle_deletion(tmp_path):
    reset_taichi_backend()
    try:
        solver = _solver(tmp_path, refinement=True)
        _add_cloud(solver, 4)
        solver.stabilization.capture_reference_state()
        solver.stabilization.reference_vortex_strength /= 2.1
        solver.step = 1

        solver.stabilization.apply_filament_refinement()

        assert solver.particles.n_particles_total == 8
        assert solver.stabilization.events == 1
        assert solver.stabilization.last_mechanism == "filament refinement"
        assert solver.stabilization.last_vortex_strength_error <= 512.0 * np.finfo(np.float32).eps
        assert solver._particles_removed_this_step == 0
        np.testing.assert_array_equal(
            solver._vortex_strength_removed_this_step,
            np.zeros(3),
        )
        assert len(solver.stabilization.reference_vortex_strength) == 8
        assert len(solver.stabilization.reference_lengths) == 8
    finally:
        reset_taichi_backend()
