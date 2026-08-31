"""Checks for the vortex-interactions tutorial configuration."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SETUP_PATH = (
    Path(__file__).resolve().parents[2]
    / "tutorials"
    / "vpm"
    / "vortex_interactions"
    / "interactions_setup.py"
)


@pytest.fixture(scope="module")
def interactions_setup():
    spec = importlib.util.spec_from_file_location("vortex_interactions_setup", SETUP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_cases_use_transposed_coupled_rk2(interactions_setup, tmp_path):
    for case_name in interactions_setup.CASES:
        setup = interactions_setup.solver_setup(case_name, tmp_path / case_name)
        assert setup.time_integration == "COUPLED"
        assert setup.advection.scheme == "RK2"
        assert setup.stretching.scheme == "RK2"
        assert setup.stretching.mode == "TRANSPOSED"
        assert setup.stabilization.max_lagrangian_cfl == pytest.approx(
            interactions_setup.MAX_LAGRANGIAN_CFL
        )


def test_runner_leaves_instability_detection_to_the_solver():
    source = SETUP_PATH.read_text(encoding="utf-8")

    assert "run_manifest" not in source
    assert "termination_reason" not in source
    assert "current_peak" not in source
    assert "vortex_strength_cpu" not in source


def test_ring_uses_the_vpm_initialization_workflow():
    source = SETUP_PATH.read_text(encoding="utf-8")

    assert "vpm.create_rectangular_distribution" in source
    assert "vpm.initialize_vortex_ring" in source
    assert "solver.remove_weak_particles" in source
    assert "create_toroidal_distribution" not in source
    assert "np.arctan2(position" not in source
    assert "represented_circulation =" not in source


def test_stabilization_cases_use_the_requested_absolute_criteria(interactions_setup, tmp_path):
    peak = interactions_setup.initial_peak_strength()
    splitting = interactions_setup.solver_setup(
        "leapfrog_les_splitting", tmp_path / "splitting"
    ).stabilization.filament_refinement
    combined = interactions_setup.solver_setup(
        "leapfrog_les_splitting_remeshing", tmp_path / "combined"
    ).stabilization

    assert np.isinf(splitting.max_vortex_strength_factor)
    assert splitting.max_absolute_vortex_strength == pytest.approx(2.0 * peak)
    assert combined.filament_refinement.max_absolute_vortex_strength == pytest.approx(2.0 * peak)
    assert combined.regularization_divergence_trigger is None
    assert combined.regularization_misalignment_trigger is None
    assert combined.regularization_core_radius_trigger == pytest.approx(
        2.0 * interactions_setup.PARTICLE_CORE_RADIUS
    )


def test_core_spreading_reaches_the_remesh_threshold_after_450_steps(
    interactions_setup,
):
    sigma_squared = interactions_setup.PARTICLE_CORE_RADIUS**2 + (
        4.0
        * interactions_setup.KINEMATIC_VISCOSITY
        * interactions_setup.TIME_STEP_SIZE
        * interactions_setup.REMESH_INTERVAL_STEPS
    )
    assert interactions_setup.REMESH_INTERVAL_STEPS == 450
    assert np.sqrt(sigma_squared) == pytest.approx(interactions_setup.REMESH_CORE_RADIUS_TRIGGER)


def test_widnall_disturbance_changes_vorticity_not_particle_geometry(
    interactions_setup, monkeypatch
):
    particles = interactions_setup._create_ring(-0.5)
    position = particles.position
    points_per_axis = [len(np.unique(position[:, axis])) for axis in range(3)]
    assert len(position) == np.prod(points_per_axis)

    monkeypatch.setattr(interactions_setup, "DISTURBANCE_AMPLITUDE", 0.0)
    alternate = interactions_setup._create_ring(-0.5)

    np.testing.assert_array_equal(alternate.position, position)
    np.testing.assert_array_equal(alternate.particle_volume, particles.particle_volume)
    np.testing.assert_array_equal(alternate.core_radius, particles.core_radius)
    assert not np.allclose(alternate.vortex_strength, particles.vortex_strength)


def test_all_cases_use_the_calibrated_smagorinsky_model(interactions_setup, tmp_path):
    for case_name in interactions_setup.CASES:
        turbulence = interactions_setup.solver_setup(case_name, tmp_path / case_name).turbulence
        assert turbulence.model == "LES_SMAGORINSKY"
        assert turbulence.smagorinsky_coefficient == pytest.approx(
            interactions_setup.SMAGORINSKY_COEFFICIENT
        )
