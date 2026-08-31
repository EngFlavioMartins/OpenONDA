"""Configuration contracts for the vortex-interactions tutorial."""

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


def test_lbm_disturbance_is_one_mode_with_the_requested_amplitude(
    interactions_setup,
):
    position = interactions_setup._single_mode_ring(-0.5)[0]
    represented_core = np.sqrt(
        interactions_setup.CORE_RADIUS**2 - interactions_setup.PARTICLE_CORE_RADIUS**2
    )
    tube_radius = represented_core * np.sqrt(-np.log(interactions_setup.TOROIDAL_TAIL_FRACTION))
    azimuth_count = max(
        8,
        int(
            np.ceil(
                2.0
                * np.pi
                * (interactions_setup.RING_RADIUS + tube_radius)
                / interactions_setup.PARTICLE_SPACING
            )
        ),
    )
    azimuth_count += (-azimuth_count) % 4
    radius = np.hypot(position[:, 1], position[:, 2]).reshape(-1, azimuth_count).mean(axis=0)
    spectrum = 2.0 * np.abs(np.fft.rfft(radius - radius.mean())) / azimuth_count

    assert len(position) == 8772
    assert np.argmax(spectrum[1:]) + 1 == interactions_setup.DISTURBANCE_MODE
    assert spectrum[interactions_setup.DISTURBANCE_MODE] == pytest.approx(
        interactions_setup.DISTURBANCE_AMPLITUDE
    )


def test_sfs_is_added_only_to_the_sfs_case(interactions_setup, tmp_path):
    baseline = interactions_setup.solver_setup("leapfrog_les", tmp_path / "baseline")
    sfs = interactions_setup.solver_setup("leapfrog_les_sfs", tmp_path / "sfs")

    assert baseline.turbulence.vortex_stretching_sfs_coefficient == 0.0
    assert not baseline.stretching.reformulated
    assert sfs.turbulence.smagorinsky_coefficient == pytest.approx(
        interactions_setup.SMAGORINSKY_COEFFICIENT
    )
    assert sfs.turbulence.vortex_stretching_sfs_coefficient == pytest.approx(
        interactions_setup.VORTEX_STRETCHING_SFS_COEFFICIENT
    )
    assert sfs.stretching.reformulated
