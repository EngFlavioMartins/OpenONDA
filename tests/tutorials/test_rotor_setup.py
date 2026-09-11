"""Source-level contracts for the ordinary rotor launcher and restart pilot."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from tutorials.vpm.rotor_flow import setup
from tutorials.vpm.rotor_flow.assets import run_matched_stabilization_pair as matched_pair
from tutorials.vpm.rotor_flow.assets import run_restart_pilot as pilot


def test_ordinary_rotor_case_keeps_native_controls() -> None:
    case = setup.build_case()

    assert case.run.steps == setup.N_STEPS
    assert case.run.initial_samples is True
    assert case.run.final_backup is True
    assert case.run.health_limit_action == "STOP"
    assert case.run.wall_time_limit_seconds is None
    assert case.run.resource_limits is not None
    assert case.run.resource_limits.max_particles == setup.RUN_PARTICLE_SOFT_LIMIT
    assert case.run.resource_limits.max_rss_bytes == setup.RUN_RSS_LIMIT_BYTES
    assert (
        case.run.resource_limits.min_available_memory_bytes
        == setup.RUN_AVAILABLE_MEMORY_FLOOR_BYTES
    )
    assert case.run.runtime_compute_device is None
    assert case.numerics.compute_device == "AUTO"
    assert case.numerics.time_step_size == setup.TIME_STEP_SIZE
    assert case.numerics.vlm.logging_interval_steps == 1
    assert case.backup.interval_steps == 4
    backup_time = case.backup.interval_steps * setup.TIME_STEP_SIZE
    assert backup_time <= 1.0 / 30.0
    assert setup.ANGULAR_VELOCITY * backup_time <= np.deg2rad(12.0)
    assert [type(item.schedule).__name__ for item in case.samplers.samples] == [
        "EveryTime",
        "EveryTime",
        "EveryTime",
    ]
    assert all(type(item).__name__ != "VLMSampler" for item in case.samplers.samples)
    assert [item.file_name for item in case.samplers.samples[1:]] == ["wake_1D", "wake_2D"]


def test_station_labels_use_authored_nominal_diameter_not_mesh_tip_radius() -> None:
    blade = json.loads((Path(setup.TUTORIAL_DIR) / "assets/blade.json").read_text())
    mesh_tip_radius = max(
        np.linalg.norm(
            (
                0.75 * np.asarray(segment["vertex_position"]["b"])
                + 0.25 * np.asarray(segment["vertex_position"]["c"])
            )[1:]
        )
        for wing in blade["wings"]
        for segment in wing["segments"]
    )
    case = setup.build_case()
    planes = case.samplers.samples[1:]

    assert np.isclose(mesh_tip_radius, 6.005739216519047)
    assert setup.STATION_REFERENCE_RADIUS == 6.0
    assert not np.isclose(mesh_tip_radius, setup.STATION_REFERENCE_RADIUS)
    assert [sampler.point[0] for sampler in planes] == [12.0, 24.0]


def test_restart_pilot_is_explicitly_bounded_and_cpu_selected() -> None:
    plan = pilot._pilot_run_plan(pilot.PILOT_STEPS)

    assert plan.steps == pilot.PILOT_STEPS
    assert plan.initial_samples is False
    assert plan.final_backup is True
    assert plan.health_limit_action == "STOP"
    assert plan.wall_time_limit_seconds == pilot.PILOT_WALL_LIMIT_SECONDS
    assert plan.runtime_compute_device == "CPU"
    assert plan.resource_limits is not None
    assert plan.resource_limits.max_particles == pilot.PILOT_PARTICLE_SOFT_LIMIT
    assert plan.resource_limits.max_rss_bytes == pilot.PILOT_RSS_LIMIT_BYTES
    assert (
        plan.resource_limits.min_available_memory_bytes == pilot.PILOT_AVAILABLE_MEMORY_FLOOR_BYTES
    )


def test_restart_preflight_has_no_scientific_samplers() -> None:
    solution_directory = Path("solution/test_pilot")
    sample_directory = Path("samples/rotor/test_pilot")
    case = setup.build_case(
        time_step_size=0.001,
        steps=pilot.PILOT_STEPS,
        solution_directory=solution_directory,
        sample_directory=sample_directory,
        run_plan=pilot._pilot_run_plan(pilot.PILOT_STEPS),
    )
    preflight = pilot._preflight_case(case, solution_directory, sample_directory, pilot.PILOT_STEPS)

    assert preflight.run.steps == pilot.PREFLIGHT_STEPS
    assert preflight.run.wall_time_limit_seconds == pilot.PREFLIGHT_WALL_LIMIT_SECONDS
    assert preflight.run.runtime_compute_device == "CPU"
    assert preflight.samplers.samples == ()
    assert preflight.backup.directory == "solution/test_pilot/preflight"
    assert preflight.samplers.directory == "rotor/test_pilot/preflight"


def test_restart_preflight_caps_steps_near_authored_endpoint() -> None:
    solution_directory = Path("solution/test_near_end")
    sample_directory = Path("samples/rotor/test_near_end")
    case = setup.build_case(
        time_step_size=0.001,
        steps=3,
        solution_directory=solution_directory,
        sample_directory=sample_directory,
        run_plan=pilot._pilot_run_plan(3),
    )
    preflight = pilot._preflight_case(case, solution_directory, sample_directory, 3)

    assert pilot._remaining_steps(setup.END_TIME - 3 * 0.001, 0.001) == 3
    assert preflight.run.steps == 3


def test_restart_pilot_help_is_lightweight() -> None:
    script = Path(__file__).parents[2] / "tutorials/vpm/rotor_flow/assets/run_restart_pilot.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=script.parent.parent,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--resume" in result.stdout
    assert "--resume-dt" in result.stdout
    assert "--attempt" in result.stdout


def test_matched_stabilization_pair_uses_fresh_public_model_variants() -> None:
    baseline = matched_pair.build_trial_case(
        "baseline",
        output_tag="matched_baseline_test",
        steps=setup.N_STEPS,
    )
    stabilized = matched_pair.build_trial_case(
        "stretching_viscosity",
        output_tag="matched_stabilized_test",
        steps=setup.N_STEPS,
    )

    assert baseline.run.runtime_compute_device == "CPU"
    assert stabilized.run.runtime_compute_device == "CPU"
    assert baseline.numerics.time_step_size == stabilized.numerics.time_step_size
    assert baseline.numerics.stabilization == setup.vpm.StabilizationConfig.disabled()
    assert (
        stabilized.numerics.stabilization.stretching_viscosity_coefficient
        == matched_pair.STRETCHING_VISCOSITY_COEFFICIENT
    )
    assert baseline.numerics.stabilization != stabilized.numerics.stabilization
    assert matched_pair._steps_to_endpoint(7.5, 9.0, setup.TIME_STEP_SIZE) == 250


def test_allrun_keeps_literal_completion_launcher() -> None:
    launcher = Path(__file__).parents[2] / "tutorials/vpm/rotor_flow/allrun.sh"
    assert launcher.read_text() == "#!/bin/bash -e\n\npython setup.py --output-tag completion\n"
