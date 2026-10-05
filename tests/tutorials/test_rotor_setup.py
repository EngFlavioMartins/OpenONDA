"""Contracts for the ordinary rotor launcher and matched physical variants."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests._tutorial_helpers import load_tutorial_module

setup = load_tutorial_module("vpm/rotor_flow")
matched_pair = __import__(
    "tests.support.vpm.rotor_flow.run_matched_stabilization_pair", fromlist=["*"]
)


def test_ordinary_rotor_case_keeps_native_controls() -> None:
    case = setup.build_case()

    assert case.run.steps == setup.N_STEPS
    assert case.run.initial_samples is True
    assert case.run.final_backup is True
    assert case.run.health_limit_action == "STOP"
    assert case.run.wall_time_limit_seconds is None
    assert case.run.runtime_compute_device is None
    assert case.numerics.compute_device == "AUTO"
    assert case.numerics.time_step_size == setup.TIME_STEP_SIZE
    assert case.numerics.turbulence == setup.vpm.TurbulenceConfig.les_smagorinsky()
    assert case.numerics.stabilization.selective_eddy_viscosity_coefficient == 0.5
    assert case.numerics.stabilization.filament_refinement.interval_steps == 5
    assert case.numerics.stabilization.filament_refinement.max_vortex_strength_factor == 2.0
    assert case.numerics.vlm.logging_interval_steps == 1
    assert case.backup.interval_steps == 4
    backup_time = case.backup.interval_steps * setup.TIME_STEP_SIZE
    assert backup_time <= 1.0 / 30.0
    assert setup.ANGULAR_VELOCITY * backup_time <= np.deg2rad(12.0)
    assert all(type(item.schedule).__name__ == "EveryTime" for item in case.samplers.samples)
    assert all(type(item).__name__ != "VLMSampler" for item in case.samplers.samples)
    planes = [item for item in case.samplers.samples if type(item).__name__ == "SurfaceSampler"]
    assert [item.file_name for item in planes] == ["wake_0D", "wake_1D", "wake_2D"]


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
    planes = [item for item in case.samplers.samples if type(item).__name__ == "SurfaceSampler"]

    assert np.isclose(mesh_tip_radius, 6.005739216519047)
    assert setup.ROTOR_RADIUS == 6.0
    assert not np.isclose(mesh_tip_radius, setup.ROTOR_RADIUS)
    assert [sampler.point[0] for sampler in planes] == [0.0, 12.0, 24.0]
    for sampler in planes:
        np.testing.assert_allclose(sampler.grid_points[:, 0], sampler.point[0])


def test_streamwise_lines_resolve_signed_fields_through_rotor_and_wake() -> None:
    lines = [
        item for item in setup.build_case().samplers.samples if type(item).__name__ == "LineSampler"
    ]
    assert [line.file_name for line in lines] == ["streamwise_r025", "streamwise_r065"]
    for line, fraction in zip(lines, (0.25, 0.65), strict=True):
        assert type(line).__name__ == "LineSampler"
        np.testing.assert_allclose(line.start, [-12.0, fraction * 6.0, 0.0])
        np.testing.assert_allclose(line.end, [36.0, fraction * 6.0, 0.0])
        assert np.max(np.diff(line.line_points[:, 0])) <= line.spacing * 1.001
        assert line.include_derivatives is False
        assert line.schedule.interval == setup.FIELD_SAMPLE_INTERVAL_TIME


def test_matched_stabilization_pair_uses_fresh_public_model_variants() -> None:
    baseline = matched_pair.build_trial_case(
        "baseline",
        output_tag="matched_baseline_test",
        steps=setup.N_STEPS,
    )
    stabilized = matched_pair.build_trial_case(
        "selective_eddy_viscosity",
        output_tag="matched_stabilized_test",
        steps=setup.N_STEPS,
    )

    assert baseline.run.runtime_compute_device == "CPU"
    assert stabilized.run.runtime_compute_device == "CPU"
    assert baseline.numerics.time_step_size == stabilized.numerics.time_step_size
    assert baseline.numerics.stabilization == setup.vpm.StabilizationConfig.disabled()
    assert (
        stabilized.numerics.stabilization.selective_eddy_viscosity_coefficient
        == matched_pair.STRETCHING_VISCOSITY_COEFFICIENT
    )
    assert baseline.numerics.stabilization != stabilized.numerics.stabilization
    assert matched_pair._steps_to_endpoint(7.5, 9.0, setup.TIME_STEP_SIZE) == 250


def test_matched_continuation_uses_native_clock_and_absolute_target(tmp_path, monkeypatch, capsys):
    calls = []

    class Solver:
        def __init__(self, case):
            assert case.run.steps == 1500
            assert not case.run.initial_samples

        def start_from(self, checkpoint):
            calls.append(checkpoint)
            self.step, self.time = 1250, 7.5

        def run(self):
            calls.append("run")
            self.run_status = "completed"

        def close(self):
            calls.append("close")

    monkeypatch.setattr(matched_pair, "TUTORIAL_DIR", tmp_path)
    monkeypatch.setattr(matched_pair.vpm, "VPMSolver", Solver)
    matched_pair.run(
        variant="baseline",
        output_tag="continued",
        endpoint=9.0,
        resume=Path("solution/prior/vpm/vpm_001250.h5"),
    )

    assert calls == [tmp_path / "solution/prior/vpm/vpm_001250.h5", "run", "close"]
    report = capsys.readouterr().out
    assert "source_step=1250; source_time=7.5" in report
    assert "target_time=9; steps=250" in report


def test_matched_continuation_propagates_native_admission_failure(tmp_path, monkeypatch):
    calls = []

    class Solver:
        def __init__(self, _case):
            pass

        def start_from(self, _checkpoint):
            raise ValueError("native checkpoint numerical identity differs")

        def run(self):
            raise AssertionError("rejected checkpoint must not advance")

        def close(self):
            calls.append("close")

    monkeypatch.setattr(matched_pair, "TUTORIAL_DIR", tmp_path)
    monkeypatch.setattr(matched_pair.vpm, "VPMSolver", Solver)
    with pytest.raises(ValueError, match="native checkpoint numerical identity"):
        matched_pair.run(
            variant="baseline",
            output_tag="rejected",
            endpoint=9.0,
            resume=Path("solution/prior/vpm/vpm_001250.h5"),
        )
    assert calls == ["close"]


def test_allrun_cleans_then_runs_the_default_resumable_case() -> None:
    launcher = Path(__file__).parents[2] / "tutorials/vpm/06_rotor_flow/allrun.sh"
    commands = [line for line in launcher.read_text().splitlines() if line.strip()]
    assert commands[0] in ("#!/bin/bash", "#!/bin/bash -e")
    if commands[0] == "#!/bin/bash":
        assert commands.pop(1) == "set -e"
    assert commands[1:] == [
        'cd "$(dirname "$0")"',
        "./allclean.sh",
        'python -m openonda.tutorial_runner . setup "$@"',
    ]


def test_default_rotor_entrypoint_uses_native_continuation(monkeypatch) -> None:
    calls = []

    class Solver:
        def __init__(self, case):
            assert case.backup.directory == "solution"
            assert case.samplers.directory == "rotor"

        def run(self, *, start_from):
            calls.append(start_from)

    monkeypatch.setattr(setup.vpm, "VPMSolver", Solver)
    setup.run()
    assert calls == ["latest"]
