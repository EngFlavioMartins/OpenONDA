"""Reference forcing matches the coupled benchmark without changing force scales."""

from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from openonda.tutorial_runner import load_case_module

CASE = Path(__file__).resolve().parents[2] / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow"


def test_reference_and_coupled_share_startup_and_nominal_force_scales():
    reference = load_case_module(CASE / "reference_flow")
    coupled = load_case_module(CASE)
    setup, _mesh = reference.build_case("phase_h004", 0.04)
    assert reference.STARTUP_DURATION == coupled.STARTUP_DURATION == 2.0
    assert reference.STARTUP_TRANSITION_DURATION == coupled.STARTUP_TRANSITION_DURATION == 1.0
    assert reference.STARTUP_FREESTREAM_VELOCITY == coupled.STARTUP_FREESTREAM_VELOCITY
    assert tuple(reference.VELOCITY) == coupled.FREESTREAM_VELOCITY
    assert setup.initial_velocity == list(reference.STARTUP_FREESTREAM_VELOCITY)
    inlet = next(patch for patch in setup.boundaries if patch.name == "inlet")
    assert inlet.velocity_value == setup.initial_velocity
    assert setup.transport.kinematic_viscosity == pytest.approx(1 / 150)
    force = next(sample for sample in setup.samplers if sample.file_name == "forces_history")
    assert force.reference_velocity == pytest.approx(1.0)
    assert force.reference_area == pytest.approx(0.96)
    assert {patch.name for patch in setup.boundaries if patch.velocity_type == "slip"} == {
        "ymin",
        "ymax",
        "zmin",
        "zmax",
    }


def test_reference_runner_passes_the_same_schedule(monkeypatch):
    reference = load_case_module(CASE / "reference_flow")
    calls = []
    solver = object()
    monkeypatch.setattr(
        reference, "run_reference_cylinder", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    reference.run_solver(solver, start_from="initial")
    assert calls == [
        (
            (solver,),
            {
                "span": reference.SPAN,
                "start_from": "initial",
                "startup_duration": 2.0,
                "startup_transition_duration": 1.0,
                "startup_freestream_velocity": (1.0, 0.1, 0.0),
                "steady_freestream_velocity": (1.0, 0.0, 0.0),
            },
        )
    ]


def test_reference_campaign_factory_accepts_independent_grid_paths(tmp_path, monkeypatch):
    reference = load_case_module(CASE / "reference_flow")
    captured = {}
    monkeypatch.setattr(
        reference.fvm, "create_fvm_solver", lambda config, **kwargs: captured.update(kwargs)
    )
    solution = tmp_path / "solution/grid"
    samples = tmp_path / "samples/grid"
    reference.create_solver(
        "grid", 0.04, output_root=tmp_path, solution_dir=solution, samples_dir=samples
    )
    assert captured["solution_dir"] == solution
    assert captured["samples_dir"] == samples


def test_reference_fresh_archive_preserves_coupled_and_diagnostic_outputs(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    script = assets / "prepare_fresh_run.py"
    shutil.copy2(CASE / "assets/prepare_fresh_run.py", script)
    for name in (
        "solution/coupled",
        "drag_recovery/evidence",
        "reference_flow/solution/reference",
        "reference_flow/samples/forces",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    subprocess.run([sys.executable, str(script), "--reference"], check=True, capture_output=True)
    assert (tmp_path / "solution/coupled").read_text() == "solution/coupled"
    assert (tmp_path / "drag_recovery/evidence").read_text() == "drag_recovery/evidence"
    assert not (tmp_path / "reference_flow/solution").exists()
    assert not (tmp_path / "reference_flow/samples").exists()
    archives = list((tmp_path / "reference_flow/previous_runs").iterdir())
    assert len(archives) == 1
    assert (archives[0] / "solution/reference").read_text() == "reference_flow/solution/reference"
    assert (archives[0] / "samples/forces").read_text() == "reference_flow/samples/forces"
