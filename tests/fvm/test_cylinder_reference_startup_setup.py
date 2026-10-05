"""Reference forcing matches the coupled benchmark without changing force scales."""

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from openonda.tutorial_runner import load_case_module

CASE = Path(__file__).resolve().parents[2] / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow"


def test_reference_and_coupled_share_startup_and_nominal_force_scales():
    reference = load_case_module(CASE / "reference_flow")
    coupled = load_case_module(CASE)
    setup, mesh = reference.build_case("phase_h004", 0.04)
    assert reference.SPAN == coupled.FVM_RESOLVED_SPAN == 1.0
    assert mesh.levels == (-0.5, 0.5)
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
    assert force.reference_area == pytest.approx(1.0)
    assert {patch.name for patch in setup.boundaries if patch.velocity_type == "slip"} == {
        "ymin",
        "ymax",
    }
    periodic = {patch.name: patch for patch in setup.boundaries if patch.velocity_type == "cyclic"}
    assert set(periodic) == {"zmin", "zmax"}
    assert periodic["zmin"].neighbour_patch == "zmax"
    assert periodic["zmax"].neighbour_patch == "zmin"
    assert all(patch.pressure_type == "cyclic" for patch in periodic.values())
    assert {
        sample.file_name for sample in setup.samplers if sample.file_name.startswith("span_")
    } == {"span_middle"}


def test_reference_runner_passes_the_same_schedule(monkeypatch):
    reference = load_case_module(CASE / "reference_flow")
    calls = []
    solver = SimpleNamespace(run=lambda **kwargs: calls.append(kwargs))
    reference.run_solver(solver, start_from="initial")
    assert len(calls) == 1
    assert calls[0]["start_from"] == "initial"
    np.testing.assert_allclose(
        calls[0]["initial_velocity"](np.array([[3.0, 0.0, 0.0]])),
        [reference.STARTUP_FREESTREAM_VELOCITY],
    )


def test_reference_parameter_study_factory_accepts_independent_grid_paths(tmp_path, monkeypatch):
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
    for name in (
        "solution/coupled",
        "drag_recovery/evidence",
        "assets/geometry",
        "study_results/evidence",
        "reference_flow/solution/reference",
        "reference_flow/solution/fvm/mesh.npz",
        "reference_flow/samples/forces",
        "reference_flow/figures/forces.png",
        "reference_flow/run.log",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    reference = tmp_path / "reference_flow"
    (reference / "setup.py").write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "assert sys.argv[1:] == ['--name', 'grid']\n"
        "assert not Path('solution').exists()\n"
        "assert not Path('samples').exists()\n"
        "assert not Path('figures').exists()\n"
        "assert not Path('run.log').exists()\n"
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "openonda.tutorial_runner",
            str(reference),
            "setup",
            "--fresh",
            "--name",
            "grid",
        ],
        check=True,
        capture_output=True,
    )
    for name in (
        "solution/coupled",
        "drag_recovery/evidence",
        "assets/geometry",
        "study_results/evidence",
    ):
        assert (tmp_path / name).read_text() == name
    archives = list((reference / "previous_runs").iterdir())
    assert len(archives) == 1
    for name in (
        "solution/reference",
        "solution/fvm/mesh.npz",
        "samples/forces",
        "figures/forces.png",
        "run.log",
    ):
        assert (archives[0] / name).read_text() == f"reference_flow/{name}"
