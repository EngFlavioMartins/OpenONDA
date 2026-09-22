"""Explicit portable backends propagate through campaign workers."""

import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.mark.parametrize("asset", ["run_pipeline", "run_sensitivity"])
def test_backend_and_core_budget_reach_every_coupled_worker(asset, tmp_path, monkeypatch):
    path = (
        Path(__file__).resolve().parents[2]
        / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/assets"
        / f"{asset}.py"
    )
    spec = importlib.util.spec_from_file_location(asset, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    cost_roots = []

    def run_trial(command, directory, **kwargs):
        calls.append(command)
        return {"returncode": 0, "wall_seconds": 1.0, "console_log": "unused"}

    monkeypatch.setattr(module, "run_trial", run_trial)

    def collect_cost(root):
        cost_roots.append(root)
        return {"unconverged_stationary_intervals": 0}

    monkeypatch.setattr(module, "collect_cost", collect_cost)
    monkeypatch.setattr(module, "load_case_module", lambda *args: object())
    mode = (
        ["--pilot", "--sensitivity", "none", "--reference-cores", "1"]
        if asset == "run_pipeline"
        else ["--screen", "--factor", "span"]
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            asset,
            "--run-dir",
            str(tmp_path / "run"),
            "--compute-device",
            "CPU",
            "--coupled-cores",
            "2",
            *mode,
        ],
    )

    assert module.main() == 0
    coupled = [command for command in calls if command[command.index("--kind") + 1] == "coupled"]
    assert coupled
    assert all("compute_device=CPU" in command and "cores=2" in command for command in coupled)
    if asset == "run_pipeline":
        reference = next(
            command for command in calls if command[command.index("--kind") + 1] == "reference"
        )
        assert reference[reference.index("--reference-cores") + 1] == "1"
        assert cost_roots[0] == tmp_path / "run/reference/solution/grid_h0p1"


def test_reference_factory_accepts_serial_execution_without_mpi_allocation(monkeypatch):
    from openonda.tutorial_runner import load_case_module

    directory = (
        Path(__file__).resolve().parents[2]
        / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/reference_flow"
    )
    module = load_case_module(directory)
    monkeypatch.setattr(module.fvm, "create_fvm_solver", lambda setup, **kwargs: setup)
    setup = module.create_solver("serial_test", 0.1, cores=1)
    assert setup.cores == 1
