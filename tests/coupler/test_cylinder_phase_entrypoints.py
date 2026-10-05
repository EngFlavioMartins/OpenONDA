"""Phase runners share the startup run_stages and preserve bounded output."""

from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from openonda.tutorial_runner import load_case_module

SUPPORT = Path(__file__).resolve().parents[1] / "support/cylinder"
CASE = Path(__file__).resolve().parents[2] / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow"
setup = load_case_module(CASE)


@pytest.fixture
def driver():
    return load_case_module(SUPPORT, "run_phase_benchmark")


@pytest.mark.parametrize("module", ["run_phase_benchmark", "check_phase_samples"])
def test_support_entrypoints_use_module_runner_without_starting_solver(tmp_path, module):
    result = subprocess.run(
        [sys.executable, "-m", "openonda.tutorial_runner", str(SUPPORT), module, "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "{reference,coupled}" in result.stdout
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("resume", [False, True])
def test_phase_reference_uses_reference_factory_and_startup_runner(
    driver, tmp_path, monkeypatch, resume
):
    calls = []
    solver = SimpleNamespace(run_status="complete", time=driver.case.END, step=12500)

    def create(name, h, **kwargs):
        calls.append((name, h, kwargs))
        return nullcontext(solver)

    def run(actual, **kwargs):
        assert actual is solver
        calls.append(kwargs)

    reference = SimpleNamespace(create_solver=create, run_solver=run)
    monkeypatch.setattr(driver.case, "load_case_module", lambda *args: reference)
    args = ["phase", "reference", "--root", str(tmp_path)] + (["--resume"] if resume else [])
    monkeypatch.setattr(sys, "argv", args)
    driver.main()
    assert calls[0] == (
        "phase_h004",
        0.04,
        {"output_root": tmp_path / "reference", "end_time": driver.case.END},
    )
    assert calls[1] == {"start_from": "latest" if resume else "initial"}
    path = tmp_path / "reference" / ("continuation-result.json" if resume else "run-result.json")
    assert json.loads(path.read_text())["status"] == "completed"


@pytest.mark.parametrize(
    "pilot,resume,last", [(True, False, 20), (True, True, 40), (False, True, 2500)]
)
def test_phase_coupled_uses_shared_schedule_and_total_pilot_cap(
    driver, tmp_path, monkeypatch, pilot, resume, last
):
    captured = {}
    factory_calls = []
    monkeypatch.setattr(
        driver.case,
        "coupled_case",
        lambda **kwargs: factory_calls.append(kwargs) or ("flow", "particles", "settings", "mesh"),
    )

    def run(**kwargs):
        captured.update(kwargs)
        return last

    def create(flow, particles, settings, **kwargs):
        assert (flow, particles, settings) == ("flow", "particles", "settings")
        assert kwargs == {"mesh": "mesh", "case_dir": tmp_path / "coupled"}
        return nullcontext(SimpleNamespace(run=run))

    monkeypatch.setattr(setup.coupling, "create_coupler", create)
    args = ["phase", "coupled", "--root", str(tmp_path), "--device", "CUDA"]
    args += ["--pilot"] if pilot else []
    args += ["--resume"] if resume else []
    monkeypatch.setattr(sys, "argv", args)
    driver.main()
    initial = captured.pop("initial_velocity")
    assert initial.func is driver.case.module.cylinder_initial_velocity
    assert "span" not in initial.keywords
    np.testing.assert_allclose(
        initial(np.array([[3.0, 0.0, -0.5], [3.0, 0.0, 0.5]])),
        [driver.case.module.STARTUP_FREESTREAM_VELOCITY] * 2,
    )
    assert captured == {
        "start_from": "latest" if resume else "initial",
        "max_coupling_steps": 20 if pilot else None,
        "backup_at_stop": True,
    }
    assert factory_calls == [{"end": 100.0, "device": "CUDA"}]
    label = "pilot" if pilot else "continuation"
    result = json.loads((tmp_path / "coupled" / f"{label}-result.json").read_text())
    assert result["step"] == last
    assert result["status"] == ("pilot-completed" if pilot else "completed")


def test_phase_failure_is_recorded_and_propagated(driver, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("native validation failed")

    monkeypatch.setattr(setup.coupling, "create_coupler", fail)
    monkeypatch.setattr(sys, "argv", ["phase", "coupled", "--root", str(tmp_path)])
    with pytest.raises(RuntimeError, match="native validation failed"):
        driver.main()
    result = json.loads((tmp_path / "coupled/run-result.json").read_text())
    assert result["status"] == "failed"
    assert "native validation failed" in result["error"]
