"""Phase runners share the startup lifecycle and preserve bounded output."""

from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from openonda.tutorial_runner import load_case_module

SUPPORT = Path(__file__).resolve().parents[1] / "support/cylinder"
CASE = Path(__file__).resolve().parents[2] / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow"
startup = load_case_module(CASE, "assets.startup")


@pytest.fixture
def driver(monkeypatch):
    for name in ("phase_benchmark", "run_phase_benchmark"):
        spec = importlib.util.spec_from_file_location(name, SUPPORT / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    return module


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
        driver.case, "coupled_case", lambda **kwargs: factory_calls.append(kwargs) or "case"
    )

    def run(build, **kwargs):
        captured.update(kwargs)
        assert build(end_time=kwargs["end_time"], overrides=None) == "case"
        return last

    monkeypatch.setattr(startup, "run_coupled_cylinder", run)
    args = ["phase", "coupled", "--root", str(tmp_path), "--device", "CUDA"]
    args += ["--pilot"] if pilot else []
    args += ["--resume"] if resume else []
    monkeypatch.setattr(sys, "argv", args)
    driver.main()
    assert captured == {
        "output_root": tmp_path / "coupled",
        "end_time": 100.0,
        "start_from": "latest" if resume else "initial",
        "max_coupling_steps": 20 if pilot else None,
        "startup_duration": 2.0,
        "startup_transition_duration": 1.0,
        "steady_freestream_velocity": (1.0, 0.0, 0.0),
        "perturbation": driver.case.module.INITIAL_PERTURBATION,
    }
    assert factory_calls == [{"end": 100.0, "device": "CUDA"}]
    label = "pilot" if pilot else "continuation"
    result = json.loads((tmp_path / "coupled" / f"{label}-result.json").read_text())
    assert result["step"] == last
    assert result["status"] == ("pilot-completed" if pilot else "completed")


def test_phase_failure_is_recorded_and_propagated(driver, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("native admission failed")

    monkeypatch.setattr(startup, "run_coupled_cylinder", fail)
    monkeypatch.setattr(sys, "argv", ["phase", "coupled", "--root", str(tmp_path)])
    with pytest.raises(RuntimeError, match="native admission failed"):
        driver.main()
    result = json.loads((tmp_path / "coupled/run-result.json").read_text())
    assert result["status"] == "failed"
    assert "native admission failed" in result["error"]
