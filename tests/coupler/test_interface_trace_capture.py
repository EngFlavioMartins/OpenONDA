"""Benchmark instrumentation copies traces without changing numerical calls."""

import importlib.util
from importlib.util import find_spec
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ASSET = (
    Path(__file__).resolve().parents[2]
    / "tests/support/cylinder/capture_interface_traces.py"
)
SPEC = importlib.util.spec_from_file_location("interface_trace_capture_asset", ASSET)
CAPTURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAPTURE)


def _owner(*, master=True):
    owner = SimpleNamespace(
        _is_master=master,
        fvm_solver=SimpleNamespace(step=1375, time=11.0, parallel=SimpleNamespace(comm=None)),
        vpm_solver=SimpleNamespace(step=276, time=11.04) if master else None,
    )
    for suffix, value in (("", 2.0), ("_old", 1.0)):
        for name, shape in (
            ("velocity_boundary_condition", (2, 3)),
            ("normal_velocity_boundary_condition", (2,)),
            ("tangential_gradient_boundary_condition", (2, 3)),
        ):
            setattr(owner, "_" + name + suffix, np.full(shape, value))
    return owner


def _fixture(*, fail=False, reject=False):
    calls = []
    module = SimpleNamespace()

    def advance(owner, *args):
        calls.append(("advance", float(args[-1][0, 0])))
        owner.fvm_solver.step += 5
        owner.fvm_solver.time += 0.04
        return 0.1

    def refresh(owner, *args):
        calls.append(("refresh",))
        for name in CAPTURE._trace(owner, old=True):
            field = {
                "velocity": "velocity_boundary_condition",
                "normal_velocity": "normal_velocity_boundary_condition",
                "tangential_gradient": "tangential_gradient_boundary_condition",
            }[name]
            getattr(owner, "_" + field + "_old")[:] += 1

    def iterate(owner, geometry, next_velocity):
        for trial in range(2):
            candidate = next_velocity if trial == 0 else np.full((2, 3), 3.0)
            owner._normal_velocity_boundary_condition[:] = candidate[:, 0]
            owner._tangential_gradient_boundary_condition[:] = candidate
            module.advance_fvm(owner, *geometry, owner._velocity_boundary_condition_old, candidate)
            module.update_boundary_history_after_replacement(owner, *geometry)
            if fail:
                raise ValueError("intentional trial failure")
        owner._last_interface_iteration_diagnostics = {
            "sweeps": 2,
            "accepted_sweep": 1 if reject else 2,
            "converged": not reject,
            "residuals": [
                {"sweep": 1, "accepted": True},
                {"sweep": 2, "accepted": not reject},
            ],
        }
        if reject:
            owner._velocity_boundary_condition_old[:] = 2
            owner._normal_velocity_boundary_condition_old[:] = 2
            owner._tangential_gradient_boundary_condition_old[:] = 2
        return "unchanged-result"

    module.advance_iterated_interface = iterate
    module.advance_fvm = advance
    module.update_boundary_history_after_replacement = refresh
    return module, calls


GEOMETRY = (np.zeros((2, 3)), np.ones((2, 3)), np.ones(2))


@pytest.mark.parametrize("reject", [False, True])
def test_explicit_trial_events_and_copied_accepted_endpoint(tmp_path, reject):
    owner = _owner()
    module, calls = _fixture(reject=reject)
    originals = vars(module).copy()
    reports = []
    with CAPTURE.capture_interface_traces(
        owner, tmp_path / "trace", reports, max_exchanges=1, iteration_module=module
    ):
        result = module.advance_iterated_interface(
            owner, GEOMETRY, owner._velocity_boundary_condition
        )
    assert result == "unchanged-result"
    assert vars(module) == originals
    assert calls == [("advance", 2.0), ("refresh",), ("advance", 3.0), ("refresh",)]
    assert reports[0]["accepted_sweep"] == (1 if reject else 2)
    with np.load(reports[0]["path"], allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata_json"]))
        events = metadata["events"]
        assert [event["label"] for event in events] == [
            "old_physical_endpoint",
            "raw_predictor",
            "trial_input",
            "trial_output",
            "trial_input",
            "trial_output",
            "accepted_endpoint",
        ]
        assert [event.get("trial") for event in events] == [None, None, 1, 1, 2, 2, None]
        assert metadata["entry_clocks"]["vpm"] == {"step": 276, "time": 11.04}
        assert metadata["status"] == "complete"
        assert np.all(saved[events[0]["arrays"]["velocity"]] == 1)
        assert np.all(saved[events[1]["arrays"]["velocity"]] == 2)
        assert np.all(saved[events[-1]["arrays"]["velocity"]] == (2 if reject else 3))
        assert metadata["interface_iteration"]["residuals"][-1]["accepted"] is not reject


def test_failure_preserves_partial_evidence_and_restores_hooks(tmp_path):
    owner = _owner()
    module, _ = _fixture(fail=True)
    originals = vars(module).copy()
    reports = []
    with (
        pytest.raises(ValueError, match="intentional trial failure"),
        CAPTURE.capture_interface_traces(
            owner, tmp_path / "failed", reports, max_exchanges=1, iteration_module=module
        ),
    ):
        module.advance_iterated_interface(owner, GEOMETRY, owner._velocity_boundary_condition)
    assert vars(module) == originals
    assert reports[0]["status"] == "failed"
    with np.load(reports[0]["path"], allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata_json"]))
        assert metadata["events"][-1]["label"] == "trial_output"
        assert metadata["accepted_sweep"] is None
        assert "intentional trial failure" in metadata["error"]


def test_existing_file_rejected_before_numerical_work(tmp_path):
    owner = _owner()
    module, calls = _fixture()
    path = tmp_path / "trace-step000276.npz"
    path.write_bytes(b"original evidence")
    with (
        CAPTURE.capture_interface_traces(
            owner, tmp_path / "trace", [], max_exchanges=1, iteration_module=module
        ),
        pytest.raises(RuntimeError, match="FileExistsError"),
    ):
        module.advance_iterated_interface(owner, GEOMETRY, owner._velocity_boundary_condition)
    assert not calls
    assert path.read_bytes() == b"original evidence"


@pytest.mark.parametrize("nonmaster", [False, True])
def test_other_owner_and_nonmaster_are_not_recorded(tmp_path, nonmaster):
    owner = _owner(master=not nonmaster)
    current = owner if nonmaster else _owner()
    module, calls = _fixture()
    reports = []
    with CAPTURE.capture_interface_traces(
        owner, tmp_path / "trace", reports, max_exchanges=1, iteration_module=module
    ):
        assert (
            module.advance_iterated_interface(
                current, GEOMETRY, current._velocity_boundary_condition
            )
            == "unchanged-result"
        )
    assert len(calls) == 4
    assert not reports
    assert not list(tmp_path.iterdir())


def test_multiple_exchanges_release_buffers_and_enforce_cap(tmp_path):
    owner = _owner()
    module, _ = _fixture()
    reports = []
    with CAPTURE.capture_interface_traces(
        owner, tmp_path / "trace", reports, max_exchanges=2, iteration_module=module
    ):
        for step in (276, 277):
            owner.vpm_solver.step = step
            module.advance_iterated_interface(owner, GEOMETRY, owner._velocity_boundary_condition)
        owner.vpm_solver.step = 278
        with pytest.raises(RuntimeError, match="exchange cap"):
            module.advance_iterated_interface(owner, GEOMETRY, owner._velocity_boundary_condition)
    assert [row["step"] for row in reports] == [276, 277]
    assert len(list(tmp_path.iterdir())) == 2


@pytest.mark.integration
def test_two_rank_master_only_capture_and_collective_errors(tmp_path):
    if find_spec("mpi4py") is None:
        pytest.skip("mpi4py is required")
    bundled = Path(sys.executable).with_name("mpiexec")
    mpiexec = str(bundled) if bundled.is_file() else shutil.which("mpiexec")
    if mpiexec is None:
        pytest.skip("mpiexec is required")
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    completed = subprocess.run(
        [
            mpiexec,
            "-n",
            "2",
            sys.executable,
            str(Path(__file__).with_name("_interface_trace_capture_mpi.py")),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=45,
        env=env,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    assert "INTERFACE_TRACE_MPI_QUALIFIED" in completed.stdout
