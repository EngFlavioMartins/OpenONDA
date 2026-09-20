"""The completion stage never mistakes a dead launcher for completed data."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "completion", ROOT / "studies/panel_removal/complete_cylinder_comparison.py"
)
stage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def pair(tmp_path):
    run, campaign = tmp_path / "run", tmp_path / "reference/campaigns/test"
    ref = campaign / "spatial"
    write(run / "experiment.json", {"end_time": 160, "coupling_dt": 0.04})
    for base in (run / "solution", ref / "solution/xy_fine"):
        write(
            base / "fvm_metadata.json",
            {"lifecycle": {"status": "complete"}, "state": {"time": 160.0000000000003}},
        )
    write(
        run / "solution/vpm_metadata.json",
        {"lifecycle": {"status": "completed"}, "state": {"time": 160}},
    )
    backup = run / "solution/backups"
    artifacts = {"fvm": "fvm_004000", "vpm": "vpm.h5", "vpm_boundary_condition": "bc.npz"}
    write(
        backup / "manifest.json",
        {
            "kind": "openonda.coupled_backup",
            "time": 160,
            "coupling_step": 4000,
            "artifacts": artifacts,
        },
    )
    for name in artifacts.values():
        (backup / name).touch()
    write(run / "solution/coupler_diagnostics.jsonl", {"time": 160, "step": 4000})
    for samples in (run / "samples", ref / "samples/xy_fine"):
        samples.mkdir(parents=True, exist_ok=True)
        (samples / "forces_history.csv").write_text("time,step,patch\n160,4000,cylinder\n")
    write(
        ref / "samples/xy_fine/grid_run.json",
        {"schema": "openonda-fvm-grid-run/1", "case": "xy_fine", "end_time": 160},
    )
    for directory, pid in ((run, 10), (campaign, 20)):
        write(directory / "process.json", {"pid": pid})
    return run, campaign


def identity(pid=10):
    return {"pid": pid, "started": "Sun Sep 20 10:50:13 2026", "command": "test", "alive": True}


def test_actual_terminal_metadata_spelling_and_clock_tolerance(tmp_path):
    run, campaign = pair(tmp_path)
    evidence = stage.completion_evidence(run, campaign)
    assert all(item["complete"] for item in evidence.values())
    write(
        run / "solution/vpm_metadata.json",
        {"lifecycle": {"status": "complete"}, "state": {"time": 160}},
    )
    assert not stage.completion_evidence(run, campaign)["experiment"]["complete"]


@pytest.mark.parametrize(
    "missing",
    [
        "solution/backups/manifest.json",
        "solution/vpm_metadata.json",
        "solution/coupler_diagnostics.jsonl",
        "samples/forces_history.csv",
    ],
)
def test_terminal_outputs_are_required(tmp_path, missing):
    run, campaign = pair(tmp_path)
    (run / missing).unlink()
    assert not stage.completion_evidence(run, campaign)["experiment"]["complete"]


def test_reference_registration_alone_is_insufficient(tmp_path):
    run, campaign = pair(tmp_path)
    write(
        campaign / "spatial/solution/xy_fine/fvm_metadata.json",
        {"lifecycle": {"status": "created"}, "state": {"time": 0}},
    )
    assert not stage.completion_evidence(run, campaign)["reference"]["complete"]


def test_reused_pid_is_rejected_even_with_complete_outputs(monkeypatch):
    monkeypatch.setattr(stage, "process_identity", lambda pid: {**identity(pid), "started": "new"})
    with pytest.raises(RuntimeError, match="reused"):
        stage.ready(
            {role: {"complete": True} for role in ("experiment", "reference")},
            {role: identity() for role in ("experiment", "reference")},
        )


def test_dead_pid_is_failure_before_completion(monkeypatch):
    monkeypatch.setattr(stage, "process_identity", lambda pid: None)
    with pytest.raises(RuntimeError, match="dead before"):
        stage.ready(
            {role: {"complete": False} for role in ("experiment", "reference")},
            {role: identity() for role in ("experiment", "reference")},
        )


def test_reference_may_still_be_queued_while_campaign_is_alive(monkeypatch):
    monkeypatch.setattr(stage, "process_identity", lambda pid: identity(pid))
    done, alive = stage.ready(
        {"experiment": {"complete": True}, "reference": {"complete": False}},
        {role: identity() for role in ("experiment", "reference")},
    )
    assert not done and all(alive.values())


def test_reference_controls_need_not_finish_after_matched_grid(monkeypatch):
    monkeypatch.setattr(stage, "process_identity", lambda pid: None if pid == 10 else identity(pid))
    done, alive = stage.ready(
        {role: {"complete": True} for role in ("experiment", "reference")},
        {"experiment": identity(10), "reference": identity(20)},
    )
    assert done and alive["reference"]


def test_duplicate_stage_excluded_and_stale_lock_reusable(tmp_path):
    with (
        stage.exclusive_lock(tmp_path),
        pytest.raises(RuntimeError, match="Another"),
        stage.exclusive_lock(tmp_path),
    ):
        pass
    with stage.exclusive_lock(tmp_path):
        pass


def test_mpi_exec_command_is_supported_but_wrong_output_rejected(tmp_path):
    run, campaign = pair(tmp_path)
    command = f"/opt/bin/prterun -n 2 /opt/bin/python {ROOT}/studies/panel_removal/run_cylinder.py --output {run}"
    stage.validate_command({"command": command}, "experiment", run, campaign)
    with pytest.raises(RuntimeError, match="expected experiment"):
        stage.validate_command({"command": command + "x"}, "experiment", run, campaign)


@pytest.mark.parametrize("passed,expected", [(True, 0), (False, 3)])
def test_completed_pair_postprocesses_once_and_preserves_scientific_failure(
    tmp_path, monkeypatch, passed, expected
):
    run, campaign = pair(tmp_path)
    monkeypatch.setattr(stage, "process_identity", lambda pid: None)
    calls = []

    def comparator(command, **kwargs):
        calls.append(command)
        destination = run / "mature_comparison"
        write(destination / "comparison.json", {"mature_gate_passed": passed})
        for name in ("fvm.png", "vpm.png"):
            (destination / name).touch()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(stage.subprocess, "run", comparator)
    assert stage.execute(run, campaign, 1) == expected
    assert stage.execute(run, campaign, 1) == expected
    assert len(calls) == 1
    assert calls[0][calls[0].index("--reference") + 1] == str(campaign / "spatial/samples/xy_fine")
    assert calls[0][calls[0].index("--start") + 1] == "80"
    assert calls[0][calls[0].index("--end") + 1] == "160"


def test_failure_marker_prevents_dead_process_success(tmp_path):
    run, campaign = pair(tmp_path)
    (run / "launcher.log").write_text("Traceback (most recent call last):\n")
    with pytest.raises(RuntimeError, match="launcher reports a failure"):
        stage.completion_evidence(run, campaign)


def test_nonzero_comparator_exit_is_operational_failure(tmp_path, monkeypatch):
    run, campaign = pair(tmp_path)
    monkeypatch.setattr(stage, "process_identity", lambda pid: None)
    monkeypatch.setattr(stage.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1))
    assert stage.execute(run, campaign, 1) == 2
    status = stage.read_json(run / "mature_comparison/status.json")
    assert status["status"] == "failed" and "Comparator exited 1" in status["error"]


def test_ps_parses_mac_start_time_and_zombie_without_accepting_it_as_alive(monkeypatch):
    monkeypatch.setattr(
        stage.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=0,
            stdout="Sun Sep 20 10:50:13 2026 Z /opt/bin/prterun -n 2 python run.py\n",
            stderr="",
        ),
    )
    result = stage.process_identity(10)
    assert result["started"] == "Sun Sep 20 10:50:13 2026"
    assert result["command"] == "/opt/bin/prterun -n 2 python run.py"
    assert not result["alive"]


def test_ps_inspection_denial_is_not_reported_as_dead(monkeypatch):
    monkeypatch.setattr(
        stage.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="operation not permitted"),
    )
    with pytest.raises(RuntimeError, match="Cannot inspect"):
        stage.process_identity(10)
