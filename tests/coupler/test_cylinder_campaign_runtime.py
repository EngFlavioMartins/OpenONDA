"""Lightweight contracts for bounded cylinder campaign orchestration."""

import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest

from openonda import cylinder_campaign, cylinder_case

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/assets"


def load_asset(name: str):
    path = ASSETS / name
    spec = importlib.util.spec_from_file_location(f"test_{name.replace('.', '_')}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_trial_terminates_a_timed_out_process_group(tmp_path):
    record = cylinder_campaign.run_trial(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        tmp_path / "trial",
        cwd=tmp_path,
        wall_limit=0.05,
    )

    assert record["timed_out"] is True
    assert record["returncode"] == 124
    assert json.loads((tmp_path / "trial" / "trial.json").read_text())["timed_out"] is True
    assert (tmp_path / "trial" / "console.log").is_file()


def test_run_trial_resume_accumulates_matching_attempt_cost(tmp_path):
    command = [sys.executable, "-c", "pass", "--resume"]
    first = cylinder_campaign.run_trial(command, tmp_path / "trial", cwd=tmp_path, wall_limit=2)
    second = cylinder_campaign.run_trial(command, tmp_path / "trial", cwd=tmp_path, wall_limit=2)
    payload = json.loads((tmp_path / "trial" / "trial.json").read_text())

    assert first["returncode"] == second["returncode"] == 0
    assert payload["normalized_command"] == [sys.executable, "-c", "pass"]
    assert len(payload["attempts"]) == 2
    assert payload["attempt_wall_seconds"] == pytest.approx(second["attempts"][-1]["wall_seconds"])
    assert payload["wall_seconds"] == pytest.approx(
        sum(item["wall_seconds"] for item in payload["attempts"])
    )
    assert payload["wall_seconds"] >= second["attempt_wall_seconds"]


def test_run_trial_records_process_tree_rss_and_collect_cost_marks_missing_rank_rss(tmp_path):
    record = cylinder_campaign.run_trial(
        [sys.executable, "-c", "import time; x=bytearray(8*1024*1024); time.sleep(.6)"],
        tmp_path / "trial",
        cwd=tmp_path,
        wall_limit=2,
    )

    assert record["peak_process_tree_rss_bytes"] > 0
    assert record["rss_sample_count"] > 0
    journal = tmp_path / "output" / "performance.jsonl"
    journal.parent.mkdir()
    journal.write_text(json.dumps({"time": 0.0, "step_seconds": {"max": 0.1}}) + "\n")
    assert (
        cylinder_campaign.collect_cost(tmp_path / "output")["peak_rank_aggregate_rss_bytes"] is None
    )


def test_manifest_works_from_an_installed_style_tree_without_git(tmp_path, monkeypatch):
    fake_module = tmp_path / "installed" / "openonda" / "cylinder_case.py"
    fake_module.parent.mkdir(parents=True)
    fake_module.write_text("# installed\n")
    input_file = tmp_path / "setup.py"
    input_file.write_text("setup = 1\n")
    monkeypatch.setattr(cylinder_case, "__file__", str(fake_module))

    destination = tmp_path / "run" / "campaign_manifest.json"
    cylinder_case.write_manifest(
        destination, status="running", config={"kind": "pilot"}, inputs=(input_file,)
    )
    payload = json.loads(destination.read_text())

    assert payload["git_diff_hash"] == "unknown"
    assert payload["inputs"][str(input_file)]
    assert payload["software_fingerprint"]["schema"] == "openonda-software-fingerprint/1"


def test_software_fingerprint_is_portable_and_tracks_solver_source(tmp_path):
    first = tmp_path / "first"
    (first / "openonda").mkdir(parents=True)
    (first / "source").mkdir()
    (first / "openonda" / "solver.py").write_text("VALUE = 1\n")
    (first / "source" / "numerics.py").write_text("VALUE = 2\n")
    second = tmp_path / "second"
    shutil.copytree(first, second)

    left = cylinder_case.software_fingerprint(first)
    right = cylinder_case.software_fingerprint(second)
    assert left["digest"] == right["digest"]
    (second / "source" / "numerics.py").write_text("VALUE = 3\n")
    changed = cylinder_case.software_fingerprint(second)
    assert changed["digest"] != left["digest"]


def test_pipeline_uses_one_wall_limit_per_case_and_resume_without_overwrite(tmp_path, monkeypatch):
    pipeline = load_asset("run_pipeline.py")
    calls = []

    def fake_trial(command, directory, *, cwd, wall_limit):
        calls.append((command, directory, cwd, wall_limit))
        return {
            "command": command,
            "returncode": 0,
            "timed_out": False,
            "console_log": str(directory / "console.log"),
        }

    monkeypatch.setattr(pipeline, "run_trial", fake_trial)
    monkeypatch.setattr(pipeline, "load_case_module", lambda *args: object())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pipeline.py",
            "--run-dir",
            str(tmp_path / "pipeline"),
            "--pilot",
            "--sensitivity",
            "none",
        ],
    )

    assert pipeline.main() == 0
    assert len(calls) == 2
    assert all(call[3] == 43200 for call in calls)
    assert [call[0][3] for call in calls] == ["reference", "coupled"]
    assert (tmp_path / "pipeline" / "pipeline_manifest.json").is_file()

    (tmp_path / "pipeline" / "reference").mkdir()
    (tmp_path / "pipeline" / "coupled" / "grid_h0p1").mkdir(parents=True)
    calls.clear()
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_pipeline.py", "--run-dir", str(tmp_path / "pipeline"), "--pilot", "--resume"],
    )
    assert pipeline.main() == 0
    assert all("--resume" in call[0] for call in calls)

    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "sentinel").write_text("keep\n")
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--run-dir", str(existing), "--pilot"])
    with pytest.raises(FileExistsError):
        pipeline.main()
    assert (existing / "sentinel").read_text() == "keep\n"


def test_sensitivity_factor_set_keeps_interface_cycles_fixed():
    sensitivity = load_asset("run_sensitivity.py")

    assert sensitivity.FACTORS["particle_spacing_ratio"] == (1.25, 1.5)
    assert "interface_iterations" not in sensitivity.FACTORS
    assert "transfer_vorticity_cutoff" not in sensitivity.FACTORS
    assert "interface iteration limits and tolerances are fixed" in (
        sensitivity.Path(sensitivity.__file__).read_text().lower()
    )


def test_coupled_campaign_identity_records_acceleration_choice(monkeypatch):
    campaign = load_asset("run_campaign.py")
    module = campaign.load_case_module(campaign.CASE_DIR)
    monkeypatch.setattr(campaign, "file_hash", lambda _path: "test-input")
    monkeypatch.setattr(campaign, "software_fingerprint", lambda: {"digest": "test-source"})

    ordinary = campaign._resolved_coupled_config(module, 0.04, {})
    accelerated = campaign._resolved_coupled_config(
        module, 0.04, {"interface_acceleration": "aitken"}
    )
    assert ordinary["interface_acceleration"] == "none"
    assert accelerated["interface_acceleration"] == "aitken"
    assert ordinary["interface_iterations"] == accelerated["interface_iterations"] == 3
    assert ordinary != accelerated


def test_span_sensitivity_keeps_particle_spacing_fixed(tmp_path, monkeypatch):
    sensitivity = load_asset("run_sensitivity.py")
    calls = []

    class Post:
        pass

    def fake_trial(command, directory, *, cwd, wall_limit):
        calls.append(command)
        directory.mkdir(parents=True, exist_ok=True)
        log = directory / "console.log"
        log.write_text("mock\n")
        return {
            "command": command,
            "returncode": 0,
            "timed_out": False,
            "wall_seconds": 1.0,
            "console_log": str(log),
        }

    monkeypatch.setattr(sensitivity, "run_trial", fake_trial)
    monkeypatch.setattr(
        sensitivity, "collect_cost", lambda _root: {"unconverged_stationary_intervals": 0}
    )
    monkeypatch.setattr(sensitivity, "load_case_module", lambda *args: Post())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_sensitivity.py",
            "--factor",
            "span",
            "--screen",
            "--run-dir",
            str(tmp_path / "sensitivity"),
        ],
    )

    assert sensitivity.main() == 0
    assert len(calls) == 3
    for command in calls:
        overrides = [
            command[index + 1] for index, value in enumerate(command[:-1]) if value == "--override"
        ]
        assert "particle_spacing_ratio=1.0" in overrides
    assert any(
        "span=0.48"
        in [command[index + 1] for index, value in enumerate(command[:-1]) if value == "--override"]
        for command in calls
    )


def test_screen_sensitivity_matches_physical_duration_across_exchange_clocks(tmp_path, monkeypatch):
    sensitivity = load_asset("run_sensitivity.py")
    calls = []

    class Post:
        pass

    def fake_trial(command, directory, *, cwd, wall_limit):
        calls.append(command)
        directory.mkdir(parents=True, exist_ok=True)
        log = directory / "console.log"
        log.write_text("mock\n")
        return {
            "command": command,
            "returncode": 0,
            "timed_out": False,
            "wall_seconds": 1.0,
            "console_log": str(log),
        }

    monkeypatch.setattr(sensitivity, "run_trial", fake_trial)
    monkeypatch.setattr(
        sensitivity, "collect_cost", lambda _root: {"unconverged_stationary_intervals": 0}
    )
    monkeypatch.setattr(sensitivity, "load_case_module", lambda *args: Post())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_sensitivity.py",
            "--factor",
            "exchange_dt",
            "--screen",
            "--max-coupling-steps",
            "20",
            "--run-dir",
            str(tmp_path / "sensitivity"),
        ],
    )

    assert sensitivity.main() == 0
    assert all("--max-coupling-steps" not in command for command in calls)
    end_times = [float(command[command.index("--end-time") + 1]) for command in calls]
    assert end_times == [0.8, 0.8, 0.8]
    report = json.loads((tmp_path / "sensitivity" / "sensitivity.json").read_text())
    assert [row["screened_physical_time"] for row in report["runs"]] == [0.8, 0.8, 0.8]
    assert [row["screened_exchange_steps"] for row in report["runs"]] == [20, 40, 10]


def test_screen_sensitivity_rejects_incompatible_step_budget():
    sensitivity = load_asset("run_sensitivity.py")

    with pytest.raises(ValueError, match="common screen duration"):
        sensitivity._screen_steps(21, 0.08)


def test_interaction_selector_rejects_invalid_body_authority_pair():
    sensitivity = load_asset("run_sensitivity.py")

    class Builder:
        def build_case(self, *, end_time, overrides):
            assert end_time == 100.0
            if (
                overrides.get("particle_spacing_ratio") == 1.5
                and overrides.get("blend_width_ratio") == 7.0
            ):
                raise ValueError("blend width exceeds body authority")

    candidates = [
        {
            "factor": "particle_spacing_ratio",
            "overrides": {"particle_spacing_ratio": 1.5},
        },
        {"factor": "blend_width_ratio", "overrides": {"blend_width_ratio": 7.0}},
        {"factor": "core_radius_ratio", "overrides": {"core_radius_ratio": 0.8}},
    ]
    interaction, rejected = sensitivity._select_interaction(candidates, Builder())

    assert interaction == (
        "interaction",
        {"particle_spacing_ratio": 1.5, "core_radius_ratio": 0.8},
        "interaction",
    )
    assert rejected == [
        {
            "factors": ["particle_spacing_ratio", "blend_width_ratio"],
            "overrides": {"particle_spacing_ratio": 1.5, "blend_width_ratio": 7.0},
            "reason": "blend width exceeds body authority",
        }
    ]


def test_reference_grid_completion_requires_matching_resolved_record(tmp_path):
    campaign = load_asset("run_campaign.py")

    class ReferenceModule:
        SPAN = 0.96
        TIME_STEP_SIZE = 0.001

    config = campaign._grid_config(ReferenceModule, "grid_h008", 0.08, 4.0)
    campaign._write_grid_record(tmp_path, config)
    samples = tmp_path / "samples" / "grid_h008"
    solution = tmp_path / "solution" / "grid_h008"
    samples.mkdir(parents=True)
    solution.mkdir(parents=True)
    (samples / "grid_run.json").write_text(
        json.dumps({"case": "grid_h008", "cell_size": 0.08, "end_time": 4.0})
    )
    (solution / "fvm_metadata.json").write_text(
        json.dumps({"lifecycle": {"status": "complete"}, "state": {"time": 4.0}})
    )

    assert campaign._grid_complete(tmp_path, ReferenceModule, "grid_h008", 0.08, 4.0)
    assert not campaign._grid_complete(tmp_path, ReferenceModule, "grid_h008", 0.04, 4.0)


def test_incomplete_reference_resume_is_rejected_without_native_restart(tmp_path, monkeypatch):
    campaign = load_asset("run_campaign.py")

    class ReferenceModule:
        SPAN = 0.96
        TIME_STEP_SIZE = 0.001

    monkeypatch.setattr(campaign, "load_case_module", lambda *args: ReferenceModule)
    partial = tmp_path / "samples" / "grid_h008"
    partial.mkdir(parents=True)
    (partial / "partial-output").write_text("incomplete\n")
    options = type(
        "Options",
        (),
        {
            "override": None,
            "pilot": True,
            "end_time": 4.0,
            "grid": ["grid_h008=0.08"],
            "no_analysis": True,
        },
    )()

    with pytest.raises(RuntimeError, match="no native restart"):
        campaign.run_reference(options, tmp_path)


def test_coupled_resume_requires_canonical_backup(tmp_path, monkeypatch):
    campaign = load_asset("run_campaign.py")

    class CoupledModule:
        END_TIME = 100.0
        VPM_TIME_STEP_SIZE = 0.04

    resolved = {"kind": "coupled", "overrides": {}, "end_time": 100.0, "source_hash": "test"}
    expected = {"kind": "coupled", "overrides": {}, "end_time": 100.0, "resolved": resolved}
    (tmp_path / "campaign_manifest.json").write_text(json.dumps({"config": expected}))
    (tmp_path / "partial").write_text("state\n")
    monkeypatch.setattr(campaign, "load_case_module", lambda *args: CoupledModule)
    monkeypatch.setattr(campaign, "_resolved_coupled_config", lambda *args: resolved)
    options = type(
        "Options",
        (),
        {
            "pilot": False,
            "max_coupling_steps": None,
            "resume": True,
            "override": None,
            "end_time": None,
        },
    )()

    with pytest.raises(RuntimeError, match="canonical solution/backups/manifest.json"):
        campaign.run_coupled(options, tmp_path)


def test_coupled_resume_skips_exact_completed_campaign_without_backup(tmp_path, monkeypatch):
    campaign = load_asset("run_campaign.py")

    class CoupledModule:
        END_TIME = 100.0
        VPM_TIME_STEP_SIZE = 0.04

    resolved = {"kind": "coupled", "overrides": {}, "end_time": 100.0, "source_hash": "test"}
    expected = {"kind": "coupled", "overrides": {}, "end_time": 100.0, "resolved": resolved}
    (tmp_path / "campaign_manifest.json").write_text(json.dumps({"config": expected}))
    (tmp_path / "COMPLETE").write_text("complete\n")
    monkeypatch.setattr(campaign, "load_case_module", lambda *args: CoupledModule)
    monkeypatch.setattr(campaign, "_resolved_coupled_config", lambda *args: resolved)
    options = type(
        "Options",
        (),
        {
            "pilot": False,
            "max_coupling_steps": None,
            "resume": True,
            "override": None,
            "end_time": None,
        },
    )()

    assert campaign.run_coupled(options, tmp_path) is None


def test_external_mpi_selects_one_broadcast_run_directory(tmp_path, monkeypatch):
    campaign = load_asset("run_campaign.py")

    class Comm:
        def __init__(self, rank):
            self.rank = rank
            self.broadcast = None

        def Get_rank(self):
            return self.rank

        def bcast(self, value, root):
            if self.rank == 0:
                self.broadcast = value
                return value
            return root_comm.broadcast

    root_comm = Comm(0)
    worker_comm = Comm(1)
    options = type(
        "Options",
        (),
        {"run_dir": None, "root": tmp_path, "kind": "coupled", "pilot": False},
    )()
    monkeypatch.setattr(campaign, "_explicit_mpi", lambda: True)
    monkeypatch.setattr(campaign, "_mpi_comm", lambda: root_comm)
    root_path = campaign.select_run_directory(options)
    assert root_path.parent == tmp_path

    monkeypatch.setattr(campaign, "_mpi_comm", lambda: worker_comm)
    worker_path = campaign.select_run_directory(options)
    assert worker_path == root_path


def test_external_mpi_collective_preflight_runs_callback_once(monkeypatch):
    campaign = load_asset("run_campaign.py")

    class Comm:
        def __init__(self, rank):
            self.rank = rank
            self.decision = None

        def Get_rank(self):
            return self.rank

        def bcast(self, value, root):
            if self.rank == 0:
                self.decision = value
                return value
            return root_comm.decision

    root_comm = Comm(0)
    worker_comm = Comm(1)
    calls = []
    monkeypatch.setattr(campaign, "_explicit_mpi", lambda: True)
    monkeypatch.setattr(campaign, "_mpi_comm", lambda: root_comm)
    assert campaign._collective_preflight(lambda: calls.append("root") or ["pending"]) == [
        "pending"
    ]
    monkeypatch.setattr(campaign, "_mpi_comm", lambda: worker_comm)
    assert campaign._collective_preflight(lambda: calls.append("worker")) == ["pending"]
    assert calls == ["root"]


def test_external_mpi_root_publication_failure_reaches_all_ranks(monkeypatch):
    campaign = load_asset("run_campaign.py")

    class Comm:
        def __init__(self, rank):
            self.rank = rank
            self.result = None

        def Get_rank(self):
            return self.rank

        def bcast(self, value, root):
            if self.rank == 0:
                self.result = value
                return value
            return root_comm.result

    root_comm = Comm(0)
    worker_comm = Comm(1)
    monkeypatch.setattr(campaign, "_explicit_mpi", lambda: True)
    monkeypatch.setattr(campaign, "_mpi_comm", lambda: root_comm)
    with pytest.raises(RuntimeError, match="root action failed"):
        campaign._collective_root_action(lambda: (_ for _ in ()).throw(OSError("read-only")))

    monkeypatch.setattr(campaign, "_mpi_comm", lambda: worker_comm)
    with pytest.raises(RuntimeError, match="root action failed"):
        campaign._collective_root_action(lambda: pytest.fail("worker executed root action"))


class _BudgetSampler:
    def __init__(self, _pid):
        self.peak_bytes = 17
        self.sample_count = 3
        self.period = 0.25

    def start(self):
        pass

    def stop(self):
        pass


class _BudgetChild:
    pid = 1234

    def __init__(self, waits):
        self.waits = waits

    def wait(self, *, timeout):
        self.waits.append(timeout)
        return 0


def _write_prior_trial(path, command, wall_seconds=2.0):
    path.mkdir(parents=True)
    (path / "trial.json").write_text(
        json.dumps(
            {
                "normalized_command": command,
                "attempts": [
                    {
                        "command": [*command, "--resume"],
                        "returncode": 124,
                        "timed_out": True,
                        "wall_seconds": wall_seconds,
                        "peak_process_tree_rss_bytes": 99,
                        "rss_sample_count": 4,
                        "rss_sample_period_seconds": 0.25,
                    }
                ],
            }
        )
    )


def test_run_trial_first_attempt_receives_full_wall_limit(monkeypatch, tmp_path):
    waits = []
    monkeypatch.setattr(cylinder_campaign, "_ProcessTreeRSSSampler", _BudgetSampler)
    monkeypatch.setattr(
        cylinder_campaign.subprocess,
        "Popen",
        lambda *args, **kwargs: _BudgetChild(waits),
    )
    command = [sys.executable, "-c", "pass"]

    record = cylinder_campaign.run_trial(command, tmp_path / "trial", cwd=tmp_path, wall_limit=3.5)

    assert waits == [pytest.approx(3.5)]
    assert record["returncode"] == 0
    assert record["remaining_wall_seconds"] == pytest.approx(3.5)


def test_run_trial_resume_uses_remaining_case_budget(monkeypatch, tmp_path):
    waits = []
    monkeypatch.setattr(cylinder_campaign, "_ProcessTreeRSSSampler", _BudgetSampler)
    monkeypatch.setattr(
        cylinder_campaign.subprocess,
        "Popen",
        lambda *args, **kwargs: _BudgetChild(waits),
    )
    command = [sys.executable, "-c", "pass", "--resume"]
    normalized = [sys.executable, "-c", "pass"]
    _write_prior_trial(tmp_path / "trial", normalized, wall_seconds=2.25)

    record = cylinder_campaign.run_trial(command, tmp_path / "trial", cwd=tmp_path, wall_limit=5.0)

    assert waits == [pytest.approx(2.75)]
    assert record["returncode"] == 0
    assert record["remaining_wall_seconds"] == pytest.approx(2.75)


def test_run_trial_changed_resume_command_consumes_prior_budget(monkeypatch, tmp_path):
    waits = []
    monkeypatch.setattr(cylinder_campaign, "_ProcessTreeRSSSampler", _BudgetSampler)
    monkeypatch.setattr(
        cylinder_campaign.subprocess,
        "Popen",
        lambda *args, **kwargs: _BudgetChild(waits),
    )
    normalized = [sys.executable, "-c", "pass", "--end-time", "1"]
    _write_prior_trial(tmp_path / "trial", normalized, wall_seconds=2.25)
    command = [sys.executable, "-c", "pass", "--end-time", "2", "--resume"]

    record = cylinder_campaign.run_trial(command, tmp_path / "trial", cwd=tmp_path, wall_limit=5.0)

    assert waits == [pytest.approx(2.75)]
    assert len(record["attempts"]) == 2
    assert record["attempts"][0]["command"][-2:] == ["1", "--resume"]
    assert record["attempts"][1]["command"][-2:] == ["2", "--resume"]


def test_run_trial_changed_resume_command_reads_legacy_record(monkeypatch, tmp_path):
    waits = []
    monkeypatch.setattr(cylinder_campaign, "_ProcessTreeRSSSampler", _BudgetSampler)
    monkeypatch.setattr(
        cylinder_campaign.subprocess,
        "Popen",
        lambda *args, **kwargs: _BudgetChild(waits),
    )
    trial = tmp_path / "trial"
    trial.mkdir()
    (trial / "trial.json").write_text(
        json.dumps(
            {
                "command": [sys.executable, "-c", "pass", "--end-time", "1"],
                "wall_seconds": 2.25,
                "returncode": 124,
            }
        )
    )
    command = [sys.executable, "-c", "pass", "--end-time", "2", "--resume"]

    record = cylinder_campaign.run_trial(command, trial, cwd=tmp_path, wall_limit=5.0)

    assert waits == [pytest.approx(2.75)]
    assert len(record["attempts"]) == 2
    assert record["attempts"][0]["command"][-2:] == ["--end-time", "1"]


def test_run_trial_exhausted_resume_does_not_spawn(monkeypatch, tmp_path):
    command = [sys.executable, "-c", "pass", "--resume"]
    normalized = [sys.executable, "-c", "pass"]
    _write_prior_trial(tmp_path / "trial", normalized, wall_seconds=2.0)

    monkeypatch.setattr(
        cylinder_campaign.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("an exhausted case must not spawn"),
    )
    record = cylinder_campaign.run_trial(command, tmp_path / "trial", cwd=tmp_path, wall_limit=2.0)

    assert record["returncode"] == 124
    assert record["timed_out"] is True
    assert record["budget_exhausted"] is True
    assert record["wall_seconds"] == pytest.approx(2.0)
    assert record["peak_process_tree_rss_bytes"] == 99
    assert len(record["attempts"]) == 1
    assert "wall budget exhausted" in (tmp_path / "trial" / "console.log").read_text()


@pytest.mark.parametrize("payload", ["{not-json", "[]"])
def test_run_trial_malformed_resume_record_fails_closed(monkeypatch, tmp_path, payload):
    trial = tmp_path / "trial"
    trial.mkdir()
    (trial / "trial.json").write_text(payload)
    monkeypatch.setattr(
        cylinder_campaign.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("malformed resume must not spawn"),
    )

    with pytest.raises(ValueError, match="malformed trial record"):
        cylinder_campaign.run_trial(
            [sys.executable, "-c", "pass", "--resume"],
            trial,
            cwd=tmp_path,
            wall_limit=2.0,
        )


def test_run_trial_invalid_prior_wall_seconds_fails_closed(monkeypatch, tmp_path):
    trial = tmp_path / "trial"
    trial.mkdir()
    (trial / "trial.json").write_text(
        json.dumps(
            {
                "normalized_command": [sys.executable, "-c", "pass"],
                "attempts": [
                    {
                        "command": [sys.executable, "-c", "pass"],
                        "wall_seconds": -1.0,
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(
        cylinder_campaign.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("invalid resume must not spawn"),
    )

    with pytest.raises(ValueError, match="malformed trial attempt"):
        cylinder_campaign.run_trial(
            [sys.executable, "-c", "pass", "--resume"],
            trial,
            cwd=tmp_path,
            wall_limit=2.0,
        )
