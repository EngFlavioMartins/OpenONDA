"""Lightweight requirements for bounded cylinder parameter study execution."""

import json
from pathlib import Path
import shutil
import sys

import pytest

from openonda.tutorial_runner import load_case_module
from tests.support.cylinder import parameter_study, run_records

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "tests/support/cylinder"


def load_asset(name: str):
    return load_case_module(ASSETS, Path(name).stem)


def test_run_trial_terminates_a_timed_out_process_group(tmp_path):
    record = parameter_study.run_trial(
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
    first = parameter_study.run_trial(command, tmp_path / "trial", cwd=tmp_path, wall_limit=2)
    second = parameter_study.run_trial(command, tmp_path / "trial", cwd=tmp_path, wall_limit=2)
    saved_data = json.loads((tmp_path / "trial" / "trial.json").read_text())

    assert first["returncode"] == second["returncode"] == 0
    assert saved_data["normalized_command"] == [sys.executable, "-c", "pass"]
    assert len(saved_data["attempts"]) == 2
    assert saved_data["attempt_wall_seconds"] == pytest.approx(
        second["attempts"][-1]["wall_seconds"]
    )
    assert saved_data["wall_seconds"] == pytest.approx(
        sum(item["wall_seconds"] for item in saved_data["attempts"])
    )
    assert saved_data["wall_seconds"] >= second["attempt_wall_seconds"]


def test_run_trial_records_process_tree_rss_and_collect_cost_marks_missing_rank_rss(tmp_path):
    record = parameter_study.run_trial(
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
        parameter_study.collect_cost(tmp_path / "output")["peak_rank_aggregate_rss_bytes"] is None
    )


def test_checkpoint_info_works_from_an_installed_style_tree_without_git(tmp_path, monkeypatch):
    fake_module = tmp_path / "installed" / "openonda" / "run_records.py"
    fake_module.parent.mkdir(parents=True)
    fake_module.write_text("# installed\n")
    input_file = tmp_path / "setup.py"
    input_file.write_text("setup = 1\n")
    monkeypatch.setattr(run_records, "__file__", str(fake_module))

    destination = tmp_path / "run" / "study_metadata.json"
    run_records.write_metadata(
        destination, status="running", config={"kind": "pilot"}, inputs=(input_file,)
    )
    saved_data = json.loads(destination.read_text())

    assert saved_data["git_diff_hash"] == "unknown"
    assert saved_data["inputs"][str(input_file)]
    assert (
        saved_data["software_versions_and_hash"]["schema"]
        == "openonda-software-versions-and-hash/2"
    )


def test_software_parameter_hash_is_portable_and_tracks_solver_source(tmp_path):
    first = tmp_path / "first"
    (first / "openonda").mkdir(parents=True)
    (first / "source").mkdir()
    (first / "openonda" / "solver.py").write_text("VALUE = 1\n")
    (first / "source" / "numerics.py").write_text("VALUE = 2\n")
    second = tmp_path / "second"
    shutil.copytree(first, second)

    left = run_records.software_versions_and_hash(first)
    right = run_records.software_versions_and_hash(second)
    assert left["digest"] == right["digest"]
    (second / "source" / "numerics.py").write_text("VALUE = 3\n")
    changed = run_records.software_versions_and_hash(second)
    assert changed["digest"] != left["digest"]


def test_solver_comparison_uses_one_wall_limit_per_case_and_resume_without_overwrite(
    tmp_path, monkeypatch
):
    solver_comparison = load_asset("compare_solvers.py")
    calls = []

    def fake_trial(command, directory, *, cwd, wall_limit):
        calls.append((command, directory, cwd, wall_limit))
        return {
            "command": command,
            "returncode": 0,
            "timed_out": False,
            "console_log": str(directory / "console.log"),
        }

    monkeypatch.setattr(solver_comparison, "run_trial", fake_trial)
    reference = solver_comparison.load_case_module(solver_comparison.CASE_DIR / "reference_flow")
    monkeypatch.setattr(solver_comparison, "load_case_module", lambda *args: reference)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_solvers.py",
            "--run-dir",
            str(tmp_path / "solver_comparison"),
            "--pilot",
            "--sensitivity",
            "none",
        ],
    )

    assert solver_comparison.main() == 0
    assert len(calls) == 2
    assert all(call[3] == 43200 for call in calls)
    assert [call[0][call[0].index("--kind") + 1] for call in calls] == ["reference", "coupled"]
    reference_command, coupled_command = calls[0][0], calls[1][0]
    assert reference_command[reference_command.index("--reference-cores") + 1] == "1"
    assert "cores=1" in coupled_command
    assert (tmp_path / "solver_comparison" / "solver_comparison.json").is_file()

    (tmp_path / "solver_comparison" / "reference").mkdir()
    (tmp_path / "solver_comparison" / "coupled" / "grid_h0p09").mkdir(parents=True)
    calls.clear()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_solvers.py",
            "--run-dir",
            str(tmp_path / "solver_comparison"),
            "--pilot",
            "--resume",
        ],
    )
    assert solver_comparison.main() == 0
    assert all("--resume" in call[0] for call in calls)

    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "sentinel").write_text("keep\n")
    monkeypatch.setattr(sys, "argv", ["compare_solvers.py", "--run-dir", str(existing), "--pilot"])
    with pytest.raises(FileExistsError):
        solver_comparison.main()
    assert (existing / "sentinel").read_text() == "keep\n"


def test_sensitivity_factor_set_keeps_interface_cycles_fixed():
    sensitivity = load_asset("run_sensitivity.py")

    assert sensitivity.FACTORS["particle_spacing_ratio"] == (1.25, 1.5)
    assert "interface_iterations" not in sensitivity.FACTORS
    assert "transfer_vorticity_cutoff" not in sensitivity.FACTORS
    assert "interface iteration limits and tolerances are fixed" in (
        sensitivity.Path(sensitivity.__file__).read_text().lower()
    )


def test_coupled_parameter_study_configuration_uses_the_actual_mesh(monkeypatch):
    parameter_study = load_asset("run_parameter_study.py")
    module = parameter_study.load_case_module(parameter_study.CASE_DIR)
    monkeypatch.setattr(parameter_study, "file_hash", lambda _: "test-input")
    monkeypatch.setattr(
        parameter_study, "software_versions_and_hash", lambda: {"digest": "test-source"}
    )
    config = parameter_study._resolved_coupled_config(module, 0.8, {"hxy": 0.048})
    assert config["hxy"] == pytest.approx(0.048)
    assert config["dz"] == config["span"] == pytest.approx(1.0)
    assert config["span_layers"] == config["particle_span_layers"] == 1
    assert config["plane_z"] == 0.0
    assert config["interface_iterations"] == module.INTERFACE_ITERATIONS == 6
    assert config["exchange_dt"] == pytest.approx(0.04)


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
        sensitivity, "collect_cost", lambda _root, **kwargs: {"unconverged_stationary_intervals": 0}
    )
    actual_case = sensitivity.load_case_module(sensitivity.CASE_DIR)
    monkeypatch.setattr(
        sensitivity,
        "load_case_module",
        lambda directory, *args: actual_case if directory == sensitivity.CASE_DIR else Post(),
    )
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
        sensitivity, "collect_cost", lambda _root, **kwargs: {"unconverged_stationary_intervals": 0}
    )
    actual_case = sensitivity.load_case_module(sensitivity.CASE_DIR)
    monkeypatch.setattr(
        sensitivity,
        "load_case_module",
        lambda directory, *args: actual_case if directory == sensitivity.CASE_DIR else Post(),
    )
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
    assert [row["screened_exchange_steps"] for row in report["runs"]] == [20, 50, 10]


def test_screen_sensitivity_rejects_incompatible_step_budget():
    sensitivity = load_asset("run_sensitivity.py")

    with pytest.raises(ValueError, match="common screen duration"):
        sensitivity._screen_steps(21, 0.08)


def test_interaction_selector_rejects_invalid_body_blend_weight_pair():
    sensitivity = load_asset("run_sensitivity.py")

    class Builder:
        def build_case(self, *, overrides):
            if overrides.get("particle_spacing_ratio") == 1.5 and overrides.get(
                "blend_width_ratio"
            ) == pytest.approx(7.0 * 0.08 / 0.12):
                raise ValueError("blend width exceeds body blend_weight")

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
            "reason": "blend width exceeds body blend_weight",
        }
    ]


def test_reference_grid_completion_requires_matching_resolved_record(tmp_path):
    parameter_study = load_asset("run_parameter_study.py")

    class ReferenceModule:
        SPAN = 1.0
        TIME_STEP_SIZE = 0.001
        CORES = 1

    config = parameter_study._grid_config(ReferenceModule, "grid_h008", 0.08, 4.0)
    assert config["span"] == config["dz"] == 1.0
    assert config["span_layers"] == 1
    parameter_study._write_grid_record(tmp_path, config)
    samples = tmp_path / "samples" / "grid_h008"
    solution = tmp_path / "solution" / "grid_h008"
    samples.mkdir(parents=True)
    solution.mkdir(parents=True)
    (samples / "grid_run.json").write_text(
        json.dumps({"case": "grid_h008", "cell_size": 0.08, "end_time": 4.0})
    )
    (solution / "fvm_metadata.json").write_text(
        json.dumps({"run_status": {"status": "complete"}, "state": {"time": 4.0}})
    )

    assert parameter_study._grid_complete(tmp_path, ReferenceModule, "grid_h008", 0.08, 4.0)
    assert not parameter_study._grid_complete(tmp_path, ReferenceModule, "grid_h008", 0.04, 4.0)


def test_incomplete_reference_output_requires_resume(tmp_path, monkeypatch):
    parameter_study = load_asset("run_parameter_study.py")

    class ReferenceModule:
        SPAN = 1.0
        TIME_STEP_SIZE = 0.001
        CORES = 1

    monkeypatch.setattr(parameter_study, "load_case_module", lambda *args: ReferenceModule)
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
            "resume": False,
        },
    )()

    with pytest.raises(RuntimeError, match="use --resume"):
        parameter_study.run_reference(options, tmp_path)


def test_coupled_resume_delegates_latest_selection_without_backup(tmp_path, monkeypatch):
    parameter_study = load_asset("run_parameter_study.py")

    class CoupledModule:
        END_TIME = 100.0
        VPM_TIME_STEP_SIZE = 0.04

        @staticmethod
        def create_solver(**kwargs):
            assert kwargs.get("restart_from") is None
            return 2500

    resolved = {
        "kind": "coupled",
        "overrides": {},
        "end_time": 100.0,
        "source_hash": "test",
        "exchange_dt": 0.04,
    }
    expected = {"kind": "coupled", "overrides": {}, "end_time": 100.0, "resolved": resolved}
    (tmp_path / "study_metadata.json").write_text(json.dumps({"config": expected}))
    (tmp_path / "partial").write_text("state\n")
    monkeypatch.setattr(parameter_study, "load_case_module", lambda *args: CoupledModule)
    monkeypatch.setattr(parameter_study, "_resolved_coupled_config", lambda *args: resolved)
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

    parameter_study.run_coupled(options, tmp_path)
    assert (tmp_path / "COMPLETE").is_file()


def test_coupled_resume_skips_exact_completed_parameter_study_without_backup(tmp_path, monkeypatch):
    parameter_study = load_asset("run_parameter_study.py")

    class CoupledModule:
        END_TIME = 100.0
        VPM_TIME_STEP_SIZE = 0.04

    resolved = {"kind": "coupled", "overrides": {}, "end_time": 100.0, "source_hash": "test"}
    expected = {"kind": "coupled", "overrides": {}, "end_time": 100.0, "resolved": resolved}
    (tmp_path / "study_metadata.json").write_text(json.dumps({"config": expected}))
    (tmp_path / "COMPLETE").write_text("complete\n")
    monkeypatch.setattr(parameter_study, "load_case_module", lambda *args: CoupledModule)
    monkeypatch.setattr(parameter_study, "_resolved_coupled_config", lambda *args: resolved)
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

    assert parameter_study.run_coupled(options, tmp_path) is None


def test_external_mpi_selects_one_broadcast_run_directory(tmp_path, monkeypatch):
    parameter_study = load_asset("run_parameter_study.py")

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
    monkeypatch.setattr(parameter_study, "_explicit_mpi", lambda: True)
    monkeypatch.setattr(parameter_study, "_mpi_comm", lambda: root_comm)
    root_path = parameter_study.select_run_directory(options)
    assert root_path.parent == tmp_path

    monkeypatch.setattr(parameter_study, "_mpi_comm", lambda: worker_comm)
    worker_path = parameter_study.select_run_directory(options)
    assert worker_path == root_path


def test_external_mpi_collective_preflight_runs_callback_once(monkeypatch):
    parameter_study = load_asset("run_parameter_study.py")

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
    monkeypatch.setattr(parameter_study, "_explicit_mpi", lambda: True)
    monkeypatch.setattr(parameter_study, "_mpi_comm", lambda: root_comm)
    assert parameter_study._collective_preflight(lambda: calls.append("root") or ["pending"]) == [
        "pending"
    ]
    monkeypatch.setattr(parameter_study, "_mpi_comm", lambda: worker_comm)
    assert parameter_study._collective_preflight(lambda: calls.append("worker")) == ["pending"]
    assert calls == ["root"]


def test_external_mpi_root_output_failure_reaches_all_ranks(monkeypatch):
    parameter_study = load_asset("run_parameter_study.py")

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
    monkeypatch.setattr(parameter_study, "_explicit_mpi", lambda: True)
    monkeypatch.setattr(parameter_study, "_mpi_comm", lambda: root_comm)
    with pytest.raises(RuntimeError, match="root action failed"):
        parameter_study._collective_root_action(lambda: (_ for _ in ()).throw(OSError("read-only")))

    monkeypatch.setattr(parameter_study, "_mpi_comm", lambda: worker_comm)
    with pytest.raises(RuntimeError, match="root action failed"):
        parameter_study._collective_root_action(lambda: pytest.fail("worker executed root action"))


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
    monkeypatch.setattr(parameter_study, "_ProcessTreeRSSSampler", _BudgetSampler)
    monkeypatch.setattr(
        parameter_study.subprocess,
        "Popen",
        lambda *args, **kwargs: _BudgetChild(waits),
    )
    command = [sys.executable, "-c", "pass"]

    record = parameter_study.run_trial(command, tmp_path / "trial", cwd=tmp_path, wall_limit=3.5)

    assert waits == [pytest.approx(3.5)]
    assert record["returncode"] == 0
    assert record["remaining_wall_seconds"] == pytest.approx(3.5)


def test_run_trial_resume_uses_remaining_case_budget(monkeypatch, tmp_path):
    waits = []
    monkeypatch.setattr(parameter_study, "_ProcessTreeRSSSampler", _BudgetSampler)
    monkeypatch.setattr(
        parameter_study.subprocess,
        "Popen",
        lambda *args, **kwargs: _BudgetChild(waits),
    )
    command = [sys.executable, "-c", "pass", "--resume"]
    normalized = [sys.executable, "-c", "pass"]
    _write_prior_trial(tmp_path / "trial", normalized, wall_seconds=2.25)

    record = parameter_study.run_trial(command, tmp_path / "trial", cwd=tmp_path, wall_limit=5.0)

    assert waits == [pytest.approx(2.75)]
    assert record["returncode"] == 0
    assert record["remaining_wall_seconds"] == pytest.approx(2.75)


def test_run_trial_changed_resume_command_consumes_prior_budget(monkeypatch, tmp_path):
    waits = []
    monkeypatch.setattr(parameter_study, "_ProcessTreeRSSSampler", _BudgetSampler)
    monkeypatch.setattr(
        parameter_study.subprocess,
        "Popen",
        lambda *args, **kwargs: _BudgetChild(waits),
    )
    normalized = [sys.executable, "-c", "pass", "--end-time", "1"]
    _write_prior_trial(tmp_path / "trial", normalized, wall_seconds=2.25)
    command = [sys.executable, "-c", "pass", "--end-time", "2", "--resume"]

    record = parameter_study.run_trial(command, tmp_path / "trial", cwd=tmp_path, wall_limit=5.0)

    assert waits == [pytest.approx(2.75)]
    assert len(record["attempts"]) == 2
    assert record["attempts"][0]["command"][-2:] == ["1", "--resume"]
    assert record["attempts"][1]["command"][-2:] == ["2", "--resume"]


def test_run_trial_exhausted_resume_does_not_spawn(monkeypatch, tmp_path):
    command = [sys.executable, "-c", "pass", "--resume"]
    normalized = [sys.executable, "-c", "pass"]
    _write_prior_trial(tmp_path / "trial", normalized, wall_seconds=2.0)

    monkeypatch.setattr(
        parameter_study.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("an exhausted case must not spawn"),
    )
    record = parameter_study.run_trial(command, tmp_path / "trial", cwd=tmp_path, wall_limit=2.0)

    assert record["returncode"] == 124
    assert record["timed_out"] is True
    assert record["budget_exhausted"] is True
    assert record["wall_seconds"] == pytest.approx(2.0)
    assert record["peak_process_tree_rss_bytes"] == 99
    assert len(record["attempts"]) == 1
    assert "wall budget exhausted" in (tmp_path / "trial" / "console.log").read_text()


@pytest.mark.parametrize("saved_data", ["{not-json", "[]"])
def test_run_trial_malformed_resume_record_fails_closed(monkeypatch, tmp_path, saved_data):
    trial = tmp_path / "trial"
    trial.mkdir()
    (trial / "trial.json").write_text(saved_data)
    monkeypatch.setattr(
        parameter_study.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("malformed resume must not spawn"),
    )

    with pytest.raises(ValueError, match="malformed trial record"):
        parameter_study.run_trial(
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
        parameter_study.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("invalid resume must not spawn"),
    )

    with pytest.raises(ValueError, match="malformed trial attempt"):
        parameter_study.run_trial(
            [sys.executable, "-c", "pass", "--resume"],
            trial,
            cwd=tmp_path,
            wall_limit=2.0,
        )
