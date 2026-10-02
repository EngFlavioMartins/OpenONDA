"""Launcher/native-continuation contracts and explicitly synthetic plot rendering."""

import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from openonda.tutorial_runner import load_case_module
from source.restart import select_backup

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow"


@pytest.mark.parametrize("arguments", [[], ["--max-coupling-steps", "3"]])
def test_default_launchers_run_one_local_case_and_preserve_outputs(tmp_path, arguments):
    for name in ("allrun.sh", "allcontinue.sh", "allclean.sh"):
        shutil.copy2(CASE / name, tmp_path / name)
    python = tmp_path / "python"
    python.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > invocation.txt\n')
    python.chmod(0o755)
    environment = {**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]}
    current = tmp_path / "solution/native_backup"
    current.parent.mkdir(parents=True)
    current.write_text("checkpoint")
    history = tmp_path / "study_results/cylinder/historical/pipeline_manifest.json"
    history.parent.mkdir(parents=True)
    history.write_text("historical")
    subprocess.run(
        ["bash", str(tmp_path / "allcontinue.sh"), *arguments],
        cwd="/tmp", env=environment, check=True
    )
    assert current.read_text() == "checkpoint"
    assert (tmp_path / "invocation.txt").read_text().splitlines() == ["setup.py", *arguments]
    subprocess.run(["bash", str(tmp_path / "allrun.sh"), *arguments],
                   cwd="/tmp", env=environment, check=True)
    assert current.read_text() == "checkpoint"
    assert history.read_text() == "historical"
    assert (tmp_path / "invocation.txt").read_text().splitlines() == ["setup.py", *arguments]


def test_reference_launcher_selects_one_mesh_and_preserves_outputs(tmp_path):
    shutil.copy2(CASE / "reference_flow/allrun.sh", tmp_path / "allrun.sh")
    python = tmp_path / "python"
    python.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > invocation.txt\n')
    python.chmod(0o755)
    checkpoint = tmp_path / "solution/backup"
    checkpoint.parent.mkdir()
    checkpoint.write_text("checkpoint")
    environment = {**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]}
    subprocess.run(["bash", str(tmp_path / "allrun.sh")], env=environment, check=True)
    assert checkpoint.read_text() == "checkpoint"
    assert (tmp_path / "invocation.txt").read_text().splitlines() == ["setup.py", "-h", "0.04"]


def test_reference_default_factory_uses_plain_case_directories(monkeypatch):
    setup = load_case_module(CASE / "reference_flow")
    captured = {}

    def factory(config, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(setup.fvm, "create_fvm_solver", factory)
    setup.create_solver("phase_h004", 0.04)
    assert captured["solution_dir"] == CASE / "reference_flow/solution"
    assert captured["samples_dir"] == CASE / "reference_flow/samples"


def campaign(monkeypatch):
    module = load_case_module(Path(__file__).resolve().parents[2] / "tests/support/cylinder", "run_campaign")
    monkeypatch.setattr(module, "_collective_preflight", lambda action: action())
    monkeypatch.setattr(module, "_collective_root_action", lambda action: action())
    monkeypatch.setattr(module, "_collective_barrier", lambda: None)
    monkeypatch.setattr(module, "write_manifest", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "initialize_cylinder_perturbation", lambda *args: None)
    return module


def test_reference_campaign_resumes_incomplete_grid_using_native_latest(tmp_path, monkeypatch):
    launcher = campaign(monkeypatch)
    calls = []

    class Solver:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def run(self, **kwargs):
            calls.append(kwargs)
            assert (
                select_backup(
                    "latest", directory=tmp_path / "solution/grid", kind="fvm", backup_path="backup"
                )
                is None
            )

    setup = SimpleNamespace(
        SPAN=1,
        create_solver=lambda *args, **kwargs: Solver(),
        fvm=SimpleNamespace(update_grid_study=lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(launcher, "load_case_module", lambda *args: setup)
    monkeypatch.setattr(launcher, "selected_grids", lambda *args: [("grid", 0.1)])
    monkeypatch.setattr(launcher, "_grid_complete", lambda *args: False)
    monkeypatch.setattr(launcher, "_grid_has_output", lambda *args: True)
    monkeypatch.setattr(launcher, "_grid_config", lambda *args: {"name": "grid"})
    monkeypatch.setattr(launcher, "_write_grid_record", lambda *args: None)
    options = SimpleNamespace(
        override=None, reference_cores=1, end_time=0.2, pilot=True, resume=True, no_analysis=True
    )
    launcher.run_reference(options, tmp_path)
    assert calls == [{"start_from": "latest"}]


def test_coupled_campaign_uses_native_latest_without_special_backup_manifest(tmp_path, monkeypatch):
    launcher = campaign(monkeypatch)
    calls = []

    def create_solver(**kwargs):
        calls.append(kwargs)
        assert (
            select_backup("latest", directory=kwargs["output_root"] / "solution", kind="coupled")
            is None
        )
        return 5

    setup = SimpleNamespace(END_TIME=1, create_solver=create_solver)
    monkeypatch.setattr(launcher, "load_case_module", lambda *args: setup)
    monkeypatch.setattr(launcher, "selected_overrides", lambda *args: {})
    monkeypatch.setattr(launcher, "_resolved_coupled_config", lambda *args: {"exchange_dt": 0.02})
    (tmp_path / "solution").mkdir()
    options = SimpleNamespace(max_coupling_steps=None, pilot=True, end_time=1, resume=True)
    launcher.run_coupled(options, tmp_path)
    assert calls[0].get("restart_from") is None
    assert calls[0]["output_root"] == tmp_path


@pytest.mark.parametrize("kind", ["reference", "coupled"])
def test_campaign_does_not_hide_corrupt_native_backup(tmp_path, monkeypatch, kind):
    launcher = campaign(monkeypatch)
    options = SimpleNamespace(
        override=None,
        reference_cores=1,
        end_time=0.2,
        pilot=True,
        resume=True,
        no_analysis=True,
        max_coupling_steps=None,
    )
    if kind == "reference":
        from source.solvers.fvm.io.backup import load_backup

        directory = tmp_path / "solution/grid"
        directory.mkdir(parents=True)
        (directory / "backup").write_bytes(b"corrupt archive")

        class Solver:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def run(self, **kwargs):
                path = select_backup(
                    kwargs["start_from"], directory=directory, kind="fvm", backup_path="backup"
                )
                load_backup(SimpleNamespace(), path)

        setup = SimpleNamespace(SPAN=1, create_solver=lambda *args, **kwargs: Solver())
        monkeypatch.setattr(launcher, "selected_grids", lambda *args: [("grid", 0.1)])
        monkeypatch.setattr(launcher, "_grid_complete", lambda *args: False)
        monkeypatch.setattr(launcher, "_grid_config", lambda *args: {"name": "grid"})
        message = "FVM restart admission"
        error = RuntimeError
    else:
        from source.coupler.backup import load_coupled_backup

        directory = tmp_path / "solution/backups"
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text("{corrupt")

        def create_solver(**kwargs):
            path = select_backup(
                "latest", directory=kwargs["output_root"] / "solution", kind="coupled"
            )
            initialized = SimpleNamespace(fvm_solver=object(), vpm_solver=object(), _is_master=True)
            return load_coupled_backup(initialized, path)

        setup = SimpleNamespace(END_TIME=0.2, create_solver=create_solver)
        monkeypatch.setattr(launcher, "selected_overrides", lambda *args: {})
        monkeypatch.setattr(
            launcher, "_resolved_coupled_config", lambda *args: {"exchange_dt": 0.02}
        )
        message = "Invalid coupled backup manifest"
        error = ValueError
    monkeypatch.setattr(launcher, "load_case_module", lambda *args: setup)
    with pytest.raises(error, match=message):
        getattr(launcher, "run_" + kind)(options, tmp_path)


@pytest.mark.parametrize("format", ["png", "pdf"])
def test_synthetic_campaign_plots_pass_thesis_contract(tmp_path, format):
    pipeline = load_case_module(Path(__file__).resolve().parents[2] / "tests/support/cylinder", "run_pipeline")
    rows = [
        {
            "h": h,
            "mean_drag": 1.3 + h,
            "rms_lift": 0.4 + h,
            "strouhal": 0.18 + h / 10,
            "uncertainty_95": {},
        }
        for h in (0.1, 0.08, 0.064)
    ]
    profile = {"y": [-1, 0, 1], "mean_velocity": [[1, 0, 0], [0.8, 0, 0], [1, 0, 0]]}
    report = {
        "reference": {"grids": rows},
        "coupled_grids": rows,
        "span_profiles": {
            name: {"reference": profile, "coupled": profile}
            for name in ("span_lower", "span_middle", "span_upper")
        },
    }
    pipeline.plot_results(report, tmp_path, format)
    for name in ("grid_comparison", "span_profiles"):
        assert (tmp_path / f"{name}.{format}").stat().st_size > 1000
