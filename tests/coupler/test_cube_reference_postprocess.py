"""Regression coverage for the non-destructive cube grid-convergence report."""

import csv
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "tutorials"
    / "coupled_fvm_vpm"
    / "02_cube_flow"
    / "reference_flow"
    / "postprocess_grid_study.py"
)
ALLRUN = SCRIPT.with_name("allrun.sh")


def test_cube_launcher_declares_geometric_levels_and_identical_temporal_mesh(tmp_path):
    stub = tmp_path / "python"
    stub.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "with open(os.environ['CALLS'], 'a') as out: out.write(json.dumps(sys.argv[1:])+'\\n')\n"
    )
    stub.chmod(0o755)
    calls = tmp_path / "calls.jsonl"
    env = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ["PATH"], CALLS=str(calls))
    subprocess.run(
        ["/bin/bash", str(ALLRUN)], cwd=tmp_path, env=env, check=True, capture_output=True
    )
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    spacings = [float(c[c.index("--dx") + 1]) for c in commands[:4]]
    np.testing.assert_allclose(np.array(spacings[:-1]) / spacings[1:], 1.5)
    fine, temporal = commands[2], commands[4]
    assert (
        temporal[temporal.index("--mesh") + 1]
        == "campaigns/geometric_r15_30s_fine/solution/grid_h0045/fvm/mesh.npz"
    )
    assert (
        float(temporal[temporal.index("--max-dt") + 1])
        == float(fine[fine.index("--max-dt") + 1]) / 2
    )
    assert [float(c[c.index("--end-time") + 1]) for c in commands] == [120, 120, 30, 120, 120]
    assert float(fine[fine.index("--output-interval") + 1]) == 0.25
    assert float(fine[fine.index("--backup-interval") + 1]) == 0.25
    for command in (commands[0], commands[1], commands[3], temporal):
        assert "--output-interval" not in command
        assert "--backup-interval" not in command


def _load_postprocessor():
    name = "cube_reference_postprocess_test"
    spec = spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_force_history(directory: Path, spacing: float) -> str:
    times = np.linspace(0.0, 20.0, 401)
    rows = [
        (
            time,
            1.0 + spacing**2,
            0.2 * np.sin(2.0 * np.pi * 0.2 * time) + spacing**2,
            0.1 * np.cos(2.0 * np.pi * 0.2 * time) + 0.5 * spacing**2,
        )
        for time in times
    ]
    path = directory / "forces_history.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "time",
                "drag_coefficient",
                "lift_coefficient",
                "side_force_coefficient",
            )
        )
        writer.writerows(rows)
    return path.read_text(encoding="utf-8")


def _write_profile(directory: Path, name: str, spacing: float, y: float) -> None:
    times = np.linspace(0.0, 20.0, 81)
    positions = np.linspace(-2.0, 5.0, 29)
    with (directory / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "time",
                "position_x",
                "position_y",
                "position_z",
                "velocity_x",
                "velocity_y",
                "velocity_z",
            )
        )
        for time in times:
            for x in positions:
                writer.writerow(
                    (
                        time,
                        x,
                        y,
                        0.0,
                        np.tanh(x) + spacing**2,
                        0.1 * np.sin(2.0 * np.pi * 0.2 * time),
                        0.0,
                    )
                )


def _write_grid(samples: Path, name: str, spacing: float, cells: int) -> str:
    directory = samples / name
    directory.mkdir(parents=True)
    force_text = _write_force_history(directory, spacing)
    _write_profile(directory, "centreline", spacing, 0.0)
    _write_profile(directory, "offaxis_y075", spacing, 0.75)
    (directory / "grid_run.json").write_text(
        json.dumps(
            {
                "schema": "openonda-fvm-grid-run/1",
                "case": name,
                "cell_size": spacing,
                "cell_count": cells,
                "end_time": 20.0,
                "profiles": ["centreline", "offaxis_y075"],
            }
        ),
        encoding="utf-8",
    )
    return force_text


def test_cube_postprocessor_compares_every_completed_grid_without_mutating_samples(tmp_path):
    postprocess = _load_postprocessor()
    samples = tmp_path / "samples"
    original_force = _write_grid(samples, "very_coarse", 0.25, 600)
    for name, spacing, cells in (
        ("coarse", 0.125, 3_000),
        ("medium", 0.0625, 20_000),
        ("fine", 0.03125, 150_000),
        ("very_fine", 0.015625, 1_100_000),
    ):
        _write_grid(samples, name, spacing, cells)

    orphan = samples / "old_unregistered_case"
    orphan.mkdir()
    (orphan / "forces_history.csv").write_text("time,drag_coefficient\n0,1\n", encoding="utf-8")
    output = tmp_path / "solution"
    report = postprocess.analyse_grid_convergence(samples, output, tolerance=0.01)

    assert [grid["case"] for grid in report["grids"]] == [
        "very_coarse",
        "coarse",
        "medium",
        "fine",
        "very_fine",
    ]
    assert report["reference_case"] == "very_fine"
    assert report["convergence"]["mean_drag"]["richardson"]["available"]
    assert report["convergence"]["mean_drag"]["richardson"]["observed_order"] == pytest.approx(2.0)
    assert report["convergence"]["mean_drag"]["finest_pair"]["meets_tolerance"]
    assert report["profiles"]["centreline"]["available"]
    assert report["profiles"]["centreline"]["comparisons"]["coarse"]["relative_l2"] > 0.0
    assert report["excluded_cases"] == [
        {
            "directory": "old_unregistered_case",
            "reason": "forces_history.csv exists but grid_run.json is absent",
        }
    ]
    assert (samples / "very_coarse" / "forces_history.csv").read_text(
        encoding="utf-8"
    ) == original_force
    for name in (
        "auxiliary/grid_convergence.json",
        "auxiliary/grid_convergence.csv",
        "auxiliary/grid_convergence.md",
        "grid_convergence.png",
        "grid_convergence_by_cells.png",
        "grid_convergence_profiles.png",
    ):
        assert (output / name).stat().st_size > 0


def test_cube_campaign_realizes_geometric_spacings_on_identical_domains(monkeypatch):
    spec = spec_from_file_location("cube_campaign_setup_test", SCRIPT.with_name("setup.py"))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    captured = []
    monkeypatch.setattr(
        module.fvm, "create_fvm_solver", lambda setup, **kwargs: captured.append((setup, kwargs))
    )
    spacings = [0.10125, 0.0675, 0.045, 0.03]
    for h in spacings:
        module.create_solver(
            "case", h, campaign=True, lean=True, cores=2, end_time=120, max_dt=0.005
        )
    for (setup, kwargs), h in zip(captured, spacings, strict=True):
        assert kwargs["mesh"].cell_size_anchor == h
        np.testing.assert_allclose(
            kwargs["mesh"].domain.bounds,
            [-6.48, 12.96, -6.48, 6.48, -6.48, 6.48],
            rtol=0,
            atol=1e-12,
        )
        assert setup.time.output_schedule.final_only
        assert setup.backup.schedule.every_time == 5
    np.testing.assert_allclose(np.array(spacings[:-1]) / spacings[1:], 1.5)
    assert "./allclean.sh" not in ALLRUN.read_text()


def test_temporal_configuration_reuses_source_and_restart_owns_its_mesh(tmp_path):
    from studies.panel_removal.cube_reference_campaign import campaign_mesh

    root = tmp_path / "campaign"
    with pytest.raises(FileNotFoundError, match="Run the spatial fine stage"):
        campaign_mesh(root / "temporal", "time_h0045_dt_half", None, None)
    fine = root / "solution/grid_h0045/fvm/mesh.npz"
    fine.parent.mkdir(parents=True)
    fine.write_bytes(b"original native mesh")
    assert campaign_mesh(root / "temporal", "time_h0045_dt_half", None, None) == fine
    assert campaign_mesh(root / "temporal", "explicit_control", fine, None) == fine
    own = root / "temporal/solution/time_h0045_dt_half/fvm/mesh.npz"
    own.parent.mkdir(parents=True)
    own.write_bytes(fine.read_bytes())
    fine.unlink()
    assert (
        campaign_mesh(root / "temporal", "time_h0045_dt_half", fine, own.parent.parent / "backup")
        == own
    )
    assert own.read_bytes() == b"original native mesh"


@pytest.mark.parametrize(
    "output_interval,backup_interval,expected_output,expected_backup",
    [(0.25, 0.25, 0.25, 0.25), (0.25, None, 0.25, 5.0), (None, 0.25, None, 0.25)],
)
def test_fine_output_and_checkpoint_schedules_are_independent(
    monkeypatch, output_interval, backup_interval, expected_output, expected_backup
):
    spec = spec_from_file_location("cube_output_schedule_test", SCRIPT.with_name("setup.py"))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    captured = []
    monkeypatch.setattr(
        module.fvm, "create_fvm_solver", lambda setup, **kwargs: captured.append(setup)
    )
    module.create_solver(
        "grid_h0045",
        0.045,
        campaign=True,
        lean=True,
        end_time=30,
        max_dt=0.005,
        output_interval=output_interval,
        backup_interval=backup_interval,
    )
    setup = captured[0]
    assert setup.time.end_time == 30
    assert setup.time.output_schedule.every_time == expected_output
    assert setup.time.output_schedule.final_only == (expected_output is None)
    assert setup.backup.schedule.every_time == expected_backup
    assert setup.backup.write_at_end
    assert [sampler.schedule.every_time for sampler in setup.samplers] == [0.05, 0.25, 0.25]
    assert setup.time.adjustment.maximum_time_step_size == 0.005
    assert setup.transport.kinematic_viscosity == 0.001


@pytest.mark.parametrize("state", ["active", "complete", "reused_command", "reused_time", "gone"])
def test_campaign_admission_checks_live_process_identity(tmp_path, state):
    import psutil

    from studies.panel_removal import cube_reference_campaign as module

    directory = tmp_path / "studies/panel_removal/runs/cube"
    directory.mkdir(parents=True)
    record = directory / "process.json"
    record.write_text(
        json.dumps(
            {
                "pid": 123,
                "command": [
                    "python",
                    "-m",
                    "studies.panel_removal.run_cube",
                    "--output",
                    str(directory),
                ],
            }
        )
    )

    class Process:
        def __init__(self, pid):
            assert pid == 123
            if state == "gone":
                raise psutil.NoSuchProcess(pid)

        def is_running(self):
            return state != "complete"

        def status(self):
            return "running"

        def create_time(self):
            return record.stat().st_mtime + (3600 if state == "reused_time" else 0)

        def cmdline(self):
            return [
                "prterun",
                "-n",
                "4",
                "python",
                str(tmp_path / "studies/panel_removal/run_cube.py"),
                "--output",
                str(directory if state != "reused_command" else tmp_path / "other"),
            ]

    assert module._recorded_experiment_is_active(record, tmp_path, Process) == (state == "active")


def test_campaign_admission_waits_then_releases_and_respects_bypass(tmp_path, monkeypatch):
    from studies.panel_removal import cube_reference_campaign as module

    fake_setup = tmp_path / "studies/panel_removal/cube_reference_campaign.py"
    monkeypatch.setattr(module, "__file__", str(fake_setup))
    record = tmp_path / "studies/panel_removal/runs/cube/process.json"
    record.parent.mkdir(parents=True)
    record.write_text("{}")
    states = iter([True, False])
    monkeypatch.setattr(module, "_recorded_experiment_is_active", lambda *args: next(states))
    sleeps = []
    monkeypatch.setattr(module.time, "sleep", sleeps.append)
    module.wait_for_coupled_experiments(True, 0.045, output_root=tmp_path)
    assert sleeps == [30]
    monkeypatch.setattr(
        module,
        "_recorded_experiment_is_active",
        lambda *args: pytest.fail("bypass inspected processes"),
    )
    module.wait_for_coupled_experiments(False, 0.045, output_root=tmp_path)
    module.wait_for_coupled_experiments(True, 0.10125, output_root=tmp_path)
    monkeypatch.setenv("CUBE_WAIT_FOR_COUPLED", "0")
    module.wait_for_coupled_experiments(True, 0.03, output_root=tmp_path)


def test_four_grid_asymptotic_check_is_independent_of_finest_triplet():
    module = _load_postprocessor()
    grids = [
        {"case": str(i), "wall_cell_size": h, "force_statistics": {"mean_drag": 1 + h * h}}
        for i, h in enumerate([0.10125, 0.0675, 0.045, 0.03])
    ]
    result = module._convergence_statistics(grids, "mean_drag", None)["richardson"]
    assert result["observed_order"] == pytest.approx(2)
    assert result["asymptotic_check"]["observed_to_predicted"] == pytest.approx(1)
    grids[-1]["force_statistics"]["mean_drag"] += 0.0002
    result = module._convergence_statistics(grids, "mean_drag", None)["richardson"]
    assert result["available"]
    assert not result["asymptotic_check"]["within_ten_percent"]


def test_frequency_screen_rejects_window_drift_and_short_periodic_record():
    module = _load_postprocessor()
    time = np.linspace(15, 30, 301)
    for signal in (time * 0.001, np.exp((time - 30) / 5), np.sin(2 * np.pi * 0.2 * time)):
        result = module._frequency_diagnostic(time, signal)
        assert result["strouhal"] is None
        assert "fewer than five cycles" in result["reason"]
    assert module._frequency_diagnostic(time, np.ones_like(time))["strouhal"] is None


def test_resolved_frequency_uses_recorded_physical_scales():
    module = _load_postprocessor()
    time = np.linspace(0, 100, 2001)
    signal = 0.02 + 0.1 * np.sin(2 * np.pi * 0.2 * time)
    result = module._frequency_diagnostic(time, signal, length=2, speed=4)
    assert result["reason"] is None
    assert result["strouhal"] == pytest.approx(0.1, rel=0.005)
    assert result["strouhal_resolution"] == pytest.approx(0.005)


def test_statistics_expose_drift_and_preserve_pressure_viscous_drag_closure():
    module = _load_postprocessor()
    time = np.array([0.0, 0.1, 0.7, 1.5, 2.0])
    pressure = 1.0 + time
    viscous = -0.05 * np.ones_like(time)
    history = module.ForceHistory(
        time,
        {
            "drag_coefficient": (pressure + viscous) / 2,
            "pressure_force_x": pressure,
            "viscous_force_x": viscous,
        },
    )
    result = module._force_statistics(
        history, 0, 2, context={"length": 1, "speed": 1, "force_scale": 2}
    )
    assert result["mean_drag"] == pytest.approx(0.975)
    assert result["drag_half_means"] == pytest.approx([0.725, 1.225])
    assert result["drag_drift_relative"] == pytest.approx(0.5 / 0.975)
    assert result["mean_pressure_drag"] + result["mean_viscous_drag"] == pytest.approx(
        result["mean_drag"]
    )
    assert result["strouhal_lift"] is None


def test_report_withholds_gci_during_drift_and_prints_statistics(tmp_path, monkeypatch, capsys):
    module = _load_postprocessor()
    for function in ("_plot_force_metrics", "_plot_profiles", "_plot_histories"):
        monkeypatch.setattr(module, function, lambda *args, **kwargs: None)
    samples = tmp_path / "samples"
    for name, spacing, cells in (("coarse", 0.2, 100), ("medium", 0.1, 200), ("fine", 0.05, 400)):
        _write_grid(samples, name, spacing, cells)
        path = samples / name / "forces_history.csv"
        rows = list(csv.DictReader(path.open()))
        for row in rows:
            row["drag_coefficient"] = str(
                float(row["drag_coefficient"]) - 0.01 * float(row["time"])
            )
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report = module.analyse_grid_convergence(samples, tmp_path / "output")
    assert not report["convergence"]["mean_drag"]["richardson"]["available"]
    assert "drift" in report["convergence"]["mean_drag"]["richardson"]["reason"]
    module.print_statistics(report)
    printed = capsys.readouterr().out
    assert "Mean Cd" in printed and "Cd drift" in printed and "St lift" in printed
    assert "NOT ESTABLISHED" in printed


def test_recorded_setting_changes_are_detected():
    module = _load_postprocessor()
    baseline = {
        "configuration": {"schemes": "backward"},
        "domain_bounds": [-1, 1],
        "mesh_controls": {"method": "same"},
        "warnings": [],
    }
    changed = {**baseline, "domain_bounds": [-2, 2]}
    report = module._comparability({"medium": changed, "fine": baseline}, "fine")
    assert not report["matching_recorded_settings"]
    assert report["differences_to_reference"] == {"medium": ["domain_bounds"]}
