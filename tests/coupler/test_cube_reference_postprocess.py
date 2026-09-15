"""Regression coverage for the non-destructive cube grid-convergence report."""

import csv
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
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
        "grid_convergence.json",
        "grid_convergence.csv",
        "grid_convergence.md",
        "grid_convergence.png",
        "grid_convergence_by_cells.png",
        "grid_convergence_profiles.png",
    ):
        assert (output / name).stat().st_size > 0


def test_cube_grid_runner_matches_the_declarative_cylinder_style():
    script = ALLRUN.read_text(encoding="utf-8")

    assert "./allclean.sh" not in script
    assert "postprocess_grid_study.py" not in script
    assert "run_case" not in script
    for name, spacing in (
        ("very_coarse", "0.12"),
        ("coarse", "0.10"),
        ("medium", "0.08"),
        ("fine", "0.06"),
    ):
        assert f"python setup.py --name {name}" in script
        assert f"--dx {spacing}" in script


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
