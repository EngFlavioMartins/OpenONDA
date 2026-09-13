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
        ("very_coarse", "0.22"),
        ("coarse", "0.20"),
        ("medium", "0.18"),
        ("fine", "0.16"),
        ("very_fine", "0.14"),
    ):
        assert f"python setup.py --name {name}" in script
        assert f"--dx {spacing}" in script
