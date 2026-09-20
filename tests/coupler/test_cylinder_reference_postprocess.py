"""Regression coverage for the cylinder reference grid-study postprocessor."""

import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "tutorials"
    / "coupled_fvm_vpm"
    / "01_cylinder_shedding_flow"
    / "reference_flow"
    / "postprocess_grid_study.py"
)


def _load_postprocessor():
    spec = importlib.util.spec_from_file_location("cylinder_grid_postprocess_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_case(samples: Path, name: str, spacing: float, cells: int) -> None:
    directory = samples / name
    directory.mkdir(parents=True)
    time = np.linspace(0.0, 20.0, 401)
    with (directory / "forces_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time", "drag_coefficient", "lift_coefficient", "side_force_coefficient"))
        for value in time:
            writer.writerow(
                (
                    value,
                    1.0 + spacing**2,
                    0.2 * np.sin(2.0 * np.pi * 0.2 * value) + spacing**2,
                    0.1 * np.cos(2.0 * np.pi * 0.2 * value) + spacing**2,
                )
            )
    with (directory / "centreline.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time", "position_x", "position_y", "position_z", "velocity_x"))
        for value in time[::4]:
            for position in np.linspace(-2.0, 8.0, 21):
                writer.writerow((value, position, 0.0, 0.0, np.tanh(position) + spacing**2))
    (directory / "grid_run.json").write_text(
        json.dumps(
            {
                "schema": "openonda-fvm-grid-run/1",
                "case": name,
                "cell_size": spacing,
                "cell_count": cells,
                "end_time": 20.0,
                "profiles": ["centreline"],
            }
        ),
        encoding="utf-8",
    )


def test_cylinder_postprocessor_compares_completed_grids_without_mutating_inputs(
    tmp_path, monkeypatch
):
    module = _load_postprocessor()
    for function in ("_plot_force_metrics", "_plot_by_cells", "_plot_histories", "_plot_profiles"):
        monkeypatch.setattr(module, function, lambda *args, **kwargs: None)
    samples = tmp_path / "samples"
    for name, spacing, cells in (
        ("very_coarse", 0.16, 500),
        ("coarse", 0.08, 2_000),
        ("medium", 0.04, 8_000),
        ("fine", 0.02, 32_000),
    ):
        _write_case(samples, name, spacing, cells)
    original = (samples / "fine" / "forces_history.csv").read_text(encoding="utf-8")

    report = module.analyse_grid_convergence(samples, tmp_path / "solution", formats=("png",))

    assert report["reference_case"] == "fine"
    assert [grid["case"] for grid in report["grids"]] == [
        "very_coarse",
        "coarse",
        "medium",
        "fine",
    ]
    assert report["convergence"]["mean_drag"]["available"]
    assert report["profiles"]["centreline"]["available"]
    assert report["profiles"]["centreline"]["comparisons"]["coarse"]["relative_l2"] > 0.0
    assert (samples / "fine" / "forces_history.csv").read_text(encoding="utf-8") == original
    assert (tmp_path / "solution/auxiliary/grid_convergence.json").is_file()
    assert (tmp_path / "solution/auxiliary/grid_convergence.csv").is_file()
    assert (tmp_path / "solution/auxiliary/grid_convergence.md").is_file()
