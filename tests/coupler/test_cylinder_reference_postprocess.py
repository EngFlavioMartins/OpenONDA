"""Force post-processing for the cylinder reference grids."""

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "tutorials"
    / "coupled_fvm_vpm"
    / "01_cylinder_shedding_flow"
    / "reference_flow"
    / "postprocess_grid_study.py"
)


def load_postprocessor():
    spec = importlib.util.spec_from_file_location("cylinder_force_postprocess", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_grid(samples: Path, name: str, h: float, cells: int) -> None:
    directory = samples / name
    directory.mkdir(parents=True)
    time = np.linspace(0.0, 20.0, 401)
    with (directory / "forces_history.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time", "drag_coefficient", "lift_coefficient", "side_force_coefficient"))
        for value in time:
            writer.writerow(
                (
                    value,
                    1.0 + h**2 + 0.01 * np.sin(4.0 * np.pi * 0.2 * value),
                    0.2 * np.sin(2.0 * np.pi * 0.2 * value),
                    0.0,
                )
            )
    (directory / "grid_run.json").write_text(
        json.dumps({"case": name, "cell_size": h, "cell_count": cells})
    )


def test_force_postprocessor_reports_grid_convergence(tmp_path, monkeypatch):
    module = load_postprocessor()
    samples = tmp_path / "samples"
    spacings = (0.08, 0.08 / np.sqrt(2.0), 0.04, 0.04 / np.sqrt(2.0))
    for index, h in enumerate(spacings):
        write_grid(samples, f"grid_h{index}", h, 1000 * 2**index)
    monkeypatch.setattr(module, "plot_forces", lambda *args: None)

    output = tmp_path / "figures"
    report = module.analyse_forces(samples, output, start=0.0, end=20.0)

    assert [grid["h"] for grid in report["grids"]] == pytest.approx(spacings)
    assert report["grids"][-1]["strouhal"] == pytest.approx(0.2, abs=0.01)
    assert report["convergence"]["mean_drag"]["order"] == pytest.approx(2.0)
    assert (output / "grid_forces.json").is_file()
    assert (output / "grid_forces.csv").is_file()
