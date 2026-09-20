"""Regression coverage for the cube reference grid study."""

import csv
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import shlex
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "tutorials/coupled_fvm_vpm/02_cube_flow/reference_flow"


def load_script(name: str):
    path = CASE / name
    spec = spec_from_file_location(f"cube_reference_{path.stem}_test", path)
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_allrun_declares_the_four_geometric_grids():
    commands = [
        shlex.split(line)
        for line in (CASE / "allrun.sh").read_text().splitlines()
        if line.startswith("python ")
    ]
    assert commands == [
        ["python", "setup.py", "--name", "grid_h010125", "-h", "0.10125"],
        ["python", "setup.py", "--name", "grid_h00675", "-h", "0.0675"],
        ["python", "setup.py", "--name", "grid_h0045", "-h", "0.045"],
        ["python", "setup.py", "--name", "grid_h003", "-h", "0.03"],
    ]
    spacings = np.array([float(command[-1]) for command in commands])
    np.testing.assert_allclose(spacings[:-1] / spacings[1:], 1.5)


def test_setup_uses_the_requested_name_and_baseline_spacing(monkeypatch):
    setup = load_script("setup.py")
    captured = {}

    def create(config, **kwargs):
        captured.update(config=config, **kwargs)
        return object()

    monkeypatch.setattr(setup.fvm, "create_fvm_solver", create)
    setup.create_solver("grid_h0045", 0.045)

    assert captured["config"].case_name == "grid_h0045"
    assert captured["solution_dir"] == CASE / "solution/grid_h0045"
    assert captured["samples_dir"] == CASE / "samples/grid_h0045"
    assert captured["mesh"].cell_size_anchor == pytest.approx(0.045)
    np.testing.assert_allclose(captured["mesh"].domain.bounds, setup.DOMAIN)


def write_grid(samples: Path, name: str, h: float, cells: int) -> str:
    directory = samples / name
    directory.mkdir(parents=True)
    time = np.linspace(0.0, 30.0, 601)
    force_path = directory / "forces_history.csv"
    with force_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "time",
                "drag_coefficient",
                "lift_coefficient",
                "side_force_coefficient",
            ]
        )
        for value in time:
            writer.writerow(
                [
                    value,
                    1.0 + h**2 + 0.01 * np.sin(2 * np.pi * 0.2 * value),
                    0.1 * np.sin(2 * np.pi * 0.2 * value),
                    0.05 * np.cos(2 * np.pi * 0.2 * value),
                ]
            )
    (directory / "grid_run.json").write_text(
        json.dumps(
            {
                "case": name,
                "cell_size": h,
                "cell_count": cells,
                "end_time": 30.0,
            }
        )
    )
    return force_path.read_text()


def test_force_postprocessor_preserves_inputs_and_reports_gci(tmp_path):
    postprocess = load_script("postprocess_grid_study.py")
    samples = tmp_path / "samples"
    original = None
    for index, (name, h) in enumerate(
        [
            ("grid_h010125", 0.10125),
            ("grid_h00675", 0.0675),
            ("grid_h0045", 0.045),
            ("grid_h003", 0.03),
        ],
        start=1,
    ):
        force_text = write_grid(samples, name, h, index * 1000)
        if name == "grid_h0045":
            original = force_text

    output = tmp_path / "figures"
    report = postprocess.analyse_forces(samples, output)

    assert [grid["name"] for grid in report["grids"]] == [
        "grid_h010125",
        "grid_h00675",
        "grid_h0045",
        "grid_h003",
    ]
    mean_drag = report["convergence"]["mean_drag"]
    assert mean_drag["order"] == pytest.approx(2.0, rel=1.0e-4)
    assert mean_drag["fine_gci"] > 0
    assert (samples / "grid_h0045/forces_history.csv").read_text() == original
    for name in ("grid_forces.json", "grid_forces.csv", "grid_forces.png"):
        assert (output / name).stat().st_size > 0
