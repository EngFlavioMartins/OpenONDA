"""The public FVM grid-study API produces numerical and plot artifacts."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import openonda.fvm as fvm


class Serial:
    is_partitioned = False
    is_root = True

    @staticmethod
    def barrier():
        return None


def write_samples(directory: Path, spacing: float) -> None:
    directory.mkdir(parents=True)
    time = np.linspace(0.0, 10.0, 201)
    with (directory / "forces_history.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "time",
                "drag_coefficient",
                "lift_coefficient",
                "side_force_coefficient",
            )
        )
        writer.writerow((0.0, 99.0, 99.0, 99.0))
        writer.writerow((10.0, 99.0, 99.0, 99.0))
        for value in time:
            writer.writerow(
                (
                    value,
                    1.0 + spacing**2,
                    0.2 * np.sin(2.0 * np.pi * 0.2 * value),
                    0.0,
                )
            )
    for name, y in (("centreline", 0.0), ("offaxis_y075", 0.75)):
        with (directory / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("time", "position_x", "position_y", "position_z", "velocity_x"))
            writer.writerow((5.0, -1.0, y, 0.0, 99.0))
            writer.writerow((10.0, -1.0, y, 0.0, 99.0))
            for value in np.linspace(5.0, 10.0, 11):
                for x in np.linspace(-1.0, 4.0, 21):
                    writer.writerow((value, x, y, 0.0, np.tanh(x) + spacing**2))


def test_three_completed_runs_create_numbers_and_plot(tmp_path):
    for index, (name, spacing) in enumerate(
        (("coarse", 0.125), ("medium", 0.0625), ("fine", 0.03125)), start=1
    ):
        samples = tmp_path / "samples" / name
        write_samples(samples, spacing)
        solver = SimpleNamespace(
            flush_output=lambda: None,
            parallel=Serial(),
            mesh_data={"n_cells": index * 1000},
            samples_dir=str(samples),
            solution_dir=str(tmp_path / "solution" / name),
            setup=SimpleNamespace(case_name=name),
            time=10.0,
        )
        report = fvm.update_grid_study(
            solver,
            spacing,
            profiles=("centreline", "offaxis_y075"),
        )

    assert report is not None
    assert report["convergence"]["mean_drag"]["observed_order"] == pytest.approx(2.0)
    assert report["convergence"]["mean_drag"]["richardson_extrapolated"] == pytest.approx(1.0)
    assert report["profiles"]["centreline"]["medium_to_fine_l2"] > 0.0
    assert report["profiles"]["offaxis_y075"]["medium_to_fine_l2"] > 0.0
    for name in ("grid_study.json", "grid_study.csv", "grid_study.md", "grid_study.png"):
        assert (tmp_path / "solution" / name).stat().st_size > 0
    saved = json.loads((tmp_path / "solution/grid_study.json").read_text())
    assert [grid["case"] for grid in saved["grids"]] == ["coarse", "medium", "fine"]
