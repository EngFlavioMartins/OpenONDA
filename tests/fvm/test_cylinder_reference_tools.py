"""Regression tests for the minimal cylinder grid-study interface."""

from __future__ import annotations

import csv
import inspect
import json

import numpy as np

from source.solvers.fvm.factory import create_fvm_solver
from tutorials.coupled_fvm_vpm.cylinder_shedding_flow.reference_flow.assets import (
    postprocess,
)


def write_force_history(path, time, drag, lift) -> None:
    path.parent.mkdir(parents=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("time", "patch", "drag_coefficient", "lift_coefficient"),
        )
        writer.writeheader()
        writer.writerows(
            {
                "time": sample_time,
                "patch": "cylinder",
                "drag_coefficient": cd,
                "lift_coefficient": cl,
            }
            for sample_time, cd, cl in zip(time, drag, lift, strict=True)
        )


def test_grid_study_force_report_uses_one_common_statistics_window(tmp_path, monkeypatch):
    time = np.arange(0.0, 60.0 + 1.0e-12, 0.02)
    omega = 2.0 * np.pi * 0.2
    for number, (case, _dx) in enumerate(postprocess.CASES, start=1):
        drag = 1.3 + (0.03 / number) * np.cos(2.0 * omega * time)
        lift = (0.4 / number) * np.sin(omega * time)
        write_force_history(
            tmp_path / "samples" / case / "forces_history.csv",
            time,
            drag,
            lift,
        )
    (tmp_path / "solution").mkdir()
    monkeypatch.setattr(postprocess, "CASE_DIR", tmp_path)

    postprocess.main()

    report = json.loads((tmp_path / "solution" / "grid_study.json").read_text())
    assert report["common_window"] == {"start": 30.0, "end": 60.0}
    np.testing.assert_allclose(report["cases"][-1]["mean_cd"], 1.3, atol=1.0e-12)
    np.testing.assert_allclose(report["cases"][-1]["strouhal"], 0.2, atol=2.0e-3)
    assert len(report["comparisons"]) == 3


def test_grid_study_uses_reasonable_monotone_wall_spacings():
    names = [case for case, _dx in postprocess.CASES]
    spacing = np.asarray([dx for _case, dx in postprocess.CASES])

    assert names == ["very_coarse", "coarse", "medium", "fine"]
    np.testing.assert_allclose(spacing, [1 / 12, 1 / 24, 1 / 36, 1 / 48])
    assert np.all(np.diff(spacing) < 0.0)


def test_fvm_solver_api_owns_named_solution_and_sample_directories():
    parameters = inspect.signature(create_fvm_solver).parameters

    assert "solution_dir" in parameters
    assert "samples_dir" in parameters
