"""Regression tests for the minimal cylinder grid-study interface."""

from __future__ import annotations

import csv
import inspect
import json

import numpy as np

from source.solvers.fvm.factory import create_fvm_solver
from tutorials.coupled_fvm_vpm.cylinder_shedding_flow.reference_flow import setup
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
    assert report["production_cases"] == ["coarse", "medium", "fine"]
    assert report["refinement_ratio"] == 1.5
    np.testing.assert_allclose(report["cases"][-1]["mean_cd"], 1.3, atol=1.0e-12)
    np.testing.assert_allclose(report["cases"][-1]["strouhal"], 0.2, atol=2.0e-3)
    assert len(report["comparisons"]) == 3
    assert report["grid_convergence"]["mean_cd"]["status"] == "converged_to_roundoff"


def test_grid_study_uses_reasonable_monotone_wall_spacings():
    names = [case for case, _dx in postprocess.CASES]
    spacing = np.asarray([dx for _case, dx in postprocess.CASES])

    assert names == ["very_coarse", "coarse", "medium", "fine"]
    np.testing.assert_allclose(spacing, [1 / 12, 1 / 24, 1 / 36, 1 / 54])
    assert np.all(np.diff(spacing) < 0.0)
    production = np.asarray([dx for _case, dx in postprocess.PRODUCTION_CASES])
    np.testing.assert_allclose(production[:-1] / production[1:], 1.5)


def test_grid_study_preserves_the_refinement_ratio_at_every_octree_level():
    meshers = [setup.grid_mesh(dx) for _case, dx in postprocess.PRODUCTION_CASES]

    np.testing.assert_allclose(
        [mesher.max_cell_size for mesher in meshers],
        8.0 * np.asarray([dx for _case, dx in postprocess.PRODUCTION_CASES]),
    )
    for mesher, (_case, dx) in zip(meshers, postprocess.PRODUCTION_CASES, strict=True):
        assert mesher.boundary_layers == ()
        assert mesher.effective_cell_size(dx) == dx
        assert mesher.effective_cell_size(2.0 * dx) == 2.0 * dx
        assert mesher.effective_cell_size(4.0 * dx) == 4.0 * dx


def test_richardson_gci_recovers_second_order_limit():
    exact = 1.25
    records = [
        {"case": case, "dx": dx, "mean_cd": exact + 2.0 * dx**2}
        for case, dx in postprocess.PRODUCTION_CASES
    ]

    result = postprocess.richardson_gci(records, "mean_cd", tolerance_percent=1.0)

    assert result["status"] == "asymptotic"
    np.testing.assert_allclose(result["observed_order"], 2.0, atol=1.0e-12)
    np.testing.assert_allclose(result["richardson_extrapolated_value"], exact, atol=1.0e-12)
    assert result["passed"]


def test_fvm_solver_api_owns_named_solution_and_sample_directories():
    parameters = inspect.signature(create_fvm_solver).parameters

    assert "solution_dir" in parameters
    assert "samples_dir" in parameters


def test_reference_grid_disables_the_unstable_extra_nonorthogonal_sweep():
    controls = setup.solver_setup("qualification", 1.0 / 36.0).pimple

    assert controls.n_correctors == 2
    assert controls.n_orthogonal_correctors == 0
