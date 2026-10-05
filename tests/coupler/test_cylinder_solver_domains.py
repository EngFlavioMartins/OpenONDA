"""Grid qualification requires matching recorded, realized FVM domains."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from openonda.tutorial_runner import load_case_module

SUPPORT = Path(__file__).resolve().parents[1] / "support/cylinder"


@pytest.mark.parametrize(
    "fine_xmax, qualified", [(2.4 + 1e-13, True), (2.432, False), (None, False)]
)
def test_solver_comparison_preserves_metrics_but_checks_gci_on_actual_domains(
    tmp_path, monkeypatch, fine_xmax, qualified
):
    solver_comparison = load_case_module(SUPPORT, "compare_solvers")
    post = load_case_module(SUPPORT, "postprocess_grid_study")
    metrics = ("mean_drag", "rms_lift", "strouhal")
    gci_calls = []

    def statistics(h):
        return {
            "mean_drag": 1.3 + 0.1 * h**2,
            "rms_lift": 0.4 + 0.01 * h**2,
            "strouhal": 0.18 + 0.01 * h**2,
            "qualified_statistics": True,
            "uncertainty_95": {},
        }

    def fake_trial(command, directory, **_kwargs):
        if command[command.index("--kind") + 1] == "coupled":
            run_dir = Path(command[command.index("--run-dir") + 1])
            samples = run_dir / "samples"
            samples.mkdir(parents=True)
            for name in ("forces_history", "span_middle"):
                (samples / (name + ".csv")).touch()
            xmax = fine_xmax if run_dir.name == "grid_h0p064" else 2.4
            if xmax is not None:
                (run_dir / "solution").mkdir()
                (run_dir / "solution/run_metadata.json").write_text(
                    json.dumps(
                        {
                            "fvm_solver": {
                                "fvm_domain": dict(
                                    zip(
                                        ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"),
                                        (-1.6, xmax, -1.6, 1.6, -0.5, 0.5),
                                        strict=True,
                                    )
                                )
                            }
                        }
                    )
                )
        return {"returncode": 0, "console_log": str(directory / "console.log")}

    def gci(grids, metric):
        gci_calls.append(metric)
        return post.richardson_gci(grids, metric)

    fake_post = SimpleNamespace(
        METRICS=metrics,
        force_statistics=lambda path, *_: statistics(
            float(path.parent.parent.name.removeprefix("grid_h").replace("p", "."))
        ),
        analyse_forces=lambda *_: {
            "force_grid_qualified": True,
            "grids": [{"name": "grid_h0p064", "h": 0.064, **statistics(0.064)}],
        },
        richardson_gci=gci,
        relative_change=post.relative_change,
    )
    actual_case = solver_comparison.load_case_module(solver_comparison.CASE_DIR)
    monkeypatch.setattr(
        solver_comparison,
        "load_case_module",
        lambda directory, *args: (
            actual_case if directory == solver_comparison.CASE_DIR else fake_post
        ),
    )
    monkeypatch.setattr(solver_comparison, "run_trial", fake_trial)
    monkeypatch.setattr(
        solver_comparison,
        "collect_cost",
        lambda _, **kwargs: {"unconverged_stationary_intervals": 0},
    )
    monkeypatch.setattr(solver_comparison, "profile_statistics", lambda *_: {"retained": True})
    monkeypatch.setattr(solver_comparison, "compare_profiles", lambda *_: {"mean_velocity_l2": 0})
    monkeypatch.setattr(solver_comparison, "plot_results", lambda *_: None)
    output = tmp_path / "study"
    monkeypatch.setattr(
        sys, "argv", ["compare_solvers.py", "--run-dir", str(output), "--sensitivity", "none"]
    )

    assert solver_comparison.main() == (0 if qualified else 2)
    report = json.loads((output / "solver_comparison.json").read_text())
    assert report["coupled_grid_domain_check"]["valid"] is qualified
    assert report["coupled_force_grid_qualified"] is qualified
    assert report["accuracy_targets_met"] is qualified
    assert gci_calls == (list(metrics) if qualified else [])
    assert report["coupled_grids"][-1]["mean_drag"] == statistics(0.064)["mean_drag"]
    assert report["coupled_reference_relative_difference"] == dict.fromkeys(metrics, 0)
    assert report["span_profiles"]["span_middle"]["coupled"] == {"retained": True}
    if not qualified:
        assert report["coupled_grid_domain_check"]["reason"]
        assert all(not row["valid"] for row in report["coupled_convergence"].values())


@pytest.mark.parametrize("contents", ["invalid json", "{}", '{"fvm_solver": []}'])
def test_unreadable_domain_metadata_does_not_supply_requested_geometry(tmp_path, contents):
    solver_comparison = load_case_module(SUPPORT, "compare_solvers")
    (tmp_path / "solution").mkdir()
    (tmp_path / "solution/run_metadata.json").write_text(contents)
    assert solver_comparison.recorded_fvm_domain(tmp_path) is None
