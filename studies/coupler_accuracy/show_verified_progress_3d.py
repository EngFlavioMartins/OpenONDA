#!/usr/bin/env python3
"""Show the measured short-run accuracy gain and explicitly scoped wall times."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def run(root, output):
    if output.exists():
        raise FileExistsError(output)

    def record(path):
        return {"path": str(path.relative_to(root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    results = root / "studies/coupler_accuracy/results"
    checks = [results / "precision-twenty-step-verification.json",
              results / "body-transport-twenty-comparison-qualified/body-transport-comparison-3d.json"]
    sources = {}
    reports = []
    for path in checks:
        value = json.loads(path.read_text())
        assert value["status"] == "complete" and value["spatial_dimensions"] == 3
        reports.append(value)
        for row in [record(path), *value["sources"]]:
            assert record(root / row["path"]) == row
            sources[row["path"]] = row
    assert reports[0]["maximum_metric_difference"] == 0
    assert reports[1]["maximum_scalar_check_difference"] == 0
    names = ["precision-twenty-step-control", "precision-twenty-step-twelve", "body-transport-twenty-iterated"]
    labels = ["Uniterated baseline", "Iterated interface", "Iteration + body transport"]
    rows = []
    histories = []
    for name, label in zip(names, labels, strict=True):
        path = results / name / "trial/cube-coupled-trial.json"
        assert sources[str(path.relative_to(root))] == record(path)
        trial = json.loads(path.read_text())
        assert trial["status"] == "complete" and trial["identical_native_shared_cells"]
        assert trial["small_cells"] == 16936 and trial["full_cells"] == 53752
        assert trial["particle_spacing"] == trial["requested_wall_spacing"] == .0625
        history = trial["comparison"]
        assert len(history) == 21 and history[0]["physical_time"] == .5 and history[-1]["physical_time"] == 1.5
        if histories:
            assert history[0] == histories[0][0]
            for a, b in zip(history, histories[0], strict=True):
                for key in ("physical_time", "full_drag_coefficient"):
                    assert a[key] == b[key]
        error = np.array([(value["hybrid_drag_coefficient"] - value["full_drag_coefficient"]) / value["full_drag_coefficient"] for value in history[1:]])
        rows.append({"label": label, "run_directory": name,
                     "drag_history_relative_rms_percent": float(100 * np.sqrt(np.mean(error**2))),
                     "maximum_absolute_relative_drag_percent": float(100 * np.max(np.abs(error))),
                     "final_near_body_velocity_error_percent_Uinf": 100 * history[-1]["fvm_near_body_velocity_rms_over_Uinf"],
                     "final_whole_fvm_velocity_error_percent_Uinf": 100 * history[-1]["fvm_velocity_rms_over_Uinf"],
                     "recorded_comparison_run_wall_seconds": trial["wall_time_seconds"]})
        histories.append(history)
    assert histories[1] == reports[1]["baseline_history"] and histories[2] == reports[1]["candidate_history"]
    reductions = {key: 100 * (1 - rows[-1][key] / rows[0][key]) for key in (
        "drag_history_relative_rms_percent", "final_near_body_velocity_error_percent_Uinf", "final_whole_fvm_velocity_error_percent_Uinf")}
    output.mkdir(parents=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    time = np.array([row["physical_time"] for row in histories[0]])
    for row, history, color, style in zip(rows, histories, ("#7a8794", "#12668a", "#b24f2b"), ("--", ":", "-"), strict=True):
        errors = [100 * (value["hybrid_drag_coefficient"] / value["full_drag_coefficient"] - 1) for value in history]
        axes[0].plot(time, errors, style, color=color, linewidth=2,
                     label=f"{row['label']} · RMS {row['drag_history_relative_rms_percent']:.3f}%")
        axes[1].plot(time, [100 * value["fvm_near_body_velocity_rms_over_Uinf"] for value in history], style,
                     color=color, linewidth=2, label=row["label"])
    axes[0].axhline(0, color=".4", linewidth=.8)
    axes[0].set(title="Drag error against independent full FVM", ylabel="Signed relative drag error (%)")
    axes[1].set(title="Near-body velocity error against full FVM", ylabel="Volume-weighted vector RMS / U∞ (%)", ylim=(0, None))
    for axis in axes:
        axis.set_xlabel("Physical flow time")
        axis.grid(alpha=.2)
        axis.legend(fontsize=8, loc="best")
    fig.suptitle("Verified fully 3D cube · identical near-body cells, h = 0.0625\nSmall FVM box: three body widths · short window, t = 0.5–1.5")
    figures = []
    for extension in ("png", "svg"):
        path = output / ("verified-progress." + extension)
        fig.savefig(path, dpi=170)
        figures.append(record(path))
    plt.close(fig)
    own = Path(__file__).resolve()
    sources[str(own.relative_to(root))] = record(own)
    result = {"schema": "openonda-verified-progress-evidence-3d/1", "status": "complete",
              "spatial_dimensions": 3, "physical_time_bounds": [.5, 1.5], "comparisons": rows,
              "relative_error_reductions_percent": reductions, "figures": figures,
              "sources": list(sources.values()), "limitations": [
                  "The baseline is the qualified uniterated experiment, not an archived pristine version of the user's original project.",
                  "The controlled study uses h=0.0625, which differs from the original tutorial's medium mesh. It does not establish tutorial or developed-wake agreement.",
                  "Recorded wall times include both independently advancing full FVM and hybrid simulations, setup, checks and output. Observer workloads and background load differ; these are not a controlled speed benchmark.",
                  "No hybrid-versus-full-FVM speedup has been established by these timings. The latest accuracy changes remain experimental."]}
    (output / "verified-progress-evidence-3d.json").write_text(json.dumps(result, indent=2) + "\n")
    archive = output / "show_verified_progress_3d.py"
    archive.write_bytes(own.read_bytes())
    print(json.dumps({"comparisons": rows, "relative_error_reductions_percent": reductions}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(Path(__file__).resolve().parents[2], args.output.resolve())
