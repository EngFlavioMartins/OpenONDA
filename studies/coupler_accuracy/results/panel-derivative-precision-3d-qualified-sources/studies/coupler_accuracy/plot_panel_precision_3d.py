#!/usr/bin/env python3
"""Recheck saved precision metrics and plot the frozen 3D panel derivative test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from studies.coupler_accuracy.cube_boundary_oracle import ROOT, hash_file
from studies.coupler_accuracy.cube_native_moment_reconstruction_3d import read_arrays


def run(args):
    path = args.source / "panel-derivative-precision-3d.json"
    report = json.loads(path.read_text())
    assert report["status"] == "complete" and report["spatial_dimensions"] == 3
    for row in report["sources"]:
        assert hash_file(ROOT / row["path"]) == row
    fields_path = ROOT / report["fields"]["path"]
    assert hash_file(fields_path) == report["fields"]
    f = read_arrays(fields_path)
    names = list(dict.fromkeys(row["source_name"] for row in report["records"]))
    count = len(names)

    def norm(value):
        return float(np.sqrt(np.dot(f["area"], np.linalg.norm(value, axis=1)**2)/f["area"].sum()))

    differences = []
    for row in report["records"]:
        state, mode = names.index(row["source_name"]), row["mode"]
        u, g, gh = (f[mode+suffix] for suffix in ("__velocity", "__gradient", "__gradient_half"))
        truth = f["quadrature_tangential_gradient"]
        response, expected = g[state+count]-g[state], truth[state+count]-truth[state]
        metrics = {"velocity_error_rms": norm(u[state]-f["quadrature_velocity"][state]),
                   "gradient_error_rms": norm(g[state]-truth[state]), "gradient_half_step_error_rms": norm(gh[state]-truth[state]),
                   "gradient_step_halving_change_rms": norm(gh[state]-g[state]),
                   "one_ulp_strength_response_rms": norm(response), "quadrature_one_ulp_strength_response_rms": norm(expected),
                   "one_ulp_strength_response_error_rms": norm(response-expected)}
        for key, value in metrics.items():
            difference = abs(value-row[key])
            assert difference < 1e-15
            differences.append(difference)
    single = [row for row in report["records"] if row["mode"] == "runtime_f32_evaluation_float32"]
    promoted = [row for row in report["records"] if row["mode"] == "runtime_f32_evaluation_float64"]
    assert [row["source_name"] for row in single] == [row["source_name"] for row in promoted] == names
    labels, x = ["Constant", "LSQ\nmoments", "Native\nmoments", "Linear\nmoments"], np.arange(count)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    for rows, shift, color, label in ((single, -.18, "#b45334", "Single-precision queries"),
                                     (promoted, .18, "#166c91", "Double-precision queries")):
        axes[0].bar(x+shift, [row["gradient_error_rms"] for row in rows], .34, color=color, label=label)
        axes[1].bar(x+shift, [row["one_ulp_strength_response_rms"] for row in rows], .34, color=color, label=label)
    axes[1].plot(x, [row["quadrature_one_ulp_strength_response_rms"] for row in single], "ko", markersize=4, label="Surface-quadrature response")
    axes[0].set(title="Tangential derivative error", ylabel="Error RMS [U∞/D]")
    axes[1].set(title="Response to one-ULP strength changes", ylabel="Derivative change RMS [U∞/D]")
    for ax in axes:
        ax.set_yscale("log")
        ax.set_xticks(x, labels)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle("Same rounded 3D cube panel field: only target evaluation precision changes", fontsize=13)
    fig.savefig(args.output, dpi=170)
    plt.close(fig)
    own = Path(__file__).resolve()
    archive = args.output.parent / (args.output.stem+"-sources") / own.relative_to(ROOT)
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(own.read_bytes())
    result = {"schema": "openonda-panel-precision-figure-3d/1", "status": "complete", "figure": hash_file(args.output),
              "independent_metrics": len(differences), "maximum_metric_difference": max(differences),
              "sources": [hash_file(path), report["fields"], hash_file(own)],
              "limitations": ["Frozen physical panel fields and controlled one-ULP perturbations; not an advancing coupling or force result."]}
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({key: result[key] for key in ("independent_metrics", "maximum_metric_difference")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.source, args.output = args.source.resolve(), args.output.resolve()
    if args.output.exists():
        parser.error("Choose a new figure path")
    run(args)
