#!/usr/bin/env python3
"""Plot a verified 3D interface-iteration trajectory against its exact control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from studies.coupler_accuracy.cube_boundary_oracle import ROOT, hash_file


def run(args):
    verification = json.loads(args.verification.read_text())
    assert verification["status"] == "complete" and verification["spatial_dimensions"] == 3
    assert verification["comparison_intervals"] > 1
    control_row = next(row for row in verification["experiments"] if row["directory"] == verification["comparison_control_directory"])
    candidates = [row for row in verification["experiments"] if row["maximum_sweeps"] > 1
                  and row["observed_intervals"] == verification["comparison_intervals"]]
    assert len(candidates) == 1
    candidate_row = candidates[0]
    reports, sources = [], [hash_file(args.verification)]
    for row in (control_row, candidate_row):
        path = ROOT / row["directory"] / "interface-iteration-3d.json"
        expected = next(item for item in verification["sources"] if item["path"] == str(path.relative_to(ROOT)))
        assert hash_file(path) == expected
        reports.append(json.loads(path.read_text()))
        sources.append(expected)
    control, candidate = reports
    time = np.array([row["physical_time"] for row in control["comparison"]])
    reference = np.array([row["full_drag_coefficient"] for row in control["comparison"]])
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8), layout="constrained")
    axes[0, 0].plot(time, reference, color="0.2", label="Full FVM")
    for report, color, label in ((control, "#718096", "Original coupling"), (candidate, "#156d91", "Iterated interface")):
        rows = report["comparison"]
        drag = np.array([row["hybrid_drag_coefficient"] for row in rows])
        axes[0, 0].plot(time, drag, "o--", color=color, label=label, markersize=3)
        axes[0, 1].plot(time, 100*(drag/reference-1), "o-", color=color, label=label, markersize=3)
        for metric, style, region in (("fvm_velocity_rms_over_Uinf", "-", "Whole small FVM"),
                                       ("fvm_near_body_velocity_rms_over_Uinf", "--", "Near body")):
            axes[1, 0].plot(time, [row[metric] for row in rows], style, color=color, label=f"{label}: {region}")
    axes[0, 0].set(title="Force history", ylabel="Drag coefficient")
    axes[0, 1].axhline(0, color="0.6", linewidth=1)
    axes[0, 1].set(title="Difference at the same physical time", ylabel="Drag coefficient difference (%)")
    axes[1, 0].set(title="Velocity agreement", ylabel="Volume-weighted velocity RMS / U∞")
    last = {row["coupling_step"]: row for row in candidate["sweeps"]}
    assert len(last) == verification["comparison_intervals"]
    final = list(last.values())
    endpoint_time = np.array([time[row["coupling_step"]] for row in final])
    for key, color, label in (("normal_residual_rms", "#156d91", "Normal velocity"),
                              ("gradient_residual_rms", "#ad572f", "Tangential derivative")):
        axes[1, 1].semilogy(endpoint_time, [row[key] for row in final], "o-", color=color, label=label, markersize=3)
    assert candidate["normal_tolerance"] == candidate["gradient_tolerance"]
    axes[1, 1].axhline(candidate["gradient_tolerance"], linestyle=":", color="0.4", label="Fixed convergence threshold")
    capped = np.array([not row["converged"] for row in final])
    axes[1, 1].scatter(endpoint_time[capped], np.array([row["gradient_residual_rms"] for row in final])[capped],
                       marker="x", s=55, color="#b42b35", label="Unconverged at sweep cap", zorder=3)
    axes[1, 1].set(title=f"Endpoint mismatch ({sum(~capped)}/{len(final)} intervals converged)",
                   ylabel="Residual (unit freestream and cube side)")
    for ax in axes.flat:
        ax.set_xlabel("Physical time")
        ax.grid(alpha=.18)
        ax.legend(fontsize=8)
    figure.suptitle("Fully 3D medium cube: fixed small domain and identical near-body cells", fontsize=13)
    figure.savefig(args.output, dpi=170)
    plt.close(figure)
    own = Path(__file__).resolve()
    archive = args.output.parent / (args.output.stem+"-sources") / own.relative_to(ROOT)
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(own.read_bytes())
    record = {"schema": "openonda-interface-trajectory-figure-3d/1", "status": "complete", "figure": hash_file(args.output),
              "sources": [*sources, hash_file(own)], "control": control_row, "candidate": candidate_row,
              "scope": "Short matched 3D trajectory with explicit convergence flags; no developed-wake or machine-precision agreement is implied."}
    args.output.with_suffix(".json").write_text(json.dumps(record, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.verification, args.output = args.verification.resolve(), args.output.resolve()
    if args.output.exists():
        parser.error("Choose a new figure path")
    run(args)
