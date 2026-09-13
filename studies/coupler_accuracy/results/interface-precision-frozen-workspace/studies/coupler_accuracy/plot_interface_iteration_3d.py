#!/usr/bin/env python3
"""Plot a verified single-interval interface iteration and its force response."""

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
    source = ROOT / verification["experiments"][-1]["directory"] / "interface-iteration-3d.json"
    expected = next(row for row in verification["sources"] if row["path"] == str(source.relative_to(ROOT)))
    assert hash_file(source) == expected
    report = json.loads(source.read_text())
    rows = report["sweeps"]
    assert {row["coupling_step"] for row in rows} == {1}
    assert rows[-1]["converged"] and report["status"] == "complete"
    baseline = verification["experiments"][0]["final_comparison"]
    ref = baseline["full_drag_coefficient"]
    sweep = [row["sweep"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    ax = axes[0]
    ax.semilogy(sweep, [row["normal_residual_rms"] for row in rows], "o-", label="Normal velocity", color="#146a91")
    ax.semilogy(sweep, [row["gradient_residual_rms"] for row in rows], "s-", label="Tangential derivative", color="#ab542d")
    assert report["normal_tolerance"] == report["gradient_tolerance"]
    ax.axhline(report["normal_tolerance"], color="0.4", linestyle=":", label="Convergence threshold")
    ax.set(title="Boundary mismatch contracts", xlabel="Sweep at the same physical time", ylabel="Residual (unit freestream and cube side)", xticks=sweep)
    ax.legend(fontsize=9)
    ax.grid(alpha=.18)
    ax = axes[1]
    force_error = 100*(np.array([row["hybrid_drag_coefficient"] for row in rows])/ref-1)
    ax.plot(sweep, force_error, "o-", color="#76508f", label="Iterated endpoint")
    ax.axhline(100*baseline["drag_coefficient_difference"]/ref, color="0.4", linestyle="--", label="Original coupling")
    ax.axhline(0, color="0.65", linewidth=1, label="Full FVM reference")
    ax.set(title="Drag moves farther from the reference", xlabel="Sweep at the same physical time", ylabel="Drag coefficient difference (%)", xticks=sweep)
    ax.legend(fontsize=9)
    ax.grid(alpha=.18)
    fig.suptitle("Fully 3D medium cube: one interval from t = 0.50 to 0.55", fontsize=13)
    fig.savefig(args.output, dpi=170)
    plt.close(fig)
    own = Path(__file__).resolve()
    archive = args.output.parent / (args.output.stem+"-sources") / own.relative_to(ROOT)
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(own.read_bytes())
    record = {"schema": "openonda-interface-iteration-figure-3d/1", "status": "complete", "figure": hash_file(args.output),
              "sources": [hash_file(own), hash_file(args.verification), expected],
              "scope": "A single short 3D interval; not the complete 20-interval or developed-wake result."}
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
