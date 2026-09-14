"""Summarize isolated cube benchmarks without reading or changing live outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from source.solvers.fvm.io.backup import decode_state


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def field_difference(before, after, step=2):
    errors = {}
    for old_path in sorted((before / f"solution/backups/fvm_{step:06d}").glob("rank-*.npz")):
        rank = old_path.name.split("-")[1]
        new_path = next((after / f"solution/backups/fvm_{step:06d}").glob(f"rank-{rank}-*.npz"))
        with np.load(old_path) as old, np.load(new_path) as new:
            a, b = decode_state(dict(old)), decode_state(dict(new))
        for name in ("global_cell_id", "global_face_id"):
            np.testing.assert_array_equal(a[name], b[name])
        for field in ("velocity", "kinematic_pressure", "volumetric_face_flux", "eddy_viscosity"):
            difference = b[field] - a[field]
            aggregate = errors.setdefault(field, [0.0, 0.0, 0.0])
            aggregate[0] = max(aggregate[0], float(np.max(np.abs(difference))))
            aggregate[1] += float(np.sum(difference**2))
            aggregate[2] += float(np.sum(a[field] ** 2))
    if not errors:
        raise ValueError("No rank backups found")
    return {
        field: {"maximum_absolute": values[0], "relative_l2": np.sqrt(values[1] / values[2])}
        for field, values in errors.items()
    }


def case_summary(path, *, reference=False):
    performance = read_jsonl(path / "solution/performance.jsonl")
    with (path / "samples/forces_history.csv").open() as stream:
        forces = [dict(row) for row in csv.DictReader(stream)]
    result = {
        "input_directory": str(path),
        "fvm_solve_count": len(performance),
        "fvm_solve_seconds": sum(row["step_seconds"]["max"] for row in performance),
        "maximum_sum_of_rank_peak_rss_bytes": max(
            row["memory"]["aggregate_peak_rss_end_bytes"] for row in performance
        ),
        "forces": [
            {key: float(row[key]) for key in ("time", "drag_coefficient")} for row in forces
        ],
    }
    if reference:
        result["steps"] = [
            {
                "time": row["time"],
                "dt": row["time_step_size"],
                "seconds": row["step_seconds"]["max"],
            }
            for row in performance
        ]
        result["interval_005_to_010_seconds"] = sum(
            row["step_seconds"]["max"]
            for row in performance
            if 0.05 + 1e-9 < row["time"] < 0.1 + 1e-9
        )
    else:
        diagnostics = read_jsonl(path / "solution/coupler_diagnostics.jsonl")
        result["steps"] = [
            {key: row[key] for key in ("time", "timing_seconds", "interface_iteration")}
            for row in diagnostics
        ]
        result["interval_005_to_010_seconds"] = next(
            row["timing_seconds"]["total"] for row in diagnostics if abs(row["time"] - 0.1) < 1e-9
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "baseline": case_summary(args.baseline),
        "optimized": case_summary(args.optimized),
        "reference": case_summary(args.reference, reference=True),
        "field_difference_including_halos_at_010": field_difference(args.baseline, args.optimized),
        "notes": [
            "Fixed original coupled box [-1.5,1.5]^3; requested fine mesh spacing 0.06.",
            "Original FVM schemes, correctors, timesteps and residual tolerances retained.",
            "All cases use four MPI ranks. Coupled cases use Metal treecode.",
            "Reference benchmark also receives the compiled FVM and PETSc workspace optimizations.",
            "Short startup comparison with concurrent reference jobs and cProfile overhead; not a developed-wake benchmark.",
            "RSS is the sum of rank process peaks, not a measurement of combined CPU/GPU memory.",
        ],
    }
    before, after, reference = (
        result[name]["interval_005_to_010_seconds"]
        for name in ("baseline", "optimized", "reference")
    )
    result["speedup"] = before / after
    result["optimized_to_reference_cost_ratio"] = after / reference
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "measurements.json").write_text(json.dumps(result, indent=2) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    values = [before, after, reference]
    bars = ax.barh(
        ["Previous coupled", "Optimized coupled", "Fine reference"],
        values,
        color=["#9b5055", "#267b97", "#65815d"],
    )
    ax.bar_label(bars, labels=[f"{v:.1f} s" for v in values], padding=6)
    ax.set_xlim(0, max(values) * 1.2)
    ax.invert_yaxis()
    ax.set_xlabel("Wall seconds to advance physical time 0.05 → 0.10 s (lower is better)")
    ax.set_title(
        "Cube runtime: unchanged mesh resolution and FVM discretization", loc="left", pad=16
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(args.output / "runtime.png", dpi=170)
    plt.close(fig)
    print(
        json.dumps({key: result[key] for key in ("speedup", "optimized_to_reference_cost_ratio")})
    )


if __name__ == "__main__":
    main()
