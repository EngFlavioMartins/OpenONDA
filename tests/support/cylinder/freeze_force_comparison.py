"""Freeze complete force samples through a requested accepted flow time.

An appending live CSV is read once. Only complete published lines with clocks
through the requested endpoint are retained; its live digest need not remain
unchanged. Complete-cycle medians are then measured in the common window.
This utility does not read, initialise, or modify any numerical solver.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from .analyze_force_cycles import FIELDS, complete_cycles, window_statistics

REPOSITORY = Path(__file__).resolve().parents[3]
CASE = REPOSITORY / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow"
STUDY = Path(__file__).resolve().parent / "force_boundary_study/20261005T214514Z"


def freeze_complete_interval(source, destination, end):
    captured_start = datetime.now(UTC).isoformat()
    captured = source.read_bytes()
    captured_end = datetime.now(UTC).isoformat()
    complete = captured[: captured.rfind(b"\n") + 1]
    lines = complete.splitlines(keepends=True)
    retained = [lines[0]]
    for line in lines[1:]:
        if float(line.split(b",", 1)[0]) <= end + 1e-8:
            retained.append(line)
    content = b"".join(retained)
    values = np.atleast_1d(np.genfromtxt(io.BytesIO(content), delimiter=",", names=True))
    if np.any(np.diff(values["time"]) <= 0) or any(
        not np.isfinite(values[field]).all() for field in ("time", *FIELDS)
    ):
        raise ValueError("Force snapshot contains invalid values or clocks")
    if abs(values["time"][-1] - end) > 1e-8:
        raise ValueError("Requested accepted endpoint is not published in the force history")
    if (
        np.max(abs(2 * values["total_force_x"] - values["drag_coefficient"])) > 1e-11
        or np.max(abs(2 * values["total_force_y"] - values["lift_coefficient"])) > 1e-11
    ):
        raise ValueError("Force coefficient normalization differs from the cylinder convention")
    destination.write_bytes(content)
    return values, {
        "source": str(source),
        "snapshot": str(destination),
        "captured_start_utc": captured_start,
        "captured_end_utc": captured_end,
        "captured_live_bytes_sha256": hashlib.sha256(captured).hexdigest(),
        "immutable_snapshot_sha256": hashlib.sha256(content).hexdigest(),
        "published_source_complete_last_time": float(lines[-1].split(b",", 1)[0]),
        "retained_last_time": float(values["time"][-1]),
        "retained_sample_count": len(values),
        "partial_final_line_bytes_omitted": len(captured) - len(complete),
        "snapshot_rule": "Complete lines from one byte read with time <= requested end + 1e-8; live source may keep appending.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coupled", type=Path, default=CASE / "samples/forces_history.csv")
    parser.add_argument(
        "--reference", type=Path, default=CASE / "reference_flow/samples/forces_history.csv"
    )
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    args = parser.parse_args()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = args.study.resolve() / f"force_baseline_{args.start:g}_{args.end:g}_{stamp}"
    output.mkdir()
    snapshots = output / "snapshots"
    snapshots.mkdir()
    sources = {}
    statistics = {}
    for label, path in (("coupled", args.coupled), ("reference", args.reference)):
        values, sources[label] = freeze_complete_interval(
            path.resolve(), snapshots / f"{label}_forces_history.csv", args.end
        )
        cycles = {field: complete_cycles(values, field, 2.0) for field in FIELDS}
        statistics[label] = window_statistics(values, cycles, args.start, args.end, 2.0)
    ratios = {
        field: {
            quantity: statistics["coupled"][field][quantity]
            / statistics["reference"][field][quantity]
            if statistics["coupled"][field][quantity] is not None
            and statistics["reference"][field][quantity] is not None
            else None
            for quantity in (
                "mean",
                "median_peak_to_peak_raw",
                "median_peak_to_peak_drift_corrected",
                "whole_window_peak_to_peak",
            )
        }
        for field in FIELDS
    }
    report = {
        "schema": "openonda-cylinder-frozen-force-comparison/1",
        "captured_at": datetime.now(UTC).isoformat(),
        "output": str(output),
        "window": [args.start, args.end],
        "sources": sources,
        "statistics": statistics,
        "coupled_to_reference_ratios": ratios,
        "normalisation": {
            "density": 1,
            "reference_speed": 1,
            "reference_area": 1,
            "coefficient_factor": 2,
        },
        "analysis_source": str(Path(__file__).resolve()),
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cycle_definition": "Peak bracketed by troughs entirely in the common window; Cd and Cl measured separately. Raw amplitude uses actual extrema; drift-corrected subtracts the interpolated trough baseline.",
        "limitations": [
            "These are measured force histories and do not identify a causal mechanism or demonstrate saturation.",
            "A later candidate run must be compared over the identical accepted-time window after excluding its own restart transient.",
        ],
    }
    (output / "force_comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "ratios": ratios,
                "cycle_counts": {
                    label: {field: values[field]["complete_cycle_count"] for field in FIELDS}
                    for label, values in statistics.items()
                },
                "raw_medians": {
                    label: {field: values[field]["median_peak_to_peak_raw"] for field in FIELDS}
                    for label, values in statistics.items()
                },
                "drift_corrected_medians": {
                    label: {
                        field: values[field]["median_peak_to_peak_drift_corrected"]
                        for field in FIELDS
                    }
                    for label, values in statistics.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
