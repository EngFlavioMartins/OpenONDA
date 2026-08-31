#!/usr/bin/env python3
"""Compare common-window cylinder force statistics across the four grids."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

CASE_DIR = Path(__file__).resolve().parents[1]
CASES = (
    ("very_coarse", 1.0 / 12.0),
    ("coarse", 1.0 / 24.0),
    ("medium", 1.0 / 36.0),
    ("fine", 1.0 / 48.0),
)
STATISTICS_WINDOW = 30.0


def force_history(case_name: str) -> dict[str, np.ndarray]:
    path = CASE_DIR / "samples" / case_name / "forces_history.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        raise ValueError(f"No force samples in {path}")
    data = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        for name in ("time", "drag_coefficient", "lift_coefficient")
    }
    if not all(np.all(np.isfinite(values)) for values in data.values()):
        raise ValueError(f"Non-finite force data in {path}")
    if np.any(np.diff(data["time"]) <= 0.0):
        raise ValueError(f"Force times are not strictly increasing in {path}")
    return data


def time_mean(time: np.ndarray, values: np.ndarray) -> float:
    return float(np.trapezoid(values, time) / (time[-1] - time[0]))


def strouhal_number(time: np.ndarray, lift: np.ndarray) -> float:
    centred = lift - time_mean(time, lift)
    indices = np.flatnonzero((centred[:-1] <= 0.0) & (centred[1:] > 0.0))
    crossings = []
    for index in indices:
        fraction = -centred[index] / (centred[index + 1] - centred[index])
        crossings.append(time[index] + fraction * (time[index + 1] - time[index]))
    periods = np.diff(crossings)
    if len(periods) < 2:
        raise ValueError("Fewer than three rising lift zero-crossings in the statistics window")
    return float(1.0 / np.median(periods))


def statistics(history: dict[str, np.ndarray], start: float, end: float) -> dict[str, float]:
    keep = (history["time"] >= start - 1.0e-12) & (history["time"] <= end + 1.0e-12)
    time = history["time"][keep]
    drag = history["drag_coefficient"][keep]
    lift = history["lift_coefficient"][keep]
    if len(time) < 4 or time[-1] - time[0] < 0.99 * (end - start):
        raise ValueError("Incomplete common force-statistics window")
    mean_drag = time_mean(time, drag)
    mean_lift = time_mean(time, lift)
    return {
        "samples": len(time),
        "mean_cd": mean_drag,
        "cd_rms": float(np.sqrt(time_mean(time, (drag - mean_drag) ** 2))),
        "cd_peak_to_peak": float(np.ptp(drag)),
        "mean_cl": mean_lift,
        "cl_rms": float(np.sqrt(time_mean(time, (lift - mean_lift) ** 2))),
        "cl_amplitude": 0.5 * float(np.ptp(lift)),
        "strouhal": strouhal_number(time, lift),
    }


def main() -> None:
    histories = {name: force_history(name) for name, _dx in CASES}
    common_end = min(float(history["time"][-1]) for history in histories.values())
    if common_end < STATISTICS_WINDOW:
        raise ValueError(
            f"Common final time is {common_end:g}; require at least {STATISTICS_WINDOW:g}"
        )
    common_start = common_end - STATISTICS_WINDOW
    records = []
    for name, dx in CASES:
        records.append(
            {
                "case": name,
                "dx": dx,
                **statistics(histories[name], common_start, common_end),
            }
        )

    metrics = ("mean_cd", "cd_rms", "cd_peak_to_peak", "cl_rms", "cl_amplitude", "strouhal")
    comparisons = []
    for coarser, finer in zip(records[:-1], records[1:], strict=True):
        comparisons.append(
            {
                "coarser": coarser["case"],
                "finer": finer["case"],
                "relative_change": {
                    metric: abs(finer[metric] - coarser[metric]) / max(abs(finer[metric]), 1.0e-14)
                    for metric in metrics
                },
            }
        )

    report = {
        "common_window": {"start": common_start, "end": common_end},
        "cases": records,
        "comparisons": comparisons,
    }
    output = CASE_DIR / "solution" / "grid_study.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
