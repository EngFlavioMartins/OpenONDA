#!/usr/bin/env python3
"""Compute force statistics for the coupled cylinder case."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

CASE_DIR = Path(__file__).resolve().parents[1]
STATISTICS_WINDOW = 30.0


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


def main() -> None:
    path = CASE_DIR / "samples" / "forces_history.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        raise ValueError(f"No force samples in {path}")

    time = np.asarray([float(row["time"]) for row in rows], dtype=np.float64)
    drag = np.asarray([float(row["drag_coefficient"]) for row in rows], dtype=np.float64)
    lift = np.asarray([float(row["lift_coefficient"]) for row in rows], dtype=np.float64)
    if not np.all(np.isfinite(np.column_stack((time, drag, lift)))):
        raise ValueError(f"Non-finite force samples in {path}")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError(f"Force-sample times are not strictly increasing in {path}")
    if time[-1] < STATISTICS_WINDOW:
        raise ValueError(f"Force history ends at t={time[-1]:g}; require t>={STATISTICS_WINDOW:g}")

    start = time[-1] - STATISTICS_WINDOW
    keep = time >= start - 1.0e-12
    time = time[keep]
    drag = drag[keep]
    lift = lift[keep]
    mean_drag = time_mean(time, drag)
    mean_lift = time_mean(time, lift)
    report = {
        "statistics_window": {"start": float(time[0]), "end": float(time[-1])},
        "samples": int(len(time)),
        "mean_cd": mean_drag,
        "cd_rms": float(np.sqrt(time_mean(time, (drag - mean_drag) ** 2))),
        "mean_cl": mean_lift,
        "cl_rms": float(np.sqrt(time_mean(time, (lift - mean_lift) ** 2))),
        "cl_amplitude": 0.5 * float(np.ptp(lift)),
        "strouhal": strouhal_number(time, lift),
    }
    output = CASE_DIR / "solution" / "cylinder_statistics.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
