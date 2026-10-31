#!/usr/bin/env python3
"""Sampling floor of the shedding statistics from sliding statistics windows.

The study compares mean drag, lift amplitude and Strouhal number over a window of a few
shedding periods. A difference between two runs is resolved only when it exceeds the spread
of the statistic between windows of the same developed flow. This script slides the study
window over every reference and coupled run that extends well beyond the window and reports
the spread of each statistic, in the percentage points used by the sensitivity figures. It
also reports the per-period mean of the lift amplitude next to the global peak-to-peak value.

Run from the case directory: python -m assets.window_uncertainty
Writes figures/auxiliary/window_uncertainty.csv.
"""

import csv
from pathlib import Path

import numpy as np

from assets.plot_forces import DEVELOPED_SHEDDING, read_forces

CASE_DIR = Path(__file__).resolve().parents[1]
WINDOW = 15.0  # s, the study window (40-55 s)
SLIDE = 2.5  # s, window offset between samples
MARGIN = 15.0  # s, a run must reach the window end plus this to be used
DIAMETER = 1.0  # m
FREESTREAM_SPEED = 1.0  # m/s


def window_statistics(forces, start, end):
    """Statistics over one window, with the per-period lift amplitude."""
    window = (forces["time"] >= start) & (forces["time"] <= end)
    time = forces["time"][window]
    lift = forces["lift_coefficient"][window]
    fluctuation = lift - lift.mean()
    up = np.flatnonzero((fluctuation[:-1] < 0.0) & (fluctuation[1:] >= 0.0))
    crossing = time[up] - fluctuation[up] * np.diff(time)[up] / np.diff(fluctuation)[up]
    if crossing.size < 2:
        return None
    period = np.diff(crossing).mean()
    per_period = [
        0.5 * np.ptp(lift[(time >= a) & (time < b)]) for a, b in zip(crossing[:-1], crossing[1:])
    ]
    return {
        "mean drag": forces["drag_coefficient"][window].mean(),
        "lift amplitude": 0.5 * np.ptp(lift),
        "lift amplitude per period": float(np.mean(per_period)),
        "Strouhal number": DIAMETER / (period * FREESTREAM_SPEED),
        "periods": crossing.size - 1,
    }


def sliding_windows(forces):
    """Statistics of every full study window from developed shedding to the end of the run."""
    end_time = forces["time"][-1]
    starts = np.arange(DEVELOPED_SHEDDING, end_time - WINDOW + 1.0e-9, SLIDE)
    rows = []
    for start in starts:
        statistics = window_statistics(forces, start, start + WINDOW)
        if statistics is not None:
            rows.append({"start": start, **statistics})
    return rows


def main():
    paths = [
        CASE_DIR / "reference_flow/samples/forces_history.csv",
        *sorted(CASE_DIR.glob("sensitivity/*/samples/forces_history.csv")),
        CASE_DIR / "samples/forces_history.csv",
    ]
    names = ("mean drag", "lift amplitude", "lift amplitude per period", "Strouhal number")
    table = []
    for path in paths:
        if not path.is_file():
            continue
        forces = np.atleast_1d(read_forces(path))
        if forces["time"][-1] < DEVELOPED_SHEDDING + WINDOW + MARGIN:
            continue  # too short to slide the window
        rows = sliding_windows(forces)
        if len(rows) < 3:
            continue
        run = path.parents[1].name if path.parents[1] != CASE_DIR else "coupled_cylinder_flow"
        record = {"run": run, "end time": forces["time"][-1], "windows": len(rows)}
        for name in names:
            values = np.array([row[name] for row in rows])
            late = values[len(values) // 2 :]  # second half of the run: after the transient
            record[f"{name} study window"] = values[0]
            record[f"{name} mean"] = late.mean()
            record[f"{name} spread [%]"] = 100.0 * np.ptp(late) / late.mean()
            record[f"{name} sigma [%]"] = 100.0 * late.std(ddof=1) / late.mean()
            record[f"{name} drift [%]"] = 100.0 * (values[0] / late.mean() - 1.0)
        table.append(record)
        print(f"{run}: {len(rows)} windows to t = {forces['time'][-1]:.0f} s")
        for name in names:
            print(
                f"  {name:26s} study window {record[f'{name} study window']:.5f}"
                f"  developed mean {record[f'{name} mean']:.5f}"
                f"  spread {record[f'{name} spread [%]']:.2f} %"
                f"  sigma {record[f'{name} sigma [%]']:.2f} %"
                f"  drift of study window {record[f'{name} drift [%]']:+.2f} %"
            )

    if not table:
        print("no run reaches far enough beyond the study window; run section h first")
        return
    output = CASE_DIR / "figures/auxiliary/window_uncertainty.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f"wrote {output.relative_to(CASE_DIR)}")


if __name__ == "__main__":
    main()
