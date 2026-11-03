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

from assets.plot_forces import DEVELOPED_SHEDDING, read_forces, shedding_statistics
from assets.plot_sensitivity import reference_paths, run_paths

CASE_DIR = Path(__file__).resolve().parents[1]
WINDOW = 15.0  # s, the study window (40-55 s)
SLIDE = 2.5  # s, window offset between samples
MARGIN = 15.0  # s, a run must reach the window end plus this to be used


def sliding_windows(forces):
    """Statistics of every full study window from developed shedding to the end of the run."""
    end_time = forces["time"][-1]
    starts = np.arange(DEVELOPED_SHEDDING, end_time - WINDOW + 1.0e-9, SLIDE)
    rows = []
    for start in starts:
        statistics = shedding_statistics(forces, start, start + WINDOW)
        if statistics is not None:
            rows.append({"start": start, **statistics})
    return rows


def main():
    paths = [*reference_paths(), *run_paths()]
    names = (
        "mean drag",
        "mean drag, whole window",
        "lift amplitude",
        "lift amplitude per period",
        "Strouhal number",
    )
    table = []
    for path in paths:
        forces = np.atleast_1d(read_forces(path))
        if forces["time"][-1] < DEVELOPED_SHEDDING + WINDOW + MARGIN:
            continue  # too short to slide the window
        rows = sliding_windows(forces)
        if len(rows) < 3:
            continue
        # reference_flow/samples/<name>/ or solution/sensitivity_analysis/<name>/samples/
        run = path.parents[1].name if path.parent.name == "samples" else path.parent.name
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
