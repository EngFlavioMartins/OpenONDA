#!/usr/bin/env python3
"""Compare completed span/time controls against the identical XY baseline."""

import argparse
import json
from pathlib import Path

import numpy as np

from postprocess_grid_study import _difference, _force_statistics, _read_table


def profile(path: Path, start: float, end: float) -> tuple[np.ndarray, np.ndarray]:
    """Return coordinates [m] and mean streamwise velocity [m/s] at each probe.

    The mean weights saved frames equally; campaign profiles have a common
    fixed 0.1 s sampling interval. Only the final restart segment is used.
    """
    table = np.genfromtxt(path, delimiter=",", names=True)
    reset = np.flatnonzero(np.diff(table["time"]) < -1.0e-10)
    if len(reset):
        table = table[reset[-1] + 1 :]
    table = table[(table["time"] >= start) & (table["time"] <= end)]
    positions = np.unique(
        np.column_stack([table["position_x"], table["position_y"], table["position_z"]]), axis=0
    )
    velocity = []
    for point in positions:
        selected = np.ones(len(table), dtype=bool)
        for index, name in enumerate(("position_x", "position_y", "position_z")):
            selected &= np.isclose(table[name], point[index])
        velocity.append(float(np.mean(table["velocity_x"][selected])))
    return positions, np.asarray(velocity)


def analyse(root: Path, start: float, end: float) -> dict:
    """Compare independent span/time controls with the matched XY baseline.

    Parameters
    ----------
    root : pathlib.Path
        Campaign directory with spatial and controls sample subdirectories.
    start, end : float
        Common physical averaging limits in s.

    Returns
    -------
    dict
        Force and profile changes, fixed engineering thresholds and their
        assessment. Passing these controls alone does not prove convergence.
    """
    baseline = root / "spatial/samples/xy_fine"
    reference = _force_statistics(_read_table(baseline / "forces_history.csv"), start, end)
    thresholds = {"mean_drag": 0.01, "rms_drag": 0.02, "rms_lift": 0.02, "strouhal_lift": 0.02}
    comparisons = {}
    for name in ("z_eight", "span_two", "dt_half"):
        directory = root / "controls/samples" / name
        table = _read_table(directory / "forces_history.csv")
        statistics = _force_statistics(table, start, end)
        metrics = {key: _difference(statistics[key], reference[key]) for key in thresholds}
        profiles = {}
        for filename in ("centreline", "transverse_x1", "transverse_x2", "transverse_x4"):
            x, u = profile(baseline / (filename + ".csv"), start, end)
            y, v = profile(directory / (filename + ".csv"), start, end)
            if x.shape != y.shape or not np.allclose(x, y):
                raise ValueError("Control profile sampling lattices differ")
            profiles[filename] = float(np.linalg.norm(v - u) / max(np.linalg.norm(u), 1.0e-14))
        comparisons[name] = {
            "statistics": statistics,
            "relative_changes": metrics,
            "profile_relative_l2": profiles,
            "within_engineering_targets": all(
                value is not None and value <= thresholds[key] for key, value in metrics.items()
            )
            and all(value <= 0.02 for value in profiles.values()),
        }
    return {
        "statistics_window": [start, end],
        "baseline": "xy_fine",
        "baseline_statistics": reference,
        "thresholds": thresholds,
        "comparisons": comparisons,
        "qualification": "Passing engineering targets does not alone establish grid convergence; control errors must also be smaller than the spatial errors being estimated.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--start", type=float, default=50.0)
    parser.add_argument("--end", type=float, default=100.0)
    args = parser.parse_args()
    result = analyse(args.root, args.start, args.end)
    output = args.root / "figures/auxiliary/control_comparisons.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output)}))
