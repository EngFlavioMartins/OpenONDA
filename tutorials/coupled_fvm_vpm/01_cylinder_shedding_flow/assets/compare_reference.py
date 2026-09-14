#!/usr/bin/env python3
"""Compare coupled forces and instantaneous velocity profiles with one FVM grid.

Use --reference medium (default), fine, or another reference samples directory
name. Comparison uses common physical times, with no phase shift, rescaling of
forces, or extrapolation beyond available samples. This measures differences;
the physical geometry and numerical configuration must also be matched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CASE_DIR = Path(__file__).resolve().parents[1]
VELOCITY = [f"velocity_{axis}" for axis in "xyz"]


def common_history(candidate, reference, columns):
    for name, frame in (("coupled", candidate), ("reference", reference)):
        values = frame[["time", *columns]].to_numpy(dtype=float)
        if len(values) < 2 or not np.all(np.isfinite(values)) or np.any(np.diff(values[:, 0]) <= 0):
            raise ValueError(f"{name} history must have finite, strictly increasing times")
    start = max(candidate.time.iloc[0], reference.time.iloc[0])
    end = min(candidate.time.iloc[-1], reference.time.iloc[-1])
    if end <= start:
        raise ValueError("Force histories have no common time interval")
    times = np.unique(np.r_[start, end, candidate.time, reference.time])
    times = times[(times >= start) & (times <= end)]
    left = np.column_stack([np.interp(times, candidate.time, candidate[key]) for key in columns])
    right = np.column_stack([np.interp(times, reference.time, reference[key]) for key in columns])
    errors = {}
    for j, key in enumerate(columns):
        delta = left[:, j] - right[:, j]
        errors[key] = {
            "rms": float(np.sqrt(np.trapezoid(delta**2, times) / (end - start))),
            "maximum": float(np.abs(delta).max()),
        }
    return times, left, right, errors


def compare(case_dir, reference_name, *, if_available=False, figure_format="png"):
    samples = case_dir / "samples"
    reference = case_dir / "reference_flow/samples" / reference_name
    force_file = reference / "forces_history.csv"
    if if_available and not force_file.is_file():
        print(f"Reference comparison skipped: {force_file} is absent")
        return
    candidate = pd.read_csv(samples / "forces_history.csv")
    reference_force = pd.read_csv(force_file)
    keys = ["drag_coefficient", "lift_coefficient"]
    times, left, right, errors = common_history(candidate, reference_force, keys)
    report = {
        "schema": "openonda-cylinder-comparison/1",
        "reference": reference_name,
        "force_time_interval": [float(times[0]), float(times[-1])],
        "force_errors": errors,
        "profile_errors": {},
        "missing_profiles": [],
    }
    figures = case_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True, constrained_layout=True)
    for i, axis in enumerate(axes):
        axis.plot(times, right[:, i], "k-", label=f"FVM reference ({reference_name})")
        axis.plot(times, left[:, i], color="#1769aa", label="Coupled", linewidth=1.1)
        axis.set_ylabel(("$C_D$", "$C_L$")[i])
        axis.grid(alpha=0.2)
    axes[0].legend()
    axes[-1].set_xlabel("t U∞/D")
    fig.savefig(figures / f"cylinder_reference_forces.{figure_format}", dpi=180)
    plt.close(fig)

    profiles = []
    for x in (1, 2, 4):
        ref_path = reference / f"transverse_x{x}.csv"
        for label in ("fvm", "vpm"):
            path = samples / f"{label}_transverse_x{x}.csv"
            if path.is_file() and ref_path.is_file():
                profiles.append((x, label, pd.read_csv(path), pd.read_csv(ref_path)))
            else:
                report["missing_profiles"].append(f"{label}_transverse_x{x}")
    # A single snapshot for every plotted curve: unequal time levels must
    # never masquerade as a spatial-profile difference.
    common_times = None
    for _, _, a, b in profiles:
        for frame in (a, b):
            values = frame[["time", "position_x", "position_y", "position_z", *VELOCITY]].to_numpy(
                dtype=float
            )
            if not np.all(np.isfinite(values)):
                raise ValueError("Velocity profiles contain non-finite samples")
            available = set(np.round(frame.time, 7))
            common_times = available if common_times is None else common_times & available
    if profiles and not common_times:
        raise ValueError("Velocity profiles have no common physical sample time")
    if profiles:
        snapshot_time = max(common_times)
        report["profile_time"] = float(snapshot_time)
        fig, axes = plt.subplots(2, 3, figsize=(11, 6), constrained_layout=True)
        shown_reference = set()
        for x, label, a, b in profiles:
            a = a[np.round(a.time, 7) == snapshot_time].sort_values("position_y")
            b = b[np.round(b.time, 7) == snapshot_time].sort_values("position_y")
            for frame in (a, b):
                if not np.allclose(frame.position_x, x) or not np.allclose(frame.position_z, 0):
                    raise ValueError("Expected a transverse line at the declared x and z=0")
                if np.any(np.diff(frame.position_y) <= 0):
                    raise ValueError("Profile y coordinates must be strictly increasing")
            ymin, ymax = (
                max(a.position_y.min(), b.position_y.min()),
                min(a.position_y.max(), b.position_y.max()),
            )
            a = a[(a.position_y >= ymin) & (a.position_y <= ymax)]
            if len(a) < 2:
                raise ValueError("Velocity profiles have insufficient overlapping spatial samples")
            y = a.position_y.to_numpy()
            expected = np.column_stack([np.interp(y, b.position_y, b[key]) for key in VELOCITY])
            actual = a[VELOCITY].to_numpy()
            delta = actual - expected
            report["profile_errors"][f"{label}_x{x}"] = {
                "y_interval": [float(y[0]), float(y[-1])],
                "velocity_rms_over_Uinf": float(
                    np.sqrt(np.trapezoid(np.sum(delta**2, axis=1), y) / (y[-1] - y[0]))
                ),
                "velocity_max_over_Uinf": float(np.linalg.norm(delta, axis=1).max()),
            }
            column = (1, 2, 4).index(x)
            for row in (0, 1):
                axis = axes[row, column]
                if x not in shown_reference:
                    axis.plot(
                        b.position_y, b[VELOCITY[row]], "k-", label=f"Reference ({reference_name})"
                    )
                axis.plot(y, actual[:, row], label=label.upper())
                axis.set(xlabel="y/D", ylabel=("u/U∞", "v/U∞")[row])
                axis.grid(alpha=0.2)
                if row == 0:
                    axis.set_title(f"x/D = {x}")
            shown_reference.add(x)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"Velocity profiles at common t U∞/D = {snapshot_time:g}")
        fig.savefig(figures / f"cylinder_reference_profiles.{figure_format}", dpi=180)
        plt.close(fig)
    destination = case_dir / "solution/cylinder_reference_comparison.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="medium")
    parser.add_argument("--case-dir", type=Path, default=CASE_DIR)
    parser.add_argument("--if-available", action="store_true")
    parser.add_argument("--format", choices=("png", "pdf"), default="png")
    args = parser.parse_args()
    compare(
        args.case_dir, args.reference, if_available=args.if_available, figure_format=args.format
    )
