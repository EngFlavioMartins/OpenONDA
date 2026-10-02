#!/usr/bin/env python3
"""Plot coupled-cylinder drag and lift histories."""

if not __package__:
    from pathlib import Path as _CasePath
    from openonda.tutorial_runner import case_package

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid

from openonda.plotting import (
    CM,
    COLORS,
    LINE_WIDTH,
    centered_subplots_adjust,
    set_thesis_style,
)

from . import postprocess as data

FORCE_COLUMNS = ("drag_coefficient", "lift_coefficient")


def time_mean(time: np.ndarray, values: np.ndarray) -> float:
    """Return the trapezoidal mean over the sampled time interval."""
    if len(time) < 2 or time[-1] <= time[0]:
        raise ValueError("Force statistics require at least two distinct saved times")
    return float(trapezoid(values, time) / (time[-1] - time[0]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("png", "pdf", "both"), default="both")
    parser.add_argument("--case-dir", type=Path, default=data.CASE_DIR)
    arguments = parser.parse_args()

    data.CASE_DIR = arguments.case_dir.resolve()
    data.FIGURES = data.CASE_DIR / "figures"
    data.AUXILIARY = data.FIGURES / "auxiliary"

    frame = data.history(data.CASE_DIR / "samples" / "forces_history.csv", FORCE_COLUMNS)
    time = frame.time.to_numpy(dtype=float)
    drag = frame.drag_coefficient.to_numpy(dtype=float)
    lift = frame.lift_coefficient.to_numpy(dtype=float)
    start = max(time[0], time[-1] - 30.0)
    statistics = time >= start - 1.0e-12
    mean_drag = time_mean(time[statistics], drag[statistics])
    mean_lift = time_mean(time[statistics], lift[statistics])
    data.write_json(
        "cylinder_force_statistics.json",
        {
            "time_interval": [float(time[statistics][0]), float(time[-1])],
            "sample_count": int(np.count_nonzero(statistics)),
            "scope": "last available 30 s or shorter; periodicity is not established by these statistics",
            "mean_drag_coefficient": mean_drag,
            "rms_drag_coefficient": float(
                np.sqrt(time_mean(time[statistics], (drag[statistics] - mean_drag) ** 2))
            ),
            "mean_lift_coefficient": mean_lift,
            "rms_lift_coefficient": float(
                np.sqrt(time_mean(time[statistics], (lift[statistics] - mean_lift) ** 2))
            ),
        },
    )
    print(f"Force history: {time[0]:g}–{time[-1]:g} s; statistics use available samples only.")

    set_thesis_style()
    figure, axes = plt.subplots(2, 1, figsize=(12.5 * CM, 8.3 * CM), sharex=True)
    axes[0].plot(time, drag, color=COLORS["hybrid"], linewidth=LINE_WIDTH)
    axes[1].plot(time, lift, color=COLORS["teal"], linewidth=LINE_WIDTH)
    axes[0].set_ylabel(r"$C_D$")
    axes[1].set_ylabel(r"$C_L$")
    axes[1].set_xlabel(r"$tU_\infty/D$")
    for axis in axes:
        axis.grid(False)
    centered_subplots_adjust(figure, outer=0.135, bottom=0.17, top=0.93, hspace=0.12)
    data.save_figure(figure, axes, "cylinder_forces", arguments.format)


if __name__ == "__main__":
    main()
