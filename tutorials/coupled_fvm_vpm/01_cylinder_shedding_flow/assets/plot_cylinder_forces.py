#!/usr/bin/env python3
"""Plot coupled-cylinder drag and lift histories."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from openonda.plotting import (
    CM,
    COLORS,
    LINE_WIDTH,
    centered_subplots_adjust,
    fit_thesis_y_label_margins,
    set_thesis_style,
)

import postprocess as data

FORCE_COLUMNS = ("drag_coefficient", "lift_coefficient")


def time_mean(time: np.ndarray, values: np.ndarray) -> float:
    """Return the trapezoidal mean over the sampled time interval."""
    return float(np.trapezoid(values, time) / (time[-1] - time[0]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("png", "pdf"), default="png")
    arguments = parser.parse_args()

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

    set_thesis_style()
    figure, axes = plt.subplots(2, 1, figsize=(12.5 * CM, 8.3 * CM), sharex=True)
    axes[0].plot(time, drag, color=COLORS["FVMorange"], linewidth=LINE_WIDTH)
    axes[1].plot(time, lift, color=COLORS["TUDcyan"], linewidth=LINE_WIDTH)
    axes[0].set_ylabel(r"$C_D$")
    axes[1].set_ylabel(r"$C_L$")
    axes[1].set_xlabel(r"$tU_\infty/D$")
    for axis in axes:
        axis.grid(alpha=0.22)
    centered_subplots_adjust(figure, outer=0.17, bottom=0.17, top=0.96, hspace=0.12)
    fit_thesis_y_label_margins(figure, axes)
    data.save_figure(figure, axes, "cylinder_forces", arguments.format)


if __name__ == "__main__":
    main()
