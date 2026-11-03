#!/usr/bin/env python3
"""Drag and lift on the cylinder: coupled FVM–VPM against the FVM reference."""

from pathlib import Path

from matplotlib.ticker import FuncFormatter
import numpy as np

from openonda.plotting import fit_thesis_y_label_margins
from openonda.plotting import COLORS, LINE_WIDTH, REFERENCE_STYLE
from openonda.plotting import save_validation_figure, validation_legend, validation_subplots

CASE_DIR = Path(__file__).resolve().parents[1]
# The baseline coupled run and its matched FVM-only reference (allrun.sh, reference_flow/allrun.sh)
BASELINE_DIR = CASE_DIR / "solution/sensitivity_analysis/box2.4_sub5_inset8.75"
REFERENCE_SAMPLES = CASE_DIR / "reference_flow/samples/h0.04_dt0.008"
REFERENCE_SOLUTION = CASE_DIR / "reference_flow/solution/h0.04_dt0.008"
DEVELOPED_SHEDDING = 40.0  # s, start of the statistics window
DIAMETER = 1.0  # m
FREESTREAM_SPEED = 1.0  # m/s


def read_forces(path):
    """Force-coefficient history, one row per sampled time."""
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")


def parabolic_extremum(time, values, index):
    """Value at the vertex of the parabola through the samples index - 1, index and index + 1.

    A sampled peak lies below the true one by up to 1 - cos(pi dt / T) of the amplitude, which
    grows with the sample interval dt; the vertex removes this bias for any sample interval.
    """
    if index == 0 or index == len(values) - 1:
        return values[index]
    t, v = time[index - 1 : index + 2], values[index - 1 : index + 2]
    a, b, c = np.polyfit(t - t[1], v, 2)
    return c - b**2 / (4.0 * a) if a != 0.0 else v[1]


def shedding_statistics(forces, start=DEVELOPED_SHEDDING, end=np.inf):
    """Mean drag, lift amplitude and Strouhal number over whole shedding periods in [start, end].

    Upward crossings of the lift fluctuation bound the periods. The mean drag is taken between the
    first and the last crossing, so that the drag oscillation averages over whole periods (over a
    window of 2.7 periods its plain mean scatters by 0.07 %, over whole periods by 0.007 %). The
    lift amplitude is half the peak-to-peak lift in the window, its extremes taken at the vertex of
    a parabola through the three samples around each, so that runs sampled at different intervals
    compare alike; its per-period mean, the plain window mean of the drag and the number of
    periods are returned as checks.
    """
    window = (forces["time"] >= start) & (forces["time"] <= end)
    time = forces["time"][window]
    drag = forces["drag_coefficient"][window]
    lift = forces["lift_coefficient"][window]

    # Upward lift crossings define the shedding period.
    fluctuation = lift - lift.mean()
    up = np.flatnonzero((fluctuation[:-1] < 0.0) & (fluctuation[1:] >= 0.0))
    crossing = time[up] - fluctuation[up] * np.diff(time)[up] / np.diff(fluctuation)[up]
    if crossing.size < 2:
        return None

    # Linear interpolation between samples, from the first to the last crossing
    grid = np.linspace(crossing[0], crossing[-1], 20 * len(time) + 1)
    drag_grid, lift_grid = np.interp(grid, time, drag), np.interp(grid, time, lift)
    per_period = [
        0.5 * np.ptp(lift_grid[(grid >= a) & (grid <= b)])
        for a, b in zip(crossing[:-1], crossing[1:])
    ]
    return {
        "mean drag": np.trapezoid(drag_grid, grid) / (grid[-1] - grid[0]),
        "lift amplitude": 0.5
        * (
            parabolic_extremum(time, lift, int(np.argmax(lift)))
            - parabolic_extremum(time, lift, int(np.argmin(lift)))
        ),
        "Strouhal number": DIAMETER * (crossing.size - 1) / ((crossing[-1] - crossing[0]) * FREESTREAM_SPEED),
        "lift amplitude per period": float(np.mean(per_period)),
        "mean drag, whole window": drag.mean(),
        "periods": crossing.size - 1,
    }


def main():
    reference = read_forces(REFERENCE_SAMPLES / "forces_history.csv")
    coupled = read_forces(BASELINE_DIR / "samples/forces_history.csv")

    # Compare developed-shedding statistics.
    expected, measured = shedding_statistics(reference), shedding_statistics(coupled)
    print(f"Developed shedding, t >= {DEVELOPED_SHEDDING:g} s")
    for name in ("mean drag", "lift amplitude", "Strouhal number"):
        deviation = 100.0 * (measured[name] / expected[name] - 1.0)
        print(
            f"  {name:16s} reference {expected[name]:.4f}  "
            f"coupled {measured[name]:.4f}  ({deviation:+.2f}%)"
        )

    figure, axes = validation_subplots(2, sharex=True)
    for axis, name, label in zip(
        axes, ("drag_coefficient", "lift_coefficient"), ("$C_D$", "$C_L$")
    ):
        axis.plot(reference["time"], reference[name], **REFERENCE_STYLE, label="FVM only")
        axis.plot(
            coupled["time"],
            coupled[name],
            color=COLORS["hybrid"],
            linewidth=LINE_WIDTH,
            label="Coupled",
        )
        axis.set_ylabel(label)
    axes[0].set_ylim(1.0, 1.6)  # developed drag range
    axes[1].set_xlabel("$t$ [s]")
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    validation_legend(figure, axes[0])
    fit_thesis_y_label_margins(figure, figure.axes)
    save_validation_figure(figure, CASE_DIR / "figures/forces")


if __name__ == "__main__":
    main()
