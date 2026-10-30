#!/usr/bin/env python3
"""Drag and lift on the cylinder: coupled FVM–VPM against the FVM reference."""

from pathlib import Path

from matplotlib.ticker import FuncFormatter
import numpy as np

from openonda.plotting import fit_thesis_y_label_margins
from openonda.plotting import COLORS, LINE_WIDTH, REFERENCE_STYLE
from openonda.plotting import save_validation_figure, validation_legend, validation_subplots

CASE_DIR = Path(__file__).resolve().parents[1]
DEVELOPED_SHEDDING = 40.0  # s, start of the statistics window
DIAMETER = 1.0  # m
FREESTREAM_SPEED = 1.0  # m/s


def read_forces(path):
    """Force-coefficient history, one row per sampled time."""
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")


def shedding_statistics(forces):
    """Mean drag, lift amplitude and Strouhal number over developed shedding."""
    window = forces["time"] >= DEVELOPED_SHEDDING
    time = forces["time"][window]
    lift = forces["lift_coefficient"][window]

    # Upward lift crossings define the shedding period.
    fluctuation = lift - lift.mean()
    up = np.flatnonzero((fluctuation[:-1] < 0.0) & (fluctuation[1:] >= 0.0))
    crossing = time[up] - fluctuation[up] * np.diff(time)[up] / np.diff(fluctuation)[up]
    period = np.diff(crossing).mean()

    return {
        "mean drag": forces["drag_coefficient"][window].mean(),
        "lift amplitude": 0.5 * np.ptp(lift),
        "Strouhal number": DIAMETER / (period * FREESTREAM_SPEED),
    }


def main():
    reference = read_forces(CASE_DIR / "reference_flow/samples/forces_history.csv")
    coupled = read_forces(CASE_DIR / "samples/forces_history.csv")

    # Compare developed-shedding statistics.
    expected, measured = shedding_statistics(reference), shedding_statistics(coupled)
    print(f"Developed shedding, t >= {DEVELOPED_SHEDDING:g} s")
    for name in expected:
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
