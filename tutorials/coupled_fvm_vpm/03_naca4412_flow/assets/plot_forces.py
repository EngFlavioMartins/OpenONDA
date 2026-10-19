#!/usr/bin/env python3
"""Lift and drag on the wing in wind axes: coupled FVM–VPM against the FVM reference."""

import math
from pathlib import Path

import numpy as np

from openonda.plotting import COLORS, LINE_WIDTH, REFERENCE_STYLE
from openonda.plotting import save_validation_figure, validation_legend, validation_subplots

CASE_DIR = Path(__file__).resolve().parents[1]
STATISTICS_START = 12.0  # s, start of the statistics window
ALPHA = math.radians(10.0)  # incidence of the freestream to the chord


def wind_axis_coefficients(path):
    """Time, drag and lift coefficients: body-axis x and y forces turned into the freestream."""
    forces = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    body_x, body_y = forces["drag_coefficient"], forces["lift_coefficient"]
    drag = body_x * math.cos(ALPHA) + body_y * math.sin(ALPHA)
    lift = -body_x * math.sin(ALPHA) + body_y * math.cos(ALPHA)
    return forces["time"], {"$C_D$": drag, "$C_L$": lift}


def window_statistics(time, coefficients):
    """Mean and rms fluctuation of each coefficient over the statistics window."""
    window = time >= STATISTICS_START
    return {
        name: (values[window].mean(), values[window].std()) for name, values in coefficients.items()
    }


def main():
    reference_time, reference = wind_axis_coefficients(
        CASE_DIR / "reference_flow/samples/forces_history.csv"
    )
    coupled_time, coupled = wind_axis_coefficients(CASE_DIR / "samples/forces_history.csv")

    # Statistics of the developed flow and the coupled deviation from the reference
    expected = window_statistics(reference_time, reference)
    measured = window_statistics(coupled_time, coupled)
    print(f"Wind-axis coefficients on the planform area, t >= {STATISTICS_START:g} s")
    for name in expected:
        (reference_mean, reference_rms), (coupled_mean, coupled_rms) = (
            expected[name],
            measured[name],
        )
        deviation = 100.0 * (coupled_mean / reference_mean - 1.0)
        print(
            f"  {name[1:-1]:5s} mean: reference {reference_mean:.4f}  coupled {coupled_mean:.4f}  "
            f"({deviation:+.2f}%);  rms: reference {reference_rms:.4f}  coupled {coupled_rms:.4f}"
        )

    figure, axes = validation_subplots(2, sharex=True)
    for axis, name in zip(axes, reference):
        axis.plot(reference_time, reference[name], **REFERENCE_STYLE, label="Reference FVM")
        axis.plot(
            coupled_time,
            coupled[name],
            color=COLORS["hybrid"],
            linewidth=LINE_WIDTH,
            label="Coupled",
        )
        axis.axvspan(
            STATISTICS_START, coupled_time[-1], color=COLORS["background_light"], alpha=0.3, lw=0
        )
        # Hide the impulsive-start peak
        steady = reference[name][reference_time >= 0.5 * STATISTICS_START]
        axis.set_ylim(0.5 * steady.min(), 1.5 * steady.max())
        axis.set_ylabel(name)
    axes[-1].set_xlabel("$t$ [s]")
    validation_legend(figure, axes[0])
    save_validation_figure(figure, CASE_DIR / "figures/forces")


if __name__ == "__main__":
    main()
