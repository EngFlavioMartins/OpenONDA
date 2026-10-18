#!/usr/bin/env python3
"""Lift and drag on the wing of the FVM reference, in wind axes."""

import math
from pathlib import Path

import numpy as np

from openonda.plotting import REFERENCE_STYLE
from openonda.plotting import save_validation_figure, validation_subplots

CASE_DIR = Path(__file__).resolve().parents[1]
STATISTICS_START = 12.0  # s, start of the statistics window
ALPHA = math.radians(10.0)  # incidence of the freestream to the chord


def main():
    forces = np.genfromtxt(
        CASE_DIR / "samples/forces_history.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    time, body_x, body_y = forces["time"], forces["drag_coefficient"], forces["lift_coefficient"]

    # Body-axis x and y forces turned into the freestream direction
    coefficients = {
        "$C_D$": body_x * math.cos(ALPHA) + body_y * math.sin(ALPHA),
        "$C_L$": -body_x * math.sin(ALPHA) + body_y * math.cos(ALPHA),
    }
    window = time >= STATISTICS_START
    figure, axes = validation_subplots(2, sharex=True)
    for axis, (name, values) in zip(axes, coefficients.items()):
        print(
            f"{name[1:-1]}: mean {values[window].mean():.4f}, rms {values[window].std():.4f} over t >= {STATISTICS_START:g} s"
        )
        axis.plot(time, values, **REFERENCE_STYLE)
        axis.set_ylabel(name)
    axes[-1].set_xlabel("$t$ [s]")
    save_validation_figure(figure, CASE_DIR / "figures/forces")


if __name__ == "__main__":
    main()
