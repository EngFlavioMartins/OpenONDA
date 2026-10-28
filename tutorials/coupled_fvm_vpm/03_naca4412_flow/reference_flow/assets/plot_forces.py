#!/usr/bin/env python3
"""Lift and drag on the wing of the FVM-only reference."""

from pathlib import Path

import numpy as np

from openonda.plotting import REFERENCE_STYLE
from openonda.plotting import save_validation_figure, validation_subplots

CASE_DIR = Path(__file__).resolve().parents[1]
STATISTICS_START = 12.0  # s, start of the statistics window


def main():
    forces = np.genfromtxt(
        CASE_DIR / "samples/forces_history.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    # The freestream runs along x
    time = forces["time"]
    coefficients = {"$C_D$": forces["drag_coefficient"], "$C_L$": forces["lift_coefficient"]}
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
