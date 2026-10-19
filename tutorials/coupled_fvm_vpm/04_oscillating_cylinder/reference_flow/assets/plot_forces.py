#!/usr/bin/env python3
"""Drag and lift history of the reference heaving-cylinder flow."""

from pathlib import Path

import numpy as np

from openonda.plotting import COLORS, LINE_WIDTH, save_validation_figure, validation_subplots

CASE_DIR = Path(__file__).resolve().parents[1]


def main():
    path = CASE_DIR / "samples/forces_history.csv"
    forces = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")

    figure, axes = validation_subplots(2, sharex=True)
    for axis, name, label in zip(
        axes, ("drag_coefficient", "lift_coefficient"), ("$C_D$", "$C_L$")
    ):
        axis.plot(forces["time"], forces[name], color=COLORS["fvm"], linewidth=LINE_WIDTH)
        axis.set_ylabel(label)
    axes[1].set_xlabel("$t$ [s]")
    save_validation_figure(figure, CASE_DIR / "figures/forces")


if __name__ == "__main__":
    main()
