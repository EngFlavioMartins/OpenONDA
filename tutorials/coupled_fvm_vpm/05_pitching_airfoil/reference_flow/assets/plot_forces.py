#!/usr/bin/env python3
"""Drag, lift and pitching-moment history of the reference pitching-airfoil flow."""

from pathlib import Path

import numpy as np

from openonda.plotting import COLORS, LINE_WIDTH, fit_thesis_y_label_margins
from openonda.plotting import save_validation_figure, validation_subplots

CASE_DIR = Path(__file__).resolve().parents[1]


def main():
    forces = np.genfromtxt(
        CASE_DIR / "samples/forces_history.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )

    figure, axes = validation_subplots(3, sharex=True)
    for axis, name, label in zip(
        axes,
        ("drag_coefficient", "lift_coefficient", "pitching_moment_coefficient"),
        ("$C_D$", "$C_L$", "$C_M$"),
    ):
        axis.plot(forces["time"], forces[name], color=COLORS["fvm"], linewidth=LINE_WIDTH)
        axis.set_ylabel(label)
    axes[-1].set_xlabel("$t$ [s]")
    fit_thesis_y_label_margins(figure, axes)
    save_validation_figure(figure, CASE_DIR / "figures/forces")


if __name__ == "__main__":
    main()
