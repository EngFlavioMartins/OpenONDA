#!/usr/bin/env python3
"""Cylinder displacement and force coefficients of the FVM-only runs of each reduced velocity."""

from pathlib import Path

import numpy as np

from openonda.plotting import COLORS, LINE_WIDTH, save_validation_figure, validation_subplots

CASE_DIR = Path(__file__).resolve().parents[1]
DIAMETER = 1.0  # m


def read_table(path):
    """Named columns of a sampler CSV."""
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")


def main():
    for motion_file in sorted((CASE_DIR / "samples").glob("vstar_*/motion_history.csv")):
        variant = motion_file.parent.name
        motion = read_table(motion_file)
        forces = read_table(motion_file.parent / "forces_history.csv")
        figure, axes = validation_subplots(4, sharex=True)
        for axis, time, values, label in (
            (axes[0], motion["time"], motion["displacement_y"] / DIAMETER, "$y/D$"),
            (axes[1], motion["time"], motion["displacement_x"] / DIAMETER, "$x/D$"),
            (axes[2], forces["time"], forces["lift_coefficient"], "$C_L$"),
            (axes[3], forces["time"], forces["drag_coefficient"], "$C_D$"),
        ):
            axis.plot(time, values, color=COLORS["fvm"], linewidth=LINE_WIDTH)
            axis.set_ylabel(label)
        axes[3].set_xlabel("$t$ [s]")
        save_validation_figure(figure, CASE_DIR / f"figures/histories_{variant}")


if __name__ == "__main__":
    main()
