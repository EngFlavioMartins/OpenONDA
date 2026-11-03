#!/usr/bin/env python3
"""Drag and lift history of every grid of the FVM-only reference (reference_flow)."""

from pathlib import Path

import numpy as np

from openonda.plotting import LINE_WIDTH, save_validation_figure, validation_subplots

CASE_DIR = Path(__file__).resolve().parents[1]
REFERENCE_DIR = CASE_DIR / "reference_flow"


def main():
    figure, axes = validation_subplots(2, sharex=True)
    for path in sorted(REFERENCE_DIR.glob("samples/*/forces_history.csv")):
        forces = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        for axis, name in zip(axes, ("drag_coefficient", "lift_coefficient")):
            axis.plot(forces["time"], forces[name], linewidth=LINE_WIDTH, label=path.parent.name)
    axes[0].set_ylabel("$C_D$")
    axes[1].set_ylabel("$C_L$")
    axes[0].set_ylim(1.0, 1.6)  # developed drag range
    axes[0].legend()
    axes[1].set_xlabel("$t$ [s]")
    save_validation_figure(figure, CASE_DIR / "figures/reference_forces")


if __name__ == "__main__":
    main()
