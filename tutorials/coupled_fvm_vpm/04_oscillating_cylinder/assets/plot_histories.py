#!/usr/bin/env python3
"""Cylinder displacement, force coefficients and trajectory of every run."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from openonda.plotting import CM, COLORS, LINE_WIDTH, MAX_FIGURE_WIDTH_CM, REFERENCE_STYLE
from openonda.plotting import export_figure, prepare_figure, set_thesis_style, validation_legend

from assets.response import CASE_DIR, DIAMETER, completed_runs, read_run, variant

FIGURES = CASE_DIR / "figures"
COUPLED = CASE_DIR / "samples"
REFERENCE = CASE_DIR / "reference_flow/samples"


def runs(run):
    """Coupled and FVM-only histories of one (Re, V*) that exist."""
    found = {}
    for label, root in (("Coupled FVM and VPM", COUPLED), ("FVM only", REFERENCE)):
        if run in completed_runs(root):
            found[label] = read_run(root / variant(*run), run[1])
    return found


def main():
    set_thesis_style()
    for run in sorted({*completed_runs(COUPLED), *completed_runs(REFERENCE)}):
        found = runs(run)
        if not found:
            continue
        height = 13.0
        figure = plt.figure(figsize=(MAX_FIGURE_WIDTH_CM * CM, height * CM))
        grid = figure.add_gridspec(
            3,
            2,
            width_ratios=(2.2, 1.0),
            left=0.12,
            right=0.88,
            bottom=0.09,
            top=0.93,
            hspace=0.12,
            wspace=0.3,
        )
        histories = [figure.add_subplot(grid[row, 0]) for row in range(3)]
        trajectory = figure.add_subplot(grid[:, 1])
        for label, (motion, forces, _) in found.items():
            style = (
                {"color": COLORS["hybrid"], "linewidth": LINE_WIDTH}
                if label.startswith("Coupled")
                else REFERENCE_STYLE
            )
            histories[0].plot(
                motion["time"], motion["displacement_y"] / DIAMETER, label=label, **style
            )
            histories[1].plot(motion["time"], motion["displacement_x"] / DIAMETER, **style)
            histories[2].plot(forces["time"], forces["lift_coefficient"], **style)
            trajectory.plot(
                motion["displacement_x"] / DIAMETER, motion["displacement_y"] / DIAMETER, **style
            )
        histories[0].set_ylabel("$y/D$")
        histories[1].set_ylabel("$x/D$")
        histories[2].set_ylabel("$C_L$")
        histories[2].set_xlabel("$t$ [s]")
        for axis in histories[:2]:
            axis.tick_params(labelbottom=False)
        for axis in histories:
            axis.yaxis.set_major_locator(MaxNLocator(4))
        # The square trajectory panel is narrow: few ticks keep the labels apart
        trajectory.xaxis.set_major_locator(MaxNLocator(3))
        trajectory.yaxis.set_major_locator(MaxNLocator(5))
        trajectory.set(xlabel="$x/D$", ylabel="$y/D$")
        trajectory.yaxis.set_label_position("right")
        trajectory.yaxis.tick_right()
        validation_legend(figure, histories[0], ncol=1)
        prepare_figure(figure)
        export_figure(figure, FIGURES / f"histories_{variant(*run)}", figure_format="png")


if __name__ == "__main__":
    main()
