#!/usr/bin/env python3
"""Plot this reference case's native drag and lift history."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from openonda.plotting import (
    COLORS,
    FORMAT_CHOICES,
    LINE_WIDTH,
    centered_subplots_adjust,
    export_figure,
    figure_size,
    fit_thesis_y_label_margins,
    prepare_figure,
    set_thesis_style,
    validate_thesis_figure,
)

CASE_DIR = Path(__file__).resolve().parents[1]
COLUMNS = ("time", "drag_coefficient", "lift_coefficient")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=FORMAT_CHOICES, default="both")
    arguments = parser.parse_args()

    path = CASE_DIR / "samples" / "forces_history.csv"
    history = pd.read_csv(path)
    missing = set(COLUMNS) - set(history.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    values = history[list(COLUMNS)].to_numpy(dtype=float)
    if len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError(f"{path} must contain at least two finite samples")
    time = values[:, 0]
    if np.any(np.diff(time) <= 0):
        raise ValueError(f"{path} times must be strictly increasing")

    plt.switch_backend("Agg")
    set_thesis_style()
    figure, axes = plt.subplots(2, 1, figsize=figure_size("wide_short"), sharex=True)
    for index, (axis, label) in enumerate(zip(axes, (r"$C_D$", r"$C_L$"), strict=True)):
        axis.plot(time, values[:, index + 1], color=COLORS["fvm"], linewidth=LINE_WIDTH)
        axis.set_ylabel(label)
        axis.grid(False)
    axes[-1].set_xlabel(r"$t$ [s]")
    centered_subplots_adjust(figure, outer=0.16, bottom=0.18, top=0.90, hspace=0.18)
    prepare_figure(figure)
    fit_thesis_y_label_margins(figure, axes)
    validate_thesis_figure(figure, axes)
    export_figure(figure, CASE_DIR / "figures" / "forces", figure_format=arguments.format)


if __name__ == "__main__":
    main()
