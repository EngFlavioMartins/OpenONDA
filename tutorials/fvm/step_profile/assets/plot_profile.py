#!/usr/bin/env python3
"""Plot streamwise-velocity profiles downstream of the step."""

from pathlib import Path


import matplotlib.pyplot as plt
from openonda.plotting import centered_subplots_adjust
import numpy as np

from ._common import (  # noqa: E402
    COLORS,
    FIGURES_DIR,
    SOLUTION_DIR,
    build_arg_parser,
    figure_size,
    read_csv_columns,
    save_fig,
)


def main():
    args = build_arg_parser().parse_args()
    data = read_csv_columns(Path(SOLUTION_DIR) / "fields.csv")

    x, y, u = (
        data["position_x_over_height"],
        data["position_y_over_height"],
        data["velocity_x"],
    )
    x_columns = np.unique(np.round(x, 10))
    stations = (1.0, 3.0, 6.0, 10.0)
    colors = ("dark", "hybrid", "teal", "vpm")

    fig, ax = plt.subplots(figsize=figure_size("single"))
    for station, color_name in zip(stations, colors, strict=True):
        x_sample = x_columns[np.argmin(np.abs(x_columns - station))]
        selected = np.isclose(x, x_sample, atol=1e-9)
        order = np.argsort(y[selected])
        ax.plot(
            u[selected][order],
            y[selected][order],
            color=COLORS[color_name],
            label=rf"$x/h={x_sample:.1f}$",
        )

    ax.axvline(0.0, color=COLORS["reference"], linewidth=0.8, linestyle="--")
    ax.set_xlabel(r"$u/U_b$")
    ax.set_ylabel(r"$y/h$")
    ax.set_ylim(0.0, 2.0)
    ax.legend()
    ax.grid(False)
    centered_subplots_adjust(fig, outer=0.100, bottom=0.20, top=0.968)
    save_fig(fig, "step_evolution.png", FIGURES_DIR, dpi=args.dpi, figure_format=args.format)


if __name__ == "__main__":
    main()
