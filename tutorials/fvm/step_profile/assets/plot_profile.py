#!/usr/bin/env python3
"""Plot streamwise-velocity profiles downstream of the step."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _common import (  # noqa: E402
    COLORS,
    FIGURES_DIR,
    SOLUTION_DIR,
    build_arg_parser,
    figure_size,
    load_csv_columns,
    save_fig,
)


def main():
    args = build_arg_parser().parse_args()
    data = load_csv_columns(Path(SOLUTION_DIR) / "fields.csv")
    if not data:
        return
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)

    x, y, u = (
        data["position_x_over_height"],
        data["position_y_over_height"],
        data["velocity_x"],
    )
    x_columns = np.unique(np.round(x, 10))
    stations = (1.0, 3.0, 6.0, 10.0)
    colors = ("TUDdark", "FVMorange", "TUDcyan", "VPMpurple")

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
    ax.set_title("Backward-facing step: downstream velocity profiles")
    ax.set_ylim(0.0, 2.0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, "step_evolution.png", FIGURES_DIR, dpi=args.dpi, figure_format=args.format)


if __name__ == "__main__":
    main()
