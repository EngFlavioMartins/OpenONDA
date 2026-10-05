#!/usr/bin/env python3
"""Plot the surface pressure distribution saved by setup.py."""

from pathlib import Path


import matplotlib.pyplot as plt
from openonda.plotting import centered_subplots_adjust

from ._common import (  # noqa: E402
    COLORS,
    FIGURES_DIR,
    RE,
    SOLUTION_DIR,
    build_arg_parser,
    figure_size,
    read_csv_columns,
    save_fig,
)


def main():
    args = build_arg_parser().parse_args()
    data = read_csv_columns(Path(SOLUTION_DIR) / "surface_cp.csv")
    x = data["position_x_over_chord"]
    y = data["position_y_over_chord"]
    cp = data["pressure_coefficient"]
    upper = y >= 0
    lower = ~upper

    fig, ax = plt.subplots(figsize=figure_size("single"))
    ax.plot(
        x[upper],
        -cp[upper],
        "o",
        color=COLORS["fvm"],
        markersize=3,
        linestyle="none",
        label="Upper surface",
    )
    ax.plot(
        x[lower],
        -cp[lower],
        "s",
        color=COLORS["fvm"],
        markersize=3,
        linestyle="none",
        label="Lower surface",
    )
    ax.set_xlabel("x / c")
    ax.set_ylabel("$-C_p$")
    ax.grid(False)
    ax.legend()

    centered_subplots_adjust(fig, outer=0.1215, bottom=0.20, top=0.985)
    save_fig(fig, "airfoil_surface_cp.png", FIGURES_DIR, dpi=args.dpi, figure_format=args.format)
    if abs(args.angle) < 1e-9:
        gap = float(abs(cp[upper].mean() - cp[lower].mean()))
        print(f"  upper/lower mean Cp difference at alpha=0: {gap:.4f} (symmetry check, ~0)")


if __name__ == "__main__":
    main()
