#!/usr/bin/env python3
"""Surface pressure distribution -Cp(x/c) from solution/surface_cp.csv
(written by airfoil_flow_setup.py at the end of the run)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import (  # noqa: E402
    COLORS,
    FIGURES_DIR,
    RE,
    SOLUTION_DIR,
    build_arg_parser,
    figure_size,
    load_csv_columns,
    save_fig,
)


def main():
    args = build_arg_parser().parse_args()
    data = load_csv_columns(Path(SOLUTION_DIR) / "surface_cp.csv")
    if not data:
        return
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
        color=COLORS["TUDdark"],
        markersize=3,
        linestyle="none",
        label="Upper surface",
    )
    ax.plot(
        x[lower],
        -cp[lower],
        "s",
        color=COLORS["FVMorange"],
        markersize=3,
        linestyle="none",
        label="Lower surface",
    )
    ax.set_xlabel("x / c")
    ax.set_ylabel("$-C_p$")
    ax.set_title(f"NACA 0012 surface pressure (Re = {RE:.0f}, $\\alpha$ = {args.angle:g}$^\\circ$)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    save_fig(fig, "airfoil_surface_cp.png", FIGURES_DIR, dpi=args.dpi, figure_format=args.format)
    if abs(args.angle) < 1e-9:
        gap = float(abs(cp[upper].mean() - cp[lower].mean()))
        print(f"  upper/lower mean Cp difference at alpha=0: {gap:.4f} (symmetry check, ~0)")


if __name__ == "__main__":
    main()
