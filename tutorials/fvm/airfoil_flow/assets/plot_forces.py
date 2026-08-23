#!/usr/bin/env python3
"""Plot drag and lift histories for the airfoil patch."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _common import (  # noqa: E402
    COLORS,
    RE,
    build_arg_parser,
    figure_size,
    load_forces_csv,
    save_fig,
)


def main():
    args = build_arg_parser().parse_args()
    data = load_forces_csv(args.solution_dir)
    if "airfoil" not in data:
        print("  No force data on patch 'airfoil' to plot.")
        return
    d = data["airfoil"]
    t = d["time"]
    drag_coefficient = d["drag_coefficient"]
    lift_coefficient = d["lift_coefficient"]

    i0 = 2 * len(t) // 3
    drag_coefficient_mean = float(np.mean(drag_coefficient[i0:]))
    lift_coefficient_mean = float(np.mean(lift_coefficient[i0:]))

    fig, axes = plt.subplots(2, 1, figsize=figure_size("stacked"), sharex=True)

    ax = axes[0]
    ax.plot(t, drag_coefficient, color=COLORS["TUDdark"], linewidth=0.9)
    ax.set_ylabel("drag coefficient")
    ax.set_title(f"NACA 0012 forces (Re = {RE:.0f}, $\\alpha$ = {args.angle:g}$^\\circ$)")
    ax.text(
        0.02,
        0.06,
        f"mean drag coefficient (last 1/3) = {drag_coefficient_mean:.4f}",
        transform=ax.transAxes,
        fontsize=8,
    )
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t, lift_coefficient, color=COLORS["TUDdark"], linewidth=0.9)
    ax.text(
        0.02,
        0.06,
        f"mean lift coefficient (last 1/3) = {lift_coefficient_mean:.4f}",
        transform=ax.transAxes,
        fontsize=8,
    )
    ax.set_ylabel("lift coefficient")
    ax.set_xlabel("t [s]")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_fig(fig, "airfoil_forces.png", args.figures_dir, dpi=args.dpi, figure_format=args.format)

    print(
        f"  airfoil: mean drag_coefficient = {drag_coefficient_mean:.4f}, "
        f"mean lift_coefficient = {lift_coefficient_mean:.4f}"
    )


if __name__ == "__main__":
    main()
