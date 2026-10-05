#!/usr/bin/env python3
"""Plot drag and lift histories for the airfoil patch."""

import matplotlib.pyplot as plt
from openonda.plotting import centered_subplots_adjust
import numpy as np

from ._common import (  # noqa: E402
    COLORS,
    FIGURES_DIR,
    RE,
    SOLUTION_DIR,
    build_arg_parser,
    figure_size,
    load_forces_csv,
    save_fig,
)


def main():
    args = build_arg_parser().parse_args()
    data = load_forces_csv(SOLUTION_DIR)
    d = data["airfoil"]
    t = d["time"]
    drag_coefficient = d["drag_coefficient"]
    lift_coefficient = d["lift_coefficient"]

    i0 = 2 * len(t) // 3
    drag_coefficient_mean = float(np.mean(drag_coefficient[i0:]))
    lift_coefficient_mean = float(np.mean(lift_coefficient[i0:]))

    fig, axes = plt.subplots(2, 1, figsize=figure_size("stacked"), sharex=True)

    ax = axes[0]
    ax.plot(
        t,
        drag_coefficient,
        color=COLORS["fvm"],
        linewidth=0.9,
        label=r"$C_D$",
    )
    ax.set_ylabel("drag coefficient")
    ax.legend(loc="best")
    ax.grid(False)

    ax = axes[1]
    ax.plot(
        t,
        lift_coefficient,
        color=COLORS["fvm"],
        linewidth=0.9,
        label=r"$C_L$",
    )
    ax.legend(loc="best")
    ax.set_ylabel("lift coefficient")
    ax.set_xlabel("t [s]")
    ax.grid(False)

    centered_subplots_adjust(fig, outer=0.135, bottom=0.12, top=0.991, hspace=0.23)
    save_fig(fig, "airfoil_forces.png", FIGURES_DIR, dpi=args.dpi, figure_format=args.format)

    print(
        f"  airfoil: mean drag_coefficient = {drag_coefficient_mean:.4f}, "
        f"mean lift_coefficient = {lift_coefficient_mean:.4f}"
    )


if __name__ == "__main__":
    main()
