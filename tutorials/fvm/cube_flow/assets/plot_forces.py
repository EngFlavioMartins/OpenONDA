#!/usr/bin/env python3
"""Plot Cd(t), Cl(t) of the square cylinder and extract the Strouhal number,
with reference bands from Okajima (1982), Sohankar et al. (1998), and
Sen et al. (2011)."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _common import (  # noqa: E402
    COLORS,
    REFERENCES,
    build_arg_parser,
    figure_size,
    load_forces_csv,
    save_fig,
    strouhal_from_lift,
)


def main():
    args = build_arg_parser().parse_args()
    ref = REFERENCES.get(args.Re, {})
    data = load_forces_csv(args.solution_dir)
    if "cube" not in data:
        print("  No force data on patch 'cube' to plot.")
        return
    d = data["cube"]
    t = d["time"]
    drag_coefficient = d["drag_coefficient"]
    lift_coefficient = d["lift_coefficient"]

    # Statistics over the settled part (last third).
    i0 = 2 * len(t) // 3
    drag_coefficient_mean = float(np.mean(drag_coefficient[i0:]))
    lift_coefficient_rms = float(
        np.sqrt(np.mean((lift_coefficient[i0:] - np.mean(lift_coefficient[i0:])) ** 2))
    )
    strouhal_number = strouhal_from_lift(t, lift_coefficient)  # f*D/U with D = U = 1

    fig, axes = plt.subplots(2, 1, figsize=figure_size("stacked"), sharex=True)

    ax = axes[0]
    ax.plot(t, drag_coefficient, color=COLORS["TUDdark"], linewidth=0.9)
    if "drag_coefficient" in ref:
        ax.axhspan(
            *ref["drag_coefficient"],
            color=COLORS["reference"],
            alpha=0.25,
            label=f"literature: {ref['drag_coefficient'][0]:.2f}-{ref['drag_coefficient'][1]:.2f}",
        )
        ax.legend(loc="upper right", fontsize=8)
    ax.set_ylabel("drag coefficient")
    ax.set_title(f"Square cylinder forces (Re = {args.Re:g})")
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
    label = f"lift coefficient rms = {lift_coefficient_rms:.4f}"
    if strouhal_number is not None:
        label += f",  strouhal_number = {strouhal_number:.4f}"
        if "strouhal_number" in ref:
            label += f" (ref {ref['strouhal_number'][0]:.3f}-{ref['strouhal_number'][1]:.3f})"
    ax.text(0.02, 0.06, label, transform=ax.transAxes, fontsize=8)
    ax.set_ylabel("lift coefficient")
    ax.set_xlabel("t [s]")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_fig(fig, "forces_cube.png", args.figures_dir, dpi=args.dpi, figure_format=args.format)

    print(f"  cube: mean drag_coefficient = {drag_coefficient_mean:.4f}", end="")
    if "drag_coefficient" in ref:
        lo, hi = ref["drag_coefficient"]
        print(
            f"  [reference {lo:.2f}-{hi:.2f}: {'OK' if lo <= drag_coefficient_mean <= hi else 'OUT OF BAND'}]",
            end="",
        )
    if strouhal_number is not None:
        print(f", strouhal_number = {strouhal_number:.4f}", end="")
        if "strouhal_number" in ref:
            lo, hi = ref["strouhal_number"]
            print(
                f"  [reference {lo:.3f}-{hi:.3f}: {'OK' if lo <= strouhal_number <= hi else 'OUT OF BAND'}]",
                end="",
            )
    print()


if __name__ == "__main__":
    main()
