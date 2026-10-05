#!/usr/bin/env python3
"""Plot Cd(t), Cl(t) of the square cylinder and extract the Strouhal number,
with reference bands from Okajima (1982), Sohankar et al. (1998), and
Sen et al. (2011)."""

import matplotlib.pyplot as plt
from openonda.plotting import centered_subplots_adjust
import numpy as np

from ._common import (  # noqa: E402
    COLORS,
    FIGURES_DIR,
    REFERENCES,
    SOLUTION_DIR,
    build_arg_parser,
    figure_size,
    load_forces_csv,
    save_fig,
    strouhal_from_lift,
)


def main():
    args = build_arg_parser().parse_args()
    ref = REFERENCES.get(args.Re, {})
    data = load_forces_csv(SOLUTION_DIR)
    d = data["cube"]
    t = d["time"]
    drag_coefficient = d["drag_coefficient"]
    lift_coefficient = d["lift_coefficient"]

    # Show descriptive statistics over the last third of the available history.
    i0 = 2 * len(t) // 3
    drag_coefficient_mean = float(np.mean(drag_coefficient[i0:]))
    lift_coefficient_rms = float(
        np.sqrt(np.mean((lift_coefficient[i0:] - np.mean(lift_coefficient[i0:])) ** 2))
    )
    strouhal_number = strouhal_from_lift(t, lift_coefficient)  # f*D/U with D = U = 1

    fig, axes = plt.subplots(2, 1, figsize=figure_size("stacked"), sharex=True)

    ax = axes[0]
    ax.plot(t, drag_coefficient, color=COLORS["fvm"], linewidth=0.9)
    if "drag_coefficient" in ref:
        ax.axhspan(
            *ref["drag_coefficient"],
            color=COLORS["reference"],
            alpha=0.25,
            label=f"literature: {ref['drag_coefficient'][0]:.2g}-{ref['drag_coefficient'][1]:.2g}",
        )
        ax.legend(loc="upper right")
    ax.set_ylabel("drag coefficient")
    print(rf"$\overline{{C_D}}$ (last 1/3) = {drag_coefficient_mean:.4f}")
    ax.grid(False)

    ax = axes[1]
    ax.plot(t, lift_coefficient, color=COLORS["fvm"], linewidth=0.9)
    label = rf"$C_{{L,\mathrm{{rms}}}}$ = {lift_coefficient_rms:.2g}"
    if strouhal_number is not None:
        label += f"\n$St$ = {strouhal_number:.2g}"
        if "strouhal_number" in ref:
            label += f" (ref {ref['strouhal_number'][0]:.2g}-{ref['strouhal_number'][1]:.2g})"
    print(label)
    ax.set_ylabel("lift coefficient")
    ax.set_xlabel("t [s]")
    ax.grid(False)

    centered_subplots_adjust(fig, outer=0.18, bottom=0.12, top=0.96, hspace=0.30)
    save_fig(fig, "forces_cube.png", FIGURES_DIR, dpi=args.dpi, figure_format=args.format)

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
