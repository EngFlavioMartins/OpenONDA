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
    t, cd, cl = d["time"], d["Cd"], d["Cl"]

    # Statistics over the settled part (last third).
    i0 = 2 * len(t) // 3
    cd_mean = float(np.mean(cd[i0:]))
    cl_rms = float(np.sqrt(np.mean((cl[i0:] - np.mean(cl[i0:])) ** 2)))
    st = strouhal_from_lift(t, cl)  # f*D/U with D = U = 1

    fig, axes = plt.subplots(2, 1, figsize=figure_size("stacked"), sharex=True)

    ax = axes[0]
    ax.plot(t, cd, color=COLORS["TUDdark"], linewidth=0.9)
    if "Cd" in ref:
        ax.axhspan(
            *ref["Cd"],
            color=COLORS["reference"],
            alpha=0.25,
            label=f"literature: {ref['Cd'][0]:.2f}-{ref['Cd'][1]:.2f}",
        )
        ax.legend(loc="upper right", fontsize=8)
    ax.set_ylabel("$C_d$")
    ax.set_title(f"Square cylinder forces (Re = {args.Re:g})")
    ax.text(
        0.02, 0.06, f"mean $C_d$ (last 1/3) = {cd_mean:.4f}", transform=ax.transAxes, fontsize=8
    )
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t, cl, color=COLORS["TUDdark"], linewidth=0.9)
    label = f"$C_l$ rms = {cl_rms:.4f}"
    if st is not None:
        label += f",  St = {st:.4f}"
        if "St" in ref:
            label += f" (ref {ref['St'][0]:.3f}-{ref['St'][1]:.3f})"
    ax.text(0.02, 0.06, label, transform=ax.transAxes, fontsize=8)
    ax.set_ylabel("$C_l$")
    ax.set_xlabel("t [s]")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_fig(fig, "forces_cube.png", args.figures_dir, dpi=args.dpi, figure_format=args.format)

    print(f"  cube: mean Cd = {cd_mean:.4f}", end="")
    if "Cd" in ref:
        lo, hi = ref["Cd"]
        print(
            f"  [reference {lo:.2f}-{hi:.2f}: {'OK' if lo <= cd_mean <= hi else 'OUT OF BAND'}]",
            end="",
        )
    if st is not None:
        print(f", St = {st:.4f}", end="")
        if "St" in ref:
            lo, hi = ref["St"]
            print(
                f"  [reference {lo:.3f}-{hi:.3f}: {'OK' if lo <= st <= hi else 'OUT OF BAND'}]",
                end="",
            )
    print()


if __name__ == "__main__":
    main()
