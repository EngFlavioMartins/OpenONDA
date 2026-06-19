#!/usr/bin/env python3
"""
Figure 3 — Total circulation  Σ|Γᵢ|/Γ₀ vs t*

Compares circulation evolution for all three simulations:
leapfrog DNS, leapfrog LES, and collision LES.

Saves: figures/vortex_rings_circulation.png
"""

import glob
from pathlib import Path

import matplotlib.pyplot as plt

from _common import (
    build_arg_parser,
    load_theme,
    load_total_circulation,
    save_fig,
    CM,
)

C_DNS = "#5C3D9B"
C_LES = "#1A8C88"
C_COL = "#B85C2A"


def main():
    args = build_arg_parser("Total circulation Σ|Γᵢ|/Γ₀ vs t* for all cases.").parse_args()
    sol = Path(args.solution_dir)
    figs = Path(args.figures_dir)
    figs.mkdir(parents=True, exist_ok=True)

    load_theme()

    runs = [
        (
            sorted(glob.glob(str(sol / "leapfrog_DNS" / "vpm_leapfrog_DNS_*.h5"))),
            C_DNS,
            "s",
            "DNS leapfrog",
        ),
        (
            sorted(glob.glob(str(sol / "leapfrog_LES" / "vpm_leapfrog_LES_*.h5"))),
            C_LES,
            "v",
            "LES leapfrog",
        ),
        (
            sorted(glob.glob(str(sol / "collide_LES" / "vpm_collide_LES_*.h5"))),
            C_COL,
            "^",
            "LES collision",
        ),
    ]

    fig, ax = plt.subplots(figsize=(12 * CM, 7 * CM))

    for h5_files, color, marker, label in runs:
        t_star, circ_norm = load_total_circulation(h5_files)
        if t_star.size == 0:
            continue
        ax.plot(
            t_star,
            circ_norm,
            "-",
            color=color,
            lw=1.1,
            marker=marker,
            ms=3,
            markevery=5,
            mew=0.4,
            label=label,
        )

    ax.axhspan(1.0, 10.0, facecolor="gray", alpha=0.15, zorder=0)
    ax.set_ylim(0, 1.5)
    ax.set_xlabel(r"Normalized time, $t\,\Gamma / R_0^2$")
    ax.set_ylabel("Total circulation, $\Sigma|\Gamma_i|/\Gamma_0$")
    ax.legend(fontsize=10)
    save_fig(fig, figs / "vortex_rings_circulation.png", dpi=args.dpi)


if __name__ == "__main__":
    main()
