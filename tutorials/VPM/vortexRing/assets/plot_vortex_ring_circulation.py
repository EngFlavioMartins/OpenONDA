#!/usr/bin/env python3
"""
Total circulation Σ|Γᵢ|/Γ₀ vs t*.
All six combinations: DNS/LES × (direct, transposed, mixed).

Saves: figures/vortex_ring_circulation.png
"""

import sys
import glob
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    build_arg_parser,
    load_theme,
    load_total_circulation,
    save_fig,
    VARIANT_STYLE,
    CM,
)


def main() -> None:
    args = build_arg_parser("Total circulation — DNS & LES, all stretching variants.").parse_args()
    sol = Path(args.solution_dir)
    figs = Path(args.figures_dir)
    figs.mkdir(parents=True, exist_ok=True)

    load_theme()

    fig, ax = plt.subplots(figsize=(12 * CM, 7 * CM))

    for variant, st in VARIANT_STYLE.items():
        h5 = sorted(glob.glob(str(sol / variant / f"vpm_{variant}_*.h5")))
        t, c = load_total_circulation(h5)
        if t.size == 0:
            continue
        ls = "--" if variant.startswith("DNS") else "-"
        label = variant.replace("_", " ")
        ax.plot(
            t,
            c,
            ls,
            color=st["color"],
            lw=1.1,
            marker=st["marker"],
            ms=3,
            markevery=3,
            mew=0.4,
            label=label,
        )

    ax.axhline(
        1.0,
        color="black",
        lw=0.6,
        ls=":",
        label=r"$\Sigma|\boldsymbol{\Gamma}_i|/\Gamma_0 = 1$",
    )
    ax.set_xlabel(r"Normalized time, $t\,\Gamma / R_0^2$")
    ax.set_ylabel(r"Total circulation, $\Sigma\,|\boldsymbol{\Gamma}_i|\,/\,\Gamma_0$")
    ax.set_ylim(0.8, 3.8)
    ax.set_xlim(0, 38)
    ax.legend(fontsize=10, ncol=1)
    save_fig(fig, figs / "vortex_ring_circulation.png", dpi=args.dpi)


if __name__ == "__main__":
    main()
