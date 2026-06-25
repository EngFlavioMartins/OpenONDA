#!/usr/bin/env python3
"""Total circulation conservation — ``rings_circulation.png``.

Plots Σ|Γᵢ|/Σ|Γᵢ|₀ versus normalised time for every case discovered under
``solution/``, read from the per-step ``stability_metrics.csv``.  A faithful
solver keeps the curve near unity; numerical blow-up shows up as runaway
growth (the shaded band marks the unphysical Σ|Γ| > Σ|Γ|₀ region).

Colour encodes the stabilization rung, linestyle the physics family — the
same key shared by every comparison figure (see ``_common.case_style``).
"""

from pathlib import Path

import matplotlib.pyplot as plt

from _common import (
    CM,
    build_arg_parser,
    case_style,
    discover_cases,
    load_theme,
    read_metric,
    save_fig,
)


def main() -> None:
    args = build_arg_parser("Total circulation Σ|Γᵢ|/Σ|Γᵢ|₀ vs t*.").parse_args()
    figs = Path(args.figures_dir)
    figs.mkdir(parents=True, exist_ok=True)

    load_theme()
    fig, ax = plt.subplots(figsize=(12.8 * CM, 7.0 * CM))

    plotted = False
    for case_dir in discover_cases(args.solution_dir):
        t_star, circ = read_metric(case_dir, "sum_gamma_magnitude")
        if t_star.size == 0 or circ[0] <= 0.0:
            continue
        st = case_style(case_dir.name)
        ax.plot(
            t_star,
            circ / circ[0],
            color=st["color"],
            linestyle=st["linestyle"],
            lw=1.1,
            marker=st["marker"],
            ms=3,
            markevery=20,
            mew=0.4,
            label=st["label"],
        )
        plotted = True

    ax.axhspan(1.0, 10.0, facecolor="gray", alpha=0.25, zorder=0)
    ax.set_ylim(0.0, 1.5)
    ax.set_xlabel(r"Normalized time, $t\,\Gamma_0 / R_0^2$")
    ax.set_ylabel(r"Total circulation, $\Sigma|\Gamma_i| / \Sigma|\Gamma_i|_0$")
    if plotted:
        ax.legend(fontsize=10, ncol=2)

    save_fig(fig, figs / "rings_circulation.png", dpi=args.dpi)


if __name__ == "__main__":
    main()
