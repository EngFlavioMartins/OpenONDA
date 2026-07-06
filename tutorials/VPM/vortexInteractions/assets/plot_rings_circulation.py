#!/usr/bin/env python3
"""Total circulation conservation — ``rings_circulation.png``.

Plots Σ|Γᵢ|/Σ|Γᵢ|₀ versus normalised time for every case discovered under
``solution/``, read from the solver log's ``FLOW DIAGNOSTICS`` sections.  A faithful
solver keeps the curve near unity; numerical blow-up shows up as runaway
growth (the shaded band marks the unphysical Σ|Γ| > Σ|Γ|₀ region).

Color encodes the stabilization method, linestyle the interaction family - the
same key shared by every comparison figure (see ``_common.case_style``).
"""

from pathlib import Path

import matplotlib.pyplot as plt

from _common import (
    build_arg_parser,
    case_style,
    discover_cases,
    figure_size,
    load_theme,
    mark_every,
    read_metric,
    reference_fill_style,
    save_fig,
)


def main() -> None:
    args = build_arg_parser("Total circulation Σ|Γᵢ|/Σ|Γᵢ|₀ vs t*.").parse_args()
    figs = Path(args.figures_dir)
    figs.mkdir(parents=True, exist_ok=True)

    load_theme()
    fig, ax = plt.subplots(figsize=figure_size("single"))

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
            lw=st["linewidth"],
            marker=st["marker"],
            ms=st["markersize"],
            markevery=mark_every(),
            mew=st["markeredgewidth"],
            label=st["label"],
        )
        plotted = True

    ax.axhspan(1.0, 10.0, **reference_fill_style())
    ax.set_ylim(0.0, 1.5)
    ax.set_xlabel(r"Normalized time, $t\,\Gamma_0 / R_0^2$")
    ax.set_ylabel(r"Total circulation, $\Sigma|\Gamma_i| / \Sigma|\Gamma_i|_0$")
    if plotted:
        ax.legend(ncol=2, loc="best")

    save_fig(fig, figs / "rings_circulation.png", dpi=args.dpi, figure_format=args.format)


if __name__ == "__main__":
    main()
