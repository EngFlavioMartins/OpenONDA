#!/usr/bin/env python3
"""Total particle-strength variation — ``rings_circulation.png``.

Plots Σ|Γᵢ|/Σ|Γᵢ|₀ versus normalised time for every case discovered under
``solution/``, read from the solver log's ``FLOW DIAGNOSTICS`` sections.
This total variation is a stretching/resolution diagnostic, not a conserved
circulation invariant for a three-dimensional vortex-particle field.

Color encodes the numerical method, linestyle the interaction family - the
same key shared by every comparison figure (see ``_common.case_style``).
"""

from pathlib import Path

import matplotlib.pyplot as plt

from _common import (
    build_arg_parser,
    case_style,
    compact_case_legend_handles,
    discover_cases,
    figure_size,
    load_theme,
    mark_every,
    read_metric,
    save_fig,
)


def main() -> None:
    args = build_arg_parser("Total circulation Σ|Γᵢ|/Σ|Γᵢ|₀ vs t*.").parse_args()
    figs = Path(args.figures_dir)
    figs.mkdir(parents=True, exist_ok=True)

    load_theme()
    fig, ax = plt.subplots(figsize=figure_size("wide_short"))

    plotted = False
    maximum_ratio = 1.0
    for case_dir in discover_cases(args.solution_dir):
        t_star, circ = read_metric(case_dir, "sum_gamma_magnitude")
        if t_star.size == 0 or circ[0] <= 0.0:
            continue
        st = case_style(case_dir.name)
        ratio = circ / circ[0]
        ax.plot(
            t_star,
            ratio,
            color=st["color"],
            linestyle=st["linestyle"],
            lw=st["linewidth"],
            marker=st["marker"],
            ms=st["markersize"],
            markevery=mark_every(),
            mew=st["markeredgewidth"],
            label=st["label"],
        )
        maximum_ratio = max(maximum_ratio, float(ratio.max()))
        plotted = True

    ax.axhline(1.0, color="0.55", linestyle=":", linewidth=0.8)
    ax.set_ylim(0.0, 1.05 * maximum_ratio)
    ax.set_xlabel(r"Normalized time, $t\,\Gamma_0 / R_0^2$")
    ax.set_ylabel(r"Strength variation, $S_\alpha/S_{\alpha,0}$")
    if plotted:
        fig.legend(
            handles=compact_case_legend_handles(),
            ncol=4,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
        )

    save_fig(
        fig,
        figs / "rings_circulation.png",
        dpi=args.dpi,
        figure_format=args.format,
        tight_rect=(0.0, 0.27, 1.0, 1.0),
    )


if __name__ == "__main__":
    main()
