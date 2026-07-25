#!/usr/bin/env python3
"""Stability ladder — ``rings_stability.png``.

Tracks the peak circulation max|Γᵢ|/max|Γᵢ|₀ versus normalised time for every
case discovered under ``solution/`` (read from ``BLOWUP CHECK`` lines in the
solver log).  A runaway curve marks the onset of numerical blow-up, and where
each curve ends marks that variant's survival time.  The shaded band is the
50× blow-up threshold used by the solver to stop a diverging run.

Color encodes the stabilization method, linestyle the interaction family - the
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
    reference_fill_style,
    save_fig,
)

BLOWUP_FACTOR = 50.0  # mirrors rings_setup.py's max|Γ| > 50× initial stop


def main() -> None:
    args = build_arg_parser("Peak circulation max|Γᵢ|/max|Γᵢ|₀ vs t*.").parse_args()
    figs = Path(args.figures_dir)
    figs.mkdir(parents=True, exist_ok=True)

    load_theme()
    fig, ax = plt.subplots(figsize=figure_size("wide_short"))

    plotted = False
    minimum_ratio = 1.0
    for case_dir in discover_cases(args.solution_dir):
        # Keep the full series (including the runaway) so survival time is visible.
        t_star, gmax = read_metric(case_dir, "max_gamma", truncate_blowup=False)
        if t_star.size == 0 or gmax[0] <= 0.0:
            continue
        st = case_style(case_dir.name)
        ratio = gmax / gmax[0]
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
        positive = ratio[ratio > 0.0]
        if positive.size:
            minimum_ratio = min(minimum_ratio, float(positive.min()))
        plotted = True

    ax.set_yscale("log")
    ax.set_xlabel(r"Normalized time, $t\,\Gamma_0 / R_0^2$")
    ax.set_ylabel(r"Peak strength, $\alpha_{\max}/\alpha_{\max,0}$")
    ax.axhspan(BLOWUP_FACTOR, 100, **reference_fill_style("strong"))
    ax.set_ylim([max(1e-3, 0.8 * minimum_ratio), 100])
    if plotted:
        fig.legend(
            handles=compact_case_legend_handles(),
            ncol=5,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
        )

    save_fig(
        fig,
        figs / "rings_stability.png",
        dpi=args.dpi,
        figure_format=args.format,
        tight_rect=(0.0, 0.27, 1.0, 1.0),
    )


if __name__ == "__main__":
    main()
