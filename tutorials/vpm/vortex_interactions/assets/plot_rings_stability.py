#!/usr/bin/env python3
"""Peak particle strength for the DNS--LES stability ladder."""

from pathlib import Path

import matplotlib.pyplot as plt

from ring_metrics import (
    FAMILIES,
    FAMILY_LABELS,
    build_arg_parser,
    case_style,
    case_legend_handles,
    discover_cases,
    figure_size,
    load_theme,
    mark_every,
    read_metric,
    reference_fill_style,
    save_fig,
)

BLOWUP_FACTOR = 50.0


def main() -> None:
    args = build_arg_parser("Peak circulation max|Γᵢ|/max|Γᵢ|₀ vs t*.").parse_args()

    load_theme()
    fig, axes = plt.subplots(1, 2, figsize=figure_size("trajectory"), sharey=True)

    plotted: list[str] = []
    min_ratio = 1.0
    max_ratio = BLOWUP_FACTOR
    for ax, family in zip(axes, FAMILIES, strict=True):
        for case_dir in discover_cases(args.solution_dir, family=family):
            time, peak_strength = read_metric(case_dir, "max_vortex_strength_magnitude")
            if time.size == 0 or peak_strength[0] <= 0.0:
                continue
            style = case_style(case_dir.name)
            ratio = peak_strength / peak_strength[0]
            ax.plot(
                time,
                ratio,
                color=style["color"],
                linestyle=style["linestyle"],
                lw=style["linewidth"],
                marker=style["marker"],
                ms=style["markersize"],
                markevery=mark_every(),
                mew=style["markeredgewidth"],
            )
            positive = ratio[ratio > 0.0]
            if positive.size:
                min_ratio = min(min_ratio, float(positive.min()))
                max_ratio = max(max_ratio, float(positive.max()))
            plotted.append(case_dir.name)
        ax.set_yscale("log")
        ax.set_title(FAMILY_LABELS[family])
        ax.set_xlabel(r"Normalized time, $t\,\Gamma_0 / R_0^2$")

    upper = max(100.0, 1.1 * max_ratio)
    for ax in axes:
        ax.axhspan(BLOWUP_FACTOR, upper, **reference_fill_style("strong"))
        ax.set_ylim(max(1e-3, 0.8 * min_ratio), upper)
    axes[0].set_ylabel(r"Peak strength, $\alpha_{\max}/\alpha_{\max,0}$")
    if plotted:
        fig.legend(
            handles=case_legend_handles(plotted),
            ncol=3,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
        )

    save_fig(
        fig,
        Path("figures") / "rings_stability.png",
        dpi=args.dpi,
        figure_format=args.format,
        tight_rect=(0.0, 0.14, 1.0, 1.0),
    )


if __name__ == "__main__":
    main()
