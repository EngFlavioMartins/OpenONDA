#!/usr/bin/env python3
"""Total particle strength for the two vortex-ring interactions."""

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
    save_fig,
)


def main() -> None:
    args = build_arg_parser("Total circulation Σ|Γᵢ|/Σ|Γᵢ|₀ vs t*.").parse_args()

    load_theme()
    fig, axes = plt.subplots(1, 2, figsize=figure_size("trajectory"), sharey=True)

    plotted: list[str] = []
    for ax, family in zip(axes, FAMILIES, strict=True):
        for case_dir in discover_cases(args.solution_dir, family=family):
            time, strength = read_metric(case_dir, "vortex_strength_magnitude_sum")
            if time.size == 0 or strength[0] <= 0.0:
                continue
            style = case_style(case_dir.name)
            ax.plot(
                time,
                strength / strength[0],
                color=style["color"],
                linestyle=style["linestyle"],
                lw=style["linewidth"],
                marker=style["marker"],
                ms=style["markersize"],
                markevery=mark_every(),
                mew=style["markeredgewidth"],
            )
            plotted.append(case_dir.name)
        ax.axhline(1.0, color="0.55", linestyle=":", linewidth=0.8)
        ax.set_title(FAMILY_LABELS[family])
        ax.set_xlabel(r"Normalized time, $t\,\Gamma_0 / R_0^2$")
        ax.margins(y=0.08)

    axes[0].set_ylabel(r"Total strength, $\sum|\alpha_i|/\sum|\alpha_i|_0$")
    if plotted:
        fig.legend(
            handles=case_legend_handles(plotted),
            ncol=3,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
        )

    save_fig(
        fig,
        Path("figures") / "rings_circulation.png",
        dpi=args.dpi,
        figure_format=args.format,
        tight_rect=(0.0, 0.14, 1.0, 1.0),
    )


if __name__ == "__main__":
    main()
