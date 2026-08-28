#!/usr/bin/env python3
"""Kinetic-energy evolution for the two vortex-ring interactions."""

from pathlib import Path

import matplotlib.pyplot as plt

from ring_metrics import (
    FAMILIES,
    FAMILY_LABELS,
    REFERENCE_TIME,
    build_arg_parser,
    case_style,
    case_legend_handles,
    discover_cases,
    figure_size,
    load_theme,
    mark_every,
    read_integrals,
    save_fig,
)


def main() -> None:
    args = build_arg_parser("Kinetic energy E/E₀ vs t*.").parse_args()

    load_theme()
    fig, axes = plt.subplots(1, 2, figsize=figure_size("trajectory"), sharey=True)

    plotted: list[str] = []
    for ax, family in zip(axes, FAMILIES, strict=True):
        for case_dir in discover_cases(args.solution_dir, family=family):
            df = read_integrals(case_dir)
            if df is None or len(df) == 0:
                continue
            energy = df["total_kinetic_energy"].to_numpy(float)
            if energy[0] <= 0.0:
                continue
            style = case_style(case_dir.name)
            ax.plot(
                df["time"].to_numpy(float) / REFERENCE_TIME,
                energy / energy[0],
                color=style["color"],
                linestyle=style["linestyle"],
                lw=style["linewidth"],
                marker=style["marker"],
                ms=style["markersize"],
                markevery=mark_every("total_kinetic_energy"),
                mew=style["markeredgewidth"],
            )
            plotted.append(case_dir.name)
        ax.axhline(1.0, color="0.55", linestyle=":", linewidth=0.8)
        ax.set_title(FAMILY_LABELS[family])
        ax.set_xlabel(r"Normalized time, $t\,\Gamma_0 / R_0^2$")
        ax.margins(y=0.08)

    axes[0].set_ylabel(r"Kinetic energy, $E/E_0$")
    if plotted:
        fig.legend(
            handles=case_legend_handles(plotted),
            ncol=3,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
        )

    save_fig(
        fig,
        Path("figures") / "rings_energy.png",
        dpi=args.dpi,
        figure_format=args.format,
        tight_rect=(0.0, 0.14, 1.0, 1.0),
    )


if __name__ == "__main__":
    main()
