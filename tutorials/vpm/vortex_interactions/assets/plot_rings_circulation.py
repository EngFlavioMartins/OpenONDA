#!/usr/bin/env python3
"""Relaxed tube-circulation histories for the two vortex-ring interactions."""

from pathlib import Path

import matplotlib.pyplot as plt

from ring_metrics import (
    FAMILIES,
    FAMILY_FILE_STEMS,
    FAMILY_LABELS,
    REFERENCE_TIME,
    build_arg_parser,
    case_style,
    case_legend_handles,
    discover_cases,
    figure_size,
    load_theme,
    mark_every,
    read_ring_diagnostics,
    save_fig,
)

RELAXED_SAMPLE_INDEX = 3


def main() -> None:
    args = build_arg_parser("Tube circulation versus t Gamma_0 / R_0^2.").parse_args()

    load_theme()
    for family in FAMILIES:
        fig, ax = plt.subplots(figsize=figure_size("single"))
        plotted: list[str] = []
        for case_dir in discover_cases(args.solution_dir, family=family):
            diagnostics = read_ring_diagnostics(case_dir)
            if diagnostics is None or "tube_circulation" not in diagnostics:
                continue
            style = case_style(case_dir.name)
            case_plotted = False
            for _, ring in diagnostics.groupby("group_id", sort=True):
                ring = ring.sort_values("step", kind="stable").iloc[RELAXED_SAMPLE_INDEX:]
                if ring.empty:
                    continue
                reference = float(ring["tube_circulation"].iloc[0])
                if reference <= 0.0:
                    continue
                ax.plot(
                    ring["time"].to_numpy(float) / REFERENCE_TIME,
                    ring["tube_circulation"].to_numpy(float) / reference,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    lw=style["linewidth"],
                    marker=style["marker"],
                    ms=style["markersize"],
                    markevery=mark_every(),
                    mew=style["markeredgewidth"],
                )
                case_plotted = True
            if not case_plotted:
                continue
            plotted.append(case_dir.name)
        ax.axhline(1.0, color="0.55", linestyle=":", linewidth=0.8)
        ax.set_title(FAMILY_LABELS[family])
        ax.set_xlabel(r"Normalized time, $t\,\Gamma_0 / R_0^2$")
        ax.set_ylabel(r"Tube circulation, $\Gamma_{\mathrm{tube}}/\Gamma_{\mathrm{tube},0}$")
        ax.margins(y=0.08)
        if plotted:
            ax.legend(
                handles=case_legend_handles(plotted),
                loc="best",
            )

        save_fig(
            fig,
            Path("figures") / f"{FAMILY_FILE_STEMS[family]}_circulation.png",
            dpi=args.dpi,
            figure_format=args.format,
        )


if __name__ == "__main__":
    main()
