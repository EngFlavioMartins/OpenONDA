#!/usr/bin/env python3
"""Discretization health — ``rings_resolution.png``.

The companion to ``rings_conservation``.  That figure shows the invariants; a
structure-preserving scheme holds those whether or not the particle field still
resolves the flow, so on its own it cannot distinguish a trustworthy run from a
conservative-but-wrong one.  This one shows the resolution instead:

* particle overlap ``h/sigma_p`` — quadrature converges only while blobs overlap;
* ``||div w|| / ||grad w||`` — the exact vorticity field is solenoidal, the
  discrete one is not, and stretching amplifies precisely its divergent part;
* the angle between ``alpha_p`` and ``omega(x_p)`` — parallel in the continuum.

Shaded bands mark the indicated resolution limits.

Every case carries its own colour, marker and legend entry — the same key
shared by every comparison figure (see ``ring_metrics.case_style``).
"""

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
    read_integrals,
    reference_fill_style,
    save_fig,
)

# Resolution limits shown by the shaded regions.
MAX_OVERLAP_RATIO = 1.25
MAX_DIVERGENCE_ERROR = 0.25
MAX_MISALIGNMENT_DEG = 45.0


def main() -> None:
    args = build_arg_parser(
        "Particle overlap, vorticity-divergence error and alpha-omega misalignment."
    ).parse_args()

    load_theme()
    for family in FAMILIES:
        width, height = figure_size("stacked")
        fig, axes = plt.subplots(
            4,
            1,
            figsize=(width, 1.2 * height),
            sharex=True,
            gridspec_kw={"height_ratios": (0.75, 1.0, 1.0, 1.0)},
        )
        fig.suptitle(f"{FAMILY_LABELS[family]}: discretization health")
        ax_legend, ax_overlap, ax_divergence, ax_angle = axes
        ax_legend.axis("off")

        panels = (
            (ax_overlap, "mean_overlap_ratio", False),
            (ax_divergence, "vorticity_divergence_error", True),
            (ax_angle, "vortex_strength_misalignment_degrees", True),
        )

        plotted: list[str] = []
        for case_dir in discover_cases(args.solution_dir, family=family):
            df = read_integrals(case_dir)
            if df is None or len(df) == 0:
                continue
            st = case_style(case_dir.name)
            nondimensional_time = df["time"].to_numpy(float) / REFERENCE_TIME
            common = dict(
                color=st["color"],
                linestyle=st["linestyle"],
                lw=st["linewidth"],
                marker=st["marker"],
                ms=st["markersize"],
                markevery=mark_every("total_kinetic_energy"),
                mew=st["markeredgewidth"],
            )
            for axis, column, positive_only in panels:
                if column not in df.columns:
                    continue
                series = df[column].to_numpy(float)
                if positive_only:
                    series = series.clip(min=1e-12)
                axis.plot(nondimensional_time, series, **common)
            plotted.append(case_dir.name)

        ax_overlap.set_ylabel(r"$h_{\mathrm{nn}}/\sigma_p$")
        ax_overlap.axhspan(MAX_OVERLAP_RATIO, 10.0, **reference_fill_style("strong"))
        ax_overlap.set_ylim(0.0, 1.2)

        ax_divergence.set_yscale("log")
        ax_divergence.set_ylabel(
            r"$\|\nabla\!\cdot\!\boldsymbol{\omega}\|"
            r"\,/\,\|\nabla\boldsymbol{\omega}\|$"
        )
        ax_divergence.axhspan(MAX_DIVERGENCE_ERROR, 10.0, **reference_fill_style("strong"))

        ax_angle.set_yscale("log")
        ax_angle.set_ylabel(
            r"$\angle(\boldsymbol{\alpha}_p,\boldsymbol{\omega}_p)$ [deg]"
        )
        ax_angle.set_xlabel(r"Normalized time, $t\,\Gamma_0 / R_0^2$")
        ax_angle.axhspan(MAX_MISALIGNMENT_DEG, 180.0, **reference_fill_style("strong"))

        if plotted:
            many_methods = len(plotted) > 3
            ax_legend.legend(
                handles=case_legend_handles(plotted),
                ncol=2 if many_methods else len(plotted),
                loc="center",
                columnspacing=1.0,
                handletextpad=0.5,
            )

        save_fig(
            fig,
            Path("figures") / f"{FAMILY_FILE_STEMS[family]}_resolution.png",
            dpi=args.dpi,
            figure_format=args.format,
        )


if __name__ == "__main__":
    main()
