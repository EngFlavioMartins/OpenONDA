#!/usr/bin/env python3
"""Discretization health — ``rings_resolution.png``.

The companion to ``rings_conservation``.  That figure shows the invariants; a
structure-preserving scheme holds those whether or not the particle field still
resolves the flow, so on its own it cannot distinguish a trustworthy run from a
conservative-but-wrong one.  This one shows the resolution instead:

* particle overlap ``h/sigma`` — quadrature converges only while blobs overlap;
* ``||div w|| / ||grad w||`` — the exact vorticity field is solenoidal, the
  discrete one is not, and stretching amplifies precisely its divergent part;
* the angle between ``alpha_p`` and ``w(x_p)`` — parallel in the continuum.

Shaded bands mark the acceptance limits enforced by ``assets/postprocess.py``.

Color encodes the numerical method, linestyle the interaction family - the
same key shared by every comparison figure (see ``_common.case_style``).
"""

from pathlib import Path

import matplotlib.pyplot as plt

from _common import (
    REFERENCE_TIME,
    build_arg_parser,
    case_style,
    compact_case_legend_handles,
    discover_cases,
    load_theme,
    mark_every,
    read_integrals,
    reference_fill_style,
    save_fig,
)

# Mirrors the hard limits in rings_setup.py and assets/check_run.py.
MAX_OVERLAP_RATIO = 1.25
MAX_DIVERGENCE_ERROR = 0.25
MAX_MISALIGNMENT_DEG = 45.0


def main() -> None:
    args = build_arg_parser(
        "Particle overlap, vorticity-divergence error and Gamma-omega misalignment."
    ).parse_args()

    load_theme()
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 7.2), sharex=True)
    ax_overlap, ax_divergence, ax_angle = axes

    panels = (
        (ax_overlap, "mean_overlap_ratio", False),
        (ax_divergence, "vorticity_divergence_error", True),
        (ax_angle, "vortex_strength_misalignment_degrees", True),
    )

    plotted = False
    for case_dir in discover_cases(args.solution_dir):
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
            axis.plot(nondimensional_time, series, label=st["label"], **common)
            plotted = True

    ax_overlap.set_ylabel(r"Particle overlap, $h_{nn}/\sigma$")
    ax_overlap.set_title("Discretization health")
    ax_overlap.axhspan(MAX_OVERLAP_RATIO, 10.0, **reference_fill_style("strong"))
    ax_overlap.set_ylim(0.0, 1.2)

    ax_divergence.set_yscale("log")
    ax_divergence.set_ylabel(r"$\|\nabla\!\cdot\!\omega\|\,/\,\|\nabla\omega\|$")
    ax_divergence.axhspan(MAX_DIVERGENCE_ERROR, 10.0, **reference_fill_style("strong"))

    ax_angle.set_yscale("log")
    ax_angle.set_ylabel(r"$\angle(\alpha_p,\ \omega(x_p))$  [deg]")
    ax_angle.set_xlabel(r"Normalized time, $t\,\Gamma_0 / R_0^2$")
    ax_angle.axhspan(MAX_MISALIGNMENT_DEG, 180.0, **reference_fill_style("strong"))

    if plotted:
        fig.legend(
            handles=compact_case_legend_handles(),
            ncol=5,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
        )

    save_fig(
        fig,
        Path("figures") / "rings_resolution.png",
        dpi=args.dpi,
        figure_format=args.format,
        tight_rect=(0.0, 0.12, 1.0, 1.0),
    )


if __name__ == "__main__":
    main()
