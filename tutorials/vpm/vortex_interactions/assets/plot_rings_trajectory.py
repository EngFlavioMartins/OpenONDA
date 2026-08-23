#!/usr/bin/env python3
"""Leapfrogging-ring trajectory comparison.

Overlays the circulation-weighted ring trajectory (R/R₀ vs x/R₀) sampled by
the VPM solver for every leapfrogging case. Each case keeps one colour and
linestyle for both rings.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ASSETS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ASSETS_DIR))
from _common import (  # noqa: E402
    RING_RADIUS,
    build_arg_parser,
    case_style,
    compact_case_legend_handles,
    discover_cases,
    figure_size,
    load_theme,
    mark_every,
    read_ring_diagnostics,
    save_fig,
)


def load_trajectory(case_dir: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Return the sampled axial position and major radius of each ring."""
    diagnostics = read_ring_diagnostics(case_dir)
    if diagnostics is None:
        return {}
    return {
        int(group_id): (
            group["vortex_centroid_x"].to_numpy(float) / RING_RADIUS,
            group["major_radius"].to_numpy(float) / RING_RADIUS,
        )
        for group_id, group in diagnostics.groupby("group_id", sort=True)
    }


def main() -> None:
    parser = build_arg_parser(__doc__)
    args = parser.parse_args()

    load_theme()
    fig, ax = plt.subplots(figsize=figure_size("trajectory"))

    plotted = False
    for case_dir in discover_cases(args.solution_dir, family="leapfrog"):
        trajectory = load_trajectory(case_dir)
        if not trajectory:
            continue
        st = case_style(case_dir.name, include_family=False)
        for axial, radius in trajectory.values():
            ax.plot(
                axial,
                radius,
                color=st["color"],
                linestyle=st["linestyle"],
                lw=st["linewidth"],
                marker=st["marker"],
                ms=st["markersize"],
                markevery=mark_every("trajectory"),
                mew=st["markeredgewidth"],
            )
        plotted = True

    legend_handles = compact_case_legend_handles(include_families=False) if plotted else []

    ax.set_xlabel(r"Axial position, $x/R_0$")
    ax.set_ylabel(r"Ring radius, $R/R_0$")
    ax.set_ylim([0.5, 1.5])
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            ncol=4,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
        )

    save_fig(
        fig,
        Path(args.figures_dir) / "rings_trajectory.png",
        dpi=args.dpi,
        figure_format=args.format,
        tight_rect=(0.0, 0.22, 1.0, 1.0),
    )


if __name__ == "__main__":
    main()
