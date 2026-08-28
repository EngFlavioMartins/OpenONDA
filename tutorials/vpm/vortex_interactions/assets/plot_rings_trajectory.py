#!/usr/bin/env python3
"""Ring trajectories for the leapfrogging and collision cases."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ring_metrics import (
    FAMILIES,
    FAMILY_LABELS,
    RING_RADIUS,
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
    fig, axes = plt.subplots(1, 2, figsize=figure_size("trajectory"), sharey=True)

    plotted: list[str] = []
    for ax, family in zip(axes, FAMILIES, strict=True):
        for case_dir in discover_cases(args.solution_dir, family=family):
            trajectory = load_trajectory(case_dir)
            if not trajectory:
                continue
            style = case_style(case_dir.name)
            for axial_position, ring_radius in trajectory.values():
                ax.plot(
                    axial_position,
                    ring_radius,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    lw=style["linewidth"],
                    marker=style["marker"],
                    ms=style["markersize"],
                    markevery=mark_every("trajectory"),
                    mew=style["markeredgewidth"],
                )
            plotted.append(case_dir.name)
        ax.set_title(FAMILY_LABELS[family])
        ax.set_xlabel(r"Axial position, $x/R_0$")
        ax.margins(y=0.08)

    axes[0].set_ylabel(r"Ring radius, $R/R_0$")
    if plotted:
        fig.legend(
            handles=case_legend_handles(plotted),
            ncol=3,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
        )

    save_fig(
        fig,
        Path("figures") / "rings_trajectory.png",
        dpi=args.dpi,
        figure_format=args.format,
        tight_rect=(0.0, 0.14, 1.0, 1.0),
    )


if __name__ == "__main__":
    main()
