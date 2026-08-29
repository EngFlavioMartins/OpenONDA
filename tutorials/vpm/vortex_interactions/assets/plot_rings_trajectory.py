#!/usr/bin/env python3
"""Ring trajectories for the leapfrogging and collision cases."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from ring_metrics import (
    ASSETS_DIR,
    FAMILIES,
    FAMILY_FILE_STEMS,
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
    reference_style,
    save_fig,
)

REFERENCE = ASSETS_DIR / "references" / "leapfrogging_lbm_trajectory.csv"
REFERENCE_INITIAL_MIDPOINT_OVER_R0 = 2.5


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


def plot_leapfrogging_reference(ax) -> bool:
    """Plot the Cheng et al. LBM core-centre trajectories when available."""
    if not REFERENCE.is_file():
        return False
    reference = pd.read_csv(REFERENCE)
    required = {"ring", "x_over_R0", "R_over_R0"}
    if not required.issubset(reference.columns):
        missing = sorted(required - set(reference.columns))
        raise ValueError(f"Missing columns in {REFERENCE}: {missing}")
    style = reference_style()
    for _, ring in reference.groupby("ring", sort=True):
        ax.plot(
            ring["x_over_R0"] - REFERENCE_INITIAL_MIDPOINT_OVER_R0,
            ring["R_over_R0"],
            zorder=1,
            **style,
        )
    return True


def main() -> None:
    parser = build_arg_parser(__doc__)
    args = parser.parse_args()

    load_theme()
    for family in FAMILIES:
        fig, ax = plt.subplots(figsize=figure_size("single"))
        has_reference = family == "leapfrog" and plot_leapfrogging_reference(ax)
        plotted: list[str] = []
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
        ax.set_ylabel(r"Ring radius, $R/R_0$")
        ax.margins(y=0.08)
        handles = case_legend_handles(plotted)
        if has_reference:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    label="LBM, Cheng et al. (2015)",
                    **reference_style(),
                )
            )
        if handles:
            ax.legend(
                handles=handles,
                loc="best",
            )

        save_fig(
            fig,
            Path("figures") / f"{FAMILY_FILE_STEMS[family]}_trajectory.png",
            dpi=args.dpi,
            figure_format=args.format,
        )


if __name__ == "__main__":
    main()
