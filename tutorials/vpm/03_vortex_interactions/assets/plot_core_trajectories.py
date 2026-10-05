"""Plot field-core trajectories against the digitized LBM reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


from openonda.results import read_csv_table, write_csv_table, write_text

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from .. import setup
from .postprocess import (
    case_style,
    comparison_legend,
    core_peak_history,
    figure_size,
    load_metadata,
    save_figure,
    theme,
    track_core_pair,
)

FIGURE_NAME = "core_trajectories"


def plot(runs, output, auxiliary_output, formats, merge_bridge, bridge_limit):
    """Render one trajectory comparison from saved meridional fields."""
    plotting = theme()
    plotting.set_thesis_style()
    reference_path = Path(__file__).parent / "references/leapfrogging_lbm_trajectory.csv"
    reference = pd.DataFrame(read_csv_table(reference_path))
    fig, ax = plt.subplots(figsize=figure_size(7.0))
    for core in (1, 2):
        values = reference[reference.ring == core]
        ax.plot(
            values.x_over_R0 - 2.5,
            values.R_over_R0,
            color=plotting.COLORS["reference"],
            lw=1.0,
            linestyle="--" if core == 1 else ":",
        )
    plotted = []
    sources = []
    for run in runs:
        metadata = load_metadata(run)
        peaks, fields = core_peak_history(run, merge_bridge)
        tracks, termination = track_core_pair(peaks, bridge_limit)
        if tracks.empty:
            print(f"Skipping {run}: no separated core pair", flush=True)
            continue
        write_csv_table(
            auxiliary_output / f"{FIGURE_NAME}_{run}_peaks.csv",
            peaks.itertuples(index=False, name=None),
            columns=peaks.columns,
        )
        write_csv_table(
            auxiliary_output / f"{FIGURE_NAME}_{run}_tracks.csv",
            tracks.itertuples(index=False, name=None),
            columns=tracks.columns,
        )
        style = case_style(run)
        for core in (1, 2):
            values = tracks[tracks.core == core]
            ax.plot(
                values.x,
                values.radius,
                color=style["color"],
                marker=style["marker"],
                ms=3.5,
                markevery=max(1, len(values) // 12),
                lw=1.1,
                linestyle="-",
                markerfacecolor=style["color"] if core == 1 else "white",
            )
        plotted.append(run)
        sources.append(
            {
                "run": run,
                "status": metadata["run_status"]["status"],
                "tracking_termination": termination,
                "tracked_until": float(tracks.time.max()),
                "fields": fields,
            }
        )
    ax.set(xlabel="$x/R_0$", ylabel="$R/R_0$", xlim=(-0.6, 7.4), ylim=(0.55, 1.48))
    handles = [
        Line2D(
            [0],
            [0],
            color=case_style(run)["color"],
            marker=case_style(run)["marker"],
            ms=4,
            lw=1.1,
            label=case_style(run)["label"],
        )
        for run in plotted
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color=plotting.COLORS["reference"],
            linestyle="--",
            lw=1,
            label="LBM (Cheng et al., 2015)",
        )
    )
    comparison_legend(fig, handles, location="bottom")
    plotting.centered_subplots_adjust(fig, outer=0.113, bottom=0.45, top=0.96)
    figure_path = output / FIGURE_NAME
    save_figure(fig, figure_path, ax, formats)
    plt.close(fig)
    plot_information = {
        "figure": FIGURE_NAME,
        "generator": Path(__file__).name,
        "reference": str(reference_path.relative_to(setup.TUTORIAL_DIR)),
        "core_definition": "positive curl(u)_z maxima on the saved z=0, y>=0 plane",
        "runs": sources,
        "exports": [{"file": f"{FIGURE_NAME}.{figure_format}"} for figure_format in formats],
    }
    write_text(
        auxiliary_output / f"{FIGURE_NAME}.json",
        json.dumps(plot_information, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--merge-bridge", type=float, default=0.9)
    parser.add_argument("--bridge-limit", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=setup.TUTORIAL_DIR / "figures")
    parser.add_argument(
        "--auxiliary-output", type=Path, default=setup.TUTORIAL_DIR / "figures/auxiliary"
    )
    parser.add_argument("--format", choices=("png", "pdf", "both"), default="both")
    args = parser.parse_args()
    formats = ("pdf", "png") if args.format == "both" else (args.format,)
    plot(
        args.runs, args.output, args.auxiliary_output, formats, args.merge_bridge, args.bridge_limit
    )


if __name__ == "__main__":
    main()
