"""Plot field-core trajectories against the digitized LBM reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

if not __package__:
    from openonda.tutorial_runner import case_package
    from pathlib import Path as _CasePath

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"

from .. import setup
from .postprocess import (
    case_style,
    comparison_legend,
    core_peak_history,
    figure_size,
    file_sha256,
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
    output.mkdir(parents=True, exist_ok=True)
    auxiliary_output.mkdir(parents=True, exist_ok=True)

    reference_path = Path(__file__).parent / "references/leapfrogging_lbm_trajectory.csv"
    reference = pd.read_csv(reference_path)
    fig, ax = plt.subplots(figsize=figure_size(7.0))
    for core in (1, 2):
        values = reference[reference.ring == core]
        ax.plot(
            values.x_over_R0 - 2.5,
            values.R_over_R0,
            color=plotting.COLORS["RefGray"],
            lw=1.0,
            linestyle="-" if core == 1 else "--",
        )

    plotted = []
    sources = []
    for run in runs:
        metadata = load_metadata(run)
        if not metadata:
            print(f"Skipping {run}: no native metadata", flush=True)
            continue
        try:
            peaks, fields = core_peak_history(run, merge_bridge)
        except ValueError as error:
            print(f"Skipping {run}: {error}", flush=True)
            continue
        tracks, termination = track_core_pair(peaks, bridge_limit)
        if tracks.empty:
            print(f"Skipping {run}: no separated core pair", flush=True)
            continue
        peaks.to_csv(auxiliary_output / f"{FIGURE_NAME}_{run}_peaks.csv", index=False)
        tracks.to_csv(auxiliary_output / f"{FIGURE_NAME}_{run}_tracks.csv", index=False)
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
                linestyle="-" if core == 1 else "--",
            )
        plotted.append(run)
        sources.append(
            {
                "run": run,
                "status": metadata.get("lifecycle", {}).get("status", "unknown"),
                "tracking_termination": termination,
                "tracked_until": float(tracks.time.max()),
                "fields": fields,
            }
        )

    if not plotted:
        plt.close(fig)
        print("No saved core trajectories are available.", flush=True)
        return
    ax.set(xlabel=r"$x/R_0$", ylabel=r"$R/R_0$", xlim=(-0.6, 7.4), ylim=(0.55, 1.48))
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
            color=plotting.COLORS["RefGray"],
            lw=1,
            label="LBM (Cheng et al., 2015)",
        )
    )
    comparison_legend(fig, handles, location="bottom")
    plotting.centered_subplots_adjust(fig, outer=0.18, bottom=0.40, top=0.96)
    figure_path = output / FIGURE_NAME
    save_figure(fig, figure_path, ax, formats)
    plt.close(fig)

    manifest = {
        "figure": FIGURE_NAME,
        "generator": Path(__file__).name,
        "generator_sha256": file_sha256(__file__),
        "reference": str(reference_path.relative_to(setup.TUTORIAL_DIR)),
        "reference_sha256": file_sha256(reference_path),
        "core_definition": "positive curl(u)_z maxima on the saved z=0, y>=0 plane",
        "runs": sources,
        "exports": [
            {
                "file": f"{FIGURE_NAME}.{figure_format}",
                "sha256": file_sha256(output / f"{FIGURE_NAME}.{figure_format}"),
            }
            for figure_format in formats
        ],
    }
    (auxiliary_output / f"{FIGURE_NAME}.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
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
    parser.add_argument("--format", choices=("png", "pdf", "both"), default="png")
    args = parser.parse_args()
    if not 0 < args.bridge_limit < args.merge_bridge < 1:
        parser.error("require 0 < bridge-limit < merge-bridge < 1")
    formats = ("pdf", "png") if args.format == "both" else (args.format,)
    plot(
        args.runs,
        args.output,
        args.auxiliary_output,
        formats,
        args.merge_bridge,
        args.bridge_limit,
    )


if __name__ == "__main__":
    main()
