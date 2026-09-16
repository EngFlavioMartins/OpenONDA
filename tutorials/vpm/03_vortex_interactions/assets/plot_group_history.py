"""Plot native group-centroid and group-radius histories."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

if not __package__:
    from openonda.tutorial_runner import case_package
    from pathlib import Path as _CasePath

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"

from .. import setup
from .postprocess import case_style, comparison_legend, figure_size, save_figure, theme

FIGURE_NAME = "group_history"


def plot(runs, output, formats):
    """Render strength-weighted histories written by RingDiagnosticsSampler."""
    plotting = theme()
    plotting.set_thesis_style()
    fig, axes = plt.subplots(2, 1, figsize=figure_size(8.3), sharex=True)
    plotted = 0
    for run in runs:
        path = setup.TUTORIAL_DIR / "samples" / run / "ring_diagnostics.csv"
        if not path.is_file():
            print(f"Skipping {run}: no ring_diagnostics.csv", flush=True)
            continue
        frame = pd.read_csv(path)
        style = case_style(run)
        for group, linestyle in ((1, "-"), (0, "--")):
            values = frame[frame.group_id == group].sort_values("time")
            time = values.time * setup.RING_CIRCULATION / setup.RING_RADIUS**2
            line = {
                "color": style["color"],
                "marker": style["marker"],
                "ms": 3,
                "markevery": max(1, len(values) // 12),
                "linestyle": linestyle,
                "lw": 1,
            }
            axes[0].plot(
                time,
                values.vortex_centroid_x / setup.RING_RADIUS,
                label=style["label"] if group == 1 else None,
                **line,
            )
            axes[1].plot(time, values.major_radius / setup.RING_RADIUS, **line)
        plotted += 1
    if not plotted:
        plt.close(fig)
        print("No native group histories are available.", flush=True)
        return
    axes[0].set_ylabel(r"$\bar{x}_g/R_0$")
    axes[1].set_ylabel(r"$R_g/R_0$")
    axes[1].set_xlabel(r"$t\Gamma_0/R_0^2$")
    handles, labels = axes[0].get_legend_handles_labels()
    comparison_legend(fig, handles, labels, location="bottom")
    plotting.centered_subplots_adjust(fig, outer=0.19, bottom=0.32, top=0.96, hspace=0.18)
    save_figure(fig, output / FIGURE_NAME, axes, formats)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--output", type=Path, default=setup.TUTORIAL_DIR / "figures")
    parser.add_argument("--format", choices=("png", "pdf", "both"), default="png")
    args = parser.parse_args()
    formats = ("pdf", "png") if args.format == "both" else (args.format,)
    plot(args.runs, args.output, formats)


if __name__ == "__main__":
    main()
