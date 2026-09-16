"""Plot native scalar histories for the vortex-interaction methods."""

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

FIGURE_NAME = "diagnostic_histories"


def plot(runs, output, formats):
    """Render six quantities directly from each native flow-integrals table."""
    plotting = theme()
    plotting.set_thesis_style()
    fig, axes = plt.subplots(3, 2, figsize=figure_size(13.5), sharex=True)
    quantities = (
        ("total_kinetic_energy", r"$E/E_0$", True),
        ("total_enstrophy", r"$Z/Z_0$", True),
        ("vorticity_divergence_error", "Divergence error", False),
        ("vortex_strength_misalignment_degrees", "Misalignment [deg]", False),
        ("lagrangian_cfl", "Lagrangian CFL", False),
        ("n_particles_total", r"$N_p/10^5$", False),
    )
    plotted = 0
    for run in runs:
        path = setup.TUTORIAL_DIR / "samples" / run / "flow_integrals.csv"
        if not path.is_file():
            print(f"Skipping {run}: no flow_integrals.csv", flush=True)
            continue
        frame = pd.read_csv(path)
        style = case_style(run)
        time = frame.time * setup.RING_CIRCULATION / setup.RING_RADIUS**2
        for axis, (column, _label, normalize) in zip(axes.flat, quantities, strict=True):
            values = frame[column] / frame[column].iloc[0] if normalize else frame[column]
            if column == "n_particles_total":
                values = values / 1e5
            axis.plot(
                time,
                values,
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                ms=3,
                markevery=max(1, len(frame) // 12),
                lw=1,
            )
        plotted += 1
    if not plotted:
        plt.close(fig)
        print("No diagnostic histories are available.", flush=True)
        return
    for index, (axis, (_column, label, _normalize)) in enumerate(
        zip(axes.flat, quantities, strict=True)
    ):
        axis.set_ylabel(label)
        label_on_right = index in {0, 1, 4}
        axis.text(
            0.96 if label_on_right else 0.04,
            0.94,
            f"({chr(97 + index)})",
            transform=axis.transAxes,
            ha="right" if label_on_right else "left",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2, "alpha": 0.8},
            zorder=5,
        )
    for axis in axes[-1]:
        axis.set_xlabel(r"$t\Gamma_0/R_0^2$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    comparison_legend(fig, handles, labels, location="bottom")
    plotting.centered_subplots_adjust(
        fig, outer=0.18, bottom=0.25, top=0.97, hspace=0.36, wspace=0.78
    )
    save_figure(fig, output / FIGURE_NAME, axes.flat, formats)
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
