"""Plot ``delta_wing_forces.png``."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from openonda import plotting as _theme

from .postprocess import FIGURES_DIR, _save_figure, force_history


def plot_forces(samples_arg=None, destination=FIGURES_DIR, figure_format="png"):
    "Export the thesis vertical-force, centroid and power-history figure."
    _theme.set_thesis_style()
    data = force_history(samples_arg)
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12.5 * _theme.CM, 10.5 * _theme.CM))
    _theme.centered_subplots_adjust(fig, outer=0.12, bottom=0.13, top=0.89, hspace=0.2)
    for surface, color, label in (
        ("front_wing", _theme.COLORS["teal"], "Front"),
        ("rear_wing", _theme.COLORS["vpm"], "Rear"),
    ):
        rows = data[data.surface == surface]
        for ax, values in zip(axes, (rows.force_z, rows.centroid_z, -rows.power), strict=True):
            ax.plot(rows.time, values, color=color, label=label)
    for ax, label in zip(axes, ("$F_z$ [N]", "$z_c$ [m]", "$P_{\\mathrm{in}}$ [W]"), strict=True):
        ax.set_ylabel(label)
        ax.axhline(0, color=_theme.COLORS["reference"], lw=0.6, linestyle="--")
        ax.locator_params(axis="y", nbins=3)
    axes[-1].set_xlabel("Time [s]")
    axes[0].legend(
        loc="best",
        ncol=2,
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="0.8",
    )
    fig.subplots_adjust(top=0.989)
    _save_figure(fig, axes, destination / "delta_wing_forces.png", figure_format)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=_theme.FORMAT_CHOICES, default="both")
    parser.add_argument(
        "--samples",
        type=Path,
        action="append",
        help="sample directory; repeat in sparse-to-dense order for a continuation",
    )
    args = parser.parse_args()
    plot_forces(args.samples, FIGURES_DIR, args.format)


if __name__ == "__main__":
    main()
