#!/usr/bin/env python3
"""Plot ``delta_wing_forces.png``."""

if not __package__:
    from pathlib import Path as _CasePath
    from openonda.tutorial_runner import case_package

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from openonda import plotting as _theme

from .postprocess import FIGURES_DIR, _save_figure, force_history, resolve_plot_sources


def plot_forces(samples_arg=None, destination=FIGURES_DIR, figure_format="png", *, partial=False):
    """Export the thesis vertical-force, centroid and power-history figure.

    Parameters
    ----------
    samples_arg : object
        ``None`` to plot the accepted lineage, otherwise sample directories to
        read directly.
    destination : Path
        ``figures/`` or ``figures/partial/`` directory for the export.
    figure_format : str
        Export extension, ``png`` or ``pdf``.
    partial : bool
        True to label the figure as a partial-run diagnostic.
    """
    _theme.set_thesis_style()
    data = force_history(samples_arg)
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12.5 * _theme.CM, 10.5 * _theme.CM))
    _theme.centered_subplots_adjust(fig, outer=0.16, bottom=0.13, top=0.89, hspace=0.20)
    for surface, color, label in (
        ("front_wing", _theme.COLORS["TUDcyan"], "Front"),
        ("rear_wing", _theme.COLORS["VPMpurple"], "Rear"),
    ):
        rows = data[data.surface == surface]
        for ax, values in zip(axes, (rows.force_z, rows.centroid_z, -rows.power), strict=True):
            ax.plot(rows.time, values, color=color, label=label)
    for ax, label in zip(axes, (r"$F_z$ [N]", r"$z_c$ [m]", r"$P_{\mathrm{in}}$ [W]"), strict=True):
        ax.set_ylabel(label)
        ax.axhline(0, color=_theme.COLORS["RefGray"], lw=0.4)
        ax.locator_params(axis="y", nbins=3)
    axes[-1].set_xlabel("Time [s]")
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.0 if not partial else 0.96),
    )
    if partial:
        fig.suptitle("Partial run", y=0.98)
        # Leave a separate line for the shared legend.
        fig.subplots_adjust(top=0.83)
    _save_figure(fig, axes, destination / "delta_wing_forces.png", figure_format)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=_theme.EXPORT_FORMATS, default="png")
    parser.add_argument(
        "--samples",
        type=Path,
        action="append",
        help="sample directory; repeat in sparse-to-dense order for a continuation",
    )
    args = parser.parse_args()
    if args.samples:
        plot_forces(args.samples, FIGURES_DIR, args.format)
        return
    sources = resolve_plot_sources()
    plot_forces(sources.samples_arg, sources.destination, args.format, partial=not sources.complete)


if __name__ == "__main__":
    main()
