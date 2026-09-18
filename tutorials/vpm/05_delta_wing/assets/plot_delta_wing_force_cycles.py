#!/usr/bin/env python3
"""Plot ``delta_wing_force_cycles.png``."""

if not __package__:
    from pathlib import Path as _CasePath
    from openonda.tutorial_runner import case_package

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from openonda import plotting as _theme

from .postprocess import (
    FIGURES_DIR,
    _available_period,
    _save_figure,
    force_history,
    last_cycles,
    resolve_plot_sources,
)


def plot_force_cycles(
    samples_arg=None, destination=FIGURES_DIR, figure_format="png", *, partial=False
):
    """Export the last three complete measured heave cycles, separated by wing.

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
    period = _available_period(data)
    if period is None:
        print("Cycle comparison skipped: fewer than two sampled heave-velocity peaks.")
        return
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12.5 * _theme.CM, 8.3 * _theme.CM))
    _theme.centered_subplots_adjust(fig, outer=0.16, bottom=0.16, top=0.80, hspace=0.27)
    for ax, surface, color, label in zip(
        axes,
        ("front_wing", "rear_wing"),
        (_theme.COLORS["TUDcyan"], _theme.COLORS["VPMpurple"]),
        ("Front", "Rear"),
        strict=True,
    ):
        for index, (cycle, phase, tail) in enumerate(
            last_cycles(data[data.surface == surface], period)
        ):
            ax.plot(
                phase,
                tail.force_z,
                color=color,
                ls=(":", "--", "-")[index],
                label=f"Cycle {cycle + 1}",
            )
        ax.set_ylabel(r"$F_z$ [N]")
        ax.set_title(label, loc="left", pad=2)
        ax.locator_params(axis="y", nbins=3)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3)
    axes[-1].set_xlabel("Cycle phase")
    if partial:
        axes[0].set_title("Front (partial run)", loc="left", pad=2)
    _save_figure(fig, axes, destination / "delta_wing_force_cycles.png", figure_format)


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
        plot_force_cycles(args.samples, FIGURES_DIR, args.format)
        return
    sources = resolve_plot_sources()
    plot_force_cycles(
        sources.samples_arg, sources.destination, args.format, partial=not sources.complete
    )


if __name__ == "__main__":
    main()
