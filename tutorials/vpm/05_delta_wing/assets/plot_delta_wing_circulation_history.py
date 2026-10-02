#!/usr/bin/env python3
"""Plot ``delta_wing_circulation_history.png``."""

if not __package__:
    from pathlib import Path as _CasePath
    from openonda.tutorial_runner import case_package

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from openonda import plotting as _theme

from .postprocess import FIGURES_DIR, _save_figure, flow_integrals, resolve_plot_sources


def plot_circulation(
    samples_arg=None, destination=FIGURES_DIR, figure_format="png", *, partial=False
):
    """Export the sampled vortex-strength history figure.

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
    data = flow_integrals(samples_arg)
    fig, ax = plt.subplots(figsize=(12.5 * _theme.CM, 7.0 * _theme.CM))
    _theme.centered_subplots_adjust(fig, outer=0.101, bottom=0.20, top=0.915)
    ax.plot(data.time, data.vortex_strength_magnitude_sum, color=_theme.COLORS["vpm"])
    ax.set(
        xlabel="Time [s]",
        ylabel=r"$\sum_p |\boldsymbol{\Gamma}_p|$ [m$^3$/s]",
        title="Partial run" if partial else "",
    )
    _save_figure(fig, (ax,), destination / "delta_wing_circulation_history.png", figure_format)


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
    if args.samples:
        plot_circulation(args.samples, FIGURES_DIR, args.format)
        return
    sources = resolve_plot_sources()
    plot_circulation(
        sources.samples_arg, sources.destination, args.format, partial=not sources.complete
    )


if __name__ == "__main__":
    main()
