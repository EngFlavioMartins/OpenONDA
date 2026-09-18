#!/usr/bin/env python3
"""Plot ``delta_wing_wake_vertical.png``."""

if not __package__:
    from pathlib import Path as _CasePath
    from openonda.tutorial_runner import case_package

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"

import argparse
from pathlib import Path

from openonda import plotting as _theme

from .postprocess import FIGURES_DIR, _plot_wake_field, resolve_plot_sources


def plot_wake_vertical(
    samples_arg=None,
    destination=FIGURES_DIR,
    figure_format="png",
    solution_dirs=None,
    *,
    partial=False,
):
    """Export the vertical wake-field figure from native wake-plane samples.

    Parameters
    ----------
    samples_arg : object
        ``None`` to plot the accepted lineage, otherwise sample directories to
        read directly.
    destination : Path
        ``figures/`` or ``figures/partial/`` directory for the export.
    figure_format : str
        Export extension, ``png`` or ``pdf``.
    solution_dirs : list[Path] | None
        Matching solver directories when ``samples_arg`` is explicit.
    partial : bool
        True to plot the latest common native wake timestamp instead of the
        measured-period mean.
    """
    _plot_wake_field(
        samples_arg,
        destination,
        figure_format,
        "vertical",
        solution_dirs=solution_dirs,
        partial=partial,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=_theme.EXPORT_FORMATS, default="png")
    parser.add_argument(
        "--samples",
        type=Path,
        action="append",
        help="sample directory; repeat in sparse-to-dense order for a continuation",
    )
    parser.add_argument(
        "--solution",
        type=Path,
        action="append",
        help="matching solution directory for each --samples directory",
    )
    args = parser.parse_args()
    if args.samples:
        plot_wake_vertical(args.samples, FIGURES_DIR, args.format, args.solution)
        return
    sources = resolve_plot_sources()
    plot_wake_vertical(
        sources.samples_arg,
        sources.destination,
        args.format,
        sources.solution_dirs,
        partial=not sources.complete,
    )


if __name__ == "__main__":
    main()
