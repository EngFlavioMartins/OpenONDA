#!/usr/bin/env python3
"""Plot downstream velocity from native wake-plane samples."""

from pathlib import Path

if not __package__:
    from openonda.tutorial_runner import case_package

    __package__ = case_package(Path(__file__).resolve().parents[1]) + ".assets"

import argparse
from ._delta_wing_plots import FIGURES_DIR, _theme, plot_wake

if __name__ == "__main__":
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
    plot_wake(args.samples, FIGURES_DIR, args.format, args.solution)
