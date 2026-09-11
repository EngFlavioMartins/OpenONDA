#!/usr/bin/env python3
"""Plot ``delta_wing_circulation_history.png``."""

if not __package__:
    from pathlib import Path as _CasePath
    from openonda.tutorial_runner import case_package

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"


import argparse
from pathlib import Path

from ._delta_wing_plots import FIGURES_DIR, _theme, plot_circulation


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
    plot_circulation(args.samples, FIGURES_DIR, args.format)


if __name__ == "__main__":
    main()
