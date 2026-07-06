#!/usr/bin/env python3
"""Plot ``quadcopter_vorticity_history.png``."""

import argparse
from pathlib import Path

from _quadcopter_plots import FIGURES_DIR, SOLUTION_DIR, _theme, plot_vorticity_history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution-dir", default=str(SOLUTION_DIR))
    parser.add_argument("--figures-dir", default=str(FIGURES_DIR))
    parser.add_argument("--format", choices=_theme.EXPORT_FORMATS, default="png")
    args = parser.parse_args()
    plot_vorticity_history(Path(args.solution_dir), Path(args.figures_dir), args.format)


if __name__ == "__main__":
    main()
