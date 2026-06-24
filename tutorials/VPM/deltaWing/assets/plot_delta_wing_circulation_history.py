#!/usr/bin/env python3
"""Plot ``delta_wing_circulation_history.png``."""

import argparse
from pathlib import Path

from _delta_wing_plots import FIGURES_DIR, SOLUTION_DIR, plot_circulation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution-dir", default=str(SOLUTION_DIR))
    parser.add_argument("--figures-dir", default=str(FIGURES_DIR))
    parser.add_argument("--pattern", default="vpm_wing_*.h5")
    args = parser.parse_args()
    plot_circulation(Path(args.solution_dir), Path(args.figures_dir), args.pattern)


if __name__ == "__main__":
    main()
