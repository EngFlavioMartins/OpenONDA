#!/usr/bin/env python3
"""Plot ``delta_wing_forces.png``."""

import argparse
from pathlib import Path

from _delta_wing_plots import FIGURES_DIR, SOLUTION_DIR, plot_forces


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution-dir", default=str(SOLUTION_DIR))
    parser.add_argument("--figures-dir", default=str(FIGURES_DIR))
    args = parser.parse_args()
    plot_forces(Path(args.solution_dir), Path(args.figures_dir))


if __name__ == "__main__":
    main()
