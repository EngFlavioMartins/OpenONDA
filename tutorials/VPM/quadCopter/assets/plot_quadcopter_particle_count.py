#!/usr/bin/env python3
"""Plot ``quadcopter_particle_count.png``."""

import argparse
from pathlib import Path

from _quadcopter_plots import FIGURES_DIR, SOLUTION_DIR, plot_particle_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution-dir", default=str(SOLUTION_DIR))
    parser.add_argument("--figures-dir", default=str(FIGURES_DIR))
    args = parser.parse_args()
    plot_particle_count(Path(args.solution_dir), Path(args.figures_dir))


if __name__ == "__main__":
    main()
