#!/usr/bin/env python3
"""Plot ``delta_wing_forces.png``."""

import argparse

from _delta_wing_plots import FIGURES_DIR, SAMPLES_DIR, _theme, plot_forces


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=_theme.EXPORT_FORMATS, default="png")
    args = parser.parse_args()
    plot_forces(SAMPLES_DIR, FIGURES_DIR, args.format)


if __name__ == "__main__":
    main()
