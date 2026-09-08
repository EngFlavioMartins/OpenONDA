#!/usr/bin/env python3
"""Plot ``quadcopter_particle_count.png``."""

if not __package__:
    from pathlib import Path as _CasePath
    from openonda.tutorial_runner import case_package

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"


import argparse

from ._quadcopter_plots import FIGURES_DIR, SAMPLES_DIR, _theme, plot_particle_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=_theme.EXPORT_FORMATS, default="png")
    args = parser.parse_args()
    plot_particle_count(SAMPLES_DIR, FIGURES_DIR, args.format)


if __name__ == "__main__":
    main()
