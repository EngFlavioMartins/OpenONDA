#!/usr/bin/env python3
"""Plot quadcopter performance from native solver samples."""

if not __package__:
    from pathlib import Path
    from openonda.tutorial_runner import case_package

    __package__ = case_package(Path(__file__).resolve().parents[1]) + ".assets"

import argparse
from ._quadcopter_plots import FIGURES_DIR, SAMPLES_DIR, _theme, plot_performance

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=_theme.EXPORT_FORMATS, default="png")
    plot_performance(SAMPLES_DIR, FIGURES_DIR, parser.parse_args().format)
