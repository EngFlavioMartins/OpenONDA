"""Plot quadcopter wake from native solver samples."""

import argparse

from ._quadcopter_plots import FIGURES_DIR, SAMPLES_DIR, _theme, plot_wake

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=_theme.FORMAT_CHOICES, default="both")
    plot_wake(SAMPLES_DIR, FIGURES_DIR, parser.parse_args().format)
