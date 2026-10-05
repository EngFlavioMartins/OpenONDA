import argparse
from pathlib import Path

from openonda.results import latest_fvm_frame, read_csv_columns, read_grouped_csv

import numpy as np

ASSETS_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = ASSETS_DIR.parent
FIGURES_DIR = SCRIPT_DIR / "figures"
SOLUTION_DIR = SCRIPT_DIR / "solution"


def _load_theme():
    from openonda import plotting as theme

    theme.set_thesis_style()
    return theme


THEME = _load_theme()
COLORS = THEME.COLORS
COLORMAPS = THEME.COLORMAPS
figure_size = THEME.figure_size

FREESTREAM_SPEED = 1.0
L_REF = 1.0
RE = 1000.0


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=THEME.FORMAT_CHOICES, default="both")
    parser.add_argument("--dpi", type=int, default=THEME.DEFAULT_DPI)
    parser.add_argument("--angle", type=float, default=0.0)
    return parser


def load_forces_csv(solution_dir):
    return read_grouped_csv(Path(solution_dir).parent / "samples/forces_history.csv", "patch")


def save_fig(fig, name, figures_dir, dpi=None, figure_format="both"):
    return THEME.save_fig(fig, Path(figures_dir) / name, figure_format=figure_format, dpi=dpi)
