import argparse
from pathlib import Path

from openonda.results import read_csv_columns, read_grouped_csv

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
D_REF = 1.0

# Reference values from Constant et al. 2017 (docs/literature/Constant2016.pdf),
# Tables 2-3, incl. the literature entries they compare against.
REFERENCES = {
    30.0: {"drag_coefficient": (1.74, 1.80), "L_over_D": (1.55, 1.70)},
    100.0: {
        "drag_coefficient": (1.35, 1.38),
        "strouhal_number": (0.164, 0.165),
    },
    185.0: {
        "drag_coefficient": (1.29, 1.43),
        "strouhal_number": (0.193, 0.199),
        "lift_coefficient_rms": (0.42, 0.46),
    },
}


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=THEME.FORMAT_CHOICES, default="both")
    parser.add_argument("--dpi", type=int, default=THEME.DEFAULT_DPI)
    parser.add_argument("--Re", type=float, default=30.0)
    return parser


def load_ibm_forces_csv(solution_dir):
    return read_grouped_csv(Path(solution_dir).parent / "samples/ibm_forces_history.csv", "body_id")


def load_markers(solution_dir):
    data = read_csv_columns(Path(solution_dir) / "ibm_markers.csv")
    return np.column_stack([data[f"position_{axis}"] for axis in "xyz"])


def save_fig(fig, name, figures_dir, dpi=None, figure_format="both"):
    return THEME.save_fig(fig, Path(figures_dir) / name, figure_format=figure_format, dpi=dpi)
