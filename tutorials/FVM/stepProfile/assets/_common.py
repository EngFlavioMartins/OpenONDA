import argparse
import csv
import importlib.util
from pathlib import Path

import numpy as np

ASSETS_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = ASSETS_DIR.parent
FIGURES_DIR = SCRIPT_DIR / "figures"
SOLUTION_DIR = SCRIPT_DIR / "solution"
THEME_PATH = SCRIPT_DIR.parents[2] / "docs" / "themes" / "matplotlib_setup.py"


def _load_theme():
    spec = importlib.util.spec_from_file_location("openonda_matplotlib_setup", THEME_PATH)
    theme = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(theme)
    theme.set_style()
    return theme


THEME = _load_theme()
COLORS = THEME.COLORS
COLORMAPS = THEME.COLORMAPS
figure_size = THEME.figure_size

# Armaly, Durst, Pereira & Schoenung, J. Fluid Mech. 127 (1983), fig. 4.
# Their Reynolds number uses the inlet-channel hydraulic diameter D = 2h and
# the mean inlet velocity, so Re_Armaly = 2 * Re_h (this case's definition).
# Measured primary reattachment at Re_Armaly = 150: x1/S = 4.2 (expansion
# ratio 1.94 vs 2.0 here).  Keyed by Re_h.
REFERENCES = {75.0: {"x_r": (3.9, 4.5)}}


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution-dir", default=str(SOLUTION_DIR))
    parser.add_argument("--figures-dir", default=str(FIGURES_DIR))
    parser.add_argument("--format", choices=THEME.EXPORT_FORMATS, default="png")
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--Re", type=float, default=75.0)
    return parser


def load_csv_columns(path):
    path = Path(path)
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return {}
    data = {}
    with open(path) as stream:
        for row in csv.DictReader(stream):
            for key, value in row.items():
                data.setdefault(key, []).append(float(value))
    return {key: np.asarray(values) for key, values in data.items()}


def save_fig(fig, name, figures_dir, dpi=400, figure_format="png"):
    path = Path(figures_dir) / name
    THEME.save_fig(fig, path, figure_format=figure_format, dpi=dpi)
