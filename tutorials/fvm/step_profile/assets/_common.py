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


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=THEME.EXPORT_FORMATS, default="png")
    parser.add_argument("--dpi", type=int, default=THEME.DEFAULT_DPI)
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


def save_fig(fig, name, figures_dir, dpi=None, figure_format="png"):
    path = Path(figures_dir) / name
    THEME.save_fig(fig, path, figure_format=figure_format, dpi=dpi)
