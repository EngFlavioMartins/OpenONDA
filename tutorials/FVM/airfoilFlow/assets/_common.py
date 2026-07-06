import argparse
import csv
import importlib.util
import os
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

U_INF = 1.0
L_REF = 1.0
RE = 1000.0
NU = U_INF * L_REF / RE


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution-dir", default=str(SOLUTION_DIR))
    parser.add_argument("--figures-dir", default=str(FIGURES_DIR))
    parser.add_argument("--format", choices=THEME.EXPORT_FORMATS, default="png")
    parser.add_argument("--dpi", type=int, default=400)
    return parser


def load_forces_csv(solution_dir):
    csv_path = os.path.join(solution_dir, "forces_history.csv")
    if not os.path.exists(csv_path):
        print(f"  WARNING: forces_history.csv not found at {csv_path}")
        return {}
    data = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pname = row["patch"]
            if pname not in data:
                data[pname] = {k: [] for k in row.keys() if k != "patch"}
            for k in data[pname].keys():
                if k != "patch":
                    try:
                        data[pname][k].append(float(row[k]) if row[k] else 0.0)
                    except ValueError:
                        data[pname][k].append(0.0)
    for pname in data:
        for k in data[pname].keys():
            if k != "patch":
                data[pname][k] = np.array(data[pname][k])
    return data


def save_fig(fig, name, figures_dir, dpi=400, figure_format="png"):
    path = Path(figures_dir) / name
    THEME.save_fig(fig, path, figure_format=figure_format, dpi=dpi)
