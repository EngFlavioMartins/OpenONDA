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
D_REF = 1.0

# Reference values from Constant et al. 2017 (docs/literature/Constant2016.pdf),
# Tables 2-3, incl. the literature entries they compare against.
REFERENCES = {
    30.0: {"Cd": (1.74, 1.80), "L_over_D": (1.55, 1.70)},
    100.0: {"Cd": (1.35, 1.38), "St": (0.164, 0.165)},
    185.0: {"Cd": (1.29, 1.43), "St": (0.193, 0.199), "Cl_rms": (0.42, 0.46)},
}


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution-dir", default=str(SOLUTION_DIR))
    parser.add_argument("--figures-dir", default=str(FIGURES_DIR))
    parser.add_argument("--format", choices=THEME.EXPORT_FORMATS, default="png")
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--Re", type=float, default=30.0)
    return parser


def load_ibm_forces_csv(solution_dir):
    """Load solution/ibm_forces_history.csv -> {body: {column: array}}."""
    csv_path = os.path.join(solution_dir, "ibm_forces_history.csv")
    if not os.path.exists(csv_path):
        print(f"  WARNING: ibm_forces_history.csv not found at {csv_path}")
        return {}
    data = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["body"]
            if name not in data:
                data[name] = {k: [] for k in row.keys() if k != "body"}
            for k, v in row.items():
                if k != "body":
                    try:
                        data[name][k].append(float(v) if v else 0.0)
                    except ValueError:
                        data[name][k].append(0.0)
    for name in data:
        for k in data[name]:
            data[name][k] = np.array(data[name][k])
    return data


def load_markers(solution_dir):
    path = os.path.join(solution_dir, "ibm_markers.csv")
    if not os.path.exists(path):
        return None
    return np.loadtxt(path, delimiter=",", skiprows=1)


def save_fig(fig, name, figures_dir, dpi=400, figure_format="png"):
    path = Path(figures_dir) / name
    THEME.save_fig(fig, path, figure_format=figure_format, dpi=dpi)
