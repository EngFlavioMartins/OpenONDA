import argparse
import csv
import glob
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


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution-dir", default=str(SOLUTION_DIR))
    parser.add_argument("--figures-dir", default=str(FIGURES_DIR))
    parser.add_argument("--format", choices=THEME.EXPORT_FORMATS, default="png")
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--angle", type=float, default=0.0)
    return parser


def load_forces_csv(solution_dir):
    """Load solution/forces_history.csv -> {patch: {column: array}}."""
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
            for k, v in row.items():
                if k != "patch":
                    try:
                        data[pname][k].append(float(v) if v else 0.0)
                    except ValueError:
                        data[pname][k].append(0.0)
    for pname in data:
        for k in data[pname]:
            data[pname][k] = np.array(data[pname][k])
    return data


def load_csv_columns(path):
    """Read a CSV with a header row into {column: float array}."""
    path = Path(path)
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return {}
    data = {}
    with open(path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            for key, value in row.items():
                data.setdefault(key, []).append(float(value))
    return {key: np.asarray(vals) for key, vals in data.items()}


def latest_vtu(solution_dir):
    """Path of the last-written .vtu snapshot, or None."""
    files = sorted(glob.glob(os.path.join(solution_dir, "*.vtu")))
    return files[-1] if files else None


def save_fig(fig, name, figures_dir, dpi=400, figure_format="png"):
    path = Path(figures_dir) / name
    THEME.save_fig(fig, path, figure_format=figure_format, dpi=dpi)
