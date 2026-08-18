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

FREESTREAM_SPEED = 1.0
L_REF = 1.0


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution-dir", default=str(SOLUTION_DIR))
    parser.add_argument("--figures-dir", default=str(FIGURES_DIR))
    parser.add_argument("--format", choices=THEME.EXPORT_FORMATS, default="png")
    parser.add_argument("--dpi", type=int, default=THEME.DEFAULT_DPI)
    parser.add_argument("--Re", type=float, default=1e4)
    return parser


def blasius_solution(eta_max=10.0, n_steps=2000):
    """Blasius similarity solution by RK4 integration.

    Solves f''' + 0.5 f f'' = 0 with f(0) = f'(0) = 0 and the classical
    shooting value f''(0) = 0.332057 (Schlichting, Boundary-Layer Theory).

    Returns:
        (eta, fprime): arrays with u/U = f'(eta).
    """

    def rhs(state):
        f, fp, fpp = state
        return np.array([fp, fpp, -0.5 * f * fpp])

    h = eta_max / n_steps
    state = np.array([0.0, 0.0, 0.332057])
    eta = np.linspace(0.0, eta_max, n_steps + 1)
    fprime = np.zeros(n_steps + 1)
    for i in range(n_steps):
        k1 = rhs(state)
        k2 = rhs(state + 0.5 * h * k1)
        k3 = rhs(state + 0.5 * h * k2)
        k4 = rhs(state + h * k3)
        state = state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        fprime[i + 1] = state[1]
    return eta, fprime


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


def save_fig(fig, name, figures_dir, dpi=None, figure_format="png"):
    path = Path(figures_dir) / name
    THEME.save_fig(fig, path, figure_format=figure_format, dpi=dpi)
