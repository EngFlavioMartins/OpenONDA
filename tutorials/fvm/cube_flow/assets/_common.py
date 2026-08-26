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

FREESTREAM_SPEED = 1.0
D_REF = 1.0

# Square cylinder, Re = 100, blockage 5%.
# Okajima, J. Fluid Mech. 123 (1982): experimental St ~ 0.14.
# Sohankar, Norberg & Davidson, IJNMF 26 (1998): St = 0.146, Cd = 1.48.
# Sen, Mittal & Biswas, IJNMF 67 (2011): St = 0.145, Cd = 1.53.
REFERENCES = {
    100.0: {
        "drag_coefficient": (1.45, 1.58),
        "strouhal_number": (0.140, 0.150),
    },
}


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=THEME.EXPORT_FORMATS, default="png")
    parser.add_argument("--dpi", type=int, default=THEME.DEFAULT_DPI)
    parser.add_argument("--Re", type=float, default=100.0)
    return parser


def load_forces_csv(solution_dir):
    """Load samples/forces_history.csv -> {patch: {column: array}}.

    Sampled output lives in samples/ at the case root, alongside solution/.
    """
    csv_path = os.path.join(os.path.dirname(solution_dir), "samples", "forces_history.csv")
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


def latest_vtu(solution_dir):
    """Path of the last-written .vtu snapshot, or None."""
    files = sorted(glob.glob(os.path.join(solution_dir, "*.vtu")))
    return files[-1] if files else None


def strouhal_from_lift(t, cl):
    """Dominant lift frequency (Hz) from the second half of the signal."""
    n = len(t)
    if n < 32:
        return None
    t2, cl2 = t[n // 2 :], cl[n // 2 :]
    if np.ptp(cl2) < 1e-6:
        return None
    tu = np.linspace(t2[0], t2[-1], len(t2))
    clu = np.interp(tu, t2, cl2)
    clu -= clu.mean()
    freqs = np.fft.rfftfreq(len(tu), tu[1] - tu[0])
    amp = np.abs(np.fft.rfft(clu))
    if amp[1:].max() < 1e-8:
        return None
    return float(freqs[1:][np.argmax(amp[1:])])


def save_fig(fig, name, figures_dir, dpi=None, figure_format="png"):
    path = Path(figures_dir) / name
    THEME.save_fig(fig, path, figure_format=figure_format, dpi=dpi)
