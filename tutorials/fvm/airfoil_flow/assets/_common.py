import argparse
import csv
import os
from pathlib import Path

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
    parser.add_argument("--format", choices=THEME.EXPORT_FORMATS, default="png")
    parser.add_argument("--dpi", type=int, default=THEME.DEFAULT_DPI)
    parser.add_argument("--angle", type=float, default=0.0)
    return parser


def load_forces_csv(solution_dir):
    """Load samples/forces_history.csv -> {patch: {column: array}}.

    Sampled output lives in samples/ at the case root, alongside solution/.
    """
    csv_path = os.path.join(os.path.dirname(solution_dir), "samples", "forces_history.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Required plotting input missing: forces_history.csv not found at {csv_path}"
        )
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
    if not data:
        raise ValueError(f"Required plotting input has no records: {csv_path}")
    return data


def load_csv_columns(path):
    """Read a CSV with a header row into {column: float array}."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required plotting input missing: {path} not found")
    data = {}
    with open(path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            for key, value in row.items():
                data.setdefault(key, []).append(float(value))
    if not data:
        raise ValueError(f"Required plotting input has no records: {path}")
    return {key: np.asarray(vals) for key, vals in data.items()}


latest_vtu = THEME.latest_fvm_snapshot


def save_fig(fig, name, figures_dir, dpi=None, figure_format="png"):
    path = Path(figures_dir) / name
    axes = fig.axes
    THEME.fit_thesis_y_label_margins(fig, axes)
    THEME.validate_thesis_figure(fig, axes)
    output = THEME.figure_path(path, figure_format)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=THEME.DEFAULT_DPI if dpi is None else dpi, bbox_inches=None)
    THEME.plt.close(fig)
    print(f"Saved: {output}")
