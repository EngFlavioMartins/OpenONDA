import argparse
import csv
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


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=THEME.EXPORT_FORMATS, default="png")
    parser.add_argument("--dpi", type=int, default=THEME.DEFAULT_DPI)
    return parser


def load_csv_columns(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required plotting input missing: {path} not found")
    data = {}
    with open(path) as stream:
        for row in csv.DictReader(stream):
            for key, value in row.items():
                data.setdefault(key, []).append(float(value))
    if not data:
        raise ValueError(f"Required plotting input has no records: {path}")
    return {key: np.asarray(values) for key, values in data.items()}


def save_fig(fig, name, figures_dir, dpi=None, figure_format="png"):
    path = Path(figures_dir) / name
    axes = fig.axes
    fig.tight_layout(pad=1.0)
    THEME.fit_thesis_y_label_margins(fig, axes)
    THEME.validate_thesis_figure(fig, axes)
    output = THEME.figure_path(path, figure_format)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=THEME.DEFAULT_DPI if dpi is None else dpi, bbox_inches=None)
    THEME.plt.close(fig)
    print(f"Saved: {output}")
