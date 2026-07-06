import argparse
import importlib.util
import os
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET

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
    parser.add_argument("--solution-dir", default=str(SOLUTION_DIR))
    parser.add_argument("--figures-dir", default=str(FIGURES_DIR))
    parser.add_argument("--format", choices=THEME.EXPORT_FORMATS, default="png")
    parser.add_argument("--dpi", type=int, default=400)
    return parser

def load_pvd_timesteps(solution_dir):
    pvd_path = os.path.join(solution_dir, "stepProfile.pvd")
    if not os.path.exists(pvd_path):
        return []
    tree = ET.parse(pvd_path)
    root = tree.getroot()
    timesteps = []
    for ds in root.iter("DataSet"):
        timesteps.append({
            "time": float(ds.get("timestep")),
            "file": os.path.join(solution_dir, ds.get("file"))
        })
    return timesteps

def save_fig(fig, name, figures_dir, dpi=400, figure_format="png"):
    path = Path(figures_dir) / name
    THEME.save_fig(fig, path, figure_format=figure_format, dpi=dpi)
