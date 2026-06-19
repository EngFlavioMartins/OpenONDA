import argparse
import os
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET

ASSETS_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = ASSETS_DIR.parent
FIGURES_DIR = SCRIPT_DIR / "figures"
SOLUTION_DIR = SCRIPT_DIR / "solution"

def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution-dir", default=str(SOLUTION_DIR))
    parser.add_argument("--figures-dir", default=str(FIGURES_DIR))
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

def save_fig(fig, name, figures_dir, dpi=400):
    path = os.path.join(figures_dir, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  Saved: {path}")
    import matplotlib.pyplot as plt
    plt.close(fig)
