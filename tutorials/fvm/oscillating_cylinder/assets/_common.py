"""Paths, plot style and output readers shared by the figures of this case."""

import argparse
import csv
from pathlib import Path

from defusedxml import ElementTree
import numpy as np

from openonda import plotting as THEME

ASSETS_DIR = Path(__file__).resolve().parent
CASE_DIR = ASSETS_DIR.parent
FIGURES_DIR = CASE_DIR / "figures"
SAMPLES_DIR = CASE_DIR / "samples"
SOLUTION_DIR = CASE_DIR / "solution"

THEME.set_thesis_style()
COLORS = THEME.COLORS
figure_size = THEME.figure_size


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=THEME.FORMAT_CHOICES, default="both")
    parser.add_argument("--dpi", type=int, default=THEME.DEFAULT_DPI)
    return parser


def read_force_history(patch):
    """Return the force-history columns of one patch as arrays."""
    columns = {}
    with (SAMPLES_DIR / "forces_history.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["patch"] == patch:
                for name, value in row.items():
                    if name != "patch":
                        columns.setdefault(name, []).append(float(value))
    return {name: np.asarray(values) for name, values in columns.items()}


def latest_fvm_frame():
    """Return ``(time, path)`` of the latest saved volume-field frame."""
    collection = SOLUTION_DIR / "fvm.pvd"
    frames = sorted(
        (float(item.attrib["timestep"]), collection.parent / item.attrib["file"])
        for item in ElementTree.parse(collection).iter("DataSet")
    )
    return frames[-1]


def save_fig(fig, name, figure_format="both", dpi=None):
    return THEME.save_fig(fig, FIGURES_DIR / name, figure_format=figure_format, dpi=dpi)
