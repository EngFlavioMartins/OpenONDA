"""Spanwise lift distribution cl(y) for moving and static plates at AoA=5 deg.

Output: figures/plate_spanwise.png
"""

from __future__ import annotations

import argparse


from openonda.results import read_csv_table

import numpy as np
import pandas as pd

from ._plot_theme import (
    FIG_DIR,
    SAMPLES_DIR,
    color,
    save_fig,
    validation_legend,
    validation_subplots,
)
from .results import parameters
from .theoretical_model import spanwise_reference

parser = argparse.ArgumentParser(description="Flat plate spanwise lift distribution")
parser.add_argument("--format", choices=("png", "pdf", "both"), default="both")
parser.add_argument("--dpi", type=int, default=400)
args = parser.parse_args()
C_MOVING = color("teal")
C_STATIC = color("vpm")
C_LL = color("reference")
physics = parameters(SAMPLES_DIR.parent)
CHORD = physics["chord"]
SPAN = physics["span"]
FREESTREAM_SPEED = physics["speed"]
ANGLE_OF_ATTACK = 5.0
angle_of_attack_radians = np.radians(ANGLE_OF_ATTACK)
y_theory = np.linspace(-SPAN / 2, SPAN / 2, 400)
df_ll = spanwise_reference(
    "lifting_line",
    y_theory,
    SPAN,
    CHORD,
    angle_of_attack_radians,
    FREESTREAM_SPEED,
    n_fourier_terms=80,
)
cl_ll = df_ll["section_lift_coefficient"].to_numpy()
y_ll_over_b = df_ll["span_coordinate_normalized"].to_numpy()


def load_spanwise_csv(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    csv = SAMPLES_DIR / name / "vlm_spanwise_flat_plate.csv"
    df = pd.DataFrame(read_csv_table(csv))
    df = df[df["step"] == df["step"].max()].sort_values("span_coordinate").reset_index(drop=True)
    y = df["span_coordinate"].to_numpy()
    cl = df["section_lift_coefficient"].to_numpy()
    y_over_b = 2.0 * y / SPAN
    return (y, cl, y_over_b)


moving_data = load_spanwise_csv("exp_moving_aoa05")
static_data = load_spanwise_csv("exp_static_aoa05")
fig, axes = validation_subplots(
    1, height_cm=5.6, outer=0.1, top_padding_cm=0.2, bottom_padding_cm=2.15
)
ax = axes[0]
y_m, cl_m, yob_m = moving_data
ax.plot(
    yob_m,
    cl_m,
    color=C_MOVING,
    lw=1.5,
    marker="o",
    ms=5,
    markerfacecolor="none",
    label="Moving",
)
y_s, cl_s, yob_s = static_data
ax.plot(yob_s, cl_s, color=C_STATIC, lw=1.5, marker="s", ms=3, label="Static")
ax.plot(y_ll_over_b, cl_ll, "--", color=C_LL, lw=1.0, label="Lifting-line")
ax.set_xlabel("Spanwise position, $2y/b$")
ax.set_ylabel("Sectional lift, $c_\\ell$")
ax.set_xlim(-1, 1)
ax.margins(y=0.08)
validation_legend(fig, ax, ncol=3, outside=True)
print("Final sampled loading compared with rectangular-wing lifting-line theory")
out = FIG_DIR / "plate_spanwise.png"
save_fig(fig, out, figure_format=args.format, dpi=args.dpi)
