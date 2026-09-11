#!/usr/bin/env python3
"""CL, CD, CM vs alpha polar for the flat-plate VLM--VPM suite.

Output: figures/plate_polar.png
"""

if not __package__:
    from pathlib import Path as _CasePath
    from openonda.tutorial_runner import case_package

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"


import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ._plot_theme import FIG_DIR, SAMPLES_DIR, centered_subplots_adjust, cm, color, save_fig
from .results import load_forces, parameters
from .theoretical_model import lifting_line_polar

parser = argparse.ArgumentParser()
parser.add_argument("--format", choices=("png", "pdf"), default="png")
parser.add_argument("--dpi", type=int, default=400)
args = parser.parse_args()
FIG_DIR.mkdir(parents=True, exist_ok=True)


C_MOVING = color("TUDcyan")
C_STATIC = color("vpm")
C_THEORY = color("ref")

# -- Physical constants --------------------------------------------------------
physics = parameters(SAMPLES_DIR.parent)
CHORD = physics["chord"]
aspect_ratio = physics["span"] / CHORD
FREESTREAM_SPEED = physics["speed"]

CM = cm()


# -- Helpers -------------------------------------------------------------------


def load_csv(name: str) -> pd.DataFrame | None:
    return load_forces(SAMPLES_DIR.parent, name)


def final_coefficients(name: str) -> tuple[float, float, float]:
    df = load_csv(name)
    if df is None:
        return float("nan"), float("nan"), float("nan")
    if df["nondimensional_distance_travelled"].max() < 5.0:
        print(f"  [SHORT RUN] {name}: insufficient travel for a steady polar")
        return float("nan"), float("nan"), float("nan")
    df = df[
        df["nondimensional_distance_travelled"]
        >= df["nondimensional_distance_travelled"].max() - 5.0
    ]
    cl = float(df["lift_coefficient"].mean())
    cd = float(df["drag_coefficient"].mean())
    cm = float(df["pitching_moment_coefficient_quarter_chord"].mean())
    return cl, cd, cm


# -- Data ----------------------------------------------------------------------

MOVING_CASES = [
    ("exp_moving_aoan10", -10),
    ("exp_moving_aoan05", -5),
    ("exp_moving_aoan02", -2),
    ("exp_moving_aoa00", 0),
    ("exp_moving_aoa02", 2),
    ("exp_moving_aoa05", 5),
    ("exp_moving_aoa08", 8),
    ("exp_moving_aoa10", 10),
    ("exp_moving_aoa12", 12),
    ("exp_moving_aoa15", 15),
]

STATIC_CASES = [
    ("exp_static_aoan10", -10),
    ("exp_static_aoan05", -5),
    ("exp_static_aoan02", -2),
    ("exp_static_aoa00", 0),
    ("exp_static_aoa02", 2),
    ("exp_static_aoa05", 5),
    ("exp_static_aoa08", 8),
    ("exp_static_aoa10", 10),
    ("exp_static_aoa12", 12),
    ("exp_static_aoa15", 15),
]

moving_aoa, moving_cl, moving_cd, moving_cm = [], [], [], []
for name, aoa in MOVING_CASES:
    cl, cd, cm_val = final_coefficients(name)
    moving_aoa.append(aoa)
    moving_cl.append(cl)
    moving_cd.append(cd)
    moving_cm.append(cm_val)

static_aoa, static_cl, static_cd, static_cm = [], [], [], []
for name, aoa in STATIC_CASES:
    cl, cd, cm_val = final_coefficients(name)
    static_aoa.append(aoa)
    static_cl.append(cl)
    static_cd.append(cd)
    static_cm.append(cm_val)

# Theory curves
alpha_range = np.linspace(-12, 18, 300)
CL_theory, CDi_theory = lifting_line_polar(alpha_range, aspect_ratio)

# -- Figure --------------------------------------------------------------------

fig, (ax_cl, ax_cd, ax_cm) = plt.subplots(3, 1, figsize=(12.5 * CM, 14.5 * CM), sharex=True)
centered_subplots_adjust(fig, outer=0.18, bottom=0.085, top=0.87, hspace=0.28)

# CL vs α
ax_cl.plot(
    alpha_range,
    CL_theory,
    "--",
    color=C_THEORY,
    lw=1.0,
    label=rf"Lifting-line (AR={aspect_ratio:.0f})",
)
ax_cl.plot(
    moving_aoa, moving_cl, "o", color=C_MOVING, ms=5, mfc="none", zorder=5, label="Moving plate"
)
ax_cl.plot(static_aoa, static_cl, "s", color=C_STATIC, ms=3, zorder=4, label="Static plate")
ax_cl.axhline(0, color=color("DarkText"), lw=0.5, alpha=0.35)
ax_cl.set_ylabel(r"Lift coefficient, $C_L$")
ax_cl.set_title(r"(a) Lift")
handles, labels = ax_cl.get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.985),
    ncol=3,
    frameon=False,
    handlelength=1.5,
    handletextpad=0.45,
    columnspacing=1.0,
)
ax_cl.set_xlim(-12, 18)

# CD vs α
ax_cd.plot(
    alpha_range,
    CDi_theory,
    "--",
    color=C_THEORY,
    lw=1.0,
    label=rf"Lifting-line (AR={aspect_ratio:.0f})",
)
ax_cd.plot(
    moving_aoa, moving_cd, "o", color=C_MOVING, ms=5, mfc="none", zorder=5, label="Moving plate"
)
ax_cd.plot(static_aoa, static_cd, "s", color=C_STATIC, ms=3, zorder=4, label="Static plate")
ax_cd.axhline(0, color=color("DarkText"), lw=0.5, alpha=0.35)
ax_cd.set_ylabel(r"Drag coefficient, $C_D$")
ax_cd.set_title(r"(b) Induced drag")
ax_cd.set_xlim(-12, 18)

# CM vs α
ax_cm.plot(
    moving_aoa, moving_cm, "o", color=C_MOVING, ms=5, mfc="none", zorder=5, label="Moving plate"
)
ax_cm.plot(static_aoa, static_cm, "s", color=C_STATIC, ms=3, zorder=4, label="Static plate")
ax_cm.axhline(0, color=color("DarkText"), lw=0.5, alpha=0.35)
ax_cm.set_xlabel(r"Angle of attack, $\alpha$ [°]")
ax_cm.set_ylabel(r"Pitching moment, $C_{m,c/4}$")
ax_cm.set_title(r"(c) Quarter-chord moment")
ax_cm.set_xlim(-12, 18)

out = FIG_DIR / "plate_polar.png"
save_fig(fig, out, figure_format=args.format, dpi=args.dpi)
