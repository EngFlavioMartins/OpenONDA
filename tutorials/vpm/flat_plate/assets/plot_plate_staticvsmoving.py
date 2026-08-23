#!/usr/bin/env python3
"""
Flat Plate VLM-VPM — Static vs. Moving Comparison Plotter
==========================================================
Generates Figure 2: CL and CD histories (vs chord-lengths travelled)
comparing the static wind-frame and moving body-frame solutions at
α = 5°, alongside Wagner impulsive-start theory for aspect_ratio = 10.

Output:
    figures/exp_static_vs_moving.png

Author:  Flavio A. C. Martins, OpenONDA Team
Date: June 2026
"""

import sys
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -- Paths ---------------------------------------------------------------------
CASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = CASE_DIR.parents[2]
THEME_PATH = REPO_ROOT / "docs" / "themes" / "matplotlib_setup.py"
SAMPLES_DIR = CASE_DIR / "samples"
FIG_DIR = CASE_DIR / "figures"
parser = argparse.ArgumentParser()
parser.add_argument("--format", choices=("png", "pdf"), default="png")
parser.add_argument("--dpi", type=int, default=300)
args = parser.parse_args()
FIG_DIR.mkdir(parents=True, exist_ok=True)
# -- Theme ---------------------------------------------------------------------
m = None
if not THEME_PATH.exists():
    raise FileNotFoundError(f"OpenONDA matplotlib theme not found: {THEME_PATH}")
spec = importlib.util.spec_from_file_location("matplotlib_setup", str(THEME_PATH))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.set_style()

from theoretical_model import prandtl_finite_span_lift_curve_slope


def _c(key):
    return m.COLORS[key]


# -- Colours -------------------------------------------------------------------
C_MOVING = _c("TUDcyan")
C_STATIC = _c("vpm")
C_THEORY = _c("ref")

# -- Physical constants --------------------------------------------------------
aspect_ratio = 10.0
CHORD = 1.0
FREESTREAM_SPEED = 10.0
finite_span_lift_curve_slope = prandtl_finite_span_lift_curve_slope(aspect_ratio)

cm = m.CM


# -- Helpers -------------------------------------------------------------------


def load_csv(name: str) -> pd.DataFrame | None:
    csv = SAMPLES_DIR / name / f"{name}.csv"
    if not csv.exists():
        print(f"  [MISSING] {csv}")
        return None
    return pd.read_csv(csv)


# -- Data ----------------------------------------------------------------------

df_static = load_csv("exp_static_aoa05")
df_moving = load_csv("exp_moving_aoa05")

steady_lift_coefficient = finite_span_lift_curve_slope * np.sin(np.radians(5.0))

# -- Figure --------------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12 * cm, 12 * cm), sharex=True)
fig.subplots_adjust(hspace=0.1, left=0.12, right=0.95, bottom=0.12, top=0.92)

if df_static is not None:
    ax1.plot(
        df_static["nondimensional_distance_travelled"],
        df_static["lift_coefficient"],
        color=C_STATIC,
        lw=1.5,
        label="Static (wind frame)",
    )
    ax2.plot(
        df_static["nondimensional_distance_travelled"],
        df_static["drag_coefficient"],
        color=C_STATIC,
        lw=1.5,
        label="Static (wind frame)",
    )

if df_moving is not None:
    ax1.plot(
        df_moving["nondimensional_distance_travelled"],
        df_moving["lift_coefficient"],
        color=C_MOVING,
        lw=1.5,
        label="Moving (body frame)",
    )
    ax2.plot(
        df_moving["nondimensional_distance_travelled"],
        df_moving["drag_coefficient"],
        color=C_MOVING,
        lw=1.5,
        label="Moving (body frame)",
    )

    _A1, _b1, _A2, _b2 = 0.165, 0.0455, 0.335, 0.300
    tau_th = np.linspace(0.0, df_moving["nondimensional_distance_travelled"].max(), 300)
    phi = 1.0 - _A1 * np.exp(-_b1 * tau_th) - _A2 * np.exp(-_b2 * tau_th)
    CL_wag = steady_lift_coefficient * phi
    CDi_wag = CL_wag**2 / (np.pi * aspect_ratio)
    ax1.plot(
        tau_th,
        CL_wag,
        "--",
        color=C_THEORY,
        lw=1.0,
        label=rf"Wagner (aspect_ratio = {aspect_ratio:.0f})",
    )
    ax2.plot(tau_th, CDi_wag, "--", color=C_THEORY, lw=1.0, label="Wagner")

ax1.set_ylabel(r"Lift coefficient, $C_L$")
ax1.set_title(r"Lift and drag buildup, $\alpha = 5°$")
ax1.legend()
ax1.set_ylim(bottom=0)

ax2.set_xlabel(r"Chord-lengths traveled, $\nondimensional_time$")
ax2.set_ylabel(r"Drag coefficient, $C_D$")
ax2.legend()
ax2.set_ylim(bottom=0)

out = FIG_DIR / "plate_staticvsmoving.png"
m.save_fig(fig, out, figure_format=args.format, dpi=args.dpi)
