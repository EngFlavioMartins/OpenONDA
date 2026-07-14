#!/usr/bin/env python3
"""
Flat Plate — Spanwise Lift Distribution Plotter
================================================
Reads the spanwise-force CSV produced by setup_plate.py (via
VLMLoadingDistribution) and plots the sectional lift coefficient
cl(y) against Prandtl lifting-line theory and an elliptic reference.

Overlays the moving and static cases at the same AoA on one figure.

Usage (standalone):
    python assets/plot_plate_spanwise.py
    python assets/plot_plate_spanwise.py --aoa 5

Called from allplot.sh with:
    --solution-dir, --figures-dir, --dpi

Output:
    figures/plate_spanwise.png

Author:  Flavio A. C. Martins, OpenONDA Team
Date: June 2026
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -- Paths ----------------------------------------------------------------------
CASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = CASE_DIR.parents[2]
THEME_PATH = REPO_ROOT / "docs" / "themes" / "matplotlib_setup.py"

sys.path.insert(0, str(CASE_DIR / "assets"))
from theoretical_model import spanwise_reference

# -- Argument parsing -----------------------------------------------------------
parser = argparse.ArgumentParser(description="Flat plate spanwise lift distribution")
parser.add_argument("--case-moving", default="exp_moving_aoa05", help="Moving case folder")
parser.add_argument("--case-static", default="exp_static_aoa05", help="Static case folder")
parser.add_argument(
    "--aoa", type=float, default=5.0, help="Angle of attack for theory curves [deg]"
)
parser.add_argument("--solution-dir", default=str(CASE_DIR / "solution"))
parser.add_argument("--figures-dir", default=str(CASE_DIR / "figures"))
parser.add_argument("--format", choices=("png", "pdf"), default="png")
parser.add_argument("--dpi", type=int, default=300)
args, _ = parser.parse_known_args()

SOL_DIR = Path(args.solution_dir).resolve()
FIG_DIR = Path(args.figures_dir).resolve()
FIG_DIR.mkdir(parents=True, exist_ok=True)

# -- Theme ----------------------------------------------------------------------
m = None
if not THEME_PATH.exists():
    raise FileNotFoundError(f"OpenONDA matplotlib theme not found: {THEME_PATH}")
spec = importlib.util.spec_from_file_location("matplotlib_setup", str(THEME_PATH))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.set_style()


def _c(key):
    return m.COLORS[key]


C_MOVING = _c("TUDcyan")
C_STATIC = _c("vpm")
C_LL = _c("ref")
C_ELL = _c("literature")

cm_inch = m.CM

# -- Physical constants ---------------------------------------------------------
AR = 10.0
CHORD = 1.0
SPAN = AR * CHORD
U_INF = 10.0
alpha_rad = np.radians(args.aoa)

# -- Theory curves --------------------------------------------------------------
y_theory = np.linspace(-SPAN / 2, SPAN / 2, 400)

df_ll = spanwise_reference("liftingline", y_theory, SPAN, CHORD, alpha_rad, U_INF, n_terms=120)
cl_ll = df_ll["cl"].to_numpy()
y_ll_over_b = df_ll["y_over_b"].to_numpy()

CL_ll = float(np.trapezoid(cl_ll * CHORD, y_theory) / (SPAN * CHORD))

df_ell = spanwise_reference(
    "elliptic", y_theory, SPAN, CHORD, alpha_rad, U_INF, CL_total=CL_ll, AR=AR
)
cl_ell = df_ell["cl"].to_numpy()


# -- Load simulation data -------------------------------------------------------
def load_spanwise_csv(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    csv = SOL_DIR / name / "samples" / f"{name}_spanwise.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv).sort_values("y").reset_index(drop=True)
    return df["y"].to_numpy(), df["cl"].to_numpy(), df["y_over_b"].to_numpy()


moving_data = load_spanwise_csv(args.case_moving)
static_data = load_spanwise_csv(args.case_static)

# -- Figure ---------------------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(12 * cm_inch, 5.5 * cm_inch))
fig.subplots_adjust(left=0.14, right=0.95, bottom=0.17, top=0.88)

if moving_data is not None:
    y_m, cl_m, yob_m = moving_data
    ax.plot(
        yob_m,
        cl_m,
        color=C_MOVING,
        lw=1.5,
        marker="o",
        ms=3,
        label=f"Moving (body frame), $\\alpha={args.aoa:.0f}°$",
    )

if static_data is not None:
    y_s, cl_s, yob_s = static_data
    ax.plot(
        yob_s,
        cl_s,
        color=C_STATIC,
        lw=1.5,
        marker="s",
        ms=3,
        label=f"Static (wind frame), $\\alpha={args.aoa:.0f}°$",
    )

# Theory curves
ax.plot(y_ll_over_b, cl_ll, "--", color=C_LL, lw=1.0, label="Lifting-line (Glauert)")
ax.plot(y_ll_over_b, cl_ell, "--", color=C_ELL, lw=1.0, label="Elliptic")

ax.set_xlabel(r"Spanwise position, $2y/b$")
ax.set_ylabel(r"Sectional lift coefficient, $c_\ell$")
ax.set_title(rf"Spanwise $c_\ell$, $\alpha={args.aoa:.0f}°$, AR={AR:.0f}")
ax.set_xlim(-1, 1)
ax.legend()

out = FIG_DIR / "plate_spanwise.png"
m.save_fig(fig, out, figure_format=args.format, dpi=args.dpi)
