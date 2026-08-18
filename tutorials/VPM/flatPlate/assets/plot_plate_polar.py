#!/usr/bin/env python3
"""
Flat Plate VLM-VPM — AoA Polar Plotter
=======================================
Generates Figure 1: CL, CD, CM vs α for the flat-plate VLM-VPM
experiment suite over α = −10° … 15°.

Output:
    figures/exp_polar.png

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

from theoretical_model import prandtl_a3D


def _c(key):
    return m.COLORS[key]


# -- Colours -------------------------------------------------------------------
C_MOVING = _c("TUDcyan")
C_STATIC = _c("vpm")
C_THEORY = _c("ref")

# -- Physical constants --------------------------------------------------------
AR = 10.0
CHORD = 1.0
FREESTREAM_SPEED = 10.0

cm = 1 / 2.54


# -- Helpers -------------------------------------------------------------------


def lifting_line_polar(alpha_deg, AR, b=10.0, chord=1.0, n_terms=40):
    alpha = np.radians(alpha_deg)
    a0 = 2.0 * np.pi
    mu = a0 * chord / (4.0 * b)

    n_odd = 2 * np.arange(1, n_terms + 1) - 1
    theta_k = np.linspace(np.pi / (2 * n_terms + 2), np.pi - np.pi / (2 * n_terms + 2), n_terms)
    sin_theta = np.sin(theta_k)

    M = np.zeros((n_terms, n_terms))
    for j, n in enumerate(n_odd):
        M[:, j] = np.sin(n * theta_k) * (sin_theta + mu * n)
    rhs = mu * alpha * sin_theta
    A_n = np.linalg.solve(M, rhs)

    CL = np.pi * AR * A_n[0]
    CDi = np.pi * AR * np.sum(n_odd * A_n**2)
    return CL, CDi


def load_csv(name: str) -> pd.DataFrame | None:
    csv = SAMPLES_DIR / name / f"{name}.csv"
    if not csv.exists():
        print(f"  [MISSING] {csv}")
        return None
    return pd.read_csv(csv)


def final_coefficients(name: str) -> tuple[float, float, float]:
    df = load_csv(name)
    if df is None:
        return float("nan"), float("nan"), float("nan")
    # Average the converged tail rather than using one noise-sensitive sample.
    if "chords" in df and df["chords"].max() >= 5.0:
        df = df[df["chords"] >= df["chords"].max() - 5.0]
    cl = float(df["CL"].mean())
    cd = float(df["CD"].mean())
    cm = float(df["Cm_c4"].mean()) if "Cm_c4" in df.columns else float("nan")
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
CL_theory = np.empty_like(alpha_range)
CDi_theory = np.empty_like(alpha_range)
for i, a_deg in enumerate(alpha_range):
    CL_theory[i], CDi_theory[i] = lifting_line_polar(a_deg, AR, b=AR * CHORD, chord=CHORD)

# -- Figure --------------------------------------------------------------------

fig, (ax_cl, ax_cd, ax_cm) = plt.subplots(
    3, 1, figsize=(12 * cm, 15 * cm), sharex=True, constrained_layout=True
)

# CL vs α
ax_cl.plot(
    alpha_range,
    CL_theory,
    "--",
    color=C_THEORY,
    lw=1.0,
    label=rf"Prandtl lifting-line (AR={AR:.0f})",
)
ax_cl.plot(moving_aoa, moving_cl, "o", color=C_MOVING, ms=5, zorder=5, label="Moving (body frame)")
ax_cl.plot(static_aoa, static_cl, "s", color=C_STATIC, ms=5, zorder=4, label="Static (wind frame)")
ax_cl.axhline(0, color=_c("DarkText"), lw=0.5, alpha=0.35)
ax_cl.set_ylabel(r"Lift coefficient, $C_L$")
ax_cl.set_title(r"$C_L$ vs $\alpha$")
ax_cl.set_ylim(bottom=-1.6, top=1.6)
ax_cl.legend()
ax_cl.set_xlim(-12, 18)

# CD vs α
ax_cd.plot(
    alpha_range,
    CDi_theory,
    "--",
    color=C_THEORY,
    lw=1.0,
    label=rf"Prandtl lifting-line (AR={AR:.0f})",
)
ax_cd.plot(moving_aoa, moving_cd, "o", color=C_MOVING, ms=5, zorder=5, label="Moving (body frame)")
ax_cd.plot(static_aoa, static_cd, "s", color=C_STATIC, ms=5, zorder=4, label="Static (wind frame)")
ax_cd.axhline(0, color=_c("DarkText"), lw=0.5, alpha=0.35)
ax_cd.set_ylabel(r"Drag coefficient, $C_D$")
ax_cd.set_title(r"$C_D$ vs $\alpha$")
ax_cd.legend()
ax_cd.set_xlim(-12, 18)
ax_cd.set_ylim(bottom=-0.01, top=0.10)

# CM vs α
ax_cm.plot(moving_aoa, moving_cm, "o", color=C_MOVING, ms=5, zorder=5, label="Moving (body frame)")
ax_cm.plot(static_aoa, static_cm, "s", color=C_STATIC, ms=5, zorder=4, label="Static (wind frame)")
ax_cm.axhline(0, color=_c("DarkText"), lw=0.5, alpha=0.35)
ax_cm.set_xlabel(r"Angle of attack, $\alpha$ [°]")
ax_cm.set_ylabel(r"Moment coeff., $C_{m,c/4}$")
ax_cm.set_title(r"$C_{m,c/4}$ vs $\alpha$")
ax_cm.legend()
ax_cm.set_ylim(bottom=-0.01, top=0.01)
ax_cm.set_xlim(-12, 18)

out = FIG_DIR / "plate_polar.png"
m.save_fig(fig, out, figure_format=args.format, dpi=args.dpi)
