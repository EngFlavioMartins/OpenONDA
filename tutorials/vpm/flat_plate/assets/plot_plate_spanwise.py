#!/usr/bin/env python3
"""Spanwise lift distribution cl(y) for moving and static plates at AoA=5 deg.

Output: figures/plate_spanwise.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _plot_theme import SAMPLES_DIR, FIG_DIR, color, cm, save_fig
from theoretical_model import spanwise_reference

parser = argparse.ArgumentParser(description="Flat plate spanwise lift distribution")
parser.add_argument("--format", choices=("png", "pdf"), default="png")
parser.add_argument("--dpi", type=int, default=300)
args = parser.parse_args()
FIG_DIR.mkdir(parents=True, exist_ok=True)


C_MOVING = color("TUDcyan")
C_STATIC = color("vpm")
C_LL = color("ref")
C_ELL = color("literature")

CM = cm()

# -- Physical constants ---------------------------------------------------------
aspect_ratio = 10.0
CHORD = 1.0
SPAN = aspect_ratio * CHORD
FREESTREAM_SPEED = 10.0
ANGLE_OF_ATTACK = 5.0
angle_of_attack_radians = np.radians(ANGLE_OF_ATTACK)

# -- Theory curves --------------------------------------------------------------
y_theory = np.linspace(-SPAN / 2, SPAN / 2, 400)

df_ll = spanwise_reference(
    "lifting_line",
    y_theory,
    SPAN,
    CHORD,
    angle_of_attack_radians,
    FREESTREAM_SPEED,
    n_fourier_terms=120,
)
cl_ll = df_ll["section_lift_coefficient"].to_numpy()
y_ll_over_b = df_ll["span_coordinate_normalized"].to_numpy()

CL_ll = float(np.trapezoid(cl_ll * CHORD, y_theory) / (SPAN * CHORD))

df_ell = spanwise_reference(
    "elliptic",
    y_theory,
    SPAN,
    CHORD,
    angle_of_attack_radians,
    FREESTREAM_SPEED,
    total_lift_coefficient=CL_ll,
    aspect_ratio=aspect_ratio,
)
cl_ell = df_ell["section_lift_coefficient"].to_numpy()


# -- Load simulation data -------------------------------------------------------
def load_spanwise_csv(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    csv = SAMPLES_DIR / name / f"{name}_spanwise.csv"
    if not csv.exists():
        print(
            f"  [MISSING] {csv}\n"
            f"  Skipping '{name}' experimental spanwise loading — its curve will be "
            f"omitted from the figure. Re-run the case (setup_plate.py) to regenerate it."
        )
        return None
    df = pd.read_csv(csv).sort_values("span_coordinate").reset_index(drop=True)
    y = df["span_coordinate"].to_numpy()
    cl = df["section_lift_coefficient"].to_numpy()
    # Convert the canonical dimensional panel-centre coordinate to span fraction.
    y_over_b = 2.0 * y / SPAN

    # VLM unknowns are cell-centred, so the outermost samples sit just inside
    # the physical tips.  Close the plotted circulation distribution with the
    # finite-wing boundary condition Γ(±b/2)=0 instead of stretching those
    # samples to ±1 (which falsely displayed non-zero lift at the tips).
    if y_over_b.size:
        if y_over_b[0] > -1.0:
            y = np.insert(y, 0, -SPAN / 2.0)
            cl = np.insert(cl, 0, 0.0)
            y_over_b = np.insert(y_over_b, 0, -1.0)
        if y_over_b[-1] < 1.0:
            y = np.append(y, SPAN / 2.0)
            cl = np.append(cl, 0.0)
            y_over_b = np.append(y_over_b, 1.0)

    return y, cl, y_over_b


moving_data = load_spanwise_csv("exp_moving_aoa05")
static_data = load_spanwise_csv("exp_static_aoa05")

# -- Figure ---------------------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(12 * CM, 5.5 * CM))
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
        label=f"Moving (body frame), $\\alpha={ANGLE_OF_ATTACK:.0f}°$",
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
        label=f"Static (wind frame), $\\alpha={ANGLE_OF_ATTACK:.0f}°$",
    )

# Theory curves
ax.plot(y_ll_over_b, cl_ll, "--", color=C_LL, lw=1.0, label="Lifting-line (Glauert)")
ax.plot(y_ll_over_b, cl_ell, "--", color=C_ELL, lw=1.0, label="Elliptic")

ax.set_xlabel(r"Spanwise position, $2y/b$")
ax.set_ylabel(r"Sectional lift coefficient, $c_\ell$")
ax.set_title(
    rf"Spanwise $c_\ell$, $\alpha={ANGLE_OF_ATTACK:.0f}°$, aspect_ratio={aspect_ratio:.0f}"
)
ax.set_xlim(-1, 1)
ax.legend()

out = FIG_DIR / "plate_spanwise.png"
save_fig(fig, out, figure_format=args.format, dpi=args.dpi)
