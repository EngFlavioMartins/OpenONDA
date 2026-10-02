#!/usr/bin/env python3
"""Settled flat-plate polar and signed differences from lifting-line theory."""

if not __package__:
    from pathlib import Path
    from openonda.tutorial_runner import case_package

    __package__ = case_package(Path(__file__).resolve().parents[1]) + ".assets"

import argparse
import numpy as np
from ._plot_theme import (
    FIG_DIR,
    SAMPLES_DIR,
    color,
    save_fig,
    validation_subplots,
    validation_legend,
)
from .results import parameters, settled_coefficients
from .theoretical_model import lifting_line_polar

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--format", choices=("png", "pdf", "both"), default="both")
parser.add_argument("--dpi", type=int, default=400)
args = parser.parse_args()
physics = parameters(SAMPLES_DIR.parent)
ar = physics["span"] / physics["chord"]
angles = np.array([-10, -5, -2, 0, 2, 5, 8, 10, 12, 15])
curves = {}
for mode in ("moving", "static"):
    curves[mode] = np.array(
        [
            settled_coefficients(
                SAMPLES_DIR.parent, f"exp_{mode}_aoa{'n' if angle < 0 else ''}{abs(angle):02d}"
            )
            for angle in angles
        ]
    )
fig, axes = validation_subplots(4, height_cm=23, sharex=True, outer=0.1215, top_padding_cm=0.11)
reference_angles = np.linspace(angles.min(), angles.max(), 200)
reference = lifting_line_polar(reference_angles, ar)
for i in range(2):
    axes[i].plot(reference_angles, reference[i], "--", color=color("ref"), label="Lifting-line")
for mode, marker, ink in [("moving", "o", color("TUDcyan")), ("static", "s", color("vpm"))]:
    for i in range(3):
        axes[i].plot(
            angles,
            curves[mode][:, i],
            marker + "-",
            color=ink,
            ms=5 if mode == "moving" else 3,
            mfc="none" if mode == "moving" else ink,
            label=mode.capitalize(),
        )
ref = np.asarray(lifting_line_polar(angles, ar)).T
nonzero = angles != 0
for i, label, marker in [(0, "Lift", "o"), (1, "Drag", "s")]:
    error = 100 * (curves["static"][nonzero, i] / ref[nonzero, i] - 1)
    axes[3].plot(
        angles[nonzero],
        error,
        marker + "-",
        ms=3,
        color=color("vpm" if i == 0 else "TUDcyan"),
        label=label,
    )
for i, label in [(0, "Lift"), (1, "Drag")]:
    error = 100 * (curves["static"][-1, i] / ref[-1, i] - 1)
    axes[3].text(16, error, label, va="center", color=color("vpm" if i == 0 else "TUDcyan"))
for axis, label in zip(
    axes, [r"$C_L$", r"$C_D$", r"$C_{m,c/4}$", r"Theory difference [\%]"], strict=True
):
    axis.set_ylabel(label)
    axis.axhline(0, color="0.6", lw=0.6)
    axis.set_xlim(-11, 20)
axes[3].margins(y=0.15)
axes[3].set_xlabel(r"Angle of attack, $\alpha$ [degrees]")
validation_legend(fig, axes[0], ncol=3)
print("Final five chord lengths; zero-incidence ratios omitted")
print("Lifting-line is a small-angle, high-AR approximation")
save_fig(fig, FIG_DIR / "plate_polar.png", figure_format=args.format, dpi=args.dpi)
