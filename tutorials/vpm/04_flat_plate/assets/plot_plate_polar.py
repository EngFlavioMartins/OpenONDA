"""Settled flat-plate lift, drag and quarter-chord moment coefficients."""

import argparse

import numpy as np

from ._plot_theme import (
    FIG_DIR,
    SAMPLES_DIR,
    centered_subplots_adjust,
    color,
    save_fig,
    validation_legend,
    validation_subplots,
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
                SAMPLES_DIR.parent, f"exp_{mode}_aoa{('n' if angle < 0 else '')}{abs(angle):02d}"
            )
            for angle in angles
        ]
    )
fig, axes = validation_subplots(
    3, height_cm=12.8, sharex=True, outer=0.129, top_padding_cm=0.11, bottom_padding_cm=2.25
)
reference_angles = np.linspace(angles.min(), angles.max(), 200)
reference = lifting_line_polar(reference_angles, ar)
for i in range(2):
    axes[i].plot(
        reference_angles, reference[i], "--", color=color("reference"), label="Lifting-line"
    )
for mode, marker, ink in [("moving", "o", color("teal")), ("static", "s", color("vpm"))]:
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
for axis, label in zip(axes, ["$C_L$", "$C_D$", "$C_{m,c/4}$"], strict=True):
    axis.set_ylabel(label)
    axis.axhline(0, color="0.6", lw=0.6)
    axis.set_xlim(-11, 16)
for axis, letter in zip(axes, "abc", strict=True):
    axis.text(0.035, 0.93, f"({letter})", transform=axis.transAxes, va="top")
axes[-1].set_xlabel("Angle of attack, $\\alpha$ [degrees]")
centered_subplots_adjust(fig, outer=0.129, bottom=2.25 / 12.8, top=1 - 0.11 / 12.8, hspace=0.24)
validation_legend(fig, axes[0], ncol=3, outside=True)
print("Final five chord lengths; lift, drag and quarter-chord moment")
print("Lifting-line is a small-angle, high-AR approximation")
save_fig(fig, FIG_DIR / "plate_polar.png", figure_format=args.format, dpi=args.dpi)
