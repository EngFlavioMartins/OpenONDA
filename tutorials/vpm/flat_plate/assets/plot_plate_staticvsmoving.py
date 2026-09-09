#!/usr/bin/env python3
"""Compare startup and subsequent wake development in the two flat-plate frames."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from openonda.tutorial_runner import case_package

__package__ = case_package(Path(__file__).resolve().parents[1]) + ".assets"
from ._plot_theme import FIG_DIR, SAMPLES_DIR, cm, color, save_fig
from .results import load_forces, parameters
from .theoretical_model import lifting_line_polar

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--format", choices=("png", "pdf"), default="png")
parser.add_argument("--dpi", type=int, default=400)
args = parser.parse_args()

physics = parameters(SAMPLES_DIR.parent, "exp_static_aoa05")
reference = lifting_line_polar(5.0, physics["span"] / physics["chord"])
histories = [
    (load_forces(SAMPLES_DIR.parent, "exp_static_aoa05"), color("vpm"), "Static (wind frame)"),
    (load_forces(SAMPLES_DIR.parent, "exp_moving_aoa05"), color("TUDcyan"), "Moving (body frame)"),
]

# Keep the impulsive-start peak visible without flattening the later comparison.
# Both figures use unmodified native loads, including the pressure-time term.
for filename, limits, title, legend_position in [
    ("plate_startup", (0.0, 2.0), "Startup", "upper right"),
    ("plate_staticvsmoving", (2.0, 24.0), "Wake development", "lower right"),
]:
    fig, axes = plt.subplots(
        2, 1, figsize=(12.5 * cm(), 12 * cm()), sharex=True, constrained_layout=True
    )
    for frame, ink, label in histories:
        travel = frame.nondimensional_distance_travelled
        selected = frame[(travel >= limits[0]) & (travel <= limits[1])]
        for axis, column in zip(axes, ("lift_coefficient", "drag_coefficient"), strict=True):
            axis.plot(
                selected.nondimensional_distance_travelled,
                selected[column],
                color=ink,
                lw=1.5,
                label=label,
            )
    for axis, value in zip(axes, reference, strict=True):
        axis.axhline(value, ls="--", color=color("ref"), lw=1.0, label="Steady lifting-line")
        axis.set_ylim(0, 1.08 * axis.get_ylim()[1])
        axis.set_xlim(*limits)
    axes[0].set_ylabel(r"Lift coefficient, $C_L$")
    axes[1].set_ylabel(r"Drag coefficient, $C_D$")
    axes[0].set_title(title + r", $\alpha=5^\circ$")
    axes[0].legend(loc=legend_position)
    axes[1].set_xlabel(r"Chord-lengths traveled, $\tau$")
    save_fig(fig, FIG_DIR / f"{filename}.png", figure_format=args.format, dpi=args.dpi)
