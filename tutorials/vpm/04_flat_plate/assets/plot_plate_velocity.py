#!/usr/bin/env python3
"""Compare sampled spanwise induced velocities with lifting-line theory."""

if not __package__:
    from pathlib import Path as _CasePath
    from openonda.tutorial_runner import case_package

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._plot_theme import FIG_DIR, SAMPLES_DIR, centered_subplots_adjust, cm, color, save_fig
from .results import parameters
from .theoretical_model import lifting_line_circulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--format", choices=("png", "pdf"), default="png")
parser.add_argument("--dpi", type=int, default=400)
args = parser.parse_args()

physics = parameters(SAMPLES_DIR.parent)
span, chord, speed = physics["span"], physics["chord"], physics["speed"]
fig, axes = plt.subplots(2, 1, figsize=(12.5 * cm(), 10.8 * cm()), sharex=True)
centered_subplots_adjust(fig, outer=0.17, bottom=0.125, top=0.92, hspace=0.12)

for mode, ink, marker in [("moving", color("TUDcyan"), "o"), ("static", color("vpm"), "s")]:
    name = f"exp_{mode}_aoa05"
    data = pd.read_csv(SAMPLES_DIR / name / "vlm_chordwise_flat_plate.csv")
    data = data[data.step == data.step.max()]
    reference = parameters(SAMPLES_DIR.parent, name)["reference_velocity"]
    stream = reference / np.linalg.norm(reference)
    lift = np.array([0.0, 0.0, 1.0]) - stream[2] * stream
    lift /= np.linalg.norm(lift)
    stations = []
    for _, strip in data.groupby("station_id"):
        # This circulation-weighted velocity is the one entering the sectional
        # Kutta-Joukowski force, and is the finite-chord analogue of lifting-line w.
        relative = strip[
            ["relative_velocity_x", "relative_velocity_y", "relative_velocity_z"]
        ].to_numpy()
        induced = np.average(relative - reference, axis=0, weights=strip.panel_circulation)
        stations.append((strip.span_coordinate.iloc[0], induced @ stream, induced @ lift))
    profile = np.array(sorted(stations))
    for axis, column in zip(axes, [2, 1], strict=True):
        axis.plot(
            2 * profile[:, 0] / span,
            profile[:, column],
            marker=marker,
            ms=5 if mode == "moving" else 3,
            markerfacecolor="none" if mode == "moving" else ink,
            lw=1.2,
            color=ink,
            label=mode.capitalize(),
        )

y = np.linspace(-0.5 * span, 0.5 * span, 401)
reference = lifting_line_circulation(y, span, chord, np.radians(5), speed)
axes[0].plot(
    2 * y / span, reference.induced_velocity_z, "--", color=color("ref"), label="Lifting-line"
)
axes[1].axhline(0, ls="--", lw=1, color=color("ref"), label="Lifting-line")
axes[0].set_ylabel(r"Downwash, $w$ [m/s]")
axes[1].set_ylabel(r"Streamwise, $u_i$ [m/s]")
axes[1].set_xlabel(r"Spanwise position, $2y/b$")
axes[0].set_title(r"Induced velocity, $\alpha=5^\circ$")
for axis in axes:
    axis.set_xlim(-1, 1)
axes[0].legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 0.035),
    ncol=3,
    handlelength=1.2,
    handletextpad=0.4,
    columnspacing=0.8,
    borderpad=0.35,
)
save_fig(fig, FIG_DIR / "plate_velocity.png", figure_format=args.format, dpi=args.dpi)
