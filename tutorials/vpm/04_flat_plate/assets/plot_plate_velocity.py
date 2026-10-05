"""Compare sampled spanwise induced velocities with lifting-line theory."""

import argparse

import numpy as np
import pandas as pd

from openonda.results import read_csv_table

from ._plot_theme import (
    FIG_DIR,
    SAMPLES_DIR,
    centered_subplots_adjust,
    cm,
    color,
    save_fig,
    validation_legend,
    validation_subplots,
)
from .results import parameters
from .theoretical_model import lifting_line_circulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--format", choices=("png", "pdf", "both"), default="both")
parser.add_argument("--dpi", type=int, default=400)
args = parser.parse_args()
physics = parameters(SAMPLES_DIR.parent)
span, chord, speed = (physics["span"], physics["chord"], physics["speed"])
fig, axes = validation_subplots(
    2, height_cm=9.2, sharex=True, outer=0.135, top_padding_cm=0.13, bottom_padding_cm=2.15
)
fig.set_size_inches(12.5 * cm(), 10.0 * cm(), forward=False)
for mode, ink, marker in [("moving", color("teal"), "o"), ("static", color("vpm"), "s")]:
    name = f"exp_{mode}_aoa05"
    data = pd.DataFrame(read_csv_table(SAMPLES_DIR / name / "vlm_chordwise_flat_plate.csv"))
    data = data[data.step == data.step.max()]
    reference = parameters(SAMPLES_DIR.parent, name)["reference_velocity"]
    stream = reference / np.linalg.norm(reference)
    lift = np.array([0.0, 0.0, 1.0]) - stream[2] * stream
    lift /= np.linalg.norm(lift)
    stations = []
    for _, strip in data.groupby("station_id"):
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
    2 * y / span, reference.induced_velocity_z, "--", color=color("reference"), label="Lifting-line"
)
axes[1].axhline(0, ls="--", lw=1, color=color("reference"), label="Lifting-line")
axes[0].set_ylabel("Downwash, $w$ [m/s]")
axes[1].set_ylabel("Streamwise, $u_i$ [m/s]")
axes[1].set_xlabel("Spanwise position, $2y/b$")
for axis in axes:
    axis.set_xlim(-1, 1)
centered_subplots_adjust(fig, outer=0.135, bottom=2.15 / 10.0, top=1 - 0.13 / 10.0, hspace=0.18)
validation_legend(fig, axes[0], ncol=3, outside=True)
print("Final sampled bound-point velocity, weighted by circulation")
print("Finite chord and wake cores affect tip downwash")
save_fig(fig, FIG_DIR / "plate_velocity.png", figure_format=args.format, dpi=args.dpi)
