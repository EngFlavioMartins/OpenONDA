"""Compare startup and subsequent wake development in the two flat-plate frames."""

import argparse

from ._plot_theme import (
    FIG_DIR,
    SAMPLES_DIR,
    centered_subplots_adjust,
    color,
    save_fig,
    validation_legend,
    validation_subplots,
)
from .results import load_forces, parameters
from .theoretical_model import lifting_line_polar

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--format", choices=("png", "pdf", "both"), default="both")
parser.add_argument("--dpi", type=int, default=400)
parser.add_argument(
    "--include-startup", action="store_true", help="Also export the optional startup comparison."
)
args = parser.parse_args()
physics = parameters(SAMPLES_DIR.parent, "exp_static_aoa05")
reference = lifting_line_polar(5.0, physics["span"] / physics["chord"])
histories = [
    (load_forces(SAMPLES_DIR.parent, "exp_static_aoa05"), color("vpm"), "Static"),
    (load_forces(SAMPLES_DIR.parent, "exp_moving_aoa05"), color("teal"), "Moving"),
]
figures = [("plate_staticvsmoving", (2.0, 24.0))]
if args.include_startup:
    figures.insert(0, ("plate_startup", (0.0, 2.0)))
for filename, limits in figures:
    fig, axes = validation_subplots(
        2, height_cm=9.2, sharex=True, outer=0.115, top_padding_cm=0.11, bottom_padding_cm=2.3
    )
    for (frame, ink, label), marker in zip(histories, ("o", "s"), strict=True):
        travel = frame.nondimensional_distance_travelled
        selected = frame[(travel >= limits[0]) & (travel <= limits[1])]
        for axis, column in zip(axes, ("lift_coefficient", "drag_coefficient"), strict=True):
            axis.plot(
                selected.nondimensional_distance_travelled,
                selected[column],
                color=ink,
                ls="-",
                marker=marker,
                markevery=80,
                ms=2.5,
                lw=1.5,
                label=label,
            )
    for axis, value in zip(axes, reference, strict=True):
        axis.axhline(value, ls="--", color=color("reference"), lw=1.0, label="Lifting-line")
        if filename == "plate_startup":
            axis.set_ylim(0, 1.08 * axis.get_ylim()[1])
        axis.set_xlim(*limits)
    axes[0].set_ylabel("Lift coefficient, $C_L$")
    axes[1].set_ylabel("Drag coefficient, $C_D$")
    validation_legend(fig, axes[0], ncol=3, outside=True)
    note = (
        "Different starts: impulsive inflow / smooth body ramp"
        if filename == "plate_startup"
        else "Steady-limit comparison; no exact transient reference"
    )
    print(note)
    axes[1].set_xlabel("Convective distance, $\\tau$")
    axes[1].tick_params(axis="x", pad=7.0)
    centered_subplots_adjust(fig, outer=0.115, bottom=2.3 / 9.2, top=1 - 0.11 / 9.2, hspace=0.18)
    save_fig(fig, FIG_DIR / f"{filename}.png", figure_format=args.format, dpi=args.dpi)
