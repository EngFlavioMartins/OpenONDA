#!/usr/bin/env python3
"""Thrust, torque and power of each blade: meshed blades against the VLM blades."""

from openonda.plotting import COLORS, LINE_WIDTH, REFERENCE_STYLE, save_validation_figure
from openonda.plotting import validation_legend, validation_subplots

from assets.rotor_data import FIGURES, ROTATION_PERIOD, fvm_blades, vlm_blades

QUANTITIES = (
    ("force_x", "Thrust [N]"),
    ("moment_x", "Torque [N m]"),
    ("power", "Power [W]"),
)
BLADE_STYLES = ("-", "--", ":")


def main():
    rotors = {"FVM and VPM": (fvm_blades(), COLORS["hybrid"])}
    vlm = vlm_blades()
    if vlm:
        rotors["VLM and VPM"] = (vlm, REFERENCE_STYLE["color"])

    figure, axes = validation_subplots(3, sharex=True)
    for label, (blades, colour) in rotors.items():
        for (name, blade), linestyle in zip(sorted(blades.items()), BLADE_STYLES):
            revolutions = blade["time"] / ROTATION_PERIOD
            for axis, (column, _) in zip(axes, QUANTITIES):
                axis.plot(
                    revolutions,
                    blade[column],
                    color=colour,
                    linestyle=linestyle,
                    linewidth=LINE_WIDTH,
                    label=f"{label}, {name.replace('_', ' ')}" if axis is axes[0] else None,
                )
    for axis, (_, unit) in zip(axes, QUANTITIES):
        axis.set_ylabel(unit)
    axes[2].set_xlabel("Nominal revolutions")
    validation_legend(figure, axes[0], ncol=2)
    save_validation_figure(figure, FIGURES / "blade_forces")


if __name__ == "__main__":
    main()
