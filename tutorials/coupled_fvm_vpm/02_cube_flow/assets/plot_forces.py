#!/usr/bin/env python3
"""Drag, side force and lift on the cube: coupled FVM–VPM against the FVM reference."""

from pathlib import Path

import numpy as np

from openonda.plotting import COLORS, LINE_WIDTH, REFERENCE_STYLE
from openonda.plotting import save_validation_figure, validation_legend, validation_subplots

CASE_DIR = Path(__file__).resolve().parents[1]
REFERENCE = CASE_DIR / "reference_flow/samples/grid_h0045/forces_history.csv"
DEVELOPED_SHEDDING = 40.0  # s, start of the statistics window
CUBE_SIDE = 1.0  # m
FREESTREAM_SPEED = 1.0  # m/s
COEFFICIENTS = {
    "drag_coefficient": "$C_D$",
    "side_force_coefficient": "$C_S$",
    "lift_coefficient": "$C_L$",
}


def read_forces(path):
    """Force-coefficient history, one row per sampled time."""
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")


def shedding_statistics(forces):
    """Mean drag, transverse force fluctuations and Strouhal number over developed shedding."""
    window = forces["time"] >= DEVELOPED_SHEDDING
    time = forces["time"][window]
    side = forces["side_force_coefficient"][window]
    lift = forces["lift_coefficient"][window]

    # The shedding frequency is the spectral peak of the stronger transverse force
    transverse = side if side.std() >= lift.std() else lift
    spectrum = np.abs(np.fft.rfft(transverse - transverse.mean()))
    frequency = np.fft.rfftfreq(len(transverse), np.diff(time).mean())

    return {
        "mean drag": forces["drag_coefficient"][window].mean(),
        "side force rms": side.std(),
        "lift rms": lift.std(),
        "Strouhal number": frequency[1:][spectrum[1:].argmax()] * CUBE_SIDE / FREESTREAM_SPEED,
    }


def main():
    reference = read_forces(REFERENCE)
    coupled = read_forces(CASE_DIR / "samples/forces_history.csv")

    # Developed-shedding statistics and the coupled deviation from the reference
    expected, measured = shedding_statistics(reference), shedding_statistics(coupled)
    print(f"Developed shedding, t >= {DEVELOPED_SHEDDING:g} s")
    for name in expected:
        deviation = 100.0 * (measured[name] / expected[name] - 1.0)
        print(
            f"  {name:16s} reference {expected[name]:.4f}  "
            f"coupled {measured[name]:.4f}  ({deviation:+.2f}%)"
        )

    figure, axes = validation_subplots(len(COEFFICIENTS), sharex=True)
    for axis, (name, label) in zip(axes, COEFFICIENTS.items()):
        axis.plot(reference["time"], reference[name], **REFERENCE_STYLE, label="Reference FVM")
        axis.plot(
            coupled["time"],
            coupled[name],
            color=COLORS["hybrid"],
            linewidth=LINE_WIDTH,
            label="Coupled",
        )
        axis.set_ylabel(label)
    axes[-1].set_xlabel("$t$ [s]")
    validation_legend(figure, axes[0])
    save_validation_figure(figure, CASE_DIR / "figures/forces")


if __name__ == "__main__":
    main()
