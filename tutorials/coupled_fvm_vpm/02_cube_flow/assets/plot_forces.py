#!/usr/bin/env python3
"""Drag, side force and lift on the cube: coupled FVM–VPM against the FVM reference."""

from pathlib import Path

import numpy as np

from openonda.plotting import COLORS, LINE_WIDTH, REFERENCE_STYLE
from openonda.plotting import save_validation_figure, validation_legend, validation_subplots

CASE_DIR = Path(__file__).resolve().parents[1]
REFERENCE = CASE_DIR / "reference_flow/samples/grid_h0045/forces_history.csv"
DEVELOPED_SHEDDING = 30.0  # s, start of the statistics window
STATISTICS_END = 60.0  # s, planned end of the coupled run
CUBE_SIDE = 1.0  # m
FREESTREAM_SPEED = 1.0  # m/s
FLOW_TIME_SCALE = FREESTREAM_SPEED / CUBE_SIDE  # 1/s, convective-time scale
COEFFICIENTS = {
    "drag_coefficient": "$C_D$",
    "side_force_coefficient": "$C_S$",
    "lift_coefficient": "$C_L$",
}


def read_forces(path):
    """Force-coefficient history, one row per sampled time."""
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")


def shedding_statistics(forces, end_time):
    """Mean drag, transverse force fluctuations and Strouhal number over developed shedding."""
    window = (forces["time"] >= DEVELOPED_SHEDDING) & (forces["time"] <= end_time)
    time = forces["time"][window]
    side = forces["side_force_coefficient"][window]
    lift = forces["lift_coefficient"][window]

    # Use the dominant transverse-force spectrum for the shedding frequency.
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
    end_time = min(STATISTICS_END, reference["time"][-1], coupled["time"][-1])
    reference = reference[reference["time"] <= end_time]
    coupled = coupled[coupled["time"] <= end_time]

    # Compare developed-shedding statistics.
    expected = shedding_statistics(reference, end_time)
    measured = shedding_statistics(coupled, end_time)
    print(
        "Developed shedding, "
        f"{DEVELOPED_SHEDDING * FLOW_TIME_SCALE:g} <= t U_inf / D <= "
        f"{end_time * FLOW_TIME_SCALE:g}"
    )
    for name in expected:
        deviation = 100.0 * (measured[name] / expected[name] - 1.0)
        print(
            f"  {name:16s} reference {expected[name]:.4f}  "
            f"coupled {measured[name]:.4f}  ({deviation:+.2f}%)"
        )

    figure, axes = validation_subplots(len(COEFFICIENTS), sharex=True)
    for axis, (name, label) in zip(axes, COEFFICIENTS.items()):
        axis.plot(
            reference["time"] * FLOW_TIME_SCALE,
            reference[name],
            **REFERENCE_STYLE,
            label="Reference FVM",
        )
        axis.plot(
            coupled["time"] * FLOW_TIME_SCALE,
            coupled[name],
            color=COLORS["hybrid"],
            linewidth=LINE_WIDTH,
            label="Coupled",
        )
        axis.set_ylabel(label)
    axes[-1].set_xlim(0.0, end_time * FLOW_TIME_SCALE)
    axes[-1].set_xlabel(r"$t U_\infty / D$")
    validation_legend(figure, axes[0])
    save_validation_figure(figure, CASE_DIR / "figures/forces")


if __name__ == "__main__":
    main()
