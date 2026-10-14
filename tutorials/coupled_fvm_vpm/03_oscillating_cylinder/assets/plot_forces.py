#!/usr/bin/env python3
"""Drag and lift on the heaving cylinder: coupled FVM–VPM against the FVM reference."""

from pathlib import Path

import numpy as np

from openonda.plotting import COLORS, LINE_WIDTH, REFERENCE_STYLE
from openonda.plotting import save_validation_figure, validation_legend, validation_subplots

CASE_DIR = Path(__file__).resolve().parents[1]
LOCKED_IN = 25.0  # s, start of the statistics window
FREQUENCY = 0.18  # Hz, heave frequency


def read_forces(path):
    """Force-coefficient history, one row per sampled time."""
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")


def lock_in_statistics(forces):
    """Mean drag, lift amplitude, lift frequency and lift phase to the motion."""
    window = forces["time"] >= LOCKED_IN
    time = forces["time"][window]
    lift = forces["lift_coefficient"][window]

    # The lift period is the mean spacing of upward zero crossings
    fluctuation = lift - lift.mean()
    up = np.flatnonzero((fluctuation[:-1] < 0.0) & (fluctuation[1:] >= 0.0))
    crossing = time[up] - fluctuation[up] * np.diff(time)[up] / np.diff(fluctuation)[up]
    frequency = 1.0 / np.diff(crossing).mean()

    # Phase of the lift harmonic at the heave frequency, relative to y = A sin(2 pi f t)
    omega = 2.0 * np.pi * FREQUENCY
    harmonic = np.sum(fluctuation * np.exp(-1j * omega * time)) / len(time)
    return {
        "mean drag": forces["drag_coefficient"][window].mean(),
        "lift amplitude": 0.5 * np.ptp(lift),
        "lift frequency / heave frequency": frequency / FREQUENCY,
        "lift phase to motion [deg]": np.degrees(np.angle(harmonic * 1j)),
    }


def main():
    reference = read_forces(CASE_DIR / "reference_flow/samples/forces_history.csv")
    coupled = read_forces(CASE_DIR / "samples/forces_history.csv")

    # Locked-in statistics and the coupled deviation from the reference
    expected, measured = lock_in_statistics(reference), lock_in_statistics(coupled)
    print(f"Locked-in wake, t >= {LOCKED_IN:g} s")
    for name in expected:
        if name.endswith("[deg]"):
            deviation = f"{measured[name] - expected[name]:+.1f} deg"
        else:
            deviation = f"{100.0 * (measured[name] / expected[name] - 1.0):+.2f}%"
        print(
            f"  {name:34s} reference {expected[name]:.4f}  "
            f"coupled {measured[name]:.4f}  ({deviation})"
        )

    figure, axes = validation_subplots(2, sharex=True)
    for axis, name, label in zip(
        axes, ("drag_coefficient", "lift_coefficient"), ("$C_D$", "$C_L$")
    ):
        axis.plot(reference["time"], reference[name], **REFERENCE_STYLE, label="Reference FVM")
        axis.plot(
            coupled["time"],
            coupled[name],
            color=COLORS["hybrid"],
            linewidth=LINE_WIDTH,
            label="Coupled",
        )
        axis.set_ylabel(label)
    axes[1].set_xlabel("$t$ [s]")
    validation_legend(figure, axes[0])
    save_validation_figure(figure, CASE_DIR / "figures/forces")


if __name__ == "__main__":
    main()
