#!/usr/bin/env python3
"""Drag, lift and pitching moment on the pitching airfoil: coupled FVM–VPM against the FVM only."""

from pathlib import Path

import numpy as np

from openonda.plotting import COLORS, LINE_WIDTH, REFERENCE_STYLE
from openonda.plotting import fit_thesis_y_label_margins, save_validation_figure
from openonda.plotting import validation_legend, validation_subplots

from assets.pitch_phase import FREQUENCY, harmonic

CASE_DIR = Path(__file__).resolve().parents[1]
PERIODIC = 15.0  # s, start of the statistics window
END = 30.0  # s, three whole pitch periods after PERIODIC
COEFFICIENTS = (
    ("drag_coefficient", "$C_D$"),
    ("lift_coefficient", "$C_L$"),
    ("pitching_moment_coefficient", "$C_M$"),
)


def read_forces(path):
    """Force-coefficient history, one row per sampled time."""
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")


def periodic_statistics(forces):
    """Mean drag and the first-harmonic amplitude and phase of lift and moment to the pitch."""
    # Whole pitch periods; the end time is left out so that each phase counts once
    window = (forces["time"] >= PERIODIC - 1e-9) & (forces["time"] < END - 1e-9)
    time = forces["time"][window]
    statistics = {"mean drag": forces["drag_coefficient"][window].mean()}
    for name, label in (("lift_coefficient", "lift"), ("pitching_moment_coefficient", "moment")):
        first = harmonic(time, forces[name][window])
        statistics[f"{label} amplitude"] = abs(first)
        statistics[f"{label} phase to pitch [deg]"] = np.degrees(np.angle(first))
    return statistics


def main():
    reference = read_forces(CASE_DIR / "reference_flow/samples/forces_history.csv")
    coupled = read_forces(CASE_DIR / "samples/forces_history.csv")

    expected, measured = periodic_statistics(reference), periodic_statistics(coupled)
    print(f"Periodic wake, {PERIODIC:g} <= t < {END:g} s, pitch frequency {FREQUENCY:g} Hz")
    for name in expected:
        if name.endswith("[deg]"):
            deviation = f"{measured[name] - expected[name]:+.1f} deg"
        else:
            deviation = f"{100.0 * (measured[name] / expected[name] - 1.0):+.2f}%"
        print(
            f"  {name:28s} FVM only {expected[name]:.4f}  "
            f"coupled {measured[name]:.4f}  ({deviation})"
        )

    figure, axes = validation_subplots(len(COEFFICIENTS), sharex=True)
    for axis, (name, label) in zip(axes, COEFFICIENTS):
        axis.plot(reference["time"], reference[name], **REFERENCE_STYLE, label="FVM only")
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
    fit_thesis_y_label_margins(figure, axes)
    save_validation_figure(figure, CASE_DIR / "figures/forces")


if __name__ == "__main__":
    main()
