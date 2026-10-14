#!/usr/bin/env python3
"""Drag, lift and lift spectrum of the heaving cylinder.

Inside the lock-in region the wake sheds at the forcing frequency instead of
its natural frequency, so the lift spectrum peaks at the forcing frequency.
"""

import matplotlib.pyplot as plt
import numpy as np

from openonda.plotting import centered_subplots_adjust

from assets._common import COLORS, build_arg_parser, figure_size, read_force_history, save_fig
from setup import (
    AMPLITUDE,
    DIAMETER,
    FREESTREAM_VELOCITY,
    FREQUENCY,
    NATURAL_STROUHAL_NUMBER,
)


def main():
    args = build_arg_parser().parse_args()
    history = read_force_history("cylinder")
    time = history["time"]
    drag = history["drag_coefficient"]
    lift = history["lift_coefficient"]
    heave = AMPLITUDE * np.sin(2.0 * np.pi * FREQUENCY * time)

    # Statistics over the second half, after the start-up transient
    late = time >= 0.5 * time[-1]
    uniform_time = np.linspace(time[late][0], time[late][-1], late.sum())
    uniform_lift = np.interp(uniform_time, time[late], lift[late])
    frequencies = np.fft.rfftfreq(len(uniform_time), uniform_time[1] - uniform_time[0])
    spectrum = np.abs(np.fft.rfft(uniform_lift - uniform_lift.mean()))
    # Lift frequency from its upward zero crossings, finer than the spectrum bins
    fluctuation = lift[late] - lift[late].mean()
    rising = np.flatnonzero((fluctuation[:-1] < 0.0) & (fluctuation[1:] >= 0.0))
    crossing = (
        time[late][rising]
        - fluctuation[rising] * np.diff(time[late])[rising] / np.diff(fluctuation)[rising]
    )
    lift_frequency = (len(crossing) - 1) / (crossing[-1] - crossing[0])
    natural_frequency = NATURAL_STROUHAL_NUMBER * FREESTREAM_VELOCITY / DIAMETER

    print(f"mean drag coefficient        {np.mean(drag[late]):.3f}")
    print(f"rms lift coefficient         {np.std(lift[late]):.3f}")
    print(f"lift frequency               {lift_frequency:.4f} Hz")
    print(f"forcing frequency            {FREQUENCY:.4f} Hz")
    print(f"lift / forcing frequency     {lift_frequency / FREQUENCY:.3f}")

    fig, axes = plt.subplots(3, 1, figsize=figure_size("stacked_tall"))
    shown = time >= 1.0  # skip the impulsive start
    axes[0].plot(time[shown], drag[shown], color=COLORS["fvm"])
    axes[0].set_ylabel("drag coefficient")
    axes[0].tick_params(labelbottom=False)

    axes[1].plot(time[shown], lift[shown], color=COLORS["fvm"], label="lift coefficient")
    axes[1].plot(
        time[shown], heave[shown] / AMPLITUDE, color=COLORS["amber"], linestyle="--", label="y / A"
    )
    axes[1].set_xlabel("t [s]")
    axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    for ax in axes[:2]:
        ax.ticklabel_format(axis="x", style="plain")

    axes[2].plot(
        frequencies[1:] * DIAMETER / FREESTREAM_VELOCITY,
        spectrum[1:] / spectrum[1:].max(),
        color=COLORS["fvm"],
    )
    axes[2].axvline(
        FREQUENCY * DIAMETER / FREESTREAM_VELOCITY,
        color=COLORS["amber"],
        linestyle="--",
        label="forcing",
    )
    axes[2].axvline(
        natural_frequency * DIAMETER / FREESTREAM_VELOCITY,
        color=COLORS["reference"],
        linestyle=":",
        label="fixed cylinder",
    )
    axes[2].set_xlim(0.0, 1.0)
    axes[2].set_xlabel(r"$fD/U_\infty$")
    axes[2].set_ylabel("lift spectrum")
    axes[2].legend(loc="upper right")

    centered_subplots_adjust(fig, outer=0.16, bottom=0.10, top=0.96, hspace=0.45)
    save_fig(fig, "oscillating_cylinder_forces.png", args.format, args.dpi)


if __name__ == "__main__":
    main()
