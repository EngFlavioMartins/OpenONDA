#!/usr/bin/env python3
"""Wake velocity profiles at x = 2 c and 5 c, across the mid-span wake and along the span:
time-mean and at every common sample time of the statistics window."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

from openonda.plotting import COLORS, LINE_WIDTH, REFERENCE_STYLE, figure_size, save_fig
from openonda.plotting import centered_subplots_adjust, set_thesis_style
from openonda.saved_times import match_saved_times

CASE_DIR = Path(__file__).resolve().parents[1]
STATISTICS_START = 12.0  # s, start of the averaging window
CHORD = 1.0  # m
FREESTREAM_SPEED = 1.0  # m/s

# Data sources at each station: reference FVM, coupled FVM box, and particle wake.
STATIONS = {2: ("Reference FVM", "Coupled FVM", "Coupled VPM"), 5: ("Reference FVM", "Coupled VPM")}
PREFIX = {
    "Reference FVM": "reference_flow/samples/",
    "Coupled FVM": "samples/fvm_",
    "Coupled VPM": "samples/vpm_",
}
STYLES = {
    "Reference FVM": REFERENCE_STYLE,
    "Coupled FVM": {"color": COLORS["fvm"], "linewidth": LINE_WIDTH},
    "Coupled VPM": {"color": COLORS["vpm"], "linewidth": LINE_WIDTH},
}

# Probe coordinate and velocity components on each line.
LINES = {
    "transverse": ("position_y", "$y/c$", ("velocity_x", "velocity_y"), ("$u_x$", "$u_y$")),
    "spanwise": ("position_z", "$z/c$", ("velocity_y", "velocity_z"), ("$u_y$", "$u_z$")),
}


def read_samples(line):
    """Sample tables of one probe line, keyed by station and source."""
    return {
        (station, label): np.genfromtxt(
            CASE_DIR / f"{PREFIX[label]}{line}_x{station}.csv", delimiter=",", names=True
        )
        for station, labels in STATIONS.items()
        for label in labels
    }


def profile(table, coordinate, components, rows):
    """Positions along the line and the scaled components, averaged over the selected rows."""
    position, point = np.unique(table[coordinate][rows].round(6), return_inverse=True)
    count = np.bincount(point)
    values = [
        np.bincount(point, table[name][rows]) / count / FREESTREAM_SPEED for name in components
    ]
    return position / CHORD, values


def profile_figure(curves, axis_label, labels):
    """Two rows of profiles at both stations, with a shared legend."""
    figure, axes = plt.subplots(2, 2, figsize=figure_size("stacked"), sharex="col", sharey="row")
    centered_subplots_adjust(figure, outer=0.13, bottom=0.2, top=0.97, hspace=0.12, wspace=0.12)
    for column, station in enumerate(STATIONS):
        for label, position, values in curves[station]:
            for row in range(2):
                axes[row, column].plot(position, values[row], **STYLES[label], label=label)
        axes[1, column].set_xlabel(f"{axis_label} at $x/c={station}$")
    for row in range(2):
        axes[row, 0].set_ylabel(labels[row])
    for axis in axes.flat:
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    handles, names = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, names, loc="lower center", ncol=3, frameon=False, handlelength=1.5)
    return figure, axes


def main():
    set_thesis_style()
    for line, (coordinate, axis_label, components, symbols) in LINES.items():
        samples = read_samples(line)

        # Compare time-mean coupled profiles with the reference.
        curves = {station: [] for station in STATIONS}
        for (station, label), table in samples.items():
            mean = profile(table, coordinate, components, table["time"] >= STATISTICS_START)
            curves[station].append((label, *mean))
        for station, entries in curves.items():
            reference_position, reference_values = entries[0][1:]
            for label, position, values in entries[1:]:
                difference = np.hypot(
                    *(
                        v - np.interp(position, reference_position, r)
                        for v, r in zip(values, reference_values)
                    )
                )
                print(
                    f"{line} x/c = {station}, {label}: time-mean velocity difference "
                    f"rms {100.0 * np.sqrt(np.mean(difference**2)):.2g}%, max {100.0 * difference.max():.2g}% U"
                )
        mean_labels = [rf"$\overline{{{symbol[1:-1]}}}/U_\infty$" for symbol in symbols]
        figure, _ = profile_figure(curves, axis_label, mean_labels)
        save_fig(figure, CASE_DIR / f"figures/velocity_profiles/{line}_mean")

        # All profile frames share one range for each velocity component.
        limits = [
            (
                min(table[name].min() for table in samples.values()) / FREESTREAM_SPEED,
                max(table[name].max() for table in samples.values()) / FREESTREAM_SPEED,
            )
            for name in components
        ]
        clocks = {key: np.unique(table["time"]) for key, table in samples.items()}
        common = match_saved_times(*clocks.values())
        for frame, time in enumerate(common.times):
            if time < STATISTICS_START:
                continue
            curves = {station: [] for station in STATIONS}
            for (station, label), indices in zip(samples, common.indices):
                table = samples[station, label]
                rows = table["time"] == clocks[station, label][indices[frame]]
                curves[station].append((label, *profile(table, coordinate, components, rows)))
            figure, axes = profile_figure(
                curves, axis_label, [rf"{s[:-1]}/U_\infty$" for s in symbols]
            )
            for row, (low, high) in enumerate(limits):
                margin = 0.05 * (high - low)
                axes[row, 0].set_ylim(low - margin, high + margin)
            save_fig(figure, CASE_DIR / f"figures/velocity_profiles/{line}/t{time:06.2f}")


if __name__ == "__main__":
    main()
