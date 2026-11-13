#!/usr/bin/env python3
"""Time-mean wake velocity at x = 2 c and 5 c, across the mid-span wake and along the span:
FVM-only reference against the coupled FVM box and particle wake, over the statistics window.

Writes figures/velocity_profiles/transverse_mean and spanwise_mean. Waits, with a message, until
both runs cover part of the window.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from openonda.plotting import CM, COLORS, LINE_WIDTH, MAX_FIGURE_WIDTH_CM, REFERENCE_STYLE
from openonda.plotting import centered_subplots_adjust, fit_thesis_y_label_margins
from openonda.plotting import save_validation_figure, set_thesis_style

CASE_DIR = Path(__file__).absolute().parents[1]
STATISTICS_WINDOW = (12.0, 24.0)  # s, the averaging window
MINIMUM_LENGTH = 4.0  # s of the window that both runs must cover
CHORD = 1.0  # m
FREESTREAM_SPEED = 1.0  # m/s

# Data sources at each station: reference FVM, coupled FVM box, and particle wake.
STATIONS = {2: ("FVM only", "FVM-VPM, FVM", "FVM-VPM, VPM"), 5: ("FVM only", "FVM-VPM, VPM")}
PREFIX = {
    "FVM only": "reference_flow/samples/",
    "FVM-VPM, FVM": "samples/fvm_",
    "FVM-VPM, VPM": "samples/vpm_",
}
STYLES = {
    "FVM only": REFERENCE_STYLE,
    "FVM-VPM, FVM": {"color": COLORS["fvm"], "linewidth": LINE_WIDTH},
    "FVM-VPM, VPM": {"color": COLORS["vpm"], "linewidth": LINE_WIDTH},
}

# Probe coordinate and velocity components on each line.
LINES = {
    "transverse": ("position_y", "$y/c$", ("velocity_x", "velocity_y"), ("u_x", "u_y")),
    "spanwise": ("position_z", "$z/c$", ("velocity_y", "velocity_z"), ("u_y", "u_z")),
}


def profile(table, coordinate, components, rows):
    """Positions along the line and the scaled components, averaged over the selected rows."""
    position, point = np.unique(table[coordinate][rows].round(6), return_inverse=True)
    count = np.bincount(point)
    values = [np.bincount(point, table[name][rows]) / count / FREESTREAM_SPEED for name in components]
    return position / CHORD, values


def main():
    paths = {(line, station, label): CASE_DIR / f"{PREFIX[label]}{line}_x{station}.csv"
             for line in LINES for station, labels in STATIONS.items() for label in labels}
    missing = sorted({str(path.relative_to(CASE_DIR)) for path in paths.values() if not path.exists()})
    if missing:
        print("velocity_profiles: wait for " + ", ".join(missing))
        return
    samples = {key: np.genfromtxt(path, delimiter=",", names=True) for key, path in paths.items()}
    end = min(STATISTICS_WINDOW[1], *(table["time"][-1] for table in samples.values()))
    if end - STATISTICS_WINDOW[0] < MINIMUM_LENGTH:
        print(f"velocity_profiles: wait until both runs pass t = "
              f"{STATISTICS_WINDOW[0] + MINIMUM_LENGTH:g} s (at {end:g} s)")
        return

    for line, (coordinate, axis_label, components, symbols) in LINES.items():
        set_thesis_style()
        figure, axes = plt.subplots(2, 2, figsize=(MAX_FIGURE_WIDTH_CM * CM, 10.0 * CM),
                                    sharex="col", sharey="row")
        centered_subplots_adjust(figure, outer=0.13, bottom=0.14, top=0.85, hspace=0.12, wspace=0.1)
        for column, (station, labels) in enumerate(STATIONS.items()):
            reference = None
            for label in labels:
                table = samples[line, station, label]
                rows = (table["time"] >= STATISTICS_WINDOW[0]) & (table["time"] <= end)
                position, values = profile(table, coordinate, components, rows)
                for row in range(2):
                    axes[row, column].plot(position, values[row], **STYLES[label], label=label)
                if reference is None:
                    reference = (position, values)
                    continue
                difference = np.hypot(*(v - np.interp(position, reference[0], r)
                                        for v, r in zip(values, reference[1])))
                print(f"{line} x/c = {station}, {label}: time-mean velocity difference rms "
                      f"{100 * np.sqrt(np.mean(difference**2)):.2g} %, max {100 * difference.max():.2g} % U")
            axes[1, column].set_xlabel(f"{axis_label} at $x/c={station}$")
        for row, symbol in enumerate(symbols):
            axes[row, 0].set_ylabel(rf"$\overline{{{symbol}}}/U_\infty$")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", ncol=3, frameon=True, edgecolor="0.8",
                      bbox_to_anchor=(0.5, 0.995), handlelength=1.5, columnspacing=0.8)
        fit_thesis_y_label_margins(figure, figure.axes)
        save_validation_figure(figure, CASE_DIR / f"figures/velocity_profiles/{line}_mean",
                               figure_format="both")
    print(f"velocity_profiles: means over {STATISTICS_WINDOW[0]:g}-{end:g} s")


if __name__ == "__main__":
    main()
