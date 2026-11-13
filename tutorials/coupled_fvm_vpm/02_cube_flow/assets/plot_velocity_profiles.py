#!/usr/bin/env python3
"""Wake velocity along the centreline and y = 0.75 D: time mean and transverse fluctuation over the
statistics window, FVM-only reference against the coupled FVM box and particle wake.

Writes figures/velocity_profiles. Waits, with a message, until both runs cover part of the window.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from openonda.plotting import CM, COLORS, LINE_WIDTH, MAX_FIGURE_WIDTH_CM, REFERENCE_STYLE
from openonda.plotting import centered_subplots_adjust, fit_thesis_y_label_margins
from openonda.plotting import save_validation_figure, set_thesis_style

CASE_DIR = Path(__file__).absolute().parents[1]
STATISTICS_WINDOW = (30.0, 60.0)  # s, the averaging window
MINIMUM_LENGTH = 10.0  # s of the window that both runs must cover
CUBE_SIDE = 1.0  # m
FREESTREAM_SPEED = 1.0  # m/s
X_RANGE = (-3.0, 12.0)  # m, compared stretch of the lines

# Probe data from the reference, the coupled FVM box, and the particle wake.
LINES = {"centreline": "$y/D=0$", "offaxis_y075": "$y/D=0.75$"}
SOURCES = {
    "FVM only": "reference_flow/samples/grid_h0045_dt0.02/{}.csv",
    "FVM-VPM, FVM": "samples/fvm_{}.csv",
    "FVM-VPM, VPM": "samples/vpm_{}.csv",
}
STYLES = {
    "FVM only": REFERENCE_STYLE,
    "FVM-VPM, FVM": {"color": COLORS["fvm"], "linewidth": LINE_WIDTH},
    "FVM-VPM, VPM": {"color": COLORS["vpm"], "linewidth": LINE_WIDTH},
}


def inside_cube(x, y):
    """Line points inside the cube."""
    return (np.abs(x) <= 0.5 * CUBE_SIDE) & (abs(y) <= 0.5 * CUBE_SIDE)


def read(path):
    return np.genfromtxt(path, delimiter=",", names=True)


def wake_profile(samples, end):
    """Time-mean streamwise velocity and rms transverse velocity at each sampled x."""
    window = (samples["time"] >= STATISTICS_WINDOW[0]) & (samples["time"] <= end)
    x, point = np.unique(samples["position_x"][window].round(6), return_inverse=True)
    count = np.bincount(point)
    mean_u = np.bincount(point, samples["velocity_x"][window]) / count
    transverse = 0.0
    for component in ("velocity_y", "velocity_z"):
        value = samples[component][window]
        mean = np.bincount(point, value) / count
        transverse = transverse + np.bincount(point, (value - mean[point]) ** 2) / count
    fluctuation = np.sqrt(transverse)
    solid = inside_cube(x, samples["position_y"][0])
    mean_u[solid], fluctuation[solid] = np.nan, np.nan
    shown = (x >= X_RANGE[0]) & (x <= X_RANGE[1])
    return x[shown] / CUBE_SIDE, mean_u[shown] / FREESTREAM_SPEED, fluctuation[shown] / FREESTREAM_SPEED


def main():
    paths = {(line, label): CASE_DIR / pattern.format(line)
             for line in LINES for label, pattern in SOURCES.items()}
    missing = sorted({str(path.relative_to(CASE_DIR)) for path in paths.values() if not path.exists()})
    if missing:
        print("velocity_profiles: wait for " + ", ".join(missing))
        return
    samples = {key: read(path) for key, path in paths.items()}
    end = min(STATISTICS_WINDOW[1], *(table["time"][-1] for table in samples.values()))
    if end - STATISTICS_WINDOW[0] < MINIMUM_LENGTH:
        print(f"velocity_profiles: wait until both runs pass t = "
              f"{STATISTICS_WINDOW[0] + MINIMUM_LENGTH:g} s (at {end:g} s)")
        return

    set_thesis_style()
    figure, axes = plt.subplots(2, 2, figsize=(MAX_FIGURE_WIDTH_CM * CM, 10.0 * CM), sharex=True,
                                sharey="row")
    centered_subplots_adjust(figure, outer=0.13, bottom=0.14, top=0.85, hspace=0.12, wspace=0.1)
    for column, (line, position) in enumerate(LINES.items()):
        for label in SOURCES:
            x, mean_u, fluctuation = wake_profile(samples[line, label], end)
            axes[0, column].plot(x, mean_u, **STYLES[label], label=label)
            axes[1, column].plot(x, fluctuation, **STYLES[label], label=label)
        axes[1, column].set_xlabel(f"$x/D$ at {position}")
        axes[1, column].set_xlim(X_RANGE[0] / CUBE_SIDE, X_RANGE[1] / CUBE_SIDE)
    axes[0, 0].set_ylabel(r"$\overline{u}/U_\infty$")
    axes[1, 0].set_ylabel(r"$\sqrt{\overline{v'^2}+\overline{w'^2}}/U_\infty$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=True, edgecolor="0.8",
                  bbox_to_anchor=(0.5, 0.995), handlelength=1.5, columnspacing=0.8)
    fit_thesis_y_label_margins(figure, figure.axes)
    save_validation_figure(figure, CASE_DIR / "figures/velocity_profiles", figure_format="png")
    print(f"velocity_profiles: means over {STATISTICS_WINDOW[0]:g}-{end:g} s")


if __name__ == "__main__":
    main()
