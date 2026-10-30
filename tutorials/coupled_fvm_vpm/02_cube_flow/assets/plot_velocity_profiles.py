#!/usr/bin/env python3
"""Wake velocity along the centreline and y = 0.75 D: time-mean, and at every common sample
time."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from openonda.plotting import COLORS, LINE_WIDTH, REFERENCE_STYLE, figure_size, save_fig
from openonda.plotting import centered_subplots_adjust, set_thesis_style
from openonda.saved_times import match_saved_times

CASE_DIR = Path(__file__).resolve().parents[1]
DEVELOPED_SHEDDING = 30.0  # s, start of the averaging window
STATISTICS_END = 60.0  # s, end of the coupled run; the reference is cut to the same window
CUBE_SIDE = 1.0  # m
FREESTREAM_SPEED = 1.0  # m/s
X_RANGE = (-3.0, 12.0)  # m, compared stretch of the lines

# Probe data from the reference, the coupled FVM box, and the particle wake.
LINES = {"centreline": "$y/D=0$", "offaxis_y075": "$y/D=0.75$"}
SOURCES = {
    "Reference FVM": "reference_flow/samples/grid_h0045/{}.csv",
    "Coupled FVM": "samples/fvm_{}.csv",
    "Coupled VPM": "samples/vpm_{}.csv",
}
STYLES = {
    "Reference FVM": REFERENCE_STYLE,
    "Coupled FVM": {"color": COLORS["fvm"], "linewidth": LINE_WIDTH},
    "Coupled VPM": {"color": COLORS["vpm"], "linewidth": LINE_WIDTH},
}


def inside_cube(x, y):
    """Line points inside the cube."""
    return (np.abs(x) <= 0.5 * CUBE_SIDE) & (abs(y) <= 0.5 * CUBE_SIDE)


def wake_profile(path):
    """Time-mean streamwise velocity and rms transverse velocity at each sampled x."""
    samples = np.genfromtxt(path, delimiter=",", names=True)
    window = (samples["time"] >= DEVELOPED_SHEDDING) & (samples["time"] <= STATISTICS_END)
    x, point = np.unique(samples["position_x"][window].round(6), return_inverse=True)
    count = np.bincount(point)

    u = samples["velocity_x"][window]
    mean_u = np.bincount(point, u) / count
    transverse = 0.0
    for component in ("velocity_y", "velocity_z"):
        value = samples[component][window]
        mean = np.bincount(point, value) / count
        transverse = transverse + np.bincount(point, (value - mean[point]) ** 2) / count

    # Mask the solid cube.
    mean_u[inside_cube(x, samples["position_y"][0])] = np.nan
    return x / CUBE_SIDE, mean_u / FREESTREAM_SPEED, np.sqrt(transverse) / FREESTREAM_SPEED


def profile_figure(curves, labels):
    """Two rows of profiles along both lines, with a shared legend."""
    figure, axes = plt.subplots(2, 2, figsize=figure_size("stacked"), sharex=True, sharey="row")
    centered_subplots_adjust(figure, outer=0.13, bottom=0.2, top=0.97, hspace=0.12, wspace=0.12)
    for column, (line, position) in enumerate(LINES.items()):
        for label, x, upper, lower in curves[line]:
            axes[0, column].plot(x, upper, **STYLES[label], label=label)
            axes[1, column].plot(x, lower, **STYLES[label], label=label)
        axes[1, column].set_xlabel(f"$x/D$ at {position}")
        axes[1, column].set_xlim(X_RANGE[0] / CUBE_SIDE, X_RANGE[1] / CUBE_SIDE)
    axes[0, 0].set_ylabel(labels[0])
    axes[1, 0].set_ylabel(labels[1])
    handles, names = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, names, loc="lower center", ncol=3, frameon=False, handlelength=1.5)
    return figure, axes


def instantaneous_profiles(samples):
    """Streamwise and transverse velocity at every sample time shared by all probes."""
    # All profile frames share one range for each velocity component.
    shown = {
        key: (table["position_x"] >= X_RANGE[0]) & (table["position_x"] <= X_RANGE[1])
        for key, table in samples.items()
    }
    limits = [
        (
            min(table[name][shown[key]].min() for key, table in samples.items()),
            max(table[name][shown[key]].max() for key, table in samples.items()),
        )
        for name in ("velocity_x", "velocity_y")
    ]
    clocks = {key: np.unique(table["time"]) for key, table in samples.items()}
    common = match_saved_times(*clocks.values())
    for frame, time in enumerate(common.times):
        curves = {line: [] for line in LINES}
        for (line, label), indices in zip(samples, common.indices):
            table = samples[line, label]
            rows = table["time"] == clocks[line, label][indices[frame]]
            order = np.argsort(table["position_x"][rows])
            x = table["position_x"][rows][order]
            # Mask the solid cube.
            fluid = np.where(inside_cube(x, table["position_y"][0]), np.nan, 1.0) / FREESTREAM_SPEED
            curves[line].append(
                (
                    label,
                    x / CUBE_SIDE,
                    table["velocity_x"][rows][order] * fluid,
                    table["velocity_y"][rows][order] * fluid,
                )
            )
        figure, axes = profile_figure(curves, (r"$u_x/U_\infty$", r"$u_y/U_\infty$"))
        for row, (low, high) in enumerate(limits):
            margin = 0.05 * (high - low)
            axes[row, 0].set_ylim(
                (low - margin) / FREESTREAM_SPEED, (high + margin) / FREESTREAM_SPEED
            )
        save_fig(figure, CASE_DIR / f"figures/velocity_profiles/t{time:06.2f}")


def main():
    set_thesis_style()
    curves = {
        line: [
            (label, *wake_profile(CASE_DIR / pattern.format(line)))
            for label, pattern in SOURCES.items()
        ]
        for line in LINES
    }
    figure, _ = profile_figure(curves, (r"$\overline{u}/U_\infty$", r"$\sqrt{v'^2+w'^2}/U_\infty$"))
    save_fig(figure, CASE_DIR / "figures/velocity_profiles")

    samples = {
        (line, label): np.genfromtxt(CASE_DIR / pattern.format(line), delimiter=",", names=True)
        for line in LINES
        for label, pattern in SOURCES.items()
    }
    instantaneous_profiles(samples)


if __name__ == "__main__":
    main()
