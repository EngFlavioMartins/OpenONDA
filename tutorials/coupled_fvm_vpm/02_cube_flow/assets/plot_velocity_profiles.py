#!/usr/bin/env python3
"""Mean streamwise velocity and transverse fluctuation along the centreline and y = 0.75 D."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from openonda.plotting import COLORS, LINE_WIDTH, REFERENCE_STYLE, figure_size, save_fig
from openonda.plotting import centered_subplots_adjust, set_thesis_style

CASE_DIR = Path(__file__).resolve().parents[1]
DEVELOPED_SHEDDING = 40.0  # s, start of the averaging window
CUBE_SIDE = 1.0  # m
FREESTREAM_SPEED = 1.0  # m/s

# Reference FVM, coupled FVM (inside its box) and the coupled particles
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


def wake_profile(path):
    """Time-mean streamwise velocity and rms transverse velocity at each sampled x."""
    samples = np.genfromtxt(path, delimiter=",", names=True)
    window = samples["time"] >= DEVELOPED_SHEDDING
    x, point = np.unique(samples["position_x"][window].round(6), return_inverse=True)
    count = np.bincount(point)

    u = samples["velocity_x"][window]
    mean_u = np.bincount(point, u) / count
    transverse = 0.0
    for component in ("velocity_y", "velocity_z"):
        value = samples[component][window]
        mean = np.bincount(point, value) / count
        transverse = transverse + np.bincount(point, (value - mean[point]) ** 2) / count

    # The cube interior carries no flow
    mean_u[np.abs(x) <= 0.5 * CUBE_SIDE] = np.nan
    return x / CUBE_SIDE, mean_u / FREESTREAM_SPEED, np.sqrt(transverse) / FREESTREAM_SPEED


def main():
    set_thesis_style()
    figure, axes = plt.subplots(2, 2, figsize=figure_size("stacked"), sharex=True, sharey="row")
    centered_subplots_adjust(figure, outer=0.13, bottom=0.2, top=0.97, hspace=0.12, wspace=0.12)
    for column, (line, position) in enumerate(LINES.items()):
        for label, pattern in SOURCES.items():
            x, mean_u, rms = wake_profile(CASE_DIR / pattern.format(line))
            axes[0, column].plot(x, mean_u, **STYLES[label], label=label)
            axes[1, column].plot(x, rms, **STYLES[label], label=label)
        axes[1, column].set_xlabel(f"$x/D$ at {position}")
        axes[1, column].set_xlim(-3.0, 12.0)
    axes[0, 0].set_ylabel(r"$\overline{u}/U_\infty$")
    axes[1, 0].set_ylabel(r"$\sqrt{v'^2+w'^2}/U_\infty$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False, handlelength=1.5)
    save_fig(figure, CASE_DIR / "figures/velocity_profiles")


if __name__ == "__main__":
    main()
