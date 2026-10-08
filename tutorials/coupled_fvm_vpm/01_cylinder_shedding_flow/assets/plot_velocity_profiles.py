#!/usr/bin/env python3
"""Mean wake deficit and transverse velocity fluctuation at x = 2 D and 4 D."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from openonda.plotting import COLORS, LINE_WIDTH, REFERENCE_STYLE, figure_size, save_fig
from openonda.plotting import centered_subplots_adjust, set_thesis_style

CASE_DIR = Path(__file__).resolve().parents[1]
DEVELOPED_SHEDDING = 40.0  # s, start of the averaging window
DIAMETER = 1.0  # m
FREESTREAM_SPEED = 1.0  # m/s

# Reference FVM, coupled FVM (inside its box at x = 2 m) and the coupled particles
SOURCES = {
    2: {
        "Reference FVM": "reference_flow/samples/transverse_x2.csv",
        "Coupled FVM": "samples/fvm_transverse_x2.csv",
        "Coupled VPM": "samples/vpm_transverse_x2.csv",
    },
    4: {
        "Reference FVM": "reference_flow/samples/transverse_x4.csv",
        "Coupled VPM": "samples/vpm_transverse_x4.csv",
    },
}
STYLES = {
    "Reference FVM": REFERENCE_STYLE,
    "Coupled FVM": {"color": COLORS["fvm"], "linewidth": LINE_WIDTH},
    "Coupled VPM": {"color": COLORS["vpm"], "linewidth": LINE_WIDTH},
}


def wake_profile(path):
    """Time-mean streamwise velocity and rms transverse velocity at each sampled y."""
    samples = np.genfromtxt(path, delimiter=",", names=True)
    window = samples["time"] >= DEVELOPED_SHEDDING
    y, point = np.unique(samples["position_y"][window].round(6), return_inverse=True)
    count = np.bincount(point)

    u = samples["velocity_x"][window]
    v = samples["velocity_y"][window]
    mean_u = np.bincount(point, u) / count
    mean_v = np.bincount(point, v) / count
    rms_v = np.sqrt(np.bincount(point, (v - mean_v[point]) ** 2) / count)
    return y / DIAMETER, mean_u / FREESTREAM_SPEED, rms_v / FREESTREAM_SPEED


def main():
    set_thesis_style()
    figure, axes = plt.subplots(2, 2, figsize=figure_size("stacked"), sharex=True, sharey="row")
    centered_subplots_adjust(figure, outer=0.13, bottom=0.2, top=0.97, hspace=0.12, wspace=0.12)
    for column, (station, files) in enumerate(SOURCES.items()):
        for label, path in files.items():
            y, mean_u, rms_v = wake_profile(CASE_DIR / path)
            axes[0, column].plot(y, mean_u, **STYLES[label], label=label)
            axes[1, column].plot(y, rms_v, **STYLES[label], label=label)
        axes[1, column].set_xlabel(f"$y/D$ at $x/D={station}$")
    axes[0, 0].set_ylabel(r"$\overline{u}/U_\infty$")
    axes[1, 0].set_ylabel(r"$v_\mathrm{rms}/U_\infty$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False, handlelength=1.5)
    save_fig(figure, CASE_DIR / "figures/velocity_profiles")


if __name__ == "__main__":
    main()
