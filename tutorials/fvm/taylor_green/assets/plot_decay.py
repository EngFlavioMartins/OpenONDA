#!/usr/bin/env python3
"""Plot Taylor–Green energy decay and solver-error histories."""

import argparse
from pathlib import Path


import matplotlib.pyplot as plt
from openonda.plotting import centered_subplots_adjust
import numpy as np

from openonda import plotting as theme
from openonda.results import read_history_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=theme.DEFAULT_DPI)
    args = parser.parse_args()

    theme.set_thesis_style()
    data = read_history_table(args.history)
    time = data["time"]

    fig, axes = plt.subplots(2, 1, figsize=theme.figure_size("stacked"), sharex=True)
    axes[0].plot(time, data["total_kinetic_energy"], "o-", label="PIMPLE")
    axes[0].plot(
        time,
        data["analytic_total_kinetic_energy"],
        "--",
        color=theme.COLORS["reference"],
        label="Analytic",
    )
    axes[0].set_ylabel("kinetic energy")
    axes[0].legend()
    axes[0].grid(False)

    axes[1].semilogy(time, np.maximum(data["velocity_l2_error"], 1e-16), label="velocity L2")
    axes[1].semilogy(
        time,
        np.maximum(data["total_kinetic_energy_relative_error"], 1e-16),
        label="total kinetic energy",
    )
    axes[1].semilogy(
        time, np.maximum(data["total_enstrophy_relative_error"], 1e-16), label="total enstrophy"
    )
    axes[1].semilogy(
        time,
        np.maximum(data["max_continuity_error"], 1e-16),
        label="maximum continuity error",
    )
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("error")
    axes[1].legend()
    axes[1].grid(False)

    centered_subplots_adjust(fig, outer=0.135, bottom=0.14, top=0.95, hspace=0.28)
    theme.export_figure(fig, args.output, dpi=args.dpi)
    plt.close(fig)


if __name__ == "__main__":
    main()
