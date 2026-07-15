#!/usr/bin/env python3
"""Plot Taylor–Green energy decay and solver-error histories."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    data = np.genfromtxt(args.history, delimiter=",", names=True)
    data = np.atleast_1d(data)
    time = data["time"]

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 7.2), sharex=True)
    axes[0].plot(time, data["kinetic_energy"], "o-", label="PIMPLE")
    axes[0].plot(time, data["analytic_energy"], "k--", label="analytic")
    axes[0].set_ylabel("kinetic energy")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].semilogy(time, np.maximum(data["velocity_l2_error"], 1e-16), label="velocity L2")
    axes[1].semilogy(time, np.maximum(data["energy_relative_error"], 1e-16), label="energy")
    axes[1].semilogy(time, np.maximum(data["enstrophy_relative_error"], 1e-16), label="enstrophy")
    axes[1].semilogy(time, np.maximum(data["continuity_max"], 1e-16), label="max |div U|")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("error")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi)


if __name__ == "__main__":
    main()
