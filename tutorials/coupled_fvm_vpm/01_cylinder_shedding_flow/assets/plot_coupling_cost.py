#!/usr/bin/env python3
"""Wall time of each FVM–VPM exchange and the number of vortex particles."""

import json
from pathlib import Path

import numpy as np

from openonda.plotting import COLORS, LINE_WIDTH, save_validation_figure, validation_legend
from openonda.plotting import validation_subplots

CASE_DIR = Path(__file__).resolve().parents[1]

# Recorded timing phases, grouped by the solver part they belong to
PHASES = {
    "VPM": (("vpm",), COLORS["vpm"]),
    "FVM": (("fvm",), COLORS["fvm"]),
    "Coupling": (("vpm_boundary_condition", "transfer"), COLORS["hybrid"]),
    "Output": (("state_checks_and_samplers", "backup", "reporting"), COLORS["gray"]),
}


def main():
    # One record per accepted exchange; a running case may still be writing the last line
    lines = (CASE_DIR / "solution/coupler_diagnostics.jsonl").read_text().split("\n")[:-1]
    records = [json.loads(line) for line in lines]
    time = np.array([record["time"] for record in records])
    particles = np.array([record["transfer"]["n_particles_after"] for record in records])
    cost = [
        sum(np.array([record["timing_seconds"][key] for record in records]) for key in keys)
        for keys, _ in PHASES.values()
    ]

    figure, axes = validation_subplots(2, sharex=True)
    axes[0].stackplot(time, *cost, labels=PHASES, colors=[color for _, color in PHASES.values()])
    axes[0].set_ylim(0.0, 1.6 * np.max(np.sum(cost, axis=0)))  # headroom for the legend
    axes[0].set_ylabel("Wall time [s]")
    axes[1].plot(time, particles / 1e6, color=COLORS["vpm"], linewidth=LINE_WIDTH)
    axes[1].set_ylabel("Particles [$10^6$]")
    axes[1].set_xlabel("$t$ [s]")
    validation_legend(figure, axes[0])
    save_validation_figure(figure, CASE_DIR / "figures/coupling_cost")


if __name__ == "__main__":
    main()
