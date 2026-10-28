#!/usr/bin/env python3
"""Wall time per second of flow time of the coupled and the reference solution."""

import json
from pathlib import Path

from matplotlib.patheffects import withStroke
import numpy as np

from openonda.plotting import COLORS, LINE_WIDTH, save_validation_figure, validation_legend
from openonda.plotting import validation_subplots

CASE_DIR = Path(__file__).resolve().parents[1]
REFERENCE_DIR = CASE_DIR / "reference_flow"

# Recorded timing phases, grouped by the solver part they belong to
PHASES = {
    "VPM": (("vpm",), COLORS["vpm"]),
    "FVM": (("fvm",), COLORS["fvm"]),
    "Coupling": (("vpm_boundary_condition", "transfer"), COLORS["hybrid"]),
    "Output": (("state_checks_and_samplers", "backup", "reporting"), COLORS["gray"]),
}


def main():
    # One record per accepted coupling step; a running case may still be writing the last line
    lines = (CASE_DIR / "solution/coupler_diagnostics.jsonl").read_text().split("\n")[:-1]
    records = [json.loads(line) for line in lines]
    time = np.array([record["time"] for record in records])
    particles = np.array([record["transfer"]["n_particles_after"] for record in records])
    # Wall time per second of flow time: each coupling step advances the flow by one interval
    coupling_interval = time[1] - time[0]
    cost = [
        sum(np.array([record["timing_seconds"][key] for record in records]) for key in keys)
        / coupling_interval
        for keys, _ in PHASES.values()
    ]
    coupled_mean = np.sum(cost, axis=0).mean()

    # Reference FVM wall time per step over its own time step
    steps = [json.loads(line) for line in (REFERENCE_DIR / "solution/performance.jsonl").open()]
    reference_mean = (
        np.mean([step["step_seconds"]["max"] for step in steps]) / steps[0]["time_step_size"]
    )
    print(
        f"Wall time per second of flow time: coupled {coupled_mean:.0f} s, "
        f"reference FVM {reference_mean:.0f} s"
    )

    figure, axes = validation_subplots(2, sharex=True)
    axes[0].stackplot(time, *cost, labels=PHASES, colors=[color for _, color in PHASES.values()])
    # Clip the start-up coupling step and isolated spikes; leave headroom for the legend
    axes[0].set_ylim(0.0, 1.6 * np.percentile(np.sum(cost, axis=0), 99.9))
    for mean, style, label in (
        (coupled_mean, "-", "Coupled"),
        (reference_mean, "--", "Reference FVM"),
    ):
        axes[0].axhline(
            mean,
            color=COLORS["text"],
            linestyle=style,
            linewidth=LINE_WIDTH,
            path_effects=[withStroke(linewidth=2.5 * LINE_WIDTH, foreground="white")],
        )
        axes[0].annotate(
            f"{label}: {mean:.0f} s",
            xy=(0.01, mean),
            xycoords=axes[0].get_yaxis_transform(),
            xytext=(0, 3),
            textcoords="offset points",
            verticalalignment="bottom",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.0},
        )
    axes[0].set_ylabel("Wall time per\nflow time [s/s]")
    axes[1].plot(time, particles / 1e6, color=COLORS["vpm"], linewidth=LINE_WIDTH)
    axes[1].set_ylabel("Particles [$10^6$]")
    axes[1].set_xlabel("$t$ [s]")
    validation_legend(figure, axes[0])
    save_validation_figure(figure, CASE_DIR / "figures/coupling_cost")


if __name__ == "__main__":
    main()
