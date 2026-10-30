#!/usr/bin/env python3
"""Wall time per second of flow time of the coupled and the reference solution."""

import json
from pathlib import Path

from matplotlib.patheffects import withStroke
import numpy as np

from openonda.plotting import COLORS, LINE_WIDTH, save_validation_figure, validation_legend
from openonda.plotting import validation_subplots

CASE_DIR = Path(__file__).resolve().parents[1]
REFERENCE_PERFORMANCE = CASE_DIR / "reference_flow/solution/grid_h0045/performance.jsonl"
CUBE_SIDE = 1.0  # m
FREESTREAM_SPEED = 1.0  # m/s
FLOW_TIME_SCALE = FREESTREAM_SPEED / CUBE_SIDE  # 1/s, convective-time scale

# Timing phases grouped by solver component.
PHASES = {
    "VPM": (("vpm",), COLORS["vpm"]),
    "FVM": (("fvm",), COLORS["fvm"]),
    "Coupling": (("vpm_boundary_condition", "transfer"), COLORS["hybrid"]),
    "Output": (("state_checks_and_samplers", "backup", "reporting"), COLORS["gray"]),
}


def main():
    records = [
        json.loads(line)
        for line in (CASE_DIR / "solution/coupler_diagnostics.jsonl").read_text().splitlines()
    ]
    time = np.array([record["time"] for record in records])
    particles = np.array([record["transfer"]["n_particles_after"] for record in records])
    # Normalize each step by the physical time it advances.
    flow_interval = np.diff(time, prepend=0.0)
    wall_seconds = [
        sum(np.array([record["timing_seconds"][key] for record in records]) for key in keys)
        for keys, _ in PHASES.values()
    ]
    cost = [phase / flow_interval for phase in wall_seconds]
    coupled_mean = np.sum(wall_seconds) / np.sum(flow_interval)

    reference_steps = [json.loads(line) for line in REFERENCE_PERFORMANCE.read_text().splitlines()]
    wall_time = sum(step["step_seconds"]["max"] for step in reference_steps)
    flow_time = sum(step["time_step_size"] for step in reference_steps)
    reference_mean = wall_time / flow_time
    print(
        f"Wall time per second of flow time: coupled {coupled_mean:.0f} s, "
        f"reference FVM {reference_mean:.0f} s"
    )

    figure, axes = validation_subplots(2, sharex=True)
    axes[0].stackplot(
        time * FLOW_TIME_SCALE,
        *cost,
        labels=PHASES,
        colors=[color for _, color in PHASES.values()],
    )
    # Leave headroom above the timing history for the legend.
    axes[0].set_ylim(0.0, 1.6 * np.percentile(np.sum(cost, axis=0), 99.9))
    axes[0].axhline(
        reference_mean,
        color=COLORS["text"],
        linestyle="--",
        linewidth=LINE_WIDTH,
        path_effects=[withStroke(linewidth=2.5 * LINE_WIDTH, foreground="white")],
    )
    axes[0].annotate(
        f"Reference FVM: {reference_mean:.0f} s",
        xy=(0.01, reference_mean),
        xycoords=axes[0].get_yaxis_transform(),
        xytext=(0, 3),
        textcoords="offset points",
        verticalalignment="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.0},
    )
    axes[0].set_ylabel("Wall time per\nflow time [s/s]")
    axes[1].plot(
        time * FLOW_TIME_SCALE, particles / 1e6, color=COLORS["vpm"], linewidth=LINE_WIDTH
    )
    axes[1].set_ylabel("Particles [$10^6$]")
    axes[1].set_xlabel(r"$t U_\infty / D$")
    validation_legend(figure, axes[0])
    save_validation_figure(figure, CASE_DIR / "figures/coupling_cost")


if __name__ == "__main__":
    main()
