#!/usr/bin/env python3
"""Wall time per second of flow time of the coupled and the FVM-only runs of each case."""

import json

from matplotlib.patheffects import withStroke
import numpy as np

from openonda.plotting import COLORS, LINE_WIDTH, save_validation_figure, validation_legend
from openonda.plotting import validation_subplots

from assets.response import CASE_DIR, completed_runs, variant

REFERENCE_DIR = CASE_DIR / "reference_flow"

# Timing phases grouped by solver component.
PHASES = {
    "VPM": (("vpm",), COLORS["vpm"]),
    "FVM": (("fvm",), COLORS["fvm"]),
    "Coupling": (("vpm_boundary_condition", "transfer"), COLORS["hybrid"]),
    "Output": (("state_checks_and_samplers", "backup", "reporting"), COLORS["gray"]),
}


def reference_cost(run):
    """Mean wall time per second of flow time of the FVM-only run, or None without one."""
    if run not in completed_runs(REFERENCE_DIR / "samples"):
        return None
    performance = REFERENCE_DIR / "solution" / variant(*run) / "performance.jsonl"
    steps = [json.loads(line) for line in performance.read_text().splitlines()]
    return np.mean([step["step_seconds"]["max"] for step in steps]) / steps[0]["time_step_size"]


def main():
    for run in completed_runs(CASE_DIR / "samples"):
        diagnostics = CASE_DIR / "solution" / variant(*run) / "coupler_diagnostics.jsonl"
        records = [json.loads(line) for line in diagnostics.read_text().splitlines()]
        time = np.array([record["time"] for record in records])
        particles = np.array([record["transfer"]["n_particles_after"] for record in records])
        # Normalize every timing phase by the coupling interval.
        coupling_interval = time[1] - time[0]
        cost = [
            sum(np.array([record["timing_seconds"][key] for record in records]) for key in keys)
            / coupling_interval
            for keys, _ in PHASES.values()
        ]
        coupled_mean = np.sum(cost, axis=0).mean()
        means = [(coupled_mean, "-", "Coupled")]
        reference_mean = reference_cost(run)
        report = (
            f"Re = {run[0]:g}, V* = {run[1]:g}: wall time per second of flow time, "
            f"coupled {coupled_mean:.0f} s"
        )
        if reference_mean is not None:
            means.append((reference_mean, "--", "FVM only"))
            report += f", FVM only {reference_mean:.0f} s"
        print(report)

        figure, axes = validation_subplots(2, sharex=True)
        axes[0].stackplot(
            time, *cost, labels=PHASES, colors=[color for _, color in PHASES.values()]
        )
        # Leave headroom above the timing history for the legend.
        axes[0].set_ylim(0.0, 1.6 * np.percentile(np.sum(cost, axis=0), 99.9))
        for mean, style, label in means:
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
        save_validation_figure(figure, CASE_DIR / f"figures/coupling_cost_{variant(*run)}")


if __name__ == "__main__":
    main()
