#!/usr/bin/env python3
"""Gate the Widnall VPM challenge against a factor-two time-step refinement."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/openonda_stage5b_dt_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/openonda_stage5b_dt_cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CIRCULATION = np.pi
MODES = np.arange(20, 25)
INK = "#20252a"
BLUE = "#286f9b"
GOLD = "#d9973b"
GREY = "#8a99a8"
GRID = "#d8dde2"


def mode_history(run_directory: Path) -> tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(run_directory / "samples" / "ring_modes.csv")
    data = data[data["group_id"] == data["group_id"].min()]
    table = data.pivot_table(
        index="flow_time",
        columns="mode",
        values="combined_amplitude",
        aggfunc="last",
    ).sort_index()
    missing = [mode for mode in MODES if mode not in table]
    if missing:
        raise ValueError(f"missing modes in {run_directory}: {missing}")
    return table.index.to_numpy(float) * CIRCULATION, table[MODES].to_numpy(float)


def health_history(run_directory: Path) -> pd.DataFrame:
    data = pd.read_csv(run_directory / "samples" / "flow_integrals.csv")
    return data.sort_values("time").drop_duplicates("step", keep="last")


def ring_history(run_directory: Path) -> pd.DataFrame:
    data = pd.read_csv(run_directory / "samples" / "ring_diagnostics.csv")
    return data.sort_values("flow_time").drop_duplicates("time_step", keep="last")


def interpolate(time: np.ndarray, values: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [np.interp(target, time, values[:, column]) for column in range(values.shape[1])]
    )


def relative_l2(value: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(value - reference) / np.linalg.norm(reference))


def evaluate(coarse_directory: Path, fine_directory: Path) -> tuple[dict[str, object], dict]:
    coarse_time, coarse_modes = mode_history(coarse_directory)
    fine_time, fine_modes = mode_history(fine_directory)
    common_end = min(coarse_time[-1], fine_time[-1])
    selected = coarse_time <= common_end + 1.0e-12
    time = coarse_time[selected]
    coarse_modes = coarse_modes[selected]
    fine_at_coarse = interpolate(fine_time, fine_modes, time)
    mode_relative_error = np.linalg.norm(coarse_modes - fine_at_coarse, axis=1) / np.maximum(
        np.linalg.norm(fine_at_coarse, axis=1),
        np.finfo(float).tiny,
    )

    coarse_health = health_history(coarse_directory)
    fine_health = health_history(fine_directory)
    coarse_ring = ring_history(coarse_directory)
    fine_ring = ring_history(fine_directory)
    coarse_ring_time = coarse_ring["flow_time"].to_numpy(float) * CIRCULATION
    fine_ring_time = fine_ring["flow_time"].to_numpy(float) * CIRCULATION
    ring_target = coarse_ring_time[coarse_ring_time <= common_end + 1.0e-12]
    coarse_radius = np.interp(
        ring_target,
        coarse_ring_time,
        coarse_ring["major_radius"].to_numpy(float),
    )
    fine_radius = np.interp(
        ring_target,
        fine_ring_time,
        fine_ring["major_radius"].to_numpy(float),
    )
    coarse_x = np.interp(
        ring_target,
        coarse_ring_time,
        coarse_ring["x_centroid"].to_numpy(float),
    )
    fine_x = np.interp(
        ring_target,
        fine_ring_time,
        fine_ring["x_centroid"].to_numpy(float),
    )

    metrics = {
        "common_end_time_star": float(common_end),
        "mode_history_relative_l2": relative_l2(coarse_modes, fine_at_coarse),
        "mode_endpoint_relative_l2": relative_l2(coarse_modes[-1], fine_at_coarse[-1]),
        "maximum_instantaneous_mode_relative_l2": float(np.max(mode_relative_error)),
        "ring_radius_relative_l2": relative_l2(coarse_radius, fine_radius),
        "centroid_trajectory_relative_l2": relative_l2(coarse_x[1:], fine_x[1:]),
        "coarse_maximum_divergence": float(coarse_health["vorticity_divergence_error"].max()),
        "fine_maximum_divergence": float(fine_health["vorticity_divergence_error"].max()),
        "coarse_maximum_misalignment_deg": float(coarse_health["strength_misalignment_deg"].max()),
        "fine_maximum_misalignment_deg": float(fine_health["strength_misalignment_deg"].max()),
    }
    checks = {
        "mode_history_within_5_percent": metrics["mode_history_relative_l2"] < 0.05,
        "mode_endpoint_within_5_percent": metrics["mode_endpoint_relative_l2"] < 0.05,
        "ring_radius_within_0p5_percent": metrics["ring_radius_relative_l2"] < 0.005,
        "centroid_trajectory_within_0p5_percent": (
            metrics["centroid_trajectory_relative_l2"] < 0.005
        ),
        "both_divergence_below_limit": max(
            metrics["coarse_maximum_divergence"], metrics["fine_maximum_divergence"]
        )
        < 0.12,
        "both_misalignment_below_limit": max(
            metrics["coarse_maximum_misalignment_deg"],
            metrics["fine_maximum_misalignment_deg"],
        )
        < 45.0,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "comparison": "dt=0.005 against dt=0.0025",
        "modes": MODES.tolist(),
        "metrics": metrics,
        "checks": checks,
    }
    histories = {
        "time": time,
        "coarse_modes": coarse_modes,
        "fine_modes": fine_at_coarse,
        "mode_relative_error": mode_relative_error,
        "coarse_health": coarse_health,
        "fine_health": fine_health,
        "ring_time": ring_target,
        "coarse_radius": coarse_radius,
        "fine_radius": fine_radius,
    }
    return result, histories


def plot(histories: dict, output: Path) -> None:
    time = histories["time"]
    coarse = histories["coarse_modes"]
    fine = histories["fine_modes"]
    coarse_envelope = np.sqrt(np.mean(coarse**2, axis=1))
    fine_envelope = np.sqrt(np.mean(fine**2, axis=1))
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4), constrained_layout=True)

    axis = axes[0, 0]
    axis.plot(time, coarse_envelope, color=BLUE, label=r"$\Delta t=0.005$")
    axis.plot(time, fine_envelope, color=GOLD, linestyle="--", label=r"$\Delta t=0.0025$")
    axis.set_xlabel(r"$t^*=t\Gamma/R^2$")
    axis.set_ylabel(r"RMS amplitude, $m=20\ldots24$")
    axis.set_title("Widnall-band envelope")
    axis.legend(frameon=False)

    axis = axes[0, 1]
    axis.plot(time, 100.0 * histories["mode_relative_error"], color=BLUE)
    axis.axhline(5.0, color=INK, linestyle="--", label="5% gate")
    axis.set_xlabel(r"$t^*=t\Gamma/R^2$")
    axis.set_ylabel("instantaneous modal difference (%)")
    axis.set_title("Time-step sensitivity")
    axis.legend(frameon=False)

    axis = axes[1, 0]
    axis.plot(histories["ring_time"], histories["coarse_radius"], color=BLUE, label="coarse")
    axis.plot(
        histories["ring_time"],
        histories["fine_radius"],
        color=GOLD,
        linestyle="--",
        label="fine",
    )
    axis.set_xlabel(r"$t^*=t\Gamma/R^2$")
    axis.set_ylabel(r"major radius $R(t)$")
    axis.set_title("Resolved ring geometry")
    axis.legend(frameon=False)

    axis = axes[1, 1]
    for health, color, label in (
        (histories["coarse_health"], BLUE, "coarse"),
        (histories["fine_health"], GOLD, "fine"),
    ):
        axis.plot(
            health["time"].to_numpy(float) * CIRCULATION,
            health["vorticity_divergence_error"],
            color=color,
            label=label,
        )
    axis.axhline(0.12, color=INK, linestyle="--", label="health limit")
    axis.set_xlabel(r"$t^*=t\Gamma/R^2$")
    axis.set_ylabel("normalized divergence error")
    axis.set_title("Particle-resolution health")
    axis.legend(frameon=False)

    for axis in axes.flat:
        axis.grid(color=GRID, linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Widnall VPM time-step gate", color=INK, fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse-directory", type=Path, required=True)
    parser.add_argument("--fine-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    result, histories = evaluate(args.coarse_directory, args.fine_directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plot(histories, args.figure)
    if result["status"] != "PASS":
        raise SystemExit("WIDNALL TIME-STEP GATE FAIL")


if __name__ == "__main__":
    main()
