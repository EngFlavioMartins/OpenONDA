#!/usr/bin/env python3
"""Replay an archived stationary interval with time-step energy diagnostics.

The paired screen stores visualization diagnostics much more slowly than the
Ornstein--Uhlenbeck force changes.  This audit starts from an archived state,
records the energy balance every time step, and requires the replayed final
state to agree exactly with the independently archived final checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/openonda_stage4b3_budget_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/openonda_stage4b3_budget_cache")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_4b1_forced_hit_pilot import budget_summary, forcing_power  # noqa: E402
from stage_4b3_stationary_pair import (  # noqa: E402
    GRID,
    MODELS,
    advance_one_step,
    load_checkpoint,
)
from stage_4b_spectral_pilot import COLORS, LABELS, VorticitySolver, diagnostics  # noqa: E402


def checkpoint_args(args: argparse.Namespace) -> argparse.Namespace:
    """Recover the physical configuration used to write the checkpoints."""
    return argparse.Namespace(
        reference_n=args.reference_n,
        les_n=args.les_n,
        viscosity=args.viscosity,
        time_step_size=args.time_step_size,
        end_time=args.original_end_time,
        save_interval=args.original_save_interval,
        checkpoint_interval=args.original_checkpoint_interval,
        forcing_rms=args.forcing_rms,
        correlation_time=args.correlation_time,
        initial_rms=args.initial_rms,
        seed=args.seed,
        restart=None,
    )


def energy_records(
    solver: VorticitySolver,
    states: dict[str, np.ndarray],
    acceleration: np.ndarray,
    gaussian_delta: float,
    time: float,
) -> dict[str, dict[str, float]]:
    records = {}
    for model, state in states.items():
        record = diagnostics(solver, state, model, gaussian_delta)
        record["time"] = time
        record["forcing_power"] = forcing_power(solver, state, acceleration)
        records[model] = record
    return records


def replay(args: argparse.Namespace) -> dict[str, object]:
    config_args = checkpoint_args(args)
    (
        start_step,
        start_time,
        reference_vorticity,
        states,
        forcing,
        _,
        _,
    ) = load_checkpoint(args.start_checkpoint, config_args)
    (
        final_step,
        final_time,
        archived_reference,
        archived_states,
        archived_forcing,
        _,
        _,
    ) = load_checkpoint(args.final_checkpoint, config_args)
    if final_step <= start_step:
        raise ValueError("final checkpoint must follow start checkpoint")

    reference_solver = VorticitySolver(args.reference_n, args.viscosity)
    les_solver = VorticitySolver(args.les_n, args.viscosity)
    gaussian_delta = 2.0 * (2.0 * np.pi / args.les_n) / np.sqrt(6.0)
    histories = {model: [] for model in MODELS}
    step = start_step
    while step <= final_step:
        records = energy_records(
            les_solver,
            states,
            forcing.field,
            gaussian_delta,
            step * args.time_step_size,
        )
        for model, record in records.items():
            histories[model].append(record)
        if step == final_step:
            break
        reference_vorticity, states = advance_one_step(
            config_args,
            reference_solver,
            les_solver,
            gaussian_delta,
            reference_vorticity,
            states,
            forcing,
        )
        step += 1
        if (step - start_step) % 100 == 0:
            print(
                f"budget-recheck progress: {100.0 * (step - start_step) / (final_step - start_step):5.1f}%"
            )

    final_differences = {
        "reference": float(np.max(np.abs(reference_vorticity - archived_reference))),
        "forcing": float(np.max(np.abs(forcing.field - archived_forcing.field))),
        **{
            model: float(np.max(np.abs(states[model] - archived_states[model]))) for model in MODELS
        },
    }
    budgets = {model: budget_summary(records) for model, records in histories.items()}
    tolerance = 2.0e-3
    checks = {
        "exact_archived_trajectory": max(final_differences.values()) == 0.0,
        "all_budget_residuals_below_0.2_percent": max(
            budget["relative_residual"] for budget in budgets.values()
        )
        < tolerance,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "purpose": "dense energy-budget audit of the archived stationary paired screen",
        "start_checkpoint": str(args.start_checkpoint),
        "final_checkpoint": str(args.final_checkpoint),
        "start_time": start_time,
        "final_time": final_time,
        "diagnostic_interval": args.time_step_size,
        "forcing_correlation_time": args.correlation_time,
        "checks": checks,
        "final_maximum_absolute_differences": final_differences,
        "budgets": budgets,
    }


def plot_budgets(result: dict[str, object], output: Path) -> None:
    budgets = result["budgets"]
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), constrained_layout=True)
    for axis, model in zip(axes.flat, MODELS, strict=True):
        budget = budgets[model]
        actual = np.asarray(budget["actual_energy_change"])
        predicted = np.asarray(budget["predicted_energy_change"])
        time = np.linspace(result["start_time"], result["final_time"], len(actual))
        axis.plot(time, actual, color="#20252a", linewidth=1.8, label="measured $E(t)-E(t_0)$")
        axis.plot(
            time,
            predicted,
            color=COLORS[model],
            linestyle="--",
            linewidth=1.5,
            label=r"$\int(P_f-2\nu Z+P_{SGS})\,dt$",
        )
        axis.set_title(f"{LABELS[model]}: residual {100.0 * budget['relative_residual']:.3f}%")
        axis.set_xlabel(r"$t$")
        axis.set_ylabel("energy change")
        axis.grid(color=GRID, linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=7)
    fig.suptitle("Time-step-resolved energy-budget audit", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-checkpoint", type=Path, required=True)
    parser.add_argument("--final-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--reference-n", type=int, default=64)
    parser.add_argument("--les-n", type=int, default=32)
    parser.add_argument("--viscosity", type=float, default=0.02)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--forcing-rms", type=float, default=0.5)
    parser.add_argument("--correlation-time", type=float, default=0.2)
    parser.add_argument("--initial-rms", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--original-end-time", type=float, default=60.0)
    parser.add_argument("--original-save-interval", type=float, default=1.0)
    parser.add_argument("--original-checkpoint-interval", type=float, default=5.0)
    args = parser.parse_args()
    result = replay(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    plot_budgets(result, args.figure)
    if result["status"] != "PASS":
        raise SystemExit("DENSE ENERGY-BUDGET RECHECK FAIL")


if __name__ == "__main__":
    main()
