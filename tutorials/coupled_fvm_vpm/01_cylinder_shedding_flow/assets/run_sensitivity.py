#!/usr/bin/env python3
"""Paired Re=150 accuracy/cost studies at fixed interface convergence controls."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

from openonda.cylinder_campaign import collect_cost, compare_profiles, profile_statistics, run_trial
from openonda.cylinder_case import new_run_directory
from openonda.tutorial_runner import load_case_module

CASE_DIR = Path(__file__).resolve().parents[1]
LAUNCHER = Path(__file__).with_name("run_campaign.py")
FACTORS = {
    "particle_spacing_ratio": (1.25, 1.5),
    "core_radius_ratio": (0.8, 1.2),
    "blend_width_ratio": (4.0, 7.0),
    "release_width_ratio": (1.0, 3.0),
    "transfer_amplification_cap": (1.4, 2.2),
    "exchange_dt": (0.02, 0.08),
    "span": (0.48, 1.92),
    "dz": (0.04, 0.16),
}


def _select_interaction(candidates, coupled_module):
    """Return the first valid two-factor interaction and rejected pairs."""
    rejected = []
    for index, left in enumerate(candidates):
        left_factor = left["factor"]
        for right in candidates[index + 1 :]:
            right_factor = right["factor"]
            if left_factor == right_factor:
                continue
            combined = {
                left_factor: left["overrides"][left_factor],
                right_factor: right["overrides"][right_factor],
            }
            overrides = {
                "hxy": 0.08,
                "cores": 4,
                "particle_spacing_ratio": 1.0,
                **combined,
            }
            try:
                coupled_module.build_case(end_time=100.0, overrides=overrides)
            except ValueError as error:
                rejected.append(
                    {
                        "factors": [left_factor, right_factor],
                        "overrides": combined,
                        "reason": str(error),
                    }
                )
                continue
            return ("interaction", combined, "interaction"), rejected
    return None, rejected


def _screen_steps(max_coupling_steps: int, exchange_dt: float) -> tuple[int, float]:
    """Convert the baseline-clock screen budget to a variant step count."""
    physical_time = 0.04 * max_coupling_steps
    requested_steps = physical_time / exchange_dt
    rounded_steps = round(requested_steps)
    if not math.isclose(requested_steps, rounded_steps, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(
            "--max-coupling-steps does not map to an integer common screen duration "
            f"at exchange_dt={exchange_dt:g}; choose a compatible step count"
        )
    return rounded_steps, physical_time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor", choices=("all", *FACTORS), default="all")
    parser.add_argument("--root", type=Path, default=CASE_DIR / "study_results" / "sensitivity")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--screen", action="store_true")
    parser.add_argument("--max-coupling-steps", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=43200)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--compute-device", choices=("AUTO", "CPU", "CUDA", "VULKAN", "METAL"), default="CPU"
    )
    parser.add_argument("--coupled-cores", type=int, default=4)
    args = parser.parse_args()
    if args.coupled_cores < 1:
        raise ValueError("coupled-cores must be positive")
    if args.max_coupling_steps <= 0:
        raise ValueError("max-coupling-steps must be positive")
    root = (
        args.run_dir.resolve() if args.run_dir else new_run_directory(args.root.resolve(), "paired")
    )
    root.mkdir(parents=True, exist_ok=True)
    post = load_case_module(CASE_DIR / "reference_flow", "postprocess_grid_study")
    variants = [("baseline", {}, "baseline")]
    for factor in FACTORS if args.factor == "all" else (args.factor,):
        variants.extend(
            (f"{factor}-{value:g}", {factor: value}, factor) for value in FACTORS[factor]
        )
    records = []
    rejected_interactions = []
    screen_physical_time = 0.04 * args.max_coupling_steps
    independent_count = len(variants)
    for label, override, factor_name in variants:
        run_dir = root / label
        values = {
            "hxy": 0.08,
            "cores": args.coupled_cores,
            "compute_device": args.compute_device,
            "particle_spacing_ratio": 1.0,
            **override,
        }
        command = [sys.executable, str(LAUNCHER), "--kind", "coupled", "--run-dir", str(run_dir)]
        if args.resume and run_dir.exists():
            command.append("--resume")
        if args.screen:
            exchange_dt = float(override.get("exchange_dt", 0.04))
            screen_steps, screen_physical_time = _screen_steps(args.max_coupling_steps, exchange_dt)
            command.extend(("--end-time", f"{screen_physical_time:g}"))
        for key, value in values.items():
            command.extend(("--override", f"{key}={value}"))
        print(f"Sensitivity {label}: {run_dir}", flush=True)
        record = {
            "label": label,
            "factor": factor_name,
            "overrides": values,
            **run_trial(command, root / "logs" / label, cwd=CASE_DIR, wall_limit=args.timeout),
        }
        record["cost"] = collect_cost(run_dir)
        physical_duration = screen_physical_time if args.screen else 100.0
        record["screened_physical_time"] = physical_duration if args.screen else None
        record["screened_exchange_steps"] = screen_steps if args.screen else None
        record["seconds_per_physical_time"] = record["wall_seconds"] / physical_duration
        if not args.screen and record["returncode"] == 0:
            forces = list((run_dir / "samples").rglob("forces_history.csv"))
            try:
                if len(forces) != 1:
                    raise ValueError("expected one coupled force history")
                record["forces"] = post.force_statistics(forces[0], 40, 100)
                record["profiles"] = {}
                for name in ("span_lower", "span_middle", "span_upper"):
                    paths = list((run_dir / "samples").rglob(name + ".csv"))
                    if len(paths) != 1:
                        raise ValueError(f"expected one {name} profile")
                    record["profiles"][name] = profile_statistics(paths[0])
            except ValueError as error:
                record["qualification_error"] = str(error)
        records.append(record)
        baseline = records[0].get("forces")
        if baseline and record.get("forces"):
            record["relative_change_from_baseline"] = {
                metric: post.relative_change(record["forces"][metric], baseline[metric])
                for metric in ("mean_drag", "rms_lift", "strouhal")
            }
            record["resolved_force_effect"] = {
                metric: abs(record["forces"][metric] - baseline[metric])
                > (
                    (record["forces"]["uncertainty_95"].get(metric) or 0)
                    + (baseline["uncertainty_95"].get(metric) or 0)
                )
                for metric in ("mean_drag", "rms_lift", "strouhal")
            }
            record["profile_errors_from_baseline"] = {
                name: compare_profiles(profile, records[0]["profiles"][name])
                for name, profile in record.get("profiles", {}).items()
                if name in records[0].get("profiles", {})
            }
        record["runtime_ratio_to_baseline"] = (
            record["seconds_per_physical_time"] / records[0]["seconds_per_physical_time"]
        )
        changes = record.get("relative_change_from_baseline", {})
        record["within_sensitivity_targets"] = bool(
            not record["returncode"]
            and (baseline or {}).get("qualified_statistics")
            and record.get("forces", {}).get("qualified_statistics")
            and changes.get("mean_drag", float("inf")) <= 0.02
            and changes.get("rms_lift", float("inf")) <= 0.05
            and changes.get("strouhal", float("inf")) <= 0.02
            and not record["cost"]["unconverged_stationary_intervals"]
            and len(record.get("profile_errors_from_baseline", {})) == 3
            and all(
                row["mean_velocity_l2"] <= 0.03
                for row in record.get("profile_errors_from_baseline", {}).values()
            )
        )
        # After single-factor trials, test one interaction between the two
        # fastest admissible controls. This is a paired confirmation, not an
        # exhaustive Cartesian product or a silently selected speed setting.
        if len(records) == independent_count and args.factor == "all" and not args.screen:
            candidates = sorted(
                (row for row in records[1:] if row["within_sensitivity_targets"]),
                key=lambda row: row["wall_seconds"],
            )
            coupled_module = load_case_module(CASE_DIR)
            interaction, rejected = _select_interaction(candidates, coupled_module)
            rejected_interactions.extend(rejected)
            if interaction is not None:
                variants.append(interaction)
        report = {
            "schema": "openonda-cylinder-sensitivity/2",
            "screen_only": args.screen,
            "scope": "Matched initial inflow with a controlled 3D perturbation; exchange_dt changes the exchange clock, not a standalone particle emission rate. Interface iteration limits and tolerances are fixed. Short screens do not qualify shedding accuracy.",
            "runs": records,
            "rejected_interactions": rejected_interactions,
            "shortlist": [
                row["label"]
                for row in sorted(records, key=lambda row: row["wall_seconds"])
                if row["within_sensitivity_targets"]
            ],
            "recommendation_scope": "Sensitivity shortlist at fixed h=.08 only; final recommendations also require the independent reference/grid/temporal/span qualifications.",
        }
        (root / "sensitivity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(root, flush=True)
    return 1 if any(row["returncode"] or row.get("qualification_error") for row in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
