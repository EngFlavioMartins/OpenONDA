#!/usr/bin/env python3
"""Aggregate the narrowly scoped full-state, timing, and formulation extension."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from .. import setup

from .run_full_checkpoint import load_checkpoint, write_csv


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    setup.mkdirs()
    summaries = []
    for path in sorted(setup.RESULTS.glob("full_replay_*_summary.json")):
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    exact_timing = read_csv(setup.RESULTS / "scale_timing_exact_pair_rk3_isolated.csv")
    gate = exact_timing[-1]
    summaries.append(
        {
            "checkpoint": "rotor",
            "configuration": "exact_pair_rk3_isolated",
            "status": "skipped_by_measured_feasibility_gate",
            "particles": 70_200,
            "steps": 0,
            "predicted_step_wall_s": gate["predicted_70200_step_wall_s_quadratic"],
            "gate_s": 120.0,
            "basis": "35000-particle complete-step timing with observed quadratic extrapolation",
        }
    )
    write_csv(setup.RESULTS / "full_checkpoint_replay.csv", summaries)

    comparison_rows = []
    pairs = (
        ("leapfrog", "exact_pair_rk3_isolated", "tree_gradient_rk3_isolated", True),
        ("leapfrog", "exact_pair_rk3_isolated", "production_numerics_unforced", False),
        ("rotor", "tree_gradient_rk3_isolated", "production_numerics_unforced", False),
    )
    for checkpoint, reference, candidate, same_equation in pairs:
        ref = np.load(setup.RESULTS / f"full_replay_{checkpoint}_{reference}_state.npz")
        trial = np.load(setup.RESULTS / f"full_replay_{checkpoint}_{candidate}_state.npz")
        dx = np.linalg.norm(trial["position"] - ref["position"], axis=1)
        dg = np.linalg.norm(trial["vortex_strength"] - ref["vortex_strength"], axis=1)
        refmag = np.linalg.norm(ref["vortex_strength"], axis=1)
        relative = dg / np.maximum(refmag, 1e-30)
        source = load_checkpoint(checkpoint)
        worst = np.argsort(relative)[-100:][::-1]
        comparison_rows.append(
            {
                "checkpoint": checkpoint,
                "reference": reference,
                "candidate": candidate,
                "same_equation_and_physics": same_equation,
                "position_relative_l2": np.linalg.norm(trial["position"] - ref["position"])
                / max(np.linalg.norm(ref["position"]), 1e-30),
                "strength_relative_l2": np.linalg.norm(
                    trial["vortex_strength"] - ref["vortex_strength"]
                )
                / max(np.linalg.norm(ref["vortex_strength"]), 1e-30),
                "per_particle_strength_error_median": np.median(relative),
                "per_particle_strength_error_p95": np.percentile(relative, 95),
                "per_particle_strength_error_max": np.max(relative),
                "position_difference_max": np.max(dx),
                "worst_checkpoint_index": int(source["checkpoint_index"][worst[0]]),
                "worst_x0": source["position"][worst[0], 0],
                "worst_y0": source["position"][worst[0], 1],
                "worst_z0": source["position"][worst[0], 2],
                "interpretation": (
                    "direct evaluator comparison"
                    if same_equation
                    else "not an accuracy comparison: RK order, diffusion, LES and freestream differ"
                ),
            }
        )
    write_csv(setup.RESULTS / "full_replay_comparisons.csv", comparison_rows)

    stage_rows = []
    for path in sorted(setup.RESULTS.glob("full_replay_*_stages.csv")):
        rows = read_csv(path)
        if not rows:
            continue

        def values(key: str, records: list[dict[str, str]] = rows) -> np.ndarray:
            return np.array([float(row[key]) for row in records if row.get(key, "")], dtype=float)

        item = {
            "checkpoint": rows[0]["checkpoint"],
            "configuration": rows[0]["configuration"],
            "stage_records": len(rows),
            "chi_s_max_over_stages": values("exact_chi_s_max").max(),
            "chi_r_max_over_stages": values("exact_chi_r_max").max(),
            "chi_gamma_max_over_stages": values("chi_gamma_max").max(),
            "net_strength_rate_norm_max": values("net_strength_rate_norm").max(),
            "source_blob_gradient_reference": True,
        }
        gradient_error = values("gradient_relative_l2_on_targets")
        rate_error = values("rate_relative_l2_on_targets")
        if len(gradient_error):
            item["tree_gradient_relative_l2_min"] = gradient_error.min()
            item["tree_gradient_relative_l2_max"] = gradient_error.max()
        if len(rate_error):
            item["rate_relative_l2_min"] = rate_error.min()
            item["rate_relative_l2_max"] = rate_error.max()
        if "exact_pair" in rows[0]["configuration"] or (
            rows[0]["checkpoint"] == "leapfrog" and "production" in rows[0]["configuration"]
        ):
            item["rate_error_interpretation"] = (
                "operator difference: pairwise rate uses mean target/source core; "
                "source-blob J uses source core"
            )
        stage_rows.append(item)
    write_csv(setup.RESULTS / "full_replay_stage_summary.csv", stage_rows)
    print(
        f"wrote {len(summaries)} replay, {len(comparison_rows)} comparison, and {len(stage_rows)} stage summaries"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
