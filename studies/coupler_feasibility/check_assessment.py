"""Reproduce the bounded numerical checks used in the feasibility assessment.

This is an operator probe and a reanalysis of recorded timings. It does not
advance a body-flow simulation or modify either solver.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from source.coupler.stable_renewal import (
    blend_represented_state,
    gaussian_represented_vortex_strength,
)

ROOT = Path(__file__).resolve().parents[2]


def represented_target_probe():
    h = 0.25
    axes = [h * np.arange(-4, 5)] * 3
    position = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    strength = (
        np.exp(-np.sum(position**2, axis=1) / 0.4)[:, None]
        * np.cross(position, [1.0, 2.0, 3.0])
        * h**3
    )
    displacement = position[:, None] - position[None, :]
    gaussian = np.exp(-np.sum(displacement**2, axis=-1) / h**2) / np.pi**1.5
    target = gaussian @ strength
    production_target = gaussian_represented_vortex_strength(strength, (9, 9, 9), h, core_radius=h)
    np.testing.assert_allclose(production_target, target, rtol=0, atol=2e-17)
    controls = {}
    for name, authority in (
        ("full_authority", np.ones(len(position))),
        ("ramped_authority", np.clip(1.0 - np.max(np.abs(position), axis=1), 0, 1)),
        ("zero_authority", np.zeros(len(position))),
    ):
        result = blend_represented_state(
            strength,
            target,
            authority,
            (9, 9, 9),
            h,
            core_radius=h,
            amplification_cap=1.8,
        )
        represented_after = gaussian @ result.vortex_strength
        controls[name] = {
            "relative_coefficient_change": float(
                np.linalg.norm(result.vortex_strength - strength) / np.linalg.norm(strength)
            ),
            "relative_represented_field_change": float(
                np.linalg.norm(represented_after - target) / np.linalg.norm(target)
            ),
        }
        if name == "zero_authority":
            np.testing.assert_array_equal(result.vortex_strength, strength)
        if name == "full_authority":
            expected = target + 0.8 * (target - gaussian @ target)
            np.testing.assert_allclose(result.vortex_strength, expected, rtol=0, atol=2e-17)
    return {
        "particle_count": len(position),
        "spacing": h,
        "core_radius": h,
        "amplification_cap": 1.8,
        "target": "Independent direct Gaussian vorticity sum times lattice-cell volume h^3",
        "scope": "One production blend; fixed positions; no pruning, body, advection, FVM, or interface iteration",
        "gaussian_operator_max_absolute_difference": float(
            np.max(np.abs(production_target - target))
        ),
        "controls": controls,
    }


def recorded_cost_analysis():
    record = json.loads(
        (
            ROOT / "studies/coupler_accuracy/results/cube-runtime-2026-09-14/measurements.json"
        ).read_text()
    )
    baseline = record["baseline"]["steps"][1]["timing_seconds"]
    optimized = record["optimized"]["steps"][1]["timing_seconds"]
    reference = record["reference"]["interval_005_to_010_seconds"]
    sweeps = record["optimized"]["steps"][1]["interface_iteration"]["sweeps"]
    return {
        "baseline_seconds": baseline,
        "optimized_seconds": optimized,
        "full_reference_seconds": reference,
        "optimized_to_reference_ratio_recomputed": optimized["total"] / reference,
        "baseline_to_optimized_ratio_recomputed": baseline["total"] / optimized["total"],
        "measured_sweeps": sweeps,
        "near_fvm_one_sweep_seconds_estimate": optimized["fvm"] / sweeps,
        "one_sweep_near_fvm_fraction_of_reference_estimate": optimized["fvm"] / sweeps / reference,
        "zero_non_fvm_overhead_cost_ratio_at_current_sweeps": optimized["fvm"] / reference,
        "cell_count_ratio_from_runtime_report": 303264 / 692604,
        "startup_relative_drag_difference": (
            record["optimized"]["forces"][-1]["drag_coefficient"]
            / record["reference"]["forces"][-1]["drag_coefficient"]
            - 1
        ),
        "scope": "Recorded 0.05-to-0.10 startup interval; profiled concurrent jobs; division by sweep count is a cost model, not a measured one-sweep run",
    }


def unfinished_later_history():
    relative = "studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/long-wake-twenty-time-iterated/trial/comparison-history.json"
    path = ROOT / relative
    raw = path.read_bytes()
    rows = json.loads(raw)
    last = rows[-1]
    return {
        "source": relative,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "rows": len(rows),
        "last_record": last,
        "last_relative_drag_difference": last["drag_coefficient_difference"]
        / last["full_drag_coefficient"],
        "scope": "Snapshot of an unfinished frozen-config record. Later rows have not received the independent checkpoint/force verification used for the published t=4 prefix. Not a result for the current tutorial.",
    }


def main():
    files = [
        "source/coupler/stable_renewal.py",
        "source/coupler/interface_iteration.py",
        "source/coupler/solver.py",
        "source/coupler/boundary.py",
        "source/coupler/config/types.py",
        "source/solvers/vpm/physics/diffusion/grid.py",
        "source/solvers/vpm/boundary_elements/panels/solver/panel_solver.py",
        "source/solvers/fvm/assemble/diffusion.py",
        "tutorials/coupled_fvm_vpm/02_cube_flow/setup.py",
        "studies/coupler_accuracy/results/cube-runtime-2026-09-14/measurements.json",
        "studies/coupler_feasibility/check_assessment.py",
    ]
    result = {
        "schema": "openonda-coupler-feasibility/1",
        "production_blend_probe": represented_target_probe(),
        "recorded_cost_analysis": recorded_cost_analysis(),
        "unfinished_later_history": unfinished_later_history(),
        "sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files},
    }
    output = Path(__file__).with_name("assessment_checks.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "sha256"}, indent=2))


if __name__ == "__main__":
    main()
