#!/usr/bin/env python3
"""Compare the corrected short ring trajectory using direct and tree velocity."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.experiments.stage_5b_relaxed_ring_analysis import (  # noqa: E402
    RunSpec,
    analyze_run,
    relative_difference,
)

SOLUTION = ROOT / "tutorials/VPM/vortexRing/solution"
LIMITS = {
    "speed_relative_difference": 2.0e-3,
    "energy_relative_difference": 2.0e-3,
    "ring_radius_relative_difference": 1.0e-3,
    "core_radius_relative_difference": 1.0e-3,
    "tree_axisymmetry_mode_amplitude": 1.0e-4,
    "tree_impulse_relative_drift": 1.0e-3,
    "tree_circulation_relative_drift": 1.0e-3,
}


def elapsed_seconds(directory: Path) -> float:
    log_path = next(directory.glob("*.log"))
    matches = re.findall(
        r"Total simulation time:\s+([0-9.eE+-]+)",
        log_path.read_text(encoding="utf-8"),
    )
    if not matches:
        raise RuntimeError(f"no elapsed step time in {log_path}")
    return float(matches[-1])


def main() -> None:
    direct_dir = SOLUTION / "relaxed_reference_tail002_cs_h010_dt002_tstar02"
    tree_dir = SOLUTION / "relaxed_reference_tail002_cs_h010_tree_dt002_tstar02"
    direct = analyze_run(RunSpec("direct", direct_dir, 0.10, 0.02, "Core Spreading"))
    tree = analyze_run(RunSpec("tree theta=0.1", tree_dir, 0.10, 0.02, "Core Spreading"))
    observed = {
        "speed_relative_difference": relative_difference(
            float(direct["measured_speed"]), float(tree["measured_speed"])
        ),
        "energy_relative_difference": relative_difference(
            float(direct["energy_final"]), float(tree["energy_final"])
        ),
        "ring_radius_relative_difference": relative_difference(
            float(direct["final"]["ring_radius_theta"]),
            float(tree["final"]["ring_radius_theta"]),
        ),
        "core_radius_relative_difference": relative_difference(
            float(direct["final"]["core_radius_theta"]),
            float(tree["final"]["core_radius_theta"]),
        ),
        "tree_axisymmetry_mode_amplitude": float(tree["maximum_mode_amplitude"]),
        "tree_impulse_relative_drift": float(tree["impulse_relative_drift"]),
        "tree_circulation_relative_drift": float(tree["circulation_relative_drift"]),
    }
    checks = {key: observed[key] <= limit for key, limit in LIMITS.items()}
    direct_elapsed = elapsed_seconds(direct_dir)
    tree_elapsed = elapsed_seconds(tree_dir)
    payload = {
        "stage": "5B corrected ring direct-versus-tree short audit",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "claim_scope": (
            "Short axisymmetric interval only. Earlier perturbed-ring work showed "
            "that small tree errors can accumulate in modal phase."
        ),
        "limits": LIMITS,
        "observed": observed,
        "checks": checks,
        "timing": {
            "direct_elapsed_seconds": direct_elapsed,
            "tree_elapsed_seconds": tree_elapsed,
            "direct_over_tree_speedup": direct_elapsed / tree_elapsed,
        },
        "decision": (
            "Use direct h=0.12 for the long relaxation: the tree passes this short "
            "screen but provides too little speedup to justify its accumulated-phase risk."
        ),
    }
    output = ROOT / "scripts/experiments/stage_5b_relaxed_ring_tree_audit_results.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
