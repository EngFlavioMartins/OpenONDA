#!/usr/bin/env python3
"""Validate that the stabilization comparison produced useful run horizons."""

from __future__ import annotations

import json
from pathlib import Path


CASE_DIR = Path(__file__).resolve().parent.parent
SOLUTION_DIR = CASE_DIR / "solution"
PLANNED_STEPS = 1200
MINIMUM_EXPERIMENTAL_STEPS = 100
FIRST_REMESH_STEP = 450
FULLY_STABILIZED_CASE = "leapfrog_les_splitting_remeshing"
EXPECTATIONS = {
    "leapfrog_les": MINIMUM_EXPERIMENTAL_STEPS,
    "leapfrog_les_splitting": MINIMUM_EXPERIMENTAL_STEPS,
    "leapfrog_les_remeshing": FIRST_REMESH_STEP + 1,
    FULLY_STABILIZED_CASE: PLANNED_STEPS,
}


def load_manifest(case_name: str) -> dict:
    """Load one solver-owned terminal manifest."""
    path = SOLUTION_DIR / case_name / "run_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing terminal manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    failures: list[str] = []
    completed_steps: dict[str, int] = {}

    print("\nStabilization-suite terminal states")
    for case_name, minimum_steps in EXPECTATIONS.items():
        try:
            manifest = load_manifest(case_name)
        except (OSError, json.JSONDecodeError) as error:
            failures.append(f"{case_name}: {error}")
            continue

        status = str(manifest.get("status", "missing"))
        step = int(manifest.get("step", -1))
        planned = int(manifest.get("planned_steps", -1))
        completed_steps[case_name] = step
        print(f"  {case_name}: status={status}, accepted_steps={step}/{planned}")

        if planned != PLANNED_STEPS:
            failures.append(f"{case_name}: planned {planned} steps, expected {PLANNED_STEPS}")
        if step < minimum_steps:
            failures.append(
                f"{case_name}: stopped at step {step}, before its useful minimum {minimum_steps}"
            )
        if case_name == FULLY_STABILIZED_CASE and status != "completed":
            failures.append(f"{case_name}: the fully stabilized case did not complete")

    fully_stabilized_steps = completed_steps.get(FULLY_STABILIZED_CASE, -1)
    longest_other_run = max(
        (step for case_name, step in completed_steps.items() if case_name != FULLY_STABILIZED_CASE),
        default=-1,
    )
    if fully_stabilized_steps < longest_other_run:
        failures.append(
            f"{FULLY_STABILIZED_CASE}: ended at step {fully_stabilized_steps}, "
            f"before another variant at step {longest_other_run}"
        )

    if failures:
        print("\nStabilization-suite validation FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("Stabilization-suite validation PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
