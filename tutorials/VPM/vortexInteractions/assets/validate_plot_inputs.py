#!/usr/bin/env python3
"""Fail before plotting an incomplete or empty vortexInteractions matrix."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

from _common import discover_cases, read_integrals


FAMILIES = ("leapfrog", "collide")
METHODS = ("baseline", "les", "les_stabilized")
EXPECTED_CASES = tuple(f"{family}_{method}" for family, method in product(FAMILIES, METHODS))
TERMINAL_STATUSES = {"completed", "terminated_nonphysical", "rejected"}


def validate(solution_dir: Path, *, allow_partial: bool) -> list[str]:
    failures: list[str] = []
    discovered = {case.name: case for case in discover_cases(solution_dir)}

    if not discovered:
        return [f"no recognized cases found under {solution_dir}"]
    if not allow_partial:
        missing = [name for name in EXPECTED_CASES if name not in discovered]
        if missing:
            failures.append("missing cases: " + ", ".join(missing))

    for name, case_dir in sorted(discovered.items()):
        manifest_path = case_dir / "run_manifest.json"
        if not manifest_path.exists():
            failures.append(f"{name}: missing run_manifest.json")
        else:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                failures.append(f"{name}: unreadable manifest ({error})")
            else:
                status = manifest.get("status")
                if status not in TERMINAL_STATUSES:
                    failures.append(f"{name}: non-terminal run status {status!r}")
                if not allow_partial and name.endswith("_les_stabilized"):
                    completed_steps = manifest.get("completed_steps")
                    requested_steps = manifest.get("requested_steps")
                    if status != "completed":
                        failures.append(
                            f"{name}: stabilized candidate ended with status {status!r}"
                        )
                    elif completed_steps != requested_steps:
                        failures.append(
                            f"{name}: completed {completed_steps!r} of "
                            f"{requested_steps!r} requested steps"
                        )

        diagnostics = read_integrals(case_dir)
        if diagnostics is None or len(diagnostics) < 2:
            failures.append(f"{name}: fewer than two flow-integral snapshots")
        if not any(case_dir.glob("vpm_*_*.h5")):
            failures.append(f"{name}: no VPM state backups")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution-dir", default="solution")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    solution_dir = Path(args.solution_dir)
    failures = validate(solution_dir, allow_partial=args.allow_partial)
    if failures:
        print("Plot input validation failed:")
        for failure in failures:
            print(f"  - {failure}")
        print("No figures were generated or overwritten.")
        return 1

    count = len(discover_cases(solution_dir))
    matrix = "partial matrix" if args.allow_partial else "complete six-case matrix"
    print(f"Plot inputs validated: {count} cases ({matrix}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
