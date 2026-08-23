#!/usr/bin/env python3
"""Fail before plotting an incomplete or empty vortex_interactions run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from _common import discover_cases, read_integrals, read_ring_diagnostics


EXPECTED_CASES = (
    "leapfrog_dns",
    "leapfrog_les",
    "leapfrog_les_stabilized",
    "collide_dns",
    "collide_les",
    "collide_les_stabilized",
)


def validate(solution_dir: Path, *, allow_partial: bool) -> list[str]:
    failures: list[str] = []
    discovered = {case.name: case for case in discover_cases(solution_dir)}

    if not discovered:
        return [f"no recognized cases found under {solution_dir}"]
    if not allow_partial:
        missing = [name for name in EXPECTED_CASES if name not in discovered]
        if missing:
            failures.append("missing cases: " + ", ".join(missing))
        discovered = {name: discovered[name] for name in EXPECTED_CASES if name in discovered}

    for name, case_dir in sorted(discovered.items()):
        manifest = None
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
                completed_steps = manifest.get("completed_steps")
                requested_steps = manifest.get("requested_steps")
                stabilized = name.endswith("_les_stabilized")
                valid_status = status == "completed" or (
                    not stabilized and status == "resolution_lost"
                )
                if not valid_status:
                    failures.append(f"{name}: invalid run status {status!r}")
                elif status == "completed" and completed_steps != requested_steps:
                    failures.append(
                        f"{name}: completed {completed_steps!r} of "
                        f"{requested_steps!r} requested steps"
                    )
                elif status == "resolution_lost" and not manifest.get("termination_reason"):
                    failures.append(f"{name}: resolution loss has no recorded reason")

        diagnostics = read_integrals(case_dir)
        if diagnostics is None or len(diagnostics) < 2:
            failures.append(f"{name}: fewer than two flow-integral samples")

        ring_diagnostics = read_ring_diagnostics(case_dir)
        if ring_diagnostics is None:
            failures.append(f"{name}: missing grouped ring diagnostics")
        elif set(ring_diagnostics["group_id"].astype(int)) != {0, 1}:
            failures.append(f"{name}: ring sampler does not contain both particle groups")

        numbered_steps = {
            int(match.group(1))
            for path in case_dir.glob("vpm_*_*.h5")
            if (match := re.search(r"_(\d{6})\.h5$", path.name))
        }
        if manifest is not None and isinstance(manifest.get("completed_steps"), int):
            completed_steps = int(manifest["completed_steps"])
            snapshot_frequency = int(manifest.get("snapshot_frequency", 0))
            if snapshot_frequency <= 0:
                failures.append(f"{name}: invalid snapshot frequency")
            else:
                expected = set(range(0, completed_steps + 1, snapshot_frequency))
                missing_snapshots = sorted(expected - numbered_steps)
                if missing_snapshots:
                    failures.append(
                        f"{name}: missing scheduled state snapshots {missing_snapshots[:5]}"
                    )
                missing_descriptors = [
                    step
                    for step in expected
                    if not (case_dir / f"vpm_{name}_{step:06d}.xdmf").is_file()
                ]
                if missing_descriptors:
                    failures.append(
                        f"{name}: state snapshots have no XDMF descriptors "
                        f"{missing_descriptors[:5]}"
                    )
        if not all(
            (case_dir / f"vpm_{name}_final{suffix}").is_file() for suffix in (".h5", ".xdmf")
        ):
            failures.append(f"{name}: final VPM state or XDMF descriptor is missing")
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
    scope = "partial result set" if args.allow_partial else "complete tutorial run"
    print(f"Plot inputs validated: {count} cases ({scope}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
