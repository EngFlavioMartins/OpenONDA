#!/usr/bin/env python3
"""Audit FVM tutorial nomenclature after Batch 05."""

from __future__ import annotations

import ast
from pathlib import Path
import re

ROOTS = (
    Path("tutorials/FVM"),
    Path("tutorials/coupled_fvm_vpm"),
)

OLD_CASES = {
    "airfoilFlow",
    "boundaryLayer",
    "cubeFlow",
    "cylinderIBM",
    "stepProfile",
    "taylorGreen",
    "coupled_FVM_VPM",
}

OLD_API = re.compile(
    r"\b(?:MeshConfig|SchemesConfig|ExecutionConfig|OutputSetup|"
    r"LogConfig|DynamicMeshConfig|type_velocity|value_velocity|"
    r"type_p|value_p|type_nut|value_nut|write_interval|"
    r"write_interval_time|initial_p|momentum_tol|pressure_tol|"
    r"alpha_u|alpha_p|n_elements)\b"
)


def main() -> int:
    failures: list[str] = []

    for old in OLD_CASES:
        if (Path("tutorials/FVM") / old).exists():
            failures.append(f"old tutorial directory remains: {old}")
    if Path("tutorials/coupled_FVM_VPM").exists():
        failures.append("old coupled tutorial directory remains: tutorials/coupled_FVM_VPM")

    for root in ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            try:
                ast.parse(source, filename=str(path))
            except SyntaxError as exc:
                failures.append(f"{path}: syntax error: {exc}")
                continue

            for lineno, line in enumerate(source.splitlines(), 1):
                if OLD_API.search(line):
                    failures.append(f"{path}:{lineno}: obsolete FVM tutorial vocabulary")
                if "from openonda.fvm import" in line:
                    failures.append(f"{path}:{lineno}: use `import openonda.fvm as fvm`")
                if "from source.solvers.FVM import" in line:
                    failures.append(f"{path}:{lineno}: internal FVM top-level import")

    print("FVM tutorial audit")
    print("=" * 72)
    if failures:
        print("\nFAILURES:")
        for item in failures:
            print(f"  {item}")
        return 1

    print("\nRequired checks: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
