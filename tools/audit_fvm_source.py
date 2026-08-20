#!/usr/bin/env python3
"""Strict final source audit for the FVM nomenclature migration."""

from __future__ import annotations

import ast
from io import BytesIO
from pathlib import Path
import re
import sys
import tokenize

FVM_ROOT = Path("source/solvers/FVM")
PUBLIC_FVM = Path("openonda/fvm.py")
LEGACY_INPUT_FILES = {
    FVM_ROOT / "config" / "types.py",
    FVM_ROOT / "io" / "checkpoint.py",
}

FORBIDDEN_IDENTIFIERS = {
    "MeshConfig",
    "SchemesConfig",
    "ExecutionConfig",
    "OutputSetup",
    "LogConfig",
    "DynamicMeshConfig",
    "type_velocity",
    "value_velocity",
    "type_p",
    "value_p",
    "type_phi",
    "value_phi",
    "type_nut",
    "value_nut",
    "write_interval",
    "write_interval_time",
    "momentum_tol",
    "momentum_rel_tol",
    "momentum_final_rel_tol",
    "momentum_maxiter",
    "pressure_tol",
    "pressure_rel_tol",
    "pressure_final_rel_tol",
    "pressure_maxiter",
    "amg_tol",
    "amg_maxiter",
    "alpha_u",
    "alpha_p",
    "initial_p",
    "U",
    "U_old",
    "U_old_old",
    "nut",
    "compute_nut",
    "n_elements",
    "startFace",
    "nFaces",
    "neighbourPatch",
}

SOLVER_ATTRIBUTE_PATTERN = re.compile(
    r"\b(?:self|solver|fvm_solver|system|fvm)\."
    r"(?:U|U_old|U_old_old|p|phi|phi_old|phi_old_old|nut|config)\b"
)


def is_time_step_name(name: str) -> bool:
    if name in {
        "dE_dt",
        "dU_dt_peak",
        "du_dt",
        "dalpha_dt",
        "dstr_dt",
        "four_nu_dt",
    }:
        return False
    return name == "dt" or name.startswith("dt_") or name.endswith("_dt")


def main() -> int:
    if not FVM_ROOT.is_dir():
        print("Run from the OpenONDA repository root.", file=sys.stderr)
        return 2

    failures: list[str] = []
    paths = list(FVM_ROOT.rglob("*.py"))
    if PUBLIC_FVM.exists():
        paths.append(PUBLIC_FVM)

    for path in paths:
        if "__pycache__" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        try:
            ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            failures.append(f"{path}: syntax error: {exc}")
            continue

        for token in tokenize.tokenize(BytesIO(source.encode()).readline):
            if token.type != tokenize.NAME:
                continue
            name = token.string
            if name in FORBIDDEN_IDENTIFIERS:
                failures.append(f"{path}:{token.start[0]}: obsolete identifier {name!r}")
            if name.startswith(("element_", "elem_")):
                failures.append(f"{path}:{token.start[0]}: obsolete mesh prefix {name!r}")
            if "center" in name.lower() or "neighbor" in name.lower():
                failures.append(f"{path}:{token.start[0]}: non-canonical spelling {name!r}")
            if is_time_step_name(name):
                failures.append(f"{path}:{token.start[0]}: non-canonical time-step name {name!r}")

        for lineno, line in enumerate(source.splitlines(), 1):
            if SOLVER_ATTRIBUTE_PATTERN.search(line):
                failures.append(f"{path}:{lineno}: obsolete FVMSolver attribute")
            if path not in LEGACY_INPUT_FILES:
                for old in (
                    "n_elements",
                    "startFace",
                    "nFaces",
                    "neighbourPatch",
                ):
                    if f'"{old}"' in line or f"'{old}'" in line:
                        failures.append(f"{path}:{lineno}: obsolete serialized key {old!r}")

    print("FVM source nomenclature audit")
    print("=" * 72)
    if failures:
        print("\nFAILURES:")
        for item in failures:
            print(f"  {item}")
        return 1

    print("\nRequired checks: PASS")
    print(
        "\nFVM source is nomenclature-complete. Legacy names remain only "
        "inside explicit setup/checkpoint input-compatibility tables."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
