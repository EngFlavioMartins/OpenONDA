#!/usr/bin/env python3
"""Final repository-level verification for the complete FVM migration."""

from __future__ import annotations

import ast
from io import BytesIO
from pathlib import Path
import re
import subprocess
import sys
import tokenize

SOURCE_ROOT = Path("source/solvers/FVM")
PUBLIC_FVM = Path("openonda/fvm.py")
TUTORIAL_ROOTS = (
    Path("tutorials/FVM"),
    Path("tutorials/coupled_fvm_vpm"),
)
TEST_ROOTS = (
    Path("tests/fvm"),
    Path("tests/coupler"),
    Path("tests/experiments"),
)

SOURCE_LEGACY_FILES = {
    SOURCE_ROOT / "config" / "types.py",
    SOURCE_ROOT / "io" / "checkpoint.py",
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

TIME_EXEMPT = {
    "dE_dt",
    "dU_dt_peak",
    "du_dt",
    "dalpha_dt",
    "dstr_dt",
    "four_nu_dt",
}

SOLVER_ATTR = re.compile(
    r"\b(?:self|solver|fvm_solver|system|fvm|dummy)\."
    r"(?:U|U_old|U_old_old|p|phi|phi_old|phi_old_old|nut|config)\b"
)


def fvm_test_relevant(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    return any(
        marker in source
        for marker in (
            "source.solvers.FVM",
            "openonda.fvm",
            "FVMSetup",
            "FVMSolver",
            "FieldState",
            "fvm_solver",
            "DummyFVM",
        )
    )


def check_path(path: Path, *, source_file: bool) -> list[str]:
    failures: list[str] = []
    text = path.read_text(encoding="utf-8")

    try:
        ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        return [f"{path}: syntax error: {exc}"]

    for token in tokenize.tokenize(BytesIO(text.encode()).readline):
        if token.type != tokenize.NAME:
            continue
        name = token.string
        if name in FORBIDDEN_IDENTIFIERS:
            failures.append(f"{path}:{token.start[0]}: obsolete identifier {name!r}")
        if name.startswith(("element_", "elem_")):
            failures.append(f"{path}:{token.start[0]}: obsolete mesh prefix {name!r}")
        if source_file and ("center" in name.lower() or "neighbor" in name.lower()):
            failures.append(f"{path}:{token.start[0]}: non-canonical spelling {name!r}")
        if name not in TIME_EXEMPT and (
            name == "dt" or name.startswith("dt_") or name.endswith("_dt")
        ):
            failures.append(f"{path}:{token.start[0]}: non-canonical time-step name {name!r}")

    for lineno, line in enumerate(text.splitlines(), 1):
        if SOLVER_ATTR.search(line):
            failures.append(f"{path}:{lineno}: obsolete FVMSolver attribute")

    return failures


def main() -> int:
    if not SOURCE_ROOT.is_dir():
        print("Run from the OpenONDA repository root.", file=sys.stderr)
        return 2

    failures: list[str] = []

    for path in SOURCE_ROOT.rglob("*.py"):
        if "__pycache__" not in path.parts:
            failures.extend(check_path(path, source_file=True))
    if PUBLIC_FVM.exists():
        failures.extend(check_path(PUBLIC_FVM, source_file=True))

    for root in TUTORIAL_ROOTS:
        if root.exists():
            for path in root.rglob("*.py"):
                failures.extend(check_path(path, source_file=False))

    for root in TEST_ROOTS:
        if root.exists():
            for path in root.rglob("*.py"):
                if fvm_test_relevant(path):
                    failures.extend(check_path(path, source_file=False))

    for old_path in (
        Path("tutorials/FVM/airfoilFlow"),
        Path("tutorials/FVM/boundaryLayer"),
        Path("tutorials/FVM/cubeFlow"),
        Path("tutorials/FVM/cylinderIBM"),
        Path("tutorials/FVM/stepProfile"),
        Path("tutorials/FVM/taylorGreen"),
        Path("tutorials/coupled_FVM_VPM"),
    ):
        if old_path.exists():
            failures.append(f"old tutorial path remains: {old_path}")

    print("Complete FVM migration audit")
    print("=" * 72)
    if failures:
        print("\nFAILURES:")
        for item in failures:
            print(f"  {item}")
        return 1

    print("\nRequired checks: PASS")

    print("\nCompiling FVM source/tutorials/tests...")
    compile_targets = [
        "source/solvers/FVM",
        "openonda",
        "tutorials/FVM",
        "tests/fvm",
    ]
    if Path("tutorials/coupled_fvm_vpm").exists():
        compile_targets.append("tutorials/coupled_fvm_vpm")

    result = subprocess.run(
        [sys.executable, "-m", "compileall", *compile_targets],
        check=False,
    )
    if result.returncode:
        return result.returncode

    print("\nFVM nomenclature migration is complete. Run the FVM test suite next:")
    print("  pytest -q tests/fvm")
    print(
        "Coupler tests should be run after the subsequent Coupler API batch, "
        "because Batch 06 changes only their FVM-facing side."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
