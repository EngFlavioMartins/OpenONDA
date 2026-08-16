"""Numerical-integrity gate for the hybrid cylinder benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CASE_DIR = Path(__file__).resolve().parents[1]


def _json_lines(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"FAIL: missing {path.relative_to(CASE_DIR)}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"FAIL: empty {path.relative_to(CASE_DIR)}")
    return rows


def main() -> None:
    metadata_path = CASE_DIR / "solution" / "run_metadata.json"
    force_path = CASE_DIR / "samples" / "ibm_forces_history.csv"
    if not metadata_path.exists() or not force_path.exists():
        raise SystemExit("FAIL: run metadata or IBM force history was not written")
    metadata = json.loads(metadata_path.read_text())
    expected_end = float(metadata["physics"]["t_end"])
    forces = np.atleast_1d(
        np.genfromtxt(force_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    )
    if (
        not forces.size
        or not np.all(np.isfinite(forces["Cd"]))
        or not np.all(np.isfinite(forces["Cl"]))
    ):
        raise SystemExit("FAIL: force history is empty or non-finite")
    if float(forces["time"][-1]) + 1.0e-10 < expected_end:
        raise SystemExit(
            f"FAIL: force history stopped at {forces['time'][-1]}, expected {expected_end}"
        )

    diagnostics = _json_lines(CASE_DIR / "solution" / "diagnostics.jsonl")
    if any(int(row["nonfinite_count"]) for row in diagnostics):
        raise SystemExit("FAIL: non-finite FVM fields detected")
    failed = [
        (row["step"], solve.get("equation", "unknown"))
        for row in diagnostics
        for solve in row.get("linear_solves", ())
        if not solve.get("converged", False)
    ]
    if failed:
        raise SystemExit(f"FAIL: unconverged FVM solves: {failed[:5]}")
    max_continuity = max(float(row["continuity_max"]) for row in diagnostics)
    if max_continuity > 1.0e-3:
        raise SystemExit(f"FAIL: excessive continuity residual {max_continuity:.3g}")

    coupling = _json_lines(CASE_DIR / "solution" / "coupler_diagnostics.jsonl")
    circulation_error = max(
        float(row["conservation"]["corrected_mismatch"]["circulation"]) for row in coupling
    )
    donor_error = max(float(row["donor_flux"]["corrected_mismatch"]) for row in coupling)
    if circulation_error > 1.0e-8 or donor_error > 1.0e-8:
        raise SystemExit(
            f"FAIL: transfer conservation failed (circulation={circulation_error:.3g}, donor={donor_error:.3g})"
        )
    print(
        f"PASS: hybrid cylinder completed through t={forces['time'][-1]:g}; "
        f"continuity<={max_continuity:.3g}, conservative handoff, finite forces."
    )


if __name__ == "__main__":
    main()
