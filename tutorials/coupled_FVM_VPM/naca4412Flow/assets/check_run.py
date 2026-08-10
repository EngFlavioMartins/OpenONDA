"""Minimal numerical-integrity gate for the native NACA 4412 workflow."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

CASE_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    metadata_path = CASE_DIR / "solution" / "run_metadata.json"
    force_path = CASE_DIR / "samples" / "ibm_forces_history.csv"
    if not metadata_path.is_file():
        raise SystemExit("FAIL: coupled run metadata was not written")
    metadata = json.loads(metadata_path.read_text())
    expected_end = float(metadata["physics"]["t_end"])

    with force_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit("FAIL: no immersed-airfoil force samples were written")
    values = np.array(
        [[float(row[key]) for key in ("time", "Fx", "Fy", "Cd", "Cl", "slip")] for row in rows]
    )
    if not np.all(np.isfinite(values)):
        raise SystemExit("FAIL: airfoil force history contains non-finite values")
    if values[-1, 0] + 1e-10 < expected_end:
        raise SystemExit(
            f"FAIL: force history ends at t={values[-1, 0]:g}, expected {expected_end:g}"
        )

    diagnostics_path = CASE_DIR / "solution" / "diagnostics.jsonl"
    diagnostics = [json.loads(line) for line in diagnostics_path.read_text().splitlines()]
    if not diagnostics:
        raise SystemExit("FAIL: no FVM diagnostics were written")
    final = diagnostics[-1]
    if final["nonfinite_count"] != 0:
        raise SystemExit("FAIL: non-finite FVM fields were detected")
    if float(final["cfl_max"]) > 5.0:
        raise SystemExit(f"FAIL: final FVM CFL is excessive ({final['cfl_max']:.3g})")
    if float(final["continuity_max"]) > 1e-3:
        raise SystemExit(
            f"FAIL: final continuity residual is excessive ({final['continuity_max']:.3g})"
        )
    if values[-1, 5] > 5.0:
        raise SystemExit(f"FAIL: final IBM no-slip error is excessive ({values[-1, 5]:.3g})")

    coupling_path = CASE_DIR / "solution" / "coupler_diagnostics.jsonl"
    coupling = [json.loads(line) for line in coupling_path.read_text().splitlines()]
    circulation_error = float(coupling[-1]["conservation"]["corrected_mismatch"]["circulation"])
    if circulation_error > 1e-8:
        raise SystemExit(f"FAIL: corrected handoff circulation mismatch is {circulation_error:.3g}")
    print(
        "PASS: native NACA 4412 run completed with bounded CFL, continuity, "
        f"no-slip, and conservative handoff through t={values[-1, 0]:g}."
    )


if __name__ == "__main__":
    main()
