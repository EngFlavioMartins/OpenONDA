"""Minimal numerical-integrity gate for the native NACA 4412 workflow."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

CASE_DIR = Path(__file__).resolve().parents[1]


def _json_lines(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"FAIL: expected diagnostics file was not written: {path}")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        raise SystemExit(f"FAIL: diagnostics file is empty: {path}")
    return records


def _check_physics(values: np.ndarray) -> str:
    alpha = math.radians(10.0)
    drag = values[:, 3] * math.cos(alpha) + values[:, 4] * math.sin(alpha)
    lift = -values[:, 3] * math.sin(alpha) + values[:, 4] * math.cos(alpha)
    settled = values[:, 0] >= 0.5 * values[-1, 0]
    if np.count_nonzero(settled) < 6:
        raise SystemExit("FAIL: production NACA run has too few settled force samples")
    mean_drag = float(np.mean(drag[settled]))
    mean_lift = float(np.mean(lift[settled]))
    if not 0.0 < mean_drag < 5.0:
        raise SystemExit(f"FAIL: settled NACA wind-axis drag is not physical ({mean_drag:.3g})")
    if not 0.05 < mean_lift < 5.0:
        raise SystemExit(f"FAIL: settled NACA lift is not physical ({mean_lift:.3g})")
    tail = np.flatnonzero(settled)
    midpoint = tail[0] + len(tail) // 2
    lift_drift = abs(float(np.mean(lift[midpoint:]) - np.mean(lift[tail[0] : midpoint])))
    lift_scale = max(abs(mean_lift), 0.1)
    if lift_drift / lift_scale > 0.50:
        raise SystemExit(
            f"FAIL: settled NACA mean lift is still drifting ({lift_drift / lift_scale:.1%})"
        )
    return f" wind-axis mean drag_coefficient={mean_drag:.3f}, lift_coefficient={mean_lift:.3f};"


def main() -> None:
    metadata_path = CASE_DIR / "solution" / "run_metadata.json"
    force_path = CASE_DIR / "samples" / "ibm_forces_history.csv"
    if not metadata_path.is_file():
        raise SystemExit("FAIL: coupled run metadata was not written")
    metadata = json.loads(metadata_path.read_text())
    expected_end = float(metadata["physics"]["end_time"])

    with force_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit("FAIL: no immersed-airfoil force samples were written")
    values = np.array(
        [
            [
                float(row[key])
                for key in (
                    "time",
                    "force_x",
                    "force_y",
                    "drag_coefficient",
                    "lift_coefficient",
                    "slip_error",
                )
            ]
            for row in rows
        ]
    )
    if not np.all(np.isfinite(values)):
        raise SystemExit("FAIL: airfoil force history contains non-finite values")
    if values[-1, 0] + 1e-10 < expected_end:
        raise SystemExit(
            f"FAIL: force history ends at t={values[-1, 0]:g}, expected {expected_end:g}"
        )

    diagnostics = _json_lines(CASE_DIR / "solution" / "diagnostics.jsonl")
    if any(int(record["nonfinite_count"]) != 0 for record in diagnostics):
        raise SystemExit("FAIL: non-finite FVM fields were detected")
    failed = [
        (record["step"], solve.get("equation", "unknown"))
        for record in diagnostics
        for solve in record.get("linear_solves", ())
        if not solve.get("converged", False)
    ]
    if failed:
        raise SystemExit(f"FAIL: unconverged FVM linear solves were recorded: {failed[:5]}")
    max_courant_number = max(float(record["max_courant_number"]) for record in diagnostics)
    max_continuity = max(float(record["max_continuity_error"]) for record in diagnostics)
    if max_courant_number > 5.0:
        raise SystemExit(
            f"FAIL: peak FVM max_courant_number is excessive ({max_courant_number:.3g})"
        )
    if max_continuity > 1e-3:
        raise SystemExit(f"FAIL: peak continuity residual is excessive ({max_continuity:.3g})")
    if values[-1, 5] > 5.0:
        raise SystemExit(f"FAIL: final IBM slip_error is excessive ({values[-1, 5]:.3g})")

    coupling = _json_lines(CASE_DIR / "solution" / "coupler_diagnostics.jsonl")
    replacement_values = np.asarray(
        [
            [
                float(record["transfer"][f"state_change_vortex_strength_net_{axis}"])
                for axis in "xyz"
            ]
            for record in coupling
        ]
    )
    if not np.all(np.isfinite(replacement_values)):
        raise SystemExit("FAIL: state-replacement circulation budget is non-finite")
    vpm_boundary_condition_error = max(
        abs(float(record["vpm_boundary_condition_flux"]["corrected_mismatch"]))
        for record in coupling
    )
    if vpm_boundary_condition_error > 1e-8:
        raise SystemExit(
            "FAIL: corrected VPM boundary-condition flux mismatch is "
            f"{vpm_boundary_condition_error:.3g}"
        )
    physics_summary = _check_physics(values) if expected_end >= 8.0 else ""
    print(
        "PASS: native NACA 4412 run completed with converged FVM solves,"
        f"{physics_summary} peak max_courant_number={max_courant_number:.3g}, "
        f"peak continuity={max_continuity:.3g}, "
        f"bounded no-slip error, and conservative handoff through t={values[-1, 0]:g}."
    )


if __name__ == "__main__":
    main()
