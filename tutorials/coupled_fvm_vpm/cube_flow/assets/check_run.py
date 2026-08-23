"""Numerical-integrity gate for the native cube FVM–VPM workflow."""

from __future__ import annotations

import csv
import json
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


def _check_solver_history(diagnostics: list[dict]) -> tuple[float, float]:
    max_courant_number = max(float(record["max_courant_number"]) for record in diagnostics)
    max_continuity = max(float(record["max_continuity_error"]) for record in diagnostics)
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
    if max_courant_number > 5.0:
        raise SystemExit(
            f"FAIL: peak FVM max_courant_number is excessive ({max_courant_number:.3g})"
        )
    if max_continuity > 1e-3:
        raise SystemExit(f"FAIL: peak continuity residual is excessive ({max_continuity:.3g})")
    return max_courant_number, max_continuity


def _check_coupling_history(coupling: list[dict]) -> None:
    vpm_boundary_condition_error = max(
        abs(float(record["vpm_boundary_condition_flux"]["corrected_mismatch"]))
        for record in coupling
    )
    if vpm_boundary_condition_error > 1e-8:
        raise SystemExit(
            "FAIL: corrected VPM boundary-condition flux mismatch is "
            f"{vpm_boundary_condition_error:.3g}"
        )
    flux_excess = max(
        float(record["vpm_boundary_condition_flux"]["raw_relative"])
        - float(record["vpm_boundary_condition_flux"]["acceptance_limit"])
        for record in coupling
    )
    if flux_excess > 0.0:
        raise SystemExit("FAIL: a physically significant VPM boundary flux was projected")
    correction_divergence = max(
        float(record["transfer"]["divergence_correction_linf"] or 0.0) for record in coupling
    )
    if correction_divergence > 1.0e-10:
        raise SystemExit(
            "FAIL: compatible-curl transfer lost solenoidality "
            f"(dimensionless Linf={correction_divergence:.3g})"
        )


def _check_reference_drag(forces: np.ndarray) -> str:
    reference_path = CASE_DIR / "reference_flow" / "samples" / "forces_history.csv"
    with reference_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    reference = np.asarray([[float(row["time"]), float(row["drag_coefficient"])] for row in rows])
    overlap = (forces[:, 0] >= reference[0, 0]) & (forces[:, 0] <= reference[-1, 0])
    if np.count_nonzero(overlap) < 10:
        raise SystemExit("FAIL: production run has too few samples for reference comparison")
    time = forces[overlap, 0]
    candidate = forces[overlap, 4]
    expected = np.interp(time, reference[:, 0], reference[:, 1])
    relative_error = np.abs(candidate - expected) / np.maximum(np.abs(expected), 0.25)
    mean_bias = abs(float(np.mean(candidate) - np.mean(expected))) / abs(float(np.mean(expected)))
    if float(np.mean(relative_error)) > 0.15 or float(np.max(relative_error)) > 0.35:
        raise SystemExit(
            "FAIL: cube drag does not track the fully meshed FVM reference "
            f"(mean/max relative error {np.mean(relative_error):.1%}/{np.max(relative_error):.1%})"
        )
    if mean_bias > 0.10:
        raise SystemExit(f"FAIL: cube mean-drag bias versus the reference is {mean_bias:.1%}")
    return (
        f" reference drag mean/max error {np.mean(relative_error):.1%}/{np.max(relative_error):.1%}"
    )


def main() -> None:
    metadata_path = CASE_DIR / "solution" / "run_metadata.json"
    force_path = CASE_DIR / "samples" / "forces_history.csv"
    if not metadata_path.is_file():
        raise SystemExit("FAIL: coupled run metadata was not written")
    expected_end = float(json.loads(metadata_path.read_text())["physics"]["end_time"])

    if not force_path.is_file():
        raise SystemExit("FAIL: cube force history was not written")
    with force_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit("FAIL: cube force history is empty")
    forces = np.asarray(
        [
            [
                float(row[key])
                for key in (
                    "time",
                    "total_force_x",
                    "total_force_y",
                    "total_force_z",
                    "drag_coefficient",
                    "lift_coefficient",
                )
            ]
            for row in rows
        ]
    )
    if not np.all(np.isfinite(forces)):
        raise SystemExit("FAIL: cube force history contains non-finite values")
    if forces[-1, 0] + 1e-10 < expected_end:
        raise SystemExit(
            f"FAIL: force history ends at t={forces[-1, 0]:g}, expected {expected_end:g}"
        )

    diagnostics = _json_lines(CASE_DIR / "solution" / "diagnostics.jsonl")
    max_courant_number, max_continuity = _check_solver_history(diagnostics)

    coupling = _json_lines(CASE_DIR / "solution" / "coupler_diagnostics.jsonl")
    _check_coupling_history(coupling)

    physics_summary = ""
    if expected_end >= 3.0:
        physics_summary = _check_reference_drag(forces)

    print(
        "PASS: native cube run completed with converged FVM solves, "
        f"peak max_courant_number={max_courant_number:.3g}, "
        f"peak continuity={max_continuity:.3g}, "
        f"and solenoidal local transfer through t={forces[-1, 0]:g}.{physics_summary}"
    )


if __name__ == "__main__":
    main()
