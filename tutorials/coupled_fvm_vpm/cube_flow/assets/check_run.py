"""Numerical-integrity gate for the native cube FVM–VPM workflow."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from measure_trial_errors import frame, load_table, profile_record

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
    for record in coupling:
        transfer = record["transfer"]
        expected_after = (
            int(transfer["n_particles_before"])
            - int(transfer["n_particles_removed"])
            + int(transfer["n_particles_injected"])
        )
        if int(transfer["n_particles_after"]) != expected_after:
            raise SystemExit("FAIL: inconsistent overlap-replacement particle budget")
        circulation_budget = [
            float(transfer[f"state_change_vortex_strength_net_{axis}"]) for axis in "xyz"
        ]
        if not np.all(np.isfinite(circulation_budget)):
            raise SystemExit("FAIL: non-finite overlap-replacement circulation budget")


def _check_reference_accuracy() -> str:
    """Require every sampled Cd and line-profile maximum to remain within 5%."""
    samples = CASE_DIR / "samples"
    reference_samples = CASE_DIR / "reference_flow" / "samples"
    candidate_force = load_table(samples / "forces_history.csv")
    reference_force = load_table(reference_samples / "forces_history.csv")
    measurements: list[tuple[str, float]] = []

    for time, drag in zip(
        candidate_force["time"], candidate_force["drag_coefficient"], strict=True
    ):
        if time < reference_force["time"].min() or time > reference_force["time"].max():
            continue
        expected = float(
            np.interp(time, reference_force["time"], reference_force["drag_coefficient"])
        )
        measurements.append((f"Cd@{time:g}", abs(float(drag) - expected) / abs(expected)))

    for name in ("centreline", "offaxis_y075"):
        reference = load_table(reference_samples / f"{name}.csv")
        for source in ("fvm", "vpm"):
            candidate = load_table(samples / f"{source}_{name}.csv")
            for time in np.unique(candidate["time"]):
                candidate_frame = frame(candidate, float(time))
                reference_frame = frame(reference, float(time))
                if candidate_frame is None or reference_frame is None:
                    continue
                record = profile_record(
                    source,
                    name,
                    float(time),
                    candidate_frame,
                    reference_frame,
                )
                measurements.append(
                    (
                        f"{source}-{name}@{time:g}",
                        float(record["max_abs_over_u_inf"]),
                    )
                )

    if not measurements:
        raise SystemExit("FAIL: no coincident reference metrics were available")
    failed = [(name, value) for name, value in measurements if value > 0.05]
    if failed:
        detail = ", ".join(f"{name}={value:.2%}" for name, value in failed[:8])
        raise SystemExit(f"FAIL: reference errors exceed 5%: {detail}")
    worst_name, worst_value = max(measurements, key=lambda item: item[1])
    return f" worst reference error {worst_name}={worst_value:.2%}"


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

    physics_summary = _check_reference_accuracy()

    print(
        "PASS: native cube run completed with converged FVM solves, "
        f"peak max_courant_number={max_courant_number:.3g}, "
        f"peak continuity={max_continuity:.3g}, "
        f"and solenoidal local transfer through t={forces[-1, 0]:g}.{physics_summary}"
    )


if __name__ == "__main__":
    main()
