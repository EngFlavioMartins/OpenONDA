"""Minimal numerical-integrity gate for the native cylinder workflow."""

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


def _check_physics(values: np.ndarray) -> str:
    settled = values[:, 0] >= 0.5 * values[-1, 0]
    time = values[settled, 0]
    cd = values[settled, 3]
    cl = values[settled, 4]
    if len(time) < 24:
        raise SystemExit("FAIL: production cylinder run has too few settled force samples")
    mean_cd = float(np.mean(cd))
    lift_rms = float(np.std(cl))
    if not 0.5 <= mean_cd <= 3.5:
        raise SystemExit(f"FAIL: settled cylinder mean Cd is not physical ({mean_cd:.3g})")
    if not 1e-3 <= lift_rms <= 3.0:
        raise SystemExit(f"FAIL: the cylinder wake has no credible periodic lift ({lift_rms:.3g})")
    uniform_time = np.linspace(time[0], time[-1], len(time))
    signal = np.interp(uniform_time, time, cl)
    signal -= np.mean(signal)
    frequencies = np.fft.rfftfreq(len(signal), d=float(uniform_time[1] - uniform_time[0]))
    peak = 1 + int(np.argmax(np.abs(np.fft.rfft(signal))[1:]))
    strouhal = float(frequencies[peak])
    if not 0.12 <= strouhal <= 0.30:
        raise SystemExit(f"FAIL: settled cylinder Strouhal number is {strouhal:.3g}")
    midpoint = len(cd) // 2
    drift = abs(float(np.mean(cd[midpoint:]) - np.mean(cd[:midpoint]))) / abs(mean_cd)
    if drift > 0.25:
        raise SystemExit(f"FAIL: settled cylinder mean drag is still drifting ({drift:.1%})")
    return f" mean Cd={mean_cd:.3f}, lift RMS={lift_rms:.3f}, St={strouhal:.3f};"


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
        raise SystemExit("FAIL: no immersed-cylinder force samples were written")
    values = np.array(
        [[float(row[key]) for key in ("time", "Fx", "Fy", "Cd", "Cl", "slip")] for row in rows]
    )
    if not np.all(np.isfinite(values)):
        raise SystemExit("FAIL: cylinder force history contains non-finite values")
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
    max_cfl = max(float(record["cfl_max"]) for record in diagnostics)
    max_continuity = max(float(record["continuity_max"]) for record in diagnostics)
    if max_cfl > 5.0:
        raise SystemExit(f"FAIL: peak FVM CFL is excessive ({max_cfl:.3g})")
    if max_continuity > 1e-3:
        raise SystemExit(f"FAIL: peak continuity residual is excessive ({max_continuity:.3g})")
    if values[-1, 5] > 5.0:
        raise SystemExit(f"FAIL: final IBM no-slip error is excessive ({values[-1, 5]:.3g})")

    coupling = _json_lines(CASE_DIR / "solution" / "coupler_diagnostics.jsonl")
    circulation_error = max(
        abs(float(record["conservation"]["corrected_mismatch"]["circulation"]))
        for record in coupling
    )
    if circulation_error > 1e-8:
        raise SystemExit(f"FAIL: corrected handoff circulation mismatch is {circulation_error:.3g}")
    donor_error = max(abs(float(record["donor_flux"]["corrected_mismatch"])) for record in coupling)
    if donor_error > 1e-8:
        raise SystemExit(f"FAIL: corrected donor-flux mismatch is {donor_error:.3g}")
    physics_summary = _check_physics(values) if expected_end >= 40.0 else ""
    print(
        "PASS: native cylinder run completed with converged FVM solves,"
        f"{physics_summary} peak CFL={max_cfl:.3g}, peak continuity={max_continuity:.3g}, "
        f"bounded no-slip error, and conservative handoff through t={values[-1, 0]:g}."
    )


if __name__ == "__main__":
    main()
