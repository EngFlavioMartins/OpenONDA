#!/usr/bin/env python3
"""State of the sensitivity study: every run of the current version, how far it is, and whether
its coupling diagnostics stay healthy.

A run is flagged when its particle image sums left the GPU (backend other than cupy_cuda, or a
memory fallback), when the mean boundary velocity mismatch exceeds 0.02 U, when coupling
iterations stop converging, when the mean drag after t = 20 s leaves 1.2-1.45, or when its
output has not advanced for two hours while the study runs.

Run from the case directory: python -m assets.study_status
"""

import json
from pathlib import Path
import time

import numpy as np

CASE_DIR = Path(__file__).resolve().parents[1]
STUDY_DIR = CASE_DIR / "solution/sensitivity_analysis"
MISMATCH_LIMIT = 0.02  # U, mean boundary velocity mismatch
DRAG_RANGE = (1.2, 1.45)  # developed mean drag after DRAG_START
DRAG_START = 20.0  # s
STALE_SECONDS = 7200.0


def last_records(path, count=50):
    """The last ``count`` records of a JSON-lines file, read from its end."""
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - 400_000 * min(count, 10)))
        lines = handle.read().decode(errors="ignore").splitlines()[1:]
    return [json.loads(line) for line in lines[-count:] if line.strip()]


def status(directory):
    forces = next(directory.glob("samples/forces_history.csv"), None)
    diagnostics = next(directory.glob("solution/coupler_diagnostics.jsonl"), None)
    row = {"run": directory.name, "time": np.nan, "flags": []}
    if forces is None:
        log = next(directory.glob("run.log"), None)
        row["state"] = "not started" if log is None else "meshing or failed"
        if log is not None and "Traceback" in log.read_text(errors="ignore"):
            row["state"] = "FAILED: " + log.read_text(errors="ignore").strip().splitlines()[-1][:90]
        return row
    data = np.atleast_1d(np.genfromtxt(forces, delimiter=",", names=True, dtype=None, encoding="utf-8"))
    row["time"] = data["time"][-1]
    row["age"] = time.time() - forces.stat().st_mtime
    late = data[data["time"] >= DRAG_START]
    if late.size > 25:
        drag = late["drag_coefficient"][-int(15.0 / np.diff(data["time"]).mean()) :].mean()
        row["drag"] = drag
        if not DRAG_RANGE[0] <= drag <= DRAG_RANGE[1]:
            row["flags"].append(f"mean drag {drag:.3f}")
    if diagnostics is not None:
        records = last_records(diagnostics)
        finite = [r["last_induction_image_call"]["mesh"]["finite_evaluation"] for r in records
                  if "last_induction_image_call" in r]
        backends = {f["execution_backend"] for f in finite}
        if backends - {"cupy_cuda"}:
            row["flags"].append("image backend " + "+".join(sorted(backends)))
        if any(f["memory_fallback"] is not None for f in finite):
            row["flags"].append("GPU memory fallback")
        mismatch = np.mean([r["fvm_boundary_data"]["mean_velocity_mismatch"] for r in records])
        row["mismatch"] = mismatch
        if mismatch > MISMATCH_LIMIT:
            row["flags"].append(f"mismatch {mismatch:.3g} U")
        converged = np.mean([r.get("coupling_iteration", {}).get("converged", True) for r in records])
        if converged < 0.9:
            row["flags"].append(f"converged {100 * converged:.0f}%")
        row["kappa"] = np.median([r["transfer"]["strength_ratio"] for r in records if "transfer" in r])
        row["particles"] = records[-1].get("transfer", {}).get("n_particles_after", 0)
        times = [r["time"] for r in records]
        if len(times) > 1:
            row["cost"] = np.mean([r["timing_seconds"]["total"] for r in records]) / np.diff(times).mean()
    return row


def main():
    # A trailing separator matches directories only
    directories = sorted(STUDY_DIR.glob("box*/"))
    print(f"{'run':52s} {'t [s]':>7s} {'s/s':>6s} {'N_p':>7s} {'kappa':>6s} {'mismatch':>9s}  flags")
    for directory in directories:
        row = status(directory)
        if "state" in row:
            if row["state"] != "not started":
                print(f"{row['run']:52s} {row['state']}")
            continue
        flags = list(row["flags"])
        if row.get("age", 0.0) > STALE_SECONDS and row["time"] < 54.9:
            flags.append(f"no output for {row['age'] / 3600:.1f} h")
        print(
            f"{row['run']:52s} {row['time']:7.2f} {row.get('cost', np.nan):6.1f} "
            f"{row.get('particles', 0):7.0f} {row.get('kappa', np.nan):6.2f} "
            f"{row.get('mismatch', np.nan):9.2e}  {'; '.join(flags)}"
        )


if __name__ == "__main__":
    main()
