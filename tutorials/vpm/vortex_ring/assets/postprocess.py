#!/usr/bin/env python3
"""Post-run validation for the vortex-ring tutorial.

Default mode runs strict completeness/physics checks.
Use ``--manifest`` to write the JSON status/provenance manifest instead.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import h5py

sys.path.insert(0, str(Path(__file__).parent))
from ring_metrics import (
    FIGURES_DIR,
    SAMPLES_DIR,
    SOLUTION_DIR,
    REFERENCE_TIME,
    REFERENCE_VELOCITY,
    load_ring_data,
    load_ring_speed,
    saffman_speed,
)


# =============================================================
# Validation
# =============================================================


VARIANTS = ("dns_direct", "dns_transposed", "dns_mixed", "les_transposed")
ALLOWED_RUN_STATUSES = {"completed", "resolution_lost"}


def _expected_backup_steps(completed_steps: int, interval_steps: int) -> set[int]:
    """Return initial, periodic, and final backups retained by a run."""
    if completed_steps < 0:
        raise ValueError("completed_steps must be non-negative")
    if interval_steps <= 0:
        raise ValueError("interval_steps must be positive")
    return {
        0,
        completed_steps,
        *range(interval_steps, completed_steps + 1, interval_steps),
    }


def _run_contract(name: str) -> tuple[dict, set[int], list[str]]:
    """Load and cross-check the manifest and final canonical backup."""
    failures: list[str] = []
    manifest_path = SOLUTION_DIR / f"run_manifest_{name}.json"
    manifest = _metadata(manifest_path)
    if not manifest:
        return {}, set(), [f"{name}: missing or unreadable {manifest_path.name}"]

    try:
        requested_steps = int(manifest["requested_steps"])
        completed_steps = int(manifest["completed_steps"])
        completed_time = float(manifest["completed_time"])
        interval_steps = int(manifest["backup"]["interval_steps"])
    except (KeyError, TypeError, ValueError) as error:
        return manifest, set(), [f"{name}: invalid run metadata ({error})"]

    final_path = SOLUTION_DIR / name / f"vpm_{completed_steps:06d}.h5"
    try:
        with h5py.File(final_path, "r") as archive:
            final_step = int(archive["solver"].attrs["step"])
            final_time = float(archive["solver"].attrs["time"])
    except (OSError, KeyError, TypeError, ValueError) as error:
        return manifest, set(), [f"{name}: invalid final backup ({error})"]

    try:
        expected_steps = _expected_backup_steps(completed_steps, interval_steps)
    except ValueError as error:
        return manifest, set(), [f"{name}: invalid backup cadence ({error})"]

    status = manifest.get("status")
    if status not in ALLOWED_RUN_STATUSES:
        failures.append(f"{name}: unsupported run status {status!r}")
    if completed_steps > requested_steps:
        failures.append(
            f"{name}: completed_steps {completed_steps} exceeds requested_steps {requested_steps}"
        )
    if status == "completed" and completed_steps != requested_steps:
        failures.append(
            f"{name}: completed run stopped at step {completed_steps}; requested {requested_steps}"
        )
    if status == "resolution_lost":
        if completed_steps >= requested_steps:
            failures.append(f"{name}: resolution_lost status does not describe an early stop")
        if not manifest.get("termination_reason"):
            failures.append(f"{name}: resolution_lost status has no termination_reason")
    if name == "les_transposed" and status != "completed":
        failures.append(f"{name}: reference LES run did not reach its requested horizon")

    if manifest.get("variant") != name:
        failures.append(f"{name}: manifest variant is {manifest.get('variant')!r}")
    if final_step != completed_steps:
        failures.append(
            f"{name}: final backup step {final_step}; manifest reports {completed_steps}"
        )
    time_tolerance = max(1.0e-12, abs(completed_time) * 1.0e-10)
    if abs(final_time - completed_time) > time_tolerance:
        failures.append(
            f"{name}: final backup time {final_time}; manifest reports {completed_time}"
        )
    return manifest, expected_steps, failures


def validate(pre_plot: bool) -> int:
    failures: list[str] = []

    speed_tolerances = {
        "dns_direct": 0.15,
        "dns_transposed": 0.10,
        "dns_mixed": 0.15,
        "les_transposed": 0.12,
    }
    for name, speed_tol in speed_tolerances.items():
        manifest, expected_steps, contract_failures = _run_contract(name)
        failures.extend(contract_failures)
        if contract_failures:
            continue

        all_files = sorted(glob.glob(str(SOLUTION_DIR / name / "vpm_*.h5")))
        numbered = {
            int(match.group(1)): path
            for path in all_files
            if (match := re.search(r"vpm_(\d{6})\.h5$", path))
        }
        if set(numbered) != expected_steps:
            missing = sorted(expected_steps - set(numbered))
            unexpected = sorted(set(numbered) - expected_steps)
            failures.append(
                f"{name}: numbered backup history disagrees with the run contract "
                f"(missing={missing}, unexpected={unexpected})"
            )
            continue
        files = [numbered[step] for step in sorted(numbered)]
        raw = load_ring_data(files)
        entries = raw.get(0, [])
        if len(entries) != len(files):
            failures.append(f"{name}: unreadable or unbounded snapshots")
            continue
        status = manifest["status"]
        completed_steps = int(manifest["completed_steps"])
        if len(entries) >= 2:
            r = np.array([x["major_radius"] for x in entries])
            impulse = np.array([x["linear_impulse_magnitude"] for x in entries])
            tube_circulation = np.array([x["tube_circulation"] for x in entries])
            r_drift = float(np.max(np.abs(r / r[0] - 1.0)))
            i_drift = float(np.max(np.abs(impulse / impulse[0] - 1.0)))
            g_drift = float(np.max(np.abs(tube_circulation / tube_circulation[0] - 1.0)))
            t, u = load_ring_speed(files)
            ref = saffman_speed(t * REFERENCE_TIME) / REFERENCE_VELOCITY
            rel_rmse = float(np.sqrt(np.mean((u - ref) ** 2)) / np.mean(ref))
            print(
                f"{name}: status={status} at step {completed_steps}, dR={r_drift:.3%}, "
                f"d|I|={i_drift:.3%}, dcirculation={g_drift:.3%}, "
                f"speed RMSE={rel_rmse:.3%}"
            )
            # A deliberately terminated DNS run remains useful evidence of a
            # formulation's stability limit.  Apply the quantitative physics
            # gate only to runs that reached their requested horizon.
            if status == "completed":
                if r_drift > 0.05:
                    failures.append(f"{name}: major-radius drift {r_drift:.3%} > 5%")
                if i_drift > 0.05 or g_drift > 0.08 or rel_rmse > speed_tol:
                    failures.append(f"{name}: conservation/speed tolerance exceeded")
        else:
            print(
                f"{name}: status={status} at step {completed_steps}; "
                "fewer than two periodic backups before the documented stop"
            )

        samples = SAMPLES_DIR / name
        for csv_name in ("flow_integrals.csv", "ring_diagnostics.csv", "ring_modes.csv"):
            csv_path = samples / csv_name
            if not csv_path.is_file():
                failures.append(f"{name}: missing {csv_name}")
                continue
            try:
                data = pd.read_csv(csv_path)
            except (OSError, ValueError, pd.errors.ParserError) as error:
                failures.append(f"{name}: unreadable {csv_name} ({error})")
                continue
            if data.empty or not np.isfinite(data.select_dtypes(include=[np.number])).all().all():
                failures.append(f"{name}: empty or non-finite {csv_name}")
        final = SOLUTION_DIR / name / f"vpm_{completed_steps:06d}.h5"
        if not final.is_file():
            failures.append(f"{name}: missing final restart state")

    if not pre_plot:
        for extension in ("png", "pdf"):
            for name in (
                "vortex_ring_motion",
                "vortex_ring_energy",
                "vortex_ring_circulation",
            ):
                figure = FIGURES_DIR / f"{name}.{extension}"
                if not figure.is_file() or figure.stat().st_size == 0:
                    failures.append(f"missing or empty figure {figure.name}")
    if failures:
        print("\n".join(f"[FAIL] {x}" for x in failures))
        return 1
    print("[OK] vortex_ring certification passed")
    return 0


# =============================================================
# Manifest generation (structural placeholder)
# =============================================================


def _metadata(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def build_manifest(samples_dir: Path, figures_dir: Path) -> dict:
    runs = {}
    solution_dir = samples_dir.parent / "solution"
    for variant in VARIANTS:
        metadata = _metadata(solution_dir / f"run_manifest_{variant}.json")
        runs[variant] = {
            "status": metadata.get("status", "missing"),
            "completed_steps": metadata.get("completed_steps"),
            "completed_time": metadata.get("completed_time"),
            "n_particles_total": metadata.get("n_particles_total"),
        }
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs": runs,
        "figures": sorted(path.name for path in figures_dir.glob("*.png")),
    }


def write_manifest() -> int:
    manifest = build_manifest(SAMPLES_DIR, FIGURES_DIR)
    output = FIGURES_DIR / "postprocessing_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(output)
    counts = {}
    for run in manifest["runs"].values():
        counts[run["status"]] = counts.get(run["status"], 0) + 1
    print(f"  [status] {counts}; wrote {output}")
    return 0


# =============================================================
# CLI
# =============================================================


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-plot", action="store_true", help="skip figure existence checks")
    parser.add_argument("--manifest", action="store_true", help="write JSON status manifest")
    args = parser.parse_args()
    if args.manifest:
        return write_manifest()
    return validate(pre_plot=args.pre_plot)


if __name__ == "__main__":
    raise SystemExit(main())
