#!/usr/bin/env python3
"""Certification checks for the four-rotor climb tutorial."""

from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd


CASE_DIR = Path(__file__).resolve().parents[1]
SAMPLES_DIR = CASE_DIR / "samples" / "quadcopter"
FIGURES_DIR = CASE_DIR / "figures"
N_STEPS = 288
TIME_STEP_SIZE = np.deg2rad(7.5) / (200.0 * 2.0 * np.pi / 60.0)


def _read_csv(path: Path, failures: list[str]) -> pd.DataFrame | None:
    if not path.is_file():
        failures.append(f"missing {path.name}")
        return None
    try:
        data = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        failures.append(f"unreadable {path.name}: {error}")
        return None
    numeric = data.select_dtypes(include=[np.number])
    if data.empty or numeric.empty or not np.isfinite(numeric.to_numpy()).all():
        failures.append(f"{path.name}: empty or non-finite numeric data")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-plot", action="store_true")
    args = parser.parse_args()
    failures: list[str] = []
    manifest_path = SAMPLES_DIR / "run_manifest.json"
    if not manifest_path.is_file():
        failures.append("missing run_manifest.json")
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            failures.append(f"unreadable run_manifest.json: {error}")
        else:
            if manifest.get("status") != "complete" or manifest.get("completed") is not True:
                failures.append("run manifest does not certify a completed run")
            if not np.isclose(float(manifest.get("final_time", np.nan)), N_STEPS * TIME_STEP_SIZE):
                failures.append("run manifest final time is inconsistent with six revolutions")
    integrals = _read_csv(SAMPLES_DIR / "flow_integrals.csv", failures)
    if integrals is not None:
        required = {"time", "step", "n_particles_total", "total_enstrophy"}
        if not required.issubset(integrals.columns):
            failures.append(
                f"flow_integrals.csv: missing {sorted(required - set(integrals.columns))}"
            )
        else:
            steps = integrals["step"].to_numpy(int)
            times = integrals["time"].to_numpy(float)
            if steps.size < 4 or steps[-1] != N_STEPS:
                failures.append(
                    f"flow-integral history ends at step {steps[-1] if steps.size else 'missing'}, expected {N_STEPS}"
                )
            if (
                steps.size
                and np.any(np.diff(steps) <= 0)
                or times.size
                and np.any(np.diff(times) <= 0.0)
            ):
                failures.append("flow-integral cadence is not strictly increasing")
            particles = integrals["n_particles_total"].to_numpy(float)
            enstrophy = integrals["total_enstrophy"].to_numpy(float)
            if np.any(particles <= 0.0) or np.any(enstrophy < 0.0):
                failures.append("particle count or enstrophy is nonphysical")
            if particles.size > 1 and np.any(np.diff(particles) < -0.5 * particles[:-1]):
                failures.append("particle population has an unexplained >50% one-sample loss")
            tail = particles[max(0, int(0.7 * particles.size)) :]
            if tail.size > 1:
                drift = float(np.ptp(tail) / max(float(np.mean(tail)), 1.0e-12))
                print(f"particle count tail relative range={drift:.3%}")

    forces = _read_csv(SAMPLES_DIR / "vlm_forces.csv", failures)
    if forces is not None:
        required = {
            "time",
            "step",
            "force_x",
            "force_y",
            "force_z",
            "moment_x",
            "moment_y",
            "moment_z",
        }
        if not required.issubset(forces.columns):
            failures.append(f"vlm_forces.csv: missing {sorted(required - set(forces.columns))}")
        elif forces["step"].max() != N_STEPS:
            failures.append("vlm force history does not reach the final step")

    blade_files = sorted(SAMPLES_DIR.glob("vlm_spanwise_rotor_*_blade_*.csv"))
    if len(blade_files) != 8:
        failures.append(f"expected 8 per-blade loading files, found {len(blade_files)}")
    for path in blade_files:
        data = _read_csv(path, failures)
        if data is not None and "section_force_z" not in data:
            failures.append(f"{path.name}: missing section_force_z")

    plane_files = list(SAMPLES_DIR.glob("sampled_zplane_*.vts"))
    if not plane_files:
        failures.append("missing sampled_zplane VTK output")

    if not args.pre_plot:
        for extension in ("png", "pdf"):
            for name in ("quadcopter_particle_count", "quadcopter_vorticity_history"):
                figure = FIGURES_DIR / f"{name}.{extension}"
                if not figure.is_file() or figure.stat().st_size == 0:
                    failures.append(f"missing or empty figure {figure.name}")

    if failures:
        print("\n".join(f"[FAIL] {failure}" for failure in failures))
        return 1
    print("[OK] quadcopter certification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
