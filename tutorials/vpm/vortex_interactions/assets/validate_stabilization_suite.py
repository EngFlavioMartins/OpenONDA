#!/usr/bin/env python3
"""Check and summarize the vortex-interaction stabilization comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .ring_metrics import (
    CASES,
    FIGURES_DIR,
    REFERENCE_TIME,
    SAMPLES_DIR,
    case_style,
    discover_cases,
    read_integrals,
    read_ring_diagnostics,
)

TERMINAL_STATES = {"horizon_reached", "resolution_lost"}
REFERENCE = Path(__file__).resolve().parent / "references" / "leapfrogging_lbm_trajectory.csv"
REFERENCE_ORIGIN = 2.5
COMMON_SETTINGS = (
    "time_step_size",
    "integrator",
    "induction_backend",
    "stretching_scheme",
    "turbulence_model",
    "smagorinsky_coefficient",
    "viscous_scheme",
    "particle_spacing",
    "particle_core_radius",
    "ring_radius",
    "ring_circulation",
    "core_radius",
    "ring_separation",
    "reynolds_number",
    "disturbance_amplitude",
    "disturbance_mode",
    "maximum_lagrangian_cfl",
    "maximum_vorticity_divergence_error",
    "maximum_vortex_misalignment_degrees",
)


def _finite_max(data, column: str) -> float:
    if data is None or column not in data:
        return np.nan
    values = data[column].to_numpy(float)
    finite = values[np.isfinite(values)]
    return float(finite.max()) if finite.size else np.nan


def _last(data, column: str) -> float:
    if data is None or column not in data:
        return np.nan
    values = data[column].to_numpy(float)
    finite = values[np.isfinite(values)]
    return float(finite[-1]) if finite.size else np.nan


def _tube_circulation_drift(case_dir: Path) -> float:
    diagnostics = read_ring_diagnostics(case_dir)
    if diagnostics is None or "tube_circulation" not in diagnostics:
        return np.nan
    largest = 0.0
    found = False
    for _, ring in diagnostics.groupby("group_id", sort=True):
        values = ring["tube_circulation"].to_numpy(float)
        values = values[np.isfinite(values)]
        if not values.size or values[0] <= 0.0:
            continue
        largest = max(largest, float(np.max(np.abs(values / values[0] - 1.0))))
        found = True
    return largest if found else np.nan


def _trajectory_radius_rmse(case_dir: Path) -> float:
    diagnostics = read_ring_diagnostics(case_dir)
    if diagnostics is None or not REFERENCE.is_file():
        return np.nan
    reference = pd.read_csv(REFERENCE)
    errors = []
    for group_id, reference_ring in ((0, 2), (1, 1)):
        numerical = diagnostics[diagnostics.group_id == group_id].sort_values("vortex_centroid_x")
        target = reference[reference.ring == reference_ring].sort_values("x_over_R0")
        if numerical.empty or target.empty:
            continue
        target_x = target.x_over_R0.to_numpy(float) - REFERENCE_ORIGIN
        selected = (target_x >= numerical.vortex_centroid_x.min()) & (
            target_x <= numerical.vortex_centroid_x.max()
        )
        if not np.any(selected):
            continue
        interpolated = np.interp(
            target_x[selected],
            numerical.vortex_centroid_x.to_numpy(float),
            numerical.major_radius.to_numpy(float),
        )
        errors.extend(interpolated - target.R_over_R0.to_numpy(float)[selected])
    return float(np.sqrt(np.mean(np.square(errors)))) if errors else np.nan


def summarize(case_dir: Path, metadata: dict) -> dict:
    """Return the diagnostics used to compare one stabilization method."""
    integrals = read_integrals(case_dir)
    energy_initial = _last(
        integrals.iloc[:1] if integrals is not None else None, "total_kinetic_energy"
    )
    energy_final = _last(integrals, "total_kinetic_energy")
    particles_initial = float(metadata.get("initial_n_particles_total", np.nan))
    particles_final = _last(integrals, "n_particles_total")
    if not np.isfinite(particles_final):
        particles_final = float(metadata.get("final_n_particles_total", np.nan))
    return {
        "case": case_dir.name,
        "label": case_style(case_dir.name)["label"],
        "status": metadata.get("status", "unknown"),
        "completed_steps": int(metadata.get("completed_steps", -1)),
        "final_time": float(metadata.get("final_time", np.nan)),
        "normalized_time": float(metadata.get("final_time", np.nan)) / REFERENCE_TIME,
        "energy_ratio": (
            energy_final / energy_initial
            if np.isfinite(energy_initial) and energy_initial > 0.0
            else np.nan
        ),
        "maximum_energy_rate": _finite_max(integrals, "kinetic_energy_rate"),
        "particle_ratio": (
            particles_final / particles_initial
            if np.isfinite(particles_initial) and particles_initial > 0.0
            else np.nan
        ),
        "stabilization_events": int(_last(integrals, "n_stabilization_events"))
        if np.isfinite(_last(integrals, "n_stabilization_events"))
        else 0,
        "maximum_lagrangian_cfl": _finite_max(integrals, "lagrangian_cfl"),
        "maximum_divergence_error": _finite_max(integrals, "vorticity_divergence_error"),
        "maximum_misalignment_degrees": _finite_max(
            integrals, "vortex_strength_misalignment_degrees"
        ),
        "maximum_tube_circulation_drift": _tube_circulation_drift(case_dir),
        "trajectory_radius_rmse": _trajectory_radius_rmse(case_dir),
    }


def _write_summary(rows: list[dict]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = FIGURES_DIR / "stabilization_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    ranked = sorted(rows, key=lambda row: row["normalized_time"], reverse=True)
    lines = [
        "# VPM stabilization comparison",
        "",
        "All cases use Smagorinsky LES, transposed stretching, SSPRK3, and the same particle field.",
        "",
        "| Method | Status | Steps | $t\\Gamma_0/R_0^2$ | $E/E_0$ | $N/N_0$ | $\\Delta\\Gamma$ | Radius RMSE | Events |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranked:
        lines.append(
            f"| {row['label']} | {row['status']} | {row['completed_steps']} | "
            f"{row['normalized_time']:.3g} | {row['energy_ratio']:.3g} | "
            f"{row['particle_ratio']:.3g} | {row['maximum_tube_circulation_drift']:.3g} | "
            f"{row['trajectory_radius_rmse']:.3g} | {row['stabilization_events']} |"
        )
    lines += [
        "",
        "## How to choose",
        "",
        "Prefer the method that extends the resolved time without producing excessive energy or tube-circulation drift. Check particle growth before choosing filament refinement, and check event count before choosing a grid-based method. A longer run is not an improvement if the ring trajectory or conserved quantities have already departed from the reference.",
        "",
    ]
    (FIGURES_DIR / "stabilization_guidelines.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="require every planned case")
    args = parser.parse_args()

    failures: list[str] = []
    available = {path.name: path for path in discover_cases(SAMPLES_DIR)}
    if args.strict:
        for case_name in CASES:
            if case_name not in available:
                failures.append(f"{case_name}: no compatible samples")
    if not available:
        print("[stabilization] no compatible samples are available")
        return 1 if args.strict else 0

    rows: list[dict] = []
    reference_settings = None
    print("\nStabilization results")
    for case_name in CASES:
        case_dir = available.get(case_name)
        if case_dir is None:
            continue
        metadata_path = case_dir / "run_metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            failures.append(f"{case_name}: invalid run metadata ({error})")
            continue

        if metadata.get("case") != case_name or metadata.get("stabilization") != case_name:
            failures.append(f"{case_name}: metadata identifies a different case")
        if metadata.get("status") not in TERMINAL_STATES:
            failures.append(f"{case_name}: result is not terminal")
        settings = tuple(metadata.get(key) for key in COMMON_SETTINGS)
        if reference_settings is None:
            reference_settings = settings
        elif settings != reference_settings:
            failures.append(f"{case_name}: common physics or numerics differ from the baseline")

        row = summarize(case_dir, metadata)
        integrals = read_integrals(case_dir)
        if integrals is None:
            failures.append(f"{case_name}: missing flow_integrals.csv")
        elif "step" not in integrals or int(integrals["step"].iloc[-1]) != row["completed_steps"]:
            failures.append(f"{case_name}: final sample does not match its terminal step")
        rows.append(row)
        print(
            f"  {row['label']:<28} {row['status']:<15} "
            f"step={row['completed_steps']:4d}  t*={row['normalized_time']:.3g}"
        )

    if rows:
        _write_summary(rows)
    if failures:
        print("\nStabilization comparison FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("Stabilization comparison data are consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
