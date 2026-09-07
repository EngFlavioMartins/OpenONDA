#!/usr/bin/env python3
"""Check and summarize the vortex-ring instability results.

Strict validation certifies a comparable four-case instability-onset campaign.
``--available`` only checks the sample files consumed by the plots, allowing
figures to be rebuilt while another variant is still running.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from tutorials.vpm.vortex_ring.assets.ring_metrics import (
    FIGURES_DIR,
    SAMPLES_DIR,
    SOLUTION_DIR,
    REFERENCE_TIME,
    REFERENCE_VELOCITY,
    VARIANT_LABEL,
    load_ring_data,
    load_ring_speed,
    load_sampled_ring_speed,
    load_sampled_ring_data,
    plot_variants,
    saffman_valid_time_limit,
    saffman_speed,
)

VARIANTS = ("dns_direct", "dns_transposed", "dns_mixed", "les_transposed")
ALLOWED_RUN_STATUSES = {"horizon_reached", "instability_detected"}
INITIAL_HEALTH_COLUMNS = {
    "strain_increment_infinity",
    "strain_increment_spectral",
    "maximum_particle_strength",
    "maximum_particle_vorticity",
}


def _expected_backup_steps(completed_steps: int, interval_steps: int) -> set[int]:
    """Return the periodic snapshots written before instability."""
    if completed_steps < 0:
        raise ValueError("completed_steps must be non-negative")
    if interval_steps <= 0:
        raise ValueError("interval_steps must be positive")
    return set(range(interval_steps, completed_steps + 1, interval_steps))


def _metadata(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _run_validation(name: str) -> tuple[dict, set[int], list[str]]:
    """Check the parameters and measured instability time for one case."""
    failures: list[str] = []
    metadata_path = SAMPLES_DIR / name / "run_metadata.json"
    metadata = _metadata(metadata_path)
    if not metadata:
        return {}, set(), [f"{name}: missing or unreadable run_metadata.json"]

    try:
        requested_steps = int(metadata["requested_steps"])
        completed_steps = int(metadata["completed_steps"])
        completed_time = float(metadata["final_time"])
        interval_steps = int(metadata["backup_interval_steps"])
    except (KeyError, TypeError, ValueError) as error:
        return metadata, set(), [f"{name}: invalid run metadata ({error})"]

    try:
        expected_steps = _expected_backup_steps(completed_steps, interval_steps)
    except ValueError as error:
        return metadata, set(), [f"{name}: invalid backup cadence ({error})"]

    expected = {
        "schema_version": 4,
        "variant": name,
        "integrator": "SSPRK3",
        "induction_backend": "TREECODE",
        "strength_rate_mode": "HIERARCHICAL_GRADIENT",
        "stretching_scheme": (
            "DIRECT" if name == "dns_direct" else "MIXED" if name == "dns_mixed" else "TRANSPOSED"
        ),
        "viscous_scheme": "CS",
        "stabilization": "DISABLED",
        "health_limit_action": "STOP",
        "experiment": "stretching_instability_onset",
        "maximum_lagrangian_cfl": 1.0,
        "maximum_vorticity_divergence_error": 0.12,
        "maximum_vortex_misalignment_degrees": 25.0,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            failures.append(f"{name}: {key} is {metadata.get(key)!r}; expected {value!r}")
    expected_turbulence = "LES_SMAGORINSKY" if name == "les_transposed" else "DNS"
    if metadata.get("turbulence_model") != expected_turbulence:
        failures.append(
            f"{name}: turbulence_model is {metadata.get('turbulence_model')!r}; "
            f"expected {expected_turbulence!r}"
        )
    status = metadata.get("status")
    if status not in ALLOWED_RUN_STATUSES:
        failures.append(f"{name}: unsupported run status {status!r}")
    if completed_steps < 0 or requested_steps < 0 or completed_steps > requested_steps:
        failures.append(
            f"{name}: invalid progress completed={completed_steps}, requested={requested_steps}"
        )
    if status == "horizon_reached" and (
        not metadata.get("completed") or completed_steps != requested_steps
    ):
        failures.append(f"{name}: horizon_reached status does not reach the requested horizon")
    if status == "instability_detected":
        if metadata.get("completed") or completed_steps >= requested_steps:
            failures.append(f"{name}: instability_detected status does not describe an early stop")
        if not metadata.get("instability_reason"):
            failures.append(f"{name}: instability_detected status has no reason")
        if metadata.get("instability_step") != completed_steps:
            failures.append(f"{name}: instability_step does not match the last step")
        if not np.isclose(
            float(metadata.get("instability_time", -1.0)),
            completed_time,
            rtol=0.0,
            atol=max(1.0e-12, abs(completed_time) * 1.0e-10),
        ):
            failures.append(f"{name}: instability_time does not match the last time")
    for csv_name in ("flow_integrals.csv", "ring_diagnostics.csv", "ring_modes.csv"):
        try:
            data = pd.read_csv(SAMPLES_DIR / name / csv_name)
            last_step = int(data["step"].iloc[-1])
            last_time = float(data["time"].iloc[-1])
        except (
            OSError,
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            pd.errors.ParserError,
        ) as error:
            failures.append(f"{name}: invalid {csv_name} ({error})")
            continue
        if last_step != completed_steps or not np.isclose(
            last_time,
            completed_time,
            rtol=0.0,
            atol=max(1.0e-12, abs(completed_time) * 1.0e-10),
        ):
            failures.append(f"{name}: {csv_name} does not end at the measured instability time")
    return metadata, expected_steps, failures


def _readable_finite_csv(path: Path) -> bool:
    try:
        data = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return False
    numeric = data.select_dtypes(include=[np.number]).copy()
    if data.empty or numeric.empty:
        return False
    if not {"time", "step"}.issubset(data.columns):
        return False
    for column in INITIAL_HEALTH_COLUMNS & set(numeric.columns):
        invalid = numeric[column].isna() & (data["step"] != 0)
        if invalid.any():
            return False
        numeric[column] = numeric[column].fillna(0.0)
    return bool(np.isfinite(numeric).all().all())


def _figure_failures() -> list[str]:
    failures = []
    for extension in ("png", "pdf"):
        for name in (
            "vortex_ring_motion",
            "vortex_ring_energy",
            "vortex_ring_circulation",
            "vortex_ring_stability",
        ):
            figure = FIGURES_DIR / f"{name}.{extension}"
            if not figure.is_file() or figure.stat().st_size == 0:
                failures.append(f"missing or empty figure {figure.name}")
    return failures


def validate_available(pre_plot: bool) -> int:
    """Validate only the existing sample histories needed to draw the figures."""
    failures: list[str] = []
    available: dict[str, list[str]] = {}
    variants = plot_variants(SAMPLES_DIR)
    for csv_name in ("ring_diagnostics.csv", "flow_integrals.csv"):
        usable = []
        for name in variants:
            csv_path = SAMPLES_DIR / name / csv_name
            if not csv_path.is_file():
                continue
            if not _readable_finite_csv(csv_path):
                failures.append(f"{name}: empty or non-finite {csv_name}")
                continue
            usable.append(name)
        available[csv_name] = usable
        if not usable:
            failures.append(f"no readable {csv_name} is available")

    if not pre_plot:
        failures.extend(_figure_failures())
    if failures:
        print("\n".join(f"[FAIL] {failure}" for failure in failures))
        return 1
    summary = "; ".join(f"{csv_name}={','.join(names)}" for csv_name, names in available.items())
    print(f"[OK] available vortex-ring plot inputs: {summary}")
    return 0


def validate(pre_plot: bool) -> int:
    """Require a complete, comparable instability-onset campaign."""
    failures: list[str] = []
    outcomes: list[tuple[float, str, str]] = []
    for name in VARIANTS:
        metadata, expected_steps, validation_failures = _run_validation(name)
        failures.extend(validation_failures)
        if validation_failures:
            continue
        outcomes.append((float(metadata["final_time"]), name, str(metadata["status"])))

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
                f"{name}: numbered backup history disagrees with run metadata "
                f"(missing={missing}, unexpected={unexpected})"
            )
            continue
        files = [numbered[step] for step in sorted(numbered)]
        raw = load_ring_data(files)
        entries = raw.get(0, [])
        if len(entries) != len(files):
            failures.append(f"{name}: unreadable or non-finite snapshots")
            continue
        if len(entries) >= 2:
            samples = load_sampled_ring_data(SAMPLES_DIR / name / "ring_diagnostics.csv")
            radii = samples["major_radius"].to_numpy()
            impulse = samples["linear_impulse_magnitude"].to_numpy()
            circulation = samples["tube_circulation"].to_numpy()
            radius_drift = float(np.max(np.abs(radii / radii[0] - 1.0)))
            impulse_drift = float(np.max(np.abs(impulse / impulse[0] - 1.0)))
            circulation_drift = float(np.max(np.abs(circulation / circulation[0] - 1.0)))
            time, velocity = load_sampled_ring_speed(SAMPLES_DIR / name / "ring_diagnostics.csv")
            valid_theory = time * REFERENCE_TIME <= saffman_valid_time_limit()
            comparison_time = time[valid_theory]
            comparison_velocity = velocity[valid_theory]
            reference = saffman_speed(comparison_time * REFERENCE_TIME) / REFERENCE_VELOCITY
            relative_rmse = float(
                np.sqrt(np.mean(((comparison_velocity - reference) / reference) ** 2))
            )
            status = metadata["status"]
            print(
                f"{name}: status={status}, step={metadata['completed_steps']}, "
                f"max |dR|={100 * radius_drift:.2g}%, "
                f"max |dI|={100 * impulse_drift:.2g}%, max |dGamma_tube|={100 * circulation_drift:.2g}%, "
                f"relative speed RMS={100 * relative_rmse:.2g}% (sample histories, initial references)"
            )
        else:
            print(
                f"{name}: status={metadata['status']}, "
                f"step={metadata['completed_steps']}; no periodic snapshot before instability"
            )

        samples = SAMPLES_DIR / name
        for csv_name in ("flow_integrals.csv", "ring_diagnostics.csv", "ring_modes.csv"):
            csv_path = samples / csv_name
            if not csv_path.is_file():
                failures.append(f"{name}: missing {csv_name}")
            elif not _readable_finite_csv(csv_path):
                failures.append(f"{name}: empty or non-finite {csv_name}")

    if not pre_plot:
        failures.extend(_figure_failures())
    if outcomes:
        ordered = sorted(outcomes, reverse=True)
        ranking = ", ".join(
            f"{VARIANT_LABEL[name]} ({time / REFERENCE_TIME:.3f}, {status})"
            for time, name, status in ordered
        )
        print(f"[stability] longest sustained time first: {ranking}")
    if failures:
        print("\n".join(f"[FAIL] {failure}" for failure in failures))
        return 1
    print("[OK] vortex_ring certification passed")
    return 0


def build_manifest(samples_dir: Path, figures_dir: Path) -> dict:
    """Summarize the measured instability times."""
    runs = {}
    for variant in plot_variants(samples_dir):
        variant_samples = samples_dir / variant
        metadata = _metadata(variant_samples / "run_metadata.json")
        if metadata.get("status") == "running":
            observed_rows = []
            for csv_name in ("flow_integrals.csv", "ring_diagnostics.csv"):
                try:
                    data = pd.read_csv(variant_samples / csv_name)
                    if not data.empty and {"step", "time"}.issubset(data.columns):
                        observed_rows.append(data.loc[data["step"].idxmax()])
                except (OSError, ValueError, pd.errors.ParserError):
                    continue
            if observed_rows:
                latest = max(observed_rows, key=lambda row: int(row["step"]))
                particle_count = latest.get("n_particles_total")
                metadata = dict(metadata)
                metadata.setdefault("status", "partial")
                metadata["completed_steps"] = int(latest["step"])
                metadata["final_time"] = float(latest["time"])
                metadata["final_n_particles_total"] = (
                    int(particle_count) if pd.notna(particle_count) else None
                )
        runs[variant] = {
            "status": metadata.get("status", "missing"),
            "completed_steps": metadata.get("completed_steps"),
            "requested_steps": metadata.get("requested_steps"),
            "completed_time": metadata.get("final_time"),
            "n_particles_total": metadata.get("final_n_particles_total"),
            "induction_backend": metadata.get("induction_backend"),
            "strength_rate_mode": metadata.get("strength_rate_mode"),
            "stretching_scheme": metadata.get("stretching_scheme"),
            "instability_step": metadata.get("instability_step"),
            "instability_time": metadata.get("instability_time"),
            "instability_reason": metadata.get("instability_reason"),
        }
    ranked = sorted(
        (
            (float(run["completed_time"]), name)
            for name, run in runs.items()
            if run["status"] in ALLOWED_RUN_STATUSES and run["completed_time"] is not None
        ),
        reverse=True,
    )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs": runs,
        "stability_ranking": [name for _, name in ranked],
        "longest_sustained_variant": ranked[0][1]
        if len(ranked) == 1 or (len(ranked) > 1 and ranked[0][0] > ranked[1][0])
        else None,
        "longest_sustained_variants": [name for time, name in ranked if time == ranked[0][0]],
        "figures": sorted(path.name for path in figures_dir.glob("*.pdf")),
    }


def write_manifest() -> int:
    manifest = build_manifest(SAMPLES_DIR, FIGURES_DIR)
    output = FIGURES_DIR / "postprocessing_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    counts: dict[str, int] = {}
    for run in manifest["runs"].values():
        status = run["status"]
        counts[status] = counts.get(status, 0) + 1
    print(f"  [status] {counts}; wrote {output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-plot", action="store_true", help="skip figure existence checks")
    parser.add_argument(
        "--available",
        action="store_true",
        help="validate available plotting inputs without requiring a complete campaign",
    )
    parser.add_argument("--manifest", action="store_true", help="write JSON status manifest")
    args = parser.parse_args()
    if args.manifest:
        return write_manifest()
    if args.available:
        return validate_available(pre_plot=args.pre_plot)
    return validate(pre_plot=args.pre_plot)


if __name__ == "__main__":
    raise SystemExit(main())
