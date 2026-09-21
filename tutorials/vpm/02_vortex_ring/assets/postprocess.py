#!/usr/bin/env python3
"""Check and summarize the vortex-ring instability results.

Strict validation certifies a comparable four-case instability-onset campaign.
``--available`` only checks the sample files consumed by the plots, allowing
figures to be rebuilt while another variant is still running.
Figure checks use PNG by default; pass ``--format pdf`` after PDF exports.
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

if not __package__:
    from openonda.tutorial_runner import case_package
    from pathlib import Path as _CasePath

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"

from ..assets.ring_metrics import (
    FIGURES_DIR,
    SAMPLES_DIR,
    SOLUTION_DIR,
    REFERENCE_TIME,
    REFERENCE_VELOCITY,
    VARIANT_LABEL,
    load_ring_data,
    load_sampled_ring_speed,
    load_sampled_ring_data,
    load_metadata,
    plot_variants,
    saffman_valid_time_limit,
    saffman_speed,
)

VARIANTS = ("dns_direct", "dns_transposed", "dns_mixed", "les_transposed")
ALLOWED_RUN_STATUSES = {"completed", "resolution_lost", "wall_time_limit"}
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


def _run_validation(name: str) -> tuple[dict, set[int], list[str]]:
    """Check one case against the universal solver-owned VPM metadata."""
    failures: list[str] = []
    metadata = load_metadata(name, SAMPLES_DIR)
    if not metadata:
        path = SOLUTION_DIR / name / "vpm_metadata.json"
        return {}, set(), [f"{name}: missing or unreadable {path}"]

    try:
        configuration = metadata["configuration"]
        numerics = configuration["numerics"]
        run = configuration["run"]
        state = metadata["state"]
        requested_steps = int(run["steps"])
        completed_steps = int(state["step"])
        completed_time = float(state["time"])
        interval_steps = int(configuration["backup"]["interval_steps"])
        status = str(metadata["lifecycle"]["status"])
    except (KeyError, TypeError, ValueError) as error:
        return metadata, set(), [f"{name}: invalid run metadata ({error})"]

    run_data = {
        "status": status,
        "requested_steps": requested_steps,
        "completed_steps": completed_steps,
        "final_time": completed_time,
        "backup_interval_steps": interval_steps,
        "initial_n_particles_total": state.get("initial_n_particles_total"),
        "final_n_particles_total": state.get("n_particles_total"),
        "induction_backend": numerics.get("induction", {}).get("method"),
        "stretching_scheme": numerics.get("induction", {}).get("stretching_scheme"),
    }

    try:
        expected_steps = _expected_backup_steps(completed_steps, interval_steps)
    except ValueError as error:
        return metadata, set(), [f"{name}: invalid backup cadence ({error})"]

    expected = {
        "solver": "VPM",
        "case_name": name,
        "schema_version": 1,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            failures.append(f"{name}: {key} is {metadata.get(key)!r}; expected {value!r}")
    expected_scheme = (
        "DIRECT" if name == "dns_direct" else "MIXED" if name == "dns_mixed" else "TRANSPOSED"
    )
    expected_numerics = {
        "integrator": (numerics.get("integrator", {}).get("name"), "SSPRK3"),
        "induction backend": (numerics.get("induction", {}).get("method"), "TREECODE"),
        "stretching scheme": (
            numerics.get("induction", {}).get("stretching_scheme"),
            expected_scheme,
        ),
        "viscous scheme": (numerics.get("viscous", {}).get("scheme"), "CS"),
        "health-limit action": (run.get("health_limit_action"), "STOP"),
    }
    for label, (actual, expected_value) in expected_numerics.items():
        if actual != expected_value:
            failures.append(f"{name}: {label} is {actual!r}; expected {expected_value!r}")
    expected_turbulence = "LES_SMAGORINSKY" if name == "les_transposed" else "DNS"
    turbulence_model = numerics.get("turbulence", {}).get("model")
    if turbulence_model != expected_turbulence:
        failures.append(
            f"{name}: turbulence model is {turbulence_model!r}; expected {expected_turbulence!r}"
        )
    if status not in ALLOWED_RUN_STATUSES:
        failures.append(f"{name}: unsupported run status {status!r}")
    if completed_steps < 0 or requested_steps < 0 or completed_steps > requested_steps:
        failures.append(
            f"{name}: invalid progress completed={completed_steps}, requested={requested_steps}"
        )
    if status == "completed" and completed_steps != requested_steps:
        failures.append(f"{name}: completed status does not reach the requested horizon")
    if status == "resolution_lost" and completed_steps >= requested_steps:
        failures.append(f"{name}: resolution_lost status does not describe an early stop")
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
    return run_data, expected_steps, failures


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


def _figure_failures(figure_format: str = "png") -> list[str]:
    failures = []
    for extension in ("png", "pdf") if figure_format == "both" else (figure_format,):
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


def validate_available(pre_plot: bool, figure_format: str = "png") -> int:
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
        failures.extend(_figure_failures(figure_format))
    if failures:
        print("\n".join(f"[FAIL] {failure}" for failure in failures))
        return 1
    summary = "; ".join(f"{csv_name}={','.join(names)}" for csv_name, names in available.items())
    print(f"[OK] available vortex-ring plot inputs: {summary}")
    return 0


def validate(pre_plot: bool, figure_format: str = "png") -> int:
    """Require a complete, comparable instability-onset campaign."""
    failures: list[str] = []
    outcomes: list[tuple[float, str, str]] = []
    for name in VARIANTS:
        metadata, expected_steps, validation_failures = _run_validation(name)
        failures.extend(validation_failures)
        if validation_failures:
            continue
        outcomes.append((float(metadata["final_time"]), name, str(metadata["status"])))

        all_files = sorted(glob.glob(str(SOLUTION_DIR / name / "vpm" / "vpm_*.h5")))
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
        failures.extend(_figure_failures(figure_format))
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


def build_summary(samples_dir: Path, figures_dir: Path) -> dict:
    """Summarize measured terminal times without writing tutorial metadata."""
    runs = {}
    for variant in plot_variants(samples_dir):
        variant_samples = samples_dir / variant
        raw = load_metadata(variant, samples_dir)
        configuration = raw.get("configuration", {})
        numerics = configuration.get("numerics", {})
        state = raw.get("state", {})
        metadata = {
            "status": raw.get("lifecycle", {}).get("status", "missing"),
            "completed_steps": state.get("step"),
            "requested_steps": configuration.get("run", {}).get("steps"),
            "final_time": state.get("time"),
            "final_n_particles_total": state.get("n_particles_total"),
            "induction_backend": numerics.get("induction", {}).get("method"),
            "stretching_scheme": numerics.get("induction", {}).get("stretching_scheme"),
        }
        if metadata["status"] == "created":
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
                metadata["status"] = "partial"
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
            "stretching_scheme": metadata.get("stretching_scheme"),
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
        "runs": runs,
        "stability_ranking": [name for _, name in ranked],
        "longest_sustained_variant": ranked[0][1]
        if len(ranked) == 1 or (len(ranked) > 1 and ranked[0][0] > ranked[1][0])
        else None,
        "longest_sustained_variants": (
            [name for time, name in ranked if time == ranked[0][0]] if ranked else []
        ),
        "figures": sorted(
            path.name
            for path in figures_dir.glob("*")
            if path.is_file() and path.suffix in (".png", ".pdf")
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-plot", action="store_true", help="skip figure existence checks")
    parser.add_argument(
        "--format",
        choices=("png", "pdf", "both"),
        default="png",
        help="Figure format to check during validation (default: png).",
    )
    parser.add_argument(
        "--available",
        action="store_true",
        help="validate available plotting inputs without requiring a complete campaign",
    )
    args = parser.parse_args()
    if args.available:
        return validate_available(pre_plot=args.pre_plot, figure_format=args.format)
    return validate(pre_plot=args.pre_plot, figure_format=args.format)


if __name__ == "__main__":
    raise SystemExit(main())
