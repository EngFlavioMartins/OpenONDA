"""Numerical-integrity gate for the native cube FVM–VPM workflow."""

from __future__ import annotations

import argparse
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
    nonfinite_counts = [
        record.get("n_nonfinite_values", record.get("nonfinite_count")) for record in diagnostics
    ]
    if any(value is None for value in nonfinite_counts):
        raise SystemExit("FAIL: FVM diagnostics omit the non-finite-value count")
    if any(int(value) != 0 for value in nonfinite_counts):
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


def _check_coupling_coverage(coupling: list[dict], metadata: dict) -> None:
    """Require one ordered diagnostic for every completed coupling step."""
    execution = metadata.get("execution")
    if not isinstance(execution, dict):
        raise SystemExit("FAIL: run metadata omits the execution segment")
    try:
        start_step = int(execution["start_coupling_step"])
        stop_step = int(execution["stop_coupling_step"])
        stop_time = float(execution["stop_time"])
        time_step_size = float(metadata["vpm_time_step_size"])
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit("FAIL: run metadata contains an invalid execution segment") from error
    if start_step < 0 or stop_step <= start_step or time_step_size <= 0.0:
        raise SystemExit("FAIL: run metadata contains an empty or invalid execution segment")

    expected_steps = list(range(start_step + 1, stop_step + 1))
    try:
        actual_steps = [int(record["step"]) for record in coupling]
        actual_times = np.asarray([float(record["time"]) for record in coupling])
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit("FAIL: coupling diagnostics contain invalid step/time fields") from error
    if actual_steps != expected_steps:
        raise SystemExit(
            "FAIL: coupling diagnostics do not cover the complete execution segment "
            f"(expected steps {expected_steps[0]}..{expected_steps[-1]}, got {actual_steps})"
        )

    expected_times = time_step_size * np.asarray(expected_steps, dtype=np.float64)
    tolerance = max(1.0e-12, 64.0 * np.finfo(np.float64).eps * max(abs(stop_time), 1.0))
    if not np.all(np.isfinite(actual_times)) or not np.allclose(
        actual_times,
        expected_times,
        rtol=0.0,
        atol=tolerance,
    ):
        raise SystemExit("FAIL: coupling diagnostic times do not match their coupling steps")
    if not np.isclose(actual_times[-1], stop_time, rtol=0.0, atol=tolerance):
        raise SystemExit(
            "FAIL: coupling diagnostics stop at "
            f"t={actual_times[-1]:g}, expected execution stop t={stop_time:g}"
        )


def _check_coupling_history(
    coupling: list[dict],
    *,
    closure_correction_limit: float,
) -> None:
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
        recovery = record["gbd_moment_recovery"]
        for name in (
            "nonzero_node_count",
            "retained_node_count",
            "pruned_node_count",
            "support_augmented_node_count",
            "correction_fraction",
            "normalized_vortex_strength_residual",
            "normalized_linear_impulse_residual",
            "normalized_angular_impulse_residual",
        ):
            if not np.isfinite(float(recovery[name])):
                raise SystemExit(f"FAIL: non-finite GBD moment-recovery diagnostic {name!r}")
        for name in (
            "nonzero_node_count",
            "retained_node_count",
            "pruned_node_count",
            "support_augmented_node_count",
        ):
            if int(recovery[name]) < 0:
                raise SystemExit(f"FAIL: negative GBD moment-recovery diagnostic {name!r}")
        if int(recovery["pruned_node_count"]) > 0 and not bool(recovery["applied"]):
            raise SystemExit("FAIL: GBD pruned vortex nodes without conservative moment recovery")
        if float(recovery["correction_fraction"]) > closure_correction_limit:
            raise SystemExit(
                "FAIL: GBD pruning required an excessive particle-strength correction "
                f"({float(recovery['correction_fraction']):.3%} > "
                f"{closure_correction_limit:.3%})"
            )
        if (
            max(
                float(recovery["normalized_vortex_strength_residual"]),
                float(recovery["normalized_linear_impulse_residual"]),
                float(recovery["normalized_angular_impulse_residual"]),
            )
            > 1.0e-5
        ):
            raise SystemExit("FAIL: GBD moment recovery exceeds its precision-aware tolerance")
        transfer = record["transfer"]
        if int(transfer["population_pruned_particles"]) != 0:
            raise SystemExit(
                "FAIL: the renewal hit its particle-capacity limit; this run is not a "
                "valid coupling comparison"
            )
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
        if transfer["transfer_method"] != "buffered_m4_renewal":
            continue
        for name in (
            "renewal_raw_vortex_strength_error",
            "renewal_applied_vortex_strength_correction",
            "renewal_conservation_error",
            "renewal_vortex_strength_tolerance",
            "renewal_raw_linear_impulse_error",
            "renewal_applied_linear_impulse_correction",
            "renewal_linear_impulse_error",
            "renewal_linear_impulse_tolerance",
            "renewal_applied_particle_strength_fraction",
        ):
            if name not in transfer:
                raise SystemExit(f"FAIL: renewal closure diagnostic {name!r} is missing")
            if not np.isfinite(float(transfer[name])):
                raise SystemExit(f"FAIL: non-finite renewal closure diagnostic {name!r}")
        if float(transfer["renewal_conservation_error"]) > float(
            transfer["renewal_vortex_strength_tolerance"]
        ):
            raise SystemExit("FAIL: renewal did not close total vortex strength")
        if float(transfer["renewal_linear_impulse_error"]) > float(
            transfer["renewal_linear_impulse_tolerance"]
        ):
            raise SystemExit("FAIL: renewal did not close linear impulse")
        correction_fraction = float(transfer["renewal_applied_particle_strength_fraction"])
        if correction_fraction > closure_correction_limit:
            raise SystemExit(
                "FAIL: renewal closure required an excessive particle-strength correction "
                f"({correction_fraction:.3%} > {closure_correction_limit:.3%})"
            )


def _check_reference_accuracy(
    case_directory: Path,
    reference_directory: Path,
    acceptance_limit: float,
    acceptance_horizon: float,
) -> str:
    """Gate every sampled Cd and each authority-stitched profile's mean error."""
    samples = case_directory / "samples"
    reference_samples = reference_directory / "samples"
    candidate_force = load_table(samples / "forces_history.csv")
    reference_force = load_table(reference_samples / "forces_history.csv")
    measurements: list[tuple[str, float]] = []
    tolerance = max(
        1.0e-12,
        64.0 * np.finfo(np.float64).eps * max(abs(acceptance_horizon), 1.0),
    )

    def require_horizon(table: dict[str, np.ndarray], label: str) -> None:
        times = np.asarray(table["time"], dtype=np.float64)
        if not np.all(np.isfinite(times)):
            raise SystemExit(f"FAIL: {label} contains non-finite sample times")
        if not np.any(np.isclose(times, acceptance_horizon, rtol=0.0, atol=tolerance)):
            raise SystemExit(
                f"FAIL: {label} does not cover the acceptance horizon t={acceptance_horizon:g}"
            )

    require_horizon(candidate_force, "candidate force history")
    require_horizon(reference_force, "reference force history")

    for time, drag in zip(
        candidate_force["time"], candidate_force["drag_coefficient"], strict=True
    ):
        if time > acceptance_horizon + tolerance:
            continue
        if time < reference_force["time"].min() or time > reference_force["time"].max():
            continue
        expected = float(
            np.interp(time, reference_force["time"], reference_force["drag_coefficient"])
        )
        measurements.append((f"Cd@{time:g}", abs(float(drag) - expected) / abs(expected)))

    for name in ("centreline", "offaxis_y075"):
        reference = load_table(reference_samples / f"{name}.csv")
        require_horizon(reference, f"reference {name} profile")
        source_tables = {
            source: load_table(samples / f"{source}_{name}.csv") for source in ("fvm", "vpm")
        }
        for source, candidate in source_tables.items():
            require_horizon(candidate, f"candidate {source} {name} profile")
        for time in np.unique(source_tables["fvm"]["time"]):
            if time > acceptance_horizon + tolerance:
                continue
            reference_frame = frame(reference, float(time))
            fvm_frame = frame(source_tables["fvm"], float(time))
            vpm_frame = frame(source_tables["vpm"], float(time))
            if reference_frame is None or fvm_frame is None or vpm_frame is None:
                continue

            fvm_x = fvm_frame["position_x"]
            finite_fvm_x = fvm_x[np.isfinite(fvm_x)]
            if finite_fvm_x.size == 0:
                raise SystemExit(f"FAIL: candidate FVM {name} profile has no finite positions")
            fvm_min = float(finite_fvm_x.min())
            fvm_max = float(finite_fvm_x.max())
            vpm_x = vpm_frame["position_x"]
            vpm_authority = (vpm_x < fvm_min) | (vpm_x > fvm_max)
            stitched = {
                "position_x": np.concatenate((fvm_x, vpm_x[vpm_authority])),
                "velocity_x": np.concatenate(
                    (fvm_frame["velocity_x"], vpm_frame["velocity_x"][vpm_authority])
                ),
            }
            record = profile_record(
                "authority_stitched",
                name,
                float(time),
                stitched,
                reference_frame,
            )
            measurements.append(
                (
                    f"mean-authority-stitched-{name}@{time:g}",
                    float(record["mean_abs_over_u_inf"]),
                )
            )

    if not measurements:
        raise SystemExit("FAIL: no coincident reference metrics were available")
    failed = [(name, value) for name, value in measurements if value > acceptance_limit]
    if failed:
        detail = ", ".join(f"{name}={value:.2%}" for name, value in failed[:8])
        raise SystemExit(f"FAIL: reference errors exceed {acceptance_limit:.1%}: {detail}")
    worst_name, worst_value = max(measurements, key=lambda item: item[1])
    return (
        f" {len(measurements)} reference metrics checked through t={acceptance_horizon:g};"
        f" worst reference error {worst_name}={worst_value:.2%}"
    )


def _resolve_acceptance_horizon(stop_time: float, requested: float | None) -> float:
    """Return the explicit physics-gate horizon for this execution segment."""
    stop = float(stop_time)
    horizon = min(stop, 2.0) if requested is None else float(requested)
    if not np.isfinite(stop) or stop <= 0.0:
        raise ValueError("execution stop time must be finite and positive")
    if not np.isfinite(horizon) or horizon <= 0.0 or horizon > stop + 1.0e-10:
        raise ValueError("acceptance horizon must lie in (0, execution stop time]")
    return horizon


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-directory", type=Path, default=CASE_DIR)
    parser.add_argument(
        "--reference-directory",
        type=Path,
        default=CASE_DIR / "reference_flow",
    )
    parser.add_argument("--acceptance-limit", type=float, default=0.05)
    parser.add_argument(
        "--acceptance-horizon",
        type=float,
        help="last physical time that must have coincident candidate/reference metrics",
    )
    arguments = parser.parse_args()
    if not 0.0 < arguments.acceptance_limit < 1.0:
        raise ValueError("acceptance limit must lie strictly between zero and one")
    case_directory = arguments.case_directory.resolve()
    reference_directory = arguments.reference_directory.resolve()

    metadata_path = case_directory / "solution" / "run_metadata.json"
    force_path = case_directory / "samples" / "forces_history.csv"
    if not metadata_path.is_file():
        raise SystemExit("FAIL: coupled run metadata was not written")
    metadata = json.loads(metadata_path.read_text())
    expected_end = float(
        metadata.get("execution", {}).get("stop_time", metadata["physics"]["end_time"])
    )
    acceptance_horizon = _resolve_acceptance_horizon(
        expected_end,
        arguments.acceptance_horizon,
    )

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

    diagnostics = _json_lines(case_directory / "solution" / "diagnostics.jsonl")
    max_courant_number, max_continuity = _check_solver_history(diagnostics)

    coupling = _json_lines(case_directory / "solution" / "coupler_diagnostics.jsonl")
    _check_coupling_coverage(coupling, metadata)
    _check_coupling_history(
        coupling,
        closure_correction_limit=float(metadata["coupler"]["transfer_discretization_error_limit"]),
    )

    physics_summary = _check_reference_accuracy(
        case_directory,
        reference_directory,
        arguments.acceptance_limit,
        acceptance_horizon,
    )

    print(
        "PASS: native cube run completed with converged FVM solves, "
        f"peak max_courant_number={max_courant_number:.3g}, "
        f"peak continuity={max_continuity:.3g}, "
        f"and solenoidal local transfer through t={forces[-1, 0]:g}.{physics_summary}"
    )


if __name__ == "__main__":
    main()
