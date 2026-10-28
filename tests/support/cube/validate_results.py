"""Numerical-integrity check for the native cube FVM–VPM workflow."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

CASE_DIR = Path(__file__).resolve().parents[3] / "tutorials/coupled_fvm_vpm/02_cube_flow"


def load_table(path: Path) -> dict[str, np.ndarray]:
    """Load one numerical CSV table and retain only its latest time sequence."""
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty table: {path}")
    numeric = {
        name: np.asarray([float(row[name]) for row in rows]) for name in rows[0] if name != "patch"
    }
    reset = np.flatnonzero(np.diff(numeric["time"]) < 0.0)
    if reset.size:
        start = int(reset[-1] + 1)
        numeric = {name: values[start:] for name, values in numeric.items()}
    return numeric


def frame(table: dict[str, np.ndarray], time: float) -> dict[str, np.ndarray] | None:
    """Return a table frame at an exactly recorded physical time."""
    available = np.unique(table["time"])
    picked = available[np.argmin(np.abs(available - time))]
    if not np.isclose(picked, time, rtol=0.0, atol=1.0e-10):
        return None
    selected = np.isclose(table["time"], picked, rtol=0.0, atol=1.0e-12)
    result = {name: values[selected] for name, values in table.items()}
    if "position_x" in result:
        order = np.argsort(result["position_x"])
        result = {name: values[order] for name, values in result.items()}
    return result


def profile_record(
    source: str,
    name: str,
    time: float,
    candidate: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
) -> dict[str, float | str]:
    """Measure one streamwise-velocity profile against the reference profile."""
    x = candidate["position_x"]
    velocity = candidate["velocity_x"]
    reference_x = reference["position_x"]
    reference_velocity = reference["velocity_x"]
    finite_reference = np.isfinite(reference_x) & np.isfinite(reference_velocity)
    reference_x = reference_x[finite_reference]
    reference_velocity = reference_velocity[finite_reference]
    valid = (
        np.isfinite(x) & np.isfinite(velocity) & (x >= reference_x.min()) & (x <= reference_x.max())
    )
    if name == "centreline":
        valid &= (x < -0.5) | (x > 0.5)
    error = np.abs(velocity[valid] - np.interp(x[valid], reference_x, reference_velocity))
    return {
        "metric": "profile",
        "source": source,
        "profile": name,
        "time": time,
        "mean_abs_over_u_inf": float(error.mean()),
        "p95_abs_over_u_inf": float(np.quantile(error, 0.95)),
        "max_abs_over_u_inf": float(error.max()),
    }


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
    moment_correction_limit: float,
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
        correction = record["gbd_moment_correction"]
        for name in (
            "nonzero_node_count",
            "retained_node_count",
            "removed_node_count",
            "support_augmented_node_count",
            "correction_fraction",
            "normalized_vortex_strength_residual",
            "normalized_linear_impulse_residual",
            "normalized_angular_impulse_residual",
        ):
            if not np.isfinite(float(correction[name])):
                raise SystemExit(f"FAIL: non-finite GBD moment-correction diagnostic {name!r}")
        for name in (
            "nonzero_node_count",
            "retained_node_count",
            "removed_node_count",
            "support_augmented_node_count",
        ):
            if int(correction[name]) < 0:
                raise SystemExit(f"FAIL: negative GBD moment-correction diagnostic {name!r}")
        if int(correction["removed_node_count"]) > 0 and not bool(correction["applied"]):
            raise SystemExit("FAIL: GBD removed weak strengths without the moment correction")
        if float(correction["correction_fraction"]) > moment_correction_limit:
            raise SystemExit(
                "FAIL: the GBD weak-strength removal needed a moment correction above its limit "
                f"({float(correction['correction_fraction']):.3%} > "
                f"{moment_correction_limit:.3%})"
            )
        if (
            max(
                float(correction["normalized_vortex_strength_residual"]),
                float(correction["normalized_linear_impulse_residual"]),
                float(correction["normalized_angular_impulse_residual"]),
            )
            > 1.0e-5
        ):
            raise SystemExit(
                "FAIL: the GBD moment correction exceeds its storage-precision tolerance"
            )
        transfer = record["transfer"]
        if int(transfer["particle_set_removed_particles"]) != 0:
            raise SystemExit(
                "FAIL: the vorticity transfer hit the particle capacity; this run is not a "
                "valid coupling comparison"
            )
        expected_after = (
            int(transfer["n_particles_before"])
            - int(transfer["n_particles_removed"])
            + int(transfer["n_particles_injected"])
        )
        if int(transfer["n_particles_after"]) != expected_after:
            raise SystemExit("FAIL: inconsistent particle count of the vorticity transfer")
        strength_change = [
            float(transfer[f"state_change_vortex_strength_net_{axis}"]) for axis in "xyz"
        ]
        if not np.all(np.isfinite(strength_change)):
            raise SystemExit("FAIL: non-finite total-strength change of the vorticity transfer")
        for name in (
            "transfer_raw_vortex_strength_error",
            "transfer_applied_vortex_strength_correction",
            "transfer_conservation_error",
            "transfer_vortex_strength_tolerance",
            "transfer_raw_linear_impulse_error",
            "transfer_applied_linear_impulse_correction",
            "transfer_linear_impulse_error",
            "transfer_linear_impulse_tolerance",
            "transfer_applied_particle_strength_fraction",
        ):
            if name not in transfer:
                raise SystemExit(f"FAIL: moment-correction diagnostic {name!r} is missing")
            if not np.isfinite(float(transfer[name])):
                raise SystemExit(f"FAIL: non-finite moment-correction diagnostic {name!r}")
        if float(transfer["transfer_conservation_error"]) > float(
            transfer["transfer_vortex_strength_tolerance"]
        ):
            raise SystemExit("FAIL: the vorticity transfer did not conserve the total strength")
        if float(transfer["transfer_linear_impulse_error"]) > float(
            transfer["transfer_linear_impulse_tolerance"]
        ):
            raise SystemExit("FAIL: the vorticity transfer did not conserve the linear impulse")
        correction_fraction = float(transfer["transfer_applied_particle_strength_fraction"])
        if correction_fraction > moment_correction_limit:
            raise SystemExit(
                "FAIL: the vorticity transfer needed a moment correction above its limit "
                f"({correction_fraction:.3%} > {moment_correction_limit:.3%})"
            )


def _check_reference_accuracy(
    case_directory: Path,
    reference_directory: Path,
    acceptance_limit: float,
    acceptance_horizon: float,
) -> str:
    """Check every sampled Cd and each blend_weight-stitched profile's mean error."""
    samples = case_directory / "samples"
    reference_samples = reference_directory / "samples"
    if (reference_samples / "fine").is_dir():
        reference_samples /= "fine"
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
            outside_fvm_mask = (vpm_x < fvm_min) | (vpm_x > fvm_max)
            stitched = {
                "position_x": np.concatenate((fvm_x, vpm_x[outside_fvm_mask])),
                "velocity_x": np.concatenate(
                    (fvm_frame["velocity_x"], vpm_frame["velocity_x"][outside_fvm_mask])
                ),
            }
            record = profile_record(
                "combined_fvm_vpm",
                name,
                float(time),
                stitched,
                reference_frame,
            )
            measurements.append(
                (
                    f"mean-blend_weight-stitched-{name}@{time:g}",
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
    """Return the explicit physics-check horizon for this execution segment."""
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
        moment_correction_limit=float(metadata["coupler"]["max_moment_correction_fraction"]),
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
