"""Scientific checks for the Lamb--Oseen physical diagnostics."""

from pathlib import Path
import re

import numpy as np
import pandas as pd

from tests._tutorial_helpers import load_tutorial_module

_physics = load_tutorial_module("vpm/lamb_oseen_vortex", "assets.postprocess")
BETA_RMAX = _physics.BETA_RMAX
CASES = _physics.CASES
COMPUTE_METHOD = {"CS": "DIRECT", "RWM": "DIRECT", "DVH": "TREECODE", "GBD": "TREECODE"}
CORE_RADIUS = _physics.CORE_RADIUS
TIME_STEP_SIZE = 0.291 / 9.0
CONFIGURED_TOTAL_TIME = 103.0 * 0.291
EXPECTED_DT = TIME_STEP_SIZE
EXPECTED_END_TIME = CONFIGURED_TOTAL_TIME
FIGURES_DIR = _physics.FIGURES_DIR
GAUSSIAN_CORE_RADIUS = _physics.GAUSSIAN_CORE_RADIUS
INITIAL_PARTICLE_COUNT = {"vortex": 2077, "dipole": 3618, "merging": 3618}
MERGING_NORMALIZED_END_TIME = _physics.MERGING_NORMALIZED_END_TIME
MINIMUM_ENSEMBLE_SIZE = 4
REFERENCE_FINAL_TIME_SECONDS = _physics.REFERENCE_FINAL_TIME_SECONDS
REFERENCE_FINAL_VISCOUS_TIME = _physics.REFERENCE_FINAL_VISCOUS_TIME
REFERENCE_VISCOUS_TIME_PER_SECOND = _physics.REFERENCE_VISCOUS_TIME_PER_SECOND
REYNOLDS_NUMBER = _physics.REYNOLDS_NUMBER
RWM_RELATIVE_STANDARD_ERROR_LIMIT = 0.075
SAMPLES_DIR = _physics.SAMPLES_DIR
SCHEMES = _physics.SCHEMES
SEPARATION = _physics.SEPARATION
SOLUTION_DIR = _physics.SOLUTION_DIR
SPACING = _physics.SPACING


def _metadata(path):
    try:
        return _physics._metadata(path)
    except FileNotFoundError:
        return {}


_reportable_energy_rate_mask = _physics._reportable_energy_rate_mask
extract_merging_timeseries = _physics.extract_merging_timeseries
lamb_oseen_gradient = _physics.lamb_oseen_gradient
lamb_oseen_profile = _physics.lamb_oseen_profile
load_merging_references = _physics.load_merging_references
load_profile = _physics.load_profile
pvd_time_map = _physics.pvd_time_map
resolve_runtime_physics = _physics.resolve_runtime_physics
theoretical_dipole_trajectory = _physics.theoretical_dipole_trajectory
uniform_cadence_mask = _physics.uniform_cadence_mask


def _read_csv(
    path: Path,
    failures: list[str],
    *,
    allow_nonfinite: bool = False,
) -> pd.DataFrame | None:
    if not path.is_file():
        failures.append(f"missing {path}")
        return None
    try:
        data = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        failures.append(f"unreadable {path}: {error}")
        return None
    numeric = data.select_dtypes(include=[np.number])
    if (
        data.empty
        or numeric.empty
        or (not allow_nonfinite and not np.isfinite(numeric.to_numpy()).all())
    ):
        failures.append(f"{path}: empty or non-finite numeric data")
    return data


def merging_normalization_audit(
    samples_dir: Path = SAMPLES_DIR,
    schemes: tuple[str, ...] = SCHEMES,
) -> tuple[dict, list[str]]:
    """Audit the paper-to-simulation mapping and requested merger horizon."""
    failures: list[str] = []
    reference = load_merging_references(CORE_RADIUS, SEPARATION)
    reference_horizons = {name: float(values[:, 0].max()) for name, values in reference.items()}
    for name, horizon in reference_horizons.items():
        if not 2.9 <= horizon <= 3.2:
            failures.append(f"merging reference {name}: normalized horizon {horizon:.6g} is not 3")

    run_report = {}
    for scheme in schemes:
        folder = Path(samples_dir) / f"merging_{scheme}"
        metadata = _metadata(folder)
        if not metadata:
            failures.append(f"merging_{scheme}: missing metadata for normalization audit")
            continue
        circulation = abs(float(metadata.get("circulations", [np.nan])[0]))
        viscosity = float(metadata.get("kinematic_viscosity", np.nan))
        a_c0 = float(metadata.get("velocity_peak_radius", np.nan))
        b0 = float(metadata.get("vortex_separation", np.nan))
        reynolds = circulation / viscosity
        core_ratio = a_c0 / b0
        physical_time_at_three = MERGING_NORMALIZED_END_TIME * a_c0**2 / viscosity
        if not np.isclose(reynolds, REYNOLDS_NUMBER, rtol=1.0e-10):
            failures.append(f"merging_{scheme}: Re_Gamma={reynolds:.9g}, expected 530")
        if not np.isclose(core_ratio, CORE_RADIUS / SEPARATION, rtol=1.0e-10):
            failures.append(f"merging_{scheme}: a_c0/b0={core_ratio:.9g}, expected 0.125")
        try:
            fields = pd.read_csv(folder / "field_diagnostics.csv")
            normalized_time = viscosity * fields["time"].to_numpy(float) / a_c0**2
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
            failures.append(f"merging_{scheme}: normalization data unreadable ({exc})")
            continue
        final_normalized_time = float(np.nanmax(normalized_time))
        if final_normalized_time < MERGING_NORMALIZED_END_TIME:
            failures.append(f"merging_{scheme}: normalized horizon {final_normalized_time:.6g} < 3")
        first_at_or_after = np.flatnonzero(normalized_time >= MERGING_NORMALIZED_END_TIME)
        endpoint = (
            float(normalized_time[first_at_or_after[0]])
            if first_at_or_after.size
            else final_normalized_time
        )
        coalesced = fields.get("is_peak_coalesced")
        if coalesced is not None:
            coalesced = coalesced.astype(str).str.lower().isin(("true", "1")).to_numpy(bool)
            if coalesced.any():
                separation = fields["vortex_separation"].to_numpy(float)
                if not np.allclose(separation[coalesced], 0.0, rtol=0.0, atol=1.0e-12):
                    failures.append(f"merging_{scheme}: b is not zero after peak coalescence")
        run_report[scheme] = {
            "circulation_reynolds_number": reynolds,
            "initial_velocity_core_to_separation_ratio": core_ratio,
            "time_scale_seconds": a_c0**2 / viscosity,
            "physical_time_at_normalized_3": physical_time_at_three,
            "available_normalized_horizon": final_normalized_time,
            "first_sample_at_or_after_3": endpoint,
        }
        timeseries = extract_merging_timeseries(Path(samples_dir), scheme, viscosity, b0, a_c0)
        agreement = {}
        if timeseries is not None:
            unresolved = np.asarray(timeseries["is_pair_unresolved"], dtype=bool)
            first_unresolved = np.flatnonzero(unresolved)
            separation_history = np.asarray(timeseries["b_over_b0"], dtype=float)
            first_zero = np.flatnonzero(
                np.isfinite(separation_history)
                & np.isclose(separation_history, 0.0, rtol=0.0, atol=1.0e-12)
            )
            finite_theta = np.flatnonzero(np.isfinite(timeseries["theta_deg"]))
            run_report[scheme]["feature_transition"] = {
                "first_statistically_unresolved_pair_time": (
                    float(timeseries["tau"][first_unresolved[0]]) if first_unresolved.size else None
                ),
                "first_zero_peak_separation_time": (
                    float(timeseries["tau"][first_zero[0]]) if first_zero.size else None
                ),
                "last_reportable_theta_time": (
                    float(timeseries["tau"][finite_theta[-1]]) if finite_theta.size else None
                ),
                "theta_definition": (
                    "axis joining the two vorticity centres before coalescence; "
                    "major-axis orientation of the merged elliptic vortex thereafter"
                ),
            }
            for feature, reference_name in (
                ("theta_deg", "theta"),
                ("a_c2_over_b02", "core"),
                ("b_over_b0", "separation"),
            ):
                if reference_name not in reference:
                    continue
                reference_values = reference[reference_name]
                x = timeseries["tau"]
                y = timeseries[feature]
                comparable = (
                    np.isfinite(x)
                    & np.isfinite(y)
                    & (x >= reference_values[:, 0].min())
                    & (x <= reference_values[:, 0].max())
                )
                if not comparable.any():
                    continue
                expected = np.interp(x[comparable], reference_values[:, 0], reference_values[:, 1])
                residual = y[comparable] - expected
                value_range = float(np.ptp(reference_values[:, 1]))
                rmse = float(np.sqrt(np.mean(residual * residual)))
                normalized_rmse = rmse / value_range
                agreement[feature] = {
                    "comparable_samples": int(np.count_nonzero(comparable)),
                    "rmse": rmse,
                    "rmse_over_reference_range": normalized_rmse,
                    "mean_bias": float(np.mean(residual)),
                    "assessment": (
                        "good"
                        if normalized_rmse <= 0.10
                        else "moderate"
                        if normalized_rmse <= 0.25
                        else "poor"
                    ),
                }
        run_report[scheme]["reference_agreement"] = agreement
    report = {
        "displayed_time": "nu*t/a_c0^2",
        "paper_time": "nu*t/b0^2",
        "paper_to_display_factor": (SEPARATION / CORE_RADIUS) ** 2,
        "reference_normalized_horizons": reference_horizons,
        "separation_reference_status": "plotted from the figure 4 b/b0 measurements",
        "separation_time_conversion": {
            "figure_4_final_time_seconds": REFERENCE_FINAL_TIME_SECONDS,
            "figure_5_final_viscous_time": REFERENCE_FINAL_VISCOUS_TIME,
            "viscous_time_per_second": REFERENCE_VISCOUS_TIME_PER_SECOND,
            "basis": ("common final acquisition of the same Re=530 experiment and timescale"),
        },
        "runs": run_report,
    }
    return report, failures


def single_vortex_error_audit(
    samples_dir: Path = SAMPLES_DIR,
    schemes: tuple[str, ...] = SCHEMES,
) -> dict:
    """Profile and core-growth errors against the exact Lamb--Oseen solution."""
    timelines = {scheme: pvd_time_map(samples_dir, "vortex", scheme) for scheme in schemes}
    latest = min(max(values.values()) for values in timelines.values() if values)
    runtime = resolve_runtime_physics(samples_dir)
    output: dict[str, dict] = {}
    for scheme in schemes:
        profile = load_profile(samples_dir, scheme, latest)
        if profile is None:
            raise ValueError(f"vortex_{scheme}: no readable common-time field profile")
        x, velocity, vorticity, selected_time = profile
        exact_velocity, exact_vorticity, _ = lamb_oseen_profile(
            x,
            runtime["t0"] + selected_time,
            runtime["circulation"],
            runtime["kinematic_viscosity"],
        )
        exact_gradient = lamb_oseen_gradient(
            x,
            runtime["t0"] + selected_time,
            runtime["circulation"],
            runtime["kinematic_viscosity"],
        )
        numerical_gradient = np.gradient(velocity, x)
        window = np.abs(x / runtime["velocity_peak_radius0"]) <= 5.5
        errors = [
            float(np.linalg.norm((numerical - exact)[window]) / np.linalg.norm(exact[window]))
            for numerical, exact in (
                (velocity, exact_velocity),
                (vorticity, exact_vorticity),
                (numerical_gradient, exact_gradient),
            )
        ]

        field_path = samples_dir / f"vortex_{scheme}" / "field_diagnostics.csv"
        fields = pd.read_csv(field_path, on_bad_lines="skip").dropna(subset=["time"])
        fields = fields.sort_values("time")
        fields = fields[fields["time"] <= EXPECTED_END_TIME + EXPECTED_DT]
        time = fields["time"].to_numpy(float)
        measured_core = fields["mean_core_radius"].to_numpy(float)
        exact_core = BETA_RMAX * np.sqrt(
            GAUSSIAN_CORE_RADIUS**2 + 4.0 * runtime["kinematic_viscosity"] * time
        )
        relative_core_error = (measured_core - exact_core) / exact_core
        finite = np.isfinite(relative_core_error)
        if not finite.any():
            raise ValueError(f"vortex_{scheme}: no finite core-radius history")
        output[scheme] = {
            "profile_time": float(selected_time),
            "relative_l2_velocity": errors[0],
            "relative_l2_vorticity": errors[1],
            "relative_l2_velocity_gradient": errors[2],
            "core_radius_comparable_samples": int(finite.sum()),
            "core_radius_relative_rmse": float(np.sqrt(np.mean(relative_core_error[finite] ** 2))),
            "core_radius_relative_mean_bias": float(np.mean(relative_core_error[finite])),
            "core_radius_relative_maximum_absolute_error": float(
                np.max(np.abs(relative_core_error[finite]))
            ),
            "core_radius_final_relative_error": float(relative_core_error[finite][-1]),
        }
    return {
        "common_profile_target_time": float(latest),
        "profile_window": "abs(x/a_c0) <= 5.5",
        "runs": output,
    }


def dipole_error_audit(
    samples_dir: Path = SAMPLES_DIR,
    schemes: tuple[str, ...] = SCHEMES,
) -> dict:
    """Compare translating-pair motion on one common physical-time window."""
    runtime = resolve_runtime_physics(samples_dir, prefix="dipole")
    frames = {}
    for scheme in schemes:
        path = samples_dir / f"dipole_{scheme}" / "field_diagnostics.csv"
        frame = pd.read_csv(path, on_bad_lines="skip").dropna(subset=["time"])
        frame = frame.sort_values("time").drop_duplicates("time", keep="last")
        if frame.empty:
            raise ValueError(f"dipole_{scheme}: no field diagnostics")
        frame = frame[uniform_cadence_mask(frame["step"].to_numpy(int))]
        frames[scheme] = frame
    common_end = min(
        EXPECTED_END_TIME,
        *(float(frame["time"].max()) for frame in frames.values()),
    )
    exact_core_end = BETA_RMAX * np.sqrt(
        GAUSSIAN_CORE_RADIUS**2 + 4.0 * runtime["kinematic_viscosity"] * common_end
    )
    reference_end = theoretical_dipole_trajectory(
        np.asarray([common_end]),
        runtime["circulation"],
        runtime["vortex_separation"],
        runtime["kinematic_viscosity"],
        runtime["t0"],
        runtime["column_length"],
    )[0]
    runs = {}
    for scheme, frame in frames.items():
        time = frame["time"].to_numpy(float)
        trajectory = frame["vortex_centre_0_x"].to_numpy(float)
        core_radius = frame["mean_core_radius"].to_numpy(float)
        separation = frame["vortex_separation"].to_numpy(float)
        comparable = (time <= common_end + 1.0e-12) & np.isfinite(trajectory)
        reference = theoretical_dipole_trajectory(
            time[comparable],
            runtime["circulation"],
            runtime["vortex_separation"],
            runtime["kinematic_viscosity"],
            runtime["t0"],
            runtime["column_length"],
        )
        reference_range = float(np.ptp(reference))
        trajectory_nrmse = float(
            np.sqrt(np.mean((trajectory[comparable] - reference) ** 2)) / reference_range
        )
        end_trajectory = float(np.interp(common_end, time, trajectory))
        end_core = float(np.interp(common_end, time, core_radius))
        end_separation = float(np.interp(common_end, time, separation))
        runs[scheme] = {
            "comparable_samples": int(comparable.sum()),
            "trajectory_rmse_over_reference_range": trajectory_nrmse,
            "end_trajectory_over_a_c0": end_trajectory / runtime["velocity_peak_radius0"],
            "end_trajectory_relative_error": (end_trajectory - reference_end) / reference_end,
            "end_separation_over_b0": end_separation / runtime["vortex_separation"],
            "end_core_radius_over_a_c0": end_core / runtime["velocity_peak_radius0"],
            "end_core_radius_relative_to_isolated_exact": (end_core - exact_core_end)
            / exact_core_end,
        }
    return {
        "common_end_time": float(common_end),
        "reference": "fixed-separation finite Lamb--Oseen filaments",
        "runs": runs,
    }


def energy_balance_audit(
    samples_dir: Path = SAMPLES_DIR,
    schemes: tuple[str, ...] = SCHEMES,
) -> dict:
    """Check every finite energy-rate sample and its recorded source."""
    runs = {}
    for case_id in CASES:
        for scheme in schemes:
            name = f"{case_id}_{scheme}"
            path = samples_dir / name / "flow_integrals.csv"
            try:
                frame = pd.read_csv(path, on_bad_lines="skip").dropna(subset=["time"])
            except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
                runs[name] = {
                    "status": "missing" if not path.is_file() else "unreadable",
                    "reason": str(exc),
                    "total_rate_samples": 0,
                    "comparable_samples": 0,
                    "raw_positive_samples": 0,
                    "comparable_positive_samples": 0,
                    "all_positive_samples_outside_reportable_definition": False,
                    "relative_rms_balance_residual": None,
                    "comparable_end_time": None,
                }
                continue
            required = {
                "time",
                "kinetic_energy_rate",
                "viscous_kinetic_energy_rate",
            }
            if not required.issubset(frame.columns):
                runs[name] = {
                    "status": "unreadable",
                    "reason": f"missing columns {sorted(required - set(frame.columns))}",
                    "total_rate_samples": 0,
                    "comparable_samples": 0,
                    "raw_positive_samples": 0,
                    "comparable_positive_samples": 0,
                    "all_positive_samples_outside_reportable_definition": False,
                    "relative_rms_balance_residual": None,
                    "comparable_end_time": None,
                }
                continue
            reportable = _reportable_energy_rate_mask(frame)
            energy_rate = frame["kinetic_energy_rate"].to_numpy(float)
            viscous_rate = frame["viscous_kinetic_energy_rate"].to_numpy(float)
            nonzero = np.isfinite(energy_rate) & (energy_rate != 0.0)
            comparable = reportable & nonzero & np.isfinite(viscous_rate)
            raw_positive = nonzero & (energy_rate > 1.0e-7)
            comparable_positive = comparable & (energy_rate > 1.0e-7)
            residual = energy_rate[comparable] - viscous_rate[comparable]
            denominator = (
                float(np.sqrt(np.mean(viscous_rate[comparable] ** 2))) if comparable.any() else 0.0
            )
            runs[name] = {
                "status": "available",
                "total_rate_samples": int(nonzero.sum()),
                "comparable_samples": int(comparable.sum()),
                "raw_positive_samples": int(raw_positive.sum()),
                "comparable_positive_samples": int(comparable_positive.sum()),
                "all_positive_samples_outside_reportable_definition": bool(
                    raw_positive.any() and not np.any(raw_positive & reportable)
                ),
                "relative_rms_balance_residual": (
                    float(np.sqrt(np.mean(residual**2)) / denominator)
                    if comparable.any() and denominator > 0.0
                    else None
                ),
                "comparable_end_time": (
                    float(frame.loc[comparable, "time"].max()) if comparable.any() else None
                ),
            }
    return {
        "finite_difference_definition": (
            "backward difference of consecutive unbounded kinetic-energy integrals; "
            "DVH output intervals include at least one resolved heat transfer"
        ),
        "large_cloud_mode": (
            "uniform-core, uniform-viscosity clouds use zero-padded linear correlations "
            "with the unbounded transverse Gaussian Green tensor"
        ),
        "runs": runs,
    }


def _solver_log_record(path: Path) -> dict | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    cumulative = [
        float(value)
        for value in re.findall(
            r"^\s*cumulative\s+([0-9.]+(?:e[+-]?\d+)?)\s+s\s*$",
            text,
            flags=re.MULTILINE | re.IGNORECASE,
        )
    ]
    if not cumulative:
        wall_clock = re.findall(
            r"WALL TIME\s+(\d+):(\d+):([0-9.]+)",
            text,
            flags=re.IGNORECASE,
        )
        cumulative = [
            3600.0 * int(hours) + 60.0 * int(minutes) + float(seconds)
            for hours, minutes, seconds in wall_clock
        ]
    if not cumulative:
        return None

    def first(pattern: str) -> str | None:
        match = re.search(pattern, text, flags=re.MULTILINE)
        return match.group(1).strip() if match else None

    return {
        "solver_cumulative_seconds": cumulative[-1],
        "backend": first(r"^\s*backend\s+(\S+)\s*$"),
        "host": first(r"^\s*host\s+(.+?)\s*$"),
        "platform": first(r"^\s*platform\s+(.+?)\s*$"),
    }


def runtime_audit(solution_dir: Path = SOLUTION_DIR) -> dict:
    """Summarize recorded solver time without conflating different hardware."""
    runs = {}
    for case_id in CASES:
        for scheme in ("cs", "gbd", "dvh"):
            name = f"{case_id}_{scheme}"
            record = _solver_log_record(solution_dir / name / "vpm.log")
            if record is not None:
                runs[name] = record

        name = f"{case_id}_rwm"
        member_records = [
            record
            for path in sorted(solution_dir.glob(f"{name}_[0-9][0-9][0-9]/vpm.log"))
            if (record := _solver_log_record(path)) is not None
        ]
        if member_records:
            seconds = np.asarray(
                [record["solver_cumulative_seconds"] for record in member_records],
                dtype=float,
            )
            runs[name] = {
                "ensemble_members": int(seconds.size),
                "ensemble_total_solver_seconds": float(seconds.sum()),
                "member_mean_solver_seconds": float(seconds.mean()),
                "member_median_solver_seconds": float(np.median(seconds)),
                "member_minimum_solver_seconds": float(seconds.min()),
                "member_maximum_solver_seconds": float(seconds.max()),
                "backend": member_records[0]["backend"],
                "host": member_records[0]["host"],
                "platform": member_records[0]["platform"],
            }
    environments = {(record.get("backend"), record.get("host")) for record in runs.values()}
    return {
        "definition": (
            "last cumulative solver-step time recorded in each solver log; "
            "external post-processing is excluded"
        ),
        "cross_scheme_wall_time_comparable": len(environments) == 1,
        "comparison_note": (
            "Wall times are comparable only when every recorded run has the same backend and host."
        ),
        "runs": runs,
    }


def _gbd_moment_recovery_failures(cases: tuple[str, ...]) -> list[str]:
    """Verify that every GBD log records successful pruning closure."""
    failures: list[str] = []
    residual_limit = 1.0e-5
    for physics_id in cases:
        name = f"{physics_id}_gbd"
        path = SOLUTION_DIR / name / "vpm.log"
        try:
            log = path.read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            failures.append(f"{name}: missing GBD recovery log ({error})")
            continue
        try:
            residuals = np.asarray(
                [float(value) for value in re.findall(r"net residual, after\s*\|?\s*(\S+)", log)],
                dtype=float,
            )
        except ValueError:
            failures.append(f"{name}: malformed GBD recovery residual")
            continue
        if residuals.size == 0:
            failures.append(f"{name}: no GBD moment-recovery closure was recorded")
        elif not np.isfinite(residuals).all():
            failures.append(f"{name}: non-finite GBD recovery residual")
        elif float(residuals.max()) > residual_limit:
            failures.append(
                f"{name}: GBD recovery residual exceeds {residual_limit:.1e} "
                f"(maximum {float(residuals.max()):.3e})"
            )
    return failures


def validate(
    pre_plot: bool,
    schemes: tuple[str, ...] = SCHEMES,
    cases: tuple[str, ...] = CASES,
    figure_format: str = "png",
) -> int:
    failures: list[str] = []
    for physics_id in cases:
        for scheme in schemes:
            name = f"{physics_id}_{scheme}"
            folder = SAMPLES_DIR / name
            metadata = _metadata(folder)
            if not metadata:
                failures.append(f"{name}: missing solver-owned VPM metadata")
                continue
            if metadata.get("status") != "completed":
                failures.append(f"{name}: metadata is not complete")
            expected_backend = COMPUTE_METHOD[scheme.upper()]
            if metadata.get("induction_backend") != expected_backend:
                failures.append(
                    f"{name}: induction backend {metadata.get('induction_backend')!r}, "
                    f"expected {expected_backend}"
                )
            if metadata.get("integrator") != "RK2":
                failures.append(f"{name}: integrator is not RK2")
            if not np.isclose(
                float(metadata.get("particle_spacing", np.nan)),
                SPACING,
                rtol=0.0,
                atol=np.finfo(float).eps,
            ):
                failures.append(f"{name}: particle spacing does not match the case setup")
            expected_initial = INITIAL_PARTICLE_COUNT[physics_id]
            if int(metadata.get("initial_n_particles_total", -1)) != expected_initial:
                failures.append(f"{name}: initial particle count is not {expected_initial}")
            final_time = float(metadata.get("final_time", np.nan))
            if final_time < EXPECTED_END_TIME - EXPECTED_DT:
                failures.append(
                    f"{name}: final time {final_time:.9g} does not cover {EXPECTED_END_TIME:.9g}"
                )

            integrals = _read_csv(folder / "flow_integrals.csv", failures)
            if integrals is not None:
                if "time" not in integrals:
                    failures.append(f"{name}: flow-integral history has no time column")
                    continue
                integral_time = integrals["time"].to_numpy(float)
                positive_cadence = np.diff(integral_time)
                positive_cadence = positive_cadence[positive_cadence > 0.0]
                cadence = (
                    float(np.median(positive_cadence)) if positive_cadence.size else EXPECTED_DT
                )
                if integrals["time"].iloc[-1] < EXPECTED_END_TIME - cadence - EXPECTED_DT:
                    failures.append(f"{name}: flow-integral history is incomplete")
                if np.any(np.diff(integral_time) <= 0.0):
                    failures.append(f"{name}: flow-integral time is not strictly increasing")
                for column in (
                    "total_kinetic_energy",
                    "kinetic_energy_rate",
                    "viscous_kinetic_energy_rate",
                ):
                    if column not in integrals:
                        failures.append(f"{name}: flow-integral history has no {column} column")
                        continue
                    if not np.isfinite(integrals[column].to_numpy(float)).all():
                        failures.append(f"{name}: non-finite values in {column}")
                if "kinetic_energy_rate_source" not in integrals:
                    failures.append(f"{name}: flow-integral history has no dE/dt source")
                else:
                    rate_sources = integrals["kinetic_energy_rate_source"].astype(str)
                    if (
                        rate_sources.str.startswith("undefined_").any()
                        or rate_sources.isin(("", "nan", "unknown")).any()
                    ):
                        failures.append(f"{name}: undefined dE/dt source")
                for column in ("kinetic_energy_rate", "viscous_kinetic_energy_rate"):
                    if column not in integrals:
                        continue
                    tested_column = column
                    if scheme == "rwm" and f"{column}_ci_lower" in integrals:
                        # A positive sample mean is not a physical violation when
                        # its confidence interval still includes zero.  Fail only
                        # when the modeled rate is significantly positive.
                        tested_column = f"{column}_ci_lower"
                    tested_values = integrals[tested_column].to_numpy(float)
                    if column == "kinetic_energy_rate":
                        tested_values = tested_values[_reportable_energy_rate_mask(integrals)]
                    if np.any(tested_values > 1.0e-7):
                        failures.append(
                            f"{name}: significantly positive modeled energy rate in {column}"
                        )

            fields = _read_csv(folder / "field_diagnostics.csv", failures, allow_nonfinite=True)
            if fields is not None:
                required = {"time", "step", "core_radius_0", "mean_core_radius"}
                if not required.issubset(fields.columns):
                    failures.append(
                        f"{name}: missing field columns {sorted(required - set(fields.columns))}"
                    )
                else:
                    if np.any(fields["core_radius_0"].to_numpy(float) <= 0.0):
                        failures.append(f"{name}: non-positive extracted core radius")
                    field_time = fields["time"].to_numpy(float)
                    field_cadence = np.diff(field_time)
                    field_cadence = field_cadence[field_cadence > 0.0]
                    allowed_gap = (
                        float(np.median(field_cadence)) if field_cadence.size else EXPECTED_DT
                    )
                    if fields["time"].iloc[-1] < EXPECTED_END_TIME - allowed_gap - EXPECTED_DT:
                        failures.append(f"{name}: field diagnostics are incomplete")
                boundary_columns = [column for column in fields if "boundary_limited" in column]
                if any(bool(fields[column].astype(bool).any()) for column in boundary_columns):
                    failures.append(f"{name}: extracted core radius is boundary limited")

            if scheme == "rwm":
                member_metadata = [
                    _metadata(path)
                    for path in sorted(SOLUTION_DIR.glob(f"{name}_*/vpm_metadata.json"))
                ]
                member_metadata = [item for item in member_metadata if item]
                ensemble_size = len(member_metadata)
                seeds = [item.get("random_seed") for item in member_metadata]
                if ensemble_size < MINIMUM_ENSEMBLE_SIZE or len(set(seeds)) != ensemble_size:
                    failures.append(f"{name}: fewer than four independent solver seeds")
                convergence = _read_csv(folder / "rwm_convergence.csv", failures)
                if convergence is not None:
                    for column in (
                        "relative_standard_error_l2_velocity",
                        "relative_standard_error_l2_vorticity",
                    ):
                        if column not in convergence:
                            failures.append(f"{name}: missing convergence column {column}")
                        elif float(convergence[column].max()) > RWM_RELATIVE_STANDARD_ERROR_LIMIT:
                            failures.append(
                                f"{name}: {column} exceeds the 7.5% Monte Carlo uncertainty limit"
                            )
                    capture = convergence.get("minimum_absolute_circulation_capture_fraction")
                    if capture is None or float(capture.min()) < 0.995:
                        failures.append(
                            f"{name}: column projection loses more than 0.5% circulation"
                        )
                if fields is not None:
                    uncertainty_columns = {
                        "core_radius_0_standard_error",
                        "vortex_centre_0_x_standard_error",
                    }
                    if not uncertainty_columns.issubset(fields.columns):
                        failures.append(
                            f"{name}: missing feature uncertainty columns "
                            f"{sorted(uncertainty_columns - set(fields.columns))}"
                        )
                    if name == "merging_rwm" and "is_pair_unresolved" in fields:
                        unresolved = (
                            fields["is_pair_unresolved"].astype(str).str.lower().isin(("true", "1"))
                        )
                        if not unresolved.any():
                            failures.append(f"{name}: statistically resolved pair never merges")
                        elif np.any(np.diff(unresolved.to_numpy(int)) < 0):
                            failures.append(f"{name}: pair resolution resurrects after merger")

            if not list(folder.glob("*_zq.pvd")):
                failures.append(f"{name}: missing sampled surface-field PVD")

    if "gbd" in schemes:
        failures.extend(_gbd_moment_recovery_failures(cases))

    if "merging" in cases:
        _, normalization_failures = merging_normalization_audit(SAMPLES_DIR, schemes)
        failures.extend(normalization_failures)

    if "vortex" in cases:
        try:
            vortex_audit = single_vortex_error_audit(SAMPLES_DIR, schemes)
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            failures.append(f"single-vortex analytic comparison failed: {error}")
        else:
            for scheme, values in vortex_audit["runs"].items():
                if scheme not in schemes:
                    continue
                velocity = values["relative_l2_velocity"]
                vorticity = values["relative_l2_vorticity"]
                gradient = values["relative_l2_velocity_gradient"]
                if scheme in {"cs", "rwm"} and max(velocity, vorticity, gradient) > 0.20:
                    failures.append(f"vortex_{scheme}: analytic profile error exceeds 20%")
                if scheme in {"dvh", "gbd"} and max(velocity, vorticity, gradient) > 0.50:
                    failures.append(f"vortex_{scheme}: analytic profile error exceeds 50%")
                if values["core_radius_relative_rmse"] > 0.20:
                    failures.append(f"vortex_{scheme}: analytic core-growth RMSE exceeds 20%")
                if abs(values["core_radius_final_relative_error"]) > 0.25:
                    failures.append(
                        f"vortex_{scheme}: final analytic core-radius error exceeds 25%"
                    )

    if "dipole" in cases:
        try:
            dipole_audit = dipole_error_audit(SAMPLES_DIR, schemes)
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            failures.append(f"dipole comparison failed: {error}")
        else:
            for scheme, values in dipole_audit["runs"].items():
                if values["trajectory_rmse_over_reference_range"] > 0.50:
                    failures.append(f"dipole_{scheme}: trajectory error exceeds 50%")
                if abs(values["end_core_radius_relative_to_isolated_exact"]) > 0.35:
                    failures.append(f"dipole_{scheme}: final core-radius error exceeds 35%")

    if not pre_plot:
        for fig_name in (
            "vortex_comparison",
            "dipole_comparison",
            "merging_comparison",
            "vortex_surface_fields",
            "lamboseen_energy",
            "merging_render_t0",
            "merging_render_final",
        ):
            for suffix in ("png", "pdf") if figure_format == "both" else (figure_format,):
                figure = FIGURES_DIR / f"{fig_name}.{suffix}"
                if not figure.is_file() or figure.stat().st_size == 0:
                    failures.append(f"missing or empty figure {figure.name}")

    if failures:
        print("\n".join(f"[FAIL] {failure}" for failure in failures))
        return 1
    return 0
