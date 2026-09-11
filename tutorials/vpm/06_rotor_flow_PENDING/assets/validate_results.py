#!/usr/bin/env python3
"""Check completion, native data, stationarity, impulse and matched rotor theory."""

if not __package__:
    from pathlib import Path
    from openonda.tutorial_runner import case_package

    __package__ = case_package(Path(__file__).resolve().parents[1]) + ".assets"

import argparse
import numpy as np
import pandas as pd

from ._common import (
    FIGURES_DIR,
    IMPULSE_WINDOW_REVOLUTIONS,
    OPERATING_WINDOW_REVOLUTIONS,
    bem_reference,
    impulse_history,
    performance,
    rotor_inputs,
)
from .finite_distance_theory import (
    AXIAL_REFERENCE_FLOOR,
    AXIAL_RELATIVE_TOLERANCE,
    TANGENTIAL_REFERENCE_FLOOR,
    TANGENTIAL_RELATIVE_TOLERANCE,
)
from .plot_rotor_wake_planes import (
    assess_wake_signal_onset,
    checkpoint_particle_front_brackets,
    native_plane_windows,
    relative_drift,
)
from .plot_rotor_wake_planes import finite_distance_profiles

VALIDATION_ROTATIONS = OPERATING_WINDOW_REVOLUTIONS


def _scaled_profile_error(actual, reference, floor):
    actual = np.asarray(actual, dtype=float)
    reference = np.asarray(reference, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(reference) & (np.abs(reference) >= floor)
    if not np.any(valid):
        return np.nan
    scaled = (actual[valid] - reference[valid]) / np.maximum(np.abs(reference[valid]), floor)
    return float(np.sqrt(np.mean(scaled**2)))


def _rms_error(actual, reference):
    actual = np.asarray(actual, dtype=float)
    reference = np.asarray(reference, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(reference)
    if not np.any(valid):
        return np.nan, 0, len(actual)
    return (
        float(np.sqrt(np.mean((actual[valid] - reference[valid]) ** 2))),
        int(valid.sum()),
        len(actual),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-plot", action="store_true")
    args = parser.parse_args()
    p = rotor_inputs()
    config, state = p.metadata["configuration"], p.metadata["state"]
    final_step = state["initial_step"] + config["run"]["steps"]
    end = state["initial_time"] + config["run"]["steps"] * p.time_step_size
    failures = []
    if p.metadata["lifecycle"]["status"] != "completed" or state["step"] != final_step:
        print(
            f"[FAIL] Run incomplete; last recorded step {state['step']}/{final_step}; convergence is unqualified"
        )
        return 1
    data = performance()
    if not np.isfinite(data.select_dtypes("number")).all().all() or data.time.duplicated().any():
        failures.append("non-finite or duplicate force samples")
    try:
        raw_force_interval = config["numerics"]["vlm"]["logging_interval_steps"]
        force_interval_steps = int(raw_force_interval)
    except (KeyError, TypeError, ValueError, OverflowError):
        raw_force_interval = None
        force_interval_steps = 0
    if isinstance(raw_force_interval, bool) or raw_force_interval != 1:
        failures.append("coupled rotor force/loading history is not on every accepted owner step")
    cadence = p.time_step_size
    if data.time.max() < end - cadence - 1e-9:
        failures.append("force samples do not cover the configured horizon")
    tail = data[data.time > end - VALIDATION_ROTATIONS * p.rotation_period]
    if (
        len(tail) < 4
        or tail.time.max() - tail.time.min()
        < VALIDATION_ROTATIONS * p.rotation_period - 2 * cadence
    ):
        print(f"[FAIL] Fewer than {VALIDATION_ROTATIONS} revolutions of final force samples")
        return 1
    bem = bem_reference()
    for key, reference_key in (("CT", "thrust_coefficient"), ("CP", "power_coefficient")):
        mean = tail[key].mean()
        reference = bem.attrs[reference_key]
        drift = relative_drift(tail[key])
        error = abs(mean / reference - 1)
        print(
            f"{key}: mean={mean:.6f}, BEM={reference:.6f}, difference={error:.2%}, window drift={drift:.2%}"
        )
        if mean <= 0 or drift > 0.02 or error > 0.15:
            failures.append(
                f"{key}: positive load, <=2% tail drift and <=15% BEM difference required"
            )
    integrals = pd.read_csv(p.samples_dir / "flow_integrals.csv")
    try:
        flow_sampler = next(
            item for item in config["samplers"]["items"] if item["type"] == "FlowIntegralsSampler"
        )
        flow_schedule = flow_sampler["schedule"]
        cadence_kwargs = {
            "flow_interval_steps": flow_schedule["interval"]
            if flow_schedule["type"] == "EverySteps"
            else None,
            "flow_interval_time": flow_schedule["interval"]
            if flow_schedule["type"] == "EveryTime"
            else None,
        }
        _, loads, fluid, relaxation = impulse_history(
            data,
            integrals,
            density=p.density,
            time_step_size=p.time_step_size,
            **cadence_kwargs,
            start_time=end - IMPULSE_WINDOW_REVOLUTIONS * p.rotation_period,
            end_time=end,
        )
        thrust_impulse = loads[-1, 0]
        ratio = fluid[-1, 0] / thrust_impulse
        transfer = relaxation[-1, 0] / thrust_impulse
        print(
            f"Coupled impulse / surface-force impulse: {ratio:.4f}; "
            f"recorded relaxation contribution: {transfer:+.2%}"
        )
        if not np.isfinite(ratio) or abs(ratio - 1) > 0.10:
            failures.append(
                "coupled impulse / surface-force impulse differs from unity by more than10%"
            )
        if not config["numerics"]["vlm"]["force"].get("unsteady", False):
            failures.append("native unsteady pressure forces are required for the rotor balance")
    except ValueError as error:
        failures.append(str(error))
    try:
        for bracket in checkpoint_particle_front_brackets(p):
            if bracket["status"] == "bracketed":
                print(
                    f"particle-front bracket at x={bracket['station_x']:.6g} m: "
                    f"t={bracket['previous_time']:.3f}–{bracket['crossing_time']:.3f} s"
                )
            else:
                status = bracket["status"]
                reason = f" ({bracket['reason']})" if bracket.get("reason") else ""
                print(
                    f"particle-front evidence {status} at x={bracket['station_x']:.6g} m{reason}; "
                    "convected arrival remains unqualified"
                )
        for name in ("wake_1D", "wake_2D"):
            onset = assess_wake_signal_onset(p, name)
            onset_time = onset["signal_onset_time"]
            if onset_time is None:
                failures.append(
                    f"{name}: induced-velocity signal onset not demonstrated at "
                    f"{onset['relative_threshold']:.1%} RMS threshold"
                )
                print(
                    f"{name}: signal onset not demonstrated; threshold={onset['threshold']:.4g} m/s"
                )
            else:
                final_window_start = state["time"] - VALIDATION_ROTATIONS * p.rotation_period
                print(
                    f"{name}: signal onset at t={onset_time:.3f} s; "
                    f"five-revolution window starts at t={final_window_start:.3f} s"
                )
                if onset_time > final_window_start + 1.0e-10:
                    failures.append(
                        f"{name}: five-revolution mean/stationarity window starts before signal onset"
                    )
        for row in native_plane_windows(
            p,
            rotations=VALIDATION_ROTATIONS,
            require_complete=True,
            required_names=("wake_1D", "wake_2D"),
        ):
            drift = row["induced_field_drift"]
            print(
                f"{row['name']}: induced-velocity field drift={drift:.2%} "
                f"over {row['compared_rotations']} revolutions"
            )
            if not np.isfinite(drift) or drift > 0.01:
                failures.append(f"{row['name']}: no resolved stationary wake within1% field drift")
    except (OSError, ValueError, KeyError) as error:
        failures.append(f"invalid native wake samples: {error}")
    try:
        for row in finite_distance_profiles(p):
            axial_error = _scaled_profile_error(
                row["actual_axial_induction"],
                row["reference_axial_induction"],
                AXIAL_REFERENCE_FLOOR,
            )
            tangential_error = _scaled_profile_error(
                row["actual_tangential_induction"],
                row["reference_tangential_induction"],
                TANGENTIAL_REFERENCE_FLOOR,
            )
            axial_dimensional, axial_count, axial_total = _rms_error(
                row["actual_axial_velocity"], row["reference_axial_velocity"]
            )
            tangential_dimensional, tangential_count, tangential_total = _rms_error(
                row["actual_tangential_velocity"], row["reference_tangential_velocity"]
            )
            print(
                f"{row['name']}: finite-distance VC diagnostic "
                f"axial={axial_dimensional:.4g} m/s ({axial_count}/{axial_total} bins, "
                f"scaled {axial_error:.2%}); azimuthal={tangential_dimensional:.4g} m/s "
                f"({tangential_count}/{tangential_total} bins, scaled {tangential_error:.2%})"
            )
            if (
                not np.isfinite(axial_error)
                or axial_error > AXIAL_RELATIVE_TOLERANCE
                or axial_count < max(1, axial_total // 2)
            ):
                print(
                    f"[DIAGNOSTIC FLAG] {row['name']}: finite-distance axial comparison "
                    f"is outside the predeclared {AXIAL_RELATIVE_TOLERANCE:.0%} screen"
                )
            if (
                not np.isfinite(tangential_error)
                or tangential_error > TANGENTIAL_RELATIVE_TOLERANCE
                or tangential_count < max(1, tangential_total // 2)
            ):
                print(
                    f"[DIAGNOSTIC FLAG] {row['name']}: finite-distance azimuthal comparison "
                    f"is outside the predeclared {TANGENTIAL_RELATIVE_TOLERANCE:.0%} screen"
                )
        print(
            "[UNQUALIFIED] Finite-distance vortex-cylinder comparisons are diagnostic "
            "only; the screens are not uncertainty-based acceptance gates."
        )
    except (OSError, ValueError, KeyError, FloatingPointError) as error:
        failures.append(f"invalid finite-distance induction reference: {error}")
    if not args.pre_plot:
        for name in (
            "rotor_performance",
            "rotor_loading_validation",
            "rotor_wake_planes",
            "rotor_induction_validation",
        ):
            for extension in ("png", "pdf"):
                if not (FIGURES_DIR / f"{name}.{extension}").is_file():
                    failures.append(f"missing {name}.{extension}")
    print(
        "\n".join(f"[FAIL] {message}" for message in failures)
        or "[OK] Core rotor checks passed; convected wake arrival and finite-distance "
        "theory remain unqualified"
    )
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
