#!/usr/bin/env python3
"""Diagnose the fully meshed cylinder-reference force and solver histories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks


def _table(path: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        raise ValueError(f"No data rows in {path}")
    return {
        name: np.asarray([float(row[name]) for row in rows], dtype=float)
        for name in rows[0]
        if name != "patch"
    }


def _quadratic_extremum(time: np.ndarray, values: np.ndarray, index: int) -> tuple[float, float]:
    """Refine a sampled extremum with the local three-point parabola."""
    if index == 0 or index == time.size - 1:
        return float(time[index]), float(values[index])
    local_time = time[index - 1 : index + 2] - time[index]
    a, b, c = np.polyfit(local_time, values[index - 1 : index + 2], 2)
    if not np.isfinite(a) or abs(a) < 1.0e-14:
        return float(time[index]), float(values[index])
    offset = -b / (2.0 * a)
    if offset < local_time[0] or offset > local_time[-1]:
        return float(time[index]), float(values[index])
    return float(time[index] + offset), float(a * offset**2 + b * offset + c)


def _extrema(time: np.ndarray, lift: np.ndarray) -> tuple[list[dict], list[dict]]:
    if time.size < 3:
        return [], []
    sample_interval = float(np.median(np.diff(time)))
    minimum_separation = max(2, int(round(2.0 / sample_interval)))
    scale = max(float(np.ptp(lift)), 1.0e-12)
    prominence = 0.02 * scale
    maxima, _ = find_peaks(lift, distance=minimum_separation, prominence=prominence)
    minima, _ = find_peaks(-lift, distance=minimum_separation, prominence=prominence)

    def records(indices: np.ndarray, kind: str) -> list[dict]:
        result = []
        for index in indices:
            peak_time, peak_value = _quadratic_extremum(time, lift, int(index))
            result.append(
                {
                    "time": peak_time,
                    "lift_coefficient": peak_value,
                    "kind": kind,
                }
            )
        return result

    return records(maxima, "maximum"), records(minima, "minimum")


def _zero_crossings(time: np.ndarray, values: np.ndarray) -> list[float]:
    crossings = []
    for index in np.flatnonzero(values[:-1] * values[1:] < 0.0):
        fraction = -values[index] / (values[index + 1] - values[index])
        crossings.append(float(time[index] + fraction * (time[index + 1] - time[index])))
    return crossings


def _solver_diagnostics(path: Path) -> dict:
    maximum_cfl = 0.0
    maximum_continuity = 0.0
    maximum_velocity_residual = 0.0
    maximum_pressure_residual = 0.0
    nonfinite_values = 0
    last_step = 0
    last_time = 0.0
    count = 0
    previous_step = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            step = int(row["step"])
            if step != previous_step + 1:
                raise ValueError(
                    f"Non-contiguous diagnostics at step {step}; expected {previous_step + 1}"
                )
            previous_step = step
            count += 1
            last_step = step
            last_time = float(row["time"])
            maximum_cfl = max(maximum_cfl, float(row["max_courant_number"]))
            maximum_continuity = max(maximum_continuity, float(row["max_continuity_error"]))
            residuals = row.get("residuals", {})
            maximum_velocity_residual = max(
                maximum_velocity_residual, float(residuals.get("velocity", 0.0))
            )
            maximum_pressure_residual = max(
                maximum_pressure_residual,
                float(residuals.get("kinematic_pressure", 0.0)),
            )
            nonfinite_values += int(row.get("n_nonfinite_values", 0))
    return {
        "records": count,
        "last_step": last_step,
        "last_time": last_time,
        "maximum_courant_number": maximum_cfl,
        "maximum_continuity_error": maximum_continuity,
        "maximum_velocity_residual": maximum_velocity_residual,
        "maximum_pressure_residual": maximum_pressure_residual,
        "nonfinite_values": nonfinite_values,
    }


def _period_records(maxima: list[dict], minima: list[dict]) -> list[dict]:
    """Return same-sign periods in chronological order."""
    result = []
    for kind, records in (("maximum", maxima), ("minimum", minima)):
        for start, end in zip(records[:-1], records[1:]):
            result.append(
                {
                    "kind": kind,
                    "start_time": float(start["time"]),
                    "end_time": float(end["time"]),
                    "period": float(end["time"] - start["time"]),
                }
            )
    return sorted(result, key=lambda record: record["end_time"])


def _half_cycle_amplitudes(extrema: list[dict]) -> list[dict]:
    """Measure lift amplitude between each pair of alternating extrema."""
    result = []
    for first, second in zip(extrema[:-1], extrema[1:]):
        if first["kind"] == second["kind"]:
            continue
        result.append(
            {
                "time": 0.5 * (float(first["time"]) + float(second["time"])),
                "lift_amplitude": 0.5
                * abs(float(second["lift_coefficient"]) - float(first["lift_coefficient"])),
            }
        )
    return result


def _time_mean(time: np.ndarray, values: np.ndarray) -> float:
    if time.size == 1:
        return float(values[0])
    return float(np.trapezoid(values, time) / (time[-1] - time[0]))


def _cycle_records(
    time: np.ndarray,
    drag: np.ndarray,
    lift: np.ndarray,
    periods: list[dict],
) -> list[dict]:
    """Calculate force statistics on every complete same-sign cycle."""
    result = []
    for period in periods:
        start = float(period["start_time"])
        end = float(period["end_time"])
        mask = (time >= start) & (time <= end)
        if np.count_nonzero(mask) < 3:
            continue
        cycle_time = time[mask]
        cycle_drag = drag[mask]
        cycle_lift = lift[mask]
        mean_lift = _time_mean(cycle_time, cycle_lift)
        result.append(
            {
                "kind": period["kind"],
                "start_time": start,
                "end_time": end,
                "period": float(period["period"]),
                "mean_drag_coefficient": _time_mean(cycle_time, cycle_drag),
                "mean_lift_coefficient": mean_lift,
                "rms_lift_fluctuation": float(
                    np.sqrt(_time_mean(cycle_time, (cycle_lift - mean_lift) ** 2))
                ),
                "lift_amplitude": float(0.5 * np.ptp(cycle_lift)),
            }
        )
    return result


def _relative_spread(values: np.ndarray) -> float:
    mean = float(np.mean(np.abs(values)))
    return float(np.ptp(values) / mean) if mean > 0.0 else float("inf")


def _strict_json(value):
    """Replace unavailable floating-point values with JSON null."""
    if isinstance(value, dict):
        return {key: _strict_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_strict_json(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def analyse(case: Path) -> dict:
    forces = _table(case / "samples" / "forces_history.csv")
    time = forces["time"]
    step = forces["step"].astype(int)
    drag = forces["drag_coefficient"]
    lift = forces["lift_coefficient"]
    if not (np.all(np.diff(time) > 0.0) and np.all(np.diff(step) > 0)):
        raise ValueError("Force history is not strictly monotone")
    if not all(np.all(np.isfinite(values)) for values in (time, drag, lift)):
        raise ValueError("Force history contains non-finite values")

    maxima, minima = _extrema(time, lift)
    all_extrema = sorted([*maxima, *minima], key=lambda row: row["time"])
    period_records = _period_records(maxima, minima)
    half_cycle_amplitudes = _half_cycle_amplitudes(all_extrema)
    cycles = _cycle_records(time, drag, lift, period_records)
    periods = np.asarray([record["period"] for record in period_records], dtype=float)
    latest_period = float(np.median(periods[-3:])) if periods.size else float("nan")
    strouhal = 1.0 / latest_period if np.isfinite(latest_period) else float("nan")

    recent_amplitudes = np.asarray(
        [record["lift_amplitude"] for record in half_cycle_amplitudes[-6:]],
        dtype=float,
    )
    amplitude_spread = (
        _relative_spread(recent_amplitudes) if recent_amplitudes.size >= 6 else float("nan")
    )
    recent_cycles = cycles[-3:]
    cycle_drag_spread = (
        _relative_spread(np.asarray([record["mean_drag_coefficient"] for record in recent_cycles]))
        if len(recent_cycles) >= 3
        else float("nan")
    )
    cycle_period_spread = (
        _relative_spread(np.asarray([record["period"] for record in recent_cycles]))
        if len(recent_cycles) >= 3
        else float("nan")
    )
    saturated = bool(
        np.isfinite(amplitude_spread)
        and amplitude_spread < 0.02
        and np.isfinite(cycle_drag_spread)
        and cycle_drag_spread < 0.01
        and np.isfinite(cycle_period_spread)
        and cycle_period_spread < 0.01
    )

    last_cycle_start = time[-1] - latest_period if np.isfinite(latest_period) else time[0]
    last_cycle = time >= last_cycle_start
    solver = _solver_diagnostics(case / "solution" / "diagnostics.jsonl")
    median_sample_step = int(round(float(np.median(np.diff(step))))) if step.size > 1 else 1
    sample_lag_steps = int(solver["last_step"] - step[-1])
    histories_aligned = bool(0 <= sample_lag_steps <= median_sample_step)
    solver_healthy = bool(
        solver["nonfinite_values"] == 0
        and solver["maximum_courant_number"] < 1.0
        and solver["maximum_continuity_error"] < 1.0e-6
        and histories_aligned
    )
    mean_lift = _time_mean(time[last_cycle], lift[last_cycle])
    force_scale = float(
        np.median(forces["total_force_x"][np.abs(drag) > 1.0e-12] / drag[np.abs(drag) > 1.0e-12])
    )
    pressure_drag = forces["pressure_force_x"] / force_scale
    viscous_drag = forces["viscous_force_x"] / force_scale
    pressure_lift = forces["pressure_force_y"] / force_scale
    viscous_lift = forces["viscous_force_y"] / force_scale
    closure = np.column_stack(
        [
            forces[f"total_force_{axis}"]
            - forces[f"pressure_force_{axis}"]
            - forces[f"viscous_force_{axis}"]
            for axis in "xyz"
        ]
    )
    closure_scale = max(
        float(
            np.max(
                np.linalg.norm(
                    np.column_stack([forces[f"total_force_{axis}"] for axis in "xyz"]),
                    axis=1,
                )
            )
        ),
        1.0e-30,
    )
    report = {
        "schema": 1,
        "case": "fully_meshed_body_fitted_reference",
        "sample_count": int(time.size),
        "last_sample_time": float(time[-1]),
        "last_sample_step": int(step[-1]),
        "force_extrema": {"maxima": maxima, "minima": minima},
        "lift_zero_crossings": _zero_crossings(time, lift),
        "period_estimates": period_records,
        "half_cycle_amplitudes": half_cycle_amplitudes,
        "complete_cycles": cycles,
        "latest_period": latest_period,
        "latest_strouhal_number": strouhal,
        "latest_cycle": {
            "start_time": float(last_cycle_start),
            "mean_drag_coefficient": _time_mean(time[last_cycle], drag[last_cycle]),
            "mean_lift_coefficient": mean_lift,
            "rms_lift_fluctuation": float(
                np.sqrt(_time_mean(time[last_cycle], (lift[last_cycle] - mean_lift) ** 2))
            ),
            "lift_amplitude": float(0.5 * np.ptp(lift[last_cycle])),
            "mean_pressure_drag_coefficient": _time_mean(
                time[last_cycle], pressure_drag[last_cycle]
            ),
            "mean_viscous_drag_coefficient": _time_mean(time[last_cycle], viscous_drag[last_cycle]),
            "rms_pressure_lift_fluctuation": float(
                np.sqrt(
                    _time_mean(
                        time[last_cycle],
                        (
                            pressure_lift[last_cycle]
                            - _time_mean(time[last_cycle], pressure_lift[last_cycle])
                        )
                        ** 2,
                    )
                )
            ),
            "rms_viscous_lift_fluctuation": float(
                np.sqrt(
                    _time_mean(
                        time[last_cycle],
                        (
                            viscous_lift[last_cycle]
                            - _time_mean(time[last_cycle], viscous_lift[last_cycle])
                        )
                        ** 2,
                    )
                )
            ),
        },
        "force_balance": {
            "coefficient_force_scale": force_scale,
            "maximum_pressure_plus_viscous_closure_relative_error": float(
                np.max(np.linalg.norm(closure, axis=1)) / closure_scale
            ),
            "maximum_absolute_side_force_coefficient": float(
                np.max(np.abs(forces["side_force_coefficient"]))
            ),
        },
        "saturation": {
            "last_six_half_cycle_amplitude_relative_spread": amplitude_spread,
            "last_three_cycle_mean_drag_relative_spread": cycle_drag_spread,
            "last_three_cycle_period_relative_spread": cycle_period_spread,
            "amplitude_limit": 0.02,
            "mean_drag_limit": 0.01,
            "period_limit": 0.01,
            "passed": saturated,
        },
        "history_alignment": {
            "diagnostic_minus_sample_steps": sample_lag_steps,
            "sample_interval_steps": median_sample_step,
            "passed": histories_aligned,
        },
        "solver": solver,
        "solver_healthy": solver_healthy,
        "status": "statistically_ready" if saturated and solver_healthy else "developing",
    }
    report = _strict_json(report)
    output = case / "solution" / "reference_diagnostics.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "case",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "reference_flow",
    )
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="exit nonzero unless the force history is saturated and the solver is healthy",
    )
    args = parser.parse_args()
    report = analyse(args.case.resolve())
    print(json.dumps(report, indent=2))
    if args.require_ready and report["status"] != "statistically_ready":
        raise SystemExit("Reference force history is not statistically ready")


if __name__ == "__main__":
    main()
