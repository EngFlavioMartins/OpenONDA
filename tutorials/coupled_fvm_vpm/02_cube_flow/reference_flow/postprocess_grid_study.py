#!/usr/bin/env python3
"""Post-process force convergence for the cube reference grids."""

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import find_peaks
from scipy.stats import t as student_t

from openonda.reference_grid import plot_force_grids, relative_change, richardson_gci

CASE_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = CASE_DIR / "samples"
OUTPUT_DIR = CASE_DIR / "figures"
STATISTICS_START = 15.0
STATISTICS_END = 30.0

FORCE_COLUMNS = (
    "drag_coefficient",
    "lift_coefficient",
    "side_force_coefficient",
)
METRICS = ("mean_drag", "rms_drag", "rms_lift", "rms_side", "strouhal")


def time_mean(time: np.ndarray, values: np.ndarray) -> float:
    return float(trapezoid(values, time) / (time[-1] - time[0]))


def _accepted_history(data: np.ndarray, path: Path) -> tuple[np.ndarray, int]:
    """Use the last repeated history only when prior records agree with it."""
    time = np.asarray(data["time"], dtype=float)
    if time.ndim != 1 or len(time) < 4 or not np.all(np.isfinite(time)):
        raise ValueError(f"{path} has too few finite force samples")
    starts = [0, *(np.flatnonzero(np.diff(time) < 0) + 1).tolist(), len(time)]
    histories = []
    duplicate_rows = 0
    for left, right in zip(starts[:-1], starts[1:]):
        segment = data[left:right]
        repeated = np.flatnonzero(np.diff(segment["time"]) == 0)
        for index in repeated:
            if not all(
                np.isclose(segment[name][index], segment[name][index + 1], rtol=1e-9, atol=1e-12)
                for name in FORCE_COLUMNS
            ):
                raise ValueError(f"{path} has conflicting duplicate-time force rows")
        if len(repeated):
            duplicate_rows += len(repeated)
            segment = np.delete(segment, repeated)
        if np.any(np.diff(segment["time"]) <= 0):
            raise ValueError(f"{path} time samples must be strictly increasing")
        histories.append(segment)
    accepted = histories[-1]
    for earlier in histories[:-1]:
        if len(earlier) > len(accepted) or not np.allclose(
            earlier["time"], accepted["time"][: len(earlier)], rtol=0.0, atol=1e-12
        ):
            raise ValueError(f"{path} has conflicting repeated force histories")
        if not all(
            np.allclose(earlier[name], accepted[name][: len(earlier)], rtol=1e-9, atol=1e-10)
            for name in FORCE_COLUMNS
        ):
            raise ValueError(f"{path} has conflicting repeated force histories")
    return accepted, len(histories) - 1 + duplicate_rows


def force_statistics(path: Path, start: float, end: float) -> dict[str, float]:
    if not path.is_file():
        raise ValueError(f"missing force history: {path}")
    data, repeated_histories = _accepted_history(
        np.genfromtxt(path, delimiter=",", names=True), path
    )
    time = np.asarray(data["time"], dtype=float)
    if time.ndim != 1 or len(time) < 4 or not np.all(np.isfinite(time)):
        raise ValueError(f"{path} has too few finite force samples")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError(f"{path} time samples must be strictly increasing")
    if time[0] > start or time[-1] < end:
        raise ValueError(f"{path} does not cover the requested window [{start}, {end}]")
    interior = (time > start) & (time < end)
    window_time = np.concatenate(([start], time[interior], [end]))
    values = {}
    for name in FORCE_COLUMNS:
        raw = np.asarray(data[name], dtype=float)
        if not np.all(np.isfinite(raw)):
            raise ValueError(f"{path} has non-finite force coefficients")
        values[name] = np.interp(window_time, time, raw)
    time = window_time
    if len(time) < 8 or np.max(np.diff(time)) > 0.1 * (end - start):
        raise ValueError(f"{path} has insufficient temporal coverage inside the window")

    means = {name: time_mean(time, value) for name, value in values.items()}
    rms = {
        name: math.sqrt(max(0.0, time_mean(time, (value - means[name]) ** 2)))
        for name, value in values.items()
    }

    uniform_time = np.linspace(time[0], time[-1], len(time))
    lift = np.interp(uniform_time, time, values["lift_coefficient"])
    side = np.interp(uniform_time, time, values["side_force_coefficient"])
    lift = lift - np.polyval(np.polyfit(uniform_time, lift, 1), uniform_time)
    side = side - np.polyval(np.polyfit(uniform_time, side, 1), uniform_time)
    frequencies = np.fft.rfftfreq(len(uniform_time), uniform_time[1] - uniform_time[0])
    window = np.hanning(len(time))
    spectrum = abs(np.fft.rfft(lift * window)) ** 2 + abs(np.fft.rfft(side * window)) ** 2
    strouhal = float(frequencies[1 + np.argmax(spectrum[1:])])
    signal = lift if np.std(lift) >= np.std(side) else side
    peaks, _ = find_peaks(
        signal,
        distance=max(1, int(0.6 / (strouhal * (uniform_time[1] - uniform_time[0])))),
        prominence=max(float(np.std(signal)) * 0.5, 1e-12),
    )
    peak_times = uniform_time[peaks]
    cycles = []
    for left, right in zip(peak_times[:-1], peak_times[1:]):
        cycle_time = np.concatenate(([left], time[(time > left) & (time < right)], [right]))
        drag = np.interp(cycle_time, time, values["drag_coefficient"])
        cl = np.interp(cycle_time, time, values["lift_coefficient"])
        cl_mean = time_mean(cycle_time, cl)
        cycles.append(
            (
                time_mean(cycle_time, drag),
                math.sqrt(max(0.0, time_mean(cycle_time, (cl - cl_mean) ** 2))),
                1 / (right - left),
            )
        )
    uncertainty = {name: None for name in ("mean_drag", "rms_lift", "strouhal")}
    reasons = []
    if len(cycles) < 10:
        reasons.append("fewer than ten complete force cycles")
    if len(cycles) >= 2:
        cycle_values = np.asarray(cycles)
        critical = float(student_t.ppf(0.975, len(cycles) - 1))
        half_width = critical * np.std(cycle_values, axis=0, ddof=1) / np.sqrt(len(cycles))
        uncertainty = dict(zip(uncertainty, map(float, half_width), strict=True))
        split = len(cycles) // 2
        drift = np.abs(cycle_values[:split].mean(axis=0) - cycle_values[split:].mean(axis=0))
        limits = np.array([0.02, 0.05, 0.02]) * np.maximum(np.abs(cycle_values.mean(axis=0)), 1e-12)
        if np.any(drift > np.maximum(limits, 2 * half_width)):
            reasons.append("force-cycle statistics drift across the window")
        if np.any(half_width > limits):
            reasons.append("cycle-block uncertainty exceeds accuracy targets")
    if max(rms["lift_coefficient"], rms["side_force_coefficient"]) < 1e-10:
        reasons.append("no resolved force oscillation")

    return {
        "mean_drag": means["drag_coefficient"],
        "rms_drag": rms["drag_coefficient"],
        "rms_lift": rms["lift_coefficient"],
        "rms_side": rms["side_force_coefficient"],
        "strouhal": strouhal,
        "complete_cycles": len(cycles),
        "uncertainty_95": uncertainty,
        "qualified_statistics": not reasons,
        "qualification_reasons": reasons,
        "repeated_history_segments": repeated_histories,
    }


def completed_grids(samples_dir: Path, start: float, end: float) -> list[dict]:
    grids = []
    for metadata_path in samples_dir.glob("grid_h*/grid_run.json"):
        metadata = json.loads(metadata_path.read_text())
        if float(metadata.get("end_time", 0.0)) < end:
            raise ValueError(f"incomplete grid run: {metadata_path}")
        name = str(metadata["case"])
        statistics = force_statistics(
            samples_dir / name / "forces_history.csv",
            start,
            end,
        )
        grids.append(
            {
                "name": name,
                "h": float(metadata["cell_size"]),
                "cells": int(metadata["cell_count"]),
                **statistics,
            }
        )
    return sorted(grids, key=lambda grid: -grid["h"])


def write_csv(grids: list[dict], path: Path) -> None:
    reference = grids[-1]
    fields = ["name", "h", "cells", *METRICS, *[f"{metric}_change" for metric in METRICS]]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for grid in grids:
            row = {name: grid[name] for name in ("name", "h", "cells", *METRICS)}
            for metric in METRICS:
                row[f"{metric}_change"] = relative_change(grid[metric], reference[metric])
            writer.writerow(row)


def plot_forces(grids: list[dict], path: Path) -> None:
    plot_force_grids(grids, path)


def analyse_forces(
    samples_dir: Path = SAMPLES_DIR,
    output_dir: Path = OUTPUT_DIR,
    start: float = STATISTICS_START,
    end: float = STATISTICS_END,
) -> dict:
    grids = completed_grids(samples_dir, start, end)
    if len(grids) < 3:
        raise ValueError("at least three completed grid_h cases are required")
    convergence = {metric: richardson_gci(grids, metric) for metric in METRICS}
    report = {
        "statistics_window": [start, end],
        "grids": grids,
        "convergence": convergence,
    }
    report["statistics_qualified"] = all(grid["qualified_statistics"] for grid in grids)
    report["force_grid_qualified"] = bool(
        report["statistics_qualified"]
        and convergence["mean_drag"]["valid"]
        and convergence["mean_drag"]["fine_gci"] <= 0.02
    )
    report["scope"] = (
        "Force-grid statistics only; temporal, domain and profile convergence remain separate gates."
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "grid_forces.json").write_text(json.dumps(report, indent=2) + "\n")
    write_csv(grids, output_dir / "grid_forces.csv")
    plot_forces(grids, output_dir / "grid_forces.png")
    return report


def print_report(report: dict) -> None:
    print(
        "case                 h [m]       cells       mean Cd       Cd rms       Cl rms       Cs rms       St"
    )
    for grid in report["grids"]:
        print(
            f"{grid['name']:<18} {grid['h']:>9.5f} {grid['cells']:>11d} "
            f"{grid['mean_drag']:>13.6g} {grid['rms_drag']:>12.6g} "
            f"{grid['rms_lift']:>12.6g} {grid['rms_side']:>12.6g} "
            f"{grid['strouhal']:>8.5f}"
        )
    print("\nRichardson/GCI from the three finest grids")
    for metric, values in report["convergence"].items():
        print(
            f"{metric:<12} p={values['order']} "
            f"extrapolated={values['extrapolated']} GCI_fine={values['fine_gci']}"
        )


if __name__ == "__main__":
    print_report(analyse_forces())
