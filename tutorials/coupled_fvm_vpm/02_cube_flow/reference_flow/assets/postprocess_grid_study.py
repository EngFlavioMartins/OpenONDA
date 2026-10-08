#!/usr/bin/env python3
"""Post-process force convergence for the cube reference grids."""

import math
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import find_peaks
from scipy.stats import t as student_t

from openonda.reference_grid import plot_force_grids, relative_change, richardson_gci
from openonda.results import (
    history_window,
    read_history_table,
    read_json,
    write_csv_table,
    write_json,
)

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


def force_statistics(path: Path, start: float, end: float) -> dict[str, float]:
    data = history_window(read_history_table(path), start, end, columns=FORCE_COLUMNS)
    time = data["time"]
    values = {name: data[name] for name in FORCE_COLUMNS}

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
    if len(cycles) >= 2:
        cycle_values = np.asarray(cycles)
        critical = float(student_t.ppf(0.975, len(cycles) - 1))
        half_width = critical * np.std(cycle_values, axis=0, ddof=1) / np.sqrt(len(cycles))
        uncertainty = dict(zip(uncertainty, map(float, half_width), strict=True))

    return {
        "mean_drag": means["drag_coefficient"],
        "rms_drag": rms["drag_coefficient"],
        "rms_lift": rms["lift_coefficient"],
        "rms_side": rms["side_force_coefficient"],
        "strouhal": strouhal,
        "complete_cycles": len(cycles),
        "uncertainty_95": uncertainty,
    }


def completed_grids(samples_dir: Path, start: float, end: float) -> list[dict]:
    grids = []
    for metadata_path in samples_dir.glob("grid_h*/grid_run.json"):
        metadata = read_json(metadata_path)
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
    rows = []
    for grid in grids:
        values = [grid[name] for name in ("name", "h", "cells", *METRICS)]
        changes = [relative_change(grid[metric], reference[metric]) for metric in METRICS]
        rows.append([*values, *changes])
    write_csv_table(path, rows, columns=fields)


def plot_forces(grids: list[dict], path: Path) -> None:
    plot_force_grids(grids, path)


def analyse_forces(
    samples_dir: Path = SAMPLES_DIR,
    output_dir: Path = OUTPUT_DIR,
    start: float = STATISTICS_START,
    end: float = STATISTICS_END,
) -> dict:
    grids = completed_grids(samples_dir, start, end)
    convergence = {metric: richardson_gci(grids, metric) for metric in METRICS}
    report = {
        "statistics_window": [start, end],
        "grids": grids,
        "convergence": convergence,
    }
    write_json(output_dir / "grid_forces.json", report)
    write_csv(grids, output_dir / "grid_forces.csv")
    plot_forces(grids, output_dir / "grid_forces.both")
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
