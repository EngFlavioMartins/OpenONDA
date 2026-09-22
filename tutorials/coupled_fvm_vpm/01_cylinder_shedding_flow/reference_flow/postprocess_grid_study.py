#!/usr/bin/env python3
"""Post-process force convergence for the cylinder reference grids."""

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import find_peaks
from scipy.stats import t as student_t

CASE_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = CASE_DIR / "samples"
OUTPUT_DIR = CASE_DIR / "figures"
STATISTICS_START = 40.0
STATISTICS_END = 100.0

FORCE_COLUMNS = (
    "drag_coefficient",
    "lift_coefficient",
    "side_force_coefficient",
)
METRICS = ("mean_drag", "rms_drag", "rms_lift", "rms_side", "strouhal")


def time_mean(time: np.ndarray, values: np.ndarray) -> float:
    return float(trapezoid(values, time) / (time[-1] - time[0]))


def force_statistics(path: Path, start: float, end: float) -> dict[str, float]:
    if not path.is_file():
        raise ValueError(f"missing force history: {path}")
    data = np.genfromtxt(path, delimiter=",", names=True)
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
    uniform_time = np.linspace(start, end, len(time))
    lift = np.interp(uniform_time, time, values["lift_coefficient"])
    detrended = lift - np.polyval(np.polyfit(uniform_time, lift, 1), uniform_time)
    frequencies = np.fft.rfftfreq(len(time), uniform_time[1] - uniform_time[0])
    spectrum = abs(np.fft.rfft(detrended * np.hanning(len(lift)))) ** 2
    peak_frequency = float(frequencies[1 + np.argmax(spectrum[1:])])
    peaks, _ = find_peaks(
        detrended,
        distance=max(1, int(0.6 / (peak_frequency * (uniform_time[1] - uniform_time[0])))),
        prominence=max(float(np.std(detrended)) * 0.5, 1e-12),
    )
    peak_times = []
    for peak in peaks:
        # Subsample peak location avoids quantizing the shedding period to a
        # force-output timestep. The spectral peak only locates this branch.
        denominator = detrended[peak - 1] - 2 * detrended[peak] + detrended[peak + 1]
        offset = (
            0.0
            if denominator == 0
            else 0.5 * (detrended[peak - 1] - detrended[peak + 1]) / denominator
        )
        peak_times.append(uniform_time[peak] + offset * (uniform_time[1] - uniform_time[0]))
    periods = np.diff(peak_times)
    cycles = []
    for left, right in zip(peak_times[:-1], peak_times[1:]):
        cycle_time = np.concatenate(([left], time[(time > left) & (time < right)], [right]))
        drag = np.interp(cycle_time, time, values["drag_coefficient"])
        cl = np.interp(cycle_time, time, values["lift_coefficient"])
        cl_mean = time_mean(cycle_time, cl)
        cycles.append(
            (
                time_mean(cycle_time, drag),
                math.sqrt(time_mean(cycle_time, (cl - cl_mean) ** 2)),
                1 / (right - left),
            )
        )
    cycle_values = np.asarray(cycles).reshape(-1, 3)
    uncertainty = {name: None for name in ("mean_drag", "rms_lift", "strouhal")}
    reasons = []
    if len(cycles) < 10:
        reasons.append("fewer than ten complete shedding periods")
    if len(cycles) >= 2:
        # Complete shedding periods are the blocks, not adjacent CSV rows.
        critical = float(student_t.ppf(0.975, len(cycles) - 1))
        half_width = critical * np.std(cycle_values, axis=0, ddof=1) / np.sqrt(len(cycles))
        uncertainty = dict(zip(uncertainty, map(float, half_width)))
        split = len(cycles) // 2
        first = cycle_values[:split].mean(axis=0)
        last = cycle_values[split:].mean(axis=0)
        drift = np.abs(last - first)
        limits = np.array([0.02, 0.05, 0.02]) * np.maximum(np.abs(cycle_values.mean(axis=0)), 1e-12)
        if np.any(drift > np.maximum(limits, 2 * half_width)):
            reasons.append("cycle statistics drift between the first and second half")
        if np.any(half_width > limits):
            reasons.append("cycle-block uncertainty exceeds the registered accuracy targets")
    if rms["lift_coefficient"] < 1e-10:
        reasons.append("no resolved shedding signal")
    return {
        "mean_drag": means["drag_coefficient"],
        "rms_drag": rms["drag_coefficient"],
        "rms_lift": rms["lift_coefficient"],
        "rms_side": rms["side_force_coefficient"],
        "strouhal": float(1 / np.mean(periods)) if len(periods) else peak_frequency,
        "complete_cycles": len(cycles),
        "uncertainty_95": uncertainty,
        "qualified_statistics": not reasons,
        "qualification_reasons": reasons,
    }


def completed_grids(samples_dir: Path, start: float, end: float) -> list[dict]:
    grids = []
    for metadata_path in samples_dir.glob("grid_h*/grid_run.json"):
        metadata = json.loads(metadata_path.read_text())
        if float(metadata.get("end_time", 0.0)) < end:
            raise ValueError(f"incomplete grid run: {metadata_path}")
        name = str(metadata["case"])
        grids.append(
            {
                "name": name,
                "h": float(metadata["cell_size"]),
                "cells": int(metadata["cell_count"]),
                **force_statistics(
                    samples_dir / name / "forces_history.csv",
                    start,
                    end,
                ),
            }
        )
    return sorted(grids, key=lambda grid: -grid["h"])


def richardson_gci(grids: list[dict], metric: str) -> dict:
    invalid = {"valid": False, "order": None, "extrapolated": None, "fine_gci": None}
    if len(grids) < 3:
        return {**invalid, "reason": "fewer than three meshes"}
    coarse, medium, fine = grids[-3:]
    if not all(
        np.isfinite(grid[metric]) and np.isfinite(grid["h"]) and grid["h"] > 0
        for grid in (coarse, medium, fine)
    ):
        return {**invalid, "reason": "nonfinite metric or invalid spacing"}
    ratio_a = coarse["h"] / medium["h"]
    ratio_b = medium["h"] / fine["h"]
    coarse_difference = coarse[metric] - medium[metric]
    fine_difference = medium[metric] - fine[metric]
    if (
        ratio_a <= 1
        or ratio_b <= 1
        or not math.isclose(ratio_a, ratio_b, rel_tol=5.0e-3)
        or coarse_difference * fine_difference <= 0
        or fine_difference == 0
    ):
        return {
            "valid": False,
            "order": None,
            "extrapolated": None,
            "fine_gci": None,
        }

    order = math.log(abs(coarse_difference / fine_difference)) / math.log(ratio_b)
    if not np.isfinite(order) or order <= 0:
        return {**invalid, "reason": "differences do not decrease with refinement"}
    uncertainty = [
        grid.get("uncertainty_95", {}).get(metric) or 0.0 for grid in (coarse, medium, fine)
    ]
    if (
        abs(coarse_difference) <= uncertainty[0] + uncertainty[1]
        or abs(fine_difference) <= uncertainty[1] + uncertainty[2]
    ):
        return {**invalid, "reason": "spatial differences unresolved within sampling uncertainty"}
    denominator = ratio_b**order - 1.0
    extrapolated = fine[metric] + (fine[metric] - medium[metric]) / denominator
    fine_gci = (
        1.25 * abs((fine[metric] - medium[metric]) / max(abs(fine[metric]), 1.0e-14)) / denominator
    )
    return {
        "valid": True,
        "order": order,
        "extrapolated": extrapolated,
        "fine_gci": fine_gci,
    }


def relative_change(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1.0e-14)


def write_csv(grids: list[dict], path: Path) -> None:
    reference = grids[-1]
    fields = ["name", "h", "cells", *METRICS, *[f"{name}_change" for name in METRICS]]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for grid in grids:
            row = {name: grid[name] for name in ("name", "h", "cells", *METRICS)}
            for metric in METRICS:
                row[f"{metric}_change"] = relative_change(grid[metric], reference[metric])
            writer.writerow(row)


def plot_forces(grids: list[dict], path: Path) -> None:
    h = [grid["h"] for grid in grids]
    labels = {
        "mean_drag": r"$\overline{C_D}$",
        "rms_drag": r"$C_{D,\mathrm{rms}}$",
        "rms_lift": r"$C_{L,\mathrm{rms}}$",
        "rms_side": r"$C_{S,\mathrm{rms}}$",
        "strouhal": r"$St$",
    }
    figure, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)
    for axis, metric in zip(axes.flat, METRICS, strict=False):
        axis.plot(h, [grid[metric] for grid in grids], "o-", color="black")
        axis.set_xlabel("wall spacing h [m]")
        axis.set_ylabel(labels[metric])
        axis.invert_xaxis()
    axes.flat[-1].axis("off")
    figure.savefig(path, dpi=200)
    plt.close(figure)


def analyse_forces(
    samples_dir: Path = SAMPLES_DIR,
    output_dir: Path = OUTPUT_DIR,
    start: float = STATISTICS_START,
    end: float = STATISTICS_END,
) -> dict:
    grids = completed_grids(samples_dir, start, end)
    if len(grids) < 3:
        raise ValueError("at least three completed grid_h cases are required")
    report = {
        "statistics_window": [start, end],
        "grids": grids,
        "convergence": {metric: richardson_gci(grids, metric) for metric in METRICS},
    }
    report["statistics_qualified"] = all(grid["qualified_statistics"] for grid in grids)
    changes = {
        metric: relative_change(grids[-2][metric], grids[-1][metric])
        for metric in ("mean_drag", "rms_lift", "strouhal")
    }
    report["fine_medium_relative_change"] = changes
    report["force_grid_qualified"] = bool(
        report["statistics_qualified"]
        and report["convergence"]["mean_drag"]["valid"]
        and report["convergence"]["mean_drag"]["fine_gci"] <= 0.02
        and changes["mean_drag"] <= 0.02
        and changes["rms_lift"] <= 0.05
        and changes["strouhal"] <= 0.02
    )
    report["scope"] = (
        "Force-grid qualification only; temporal, domain, span and velocity-profile convergence remain separate gates."
    )

    fine = grids[-1]
    selection = {
        "schema": "openonda-cylinder-reference-selection/1",
        "case": fine["name"],
        "samples_relative": str(Path("samples") / fine["name"]),
        "h": fine["h"],
        "statistics_window": [start, end],
    }
    selection["force_grid_qualified"] = report["force_grid_qualified"]
    selection_path = samples_dir.parent / "reference_selection.json"
    selection_path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "grid_forces.json").write_text(json.dumps(report, indent=2) + "\n")
    write_csv(grids, output_dir / "grid_forces.csv")
    plot_forces(grids, output_dir / "grid_forces.png")
    return report


def print_report(report: dict) -> None:
    print(
        "case                 h [m]       cells       mean Cd       Cd rms"
        "       Cl rms       Cs rms       St"
    )
    for grid in report["grids"]:
        print(
            f"{grid['name']:<20} {grid['h']:>7.5f} {grid['cells']:>11d} "
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
