#!/usr/bin/env python3
"""Post-process force convergence for the cube reference grids."""

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
    return float(np.trapezoid(values, time) / (time[-1] - time[0]))


def force_statistics(path: Path, start: float, end: float) -> dict[str, float]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    time = np.asarray(data["time"], dtype=float)
    selected = (time >= start) & (time <= end)
    time = time[selected]
    values = {
        name: np.asarray(data[name], dtype=float)[selected]
        for name in FORCE_COLUMNS
    }

    means = {name: time_mean(time, value) for name, value in values.items()}
    rms = {
        name: math.sqrt(time_mean(time, (value - means[name]) ** 2))
        for name, value in values.items()
    }

    uniform_time = np.linspace(time[0], time[-1], len(time))
    lift = np.interp(uniform_time, time, values["lift_coefficient"])
    side = np.interp(uniform_time, time, values["side_force_coefficient"])
    lift -= np.polyval(np.polyfit(uniform_time, lift, 1), uniform_time)
    side -= np.polyval(np.polyfit(uniform_time, side, 1), uniform_time)
    frequencies = np.fft.rfftfreq(len(uniform_time), uniform_time[1] - uniform_time[0])
    spectrum = abs(np.fft.rfft(lift)) ** 2 + abs(np.fft.rfft(side)) ** 2
    strouhal = frequencies[1 + np.argmax(spectrum[1:])]

    return {
        "mean_drag": means["drag_coefficient"],
        "rms_drag": rms["drag_coefficient"],
        "rms_lift": rms["lift_coefficient"],
        "rms_side": rms["side_force_coefficient"],
        "strouhal": float(strouhal),
    }


def completed_grids(samples_dir: Path, start: float, end: float) -> list[dict]:
    grids = []
    for metadata_path in samples_dir.glob("*/grid_run.json"):
        metadata = json.loads(metadata_path.read_text())
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


def richardson_gci(grids: list[dict], metric: str) -> dict:
    coarse, medium, fine = grids[-3:]
    r = medium["h"] / fine["h"]
    coarse_difference = coarse[metric] - medium[metric]
    fine_difference = medium[metric] - fine[metric]
    if coarse_difference * fine_difference <= 0 or fine_difference == 0:
        return {"order": None, "extrapolated": None, "fine_gci": None}

    order = math.log(abs(coarse_difference / fine_difference)) / math.log(r)
    denominator = r**order - 1.0
    extrapolated = fine[metric] + (fine[metric] - medium[metric]) / denominator
    fine_gci = 1.25 * abs((fine[metric] - medium[metric]) / fine[metric]) / denominator
    return {
        "order": order,
        "extrapolated": extrapolated,
        "fine_gci": fine_gci,
    }


def write_csv(grids: list[dict], path: Path) -> None:
    reference = grids[-1]
    fields = ["name", "h", "cells", *METRICS, *[f"{metric}_change" for metric in METRICS]]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for grid in grids:
            row = dict(grid)
            for metric in METRICS:
                row[f"{metric}_change"] = abs(grid[metric] - reference[metric]) / abs(
                    reference[metric]
                )
            writer.writerow(row)


def plot_forces(grids: list[dict], path: Path) -> None:
    h = [grid["h"] for grid in grids]
    figure, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)
    labels = {
        "mean_drag": r"$\overline{C_D}$",
        "rms_drag": r"$C_{D,\mathrm{rms}}$",
        "rms_lift": r"$C_{L,\mathrm{rms}}$",
        "rms_side": r"$C_{S,\mathrm{rms}}$",
        "strouhal": r"$St$",
    }
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
    convergence = {metric: richardson_gci(grids, metric) for metric in METRICS}
    report = {
        "statistics_window": [start, end],
        "grids": grids,
        "convergence": convergence,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "grid_forces.json").write_text(json.dumps(report, indent=2) + "\n")
    write_csv(grids, output_dir / "grid_forces.csv")
    plot_forces(grids, output_dir / "grid_forces.png")
    return report


def print_report(report: dict) -> None:
    print("case                 h [m]       cells       mean Cd       Cd rms       Cl rms       Cs rms       St")
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
