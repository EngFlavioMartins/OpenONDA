"""Check eight spanwise cylinder slabs against the sixteen-slab reference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eight_slab_samples", type=Path)
    parser.add_argument("sixteen_slab_samples", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _force_comparison(eight: Path, sixteen: Path) -> list[dict]:
    rows_eight = _rows(eight / "forces_history.csv")
    rows_sixteen = _rows(sixteen / "forces_history.csv")
    by_time = {round(float(row["time"]), 10): row for row in rows_sixteen}
    comparisons = []
    for row in rows_eight:
        time = round(float(row["time"]), 10)
        if time not in by_time:
            continue
        reference = by_time[time]
        drag = float(row["drag_coefficient"])
        drag_reference = float(reference["drag_coefficient"])
        comparisons.append(
            {
                "time": time,
                "drag_coefficient_eight_slabs": drag,
                "drag_coefficient_sixteen_slabs": drag_reference,
                "relative_drag_difference": abs(drag - drag_reference)
                / max(abs(drag_reference), 1.0e-30),
                "absolute_lift_difference": abs(
                    float(row["lift_coefficient"]) - float(reference["lift_coefficient"])
                ),
            }
        )
    if not comparisons:
        raise ValueError("The force histories have no common sample times")
    return comparisons


def _spanwise_uniformity(samples: Path) -> dict:
    rows = _rows(samples / "spanwise_line.csv")
    latest_time = max(float(row["time"]) for row in rows)
    latest = [row for row in rows if np.isclose(float(row["time"]), latest_time)]
    velocity = np.asarray(
        [
            [float(row["velocity_x"]), float(row["velocity_y"]), float(row["velocity_z"])]
            for row in latest
        ]
    )
    return {
        "time": latest_time,
        "points": len(latest),
        "mean_streamwise_velocity": float(np.mean(velocity[:, 0])),
        "mean_transverse_velocity": float(np.mean(velocity[:, 1])),
        "streamwise_velocity_range": float(np.ptp(velocity[:, 0])),
        "transverse_velocity_range": float(np.ptp(velocity[:, 1])),
        "maximum_absolute_spanwise_velocity": float(np.max(np.abs(velocity[:, 2]))),
    }


def main() -> None:
    arguments = _arguments()
    forces = _force_comparison(arguments.eight_slab_samples, arguments.sixteen_slab_samples)
    eight_uniformity = _spanwise_uniformity(arguments.eight_slab_samples)
    sixteen_uniformity = _spanwise_uniformity(arguments.sixteen_slab_samples)
    maximum_drag_difference = max(row["relative_drag_difference"] for row in forces)
    mean_streamwise_difference = abs(
        eight_uniformity["mean_streamwise_velocity"]
        - sixteen_uniformity["mean_streamwise_velocity"]
    ) / max(abs(sixteen_uniformity["mean_streamwise_velocity"]), 1.0e-30)
    mean_transverse_difference = abs(
        eight_uniformity["mean_transverse_velocity"]
        - sixteen_uniformity["mean_transverse_velocity"]
    )
    streamwise_range_ratio = eight_uniformity["streamwise_velocity_range"] / max(
        sixteen_uniformity["streamwise_velocity_range"], 1.0e-30
    )
    transverse_range_ratio = eight_uniformity["transverse_velocity_range"] / max(
        sixteen_uniformity["transverse_velocity_range"], 1.0e-30
    )
    passed = (
        maximum_drag_difference < 0.01
        and mean_streamwise_difference < 0.005
        and mean_transverse_difference < 0.002
        and eight_uniformity["streamwise_velocity_range"] < 0.002
        and eight_uniformity["transverse_velocity_range"] < 0.002
        and sixteen_uniformity["streamwise_velocity_range"] < 0.002
        and sixteen_uniformity["transverse_velocity_range"] < 0.002
        and eight_uniformity["maximum_absolute_spanwise_velocity"] < 1.0e-3
        and sixteen_uniformity["maximum_absolute_spanwise_velocity"] < 1.0e-3
    )
    result = {
        "schema": "openonda-cylinder-spanwise-independence/1",
        "eight_slab_samples": str(arguments.eight_slab_samples.resolve()),
        "sixteen_slab_samples": str(arguments.sixteen_slab_samples.resolve()),
        "criteria": {
            "maximum_relative_drag_difference": 0.01,
            "maximum_relative_mean_streamwise_velocity_difference": 0.005,
            "maximum_absolute_mean_transverse_velocity_difference": 0.002,
            "maximum_absolute_in_plane_spanwise_range": 0.002,
            "maximum_absolute_spanwise_velocity": 1.0e-3,
        },
        "force_samples": forces,
        "maximum_relative_drag_difference": maximum_drag_difference,
        "relative_mean_streamwise_velocity_difference": mean_streamwise_difference,
        "absolute_mean_transverse_velocity_difference": mean_transverse_difference,
        "streamwise_velocity_range_ratio": streamwise_range_ratio,
        "transverse_velocity_range_ratio": transverse_range_ratio,
        "eight_slab_spanwise_uniformity": eight_uniformity,
        "sixteen_slab_spanwise_uniformity": sixteen_uniformity,
        "passed": passed,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit("Spanwise-independence check failed")


if __name__ == "__main__":
    main()
