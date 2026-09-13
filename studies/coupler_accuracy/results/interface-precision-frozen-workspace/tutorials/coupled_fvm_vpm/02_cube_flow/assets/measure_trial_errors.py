"""Report force and velocity-profile errors for one isolated cube trial."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_table(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty table: {path}")
    numeric = {
        name: np.asarray([float(row[name]) for row in rows]) for name in rows[0] if name != "patch"
    }
    if "step" in numeric:
        reset = np.flatnonzero(np.diff(numeric["step"]) < 0.0)
        if reset.size:
            start = int(reset[-1] + 1)
            numeric = {name: values[start:] for name, values in numeric.items()}
    return numeric


def frame(table: dict[str, np.ndarray], time: float) -> dict[str, np.ndarray] | None:
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
        # The exact panel/wall trace is a two-sided jump value for the VPM
        # evaluator, not a fluid sample. Compare only open-fluid points.
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_directory", type=Path)
    parser.add_argument("reference_directory", type=Path)
    parser.add_argument("--acceptance-limit", type=float, default=0.05)
    arguments = parser.parse_args()
    if not 0.0 < arguments.acceptance_limit < 1.0:
        raise ValueError("acceptance limit must lie strictly between zero and one")
    samples = arguments.case_directory / "samples"
    reference_samples = arguments.reference_directory / "samples"
    acceptance_values: list[tuple[str, float]] = []

    candidate_force = load_table(samples / "forces_history.csv")
    reference_force = load_table(reference_samples / "forces_history.csv")
    for time, drag in zip(
        candidate_force["time"], candidate_force["drag_coefficient"], strict=True
    ):
        expected = float(
            np.interp(time, reference_force["time"], reference_force["drag_coefficient"])
        )
        record = {
            "metric": "drag_coefficient",
            "time": float(time),
            "candidate": float(drag),
            "reference": expected,
            "relative_error": abs(float(drag) - expected) / abs(expected),
        }
        print(json.dumps(record))
        acceptance_values.append((f"Cd at t={time:g}", float(record["relative_error"])))

    for name in ("centreline", "offaxis_y075"):
        reference_table = load_table(reference_samples / f"{name}.csv")
        source_tables = {
            source: load_table(samples / f"{source}_{name}.csv") for source in ("fvm", "vpm")
        }
        common_times = [
            time
            for time in np.unique(source_tables["fvm"]["time"])
            if np.any(np.isclose(reference_table["time"], time, rtol=0.0, atol=1.0e-10))
            and np.any(np.isclose(source_tables["vpm"]["time"], time, rtol=0.0, atol=1.0e-10))
        ]
        for time in common_times:
            reference_frame = frame(reference_table, time)
            assert reference_frame is not None
            for source, table in source_tables.items():
                candidate_frame = frame(table, time)
                assert candidate_frame is not None
                record = profile_record(source, name, time, candidate_frame, reference_frame)
                print(json.dumps(record))
                acceptance_values.append(
                    (
                        f"{source} {name} max at t={time:g}",
                        float(record["max_abs_over_u_inf"]),
                    )
                )

    failed = [
        (name, value) for name, value in acceptance_values if value > arguments.acceptance_limit
    ]
    if failed:
        details = ", ".join(f"{name}={value:.3%}" for name, value in failed)
        raise SystemExit(
            f"FAIL: metrics exceed the {arguments.acceptance_limit:.1%} limit: {details}"
        )
    worst_name, worst_value = max(acceptance_values, key=lambda item: item[1])
    print(
        f"PASS: all {len(acceptance_values)} acceptance metrics are at most "
        f"{arguments.acceptance_limit:.1%}; worst is {worst_name}={worst_value:.3%}"
    )


if __name__ == "__main__":
    main()
