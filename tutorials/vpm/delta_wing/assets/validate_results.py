#!/usr/bin/env python3
"""Certification checks for the two-wing wake-crossing tutorial."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CASE_DIR = Path(__file__).resolve().parents[1]
SAMPLES_DIR = CASE_DIR / "samples" / "delta_wing"
FIGURES_DIR = CASE_DIR / "figures"


def _finite_numeric(data: pd.DataFrame, label: str, failures: list[str]) -> None:
    numeric = data.select_dtypes(include=[np.number])
    if data.empty or numeric.empty or not np.isfinite(numeric.to_numpy()).all():
        failures.append(f"{label}: empty or non-finite numeric data")


def _cycle_drift(time: np.ndarray, force: np.ndarray) -> float:
    """Return the worst relative change in the last three one-second cycles."""
    cycle = np.floor(time).astype(int)
    ids = sorted(set(cycle))
    if len(ids) < 4:
        return float("inf")
    ids = ids[-3:]
    means: list[float] = []
    amplitudes: list[float] = []
    for item in ids:
        values = force[cycle == item]
        if values.size < 2:
            return float("inf")
        means.append(float(values.mean()))
        amplitudes.append(float(values.max() - values.min()))
    scale_mean = max(float(np.max(np.abs(means))), 1.0e-12)
    scale_amplitude = max(float(np.max(np.abs(amplitudes))), 1.0e-12)
    return max(
        float(np.ptp(means) / scale_mean),
        float(np.ptp(amplitudes) / scale_amplitude),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-plot", action="store_true")
    args = parser.parse_args()
    failures: list[str] = []
    metadata_path = SAMPLES_DIR / "motion_params.json"
    if not metadata_path.is_file():
        failures.append("missing motion_params.json")
        metadata: dict = {}
    else:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    expected_steps = int(metadata.get("n_steps", 0))
    expected_dt = float(metadata.get("time_step_size", np.nan))
    expected_end = expected_steps * expected_dt
    if expected_steps != 3520 or not np.isclose(expected_end, 8.8, atol=expected_dt):
        failures.append(f"unexpected configured horizon ({expected_steps=}, {expected_end=})")
    if not np.isclose(float(metadata.get("wings", {}).get("front_wing", np.nan)), 0.0):
        failures.append("front-wing phase metadata is not zero")
    if not np.isclose(float(metadata.get("wings", {}).get("rear_wing", np.nan)), np.pi):
        failures.append("rear-wing phase metadata is not pi")
    if metadata.get("status") != "complete" or metadata.get("completed") is not True:
        failures.append("motion metadata does not certify a completed run")

    for surface in ("front_wing", "rear_wing"):
        path = SAMPLES_DIR / f"vlm_spanwise_{surface}.csv"
        if not path.is_file():
            failures.append(f"missing {path.name}")
            continue
        try:
            data = pd.read_csv(path)
        except (OSError, ValueError, pd.errors.ParserError) as error:
            failures.append(f"{surface}: unreadable spanwise CSV ({error})")
            continue
        _finite_numeric(data, surface, failures)
        required = {"step", "time", "section_force_z"}
        if not required.issubset(data.columns):
            failures.append(f"{surface}: missing columns {sorted(required - set(data.columns))}")
            continue
        history = data.groupby("step", sort=True).agg(
            time=("time", "first"), force=("section_force_z", "sum")
        )
        time = history["time"].to_numpy(float)
        force = history["force"].to_numpy(float)
        if time.size < 10 or time[-1] < expected_end - expected_dt:
            failures.append(
                f"{surface}: incomplete force history ending at "
                f"{time[-1] if time.size else 'missing'}"
            )
        if time.size and np.any(np.diff(time) <= 0.0):
            failures.append(f"{surface}: force time is not strictly increasing")
        drift = _cycle_drift(time, force)
        print(
            f"{surface}: final_time={time[-1] if time.size else float('nan'):.6g}, "
            f"tail_cycle_drift={drift:.3%}"
        )
        if not np.isfinite(drift) or drift > 0.05:
            failures.append(f"{surface}: tail-cycle force drift {drift:.3%} > 5%")

    integrals_path = SAMPLES_DIR / "flow_integrals.csv"
    if not integrals_path.is_file():
        failures.append("missing flow_integrals.csv")
    else:
        try:
            integrals = pd.read_csv(integrals_path)
        except (OSError, ValueError, pd.errors.ParserError) as error:
            failures.append(f"unreadable flow_integrals.csv ({error})")
        else:
            _finite_numeric(integrals, "flow_integrals.csv", failures)
            if "time" not in integrals or integrals.empty:
                failures.append("flow integrals have no time history")
            else:
                if integrals["time"].iloc[-1] < expected_end - expected_dt:
                    failures.append("flow integrals do not reach the configured end time")
                if "vortex_strength_magnitude_sum" in integrals:
                    strength = integrals["vortex_strength_magnitude_sum"].to_numpy(float)
                    if np.any(strength <= 0.0) or strength[-1] > 10.0 * strength[0]:
                        failures.append("wake circulation history is nonphysical or unbounded")

    if not list(SAMPLES_DIR.glob("wake_*")):
        failures.append("no wake-plane samples found")

    if not args.pre_plot:
        for extension in ("png", "pdf"):
            for name in ("delta_wing_forces", "delta_wing_circulation_history"):
                figure = FIGURES_DIR / f"{name}.{extension}"
                if not figure.is_file() or figure.stat().st_size == 0:
                    failures.append(f"missing or empty figure {figure.name}")

    if failures:
        print("\n".join(f"[FAIL] {failure}" for failure in failures))
        return 1
    print("[OK] delta_wing certification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
