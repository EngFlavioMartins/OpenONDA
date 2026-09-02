#!/usr/bin/env python3
"""Compute force statistics and formal spatial convergence for the cylinder grids."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

CASE_DIR = Path(__file__).resolve().parents[1]
PREFLIGHT_CASE = ("very_coarse", 1.0 / 12.0)
PRODUCTION_CASES = (
    ("coarse", 1.0 / 24.0),
    ("medium", 1.0 / 36.0),
    ("fine", 1.0 / 54.0),
)
CASES = (PREFLIGHT_CASE, *PRODUCTION_CASES)
REFINEMENT_RATIO = 1.5
STATISTICS_WINDOW = 30.0
CONVERGENCE_TOLERANCE_PERCENT = {
    "mean_cd": 1.0,
    "cd_rms": 2.0,
    "cd_peak_to_peak": 2.0,
    "cl_rms": 2.0,
    "cl_amplitude": 2.0,
    "strouhal": 1.0,
}


def force_history(case_name: str) -> dict[str, np.ndarray]:
    path = CASE_DIR / "samples" / case_name / "forces_history.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        raise ValueError(f"No force samples in {path}")
    data = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        for name in ("time", "drag_coefficient", "lift_coefficient")
    }
    if not all(np.all(np.isfinite(values)) for values in data.values()):
        raise ValueError(f"Non-finite force data in {path}")
    if np.any(np.diff(data["time"]) <= 0.0):
        raise ValueError(f"Force times are not strictly increasing in {path}")
    return data


def time_mean(time: np.ndarray, values: np.ndarray) -> float:
    return float(np.trapezoid(values, time) / (time[-1] - time[0]))


def strouhal_number(time: np.ndarray, lift: np.ndarray) -> float:
    centred = lift - time_mean(time, lift)
    indices = np.flatnonzero((centred[:-1] <= 0.0) & (centred[1:] > 0.0))
    crossings = []
    for index in indices:
        fraction = -centred[index] / (centred[index + 1] - centred[index])
        crossings.append(time[index] + fraction * (time[index + 1] - time[index]))
    periods = np.diff(crossings)
    if len(periods) < 2:
        raise ValueError("Fewer than three rising lift zero-crossings in the statistics window")
    return float(1.0 / np.median(periods))


def statistics(history: dict[str, np.ndarray], start: float, end: float) -> dict[str, float]:
    keep = (history["time"] >= start - 1.0e-12) & (history["time"] <= end + 1.0e-12)
    time = history["time"][keep]
    drag = history["drag_coefficient"][keep]
    lift = history["lift_coefficient"][keep]
    if len(time) < 4 or time[-1] - time[0] < 0.99 * (end - start):
        raise ValueError("Incomplete common force-statistics window")
    mean_drag = time_mean(time, drag)
    mean_lift = time_mean(time, lift)
    return {
        "samples": len(time),
        "mean_cd": mean_drag,
        "cd_rms": float(np.sqrt(time_mean(time, (drag - mean_drag) ** 2))),
        "cd_peak_to_peak": float(np.ptp(drag)),
        "mean_cl": mean_lift,
        "cl_rms": float(np.sqrt(time_mean(time, (lift - mean_lift) ** 2))),
        "cl_amplitude": 0.5 * float(np.ptp(lift)),
        "strouhal": strouhal_number(time, lift),
    }


def richardson_gci(records: list[dict], metric: str, tolerance_percent: float) -> dict:
    """Return observed order, Richardson limit, and fine-grid GCI for three grids."""
    if len(records) != 3:
        raise ValueError("Richardson/GCI analysis requires exactly three production grids")
    spacing = np.asarray([record["dx"] for record in records], dtype=np.float64)
    ratios = spacing[:-1] / spacing[1:]
    if not np.allclose(ratios, REFINEMENT_RATIO, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            f"Production grids must have constant refinement ratio {REFINEMENT_RATIO:g}; "
            f"received {ratios.tolist()}"
        )
    values = np.asarray([record[metric] for record in records], dtype=np.float64)
    coarse_medium = values[0] - values[1]
    medium_fine = values[1] - values[2]
    scale = max(float(np.max(np.abs(values))), 1.0)
    roundoff = 1.0e-12 * scale
    base = {
        "metric": metric,
        "grids": [record["case"] for record in records],
        "refinement_ratio": REFINEMENT_RATIO,
        "tolerance_percent": tolerance_percent,
        "monotone": bool(coarse_medium * medium_fine > 0.0),
    }
    if abs(coarse_medium) <= roundoff and abs(medium_fine) <= roundoff:
        return {
            **base,
            "status": "converged_to_roundoff",
            "monotone": True,
            "observed_order": None,
            "richardson_extrapolated_value": float(values[2]),
            "fine_grid_relative_change_percent": 0.0,
            "fine_grid_gci_percent": 0.0,
            "asymptotic_ratio": None,
            "passed": True,
        }
    if coarse_medium * medium_fine <= 0.0:
        return {
            **base,
            "status": "oscillatory_or_non_monotone",
            "observed_order": None,
            "richardson_extrapolated_value": None,
            "fine_grid_relative_change_percent": (
                100.0 * abs(medium_fine) / max(abs(float(values[2])), 1.0e-14)
            ),
            "fine_grid_gci_percent": None,
            "asymptotic_ratio": None,
            "passed": False,
        }

    observed_order = float(
        np.log(abs(coarse_medium / medium_fine)) / np.log(REFINEMENT_RATIO)
    )
    if not np.isfinite(observed_order) or observed_order <= 0.0:
        return {
            **base,
            "status": "not_in_asymptotic_range",
            "observed_order": observed_order if np.isfinite(observed_order) else None,
            "richardson_extrapolated_value": None,
            "fine_grid_relative_change_percent": (
                100.0 * abs(medium_fine) / max(abs(float(values[2])), 1.0e-14)
            ),
            "fine_grid_gci_percent": None,
            "asymptotic_ratio": None,
            "passed": False,
        }

    denominator = REFINEMENT_RATIO**observed_order - 1.0
    extrapolated = float(values[2] + (values[2] - values[1]) / denominator)
    fine_change = 100.0 * abs(medium_fine) / max(abs(float(values[2])), 1.0e-14)
    safety_factor = 1.25
    fine_gci = safety_factor * fine_change / denominator
    medium_change = 100.0 * abs(coarse_medium) / max(abs(float(values[1])), 1.0e-14)
    medium_gci = safety_factor * medium_change / denominator
    asymptotic_ratio = medium_gci / max(
        REFINEMENT_RATIO**observed_order * fine_gci,
        1.0e-30,
    )
    return {
        **base,
        "status": "asymptotic",
        "observed_order": observed_order,
        "richardson_extrapolated_value": extrapolated,
        "fine_grid_relative_change_percent": fine_change,
        "fine_grid_gci_percent": fine_gci,
        "asymptotic_ratio": asymptotic_ratio,
        "passed": bool(fine_gci <= tolerance_percent),
    }


def main() -> None:
    histories = {name: force_history(name) for name, _dx in CASES}
    common_end = min(float(history["time"][-1]) for history in histories.values())
    if common_end < STATISTICS_WINDOW:
        raise ValueError(
            f"Common final time is {common_end:g}; require at least {STATISTICS_WINDOW:g}"
        )
    common_start = common_end - STATISTICS_WINDOW
    records = []
    for name, dx in CASES:
        records.append(
            {
                "case": name,
                "dx": dx,
                **statistics(histories[name], common_start, common_end),
            }
        )

    metrics = tuple(CONVERGENCE_TOLERANCE_PERCENT)
    comparisons = []
    for coarser, finer in zip(records[:-1], records[1:], strict=True):
        comparisons.append(
            {
                "coarser": coarser["case"],
                "finer": finer["case"],
                "relative_change": {
                    metric: abs(finer[metric] - coarser[metric]) / max(abs(finer[metric]), 1.0e-14)
                    for metric in metrics
                },
            }
        )

    production_names = {case for case, _dx in PRODUCTION_CASES}
    production_records = [record for record in records if record["case"] in production_names]
    convergence = {
        metric: richardson_gci(
            production_records,
            metric,
            CONVERGENCE_TOLERANCE_PERCENT[metric],
        )
        for metric in metrics
    }
    report = {
        "common_window": {"start": common_start, "end": common_end},
        "preflight_case": PREFLIGHT_CASE[0],
        "production_cases": [case for case, _dx in PRODUCTION_CASES],
        "refinement_ratio": REFINEMENT_RATIO,
        "cases": records,
        "comparisons": comparisons,
        "grid_convergence": convergence,
        "grid_independent": all(result["passed"] for result in convergence.values()),
    }
    output = CASE_DIR / "solution" / "grid_study.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    for metric, result in convergence.items():
        order = result["observed_order"]
        gci = result["fine_grid_gci_percent"]
        detail = (
            f"p={order:.3f}, GCI_fine={gci:.3f}%"
            if order is not None and gci is not None
            else result["status"]
        )
        print(f"  {metric}: {detail}, passed={result['passed']}")
    print(f"Grid-independent at configured tolerances: {report['grid_independent']}")


if __name__ == "__main__":
    main()
