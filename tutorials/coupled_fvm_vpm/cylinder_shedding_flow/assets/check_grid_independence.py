#!/usr/bin/env python3
"""Apply the cylinder reference spatial, temporal, and domain gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse_coupled_benchmark import _force_metrics, _relative, _table  # noqa: E402


METRICS = ("mean_cd", "cd_peak_to_peak", "cl_first_harmonic", "strouhal")
LIMITS = {
    "mean_cd": 0.01,
    "cd_peak_to_peak": 0.02,
    "cl_first_harmonic": 0.01,
    "strouhal": 0.005,
}


def _forces(path: Path) -> dict[str, np.ndarray]:
    if path.is_dir():
        path = path / "forces_history.csv"
    return _table(path)


def _differences(candidate: dict, reference: dict) -> dict[str, float]:
    return {name: _relative(float(candidate[name]), float(reference[name])) for name in METRICS}


def _spatial_convergence(g0: dict, g1: dict, g2: dict) -> dict:
    result = {}
    for name in METRICS:
        coarse = float(g0[name])
        medium = float(g1[name])
        fine = float(g2[name])
        coarse_change = _relative(medium, coarse)
        fine_change = _relative(fine, medium)
        numerator = abs(coarse - medium)
        denominator = abs(medium - fine)
        observed_order = (
            float(np.log(numerator / denominator) / np.log(2.0))
            if numerator > 0.0 and denominator > 0.0
            else float("nan")
        )
        monotone = bool((medium - coarse) * (fine - medium) > 0.0)
        gci = float("nan")
        if monotone and np.isfinite(observed_order) and observed_order > 0.0:
            gci = float(
                1.25 * abs(fine - medium) / max(abs(fine), 1.0e-30) / (2.0**observed_order - 1.0)
            )
        result[name] = {
            "g0": coarse,
            "g1": medium,
            "g2": fine,
            "g0_to_g1_relative_change": coarse_change,
            "g1_to_g2_relative_change": fine_change,
            "changes_decrease": bool(fine_change < coarse_change),
            "monotone": monotone,
            "observed_order": observed_order,
            "fine_grid_gci": gci,
            "limit": LIMITS[name],
            "passed": bool(fine_change < LIMITS[name] and fine_change < coarse_change),
        }
    return result


def _strict_json(value):
    if isinstance(value, dict):
        return {key: _strict_json(item) for key, item in value.items()}
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g0", type=Path, required=True)
    parser.add_argument("--g1", type=Path, required=True)
    parser.add_argument("--g2", type=Path, required=True)
    parser.add_argument("--half-dt", type=Path, required=True)
    parser.add_argument("--large-domain", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "reference_flow"
        / "solution"
        / "grid_independence.json",
    )
    args = parser.parse_args()
    tables = {
        "g0": _forces(args.g0),
        "g1": _forces(args.g1),
        "g2": _forces(args.g2),
        "half_dt": _forces(args.half_dt),
        "large_domain": _forces(args.large_domain),
    }
    common_end = min(float(np.max(table["time"])) for table in tables.values())
    if common_end < 30.0:
        raise SystemExit(f"Grid-independence histories end at t={common_end:g}; require t>=30")
    start = max(30.0, common_end - 30.0)
    metrics = {name: _force_metrics(table, start) for name, table in tables.items()}
    spatial = _spatial_convergence(metrics["g0"], metrics["g1"], metrics["g2"])
    temporal = _differences(metrics["g1"], metrics["half_dt"])
    domain = _differences(metrics["g1"], metrics["large_domain"])
    temporal_pass = all(temporal[name] < LIMITS[name] for name in METRICS)
    domain_pass = all(domain[name] < LIMITS[name] for name in METRICS)
    passed = all(record["passed"] for record in spatial.values()) and temporal_pass and domain_pass
    report = _strict_json(
        {
            "schema": "openonda-cylinder-reference-independence/1",
            "status": "passed" if passed else "failed",
            "common_window": {"start": start, "end": common_end},
            "limits": LIMITS,
            "metrics": metrics,
            "spatial": spatial,
            "half_time_step_relative_change": temporal,
            "large_domain_relative_change": domain,
            "half_time_step_passed": temporal_pass,
            "large_domain_passed": domain_pass,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit("Reference grid/time/domain independence gate failed")


if __name__ == "__main__":
    main()
