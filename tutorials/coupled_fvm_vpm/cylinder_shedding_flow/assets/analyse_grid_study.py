#!/usr/bin/env python3
"""Compare common-window Cd/Cl statistics for the four cylinder grid cases.

The script deliberately reports measured changes rather than declaring a grid
independent result from a single snapshot.  It rejects non-finite, duplicate,
or too-short histories, then writes a machine-readable JSON report and a
compact Markdown table beside the case outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CASES = ("very_coarse", "coarse", "medium", "fine")
EPSILON = 1.0e-14


def _load_force_history(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise ValueError(f"Missing force history: {path}")
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    rows = [row for row in rows if row.get("patch", "cylinder") == "cylinder"]
    if len(rows) < 4:
        raise ValueError(f"Need at least four cylinder force samples in {path}")
    required = ("time", "drag_coefficient", "lift_coefficient")
    if any(name not in rows[0] for name in required):
        raise ValueError(f"Force history {path} is missing one of {required}")
    values = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        for name in required
    }
    if not all(np.all(np.isfinite(value)) for value in values.values()):
        raise ValueError(f"Force history {path} contains non-finite values")
    order = np.argsort(values["time"])
    values = {name: value[order] for name, value in values.items()}
    if np.any(np.diff(values["time"]) <= 0.0):
        raise ValueError(f"Force history {path} has duplicate or non-increasing times")
    return values


def _time_mean(time: np.ndarray, values: np.ndarray) -> float:
    duration = float(time[-1] - time[0])
    if duration <= 0.0:
        raise ValueError("A statistics window must have positive duration")
    return float(np.trapezoid(values, time) / duration)


def _rms_fluctuation(time: np.ndarray, values: np.ndarray, mean: float) -> float:
    return float(np.sqrt(_time_mean(time, (values - mean) ** 2)))


def _strouhal(time: np.ndarray, lift: np.ndarray) -> float | None:
    centred = lift - _time_mean(time, lift)
    rising = np.flatnonzero((centred[:-1] <= 0.0) & (centred[1:] > 0.0))
    if rising.size < 3:
        return None
    crossing_times = []
    for index in rising:
        fraction = -centred[index] / (centred[index + 1] - centred[index])
        crossing_times.append(time[index] + fraction * (time[index + 1] - time[index]))
    periods = np.diff(crossing_times)
    periods = periods[np.isfinite(periods) & (periods > 0.0)]
    if periods.size < 2:
        return None
    return float(1.0 / np.median(periods))


def _window_metrics(history: dict[str, np.ndarray], start: float, end: float) -> dict[str, Any]:
    mask = (history["time"] >= start - 1.0e-12) & (history["time"] <= end + 1.0e-12)
    time = history["time"][mask]
    drag = history["drag_coefficient"][mask]
    lift = history["lift_coefficient"][mask]
    if time.size < 4 or float(time[-1] - time[0]) < 0.9 * (end - start):
        raise ValueError("Insufficient samples in the common statistics window")
    mean_drag = _time_mean(time, drag)
    mean_lift = _time_mean(time, lift)
    return {
        "samples": int(time.size),
        "sample_interval": float(np.median(np.diff(time))),
        "mean_cd": mean_drag,
        "cd_rms": _rms_fluctuation(time, drag, mean_drag),
        "cd_peak_to_peak": float(np.ptp(drag)),
        "mean_cl": mean_lift,
        "cl_rms": _rms_fluctuation(time, lift, mean_lift),
        "cl_amplitude": 0.5 * float(np.ptp(lift)),
        "strouhal": _strouhal(time, lift),
    }


def _relative_change(coarser: float | None, finer: float | None) -> float | None:
    if coarser is None or finer is None:
        return None
    return float(abs(finer - coarser) / max(abs(finer), EPSILON))


def _load_mesh_contract(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Missing case metadata: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    study = payload.get("mesh", {}).get("grid_study")
    if not isinstance(study, dict):
        raise ValueError(f"Metadata {path} does not describe a grid-study mesh")
    wall = float(study["wall_dx"])
    expected = {"near_body_dx": 2.0 * wall, "wake_dx": 4.0 * wall, "far_field_dx": 12.0 * wall}
    for name, value in expected.items():
        if not np.isclose(float(study[name]), value, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"Metadata {path} violates the {name} grid-study contract")
    return {
        "wall_dx": wall,
        "near_body_dx": float(study["near_body_dx"]),
        "wake_dx": float(study["wake_dx"]),
        "far_field_dx": float(study["far_field_dx"]),
        "cell_count": int(payload.get("mesh", {}).get("cell_count", 0)),
    }


def build_report(reference_dir: Path, cases: tuple[str, ...], window: float) -> dict[str, Any]:
    histories = {}
    mesh_contracts = {}
    for case in cases:
        histories[case] = _load_force_history(reference_dir / "samples" / case / "forces_history.csv")
        mesh_contracts[case] = _load_mesh_contract(
            reference_dir / "solution" / case / "benchmark_metadata.json"
        )
    common_end = min(float(history["time"][-1]) for history in histories.values())
    if common_end < window:
        raise ValueError(
            f"All cases must reach at least t={window:g}; their common final time is {common_end:g}"
        )
    start = common_end - window
    metrics = {
        case: _window_metrics(history, start, common_end) for case, history in histories.items()
    }
    comparisons = []
    metric_names = ("mean_cd", "cd_rms", "cd_peak_to_peak", "cl_rms", "cl_amplitude", "strouhal")
    for coarser, finer in zip(cases[:-1], cases[1:], strict=True):
        comparisons.append(
            {
                "coarser": coarser,
                "finer": finer,
                "relative_change": {
                    name: _relative_change(metrics[coarser][name], metrics[finer][name])
                    for name in metric_names
                },
            }
        )
    return {
        "schema": "openonda-cylinder-grid-study/1",
        "status": "evidence_ready",
        "common_statistics_window": {"start": start, "end": common_end, "duration": window},
        "cases": {
            case: {"mesh": mesh_contracts[case], "force_statistics": metrics[case]}
            for case in cases
        },
        "sequential_relative_changes": comparisons,
        "notes": [
            "Cd and Cl statistics are time-weighted over the same final window.",
            "A grid-independent conclusion requires the user to review the measured sequential changes and waveform quality.",
        ],
    }


def _format(value: float | None, digits: int = 5) -> str:
    return "n/a" if value is None else f"{value:.{digits}g}"


def _markdown(report: dict[str, Any], cases: tuple[str, ...]) -> str:
    window = report["common_statistics_window"]
    lines = [
        "# Cylinder grid-study force comparison",
        "",
        f"Common statistics window: t={window['start']:.3f} to {window['end']:.3f}.",
        "",
        "| case | wall dx | cells | mean Cd | Cd rms | Cl rms | Cl amplitude | St | force dt |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in cases:
        record = report["cases"][case]
        mesh = record["mesh"]
        stats = record["force_statistics"]
        lines.append(
            "| {case} | {dx:.6g} | {cells:,} | {mean_cd:.6g} | {cd_rms:.6g} | "
            "{cl_rms:.6g} | {cl_amplitude:.6g} | {st} | {dt:.6g} |".format(
                case=case,
                dx=mesh["wall_dx"],
                cells=mesh["cell_count"],
                mean_cd=stats["mean_cd"],
                cd_rms=stats["cd_rms"],
                cl_rms=stats["cl_rms"],
                cl_amplitude=stats["cl_amplitude"],
                st=_format(stats["strouhal"]),
                dt=stats["sample_interval"],
            )
        )
    lines.extend(["", "| pair | Δmean Cd | ΔCd rms | ΔCl rms | ΔCl amplitude | ΔSt |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for comparison in report["sequential_relative_changes"]:
        change = comparison["relative_change"]
        lines.append(
            "| {coarser} → {finer} | {mean_cd} | {cd_rms} | {cl_rms} | {cl_amplitude} | {st} |".format(
                coarser=comparison["coarser"],
                finer=comparison["finer"],
                mean_cd=_format(change["mean_cd"], 4),
                cd_rms=_format(change["cd_rms"], 4),
                cl_rms=_format(change["cl_rms"], 4),
                cl_amplitude=_format(change["cl_amplitude"], 4),
                st=_format(change["strouhal"], 4),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "reference_flow",
    )
    parser.add_argument("--window", type=float, default=30.0)
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.window <= 0.0:
        raise SystemExit("--window must be positive")
    reference_dir = arguments.reference_dir.resolve()
    cases = tuple(arguments.cases)
    output = arguments.output or reference_dir / "solution" / "grid_study_report.json"
    report = build_report(reference_dir, cases, arguments.window)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    markdown_path = output.with_suffix(".md")
    markdown_path.write_text(_markdown(report, cases), encoding="utf-8")
    print(f"Wrote {output}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
