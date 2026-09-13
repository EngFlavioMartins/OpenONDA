#!/usr/bin/env python3
"""Compare every completed cube reference grid without touching raw results.

By default, the report measures differences relative to the finest *available*
grid. Such differences are useful convergence indicators, but they are not
exact discretisation errors. A Richardson extrapolation and GCI are emitted
only when the three finest distinct mesh levels satisfy their basic assumptions.

Examples
--------
Run after one or more completed reference cases::

    python -u postprocess_grid_study.py

Use a known post-transient common statistics interval::

    python -u postprocess_grid_study.py --statistics-start 10 --statistics-end 20

Only derived ``grid_convergence.*`` files below ``solution/`` are written.  No
field, mesh, sample, backup, or case directory is removed or overwritten.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


CASE_DIR = Path(__file__).resolve().parent
TIME_TOLERANCE = 1.0e-10
VALUE_FLOOR = 1.0e-14
FORCE_COLUMNS = (
    "drag_coefficient",
    "lift_coefficient",
    "side_force_coefficient",
)
METRICS = (
    "mean_drag",
    "rms_drag",
    "mean_lift",
    "rms_lift",
    "mean_side",
    "rms_side",
    "strouhal_lift",
    "strouhal_side",
)
PROFILE_NAMES = ("centreline", "offaxis_y075")


@dataclass(frozen=True)
class GridCase:
    """One completed grid registered by ``fvm.update_grid_study``."""

    name: str
    samples_dir: Path
    wall_cell_size: float
    cell_count: int
    declared_end_time: float


@dataclass(frozen=True)
class ForceHistory:
    """A restart-free, strictly increasing force history."""

    time: np.ndarray
    values: dict[str, np.ndarray]


@dataclass(frozen=True)
class MeanProfile:
    """A time-averaged velocity profile along one sampled line."""

    position: np.ndarray
    velocity: np.ndarray
    axis: str


def _finite_float(value: object, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive_float(value: object, label: str) -> float:
    result = _finite_float(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _positive_int(value: object, label: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _last_restart_segment(
    time: np.ndarray,
    values: dict[str, np.ndarray],
    *,
    source: Path,
    allow_duplicate_times: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Keep the final run segment and, when appropriate, collapse duplicate times."""

    resets = np.flatnonzero(np.diff(time) < -TIME_TOLERANCE)
    start = int(resets[-1]) + 1 if len(resets) else 0
    time = time[start:]
    values = {name: data[start:] for name, data in values.items()}
    if len(time) < 2:
        raise ValueError(f"{source} has fewer than two samples after its final restart")
    if allow_duplicate_times:
        if np.any(np.diff(time) < -TIME_TOLERANCE):
            raise ValueError(f"{source} has decreasing times after its final restart")
        return time, values

    keep: list[int] = []
    for index, value in enumerate(time):
        if not keep:
            keep.append(index)
            continue
        previous = time[keep[-1]]
        if value > previous + TIME_TOLERANCE:
            keep.append(index)
        elif abs(value - previous) <= TIME_TOLERANCE:
            # A restart can reproduce its terminal sample.  The later row is
            # the state that belongs to the final run segment.
            keep[-1] = index
        else:
            raise ValueError(f"{source} has decreasing times after its final restart")

    selected = np.asarray(keep, dtype=np.int64)
    time = time[selected]
    values = {name: data[selected] for name, data in values.items()}
    if len(time) < 2 or np.any(np.diff(time) <= 0.0):
        raise ValueError(f"{source} has no strictly increasing final time history")
    return time, values


def _read_force_history(path: Path) -> ForceHistory:
    """Read required force data, retaining the last complete restart segment."""

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = tuple(reader.fieldnames or ())
        if "time" not in fieldnames:
            raise ValueError(f"{path} has no 'time' column")
        available = tuple(name for name in FORCE_COLUMNS if name in fieldnames)
        if "drag_coefficient" not in available:
            raise ValueError(f"{path} has no 'drag_coefficient' column")
        rows = list(reader)

    if not rows:
        raise ValueError(f"{path} is empty")
    try:
        time = np.asarray([float(row["time"]) for row in rows], dtype=np.float64)
        values = {
            name: np.asarray([float(row[name]) for row in rows], dtype=np.float64)
            for name in available
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path} contains a non-numeric force sample") from error
    if not np.all(np.isfinite(time)) or any(
        not np.all(np.isfinite(data)) for data in values.values()
    ):
        raise ValueError(f"{path} contains non-finite force samples")
    time, values = _last_restart_segment(time, values, source=path)
    return ForceHistory(time=time, values=values)


def _discover_cases(
    samples_root: Path,
) -> tuple[list[GridCase], list[dict[str, str]]]:
    """Discover every usable registered grid, retaining exclusion diagnostics."""

    if not samples_root.is_dir():
        raise FileNotFoundError(f"samples root does not exist: {samples_root}")

    cases: list[GridCase] = []
    excluded: list[dict[str, str]] = []
    for directory in sorted(path for path in samples_root.iterdir() if path.is_dir()):
        metadata_path = directory / "grid_run.json"
        forces_path = directory / "forces_history.csv"
        if not metadata_path.is_file():
            if forces_path.is_file():
                excluded.append(
                    {
                        "directory": directory.name,
                        "reason": "forces_history.csv exists but grid_run.json is absent",
                    }
                )
            continue
        if not forces_path.is_file():
            excluded.append(
                {
                    "directory": directory.name,
                    "reason": "grid_run.json exists but forces_history.csv is absent",
                }
            )
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            name = str(metadata["case"]).strip()
            if not name:
                raise ValueError("case is empty")
            requested = _positive_float(metadata["cell_size"], "cell_size")
            cell_count = _positive_int(metadata["cell_count"], "cell_count")
            end_time = _positive_float(metadata["end_time"], "end_time")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            excluded.append({"directory": directory.name, "reason": str(error)})
            continue
        cases.append(
            GridCase(
                name=name,
                samples_dir=directory,
                wall_cell_size=requested,
                cell_count=cell_count,
                declared_end_time=end_time,
            )
        )

    duplicate_names = {
        case.name for case in cases if sum(item.name == case.name for item in cases) > 1
    }
    if duplicate_names:
        names = ", ".join(sorted(duplicate_names))
        raise ValueError(f"grid_run.json registers duplicate case names: {names}")
    cases.sort(key=lambda case: (-case.wall_cell_size, case.cell_count, case.name))
    return cases, excluded


def _resolve_statistics_window(
    histories: dict[str, ForceHistory],
    statistics_start: float | None,
    statistics_end: float | None,
) -> tuple[float, float]:
    """Return an interval shared by all histories, defaulting to its final half."""

    common_start = max(float(history.time[0]) for history in histories.values())
    common_end = min(float(history.time[-1]) for history in histories.values())
    if common_end <= common_start + TIME_TOLERANCE:
        raise ValueError("completed grid cases have no common force-history interval")
    start = (
        _finite_float(statistics_start, "statistics start")
        if statistics_start is not None
        else 0.5 * (common_start + common_end)
    )
    end = (
        _finite_float(statistics_end, "statistics end")
        if statistics_end is not None
        else common_end
    )
    if start < common_start - TIME_TOLERANCE or end > common_end + TIME_TOLERANCE:
        raise ValueError(
            f"statistics interval [{start:g}, {end:g}] lies outside the common force interval "
            f"[{common_start:g}, {common_end:g}]"
        )
    if end <= start + TIME_TOLERANCE:
        raise ValueError("statistics end must be later than statistics start")
    return start, end


def _window_series(
    time: np.ndarray,
    values: np.ndarray,
    start: float,
    end: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Clip a sampled signal to an interval, interpolating only its endpoints."""

    if start < time[0] - TIME_TOLERANCE or end > time[-1] + TIME_TOLERANCE:
        raise ValueError("requested statistics interval is outside a sampled history")
    start = max(start, float(time[0]))
    end = min(end, float(time[-1]))
    interior = (time > start + TIME_TOLERANCE) & (time < end - TIME_TOLERANCE)
    output_time = np.concatenate(([start], time[interior], [end]))
    output_values = np.concatenate(
        ([np.interp(start, time, values)], values[interior], [np.interp(end, time, values)])
    )
    if len(output_time) < 2 or output_time[-1] <= output_time[0]:
        raise ValueError("statistics interval has insufficient samples")
    return output_time, output_values


def _time_mean(time: np.ndarray, values: np.ndarray) -> float:
    return float(np.trapezoid(values, time) / (time[-1] - time[0]))


def _dominant_strouhal(time: np.ndarray, values: np.ndarray) -> float | None:
    """Estimate the dominant nondimensional frequency from a force signal."""

    if len(time) < 16:
        return None
    centred = values - _time_mean(time, values)
    if float(np.max(np.abs(centred))) <= VALUE_FLOOR:
        return None
    uniform_time = np.linspace(time[0], time[-1], len(time))
    signal = np.interp(uniform_time, time, centred)
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
    frequency = np.fft.rfftfreq(len(signal), uniform_time[1] - uniform_time[0])
    if len(spectrum) < 2 or not np.any(np.isfinite(spectrum[1:])):
        return None
    return float(frequency[1 + int(np.argmax(spectrum[1:]))])


def _force_statistics(
    history: ForceHistory, start: float, end: float
) -> dict[str, float | int | None]:
    """Compute time-weighted force statistics over a shared physical interval."""

    result: dict[str, float | int | None] = {"samples": None}
    signals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in FORCE_COLUMNS:
        if name not in history.values:
            continue
        signals[name] = _window_series(history.time, history.values[name], start, end)
    if "drag_coefficient" not in signals:
        raise ValueError("force history has no drag coefficient in the statistics interval")
    result["samples"] = int(len(signals["drag_coefficient"][0]))

    for prefix, column in (
        ("drag", "drag_coefficient"),
        ("lift", "lift_coefficient"),
        ("side", "side_force_coefficient"),
    ):
        if column not in signals:
            result[f"mean_{prefix}"] = None
            result[f"rms_{prefix}"] = None
            continue
        time, values = signals[column]
        mean = _time_mean(time, values)
        result[f"mean_{prefix}"] = mean
        result[f"rms_{prefix}"] = float(np.sqrt(_time_mean(time, (values - mean) ** 2)))

    for metric, column in (
        ("strouhal_lift", "lift_coefficient"),
        ("strouhal_side", "side_force_coefficient"),
    ):
        result[metric] = _dominant_strouhal(*signals[column]) if column in signals else None
    return result


def _read_profile(path: Path, start: float, end: float) -> MeanProfile:
    """Build a time-weighted mean vector profile over the force-statistics window."""

    required = (
        "time",
        "position_x",
        "position_y",
        "position_z",
        "velocity_x",
        "velocity_y",
        "velocity_z",
    )
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = tuple(reader.fieldnames or ())
        missing = [name for name in required if name not in fieldnames]
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} is empty")

    try:
        time = np.asarray([float(row["time"]) for row in rows], dtype=np.float64)
        coordinates = np.column_stack(
            [np.asarray([float(row[f"position_{axis}"]) for row in rows]) for axis in "xyz"]
        )
        velocity = np.column_stack(
            [np.asarray([float(row[f"velocity_{axis}"]) for row in rows]) for axis in "xyz"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path} contains non-numeric profile data") from error
    if not np.all(np.isfinite(np.column_stack((time, coordinates, velocity)))):
        raise ValueError(f"{path} contains non-finite profile data")

    time, data = _last_restart_segment(
        time,
        {
            "coordinates": coordinates,
            "velocity": velocity,
        },
        source=path,
        allow_duplicate_times=True,
    )
    coordinates = data["coordinates"]
    velocity = data["velocity"]
    selected = (time >= start - TIME_TOLERANCE) & (time <= end + TIME_TOLERANCE)
    if not np.any(selected):
        raise ValueError(f"{path} has no profile samples in the statistics window")
    axis_index = int(np.argmax(np.ptp(coordinates[selected], axis=0)))
    axis = "xyz"[axis_index]
    positions = coordinates[:, axis_index]
    rounded_positions = np.round(positions, decimals=10)
    unique_positions, inverse = np.unique(rounded_positions, return_inverse=True)

    mean_positions: list[float] = []
    mean_velocity: list[np.ndarray] = []
    for group in range(len(unique_positions)):
        indices = np.flatnonzero(inverse == group)
        group_time = time[indices]
        group_velocity = velocity[indices]
        group_time, values = _last_restart_segment(
            group_time,
            {"velocity": group_velocity},
            source=path,
        )
        group_velocity = values["velocity"]
        averaged = []
        for component in range(3):
            clipped_time, clipped_values = _window_series(
                group_time,
                group_velocity[:, component],
                start,
                end,
            )
            averaged.append(_time_mean(clipped_time, clipped_values))
        mean_positions.append(float(np.mean(positions[indices])))
        mean_velocity.append(np.asarray(averaged, dtype=np.float64))

    order = np.argsort(mean_positions)
    result_position = np.asarray(mean_positions, dtype=np.float64)[order]
    result_velocity = np.asarray(mean_velocity, dtype=np.float64)[order]
    if len(result_position) < 2 or np.any(np.diff(result_position) <= 0.0):
        raise ValueError(f"{path} has no strictly increasing sampled profile positions")
    return MeanProfile(position=result_position, velocity=result_velocity, axis=axis)


def _profile_difference(
    candidate: MeanProfile, reference: MeanProfile
) -> dict[str, float | list[float] | None]:
    """Measure a candidate mean profile relative to the reference mean profile."""

    if candidate.axis != reference.axis:
        raise ValueError(
            f"profile axes differ: candidate varies along {candidate.axis}, reference along {reference.axis}"
        )
    lower = max(float(candidate.position[0]), float(reference.position[0]))
    upper = min(float(candidate.position[-1]), float(reference.position[-1]))
    selected = (reference.position >= lower) & (reference.position <= upper)
    position = reference.position[selected]
    if len(position) < 2 or upper <= lower:
        raise ValueError("profiles have insufficient overlapping spatial support")
    interpolated = np.column_stack(
        [
            np.interp(position, candidate.position, candidate.velocity[:, component])
            for component in range(3)
        ]
    )
    reference_velocity = reference.velocity[selected]
    difference = interpolated - reference_velocity
    span = position[-1] - position[0]
    absolute_rms = float(np.sqrt(np.trapezoid(np.sum(difference**2, axis=1), position) / span))
    reference_rms = float(
        np.sqrt(np.trapezoid(np.sum(reference_velocity**2, axis=1), position) / span)
    )
    return {
        "position_interval": [float(position[0]), float(position[-1])],
        "absolute_rms_velocity_difference": absolute_rms,
        "relative_l2": absolute_rms / reference_rms if reference_rms > VALUE_FLOOR else None,
    }


def _difference(
    value: float | int | None, reference: float | int | None
) -> dict[str, float | None]:
    """Return absolute and safely normalized differences from a reference value."""

    if value is None or reference is None:
        return {"absolute": None, "relative": None}
    absolute = abs(float(value) - float(reference))
    scale = abs(float(reference))
    return {
        "absolute": absolute,
        "relative": absolute / scale if scale > VALUE_FLOOR else None,
    }


def _distinct_levels(
    grids: list[dict[str, Any]],
    metric: str,
) -> tuple[list[dict[str, Any]], list[float]]:
    """Keep metric-bearing levels with unique realized wall spacing."""

    grouped: dict[float, list[dict[str, Any]]] = {}
    for grid in grids:
        if grid["force_statistics"].get(metric) is None:
            continue
        spacing = float(grid["wall_cell_size"])
        grouped.setdefault(spacing, []).append(grid)
    duplicate_spacings = [spacing for spacing, items in grouped.items() if len(items) > 1]
    levels = [items[0] for spacing, items in grouped.items() if spacing not in duplicate_spacings]
    levels.sort(key=lambda grid: -float(grid["wall_cell_size"]))
    return levels, sorted(duplicate_spacings, reverse=True)


def _convergence_statistics(
    grids: list[dict[str, Any]],
    metric: str,
    tolerance: float | None,
) -> dict[str, Any]:
    """Compute adjacent-grid changes plus guarded Richardson/GCI diagnostics."""

    levels, duplicate_spacings = _distinct_levels(grids, metric)
    result: dict[str, Any] = {
        "unique_level_count": len(levels),
        "duplicate_wall_cell_sizes": duplicate_spacings,
        "finest_pair": None,
        "richardson": {
            "available": False,
            "reason": "fewer than three distinct levels with this metric",
            "monotone": None,
            "observed_order": None,
            "extrapolated": None,
            "fine_grid_gci": None,
        },
    }
    if len(levels) >= 2:
        coarse, fine = levels[-2:]
        coarse_value = float(coarse["force_statistics"][metric])
        fine_value = float(fine["force_statistics"][metric])
        difference = _difference(coarse_value, fine_value)
        result["finest_pair"] = {
            "coarser_case": coarse["case"],
            "finer_case": fine["case"],
            "coarser_wall_cell_size": coarse["wall_cell_size"],
            "finer_wall_cell_size": fine["wall_cell_size"],
            "absolute_change": difference["absolute"],
            "relative_change": difference["relative"],
            "meets_tolerance": (
                None
                if tolerance is None or difference["relative"] is None
                else bool(difference["relative"] <= tolerance)
            ),
        }

    if len(levels) < 3:
        return result
    coarse, medium, fine = levels[-3:]
    h_coarse = float(coarse["wall_cell_size"])
    h_medium = float(medium["wall_cell_size"])
    h_fine = float(fine["wall_cell_size"])
    ratio_coarse = h_coarse / h_medium
    ratio_fine = h_medium / h_fine
    richardson = result["richardson"]
    richardson["coarse_to_medium_ratio"] = ratio_coarse
    richardson["medium_to_fine_ratio"] = ratio_fine
    if not np.isclose(ratio_coarse, ratio_fine, rtol=5.0e-3, atol=1.0e-12):
        richardson["reason"] = "the three finest ratios are not equal"
        return result

    coarse_value = float(coarse["force_statistics"][metric])
    medium_value = float(medium["force_statistics"][metric])
    fine_value = float(fine["force_statistics"][metric])
    delta_coarse = medium_value - coarse_value
    delta_fine = fine_value - medium_value
    monotone = bool(delta_coarse * delta_fine > 0.0)
    richardson["monotone"] = monotone
    if not monotone:
        richardson["reason"] = "the three finest values are not monotone"
        return result
    if abs(delta_coarse) <= VALUE_FLOOR or abs(delta_fine) <= VALUE_FLOOR:
        richardson["reason"] = "an adjacent difference is too small to estimate an order"
        return result

    observed_order = math.log(abs(delta_coarse / delta_fine)) / math.log(ratio_fine)
    if not math.isfinite(observed_order) or observed_order <= 0.0:
        richardson["reason"] = "the observed order is not positive and finite"
        return result
    extrapolated = fine_value + delta_fine / (ratio_fine**observed_order - 1.0)
    scale = max(abs(extrapolated), VALUE_FLOOR)
    richardson.update(
        {
            "available": True,
            "reason": None,
            "observed_order": observed_order,
            "extrapolated": extrapolated,
            "fine_grid_gci": 1.25 * abs(extrapolated - fine_value) / scale,
        }
    )
    return result


def _cell_cost_statistics(grids: list[dict[str, Any]]) -> list[dict[str, float | str]]:
    """Describe the observed mesh-cost growth for adjacent unique levels."""

    grouped: dict[float, list[dict[str, Any]]] = {}
    for grid in grids:
        grouped.setdefault(float(grid["wall_cell_size"]), []).append(grid)
    levels = [items[0] for spacing, items in grouped.items() if len(items) == 1]
    levels.sort(key=lambda grid: -float(grid["wall_cell_size"]))
    pairs: list[dict[str, float | str]] = []
    for coarse, fine in zip(levels, levels[1:], strict=False):
        h_ratio = float(coarse["wall_cell_size"]) / float(fine["wall_cell_size"])
        cell_ratio = float(fine["cell_count"]) / float(coarse["cell_count"])
        effective_dimension = math.log(cell_ratio) / math.log(h_ratio)
        pairs.append(
            {
                "coarser_case": str(coarse["case"]),
                "finer_case": str(fine["case"]),
                "wall_cell_size_ratio": h_ratio,
                "cell_count_ratio": cell_ratio,
                "observed_cell_count_exponent": effective_dimension,
            }
        )
    return pairs


def _profile_statistics(
    cases: list[GridCase],
    grids: list[dict[str, Any]],
    reference_case: str,
    start: float,
    end: float,
) -> dict[str, Any]:
    """Compare all available time-mean sampled profiles with the finest reference."""

    case_by_name = {case.name: case for case in cases}
    profiles: dict[str, Any] = {}
    for name in PROFILE_NAMES:
        reference_path = case_by_name[reference_case].samples_dir / f"{name}.csv"
        if not reference_path.is_file():
            profiles[name] = {
                "available": False,
                "reason": f"reference profile is absent: {reference_path.name}",
                "comparisons": {},
            }
            continue
        try:
            reference = _read_profile(reference_path, start, end)
        except ValueError as error:
            profiles[name] = {
                "available": False,
                "reason": str(error),
                "comparisons": {},
            }
            continue

        comparisons: dict[str, Any] = {}
        for case in cases:
            path = case.samples_dir / f"{name}.csv"
            if not path.is_file():
                comparisons[case.name] = {"available": False, "reason": "profile file is absent"}
                continue
            try:
                candidate = _read_profile(path, start, end)
                comparison = _profile_difference(candidate, reference)
            except ValueError as error:
                comparisons[case.name] = {"available": False, "reason": str(error)}
                continue
            comparisons[case.name] = {"available": True, **comparison}
        profiles[name] = {
            "available": True,
            "axis": reference.axis,
            "comparisons": comparisons,
        }

    by_name = {grid["case"]: grid for grid in grids}
    for case in cases:
        by_name[case.name]["profile_differences_to_reference"] = {
            name: profiles[name]["comparisons"].get(case.name)
            for name in PROFILE_NAMES
            if profiles[name].get("available")
        }
    return profiles


def _display_number(value: object, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}g}"


def _display_percent(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.3g}%"


def _write_csv(report: dict[str, Any], destination: Path) -> None:
    profile_columns = [f"{name}_relative_l2" for name in PROFILE_NAMES]
    difference_columns = [f"difference_to_reference_{metric}" for metric in METRICS]
    fieldnames = [
        "case",
        "wall_cell_size",
        "cell_count",
        "declared_end_time",
        "force_samples",
        *METRICS,
        *difference_columns,
        *profile_columns,
    ]
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for grid in report["grids"]:
            statistics = grid["force_statistics"]
            differences = grid["difference_to_reference"]
            profile_differences = grid.get("profile_differences_to_reference", {})
            row = {
                "case": grid["case"],
                "wall_cell_size": grid["wall_cell_size"],
                "cell_count": grid["cell_count"],
                "declared_end_time": grid["declared_end_time"],
                "force_samples": statistics["samples"],
                **{metric: statistics.get(metric) for metric in METRICS},
                **{
                    f"difference_to_reference_{metric}": differences[metric]["relative"]
                    for metric in METRICS
                },
                **{
                    f"{name}_relative_l2": (
                        profile_differences.get(name, {}).get("relative_l2")
                        if profile_differences.get(name, {}).get("available")
                        else None
                    )
                    for name in PROFILE_NAMES
                },
            }
            writer.writerow(row)


def _write_markdown(report: dict[str, Any], destination: Path) -> None:
    window = report["statistics_window"]
    lines = [
        "# Cube grid-convergence report",
        "",
        f"Common statistics window: `{window['start']:g} <= t <= {window['end']:g}`.",
        f"Reference grid: `{report['reference_case']}` ({report['reference_definition']}).",
        "",
        "Differences are relative to that grid, not exact discretisation errors.",
        "",
        "| Case | Wall h/D | Cells | Mean Cd | RMS Cd | RMS Cl | RMS Cs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for grid in report["grids"]:
        force = grid["force_statistics"]
        lines.append(
            f"| {grid['case']} | {_display_number(grid['wall_cell_size'])} | "
            f"{grid['cell_count']:,} | "
            f"{_display_number(force['mean_drag'])} | {_display_number(force['rms_drag'])} | "
            f"{_display_number(force['rms_lift'])} | {_display_number(force['rms_side'])} |"
        )

    lines.extend(
        [
            "",
            "## Finest-pair and Richardson diagnostics",
            "",
            "| Metric | Finest-pair change | Observed order | Fine-grid GCI |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric in METRICS:
        convergence = report["convergence"][metric]
        pair = convergence["finest_pair"] or {}
        richardson = convergence["richardson"]
        lines.append(
            f"| {metric} | {_display_percent(pair.get('relative_change'))} | "
            f"{_display_number(richardson['observed_order'])} | "
            f"{_display_percent(richardson['fine_grid_gci'])} |"
        )

    largest = report["assessment"].get("largest_finest_pair_relative_change")
    if largest is not None:
        lines.extend(
            [
                "",
                "The largest finite force-statistic change between the two finest levels is "
                f"`{_display_percent(largest['relative_change'])}` for `{largest['metric']}`.",
            ]
        )
    if report["excluded_cases"]:
        lines.extend(["", "## Excluded directories", ""])
        for item in report["excluded_cases"]:
            lines.append(f"- `{item['directory']}`: {item['reason']}")
    lines.append("")
    destination.write_text("\n".join(lines), encoding="utf-8")


def _spacing_axis(axis, label: str = "Wall cell size, h/D") -> None:
    from matplotlib.ticker import FuncFormatter

    axis.set_xscale("log", base=2)
    axis.invert_xaxis()
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:.5g}"))
    axis.set_xlabel(label)
    axis.grid(alpha=0.25)


def _plot_metric(axis, grids: list[dict[str, Any]], metric: str, label: str, x: np.ndarray) -> None:
    values = np.asarray(
        [
            np.nan
            if grid["force_statistics"].get(metric) is None
            else grid["force_statistics"][metric]
            for grid in grids
        ],
        dtype=np.float64,
    )
    if np.any(np.isfinite(values)):
        axis.plot(x, values, "o-", linewidth=1.4, label=label)


def _relative_difference_values(grids: list[dict[str, Any]], metric: str) -> np.ndarray:
    """Return plot-ready differences, omitting exact/roundoff-level reference points."""

    values = np.asarray(
        [grid["difference_to_reference"][metric]["relative"] for grid in grids],
        dtype=np.float64,
    )
    return np.where(values > VALUE_FLOOR, values, np.nan)


def _plot_by_spacing(report: dict[str, Any], destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grids = report["grids"]
    spacing = np.asarray([grid["wall_cell_size"] for grid in grids], dtype=np.float64)
    cells = np.asarray([grid["cell_count"] for grid in grids], dtype=np.float64)
    figure, axes = plt.subplots(2, 3, figsize=(13.0, 7.6), constrained_layout=True)

    for metric, label in (
        ("mean_drag", "Mean Cd"),
        ("mean_lift", "Mean Cl"),
        ("mean_side", "Mean Cs"),
    ):
        _plot_metric(axes[0, 0], grids, metric, label, spacing)
    axes[0, 0].set_ylabel("Mean force coefficient")
    axes[0, 0].legend(fontsize=8)
    _spacing_axis(axes[0, 0])

    for metric, label in (("rms_drag", "RMS Cd"), ("rms_lift", "RMS Cl"), ("rms_side", "RMS Cs")):
        _plot_metric(axes[0, 1], grids, metric, label, spacing)
    axes[0, 1].set_ylabel("RMS force coefficient")
    axes[0, 1].legend(fontsize=8)
    _spacing_axis(axes[0, 1])

    for metric, label in (("strouhal_lift", "Lift"), ("strouhal_side", "Side")):
        _plot_metric(axes[0, 2], grids, metric, label, spacing)
    axes[0, 2].set_ylabel("Dominant St")
    axes[0, 2].legend(fontsize=8)
    _spacing_axis(axes[0, 2])

    axes[1, 0].plot(spacing, cells, "o-", linewidth=1.4, color="#6a3d9a")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylabel("Fluid cells")
    _spacing_axis(axes[1, 0])

    positive_differences = []
    for metric in ("mean_drag", "rms_drag", "rms_lift", "rms_side", "strouhal_lift"):
        values = _relative_difference_values(grids, metric)
        positive_differences.extend(value for value in values if np.isfinite(value))
        axes[1, 1].plot(spacing, values, "o-", linewidth=1.2, label=metric)
    if positive_differences:
        axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylabel("Difference from reference grid")
    axes[1, 1].legend(fontsize=7)
    _spacing_axis(axes[1, 1])

    profile_plotted = False
    for name, profile in report["profiles"].items():
        if not profile.get("available"):
            continue
        values = np.asarray(
            [
                profile["comparisons"].get(grid["case"], {}).get("relative_l2", np.nan)
                for grid in grids
            ],
            dtype=np.float64,
        )
        axes[1, 2].plot(spacing, values, "o-", linewidth=1.4, label=name)
        profile_plotted = True
    if profile_plotted:
        axes[1, 2].set_ylabel("Profile L2 difference from reference")
        axes[1, 2].legend(fontsize=8)
    else:
        axes[1, 2].text(0.5, 0.5, "No common profile data", ha="center", va="center")
    _spacing_axis(axes[1, 2])

    window = report["statistics_window"]
    figure.suptitle(
        f"Cube grid convergence: common statistics window {window['start']:g} ≤ t ≤ {window['end']:g}"
    )
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _plot_by_cells(report: dict[str, Any], destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grids = sorted(report["grids"], key=lambda grid: grid["cell_count"])
    cells = np.asarray([grid["cell_count"] for grid in grids], dtype=np.float64)
    figure, axes = plt.subplots(2, 2, figsize=(9.8, 7.0), constrained_layout=True)

    for metric, label in (
        ("mean_drag", "Mean Cd"),
        ("mean_lift", "Mean Cl"),
        ("mean_side", "Mean Cs"),
    ):
        _plot_metric(axes[0, 0], grids, metric, label, cells)
    axes[0, 0].set_ylabel("Mean force coefficient")
    axes[0, 0].legend(fontsize=8)

    for metric, label in (("rms_drag", "RMS Cd"), ("rms_lift", "RMS Cl"), ("rms_side", "RMS Cs")):
        _plot_metric(axes[0, 1], grids, metric, label, cells)
    axes[0, 1].set_ylabel("RMS force coefficient")
    axes[0, 1].legend(fontsize=8)

    for metric, label in (("strouhal_lift", "Lift"), ("strouhal_side", "Side")):
        _plot_metric(axes[1, 0], grids, metric, label, cells)
    axes[1, 0].set_ylabel("Dominant St")
    axes[1, 0].legend(fontsize=8)

    positive_differences = []
    for metric in ("mean_drag", "rms_drag", "rms_lift", "rms_side", "strouhal_lift"):
        values = _relative_difference_values(grids, metric)
        positive_differences.extend(value for value in values if np.isfinite(value))
        axes[1, 1].plot(cells, values, "o-", linewidth=1.2, label=metric)
    if positive_differences:
        axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylabel("Difference from reference grid")
    axes[1, 1].legend(fontsize=7)

    for axis in axes.flat:
        axis.set_xscale("log")
        axis.set_xlabel("Fluid cells")
        axis.grid(alpha=0.25)
    figure.suptitle("Cube grid convergence by fluid-cell count")
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _plot_profiles(report: dict[str, Any], destination: Path) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    available = [
        (name, profile) for name, profile in report["profiles"].items() if profile.get("available")
    ]
    if not available:
        figure, axis = plt.subplots(figsize=(5.0, 3.6), constrained_layout=True)
        axis.text(0.5, 0.5, "No common profile data", ha="center", va="center")
        axis.set_axis_off()
        figure.savefig(destination, dpi=180)
        plt.close(figure)
        return False
    grids = report["grids"]
    spacing = np.asarray([grid["wall_cell_size"] for grid in grids], dtype=np.float64)
    figure, axes = plt.subplots(
        1, len(available), figsize=(5.0 * len(available), 3.6), squeeze=False
    )
    for axis, (name, profile) in zip(axes.flat, available, strict=True):
        values = np.asarray(
            [
                profile["comparisons"].get(grid["case"], {}).get("relative_l2", np.nan)
                for grid in grids
            ],
            dtype=np.float64,
        )
        axis.plot(spacing, values, "o-", color="#1b9e77", linewidth=1.4)
        axis.set_title(name)
        axis.set_ylabel("Relative L2 difference")
        _spacing_axis(axis)
    figure.suptitle("Time-mean velocity-profile differences from the reference grid")
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return True


def analyse_grid_convergence(
    samples_root: str | Path,
    output_dir: str | Path,
    *,
    statistics_start: float | None = None,
    statistics_end: float | None = None,
    reference_case: str | None = None,
    tolerance: float | None = None,
) -> dict[str, Any]:
    """Analyse every completed cube grid and write derived convergence artifacts.

    The function only reads the registered case directories below
    ``samples_root``.  It creates or replaces the named report files in
    ``output_dir``; it never changes individual grid outputs.
    """

    samples_root = Path(samples_root)
    output_dir = Path(output_dir)
    if tolerance is not None:
        tolerance = _positive_float(tolerance, "convergence tolerance")
    cases, excluded = _discover_cases(samples_root)
    histories: dict[str, ForceHistory] = {}
    usable_cases: list[GridCase] = []
    for case in cases:
        try:
            histories[case.name] = _read_force_history(case.samples_dir / "forces_history.csv")
        except ValueError as error:
            excluded.append({"directory": case.samples_dir.name, "reason": str(error)})
            continue
        usable_cases.append(case)
    cases = usable_cases
    if len(cases) < 2:
        raise ValueError(
            "at least two completed grid cases with valid force histories are required"
        )

    if reference_case is None:
        reference = min(cases, key=lambda case: (case.wall_cell_size, -case.cell_count, case.name))
    else:
        matches = [case for case in cases if case.name == reference_case]
        if not matches:
            raise ValueError(f"reference case {reference_case!r} is not a valid completed grid")
        reference = matches[0]

    start, end = _resolve_statistics_window(histories, statistics_start, statistics_end)
    grids: list[dict[str, Any]] = []
    for case in cases:
        statistics = _force_statistics(histories[case.name], start, end)
        grids.append(
            {
                "case": case.name,
                "wall_cell_size": case.wall_cell_size,
                "cell_count": case.cell_count,
                "declared_end_time": case.declared_end_time,
                "force_statistics": statistics,
            }
        )
    grids.sort(key=lambda grid: (-grid["wall_cell_size"], grid["cell_count"], grid["case"]))
    reference_statistics = next(
        grid["force_statistics"] for grid in grids if grid["case"] == reference.name
    )
    for grid in grids:
        grid["difference_to_reference"] = {
            metric: _difference(
                grid["force_statistics"].get(metric), reference_statistics.get(metric)
            )
            for metric in METRICS
        }

    convergence = {metric: _convergence_statistics(grids, metric, tolerance) for metric in METRICS}
    profiles = _profile_statistics(cases, grids, reference.name, start, end)
    pair_changes = [
        (metric, details["finest_pair"]["relative_change"])
        for metric, details in convergence.items()
        if details["finest_pair"] is not None
        and details["finest_pair"]["relative_change"] is not None
    ]
    largest_change = None
    if pair_changes:
        metric, relative_change = max(pair_changes, key=lambda item: float(item[1]))
        largest_change = {"metric": metric, "relative_change": relative_change}

    reference_definition = (
        "finest available wall-cell-size target"
        if reference_case is None
        else "user-selected completed grid"
    )
    report: dict[str, Any] = {
        "schema": "openonda-cube-grid-convergence/1",
        "statistics_window": {"start": start, "end": end},
        "reference_case": reference.name,
        "reference_definition": reference_definition,
        "wall_cell_size_definition": "target cube-patch size supplied as setup.py --dx",
        "grids": grids,
        "convergence": convergence,
        "cell_cost": _cell_cost_statistics(grids),
        "profiles": profiles,
        "excluded_cases": excluded,
        "assessment": {
            "difference_interpretation": (
                "Differences use the selected reference grid and are not exact discretisation errors."
            ),
            "tolerance": tolerance,
            "largest_finest_pair_relative_change": largest_change,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "grid_convergence.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(report, output_dir / "grid_convergence.csv")
    _write_markdown(report, output_dir / "grid_convergence.md")
    _plot_by_spacing(report, output_dir / "grid_convergence.png")
    _plot_by_cells(report, output_dir / "grid_convergence_by_cells.png")
    _plot_profiles(report, output_dir / "grid_convergence_profiles.png")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-root", type=Path, default=CASE_DIR / "samples")
    parser.add_argument("--output-dir", type=Path, default=CASE_DIR / "solution")
    parser.add_argument("--statistics-start", type=float)
    parser.add_argument("--statistics-end", type=float)
    parser.add_argument("--reference-case")
    parser.add_argument(
        "--tolerance",
        type=float,
        help="Optional relative finest-pair tolerance recorded in the report; no default is imposed.",
    )
    arguments = parser.parse_args()
    report = analyse_grid_convergence(
        arguments.samples_root,
        arguments.output_dir,
        statistics_start=arguments.statistics_start,
        statistics_end=arguments.statistics_end,
        reference_case=arguments.reference_case,
        tolerance=arguments.tolerance,
    )
    print(
        f"Wrote convergence report for {len(report['grids'])} completed grids to {arguments.output_dir}"
    )


if __name__ == "__main__":
    # Keep Matplotlib's cache outside a source checkout when this script is
    # launched directly from a tutorial directory.
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/openonda-matplotlib-cache")
    main()
