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
import hashlib
import json
import math
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
OPTIONAL_COLUMNS = ("accepted_time_step_size", "pressure_force_x", "viscous_force_x")
METRICS = (
    "mean_drag",
    "rms_drag",
    "mean_lift",
    "rms_lift",
    "mean_side",
    "rms_side",
    "strouhal_lift",
    "strouhal_side",
    "mean_pressure_drag",
    "mean_viscous_drag",
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
        available = tuple(
            name for name in (*FORCE_COLUMNS, *OPTIONAL_COLUMNS) if name in fieldnames
        )
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


def _case_context(case: GridCase, solution_root: Path) -> dict[str, Any]:
    """Read original run settings and global mesh data, never the current setup."""
    directory = solution_root / case.samples_dir.name
    metadata_path = directory / "fvm_metadata.json"
    result: dict[str, Any] = dict(
        length=1.0,
        speed=1.0,
        force_scale=None,
        configuration=None,
        scales_source="cube tutorial defaults D=1, U=1; run metadata absent",
        domain_bounds=None,
        realized_wall_cell_size=None,
        global_cell_count=None,
        mesh_controls=None,
        warnings=[],
    )
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("lifecycle", {}).get("status") != "complete":
            raise ValueError("native FVM metadata does not mark this run complete")
        if not np.isclose(
            metadata["state"]["time"], case.declared_end_time, atol=TIME_TOLERANCE, rtol=0
        ):
            raise ValueError("native completion time disagrees with grid_run.json")
        configuration = metadata["configuration"]
        samplers = [
            s
            for s in configuration["samplers"]
            if s["type"] == "ForceSampler" and s.get("file_name") == "forces_history"
        ]
        if len(samplers) != 1:
            raise ValueError("cannot identify one forces_history normalization in native metadata")
        sampler = samplers[0]
        result["length"] = _positive_float(sampler["reference_length"], "reference length")
        result["speed"] = _positive_float(sampler["reference_velocity"], "reference velocity")
        result["force_scale"] = (
            0.5
            * _positive_float(configuration["transport"]["density"], "density")
            * result["speed"] ** 2
            * _positive_float(sampler["reference_area"], "reference area")
        )
        result["scales_source"] = str(metadata_path.resolve())
        keys = (
            "transport",
            "turbulence",
            "boundaries",
            "schemes",
            "linear",
            "pimple",
            "initial_velocity",
            "initial_kinematic_pressure",
        )
        result["configuration"] = {key: configuration.get(key) for key in keys}
        result["configuration"]["force_normalization"] = {
            key: sampler[key]
            for key in ("reference_length", "reference_velocity", "reference_area", "patch_names")
        }
        result["configuration"]["time_integration"] = {
            key: configuration["time"].get(key) for key in ("time_step_size", "adjustment")
        }
    else:
        result["warnings"].append("native settings unavailable; matching physics is unverified")

    mesh_path = directory / "mesh.npz"
    if mesh_path.exists():
        with np.load(mesh_path, allow_pickle=False) as mesh:
            points = mesh["vertex_position"]
            result["domain_bounds"] = (
                np.column_stack((points.min(axis=0), points.max(axis=0))).ravel().tolist()
            )
            result["global_cell_count"] = len(mesh["cell_sizes"])
            generation = json.loads(str(mesh["metadata"]))["mesh_generation"]
            result["realized_wall_cell_size"] = generation["resolved_surface_patch_sizes"]["cube"]
            result["mesh_controls"] = {
                "method": generation["method"],
                "surface_hashes": generation["cartesian_report"].get("surface_hashes"),
                "requested_size_ratios": {
                    entry["name"]: round(entry["requested"] / case.wall_cell_size, 10)
                    for entry in generation["requested_sizes"]
                },
            }
        if result["global_cell_count"] != case.cell_count:
            raise ValueError("global mesh cell count disagrees with grid_run.json")
    else:
        result["warnings"].append("global mesh unavailable; realized spacing and domain unverified")
    return result


def _comparability(contexts: dict[str, dict], reference: str) -> dict:
    differences = {}
    ref = contexts[reference]
    for name, context in contexts.items():
        changed = []
        for key in ("configuration", "domain_bounds", "mesh_controls"):
            if context[key] is not None and ref[key] is not None and context[key] != ref[key]:
                changed.append(key)
        if changed:
            differences[name] = changed
    return {
        "differences_to_reference": differences,
        "matching_recorded_settings": not differences
        and all(not context["warnings"] for context in contexts.values()),
        "limitation": "Checks saved physics, solver controls, domain bounds, surface hashes and size ratios; no independent time-step convergence test.",
    }


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


def _spectral_peak(time: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    """Hann-windowed peak and its three-bin share of nonzero-frequency power."""
    uniform = np.linspace(time[0], time[-1], len(time))
    signal = np.interp(uniform, time, values)
    signal -= signal.mean()
    power = np.abs(np.fft.rfft(signal * np.hanning(len(signal)))) ** 2
    power[0] = 0.0
    peak = 1 + int(np.argmax(power[1:]))
    frequency = np.fft.rfftfreq(len(signal), uniform[1] - uniform[0])
    share = float(power[max(1, peak - 1) : peak + 2].sum() / max(power.sum(), VALUE_FLOOR**2))
    return float(frequency[peak]), share


def _frequency_diagnostic(
    time: np.ndarray, values: np.ndarray, length: float = 1.0, speed: float = 1.0
) -> dict[str, Any]:
    """Screen a candidate frequency; a window-scale drift is not a shedding St.

    Five observed cycles, eight samples per cycle, a three-bin power share of
    50%, and agreement between half-window peaks are practical screening rules,
    not confidence limits or proof of a statistically converged spectrum.
    """
    duration = float(time[-1] - time[0])
    result = dict(
        strouhal=None,
        reason=None,
        candidate_frequency_hz=None,
        observed_cycles=None,
        strouhal_resolution=length / speed / duration,
    )
    if len(time) < 32:
        result["reason"] = "fewer than 32 samples"
        return result
    centred = values - _time_mean(time, values)
    rms = float(np.sqrt(_time_mean(time, centred**2)))
    if rms < 1.0e-7:
        result["reason"] = "force fluctuations below 1e-7 coefficient"
        return result
    frequency, share = _spectral_peak(time, centred)
    cycles = frequency * duration
    result.update(
        candidate_frequency_hz=frequency, observed_cycles=cycles, peak_power_fraction=share
    )
    if cycles < 5.0:
        result["reason"] = "fewer than five cycles; peak is not resolved from slow drift"
        return result
    if frequency * float(np.max(np.diff(time))) > 0.125:
        result["reason"] = "fewer than eight samples per candidate cycle"
        return result
    slope = float(np.polyfit(time - time[0], values, 1)[0])
    trend_rms = abs(slope) * duration / np.sqrt(12.0)
    if trend_rms > 0.5 * rms:
        result["reason"] = "linear drift exceeds half the fluctuation RMS"
        return result
    midpoint = 0.5 * (time[0] + time[-1])
    half_peaks = [
        _spectral_peak(*_window_series(time, values, a, b))[0]
        for a, b in ((time[0], midpoint), (midpoint, time[-1]))
    ]
    result["half_window_frequencies_hz"] = half_peaks
    if any(abs(peak - frequency) > max(2.0 / duration, 0.2 * frequency) for peak in half_peaks):
        result["reason"] = "half-window frequency estimates disagree"
    elif share < 0.5:
        result["reason"] = "no concentrated spectral peak (three-bin power below 50%)"
    else:
        result["strouhal"] = frequency * length / speed
    return result


def _force_statistics(
    history: ForceHistory, start: float, end: float, *, context: dict | None = None
) -> dict[str, Any]:
    """Compute time-weighted force statistics over a shared physical interval."""

    result: dict[str, Any] = {"samples": None}
    context = context or {"length": 1.0, "speed": 1.0, "force_scale": None}
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
        if result[f"rms_{prefix}"] <= VALUE_FLOOR:
            result[f"rms_{prefix}"] = 0.0
        midpoint = 0.5 * (start + end)
        halves = [
            _time_mean(*_window_series(time, values, a, b))
            for a, b in ((start, midpoint), (midpoint, end))
        ]
        result[f"{prefix}_half_means"] = halves
        result[f"{prefix}_half_mean_change"] = float(halves[1] - halves[0])
        result[f"{prefix}_range"] = [float(values.min()), float(values.max())]
        result[f"{prefix}_block_means"] = [
            _time_mean(*_window_series(time, values, a, b))
            for a, b in zip(np.linspace(start, end, 5)[:-1], np.linspace(start, end, 5)[1:])
        ]

    result["drag_drift_relative"] = (
        abs(result["drag_half_mean_change"]) / abs(result["mean_drag"])
        if abs(result["mean_drag"]) > VALUE_FLOOR
        else None
    )
    for prefix, column in (("pressure", "pressure_force_x"), ("viscous", "viscous_force_x")):
        result[f"mean_{prefix}_drag"] = (
            _time_mean(*_window_series(history.time, history.values[column], start, end))
            / context["force_scale"]
            if column in history.values and context.get("force_scale")
            else None
        )
    result["sampled_time_step_range"] = None
    if "accepted_time_step_size" in history.values:
        _, step_sizes = _window_series(
            history.time, history.values["accepted_time_step_size"], start, end
        )
        result["sampled_time_step_range"] = [float(step_sizes.min()), float(step_sizes.max())]

    for metric, column in (
        ("strouhal_lift", "lift_coefficient"),
        ("strouhal_side", "side_force_coefficient"),
    ):
        diagnostic = (
            _frequency_diagnostic(*signals[column], context["length"], context["speed"])
            if column in signals
            else {"strouhal": None, "reason": "force column absent"}
        )
        result[metric] = diagnostic["strouhal"]
        result[f"{metric}_diagnostic"] = diagnostic
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
    candidate: MeanProfile, reference: MeanProfile, interval: tuple[float, float] | None = None
) -> dict[str, float | list[float] | None]:
    """Measure a candidate mean profile relative to the reference mean profile."""

    if candidate.axis != reference.axis:
        raise ValueError(
            f"profile axes differ: candidate varies along {candidate.axis}, reference along {reference.axis}"
        )
    lower = max(float(candidate.position[0]), float(reference.position[0]))
    upper = min(float(candidate.position[-1]), float(reference.position[-1]))
    if interval is not None:
        lower, upper = max(lower, interval[0]), min(upper, interval[1])
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
    scale = max(abs(fine_value), VALUE_FLOOR)
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
                comparison["wake"] = _profile_difference(candidate, reference, (0.5, 8.0))
            except ValueError as error:
                comparisons[case.name] = {"available": False, "reason": str(error)}
                continue
            # Retain the actual mean profiles for inspection, not just their norms.
            comparisons[case.name] = {
                "available": True,
                **comparison,
                "position": candidate.position.tolist(),
                "mean_velocity": candidate.velocity.tolist(),
            }
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
        "statistics_start",
        "statistics_end",
        "realized_wall_cell_size",
        "drag_drift_relative",
        "drag_first_half_mean",
        "drag_second_half_mean",
        "strouhal_lift_reason",
        "strouhal_side_reason",
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
                "statistics_start": report["statistics_window"]["start"],
                "statistics_end": report["statistics_window"]["end"],
                "realized_wall_cell_size": grid["context"]["realized_wall_cell_size"],
                "drag_drift_relative": statistics["drag_drift_relative"],
                "drag_first_half_mean": statistics["drag_half_means"][0],
                "drag_second_half_mean": statistics["drag_half_means"][1],
                "strouhal_lift_reason": statistics["strouhal_lift_diagnostic"]["reason"],
                "strouhal_side_reason": statistics["strouhal_side_diagnostic"]["reason"],
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
        "RMS denotes fluctuations about the time-weighted mean, not uncertainty in that mean.",
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

    lines += ["", "## Assessment", "", report["assessment"]["conclusion"], ""]
    lines += [f"- {reason}" for reason in report["assessment"]["reasons"]]
    lines += [
        "",
        "## Averaging-window sensitivity and drag components",
        "",
        "| Case | Cd first half | Cd second half | Change / mean Cd | Pressure Cd | Viscous Cd | St (lift) | St (side) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for grid in report["grids"]:
        f = grid["force_statistics"]
        lines.append(
            f"| {grid['case']} | {_display_number(f['drag_half_means'][0])} | "
            f"{_display_number(f['drag_half_means'][1])} | {_display_percent(f['drag_drift_relative'])} | "
            f"{_display_number(f['mean_pressure_drag'])} | {_display_number(f['mean_viscous_drag'])} | "
            f"{_display_number(f['strouhal_lift'])} | {_display_number(f['strouhal_side'])} |"
        )
    lines += [
        "",
        "Half-window changes and four-block means (in JSON) expose drift; they are not confidence intervals.",
        "The automatic final-half window does not establish that transients have ended.",
        "",
        "## Frequency qualification",
        "",
        "St = f D / U. A Hann-windowed FFT candidate is reported only with at least five cycles,",
        "at least eight samples per cycle, limited linear drift, consistent half-window peaks,",
        "and at least 50% of nonzero-frequency power in its three-bin peak neighborhood.",
        "These are screening rules, not a confidence level. The resolution is D/(U T).",
        "",
    ]
    for grid in report["grids"]:
        for component in ("lift", "side"):
            d = grid["force_statistics"][f"strouhal_{component}_diagnostic"]
            lines.append(
                f"- {grid['case']}, {component}: {d['reason'] or 'screen passed; spectral estimate only'}; "
                f"candidate cycles = {_display_number(d.get('observed_cycles'))}."
            )
    lines += [
        "",
        "## Wake profile changes",
        "",
        "Mean vector-velocity differences use the same physical time interval.",
        "The wake-only norm covers x/D = 0.5–8; whole-line norms also include the upstream flow.",
        "",
        "| Case | Centreline wake L2 difference | Off-axis wake L2 difference |",
        "|---|---:|---:|",
    ]
    for grid in report["grids"]:
        values = [
            report["profiles"][name]
            .get("comparisons", {})
            .get(grid["case"], {})
            .get("wake", {})
            .get("relative_l2")
            for name in PROFILE_NAMES
        ]
        lines.append(
            f"| {grid['case']} | {_display_percent(values[0])} | {_display_percent(values[1])} |"
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
    lines += [
        "",
        "## Provenance and interpretation",
        "",
        "Recorded physics, solver settings and domain/mesh controls match: "
        f"`{report['comparability']['matching_recorded_settings']}`.",
        "The JSON records differences, actual mesh sizes, input hashes and frequency rejection reasons.",
        "Cell counts come from the registered global mesh, not the rank-local metadata count.",
        "Grid refinement also changes the LES filter scale. A dense run alone does not establish statistical or time-step convergence.",
        "Richardson/GCI values are withheld for incompatible settings, unequal refinement ratios, nonmonotone values, or more than 1% drag half-window drift.",
        "The 1% drift screen is a diagnostic convention, not proof of stationarity.",
        "Method reference: [NASA spatial convergence guidance](https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html) "
        "and [temporal convergence guidance](https://www.grc.nasa.gov/www/wind/valid/tutorial/tempconv.html).",
    ]
    lines.append("")
    destination.write_text("\n".join(lines), encoding="utf-8")


def print_statistics(report: dict) -> None:
    window = report["statistics_window"]
    print(f"\nCube reference grids: common window t = {window['start']:g} to {window['end']:g} s")
    print(
        f"Differences use {report['reference_case']}; fluctuation RMS is not uncertainty in the mean.\n"
    )
    print(
        f"{'Grid':<14} {'Cells':>9} {'h target':>9} {'Mean Cd':>10} {'RMS Cd':>10} {'RMS Cl':>10} {'RMS Cs':>10} {'dCd/ref':>10} {'Cd drift':>10}"
    )
    for grid in report["grids"]:
        f = grid["force_statistics"]
        numbers = " ".join(
            f"{_display_number(f[key]):>10}"
            for key in ("mean_drag", "rms_drag", "rms_lift", "rms_side")
        )
        print(
            f"{grid['case']:<14} {grid['cell_count']:>9,} {grid['wall_cell_size']:>9.4g} {numbers} "
            f"{_display_percent(grid['difference_to_reference']['mean_drag']['relative']):>10} "
            f"{_display_percent(f['drag_drift_relative']):>10}"
        )
    print("\nCd drift = absolute difference between half-window means / full-window mean.")
    print(f"\n{'Grid':<14} {'Pressure Cd':>12} {'Viscous Cd':>12} {'St lift':>12} {'St side':>12}")
    for grid in report["grids"]:
        f = grid["force_statistics"]
        print(
            f"{grid['case']:<14} "
            + " ".join(
                f"{_display_number(f[k]):>12}"
                for k in (
                    "mean_pressure_drag",
                    "mean_viscous_drag",
                    "strouhal_lift",
                    "strouhal_side",
                )
            )
        )
        for component in ("lift", "side"):
            reason = f[f"strouhal_{component}_diagnostic"]["reason"]
            if reason:
                print(f"  St {component}: {reason}")
    print(f"\n{report['assessment']['conclusion']}")
    for reason in report["assessment"]["reasons"]:
        print(f"  {reason}")
    print("\nWake profile L2 differences from the reference (x/D = 0.5 to 8):")
    for grid in report["grids"]:
        values = [
            report["profiles"][name]
            .get("comparisons", {})
            .get(grid["case"], {})
            .get("wake", {})
            .get("relative_l2")
            for name in PROFILE_NAMES
        ]
        print(
            f"  {grid['case']}: centreline {_display_percent(values[0])}, off-axis {_display_percent(values[1])}"
        )


def _plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from openonda import plotting as theme

    theme.set_thesis_style()
    return plt, theme


def _grid_styles(grids: list[dict]) -> dict[str, dict]:
    _, theme = _plotting()
    names = ("RefGray", "FVMorange", "VPMpurple", "TUDdark", "TUDcyan", "AccentGreen")
    markers = ("v", "s", "D", "o", "^", "P")
    return {
        grid["case"]: dict(
            color=theme.COLORS[names[i % len(names)]],
            marker=markers[i % len(markers)],
            label=grid["case"].replace("_", " ").capitalize(),
        )
        for i, grid in enumerate(grids)
    }


def _save_plot(figure, destination: Path, formats: tuple[str, ...]) -> None:
    plt, theme = _plotting()
    theme.fit_thesis_y_label_margins(figure, figure.axes)
    theme.validate_thesis_figure(figure, figure.axes)
    for fmt in formats:
        figure.savefig(
            destination.with_suffix(f".{fmt}"),
            dpi=theme.DEFAULT_DPI,
            format=fmt,
            bbox_inches=None,
            facecolor="white",
        )
    plt.close(figure)


def _plot_force_metrics(
    report: dict, destination: Path, *, by_cells=False, formats=("png",)
) -> None:
    plt, theme = _plotting()
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    grids = report["grids"]
    x = np.asarray(
        [
            g["cell_count"] / 1000 if by_cells else g["wall_cell_size"] / g["context"]["length"]
            for g in grids
        ]
    )
    fig, axes = plt.subplots(3, 1, figsize=(12.5 / 2.54, 12.0 / 2.54), sharex=True)
    fig.subplots_adjust(left=0.2, right=0.8, bottom=0.13, top=0.97, hspace=0.32)
    for axis, metrics, label in zip(
        axes,
        (("mean_drag",), ("rms_drag",), ("rms_lift", "rms_side")),
        (r"$\overline{C_D}$", r"$\sigma(C_D)$", r"$\sigma(C_\perp)$"),
        strict=True,
    ):
        for i, metric in enumerate(metrics):
            values = [g["force_statistics"].get(metric, np.nan) for g in grids]
            axis.plot(
                x,
                values,
                marker=("o", "s")[i],
                linestyle=("-", "--")[i],
                color=theme.COLORS[("TUDdark", "VPMpurple")[i]],
                label=(r"$C_L$", r"$C_S$")[i],
            )
        axis.set_ylabel(label)
        axis.yaxis.set_major_locator(MaxNLocator(3))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.4g}"))
        if len(metrics) == 2:
            axis.legend(loc="best", frameon=False, ncol=2, handlelength=1.3, columnspacing=1.0)
    if by_cells:
        axes[-1].set_xlabel(r"Fluid cells [$10^3$]")
        axes[-1].xaxis.set_major_locator(MaxNLocator(4))
    else:
        axes[-1].set_xlabel(r"Target $h/D$")
        if x.max() / x.min() > 3:
            axes[-1].set_xscale("log", base=2)
        axes[-1].set_xticks(x)
        axes[-1].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.3g}"))
        axes[-1].invert_xaxis()
    _save_plot(fig, destination, formats)


def _plot_profiles(report: dict, destination: Path, formats=("png",)) -> None:
    plt, theme = _plotting()
    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(2, 1, figsize=(12.5 / 2.54, 10.5 / 2.54), sharex=True)
    legend_rows = int(np.ceil(len(report["grids"]) / 2))
    fig.subplots_adjust(
        left=0.2, right=0.8, bottom=0.15, top=0.93 - 0.05 * legend_rows, hspace=0.38
    )
    styles = _grid_styles(report["grids"])
    for axis, name in zip(axes, PROFILE_NAMES, strict=True):
        profile = report["profiles"][name]
        for grid in report["grids"]:
            values = profile.get("comparisons", {}).get(grid["case"], {})
            if not values.get("available"):
                continue
            x = np.asarray(values["position"]) / grid["context"]["length"]
            ux = np.asarray(values["mean_velocity"])[:, 0] / grid["context"]["speed"]
            selected = (x >= 0.5) & (x <= 8)
            axis.plot(x[selected], ux[selected], **styles[grid["case"]], markevery=12, markersize=3)
        axis.set_ylabel(r"$\overline{u_x}/U_\infty$")
        axis.set_title(r"$y/D=0$" if name == "centreline" else r"$y/D=0.75$")
        axis.set_xlim(0.5, 8.0)
        axis.yaxis.set_major_locator(MaxNLocator(3))
        if not axis.lines:
            axis.text(0.5, 0.5, "No common profile data", transform=axis.transAxes, ha="center")
    axes[-1].set_xlabel(r"$x/D$")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        legend = fig.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False
        )
        fig.canvas.draw()
        legend_bottom = legend.get_window_extent(fig.canvas.get_renderer()).y0 / fig.bbox.height
        fig.subplots_adjust(top=legend_bottom - 0.08)
    _save_plot(fig, destination, formats)


def _plot_histories(report: dict, histories: dict, destination: Path, formats=("png",)) -> None:
    plt, theme = _plotting()
    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(2, 1, figsize=(12.5 / 2.54, 10.5 / 2.54), sharex=True)
    legend_rows = int(np.ceil(len(report["grids"]) / 2))
    fig.subplots_adjust(
        left=0.2, right=0.8, bottom=0.15, top=0.92 - 0.05 * legend_rows, hspace=0.35
    )
    styles = _grid_styles(report["grids"])
    start, end = (report["statistics_window"][k] for k in ("start", "end"))
    for grid in report["grids"]:
        history = histories[grid["case"]]
        for axis, column in zip(axes, ("drag_coefficient", "lift_coefficient"), strict=True):
            if column not in history.values:
                continue
            t, v = _window_series(history.time, history.values[column], start, end)
            axis.plot(t, v, **styles[grid["case"]], markevery=max(1, len(t) // 10), markersize=3)
    for axis, label in zip(axes, (r"$C_D$", r"$C_L$"), strict=True):
        axis.set_ylabel(label)
        axis.yaxis.set_major_locator(MaxNLocator(4))
    axes[-1].set_xlabel(r"$t$ [s]")
    legend = fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
    )
    fig.canvas.draw()
    legend_bottom = legend.get_window_extent(fig.canvas.get_renderer()).y0 / fig.bbox.height
    fig.subplots_adjust(top=legend_bottom - 0.04)
    _save_plot(fig, destination, formats)


def analyse_grid_convergence(
    samples_root: str | Path,
    output_dir: str | Path,
    *,
    statistics_start: float | None = None,
    statistics_end: float | None = None,
    reference_case: str | None = None,
    tolerance: float | None = None,
    solution_root: str | Path | None = None,
    formats: tuple[str, ...] = ("png",),
) -> dict[str, Any]:
    """Analyse every completed cube grid and write derived convergence artifacts.

    The function only reads the registered case directories below
    ``samples_root``.  It creates or replaces the named report files in
    ``output_dir``; it never changes individual grid outputs.
    """

    samples_root = Path(samples_root)
    output_dir = Path(output_dir)
    solution_root = (
        Path(solution_root) if solution_root is not None else samples_root.parent / "solution"
    )
    if not formats or any(fmt not in ("png", "pdf") for fmt in formats):
        raise ValueError("formats must contain png and/or pdf")
    if tolerance is not None:
        tolerance = _positive_float(tolerance, "convergence tolerance")
    cases, excluded = _discover_cases(samples_root)
    histories: dict[str, ForceHistory] = {}
    contexts: dict[str, dict] = {}
    usable_cases: list[GridCase] = []
    for case in cases:
        try:
            history = _read_force_history(case.samples_dir / "forces_history.csv")
            if not np.isclose(
                history.time[-1], case.declared_end_time, atol=TIME_TOLERANCE, rtol=0
            ):
                raise ValueError("force history does not reach the registered completion time")
            contexts[case.name] = _case_context(case, solution_root)
            histories[case.name] = history
        except (ValueError, KeyError) as error:
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
        statistics = _force_statistics(
            histories[case.name], start, end, context=contexts[case.name]
        )
        grids.append(
            {
                "case": case.name,
                "wall_cell_size": case.wall_cell_size,
                "cell_count": case.cell_count,
                "declared_end_time": case.declared_end_time,
                "force_statistics": statistics,
                "context": contexts[case.name],
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
    comparability = _comparability(contexts, reference.name)
    drifting = [
        g["case"] for g in grids if (g["force_statistics"]["drag_drift_relative"] or 0) > 0.01
    ]
    gci_reasons = []
    if comparability["differences_to_reference"]:
        gci_reasons.append("saved physical/numerical settings or mesh controls differ")
    if drifting:
        gci_reasons.append("drag half-window drift exceeds the 1% screening threshold")
    if gci_reasons:
        for entry in convergence.values():
            entry["richardson"].update(
                available=False,
                reason="; ".join(gci_reasons),
                observed_order=None,
                extrapolated=None,
                fine_grid_gci=None,
            )
    reasons = []
    pair = convergence["mean_drag"]["finest_pair"]
    if pair:
        reasons.append(
            f"Mean Cd changes {_display_percent(pair['relative_change'])} from "
            f"{pair['coarser_case']} to {pair['finer_case']}."
        )
        fine = next(g for g in grids if g["case"] == pair["finer_case"])
        drift = fine["force_statistics"]["drag_drift_relative"]
        reasons.append(
            f"{fine['case']} Cd half-window drift is {_display_percent(drift)}; "
            "the averaging-window choice must be checked before interpreting a grid plateau."
        )
    for metric in ("rms_drag", "rms_lift", "rms_side"):
        pair = convergence[metric]["finest_pair"]
        if pair:
            reasons.append(
                f"Finest-pair {metric} change: {_display_percent(pair['relative_change'])}."
            )
    if all(
        g["force_statistics"]["strouhal_lift"] is None
        and g["force_statistics"]["strouhal_side"] is None
        for g in grids
    ):
        reasons.append("No resolved shedding Strouhal number in the selected window.")
    if not comparability["matching_recorded_settings"]:
        reasons.append(
            "Matching run settings are unverified or differ; inspect comparability and context in JSON."
        )
    reasons.append(
        "A dense grid can test spatial sensitivity, but longer stationary records and a separate time-step check are still needed."
    )
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
        "schema": "openonda-cube-grid-convergence/2",
        "statistics_window": {"start": start, "end": end},
        "reference_case": reference.name,
        "reference_definition": reference_definition,
        "wall_cell_size_definition": "target cube-patch size supplied as setup.py --dx",
        "grids": grids,
        "convergence": convergence,
        "cell_cost": _cell_cost_statistics(grids),
        "profiles": profiles,
        "excluded_cases": excluded,
        "comparability": comparability,
        "provenance": {
            "postprocessor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "input_sha256": {
                str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                for case in cases
                for path in (
                    case.samples_dir / "grid_run.json",
                    case.samples_dir / "forces_history.csv",
                    *(case.samples_dir / f"{name}.csv" for name in PROFILE_NAMES),
                    solution_root / case.samples_dir.name / "fvm_metadata.json",
                )
                if path.is_file()
            },
        },
        "assessment": {
            "conclusion": "Grid independence: NOT ESTABLISHED by these statistics alone.",
            "reasons": reasons,
            "drag_drift_screen": 0.01,
            "grids_exceeding_drag_drift_screen": drifting,
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
    _plot_force_metrics(report, output_dir / "grid_convergence.png", formats=formats)
    _plot_force_metrics(
        report, output_dir / "grid_convergence_by_cells.png", by_cells=True, formats=formats
    )
    _plot_profiles(report, output_dir / "grid_convergence_profiles.png", formats=formats)
    _plot_histories(
        report, histories, output_dir / "grid_convergence_histories.png", formats=formats
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-root", type=Path, default=CASE_DIR / "samples")
    parser.add_argument("--output-dir", type=Path, default=CASE_DIR / "solution")
    parser.add_argument(
        "--solution-root", type=Path, help="Native solution root; defaults beside samples-root"
    )
    parser.add_argument("--format", choices=("png", "pdf", "both"), default="both")
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
        solution_root=arguments.solution_root,
        formats=("png", "pdf") if arguments.format == "both" else (arguments.format,),
    )
    print_statistics(report)
    print(
        f"Wrote convergence report for {len(report['grids'])} completed grids to {arguments.output_dir}"
    )


if __name__ == "__main__":
    # Keep Matplotlib's cache outside a source checkout when this script is
    # launched directly from a tutorial directory.
    main()
