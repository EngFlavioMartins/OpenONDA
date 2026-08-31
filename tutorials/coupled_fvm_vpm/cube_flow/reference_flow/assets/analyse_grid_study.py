#!/usr/bin/env python3
"""Prepare and analyse compact cube-reference grid-study samples."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np


GRID_ORDER = ("h16", "h32", "h64")
GRID_SPACING = {"h16": 1.0 / 16.0, "h32": 1.0 / 32.0, "h64": 1.0 / 64.0}
LINE_FILES = ("centreline.csv", "offaxis_y075.csv")
EPSILON = 1.0e-30


def _strict(value):
    if isinstance(value, dict):
        return {key: _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _relative(first: float, second: float) -> float:
    return abs(first - second) / max(abs(first), abs(second), EPSILON)


def _load_table(path: Path, *, strings: bool = False) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Required grid-study sample is missing: {path}")
    table = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        dtype=None if strings else float,
        encoding="utf-8" if strings else None,
    )
    table = np.atleast_1d(table)
    if table.size == 0 or not table.dtype.names:
        raise ValueError(f"Grid-study sample is empty or malformed: {path}")
    return table


def _column(table: np.ndarray, name: str, path: Path) -> np.ndarray:
    if name not in (table.dtype.names or ()):
        raise ValueError(f"{path} has no {name!r} column")
    return np.asarray(table[name])


def _compact_line(source: Path, destination: Path, interval: float) -> int:
    kept = 0
    with source.open(newline="", encoding="utf-8") as input_stream:
        reader = csv.DictReader(input_stream)
        if not reader.fieldnames or "time" not in reader.fieldnames:
            raise ValueError(f"{source} is not a time-aware sampler CSV")
        with destination.open("w", newline="", encoding="utf-8") as output_stream:
            writer = csv.DictWriter(output_stream, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                time_value = float(row["time"])
                sample_index = round(time_value / interval)
                if sample_index >= 1 and math.isclose(
                    time_value,
                    sample_index * interval,
                    rel_tol=0.0,
                    abs_tol=1.0e-8,
                ):
                    writer.writerow(row)
                    kept += 1
    if kept == 0:
        raise ValueError(f"No rows in {source} matched interval {interval:g}")
    return kept


def prepare_existing_fine(source: Path, destination: Path, line_interval: float) -> None:
    source = source.resolve()
    destination = destination.resolve()
    required = (source / "forces_history.csv", *(source / name for name in LINE_FILES))
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing existing fine samples: " + ", ".join(map(str, missing)))
    if destination.exists():
        metadata_path = destination / "grid_metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("status") == "completed":
                print(f"Fine-grid compact samples already exist in {destination}")
                return
        raise FileExistsError(f"Refusing to overwrite incomplete directory {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".h64-", dir=destination.parent))
    try:
        shutil.copy2(source / "forces_history.csv", temporary / "forces_history.csv")
        row_counts = {
            name: _compact_line(source / name, temporary / name, line_interval)
            for name in LINE_FILES
        }
        forces = _load_table(temporary / "forces_history.csv", strings=True)
        maximum_time = float(np.max(_column(forces, "time", temporary / "forces_history.csv")))
        metadata = {
            "schema": "openonda-cube-grid-trial/1",
            "status": "completed",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "grid": "h64",
            "physics": {"reynolds_number": 1000.0},
            "mesh": {
                "requested_surface_cell_size": GRID_SPACING["h64"],
                "surface_cell_size": GRID_SPACING["h64"],
                "near_wake_cell_size": 2.0 * GRID_SPACING["h64"],
                "downstream_wake_cell_size": 4.0 * GRID_SPACING["h64"],
                "background_cell_size": 0.5,
                "cell_count": None,
            },
            "time": {
                "time_step_size": 0.01,
                "end_time": maximum_time,
                "force_interval": 0.05,
                "line_interval": line_interval,
            },
            "execution": {
                "reused_existing_samples": True,
                "source_directory": str(source),
                "retained_line_rows": row_counts,
                "vtk_output": False,
            },
        }
        (temporary / "grid_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"Prepared compact fine-grid samples in {destination}")


def _force_metrics(path: Path, start: float, end: float) -> dict:
    table = _load_table(path, strings=True)
    time_values = _column(table, "time", path).astype(float)
    if "patch" in (table.dtype.names or ()):
        patch = _column(table, "patch", path).astype(str)
        patch_mask = patch == "cube"
    else:
        patch_mask = np.ones(len(table), dtype=bool)
    mask = patch_mask & (time_values >= start - 1.0e-9) & (time_values <= end + 1.0e-9)
    if np.count_nonzero(mask) < 4:
        raise ValueError(f"{path} has too few force samples in [{start:g}, {end:g}]")
    midpoint = 0.5 * (start + end)
    first = mask & (time_values <= midpoint)
    second = mask & (time_values > midpoint)
    drag = _column(table, "drag_coefficient", path).astype(float)
    lift = _column(table, "lift_coefficient", path).astype(float)
    side = _column(table, "side_force_coefficient", path).astype(float)
    mean_drag = float(np.mean(drag[mask]))
    first_drag = float(np.mean(drag[first]))
    second_drag = float(np.mean(drag[second]))
    result = {
        "mean_cd": mean_drag,
        "std_cd": float(np.std(drag[mask])),
        "rms_cl": float(np.sqrt(np.mean(lift[mask] ** 2))),
        "rms_cs": float(np.sqrt(np.mean(side[mask] ** 2))),
        "first_half_mean_cd": first_drag,
        "second_half_mean_cd": second_drag,
        "mean_cd_half_window_change": _relative(first_drag, second_drag),
        "sample_count": int(np.count_nonzero(mask)),
    }
    if "pressure_force_x" in (table.dtype.names or ()):
        result["mean_pressure_cd"] = float(
            2.0 * np.mean(_column(table, "pressure_force_x", path).astype(float)[mask])
        )
    if "viscous_force_x" in (table.dtype.names or ()):
        result["mean_viscous_cd"] = float(
            2.0 * np.mean(_column(table, "viscous_force_x", path).astype(float)[mask])
        )
    return result


def _mean_profile(table: np.ndarray, path: Path, start: float, end: float):
    time_values = _column(table, "time", path)
    positions = _column(table, "position_x", path)
    velocity = _column(table, "velocity_x", path)
    mask = (time_values >= start - 1.0e-9) & (time_values <= end + 1.0e-9)
    if np.count_nonzero(mask) == 0:
        raise ValueError(f"{path} has no line samples in [{start:g}, {end:g}]")
    x, inverse = np.unique(positions[mask], return_inverse=True)
    sums = np.bincount(inverse, weights=velocity[mask])
    counts = np.bincount(inverse)
    return x, sums / counts


def _recirculation_length(x: np.ndarray, velocity: np.ndarray) -> float:
    mask = (x >= 0.5) & (x <= 5.0)
    wake_x = x[mask]
    wake_u = velocity[mask]
    for index in range(len(wake_x) - 1):
        if wake_u[index] <= 0.0 < wake_u[index + 1]:
            fraction = -wake_u[index] / (wake_u[index + 1] - wake_u[index])
            return float(wake_x[index] + fraction * (wake_x[index + 1] - wake_x[index]))
    return float("nan")


def _line_metrics(
    path: Path, start: float, end: float
) -> tuple[dict, tuple[np.ndarray, np.ndarray]]:
    table = _load_table(path)
    midpoint = 0.5 * (start + end)
    x, velocity = _mean_profile(table, path, start, end)
    first_x, first_velocity = _mean_profile(table, path, start, midpoint)
    second_x, second_velocity = _mean_profile(table, path, midpoint, end)
    if not np.array_equal(x, first_x) or not np.array_equal(x, second_x):
        raise ValueError(f"Probe locations changed during {path}")
    wake = (x >= 0.52) & (x <= 5.0)
    profile_change = float(
        np.linalg.norm(second_velocity[wake] - first_velocity[wake])
        / max(np.linalg.norm(velocity[wake]), EPSILON)
    )
    full_recirculation = _recirculation_length(x, velocity)
    first_recirculation = _recirculation_length(x, first_velocity)
    second_recirculation = _recirculation_length(x, second_velocity)
    metrics = {
        "u_x1": float(np.interp(1.0, x, velocity)),
        "u_x2": float(np.interp(2.0, x, velocity)),
        "u_x4": float(np.interp(4.0, x, velocity)),
        "recirculation_length": full_recirculation,
        "first_half_recirculation_length": first_recirculation,
        "second_half_recirculation_length": second_recirculation,
        "recirculation_half_window_change": _relative(first_recirculation, second_recirculation),
        "wake_profile_half_window_l2_change": profile_change,
    }
    return metrics, (x, velocity)


def _spatial_convergence(values: dict[str, float], tolerance: float) -> dict:
    coarse, medium, fine = (float(values[name]) for name in GRID_ORDER)
    coarse_change = _relative(coarse, medium)
    fine_change = _relative(medium, fine)
    delta_coarse = medium - coarse
    delta_fine = fine - medium
    monotone = bool(delta_coarse * delta_fine > 0.0)
    observed_order = float("nan")
    extrapolated = float("nan")
    fine_gci = float("nan")
    estimated_errors: dict[str, float] = {}
    recommended_h = float("nan")
    if monotone and abs(delta_fine) > EPSILON and abs(delta_coarse) > EPSILON:
        observed_order = math.log(abs(delta_coarse / delta_fine)) / math.log(2.0)
        if 0.0 < observed_order <= 10.0:
            denominator = 2.0**observed_order - 1.0
            extrapolated = fine + delta_fine / denominator
            fine_gci = 1.25 * abs(delta_fine) / max(abs(fine), EPSILON) / denominator
            coefficient = (fine - medium) / (
                GRID_SPACING["h64"] ** observed_order - GRID_SPACING["h32"] ** observed_order
            )
            for grid in GRID_ORDER:
                estimated_errors[grid] = (
                    1.25
                    * abs(coefficient * GRID_SPACING[grid] ** observed_order)
                    / max(abs(extrapolated), EPSILON)
                )
            for level in range(4, 12):
                spacing = 1.0 / (2**level)
                predicted = (
                    1.25
                    * abs(coefficient * spacing**observed_order)
                    / max(abs(extrapolated), EPSILON)
                )
                if predicted <= tolerance:
                    recommended_h = spacing
                    break
    passed = bool(np.isfinite(fine_gci) and fine_gci <= tolerance and fine_change < coarse_change)
    return {
        "values": values,
        "coarse_to_medium_relative_change": coarse_change,
        "medium_to_fine_relative_change": fine_change,
        "changes_decrease": bool(fine_change < coarse_change),
        "monotone": monotone,
        "observed_order": observed_order,
        "richardson_extrapolated_value": extrapolated,
        "fine_grid_gci": fine_gci,
        "estimated_gci_by_grid": estimated_errors,
        "tolerance": tolerance,
        "passed": passed,
        "recommended_surface_cell_size": recommended_h,
    }


def _profile_convergence(
    profiles: dict[str, tuple[np.ndarray, np.ndarray]], tolerance: float
) -> dict:
    coarse_x, coarse_u = profiles["h16"]
    medium_x, medium_u = profiles["h32"]
    fine_x, fine_u = profiles["h64"]
    if not np.array_equal(coarse_x, medium_x) or not np.array_equal(medium_x, fine_x):
        raise ValueError("Grid-study profile coordinates do not match")
    wake = (fine_x >= 0.52) & (fine_x <= 5.0)
    coarse_change = float(
        np.linalg.norm(medium_u[wake] - coarse_u[wake])
        / max(np.linalg.norm(medium_u[wake]), EPSILON)
    )
    fine_change = float(
        np.linalg.norm(fine_u[wake] - medium_u[wake]) / max(np.linalg.norm(fine_u[wake]), EPSILON)
    )
    return {
        "coarse_to_medium_relative_l2_change": coarse_change,
        "medium_to_fine_relative_l2_change": fine_change,
        "changes_decrease": bool(fine_change < coarse_change),
        "tolerance": tolerance,
        "passed": bool(fine_change <= tolerance and fine_change < coarse_change),
    }


def _markdown(report: dict) -> str:
    lines = [
        "# Cube reference-flow grid convergence",
        "",
        f"Analysis window: `{report['analysis_window']['start']:g}` to "
        f"`{report['analysis_window']['end']:g}` convective time units.",
        "",
        "| Grid | h/D | Cells | Mean Cd | Cd stationarity | Recirculation length |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for grid in GRID_ORDER:
        record = report["grids"][grid]
        cells = record["metadata"]["mesh"].get("cell_count")
        cells_text = f"{cells:,}" if isinstance(cells, int) else "existing archive"
        lines.append(
            f"| {grid} | {GRID_SPACING[grid]:.8f} | {cells_text} | "
            f"{record['forces']['mean_cd']:.8f} | "
            f"{100.0 * record['forces']['mean_cd_half_window_change']:.3f}% | "
            f"{record['centreline']['recirculation_length']:.6f} |"
        )
    primary = report["convergence"]["mean_cd"]
    lines.extend(
        [
            "",
            f"Load-convergence status: **{report['load_status']}**.",
            f"Flow-field-convergence status: **{report['flow_field_status']}**.",
            "",
            f"Observed drag order: `{primary['observed_order']}`.",
            f"Fine-grid drag GCI: `{primary['fine_grid_gci']}`.",
        ]
    )
    recommendation = report.get("recommendation")
    if recommendation:
        lines.extend(
            [
                "",
                f"Recommended load-converged surface spacing: "
                f"`h/D = {recommendation['surface_cell_size']:.8f}`.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "No mesh spacing was applied: the available histories do not support "
                "a monotone, stationary load extrapolation.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def analyse(
    samples_root: Path,
    output: Path,
    start: float,
    requested_end: float,
    drag_tolerance: float,
    field_tolerance: float,
) -> dict:
    samples_root = samples_root.resolve()
    metadata = {
        grid: json.loads((samples_root / grid / "grid_metadata.json").read_text(encoding="utf-8"))
        for grid in GRID_ORDER
    }
    end_times = []
    for grid in GRID_ORDER:
        forces_path = samples_root / grid / "forces_history.csv"
        forces = _load_table(forces_path, strings=True)
        end_times.append(float(np.max(_column(forces, "time", forces_path).astype(float))))
    end = min(requested_end, *end_times)
    if end - start < 2.0:
        raise ValueError(
            f"Common analysis window [{start:g}, {end:g}] is shorter than two time units"
        )

    grids = {}
    centreline_profiles = {}
    offaxis_profiles = {}
    for grid in GRID_ORDER:
        grid_dir = samples_root / grid
        force_metrics = _force_metrics(grid_dir / "forces_history.csv", start, end)
        centreline, centreline_profile = _line_metrics(grid_dir / "centreline.csv", start, end)
        offaxis, offaxis_profile = _line_metrics(grid_dir / "offaxis_y075.csv", start, end)
        grids[grid] = {
            "metadata": metadata[grid],
            "forces": force_metrics,
            "centreline": centreline,
            "offaxis_y075": offaxis,
        }
        centreline_profiles[grid] = centreline_profile
        offaxis_profiles[grid] = offaxis_profile

    scalar_specs = {
        "mean_cd": ("forces", "mean_cd", drag_tolerance),
        "std_cd": ("forces", "std_cd", 0.05),
        "recirculation_length": (
            "centreline",
            "recirculation_length",
            field_tolerance,
        ),
        "centreline_u_x1": ("centreline", "u_x1", field_tolerance),
        "centreline_u_x2": ("centreline", "u_x2", field_tolerance),
        "centreline_u_x4": ("centreline", "u_x4", field_tolerance),
        "offaxis_u_x1": ("offaxis_y075", "u_x1", field_tolerance),
        "offaxis_u_x2": ("offaxis_y075", "u_x2", field_tolerance),
        "offaxis_u_x4": ("offaxis_y075", "u_x4", field_tolerance),
    }
    convergence = {}
    for name, (section, metric, tolerance) in scalar_specs.items():
        values = {grid: grids[grid][section][metric] for grid in GRID_ORDER}
        convergence[name] = _spatial_convergence(values, tolerance)
    convergence["centreline_wake_profile"] = _profile_convergence(
        centreline_profiles, field_tolerance
    )
    convergence["offaxis_wake_profile"] = _profile_convergence(offaxis_profiles, field_tolerance)

    load_stationary = all(
        grids[grid]["forces"]["mean_cd_half_window_change"] <= drag_tolerance for grid in GRID_ORDER
    )
    primary = convergence["mean_cd"]
    if not load_stationary:
        load_status = "inconclusive-nonstationary"
    elif not primary["monotone"] or not np.isfinite(primary["observed_order"]):
        load_status = "inconclusive-nonmonotone"
    elif primary["passed"]:
        load_status = "passed"
    else:
        load_status = "failed"

    field_stationary = all(
        grids[grid]["centreline"]["wake_profile_half_window_l2_change"] <= field_tolerance
        and grids[grid]["centreline"]["recirculation_half_window_change"] <= field_tolerance
        and grids[grid]["offaxis_y075"]["wake_profile_half_window_l2_change"] <= field_tolerance
        for grid in GRID_ORDER
    )
    field_names = (
        "recirculation_length",
        "centreline_u_x1",
        "centreline_u_x2",
        "centreline_u_x4",
        "offaxis_u_x1",
        "offaxis_u_x2",
        "offaxis_u_x4",
        "centreline_wake_profile",
        "offaxis_wake_profile",
    )
    if not field_stationary:
        field_status = "inconclusive-nonstationary"
    elif all(convergence[name]["passed"] for name in field_names):
        field_status = "passed"
    else:
        field_status = "failed"

    recommended_h = primary["recommended_surface_cell_size"]
    recommendation = None
    if load_status == "passed" and np.isfinite(recommended_h):
        recommendation = {
            "scope": "mean-drag load convergence",
            "surface_cell_size": float(recommended_h),
            "near_wake_cell_size": float(2.0 * recommended_h),
            "downstream_wake_cell_size": float(4.0 * recommended_h),
            "background_cell_size": 0.5,
            "drag_tolerance": drag_tolerance,
        }

    report = _strict(
        {
            "schema": "openonda-cube-grid-convergence/1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "analysis_window": {"start": start, "end": end},
            "grid_order": list(GRID_ORDER),
            "load_status": load_status,
            "flow_field_status": field_status,
            "load_stationary": load_stationary,
            "flow_field_stationary": field_stationary,
            "tolerances": {
                "mean_drag": drag_tolerance,
                "wake_fields": field_tolerance,
            },
            "grids": grids,
            "convergence": convergence,
            "recommendation": recommendation,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    recommendation_path = output.parent / "grid_recommendation.json"
    if recommendation:
        recommendation_path.write_text(
            json.dumps(recommendation, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    elif recommendation_path.exists():
        recommendation_path.unlink()
    print(json.dumps(report, indent=2, allow_nan=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-fine", help="compact the existing h64 archive")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--destination", type=Path, required=True)
    prepare.add_argument("--line-interval", type=float, default=0.25)
    analysis = subparsers.add_parser("analyse", help="analyse h16, h32, and h64 samples")
    analysis.add_argument("--samples-root", type=Path, required=True)
    analysis.add_argument("--output", type=Path, required=True)
    analysis.add_argument("--window-start", type=float, default=15.0)
    analysis.add_argument("--window-end", type=float, default=20.0)
    analysis.add_argument("--drag-tolerance", type=float, default=0.01)
    analysis.add_argument("--field-tolerance", type=float, default=0.02)
    arguments = parser.parse_args()
    if arguments.command == "prepare-fine":
        prepare_existing_fine(arguments.source, arguments.destination, arguments.line_interval)
    else:
        analyse(
            arguments.samples_root,
            arguments.output,
            arguments.window_start,
            arguments.window_end,
            arguments.drag_tolerance,
            arguments.field_tolerance,
        )


if __name__ == "__main__":
    main()
