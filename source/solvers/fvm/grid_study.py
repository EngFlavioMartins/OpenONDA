"""Grid-convergence reports from completed FVM force and line samples."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import trapezoid


def _table(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    data = np.atleast_1d(data)
    if data.size == 0 or not data.dtype.names:
        raise ValueError(f"Empty or malformed grid-study sample: {path}")
    if "time" in data.dtype.names and len(data) > 1:
        time = np.asarray(data["time"], dtype=float)
        resets = np.flatnonzero(np.diff(time) < -1.0e-12)
        if resets.size:
            data = data[int(resets[-1]) + 1 :]
    return data


def _values(table: np.ndarray, name: str, path: Path, dtype=float) -> np.ndarray:
    if name not in (table.dtype.names or ()):
        raise ValueError(f"{path} has no {name!r} column")
    return np.asarray(table[name], dtype=dtype)


def _mean(time: np.ndarray, values: np.ndarray) -> float:
    if len(time) == 1 or time[-1] == time[0]:
        return float(values[-1])
    return float(trapezoid(values, time) / (time[-1] - time[0]))


def _strouhal(time: np.ndarray, lift: np.ndarray) -> float | None:
    lift = lift - _mean(time, lift)
    if len(lift) < 16 or np.max(np.abs(lift)) < 1.0e-10:
        return None
    sample_time = np.linspace(time[0], time[-1], len(time))
    signal = np.interp(sample_time, time, lift)
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
    frequency = np.fft.rfftfreq(len(signal), sample_time[1] - sample_time[0])
    if len(spectrum) < 2:
        return None
    return float(frequency[1 + np.argmax(spectrum[1:])])


def _force_metrics(path: Path, start: float, end: float) -> dict[str, float | None]:
    table = _table(path)
    time = _values(table, "time", path)
    mask = (time >= start - 1.0e-10) & (time <= end + 1.0e-10)
    if np.count_nonzero(mask) < 8:
        raise ValueError(f"Too few force samples in the common window for {path}")
    time = time[mask]
    drag = _values(table, "drag_coefficient", path)[mask]
    lift = _values(table, "lift_coefficient", path)[mask]
    side = _values(table, "side_force_coefficient", path)[mask]
    mean_drag = _mean(time, drag)
    mean_lift = _mean(time, lift)
    mean_side = _mean(time, side)
    return {
        "mean_drag": mean_drag,
        "rms_drag": _mean(time, (drag - mean_drag) ** 2) ** 0.5,
        "rms_lift": _mean(time, (lift - mean_lift) ** 2) ** 0.5,
        "rms_side": _mean(time, (side - mean_side) ** 2) ** 0.5,
        "strouhal": _strouhal(time, lift),
        "samples": int(len(time)),
    }


def _profile(path: Path, start: float, end: float) -> tuple[np.ndarray, np.ndarray]:
    table = _table(path)
    time = _values(table, "time", path)
    mask = (time >= start - 1.0e-10) & (time <= end + 1.0e-10)
    coordinates = np.column_stack(
        [_values(table, f"position_{axis}", path) for axis in ("x", "y", "z")]
    )
    axis = int(np.argmax(np.ptp(coordinates[mask], axis=0)))
    position = coordinates[mask, axis]
    velocity = _values(table, "velocity_x", path)[mask]
    points, inverse = np.unique(position, return_inverse=True)
    return points, np.bincount(inverse, weights=velocity) / np.bincount(inverse)


def _convergence(values: list[float], spacing: list[float]) -> dict[str, Any]:
    coarse, medium, fine = values[-3:]
    h_coarse, h_medium, h_fine = spacing[-3:]
    ratio = 0.5 * (h_coarse / h_medium + h_medium / h_fine)
    delta_coarse = medium - coarse
    delta_fine = fine - medium
    monotone = bool(delta_coarse * delta_fine > 0.0)
    order = extrapolated = gci = None
    if monotone and delta_coarse != 0.0 and delta_fine != 0.0 and ratio > 1.0:
        order_value = np.log(abs(delta_coarse / delta_fine)) / np.log(ratio)
        if np.isfinite(order_value) and order_value > 0.0:
            order = float(order_value)
            extrapolated = float(fine + delta_fine / (ratio**order - 1.0))
            scale = max(abs(extrapolated), 1.0e-30)
            gci = float(1.25 * abs(extrapolated - fine) / scale)
    return {
        "monotone": monotone,
        "refinement_ratio": float(ratio),
        "observed_order": order,
        "richardson_extrapolated": extrapolated,
        "fine_grid_gci": gci,
        "medium_to_fine_change": float(abs(fine - medium) / max(abs(fine), abs(medium), 1.0e-30)),
    }


def _plot(report: dict[str, Any], destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grids = report["grids"]
    spacing = np.asarray([grid["cell_size"] for grid in grids])
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 6.5), constrained_layout=True)
    specifications = (
        ("mean_drag", "Mean drag coefficient"),
        ("rms_lift", "Lift RMS"),
        ("strouhal", "Strouhal number"),
    )
    for axis, (metric, label) in zip(axes.flat[:3], specifications, strict=True):
        values = [grid["forces"][metric] for grid in grids]
        if any(value is not None for value in values):
            axis.plot(spacing, values, "o-", linewidth=1.4)
        axis.set_xscale("log", base=2)
        axis.invert_xaxis()
        axis.set_xlabel("Wall cell size, h/D")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    cell_counts = [grid["cell_count"] for grid in grids]
    axes[1, 1].plot(spacing, cell_counts, "o-", linewidth=1.4)
    axes[1, 1].set_xscale("log", base=2)
    axes[1, 1].set_yscale("log")
    axes[1, 1].invert_xaxis()
    axes[1, 1].set_xlabel("Wall cell size, h/D")
    axes[1, 1].set_ylabel("Cells")
    axes[1, 1].grid(alpha=0.25)
    figure.suptitle("FVM grid-convergence study")
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def analyse_grid_study(samples_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    """Aggregate at least three registered FVM runs into a grid-convergence report.

    Parameters
    ----------
    samples_root : str or pathlib.Path
        Directory containing one case subdirectory per grid, each with
        ``grid_run.json`` and ``forces_history.csv``.
    output_root : str or pathlib.Path
        Destination for ``grid_study.json``, ``grid_study.csv``,
        ``grid_study.md``, and ``grid_study.png``. It is created if needed.

    Returns
    -------
    dict[str, Any]
        JSON-compatible report containing grid metadata, force statistics,
        observed-order/GCI estimates, and common profile errors. Time is in
        seconds; force coefficients and normalized errors are dimensionless.

    Raises
    ------
    ValueError
        If fewer than three runs exist or a required sample is malformed or
        has too few rows in the common averaging window.

    Notes
    -----
    Statistics use the second half of the shortest completed physical-time
    interval. The three finest grids determine Richardson/GCI estimates.
    """
    samples_root = Path(samples_root)
    metadata = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in samples_root.glob("*/grid_run.json")
    ]
    if len(metadata) < 3:
        raise ValueError("A grid study requires at least three completed mesh sizes")
    metadata.sort(key=lambda item: item["cell_size"], reverse=True)
    end = min(float(item["end_time"]) for item in metadata)
    start = 0.5 * end
    grids = []
    for item in metadata:
        case = item["case"]
        sample_dir = samples_root / case
        grids.append(
            {
                **item,
                "forces": _force_metrics(sample_dir / "forces_history.csv", start, end),
            }
        )

    spacing = [float(grid["cell_size"]) for grid in grids]
    convergence = {}
    for metric in ("mean_drag", "rms_drag", "rms_lift", "rms_side", "strouhal"):
        values = [grid["forces"][metric] for grid in grids]
        if all(value is not None for value in values[-3:]):
            convergence[metric] = _convergence([float(value) for value in values], spacing)

    profile_names = set(grids[0].get("profiles", ()))
    for grid in grids[1:]:
        profile_names &= set(grid.get("profiles", ()))
    profiles = {}
    for name in sorted(profile_names):
        curves = [
            _profile(samples_root / grid["case"] / f"{name}.csv", start, end) for grid in grids[-3:]
        ]
        fine_x, fine_u = curves[-1]
        errors = []
        for x, velocity in curves[:-1]:
            lower = max(x[0], fine_x[0])
            upper = min(x[-1], fine_x[-1])
            mask = (fine_x >= lower) & (fine_x <= upper)
            difference = np.interp(fine_x[mask], x, velocity) - fine_u[mask]
            errors.append(
                float(np.linalg.norm(difference) / max(np.linalg.norm(fine_u[mask]), 1.0e-30))
            )
        profiles[name] = {
            "coarse_to_fine_l2": errors[0],
            "medium_to_fine_l2": errors[1],
        }

    report = {
        "schema": "openonda-fvm-grid-study/1",
        "analysis_window": {"start": start, "end": end},
        "grids": grids,
        "convergence": convergence,
        "profiles": profiles,
    }
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "grid_study.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (output_root / "grid_study.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("case", "cell_size", "cell_count", "mean_drag", "rms_lift", "strouhal"))
        for grid in grids:
            writer.writerow(
                (
                    grid["case"],
                    grid["cell_size"],
                    grid["cell_count"],
                    grid["forces"]["mean_drag"],
                    grid["forces"]["rms_lift"],
                    grid["forces"]["strouhal"],
                )
            )
    drag = convergence.get("mean_drag", {})
    markdown = [
        "# FVM grid-convergence study",
        "",
        f"Common statistics window: `{start:g} <= t <= {end:g}`.",
        "",
        "| Case | h/D | Cells | Mean Cd | RMS Cl | St |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for grid in grids:
        force = grid["forces"]
        st = "n/a" if force["strouhal"] is None else f"{force['strouhal']:.6g}"
        markdown.append(
            f"| {grid['case']} | {grid['cell_size']:.8g} | {grid['cell_count']:,} | "
            f"{force['mean_drag']:.8g} | {force['rms_lift']:.8g} | {st} |"
        )
    markdown.extend(
        [
            "",
            f"Mean-drag observed order: `{drag.get('observed_order')}`.",
            f"Mean-drag fine-grid GCI: `{drag.get('fine_grid_gci')}`.",
            "",
        ]
    )
    (output_root / "grid_study.md").write_text("\n".join(markdown), encoding="utf-8")
    _plot(report, output_root / "grid_study.png")
    return report


def update_grid_study(
    solver: Any,
    cell_size: float,
    *,
    profiles: tuple[str, ...] = (),
) -> dict[str, Any] | None:
    """Register one completed FVM run and refresh its grid study when possible.

    Parameters
    ----------
    solver : FVMSolver
        Completed solver. Pending asynchronous output is flushed and MPI-owned
        cell counts are reduced before metadata is written.
    cell_size : float
        Representative grid spacing, normally nondimensionalized by the
        reference body length (for example ``h/D``).
    profiles : tuple[str, ...], optional
        Line-sampler basenames to compare when present on every registered grid.

    Returns
    -------
    dict[str, Any] or None
        Refreshed report on the root rank once at least three runs exist;
        otherwise ``None``. Non-root ranks also return ``None``.

    Notes
    -----
    Writes ``grid_run.json`` below the solver's samples directory and performs
    an MPI barrier before returning.
    """
    solver.flush_output()
    if solver.parallel.is_partitioned:
        cell_count = int(solver.parallel.global_sum(int(solver.parallel.n_owned)))
    else:
        cell_count = int(solver.mesh_data["n_cells"])
    report = None
    failure = None
    try:
        if solver.parallel.is_root:
            samples_dir = Path(solver.samples_dir)
            samples_dir.mkdir(parents=True, exist_ok=True)
            metadata = {
                "schema": "openonda-fvm-grid-run/1",
                "case": solver.setup.case_name,
                "cell_size": float(cell_size),
                "cell_count": cell_count,
                "end_time": float(solver.time),
                "profiles": list(profiles),
            }
            (samples_dir / "grid_run.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
            completed = list(samples_dir.parent.glob("*/grid_run.json"))
            if len(completed) >= 3:
                report = analyse_grid_study(samples_dir.parent, Path(solver.solution_dir).parent)
    except BaseException as error:
        failure = error
    solver._collective_io_failure(failure, "grid-study output")
    return report


__all__ = ["analyse_grid_study", "update_grid_study"]
