#!/usr/bin/env python3
"""Assess cylinder-grid convergence from completed force and velocity records.

Use a common physical-time window and inspect observed order, Richardson/GCI
estimates, force fluctuations and mean wake profiles. Nominal realized XY
spacing, span resolution and statistical uncertainty are reported separately.

Examples
--------
    python postprocess_grid_study.py --statistics-start 50 --statistics-end 100
    python postprocess_grid_study.py --format pdf --output-dir figures
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


CASE_DIR = Path(__file__).resolve().parent
TIME_TOLERANCE = 1.0e-10
VALUE_FLOOR = 1.0e-14


def _read_table(path: Path) -> dict[str, np.ndarray]:
    """Read a CSV sample table and retain only its final restart segment."""

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        names = tuple(reader.fieldnames or ())
        rows = list(reader)
    if not rows or "time" not in names:
        raise ValueError(f"{path} is empty or has no time column")
    numeric: dict[str, np.ndarray] = {}
    for name in names:
        try:
            numeric[name] = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            continue
    if "time" not in numeric:
        raise ValueError(f"{path} has a non-numeric time column")
    time = numeric["time"]
    reset = np.flatnonzero(np.diff(time) < -TIME_TOLERANCE)
    if reset.size:
        start = int(reset[-1]) + 1
        numeric = {name: values[start:] for name, values in numeric.items()}
        time = time[int(reset[-1]) + 1 :]
    keep: list[int] = []
    for index, value in enumerate(time):
        if keep and value < time[keep[-1]] - TIME_TOLERANCE:
            raise ValueError(f"{path} has decreasing times")
        if keep and abs(value - time[keep[-1]]) <= TIME_TOLERANCE:
            keep[-1] = index
        else:
            keep.append(index)
    selected = np.asarray(keep, dtype=np.int64)
    numeric = {name: values[selected] for name, values in numeric.items()}
    time = numeric["time"]
    if len(time) < 2 or np.any(np.diff(time) <= 0.0):
        raise ValueError(f"{path} has no strictly increasing final history")
    if any(not np.all(np.isfinite(values)) for values in numeric.values()):
        raise ValueError(f"{path} contains non-finite samples")
    return numeric


def _discover(samples_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Discover registered cylinder grids and explain incomplete directories."""

    if not samples_root.is_dir():
        raise FileNotFoundError(f"samples root does not exist: {samples_root}")
    cases: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for directory in sorted(path for path in samples_root.iterdir() if path.is_dir()):
        metadata_path = directory / "grid_run.json"
        force_path = directory / "forces_history.csv"
        if not metadata_path.is_file():
            if force_path.is_file():
                excluded.append({"directory": directory.name, "reason": "missing grid_run.json"})
            continue
        if not force_path.is_file():
            excluded.append({"directory": directory.name, "reason": "missing forces_history.csv"})
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            name = str(metadata["case"]).strip()
            cell_size = float(metadata["cell_size"])
            cell_count = int(metadata["cell_count"])
            end_time = float(metadata["end_time"])
            if not name or not math.isfinite(cell_size) or cell_size <= 0.0:
                raise ValueError("invalid case name or cell_size")
            if cell_count <= 0 or not math.isfinite(end_time) or end_time <= 0.0:
                raise ValueError("invalid cell_count or end_time")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            excluded.append({"directory": directory.name, "reason": str(error)})
            continue
        cases.append(
            {
                "case": name,
                "directory": directory,
                "cell_size": cell_size,
                "cell_count": cell_count,
                "end_time": end_time,
                "metadata_path": metadata_path,
                "force_path": force_path,
            }
        )
    names = [case["case"] for case in cases]
    if len(names) != len(set(names)):
        raise ValueError("grid_run.json registers duplicate case names")
    cases.sort(key=lambda case: (-case["cell_size"], case["cell_count"], case["case"]))
    return cases, excluded


def _window(
    time: np.ndarray, values: np.ndarray, start: float, end: float
) -> tuple[np.ndarray, np.ndarray]:
    """Clip a sampled signal and interpolate only the two requested endpoints."""

    if start < time[0] - TIME_TOLERANCE or end > time[-1] + TIME_TOLERANCE:
        raise ValueError("requested interval lies outside the saved history")
    interior = (time > start + TIME_TOLERANCE) & (time < end - TIME_TOLERANCE)
    clipped_time = np.concatenate(([start], time[interior], [end]))
    clipped_values = np.concatenate(
        ([np.interp(start, time, values)], values[interior], [np.interp(end, time, values)])
    )
    return clipped_time, clipped_values


def _mean(time: np.ndarray, values: np.ndarray) -> float:
    """Return the trapezoidal time mean, preserving the input quantity units."""
    return float(np.trapezoid(values, time) / (time[-1] - time[0]))


def _strouhal(time: np.ndarray, values: np.ndarray) -> float | None:
    """Return a screened St estimate, or None when the record is unresolved."""

    if len(time) < 32:
        return None
    uniform = np.linspace(time[0], time[-1], len(time))
    signal = np.interp(uniform, time, values)
    signal -= signal.mean()
    rms = float(np.sqrt(np.mean(signal**2)))
    if rms < 1.0e-7:
        return None
    power = np.abs(np.fft.rfft(signal * np.hanning(len(signal)))) ** 2
    power[0] = 0.0
    index = 1 + int(np.argmax(power[1:]))
    frequency = float(np.fft.rfftfreq(len(signal), uniform[1] - uniform[0])[index])
    if frequency * (uniform[-1] - uniform[0]) < 5.0:
        return None
    return frequency


def _force_statistics(table: dict[str, np.ndarray], start: float, end: float) -> dict[str, Any]:
    """Compute time-weighted force statistics in the common physical window."""

    time = table["time"]
    clipped_time, _ = _window(time, table["drag_coefficient"], start, end)
    result: dict[str, Any] = {"samples": int(len(clipped_time))}
    for name, column in (
        ("drag", "drag_coefficient"),
        ("lift", "lift_coefficient"),
        ("side", "side_force_coefficient"),
    ):
        if column not in table:
            result[f"mean_{name}"] = None
            result[f"rms_{name}"] = None
            result[f"strouhal_{name}"] = None
            continue
        _, values = _window(time, table[column], start, end)
        mean = _mean(clipped_time, values)
        result[f"mean_{name}"] = mean
        result[f"rms_{name}"] = float(np.sqrt(_mean(clipped_time, (values - mean) ** 2)))
        result[f"strouhal_{name}"] = (
            _strouhal(clipped_time, values - mean) if name in ("lift", "side") else None
        )
    midpoint = 0.5 * (start + end)
    first_time, first_drag = _window(time, table["drag_coefficient"], start, midpoint)
    second_time, second_drag = _window(time, table["drag_coefficient"], midpoint, end)
    first_mean = _mean(first_time, first_drag)
    second_mean = _mean(second_time, second_drag)
    result["drag_half_means"] = [first_mean, second_mean]
    result["drag_drift_relative"] = abs(second_mean - first_mean) / max(
        abs(result["mean_drag"]), VALUE_FLOOR
    )
    # Four long contiguous batches expose correlated oscillations and slow drift.
    edges = np.linspace(start, end, 5)
    batch_means = []
    for left, right in zip(edges[:-1], edges[1:]):
        bt, bv = _window(time, table["drag_coefficient"], left, right)
        batch_means.append(_mean(bt, bv))
    result["drag_batch_means"] = batch_means
    result["drag_batch_standard_error"] = float(np.std(batch_means, ddof=1) / 2.0)
    result["drag_batch_95_half_width"] = 3.182446305 * result["drag_batch_standard_error"]
    result["sampling_uncertainty_note"] = (
        "Student-t interval from four contiguous batches; diagnostic only until each "
        "batch contains several shedding cycles and residual batch correlation is negligible."
    )
    return result


def _read_profile(path: Path, start: float, end: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the time-mean streamwise centreline profile."""

    table = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding="utf-8")
    if table.size == 0 or table.dtype.names is None:
        raise ValueError(f"{path} is empty or malformed")
    table = np.atleast_1d(table)
    names = tuple(table.dtype.names)
    required = ("time", "position_x", "velocity_x")
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"{path} is missing {', '.join(missing)}")
    time = np.asarray(table["time"], dtype=np.float64)
    reset = np.flatnonzero(np.diff(time) < -TIME_TOLERANCE)
    if reset.size:
        table = table[int(reset[-1]) + 1 :]
    if any(not np.all(np.isfinite(table[name])) for name in required):
        raise ValueError(f"{path} contains non-finite profile data")
    selected = (table["time"] >= start - TIME_TOLERANCE) & (table["time"] <= end + TIME_TOLERANCE)
    positions = np.asarray(table["position_x"][selected], dtype=np.float64)
    velocities = np.asarray(table["velocity_x"][selected], dtype=np.float64)
    if len(positions) == 0:
        raise ValueError(f"{path} has no samples in the statistics window")
    unique = np.unique(np.round(positions, decimals=10))
    profile = np.asarray(
        [velocities[np.isclose(positions, position)].mean() for position in unique]
    )
    if len(unique) < 2 or np.any(np.diff(unique) <= 0.0):
        raise ValueError(f"{path} has no increasing profile positions")
    return unique, profile


def _difference(value: float | None, reference: float | None) -> float | None:
    """Return absolute relative error; missing values remain unavailable."""
    if value is None or reference is None:
        return None
    return abs(float(value) - float(reference)) / max(abs(float(reference)), VALUE_FLOOR)


def _convergence(grids: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    """Compute a guarded three-level Richardson/GCI estimate."""

    values = [grid["statistics"].get(metric) for grid in grids]
    if len(grids) < 3 or any(value is None for value in values[-3:]):
        return {"available": False, "reason": "fewer than three levels with this metric"}
    coarse, medium, fine = (float(value) for value in values[-3:])
    spacings = [grid.get("effective_cell_size", grid["cell_size"]) for grid in grids[-3:]]
    ratio_a = spacings[0] / spacings[1]
    ratio_b = spacings[1] / spacings[2]
    result: dict[str, Any] = {
        "available": False,
        "coarse_to_medium_ratio": ratio_a,
        "medium_to_fine_ratio": ratio_b,
        "finest_pair_relative_change": _difference(medium, fine),
    }
    if min(ratio_a, ratio_b) < 1.1:
        result["reason"] = "refinement ratios below 1.1 cannot resolve grid error reliably"
        return result
    if not np.isclose(ratio_a, ratio_b, rtol=5.0e-3, atol=1.0e-12):
        result["reason"] = "the three finest refinement ratios are not equal"
        return result
    delta_coarse = medium - coarse
    delta_fine = fine - medium
    if delta_coarse * delta_fine <= 0.0 or abs(delta_fine) <= VALUE_FLOOR:
        result["reason"] = "the three finest values are not monotone"
        return result
    order = math.log(abs(delta_coarse / delta_fine)) / math.log(ratio_b)
    if not math.isfinite(order) or order <= 0.0:
        result["reason"] = "the observed order is not positive and finite"
        return result
    if order > 4.0:
        result["reason"] = "observed order exceeds four; possible cancellation or sampling noise"
        result["observed_order"] = order
        return result
    if metric == "mean_drag":
        uncertainty = sum(
            grid["statistics"].get("drag_batch_95_half_width", 0.0) for grid in grids[-2:]
        )
        result["combined_sampling_half_width"] = uncertainty
        if uncertainty >= abs(delta_fine):
            result["reason"] = (
                "finest-grid difference is not resolved above batch sampling uncertainty"
            )
            return result
    extrapolated = fine + delta_fine / (ratio_b**order - 1.0)
    result.update(
        available=True,
        reason=None,
        observed_order=order,
        richardson_extrapolated=extrapolated,
        fine_grid_gci=1.25 * abs(extrapolated - fine) / max(abs(fine), VALUE_FLOOR),
        coarse_grid_gci=1.25
        * abs(delta_coarse)
        / ((ratio_a**order - 1.0) * max(abs(medium), VALUE_FLOOR)),
        asymptotic_ratio=abs(delta_coarse / delta_fine)
        / ratio_b**order
        * abs(fine)
        / max(abs(medium), VALUE_FLOOR),
        qualification="Three-grid asymptotic ratio is algebraically dependent on fitted order; compare adjacent triplets and temporal/span controls before accepting.",
    )
    return result


def _context(case: dict[str, Any], solution_root: Path) -> dict[str, Any]:
    """Read saved native metadata when present; never infer settings from names."""

    metadata_path = solution_root / case["directory"].name / "fvm_metadata.json"
    context: dict[str, Any] = {"metadata": None, "metadata_path": None, "warnings": []}
    if not metadata_path.is_file():
        context["warnings"].append("native fvm_metadata.json is absent")
        return context
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    lifecycle = metadata.get("lifecycle", {})
    if lifecycle.get("status") != "complete":
        raise ValueError(f"{metadata_path} does not mark the run complete")
    mesh_path = metadata_path.parent / "fvm" / "mesh.npz"
    if mesh_path.is_file():
        with np.load(mesh_path, allow_pickle=False) as mesh:
            provenance = json.loads(str(mesh["metadata"]))
            generation = provenance.get("mesh_generation", {})
            levels = generation.get("extrusion_levels", [])
            nominal = generation.get("resolved_surface_patch_sizes", {}).get("cylinder")
            context["mesh_realization"] = {
                "effective_wall_lattice": nominal,
                "domain": generation.get("domain"),
                "span_layers": len(levels) - 1 if levels else None,
                "span": levels[-1] - levels[0] if levels else None,
                "section_cells": generation.get("section_cells"),
                "global_cells": provenance.get("n_cells"),
                "definition": "nominal octree cylinder size before projection, not all physical cell edges",
            }
    context["metadata"] = metadata
    context["metadata_path"] = str(metadata_path.resolve())
    return context


def _styles(grids: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Assign named thesis colours and distinct markers to the ordered grids."""
    from openonda import plotting as theme

    names = ("RefGray", "FVMorange", "VPMpurple", "TUDdark", "TUDcyan")
    markers = ("v", "s", "D", "o", "^")
    return {
        grid["case"]: {
            "color": theme.COLORS[names[index % len(names)]],
            "marker": markers[index % len(markers)],
            "label": grid["case"].replace("_", " ").capitalize(),
        }
        for index, grid in enumerate(grids)
    }


def _save_plot(figure: Any, destination: Path, formats: tuple[str, ...]) -> None:
    """Validate fixed thesis dimensions and save each requested format without cropping."""
    from openonda import plotting as theme

    theme.fit_thesis_y_label_margins(figure, figure.axes)
    theme.validate_thesis_figure(figure, figure.axes)
    for output_format in formats:
        figure.savefig(
            destination.with_suffix(f".{output_format}"),
            dpi=theme.DEFAULT_DPI,
            format=output_format,
            bbox_inches=None,
            facecolor="white",
        )
    import matplotlib.pyplot as plt

    plt.close(figure)


def _plot_force_metrics(
    report: dict[str, Any], destination: Path, formats: tuple[str, ...]
) -> None:
    """Plot mean drag and transverse force RMS versus requested spacing divided by D."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    from openonda import plotting as theme

    theme.set_thesis_style()
    grids = report["grids"]
    x = np.asarray([grid["cell_size"] for grid in grids])
    figure, axes = plt.subplots(2, 1, figsize=(12.5 * theme.CM, 8.3 * theme.CM), sharex=True)
    theme.centered_subplots_adjust(figure, outer=0.2, bottom=0.22, top=0.96, hspace=0.35)
    for axis, metrics, ylabel in zip(
        axes,
        (("mean_drag",), ("rms_lift", "rms_side")),
        (r"$\overline{C_D}$", r"$\sigma(C_\perp)$"),
        strict=True,
    ):
        for index, metric in enumerate(metrics):
            axis.plot(
                x,
                [grid["statistics"].get(metric, np.nan) for grid in grids],
                marker=("o", "s")[index],
                linestyle=("-", "--")[index],
                color=theme.COLORS[("TUDdark", "VPMpurple")[index]],
                label=(r"$C_L$", r"$C_S$")[index],
            )
        axis.set_ylabel(ylabel)
        axis.yaxis.set_major_locator(MaxNLocator(3))
        if len(metrics) > 1:
            axis.legend(frameon=False, ncol=2, loc="best")
    axes[-1].set_xlabel(r"Target $h/D$")
    axes[-1].set_xticks(x)
    axes[-1].invert_xaxis()
    _save_plot(figure, destination, formats)


def _plot_by_cells(report: dict[str, Any], destination: Path, formats: tuple[str, ...]) -> None:
    """Plot force statistics versus global fluid-cell count."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from openonda import plotting as theme

    theme.set_thesis_style()
    grids = report["grids"]
    x = np.asarray([grid["cell_count"] / 1000.0 for grid in grids])
    figure, axis = plt.subplots(figsize=(12.5 * theme.CM, 7.0 * theme.CM))
    theme.centered_subplots_adjust(figure, outer=0.2, bottom=0.23, top=0.95)
    for metric, label, color, marker in (
        ("mean_drag", r"$\overline{C_D}$", "TUDdark", "o"),
        ("rms_lift", r"$\sigma(C_L)$", "VPMpurple", "s"),
    ):
        axis.plot(
            x,
            [grid["statistics"].get(metric, np.nan) for grid in grids],
            marker=marker,
            color=theme.COLORS[color],
            label=label,
        )
    axis.set_xlabel(r"Fluid cells [$10^3$]")
    axis.set_ylabel("Statistic")
    axis.legend(frameon=False)
    _save_plot(figure, destination, formats)


def _plot_histories(
    report: dict[str, Any],
    tables: dict[str, dict[str, np.ndarray]],
    destination: Path,
    formats: tuple[str, ...],
) -> None:
    """Plot unshifted drag and lift histories over the common interval in seconds."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from openonda import plotting as theme

    theme.set_thesis_style()
    start, end = report["statistics_window"]
    figure, axes = plt.subplots(2, 1, figsize=(12.5 * theme.CM, 10.5 * theme.CM), sharex=True)
    theme.centered_subplots_adjust(figure, outer=0.2, bottom=0.17, top=0.70, hspace=0.32)
    styles = _styles(report["grids"])
    for grid in report["grids"]:
        table = tables[grid["case"]]
        for axis, column in zip(axes, ("drag_coefficient", "lift_coefficient"), strict=True):
            time, values = _window(table["time"], table[column], start, end)
            axis.plot(
                time,
                values,
                **styles[grid["case"]],
                markersize=3,
                markevery=max(1, len(time) // 10),
            )
    axes[0].set_ylabel(r"$C_D$")
    axes[1].set_ylabel(r"$C_L$")
    axes[1].set_xlabel(r"$t$ [s]")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        handlelength=1.5,
    )
    _save_plot(figure, destination, formats)


def _plot_profiles(
    report: dict[str, Any],
    profiles: dict[str, dict[str, np.ndarray]],
    destination: Path,
    formats: tuple[str, ...],
) -> None:
    """Plot time-mean streamwise velocity divided by U against centreline x/D."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from openonda import plotting as theme

    theme.set_thesis_style()
    figure, axis = plt.subplots(figsize=(12.5 * theme.CM, 7.0 * theme.CM))
    theme.centered_subplots_adjust(figure, outer=0.2, bottom=0.24, top=0.92)
    styles = _styles(report["grids"])
    for grid in report["grids"]:
        profile = profiles.get(grid["case"])
        if profile is not None:
            # Cell interpolation inside the solid is not a fluid velocity.
            for index, fluid in enumerate((profile["position"] < -0.5, profile["position"] > 0.5)):
                style = dict(styles[grid["case"]])
                if index:
                    style["label"] = "_nolegend_"
                axis.plot(
                    profile["position"][fluid],
                    profile["velocity"][fluid],
                    **style,
                    markersize=3,
                    markevery=12,
                )
    from matplotlib.ticker import MaxNLocator

    # Keep corner tick labels apart at the prescribed thesis font size.
    axis.xaxis.set_major_locator(MaxNLocator(4, prune="lower"))
    axis.set_xlabel(r"$x/D$")
    axis.set_ylabel(r"$\overline{u_x}/U_\infty$")
    axis.legend(frameon=False, ncol=2)
    _save_plot(figure, destination, formats)


def _write_csv(report: dict[str, Any], destination: Path) -> None:
    """Write per-grid spacing, force statistics and profile errors as a derived table."""
    fields = (
        "case",
        "cell_size",
        "cell_count",
        "end_time",
        "mean_drag",
        "rms_drag",
        "rms_lift",
        "rms_side",
        "strouhal_lift",
        "strouhal_side",
        "drag_drift_relative",
        "difference_to_reference_mean_drag",
    )
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for grid in report["grids"]:
            statistics = grid["statistics"]
            writer.writerow(
                {field: grid.get(field, statistics.get(field)) for field in fields}
                | {
                    "difference_to_reference_mean_drag": grid["difference_to_reference"][
                        "mean_drag"
                    ]
                }
            )


def _write_markdown(report: dict[str, Any], destination: Path) -> None:
    """Write the statistics window, grid comparison and qualification limits."""
    start, end = report["statistics_window"]
    lines = [
        "# Cylinder grid-convergence report",
        "",
        f"Common statistics window: `{start:g} <= t <= {end:g}`.",
        f"Reference grid: `{report['reference_case']}` (finest available target spacing).",
        "",
        "Differences are relative to the selected finest grid and are not exact discretisation errors.",
        "",
        "| Case | Target h/D | Cells | Mean Cd | RMS Cd | RMS Cl | St (lift) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for grid in report["grids"]:
        values = grid["statistics"]
        lines.append(
            f"| {grid['case']} | {grid['cell_size']:.6g} | {grid['cell_count']:,} | "
            f"{values['mean_drag']:.6g} | {values['rms_drag']:.6g} | {values['rms_lift']:.6g} | "
            f"{values['strouhal_lift'] if values['strouhal_lift'] is not None else 'n/a'} |"
        )
    lines.extend(["", "## Interpretation", "", report["assessment"], ""])
    for metric, result in report["convergence"].items():
        if result.get("reason"):
            summary = str(result["reason"])
        else:
            summary = (
                f"observed order {result['observed_order']:.4g}, "
                f"fine-grid GCI {result['fine_grid_gci']:.4g}"
            )
        lines.append(f"- `{metric}`: {summary}")
    if report["excluded_cases"]:
        lines.extend(["", "## Excluded directories", ""])
        lines.extend(
            f"- `{item['directory']}`: {item['reason']}" for item in report["excluded_cases"]
        )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyse_grid_convergence(
    samples_root: str | Path,
    output_dir: str | Path,
    *,
    statistics_start: float | None = None,
    statistics_end: float | None = None,
    formats: tuple[str, ...] = ("png",),
    solution_root: str | Path | None = None,
) -> dict[str, Any]:
    """Compare completed grids over one common physical-time interval.

    Parameters
    ----------
    samples_root : str or pathlib.Path
        Directory containing registered per-grid force and line samples.
    output_dir : str or pathlib.Path
        Figure directory; tables and provenance are written under auxiliary/.
    statistics_start, statistics_end : float or None
        Averaging limits in s. None selects the final half of the common record.
    formats : tuple of str
        Figure formats, png and/or pdf. PNG is the default.
    solution_root : str or pathlib.Path or None
        Native solver records. None uses the samples root's sibling solution/.

    Returns
    -------
    dict
        Force statistics, mean-profile errors, convergence estimates, coverage
        exclusions and source hashes. Input records are not modified.

    Raises
    ------
    ValueError
        If fewer than two completed grids are usable, the averaging window is
        unsupported, or a requested figure format is invalid.
    """

    samples_root = Path(samples_root)
    output_dir = Path(output_dir)
    solution_root = (
        Path(solution_root) if solution_root is not None else samples_root.parent / "solution"
    )
    if not formats or any(value not in ("png", "pdf") for value in formats):
        raise ValueError("formats must contain png and/or pdf")
    cases, excluded = _discover(samples_root)
    tables: dict[str, dict[str, np.ndarray]] = {}
    usable: list[dict[str, Any]] = []
    for case in cases:
        try:
            table = _read_table(case["force_path"])
            if not np.isclose(table["time"][-1], case["end_time"], atol=TIME_TOLERANCE, rtol=0.0):
                raise ValueError("force history does not reach registered end_time")
            case["context"] = _context(case, solution_root)
            effective = case["context"].get("mesh_realization", {}).get("effective_wall_lattice")
            if effective is not None:
                case["effective_cell_size"] = float(effective)
        except (ValueError, json.JSONDecodeError) as error:
            excluded.append({"directory": case["directory"].name, "reason": str(error)})
            continue
        tables[case["case"]] = table
        usable.append(case)
    cases = usable
    if len(cases) < 2:
        raise ValueError("at least two completed cylinder grids are required")
    common_start = max(float(table["time"][0]) for table in tables.values())
    common_end = min(float(table["time"][-1]) for table in tables.values())
    start = (
        0.5 * (common_start + common_end) if statistics_start is None else float(statistics_start)
    )
    end = common_end if statistics_end is None else float(statistics_end)
    if start < common_start - TIME_TOLERANCE or end > common_end + TIME_TOLERANCE or end <= start:
        raise ValueError(
            f"statistics interval [{start:g}, {end:g}] is outside common interval [{common_start:g}, {common_end:g}]"
        )
    grids: list[dict[str, Any]] = []
    for case in cases:
        grids.append({**case, "statistics": _force_statistics(tables[case["case"]], start, end)})
    grids.sort(key=lambda grid: (-grid["cell_size"], grid["cell_count"], grid["case"]))
    reference = min(grids, key=lambda grid: (grid["cell_size"], -grid["cell_count"], grid["case"]))
    for grid in grids:
        grid["difference_to_reference"] = {
            metric: _difference(grid["statistics"].get(metric), reference["statistics"].get(metric))
            for metric in (
                "mean_drag",
                "rms_drag",
                "rms_lift",
                "rms_side",
                "strouhal_lift",
                "strouhal_side",
            )
        }
    profiles: dict[str, dict[str, np.ndarray]] = {}
    for grid in grids:
        path = grid["directory"] / "centreline.csv"
        if path.is_file():
            try:
                position, velocity = _read_profile(path, start, end)
                profiles[grid["case"]] = {"position": position, "velocity": velocity}
            except ValueError:
                pass
    profile_report: dict[str, Any] = {"available": False, "comparisons": {}}
    reference_profile = profiles.get(reference["case"])
    if reference_profile is not None:
        reference_position = reference_profile["position"]
        reference_velocity = reference_profile["velocity"]
        profile_report.update(
            available=True,
            position=reference_position.tolist(),
            velocity=reference_velocity.tolist(),
        )
        for grid in grids:
            candidate = profiles.get(grid["case"])
            if candidate is None:
                profile_report["comparisons"][grid["case"]] = {
                    "available": False,
                    "reason": "centreline.csv is absent or has no common samples",
                }
                continue
            lower = max(float(candidate["position"][0]), float(reference_position[0]))
            upper = min(float(candidate["position"][-1]), float(reference_position[-1]))
            selected = (reference_position >= lower) & (reference_position <= upper)
            position = reference_position[selected]
            if len(position) < 2:
                profile_report["comparisons"][grid["case"]] = {
                    "available": False,
                    "reason": "centreline profiles have insufficient overlap",
                }
                continue
            difference = (
                np.interp(position, candidate["position"], candidate["velocity"])
                - reference_velocity[selected]
            )
            reference_norm = float(
                np.sqrt(np.trapezoid(reference_velocity[selected] ** 2, position))
            )
            difference_norm = float(np.sqrt(np.trapezoid(difference**2, position)))
            profile_report["comparisons"][grid["case"]] = {
                "available": True,
                "position_interval": [float(position[0]), float(position[-1])],
                "relative_l2": difference_norm / max(reference_norm, VALUE_FLOOR),
            }
    report: dict[str, Any] = {
        "schema": "openonda-cylinder-grid-convergence/1",
        "statistics_window": [start, end],
        "reference_case": reference["case"],
        "wall_cell_size_definition": "target cylinder surface cell size supplied as setup.py --dx, in D",
        "grids": [
            {
                "case": grid["case"],
                "cell_size": grid["cell_size"],
                "cell_count": grid["cell_count"],
                "effective_cell_size": grid.get("effective_cell_size"),
                "end_time": grid["end_time"],
                "statistics": grid["statistics"],
                "difference_to_reference": grid["difference_to_reference"],
                "context": grid["context"],
            }
            for grid in grids
        ],
        "convergence": {
            metric: _convergence(grids, metric)
            for metric in ("mean_drag", "rms_drag", "rms_lift", "rms_side")
        },
        "overlapping_triplets": [
            {
                "cases": [grid["case"] for grid in grids[index : index + 3]],
                "convergence": {
                    metric: _convergence(grids[index : index + 3], metric)
                    for metric in ("mean_drag", "rms_drag", "rms_lift")
                },
            }
            for index in range(max(0, len(grids) - 2))
        ],
        "profiles": {"centreline": profile_report},
        "excluded_cases": excluded,
        "assessment": "Grid independence is not established by these statistics alone; inspect the finest-pair changes, averaging drift, and the separate time-step convergence requirement.",
        "provenance": {
            "postprocessor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "input_sha256": {
                str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                for case in cases
                for path in (
                    case["metadata_path"],
                    case["force_path"],
                    case["directory"] / "centreline.csv",
                )
                if path.is_file()
            },
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    auxiliary = output_dir / "auxiliary"
    auxiliary.mkdir(exist_ok=True)
    (auxiliary / "grid_convergence.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    _write_csv(report, auxiliary / "grid_convergence.csv")
    _write_markdown(report, auxiliary / "grid_convergence.md")
    _plot_force_metrics(report, output_dir / "grid_convergence.png", formats)
    _plot_by_cells(report, output_dir / "grid_convergence_by_cells.png", formats)
    _plot_histories(report, tables, output_dir / "grid_convergence_histories.png", formats)
    _plot_profiles(report, profiles, output_dir / "grid_convergence_profiles.png", formats)
    return report


def main() -> None:
    """Parse a common statistics window and write the convergence assessment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-root", type=Path, default=CASE_DIR / "samples")
    parser.add_argument("--solution-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=CASE_DIR / "figures")
    parser.add_argument("--statistics-start", type=float)
    parser.add_argument("--statistics-end", type=float)
    parser.add_argument("--format", choices=("png", "pdf", "both"), default="png")
    arguments = parser.parse_args()
    formats = ("png", "pdf") if arguments.format == "both" else (arguments.format,)
    report = analyse_grid_convergence(
        arguments.samples_root,
        arguments.output_dir,
        statistics_start=arguments.statistics_start,
        statistics_end=arguments.statistics_end,
        formats=formats,
        solution_root=arguments.solution_root,
    )
    print(
        json.dumps(
            {
                "reference_case": report["reference_case"],
                "statistics_window": report["statistics_window"],
                "output_dir": str(arguments.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
