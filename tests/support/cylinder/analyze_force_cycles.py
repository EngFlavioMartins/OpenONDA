"""Freeze and compare complete cylinder force cycles without editing live samples.

Run from the repository root, for example::

    python -m tests.support.cylinder.analyze_force_cycles --end 44

The reports and source snapshots stay under tests/support/cylinder. Whole-window
ranges are reported separately because mean-drag growth can inflate them. Each
complete oscillation requires a peak between two troughs inside the requested
window. No startup spike is masked or removed from the source CSV.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import find_peaks

CASE = Path(__file__).resolve().parents[3] / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow"
FIELDS = ("drag_coefficient", "lift_coefficient")


def freeze_csv(source: Path, destination: Path) -> tuple[np.ndarray, dict]:
    """Copy complete published lines from one byte snapshot of an appending CSV."""
    content = source.read_bytes()
    if not content.endswith(b"\n"):
        content = content[: content.rfind(b"\n") + 1]
    values = np.atleast_1d(np.genfromtxt(io.BytesIO(content), delimiter=",", names=True))
    for field in ("time", *FIELDS):
        if not np.all(np.isfinite(values[field])):
            raise ValueError(f"Non-finite {field} in {source}")
    if np.any(np.diff(values["time"]) <= 0):
        raise ValueError(f"Non-increasing force times in {source}")
    destination.write_bytes(content)
    return values, {
        "live_source": str(source),
        "snapshot": str(destination),
        "sha256": hashlib.sha256(content).hexdigest(),
        "sample_count": len(values),
        "first_time": float(values["time"][0]),
        "last_time": float(values["time"][-1]),
    }


def complete_cycles(values: np.ndarray, field: str, coefficient_factor: float) -> list[dict]:
    """Measure trough-bracketed oscillations and resolve their force components."""
    time, force = values["time"], values[field]
    is_drag = field == "drag_coefficient"
    axis = "x" if is_drag else "y"
    minimum_period, maximum_period = (1.5, 5.0) if is_drag else (4.0, 8.0)
    separation = max(1, round(minimum_period / np.median(np.diff(time))))
    peaks, _ = find_peaks(force, distance=separation, prominence=1e-4)
    troughs, _ = find_peaks(-force, distance=separation, prominence=1e-4)
    records = []
    for peak in peaks:
        lower, upper = troughs[troughs < peak], troughs[troughs > peak]
        if not len(lower) or not len(upper):
            continue
        left, right = lower[-1], upper[0]
        period = time[right] - time[left]
        baseline = np.interp(time[peak], time[[left, right]], force[[left, right]])
        if not minimum_period <= period <= maximum_period or force[peak] <= baseline:
            continue
        indices = np.arange(left, right + 1)
        maximum = indices[np.argmax(force[indices])]
        minimum = indices[np.argmin(force[indices])]
        record = {
            "left_trough_time": float(time[left]),
            "peak_time": float(time[peak]),
            "right_trough_time": float(time[right]),
            "period": float(period),
            "peak_to_peak_raw": float(force[maximum] - force[minimum]),
            "peak_to_peak_drift_corrected": float(force[peak] - baseline),
            "minimum_coefficient": float(force[minimum]),
            "maximum_coefficient": float(force[maximum]),
        }
        for component in ("pressure", "viscous"):
            coefficient = coefficient_factor * values[f"{component}_force_{axis}"]
            component_baseline = np.interp(
                time[peak], time[[left, right]], coefficient[[left, right]]
            )
            record[f"{component}_contribution_at_raw_extrema"] = float(
                coefficient[maximum] - coefficient[minimum]
            )
            record[f"{component}_contribution_drift_corrected"] = float(
                coefficient[peak] - component_baseline
            )
        records.append(record)
    return records


def window_statistics(values, cycles, start, end, coefficient_factor):
    """Measure a fully covered common window and complete cycles within it."""
    if values["time"][0] > start + 1e-8 or values["time"][-1] < end - 1e-8:
        raise ValueError(f"History does not cover [{start}, {end}]")
    time = values["time"]
    times = np.r_[start, time[(time > start) & (time < end)], end]
    result = {}
    for field in FIELDS:
        samples = np.interp(times, time, values[field])
        mean = trapezoid(samples, times) / (end - start)
        selected = [
            record
            for record in cycles[field]
            if record["left_trough_time"] >= start - 1e-8
            and record["right_trough_time"] <= end + 1e-8
        ]
        record = {
            "mean": float(mean),
            "whole_window_peak_to_peak": float(np.ptp(samples)),
            "rms_fluctuation": float(
                np.sqrt(trapezoid((samples - mean) ** 2, times) / (end - start))
            ),
            "complete_cycle_count": len(selected),
            "cycles": selected,
        }
        for quantity in (
            "peak_to_peak_raw",
            "peak_to_peak_drift_corrected",
            "period",
            "pressure_contribution_at_raw_extrema",
            "viscous_contribution_at_raw_extrema",
            "pressure_contribution_drift_corrected",
            "viscous_contribution_drift_corrected",
        ):
            measured = [cycle[quantity] for cycle in selected]
            record[f"median_{quantity}"] = float(np.median(measured)) if measured else None
            record[f"mean_{quantity}"] = float(np.mean(measured)) if measured else None
        axis = "x" if field == "drag_coefficient" else "y"
        for component in ("pressure", "viscous"):
            samples = np.interp(
                times, time, coefficient_factor * values[f"{component}_force_{axis}"]
            )
            record[f"mean_{component}_coefficient"] = float(
                trapezoid(samples, times) / (end - start)
            )
        result[field] = record
    return result


def plot_comparison(histories, cycles, output, start, end):
    figure, axes = plt.subplots(2, 2, figsize=(11, 6.5), constrained_layout=True)
    for row, field in enumerate(FIELDS):
        for label, values in histories.items():
            axes[row, 0].plot(values["time"], values[field], label=label, linewidth=1.3)
            accepted = [cycle for cycle in cycles[label][field] if cycle["left_trough_time"] >= 10]
            axes[row, 1].plot(
                [cycle["peak_time"] for cycle in accepted],
                [cycle["peak_to_peak_drift_corrected"] for cycle in accepted],
                ".-",
                label=label,
            )
        symbol = "$C_D$" if row == 0 else "$C_L$"
        axes[row, 0].set_ylabel(symbol)
        axes[row, 0].set_xlim(start, end)
        visible = np.concatenate(
            [
                values[field][(values["time"] >= start - 1e-8) & (values["time"] <= end + 1e-8)]
                for values in histories.values()
            ]
        )
        padding = max(float(np.ptp(visible)) * 0.08, 1e-4)
        axes[row, 0].set_ylim(float(np.min(visible)) - padding, float(np.max(visible)) + padding)
        axes[row, 1].set_ylabel(f"{symbol} complete-cycle peak-to-peak")
        axes[row, 1].set_xlim(10, end)
        for axis in axes[row]:
            axis.grid(alpha=0.25)
            axis.set_xlabel("Time (s)")
    axes[0, 0].legend()
    axes[0, 1].legend()
    figure.suptitle(
        f"Matching planar runs: forces {start:g}–{end:g} s and cycle amplitude evolution\n"
        "Cycle amplitudes remove the linear drift between their bracketing troughs"
    )
    figure.savefig(output / "force_cycle_comparison.png", dpi=170)
    figure.savefig(output / "force_cycle_comparison.pdf")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=CASE)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path(__file__).resolve().parent / "force_boundary_study" / stamp
    output.mkdir(parents=True, exist_ok=False)
    snapshots = output / "snapshots"
    snapshots.mkdir()
    histories, metadata, source_information, all_cycles, all_windows = {}, {}, {}, {}, {}
    windows = [(10, 20), (20, 30), (30, args.end), (args.end - 10, args.end)]
    for label, directory in (("coupled", args.case), ("reference", args.case / "reference_flow")):
        histories[label], source_information[label] = freeze_csv(
            directory / "samples/forces_history.csv", snapshots / f"{label}_forces_history.csv"
        )
        content = (directory / "solution/fvm_metadata.json").read_bytes()
        (snapshots / f"{label}_fvm_metadata.json").write_bytes(content)
        metadata[label] = json.loads(content)
        settings = metadata[label]["configuration"]
        sampler = next(value for value in settings["samplers"] if value["type"] == "ForceSampler")
        factor = 2.0 / (
            settings["transport"]["density"]
            * sampler["reference_velocity"] ** 2
            * sampler["reference_area"]
        )
        # Check the reported coefficients against the dimensional force columns.
        for field, axis in zip(FIELDS, ("x", "y"), strict=True):
            error = np.max(
                abs(histories[label][field] - factor * histories[label][f"total_force_{axis}"])
            )
            if error > 1e-10:
                raise ValueError(f"Inconsistent force normalization in {label}: {error}")
        all_cycles[label] = {
            field: complete_cycles(histories[label], field, factor) for field in FIELDS
        }
        all_windows[label] = {
            f"{start:g}-{end:g}": window_statistics(
                histories[label], all_cycles[label], start, end, factor
            )
            for start, end in windows
        }
    comparisons = {}
    for start, end in windows:
        window = f"{start:g}-{end:g}"
        comparisons[window] = {}
        for field in FIELDS:
            metrics = {}
            for quantity in (
                "mean",
                "whole_window_peak_to_peak",
                "median_peak_to_peak_raw",
                "median_peak_to_peak_drift_corrected",
            ):
                measured, reference = (
                    all_windows[label][window][field][quantity]
                    for label in ("coupled", "reference")
                )
                metrics[quantity] = {
                    "coupled": measured,
                    "reference": reference,
                    "ratio": measured / reference if measured is not None and reference else None,
                    "relative_difference": measured / reference - 1
                    if measured is not None and reference
                    else None,
                }
            comparisons[window][field] = metrics
    checkpoint_bytes = (args.case / "solution/backups/checkpoint_info.json").read_bytes()
    (snapshots / "coupled_checkpoint_info.json").write_bytes(checkpoint_bytes)
    checkpoint = json.loads(checkpoint_bytes)
    report = {
        "schema": "openonda-cylinder-force-cycle-comparison/1",
        "generated_utc": datetime.now(UTC).isoformat(),
        "sources": source_information,
        "coupled_committed_checkpoint": {
            key: checkpoint[key]
            for key in ("created_utc", "time", "coupling_step", "fvm_step", "config_sha256")
        },
        "reference_final_time": metadata["reference"]["state"]["time"],
        "method": {
            "drag_period_bounds": [1.5, 5.0],
            "lift_period_bounds": [4.0, 8.0],
            "extremum_prominence": 1e-4,
            "complete_cycle": "A peak bracketed by troughs, both inside the window; tail and startup fragments excluded.",
            "raw_amplitude": "Maximum minus minimum within that complete cycle; includes slow mean drift.",
            "drift_corrected_amplitude": "Peak minus the linearly interpolated baseline through both bracketing troughs.",
            "force_components": "Pressure and viscous coefficients evaluated at the same total-force extrema; their signed contributions sum to total amplitude.",
        },
        "comparability": {
            label: {
                "mesh_cells": value["mesh"]["n_cells"],
                "mesh_geometry": value["mesh"]["mesh_generation"],
                "transport": value["configuration"]["transport"],
                "schemes": value["configuration"]["schemes"],
                "pimple": value["configuration"]["pimple"],
                "execution": value["configuration"]["execution"],
                "boundaries": value["configuration"]["boundaries"],
                "velocity_boundaries": value["configuration"]["velocity_boundaries"],
                "force_sampler": next(
                    item
                    for item in value["configuration"]["samplers"]
                    if item["type"] == "ForceSampler"
                ),
            }
            for label, value in metadata.items()
        },
        "windows": all_windows,
        "comparisons": comparisons,
        "all_complete_cycles": all_cycles,
        "limitations": [
            "The coupled history has not reached its 100 s horizon; no mature saturated-amplitude recovery is claimed.",
            "Both are unit-span planar cases with equivalent reference scales and effective 0.04 m cylinder spacing. Domains and detailed meshes differ.",
            "Operator backends differ (Numba coupled, NumPy reference); no causal inference is made from that difference.",
            "The 10–20 s drag range is dominated by mean-drag growth. Complete-cycle and drift-corrected amplitudes should be used.",
            "Lift cycles take about 5.4 s; some 10 s windows contain zero complete trough-bracketed cycles despite containing extrema.",
            "Pressure dominance identifies the measured force contribution, not the root cause of pressure-field differences.",
        ],
    }
    (output / "force_cycle_statistics.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_comparison(histories, all_cycles, output, args.end - 10, args.end)
    print(
        json.dumps(
            {
                "output": str(output),
                "latest_common_window": comparisons[f"{args.end - 10:g}-{args.end:g}"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
