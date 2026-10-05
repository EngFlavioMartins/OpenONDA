"""Compare cylinder forces without confusing startup drift with drag oscillations.

Example::

    python tests/support/cylinder/compare_drag_cycles.py \
        --reference reference_flow/samples/forces_history.csv \
        --case coupled=samples/forces_history.csv --start 40 --end 54 \
        --output drag_recovery/baseline

The requested window is explicit. Partial histories are reported as partial,
and this script never infers that a run has reached a saturated limit cycle.
"""

from __future__ import annotations

import argparse
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


def read_forces(path: Path) -> tuple[np.ndarray, str]:
    """Read one consistent CSV snapshot, including when a solver is appending."""
    csv_bytes = path.read_bytes()
    if not csv_bytes.endswith(b"\n"):
        csv_bytes = csv_bytes[: csv_bytes.rfind(b"\n") + 1]
    values = np.atleast_1d(np.genfromtxt(io.BytesIO(csv_bytes), delimiter=",", names=True))
    required = ("time", "drag_coefficient", "lift_coefficient")
    if not all(name in (values.dtype.names or ()) for name in required):
        raise ValueError(f"Missing force columns in {path}")
    if len(values) < 3 or any(not np.all(np.isfinite(values[key])) for key in required):
        raise ValueError(f"Need at least three finite force samples in {path}")
    if np.any(np.diff(values["time"]) <= 0):
        raise ValueError(f"Force times must be strictly increasing in {path}")
    return values, hashlib.sha256(csv_bytes).hexdigest()


def drag_cycles(time, drag, *, min_period=1.5, max_period=5.0, prominence=1e-4) -> list[dict]:
    """Measure each peak above the line through its bracketing troughs.

    The linear baseline removes slow mean-drag drift. A complete measurement
    needs both troughs, so unbracketed startup and tail peaks are omitted.
    The minimum period is a peak-selection control, not a measured frequency.
    """
    separation = max(1, int(round(min_period / np.median(np.diff(time)))))
    peaks, _ = find_peaks(drag, distance=separation, prominence=prominence)
    troughs, _ = find_peaks(-drag, distance=separation, prominence=prominence)
    cycles = []
    for peak in peaks:
        left, right = troughs[troughs < peak], troughs[troughs > peak]
        if not len(left) or not len(right):
            continue
        lower, upper = left[-1], right[0]
        baseline = np.interp(time[peak], time[[lower, upper]], drag[[lower, upper]])
        if time[upper] - time[lower] > max_period or drag[peak] <= baseline:
            continue
        cycles.append(
            {
                "peak_time": float(time[peak]),
                "left_trough_time": float(time[lower]),
                "right_trough_time": float(time[upper]),
                "peak_drag_coefficient": float(drag[peak]),
                "interpolated_trough_baseline": float(baseline),
                "drag_peak_to_peak_detrended": float(drag[peak] - baseline),
                "trough_to_trough_period": float(time[upper] - time[lower]),
                "baseline_drag_drift_per_second": float(
                    (drag[upper] - drag[lower]) / (time[upper] - time[lower])
                ),
            }
        )
    return cycles


def summarize(values, start, end, *, min_period=1.5, max_period=5.0, prominence=1e-4) -> dict:
    time, drag, lift = (values[name] for name in ("time", "drag_coefficient", "lift_coefficient"))
    cycles = drag_cycles(
        time, drag, min_period=min_period, max_period=max_period, prominence=prominence
    )
    lower, upper = max(start, time[0]), min(end, time[-1])
    covered = bool(time[0] <= start + 1e-9 and time[-1] >= end - 1e-9)
    result = {
        "available_interval": [float(time[0]), float(time[-1])],
        "requested_interval": [start, end],
        "requested_interval_covered": covered,
        "sample_count": len(time),
        "all_complete_drag_cycles": cycles,
        "window": None,
    }
    if upper <= lower:
        result["limitation"] = "No data overlap with the requested window."
        return result
    sample_time = np.r_[lower, time[(time > lower) & (time < upper)], upper]
    sample_drag, sample_lift = (np.interp(sample_time, time, value) for value in (drag, lift))
    mean_drag = float(trapezoid(sample_drag, sample_time) / (upper - lower))
    mean_lift = float(trapezoid(sample_lift, sample_time) / (upper - lower))
    selected = [
        cycle
        for cycle in cycles
        if cycle["left_trough_time"] >= lower - 1e-9 and cycle["right_trough_time"] <= upper + 1e-9
    ]
    amplitudes = [cycle["drag_peak_to_peak_detrended"] for cycle in selected]
    periods = [cycle["trough_to_trough_period"] for cycle in selected]
    result["window"] = {
        "measured_interval": [float(lower), float(upper)],
        "mean_drag_coefficient": mean_drag,
        "mean_lift_coefficient": mean_lift,
        "raw_drag_peak_to_peak": float(np.ptp(sample_drag)),
        "lift_peak_to_peak": float(np.ptp(sample_lift)),
        "drag_fluctuation_rms": float(
            np.sqrt(trapezoid((sample_drag - mean_drag) ** 2, sample_time) / (upper - lower))
        ),
        "complete_drag_cycle_count": len(selected),
        "mean_drag_peak_to_peak_detrended": float(np.mean(amplitudes)) if amplitudes else None,
        "mean_drag_cycle_period_seconds": float(np.mean(periods)) if periods else None,
        # This benchmark has D/U=1 s and two drag cycles per shedding cycle.
        "shedding_strouhal_from_drag_period": float(0.5 / np.mean(periods)) if periods else None,
        "drag_peak_to_peak_detrended_range": [min(amplitudes), max(amplitudes)]
        if amplitudes
        else None,
        "complete_drag_cycles": selected,
    }
    result["limitation"] = (
        "Window coverage is complete; saturation is not established by coverage or these metrics."
        if covered
        else "Only the available portion of the requested window is measured; tail is incomplete."
    )
    if len(selected) < 3:
        result["limitation"] += " Fewer than three complete drag cycles are available."
    return result


def create_plots(histories, metrics, output, start, end):
    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
    for label, values in histories.items():
        time = values["time"]
        axes[0].plot(time, values["drag_coefficient"], label=label, linewidth=1.2)
        axes[1].plot(time, values["lift_coefficient"], label=label, linewidth=1.2)
    for axis, ylabel in zip(
        axes, ("Drag coefficient $C_D$", "Lift coefficient $C_L$"), strict=True
    ):
        axis.set_ylabel(ylabel)
        axis.set_xlim(start, end)
        axis.grid(alpha=0.2)
        visible = []
        key = "drag_coefficient" if axis is axes[0] else "lift_coefficient"
        for values in histories.values():
            mask = (values["time"] >= start - 1e-9) & (values["time"] <= end + 1e-9)
            visible.extend(values[key][mask])
        if visible:
            minimum, maximum = min(visible), max(visible)
            padding = max((maximum - minimum) * 0.06, 1e-4)
            axis.set_ylim(minimum - padding, maximum + padding)
    axes[0].legend()
    axes[1].set_xlabel("Time $tU_\\infty/D$")
    figure.suptitle(
        f"Cylinder forces: requested window {start:g}–{end:g}\nAvailable data only; saturation is not assumed"
    )
    figure.savefig(output / "force_overlay.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
    for label, values in histories.items():
        cycles = metrics[label]["all_complete_drag_cycles"]
        axes[0].plot(
            [cycle["peak_time"] for cycle in cycles],
            [cycle["drag_peak_to_peak_detrended"] for cycle in cycles],
            ".-",
            markersize=4,
            label=label,
        )
        absolute_lift = abs(values["lift_coefficient"])
        separation = max(1, int(round(1.5 / np.median(np.diff(values["time"])))))
        peaks, _ = find_peaks(absolute_lift, distance=separation, prominence=1e-4)
        axes[1].plot(values["time"][peaks], absolute_lift[peaks], ".-", markersize=4, label=label)
    axes[0].set_ylabel("Drift-corrected $C_D$ peak-to-peak")
    axes[1].set_ylabel("Lift extrema $|C_L|$")
    axes[1].set_xlabel("Time $tU_\\infty/D$")
    axes[0].legend()
    for axis in axes:
        axis.axvspan(start, end, color="grey", alpha=0.1, zorder=-10)
        axis.grid(alpha=0.2)
        axis.set_xlim(left=2)
    figure.suptitle(
        "Amplitude evolution; shaded region is the requested statistics window\nDrag peaks require two bracketing troughs; incomplete tail cycles are omitted"
    )
    figure.savefig(output / "amplitude_evolution.png", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[], metavar="LABEL=CSV")
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--min-cycle-period", type=float, default=1.5)
    parser.add_argument("--max-cycle-period", type=float, default=5.0)
    parser.add_argument("--peak-prominence", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not (np.isfinite(args.start) and np.isfinite(args.end) and args.end > args.start):
        parser.error("--end must be finite and greater than finite --start")
    if not (
        np.isfinite(args.min_cycle_period)
        and np.isfinite(args.max_cycle_period)
        and np.isfinite(args.peak_prominence)
        and 0 < args.min_cycle_period < args.max_cycle_period
        and args.peak_prominence >= 0
    ):
        parser.error(
            "Peak selection needs 0 < min period < max period and finite nonnegative prominence"
        )
    paths = {"reference": args.reference}
    for specification in args.case:
        label, separator, path = specification.partition("=")
        if not separator or not label or not path or label in paths:
            parser.error("Each --case needs a unique nonempty LABEL=CSV")
        paths[label] = Path(path)
    histories, metrics = {}, {}
    for label, path in paths.items():
        values, digest = read_forces(path)
        histories[label] = values
        metrics[label] = summarize(
            values,
            args.start,
            args.end,
            min_period=args.min_cycle_period,
            max_period=args.max_cycle_period,
            prominence=args.peak_prominence,
        )
        metrics[label].update(source=str(path.resolve()), snapshot_sha256=digest)
    result = {
        "scope": "Saved force histories only. No saturation or grid-convergence claim.",
        "drag_cycle_definition": "Peak minus linear interpolation between its two adjacent troughs.",
        "startup_exclusion": "Nonpositive corrected amplitudes and trough separations above max-cycle-period are excluded.",
        "window_definition": "Explicit requested interval; boundary values linearly interpolated; incomplete coverage labelled.",
        "peak_selection": {
            "minimum_same_kind_peak_separation_seconds": args.min_cycle_period,
            "maximum_trough_to_trough_period_seconds": args.max_cycle_period,
            "minimum_prominence": args.peak_prominence,
        },
        "series": metrics,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "force_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    create_plots(histories, metrics, args.output, args.start, args.end)
    for label, metric in metrics.items():
        window = metric["window"]
        if window:
            print(
                f"{label}: {window['measured_interval']}; mean Cd={window['mean_drag_coefficient']:.6f}; "
                f"cycle Cd p2p={window['mean_drag_peak_to_peak_detrended']}; "
                f"Cl p2p={window['lift_peak_to_peak']:.6f}; "
                f"complete cycles={window['complete_drag_cycle_count']}"
            )
        print(f"  {metric['limitation']}")
    print(f"Evidence saved under {args.output.resolve()}")


if __name__ == "__main__":
    main()
