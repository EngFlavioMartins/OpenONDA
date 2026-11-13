#!/usr/bin/env python3
"""Coupled wing against the FVM-only reference: forces, errors, interface measures and cost.

Reads the coupled run in solution/ and samples/ and the FVM-only reference in
reference_flow/{samples,solution}, and writes to figures/ (PDF and PNG):

    forces               drag and lift against the reference
    errors_by_window     mean drag and lift errors in 2 s windows with the 2 % acceptance band
    interface_measures   rms velocity mismatch per box side, budget error, near-wall ratio
    coupling_cost        wall time per flow time by solver part against the reference, particles
    comparison_summary.csv  statistics, errors, interface measures and cost

The forces are compared without a blockage correction (see the README). A figure whose data are
missing is skipped with a message.

Run from the case directory: python -m assets.plot_comparison
"""

import csv
import json
from pathlib import Path

from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np

from openonda.plotting import COLORS, LINE_WIDTH, REFERENCE_STYLE, fit_thesis_y_label_margins
from openonda.plotting import save_validation_figure, validation_subplots

CASE_DIR = Path(__file__).absolute().parents[1]
FIGURE_DIR = CASE_DIR / "figures"
REFERENCE_NAME = "reference_flow"
REFERENCE_SAMPLES = CASE_DIR / "reference_flow/samples"
REFERENCE_SOLUTION = CASE_DIR / "reference_flow/solution"
REFERENCE_RANKS = 4  # MPI processes of the reference run
COUPLED_RANKS = 8  # FVM MPI processes of the coupled run, with the particles on one GPU

CHORD = 1.0  # m
FREESTREAM_SPEED = 1.0  # m/s
TIME_SCALE = FREESTREAM_SPEED / CHORD  # 1/s, convective-time scale
STATISTICS_WINDOW = (12.0, 24.0)  # s
MINIMUM_STATISTICS_LENGTH = 4.0  # s of the window that both runs must cover
FORCE_WINDOW = 2.0  # s, windows of the mean-force errors
START_UP = 0.0  # s, the error windows start here
TOLERANCE = 0.02  # acceptance: mean lift and drag within 2 % of the reference in every window
RATIO_SCALE_START = 1.0  # s
RATIO_BOUNDS = (0.97, 1.03)  # acceptance: near-wall circulation ratio of the wall transfer
SIDES = ("x-", "x+", "y-", "y+", "z-", "z+")
SIDE_NAMES = {"x-": "upstream", "x+": "downstream", "y-": "$y^-$", "y+": "$y^+$", "z-": "$z^-$",
              "z+": "$z^+$"}
SIDE_STYLES = {
    "x-": dict(color=COLORS["teal"], linestyle="-"),
    "x+": dict(color=COLORS["red"], linestyle="-"),
    "y-": dict(color=COLORS["green"], linestyle="-"),
    "y+": dict(color=COLORS["green"], linestyle="--"),
    "z-": dict(color=COLORS["blue"], linestyle="-"),
    "z+": dict(color=COLORS["blue"], linestyle="--"),
}
COEFFICIENTS = {"drag_coefficient": "drag coefficient", "lift_coefficient": "lift coefficient"}
STATISTIC_NAMES = {"CD": "mean drag", "CL": "mean lift", "CD_rms": "drag rms", "CL_rms": "lift rms"}
ERROR_STYLES = {"CD": (r"$\overline{C_D}$", COLORS["teal"], "s"),
                "CL": (r"$\overline{C_L}$", COLORS["hybrid"], "o")}
COUPLED_STYLE = dict(color=COLORS["hybrid"], linewidth=LINE_WIDTH)
PHASES = {
    "VPM": (("vpm",), COLORS["vpm"]),
    "FVM": (("fvm",), COLORS["fvm"]),
    "coupling": (("vpm_boundary_condition", "transfer"), COLORS["hybrid"]),
    "output": (("state_checks_and_samplers", "backup", "reporting"), COLORS["gray"]),
}


# ------------------------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------------------------


def read_forces(path):
    """Force-coefficient history, or None if the file is missing or empty."""
    if not path.exists():
        return None
    data = np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8")
    if data.size < 2:
        return None
    return {key: np.atleast_1d(data[key]) for key in ("time", *COEFFICIENTS)}


def jsonl(path):
    records = []
    if path.exists():
        for line in path.read_text(errors="ignore").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def statistics(forces, start, end):
    """Mean and rms fluctuation of drag and lift over [start, end]."""
    inside = (forces["time"] >= start) & (forces["time"] <= end)
    time = forces["time"][inside]
    if time.size < 4:
        raise ValueError("fewer than four samples in the window")
    drag, lift = (forces[key][inside] for key in COEFFICIENTS)
    duration = time[-1] - time[0]
    return {"CD": np.trapezoid(drag, time) / duration, "CL": np.trapezoid(lift, time) / duration,
            "CD_rms": drag.std(), "CL_rms": lift.std()}


def errors(coupled, reference, start, end):
    """Relative errors of the coupled statistics against the reference over [start, end]."""
    expected = statistics(reference, start, end)
    measured = statistics(coupled, start, end)
    return {key: measured[key] / expected[key] - 1.0 for key in expected}, measured, expected


def window_errors(coupled, reference, length, end, key):
    """Error of one statistic in consecutive windows of the given length from START_UP to end."""
    centres, values = [], []
    for start in np.arange(START_UP, end - length + 1e-9, length):
        try:
            values.append(errors(coupled, reference, start, start + length)[0][key])
        except ValueError:
            continue
        centres.append(start + 0.5 * length)
    return np.array(centres), np.array(values)


def reference_cost():
    """Wall seconds per flow second of the reference run, from its performance log."""
    records = [r for r in jsonl(REFERENCE_SOLUTION / "performance.jsonl") if "step_seconds" in r]
    if not records:
        return np.nan
    return sum(r["step_seconds"]["max"] for r in records) / records[-1]["time"]


def interface_series(records):
    """Time series of the interface measures, one value per coupling interval."""
    time = np.array([r["time"] for r in records])
    boundary = [r.get("fvm_boundary_data", {}) for r in records]
    transfer = [r.get("transfer", {}) for r in records]

    def column(rows, key):
        return np.array([row.get(key, np.nan) for row in rows], float)

    return {
        "time": time,
        **{side: column(boundary, f"rms_velocity_mismatch_{side}") / FREESTREAM_SPEED for side in SIDES},
        "maximum": column(boundary, "maximum_velocity_mismatch") / FREESTREAM_SPEED,
        "budget": column(transfer, "fvm_budget_relative_error"),
        "ratio": column(transfer, "fvm_budget_wall_circulation_ratio"),
        "particles": column(transfer, "n_particles_after"),
        "iterations": np.array([r.get("coupling_iteration", {}).get("iterations", np.nan)
                                for r in records], float),
    }


# ------------------------------------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------------------------------------


def top_legend(figure, axis, ncol, handles=None):
    """One legend above the top panel, clear of the data."""
    handles, labels = (handles, None) if handles else axis.get_legend_handles_labels()
    arguments = dict(loc="upper center", ncol=ncol, frameon=True, framealpha=1.0, edgecolor="0.8",
                     bbox_to_anchor=(0.5, 0.995), handlelength=1.5, columnspacing=0.8,
                     handletextpad=0.4)
    if labels is None:
        figure.legend(handles=handles, **arguments)
    else:
        figure.legend(handles, labels, **arguments)


def save(figure, name, fit_margins=True):
    if fit_margins:
        fit_thesis_y_label_margins(figure, figure.axes)
    save_validation_figure(figure, FIGURE_DIR / name, figure_format="png")


def shade_window(axis, end):
    """Light band over the part of the statistics window that the data cover."""
    if end > STATISTICS_WINDOW[0]:
        axis.axvspan(STATISTICS_WINDOW[0] * TIME_SCALE, min(end, STATISTICS_WINDOW[1]) * TIME_SCALE,
                     color=COLORS["background_light"], alpha=0.3, linewidth=0)


def figure_forces(coupled, reference, end):
    figure, axes = validation_subplots(2, sharex=True, height_cm=9.5, outer=0.13, top_padding_cm=1.3)
    for axis, (key, label) in zip(axes, COEFFICIENTS.items()):
        shown = reference["time"] <= end
        axis.plot(reference["time"][shown] * TIME_SCALE, reference[key][shown],
                  label="FVM", **REFERENCE_STYLE)
        shown = coupled["time"] <= end
        axis.plot(coupled["time"][shown] * TIME_SCALE, coupled[key][shown], label="FVM-VPM",
                  **COUPLED_STYLE)
        shade_window(axis, end)
        axis.set_ylabel(label)
    # Scale to the flow after the impulsive start.
    for axis, key in zip(axes, COEFFICIENTS):
        values = np.concatenate([forces[key][forces["time"] >= min(1.0, 0.5 * end)]
                                 for forces in (coupled, reference)])
        span = max(values.max() - values.min(), 0.02 * abs(values).max())
        axis.set_ylim(values.min() - 0.15 * span, values.max() + 0.15 * span)
    axes[-1].set_xlim(0, end * TIME_SCALE)
    axes[-1].set_xlabel(r"$tU_\infty/c$")
    top_legend(figure, axes[0], 2)
    save(figure, "forces")


def figure_errors(coupled, reference, end):
    series = {key: window_errors(coupled, reference, FORCE_WINDOW, end, key) for key in ERROR_STYLES}
    if series["CD"][0].size == 0:
        return f"needs one {FORCE_WINDOW:g} s window in both runs"
    figure, axes = validation_subplots(1, height_cm=7.5, outer=0.13, top_padding_cm=1.3)
    axis = axes[0]
    axis.axhspan(-100 * TOLERANCE, 100 * TOLERANCE, color=COLORS["background_light"], alpha=0.4,
                 linewidth=0, label=r"$\pm2\,\%$ acceptance")
    axis.axhline(0, color=COLORS["black"], linewidth=LINE_WIDTH / 2)
    for key, (label, color, marker) in ERROR_STYLES.items():
        centres, values = series[key]
        axis.plot(centres * TIME_SCALE, 100 * values, marker=marker, markersize=3.5, color=color,
                  linewidth=LINE_WIDTH, label=label)
    values = 100 * np.concatenate([series[key][1] for key in ERROR_STYLES])
    axis.set_ylim(min(-3.0, values.min() - 0.5), max(3.0, values.max() + 0.5))
    axis.set_ylabel(r"error ($\%$)")
    shade_window(axis, end)
    top_legend(figure, axis, 3)
    last = np.ceil(end / FORCE_WINDOW) * FORCE_WINDOW
    axis.set_xlim(START_UP * TIME_SCALE, last * TIME_SCALE)
    axis.set_xlabel(r"window centre $tU_\infty/c$")
    save(figure, "errors_by_window")
    return None


def log_axis(axis):
    axis.set_yscale("log")
    axis.yaxis.set_major_locator(LogLocator(numticks=6))
    axis.yaxis.set_minor_formatter(NullFormatter())


def figure_interface(series):
    time = series["time"] * TIME_SCALE
    figure, axes = validation_subplots(3, sharex=True, height_cm=14.0, outer=0.13, top_padding_cm=1.9)
    for side in SIDES:
        axes[0].plot(time, series[side], linewidth=LINE_WIDTH, label=SIDE_NAMES[side], **SIDE_STYLES[side])
    axes[0].plot(time, series["maximum"], color=COLORS["gray"], linewidth=LINE_WIDTH / 2,
                 label="maximum")
    log_axis(axes[0])
    finite = np.concatenate([series[side][np.isfinite(series[side])] for side in SIDES])
    axes[0].set_ylim(0.5 * finite[finite > 0].min(), 3 * np.nanmax(series["maximum"]))
    axes[0].set_ylabel(r"$|\Delta u|/U_\infty$")
    top_legend(figure, axes[0], 4)
    axes[1].plot(time, series["budget"], color=COLORS["teal"], linewidth=LINE_WIDTH)
    log_axis(axes[1])
    axes[1].set_ylabel("budget error")
    axes[2].axhspan(*RATIO_BOUNDS, color=COLORS["background_light"], alpha=0.4, linewidth=0)
    axes[2].axhline(1, color=COLORS["black"], linewidth=LINE_WIDTH / 2)
    axes[2].plot(time, series["ratio"], color=COLORS["hybrid"], linewidth=LINE_WIDTH)
    # The first coupling intervals divide two small totals; the scale follows the rest.
    ratio = series["ratio"][np.isfinite(series["ratio"]) & (series["time"] >= RATIO_SCALE_START)]
    if ratio.size:
        axes[2].set_ylim(min(0.95, ratio.min() - 0.01), max(1.05, ratio.max() + 0.01))
    axes[2].set_ylabel("near-wall ratio")
    axes[-1].set_xlim(0, time[-1])
    axes[-1].set_xlabel(r"$tU_\infty/c$")
    save(figure, "interface_measures")


def figure_cost(records, series, cost_reference):
    time = series["time"]
    interval = np.diff(time, prepend=0.0)
    keep = interval > 0
    wall = [np.array([sum(r["timing_seconds"].get(key, 0.0) for key in keys) for r in records])
            for keys, _ in PHASES.values()]
    cost = [phase[keep] / interval[keep] for phase in wall]
    total = sum(np.sum(phase) for phase in wall) / time[-1]
    figure, axes = validation_subplots(2, sharex=True, height_cm=10.0, outer=0.13, top_padding_cm=1.9)
    cost = [phase / 1e3 for phase in cost]
    axes[0].stackplot(time[keep] * TIME_SCALE, *cost, labels=list(PHASES),
                      colors=[color for _, color in PHASES.values()], linewidth=0)
    axes[0].axhline(total / 1e3, color=COLORS["text"], linestyle="-.", linewidth=LINE_WIDTH,
                    label="FVM-VPM mean")
    if not np.isnan(cost_reference):
        axes[0].axhline(cost_reference / 1e3, label="FVM only", **REFERENCE_STYLE)
    axes[0].set_ylim(0, 1.15 * np.percentile(np.sum(cost, axis=0), 99.5))
    axes[0].set_ylabel(r"cost ($10^3$ s/s)")
    top_legend(figure, axes[0], 3)
    axes[1].plot(time * TIME_SCALE, series["particles"] / 1e6, color=COLORS["vpm"], linewidth=LINE_WIDTH)
    axes[1].set_ylabel(r"particles ($10^6$)")
    axes[1].set_xlim(0, time[-1] * TIME_SCALE)
    axes[1].set_xlabel(r"$tU_\infty/c$")
    save(figure, "coupling_cost")
    return total


# ------------------------------------------------------------------------------------------------


def window_mean(series, key, start, end):
    inside = (series["time"] >= start) & (series["time"] <= end)
    values = series[key][inside]
    values = values[np.isfinite(values)]
    return values if values.size else np.array([np.nan])


def main():
    FIGURE_DIR.mkdir(exist_ok=True)
    coupled = read_forces(CASE_DIR / "samples/forces_history.csv")
    reference = read_forces(REFERENCE_SAMPLES / "forces_history.csv")
    records = jsonl(CASE_DIR / "solution/coupler_diagnostics.jsonl")
    cost_reference = reference_cost()
    summary = {"coupled_end_s": np.nan, "reference_end_s": np.nan}

    if coupled is None or reference is None:
        print("forces, errors_by_window: wait for "
              + " and ".join(name for name, data in (("the coupled forces", coupled),
                                                     (f"the reference {REFERENCE_NAME}", reference))
                             if data is None))
    else:
        end = min(coupled["time"][-1], reference["time"][-1], STATISTICS_WINDOW[1])
        summary.update(coupled_end_s=coupled["time"][-1], reference_end_s=reference["time"][-1])
        print(f"Coupled run to t = {coupled['time'][-1]:.2f} s, reference {REFERENCE_NAME} to "
              f"t = {reference['time'][-1]:.2f} s; compared to t = {end:.2f} s")
        figure_forces(coupled, reference, end)
        reason = figure_errors(coupled, reference, end)
        if reason:
            print(f"errors_by_window: waits, {reason}")
        statistics_end = min(end, STATISTICS_WINDOW[1])
        if statistics_end - STATISTICS_WINDOW[0] >= MINIMUM_STATISTICS_LENGTH:
            error, measured, expected = errors(coupled, reference, STATISTICS_WINDOW[0], statistics_end)
            print(f"Statistics over {STATISTICS_WINDOW[0]:g}-{statistics_end:.1f} s:")
            for key in error:
                print(f"  {STATISTIC_NAMES[key]:16s} FVM {expected[key]:.4f}  FVM-VPM {measured[key]:.4f}"
                      f"  ({100 * error[key]:+.2f} %)")
                summary.update({f"{key}_reference": expected[key], f"{key}_coupled": measured[key],
                                f"{key}_error_percent": 100 * error[key]})
            summary["statistics_end_s"] = statistics_end
            for key in ERROR_STYLES:
                verdict = "within" if abs(error[key]) <= TOLERANCE else "outside"
                print(f"  {STATISTIC_NAMES[key]} {verdict} the {100 * TOLERANCE:g} % acceptance")
        else:
            print(f"statistics: wait until both runs pass t = "
                  f"{STATISTICS_WINDOW[0] + MINIMUM_STATISTICS_LENGTH:g} s")

    if not records:
        print("interface_measures, coupling_cost: wait for solution/coupler_diagnostics.jsonl")
    else:
        series = interface_series(records)
        figure_interface(series)
        total = figure_cost(records, series, cost_reference)
        start, end = STATISTICS_WINDOW[0], series["time"][-1]
        if end <= start:
            start = RATIO_SCALE_START
        for side in SIDES:
            summary[f"mismatch_rms_{side}"] = float(np.sqrt(np.mean(window_mean(series, side, start, end) ** 2)))
        summary["mismatch_maximum"] = float(np.nanmax(series["maximum"]))
        summary["budget_error_max"] = float(np.nanmax(series["budget"]))
        ratio = window_mean(series, "ratio", start, end)
        summary.update(wall_ratio_median=float(np.median(ratio)), wall_ratio_min=float(ratio.min()),
                       wall_ratio_max=float(ratio.max()),
                       iterations_mean=float(np.nanmean(series["iterations"])))
        summary.update(cost_coupled=total, cost_reference=cost_reference)
        print(f"Interface measures over {start:g}-{end:.1f} s: rms mismatch "
              + ", ".join(f"{side} {summary[f'mismatch_rms_{side}']:.2e}" for side in SIDES)
              + f" U; maximum over the run {summary['mismatch_maximum']:.3f} U; "
              f"near-wall ratio {ratio.min():.3f}-{ratio.max():.3f} (acceptance "
              f"{RATIO_BOUNDS[0]:g}-{RATIO_BOUNDS[1]:g})")
        print(f"Wall time per flow time: FVM-VPM {total:.0f} s/s ({COUPLED_RANKS} FVM ranks and a GPU), "
              f"FVM only {cost_reference:.0f} s/s ({REFERENCE_RANKS} ranks)")

    with open(FIGURE_DIR / "comparison_summary.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["quantity", "value"])
        for key, value in summary.items():
            writer.writerow([key, f"{value:.6g}"])
    print(f"Saved: {FIGURE_DIR / 'comparison_summary.csv'}")


if __name__ == "__main__":
    main()
