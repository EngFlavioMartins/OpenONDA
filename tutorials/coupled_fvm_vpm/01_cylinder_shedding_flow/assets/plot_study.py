#!/usr/bin/env python3
"""Calibration study of the coupled cylinder at Re = 150: statistics, errors and figures.

Reads every coupled run in solution/sensitivity_analysis/<name> and the FVM-only references in
reference_flow/samples/h<cell>_dt<step>, and writes to figures/ (PDF and PNG):

    forces, velocity_profiles,     1. baseline against the FVM-only reference
    vorticity_profiles
    error_overview                 2. error of C_D, St and C_L rms of every run, with the window
                                      uncertainty as error bars
    error_coupling_interval        3. temporal convergence: error against the coupling interval
    error_cell_size                4. spatial convergence at a fixed interval, particle spacing
    error_domain                   5. error against the box margin and the transfer inset
    error_coupling_iterations      6. error and mean iteration count against the tolerance
    interface_measures             7. boundary mismatch per side, budget error, near-wall ratio
    cost_accuracy                  8. wall time per flow time against error
    study_summary.csv              9. one row per run with every number above

A run belongs to the study when it uses the current wall transfer: its coupler configuration has
the near-wall circulation bounds (0.97, 1.03) of 37d7d7e3. Older runs in the same folder are
listed and skipped. The settings of a run are read from its metadata, not from its name; a run
enters the group of the one setting in which it differs from the baseline. A figure whose runs are
missing, or have not reached the end of the statistics window, is skipped with a message.

The error of a run is Q_coupled / Q_FVM - 1 against the FVM-only run with the same cells and FVM
step, over the whole shedding periods in the study window. The FVM domain is bounded by slip
walls 32 D apart and the coupled one is not, so every FVM value is corrected for blockage
(Allen and Vincenti): forces by (U'/U)^2, St by U'/U.

Usage, from the case directory:
    python -m assets.plot_study [--runs DIR] [--figures DIR] [--window START END]
"""

import argparse
import csv
import functools
import json
from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import matplotlib.pyplot as plt
import numpy as np

from openonda.plotting import CM, COLORS, LINE_WIDTH, MAX_FIGURE_WIDTH_CM, REFERENCE_STYLE
from openonda.plotting import centered_subplots_adjust, fit_thesis_y_label_margins
from openonda.plotting import save_validation_figure, set_thesis_style
from openonda.plotting import validation_subplots

CASE_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = CASE_DIR / "solution/sensitivity_analysis"
REFERENCE_DIR = CASE_DIR / "reference_flow/samples"
REFERENCE_SOLUTION_DIR = CASE_DIR / "reference_flow/solution"
FIGURE_DIR = CASE_DIR / "figures"

DIAMETER = 1.0  # m
FREESTREAM_SPEED = 1.0  # m/s
REFERENCE_HEIGHT = 32.0  # m, distance between the slip walls of the FVM-only domain
CURRENT_TRANSFER_BOUNDS = [0.97, 1.03]  # near-wall ratio bounds of the current wall transfer

# The baseline (allrun.sh, first line): h = 0.04 m, dt_FVM = 0.008 s, N = 5, box margin 2.4 D,
# inset 8.75 particle spacings, particle spacing h, iterations to 1e-5 with linear extrapolation.
BASELINE = {
    "N": 5,
    "cell": 0.04,
    "fvm_step": 0.008,
    "margin": 2.4,
    "inset": 8.75,
    "particle_spacing": 0.04,
    "max_iterations": 6,
    "tolerance": 1e-5,
    "extrapolation": "linear",
}
ITERATION_KEYS = ("max_iterations", "tolerance", "extrapolation")
FIXED_INTERVAL = 0.048  # s, coupling interval of the cell-size series

# Every study run reaches 55 s, so runs are compared over 40-55 s. The baseline continues to
# 100 s; its forces and profiles use the whole periods from 55 s on.
STUDY_WINDOW = (40.0, 55.0)
STEADY_START = 55.0
MINIMUM_STEADY_LENGTH = 12.0  # s, two periods: the baseline figures wait for this much past 55 s

STATISTICS = ("CD", "St", "CL_rms")
ALL_STATISTICS = ("CD", "CL", "St", "CL_rms")
STATISTIC_NAMES = {"CD": "mean drag", "CL": "lift amplitude", "St": "Strouhal number",
                   "CL_rms": "rms lift"}
STATISTIC_SYMBOLS = {"CD": r"$C_D$", "CL": r"$\hat C_L$", "St": r"$St$",
                     "CL_rms": r"$C_{L,\mathrm{rms}}$"}
STATISTIC_STYLES = {
    "CD": dict(color=COLORS["teal"], marker="o"),
    "St": dict(color=COLORS["green"], marker="^"),
    "CL_rms": dict(color=COLORS["red"], marker="s"),
    "CL": dict(color=COLORS["amber"], marker="D"),
}
GROUPS = {
    "baseline": ("baseline", COLORS["black"]),
    "T": ("coupling interval", COLORS["teal"]),
    "M": ("coupling iterations", COLORS["amber"]),
    "D": ("box margin", COLORS["green"]),
    "I": ("transfer inset", COLORS["blue"]),
    "S": ("cell size", COLORS["red"]),
    "P": ("particle spacing", COLORS["gray"]),
    "other": ("other", COLORS["strong_gray"]),
}
SIDES = ("x-", "x+", "y-", "y+")
SIDE_NAMES = {"x-": "upstream", "x+": "downstream", "y-": "lower", "y+": "upper"}
COUPLED_STYLE = dict(color=COLORS["hybrid"], linewidth=LINE_WIDTH)
WIDTH = MAX_FIGURE_WIDTH_CM * CM


# ------------------------------------------------------------------------------------------------
# Statistics (also used by ~/openonda_runs/f0/metrics/f0_metrics.py and the paper evidence)
# ------------------------------------------------------------------------------------------------


def read_columns(path, columns):
    data = np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8")
    return {column: np.atleast_1d(data[column]) for column in columns}


@functools.lru_cache(maxsize=None)
def _read_forces_file(path):
    return read_columns(path, ("time", "drag_coefficient", "lift_coefficient"))


def read_forces(directory):
    """Force-coefficient history of a coupled run or an FVM-only reference."""
    path = next(Path(directory).glob("samples/forces_history.csv"), Path(directory) / "forces_history.csv")
    return _read_forces_file(path.resolve())


def blockage_factor(drag):
    """Freestream speed-up U'/U between slip walls (Allen and Vincenti)."""
    ratio = DIAMETER / REFERENCE_HEIGHT
    return 1 + 0.25 * drag * ratio + 0.82 * ratio**2


def corrected(stats):
    """Unbounded-flow statistics of an FVM-only run: forces scale with U^2, St with U."""
    factor = blockage_factor(stats["CD"])
    out = {"CD": stats["CD"] / factor**2, "CL": stats["CL"] / factor**2, "St": stats["St"] / factor}
    if "CL_rms" in stats:
        out["CL_rms"] = stats["CL_rms"] / factor**2
    return out


def upward_crossings(time, lift):
    """Upward zero crossings of the lift fluctuation: the bounds of the shedding periods."""
    fluctuation = lift - lift.mean()
    up = np.flatnonzero((fluctuation[:-1] < 0) & (fluctuation[1:] >= 0))
    return time[up] - fluctuation[up] * np.diff(time)[up] / np.diff(fluctuation)[up]


def whole_periods(forces, start, end=np.inf):
    """First and last upward lift crossing in [start, end], and the number of periods between."""
    inside = (forces["time"] >= start) & (forces["time"] <= end)
    crossings = upward_crossings(forces["time"][inside], forces["lift_coefficient"][inside])
    return crossings[0], crossings[-1], len(crossings) - 1


def vertex(time, values, index):
    """Extremum at the vertex of the parabola through three samples."""
    if index in (0, len(values) - 1):
        return values[index]
    t, v = time[index - 1 : index + 2] - time[index], values[index - 1 : index + 2]
    a, b, c = np.linalg.solve(np.vander(t, 3), v)
    return c - b * b / (4 * a) if a else v[1]


def shedding_statistics(forces, start, end=np.inf):
    """Mean drag, lift amplitude, Strouhal number and rms lift over the whole periods in [start, end]."""
    first, last, periods = whole_periods(forces, start, end)
    if periods < 1:
        raise ValueError("no whole shedding period in the window")
    inside = (forces["time"] >= first) & (forces["time"] <= last)
    time, drag, lift = (
        forces[key][inside] for key in ("time", "drag_coefficient", "lift_coefficient")
    )
    return {
        "CD": np.trapezoid(drag, time) / (time[-1] - time[0]),
        "CL": 0.5 * (vertex(time, lift, np.argmax(lift)) - vertex(time, lift, np.argmin(lift))),
        "St": DIAMETER * periods / ((last - first) * FREESTREAM_SPEED),
        "CL_rms": float(np.sqrt(np.mean((lift - lift.mean()) ** 2))),
    }


# ------------------------------------------------------------------------------------------------
# Runs, references, diagnostics
# ------------------------------------------------------------------------------------------------


def jsonl(path):
    records = []
    for line in Path(path).read_text(errors="ignore").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def run_settings(directory):
    """Settings of a coupled run from its coupler and FVM metadata."""
    coupler = json.loads((directory / "solution/coupler_metadata.json").read_text())
    fvm = json.loads((directory / "solution/fvm_metadata.json").read_text())
    configuration, coupling = coupler["configuration"], coupler["coupling"]
    cell = fvm["mesh"]["mesh_generation"]["resolved_background_cell_size"]
    return {
        "N": int(coupling["nominal_fvm_substeps_per_vpm_step"]),
        "cell": round(float(cell), 6),
        "fvm_step": round(float(coupling["fvm_time_step_size"]), 6),
        "interval": round(float(coupling["vpm_time_step_size"]), 6),
        "margin": round(-float(coupling["fvm_domain_bounds"]["xmin"]), 6),
        "inset": float(configuration["transfer_inset_spacings"]),
        "particle_spacing": round(float(coupling["particle_spacing"]), 6),
        "max_iterations": int(configuration["max_coupling_iterations"]),
        "tolerance": float(configuration["normal_velocity_tolerance"]),
        "extrapolation": configuration.get("boundary_data_extrapolation", "linear"),
        "current": configuration.get("wall_circulation_ratio_bounds") == CURRENT_TRANSFER_BOUNDS,
    }


def differences(settings):
    """Settings in which a run differs from the baseline."""
    changed = set()
    for key, value in BASELINE.items():
        same = settings[key] == value if isinstance(value, str) else np.isclose(settings[key], value)
        if not same:
            changed.add(key)
    return changed


def group_of(settings):
    """Study group of a run: the one setting in which it differs from the baseline."""
    changed = differences(settings)
    if not changed:
        return "baseline"
    if changed == {"N"}:
        return "T"
    if changed <= set(ITERATION_KEYS):
        return "M"
    if changed == {"margin"}:
        return "D"
    if changed == {"inset"}:
        return "I"
    if np.isclose(settings["interval"], FIXED_INTERVAL) and "cell" in changed and changed <= {
        "cell", "fvm_step", "N", "particle_spacing"
    }:
        return "S"
    if changed == {"particle_spacing"}:
        return "P"
    return "other"


def label_of(run):
    s = run["settings"]
    group = run["group"]
    if group == "baseline":
        return "baseline"
    if group == "T":
        return rf"$N={s['N']}$"
    if group == "M":
        if s["max_iterations"] == 1:
            return "one pass"
        text = rf"tol. $10^{{{int(round(np.log10(s['tolerance'])))}}}$"
        if s["extrapolation"] != BASELINE["extrapolation"]:
            return f"{s['extrapolation']} first estimate"
        return text
    if group == "D":
        return rf"margin ${s['margin']:g}D$"
    if group == "I":
        return rf"inset ${s['inset']:g}h$"
    if group == "S":
        return rf"$h={s['cell']:g}$ m"
    if group == "P":
        return rf"$h_p={s['particle_spacing']:g}$ m"
    return run["run"]


def reference_name(settings):
    """FVM-only reference on the run's cells and step: h<cell>_dt<step>, or h<cell>_dt<step>_fb<margin>
    when a reference with the run's fine box (box margins beyond the default fine box) exists."""
    name = f"h{settings['cell']:.2f}_dt{settings['fvm_step']:.3f}"
    fine_box = f"{name}_fb{settings['margin']:g}"
    path = REFERENCE_DIR / fine_box
    return fine_box if path.exists() or path.is_symlink() else name


def reference_cost(name):
    """Wall seconds per flow second of an FVM-only reference, from its performance log."""
    path = REFERENCE_SOLUTION_DIR / name / "performance.jsonl"
    if not path.exists():
        return np.nan
    records = jsonl(path)
    seconds = sum(r["step_seconds"]["mean"] for r in records if "step_seconds" in r)
    return seconds / records[-1]["time"] if records else np.nan


def read_references(window):
    references = {}
    for directory in sorted(REFERENCE_DIR.glob("h*_dt*")):
        if not (directory / "forces_history.csv").exists():
            references[directory.name] = {"name": directory.name, "directory": directory, "stats": None,
                                          "end": 0.0, "cost": np.nan, "waiting": "no samples yet"}
            continue
        forces = read_forces(directory)
        row = {"name": directory.name, "directory": directory, "forces": forces,
               "end": float(forces["time"][-1]), "cost": reference_cost(directory.name)}
        try:
            row["stats"] = corrected(shedding_statistics(forces, *window))
        except (IndexError, ValueError):
            row["stats"] = None
        references[directory.name] = row
    return references


def window_records(records, window):
    return [r for r in records if window[0] <= r["time"] <= window[1]]


def interface_measures(directory, window):
    """Coupling measures over the statistics window and the cost over the whole run."""
    path = directory / "solution/coupler_diagnostics.jsonl"
    out = {"iterations": np.nan, "cost": np.nan, "budget_error_median": np.nan,
           "budget_error_max": np.nan, "wall_ratio_median": np.nan, "wall_ratio_p01": np.nan,
           "wall_ratio_p99": np.nan, **{f"mismatch_rms_{side}": np.nan for side in SIDES}}
    if not path.exists():
        return out
    records = jsonl(path)
    if not records:
        return out
    total = sum(r.get("timing_seconds", {}).get("total", 0.0) for r in records)
    out["cost"] = total / records[-1]["time"]
    inside = window_records(records, window)
    if not inside:
        return out
    out["iterations"] = float(np.mean([r["coupling_iteration"]["iterations"] for r in inside
                                       if "coupling_iteration" in r]))
    for side in SIDES:
        values = [r["fvm_boundary_data"].get(f"rms_velocity_mismatch_{side}") for r in inside]
        values = np.array([v for v in values if v is not None], float)
        if values.size:
            out[f"mismatch_rms_{side}"] = float(np.sqrt(np.mean(values**2)) / FREESTREAM_SPEED)
    budget = np.array([r["transfer"]["fvm_budget_relative_error"] for r in inside
                       if "fvm_budget_relative_error" in r.get("transfer", {})])
    if budget.size:
        out["budget_error_median"], out["budget_error_max"] = float(np.median(budget)), float(budget.max())
    ratio = np.array([r["transfer"]["fvm_budget_wall_circulation_ratio"] for r in inside
                      if "fvm_budget_wall_circulation_ratio" in r.get("transfer", {})])
    if ratio.size:
        out["wall_ratio_median"] = float(np.median(ratio))
        out["wall_ratio_p01"], out["wall_ratio_p99"] = (float(v) for v in np.percentile(ratio, (1, 99)))
    return out


def read_runs(runs_dir, references, window):
    """Every run of the current transfer with its settings, statistics, errors and measures."""
    runs, skipped = [], []
    for directory in sorted(p for p in Path(runs_dir).iterdir() if p.is_dir()):
        forces_path = directory / "samples/forces_history.csv"
        if not forces_path.exists() or not (directory / "solution/coupler_metadata.json").exists():
            skipped.append((directory.name, "no samples or metadata yet"))
            continue
        settings = run_settings(directory)
        if not settings["current"]:
            skipped.append((directory.name, "older wall transfer"))
            continue
        forces = read_forces(directory)
        run = {"run": directory.name, "directory": directory, "settings": settings,
               "forces": forces, "end": float(forces["time"][-1])}
        run["group"] = group_of(settings)
        run["label"] = label_of(run)
        run["reference"] = references.get(reference_name(settings))
        run["stats"] = run["error"] = None
        if run["end"] < window[1] - 1e-6:
            run["waiting"] = f"t = {run['end']:.1f} s of {window[1]:g} s"
        elif run["reference"] is None or run["reference"]["stats"] is None:
            run["waiting"] = f"FVM-only reference {reference_name(settings)} not ready"
        else:
            run["stats"] = shedding_statistics(forces, *window)
            run["error"] = {key: run["stats"][key] / run["reference"]["stats"][key] - 1
                            for key in ALL_STATISTICS}
        run.update(interface_measures(directory, window))
        runs.append(run)
    return runs, skipped


def long_runs(runs_dir=RUNS_DIR):
    """Force histories of the references and the coupled baseline (the window uncertainty)."""
    out = {f"FVM {d.name}": read_forces(d) for d in sorted(REFERENCE_DIR.glob("h*_dt*"))
           if (d / "forces_history.csv").exists()}
    for directory in sorted(p for p in Path(runs_dir).glob("*/") if (p / "samples").exists()):
        try:
            settings = run_settings(directory)
        except (FileNotFoundError, KeyError):
            continue
        if settings["current"] and group_of(settings) == "baseline":
            out[directory.name] = read_forces(directory)
    return out


# ------------------------------------------------------------------------------------------------
# Summary table
# ------------------------------------------------------------------------------------------------


def write_summary(runs, references, uncertainty, window, figure_dir):
    figure_dir.mkdir(parents=True, exist_ok=True)
    settings_keys = ("N", "cell", "fvm_step", "interval", "margin", "inset", "particle_spacing",
                     "max_iterations", "tolerance", "extrapolation")
    measure_keys = ("iterations", *(f"mismatch_rms_{side}" for side in SIDES), "budget_error_median",
                    "budget_error_max", "wall_ratio_median", "wall_ratio_p01", "wall_ratio_p99", "cost")
    with open(figure_dir / "study_summary.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([f"# statistics over the whole shedding periods in {window[0]:g}-{window[1]:g} s; "
                         "FVM-only values corrected for blockage (Allen-Vincenti, D/H = 1/32)"])
        writer.writerow(["# error = coupled / FVM-only (same cells and FVM step) - 1; uncertainty = 2 sigma "
                         "of the window (assets/window_uncertainty.py), from: "
                         + (", ".join(uncertainty["runs"]) or "no run reaching 95 s")])
        writer.writerow(["# interface measures over the window; cost = wall seconds per flow second, "
                         "whole run; reference_cost of the matched FVM-only run"])
        writer.writerow(["run", "group", "end_time", *settings_keys, "reference",
                         *ALL_STATISTICS, *(f"error_{k}_%" for k in ALL_STATISTICS),
                         *(f"uncertainty_{k}_%" for k in ALL_STATISTICS),
                         *(f"resolved_{k}" for k in ALL_STATISTICS), *measure_keys, "reference_cost"])
        for run in runs:
            s = run["settings"]
            reference = run["reference"]
            writer.writerow(
                [run["run"], run["group"], f"{run['end']:.2f}", *(s[k] for k in settings_keys),
                 reference["name"] if reference else ""]
                + [f"{run['stats'][k]:.6f}" if run["stats"] else "" for k in ALL_STATISTICS]
                + [f"{100 * run['error'][k]:.3f}" if run["error"] else "" for k in ALL_STATISTICS]
                + [f"{200 * uncertainty['sigma'][k]:.3f}" for k in ALL_STATISTICS]
                + [resolved(run, k, uncertainty) if run["error"] else "" for k in ALL_STATISTICS]
                + [f"{run[k]:.4g}" if k == "cost" or run["end"] >= window[1] - 1e-6 else ""
                   for k in measure_keys]
                + [f"{reference['cost']:.4g}" if reference else ""]
            )
        for name, reference in references.items():
            if reference["stats"] is None:
                continue
            writer.writerow([f"FVM {name}", "reference", f"{reference['end']:.2f}"]
                            + [""] * (len(settings_keys) + 1)
                            + [f"{reference['stats'][k]:.6f}" for k in ALL_STATISTICS]
                            + [""] * (3 * len(ALL_STATISTICS) + len(measure_keys))
                            + [f"{reference['cost']:.4g}"])


def resolved(run, key, uncertainty):
    bar = 2 * uncertainty["sigma"][key]
    return "" if np.isnan(bar) else str(bool(abs(run["error"][key]) > bar))


# ------------------------------------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------------------------------------


def save(figure, name, figure_dir, fit_margins=True):
    """Save in PDF and PNG; with fit_margins, set the side margins from the y labels first."""
    if fit_margins:
        fit_thesis_y_label_margins(figure, figure.axes)
    save_validation_figure(figure, figure_dir / name, figure_format="png")


def figure_forces(run, figure_dir, steady_start):
    """1. Drag and lift histories; dotted lines are the means over the steady periods."""
    reference_dir = run["reference"]["directory"]
    histories = (
        ("FVM", read_forces(reference_dir), REFERENCE_STYLE, True),
        ("FVM-VPM", run["forces"], COUPLED_STYLE, False),
    )
    figure, axes = validation_subplots(2, sharex=True, height_cm=9.5, outer=0.13)
    end = min(forces["time"][-1] for _, forces, _, _ in histories)
    for label, forces, style, bounded in histories:
        first, last, _ = whole_periods(forces, steady_start, end)
        stats = shedding_statistics(forces, steady_start, end)
        factor = blockage_factor(stats["CD"]) if bounded else 1.0
        time = forces["time"] * FREESTREAM_SPEED / DIAMETER
        for axis, key in zip(axes, ("drag_coefficient", "lift_coefficient")):
            axis.plot(time, forces[key] / factor**2, label=label, **style)
        mean = stats["CD"] / factor**2
        axes[0].plot([first, last], [mean, mean], color=style["color"], linestyle=":", linewidth=LINE_WIDTH)
    axes[0].set_ylabel("drag coefficient")
    axes[1].set_ylabel("lift coefficient")
    axes[0].set_ylim(1.05, 1.75)
    axes[1].set_ylim(-0.75, 0.75)
    axes[1].set_xlim(0, end * FREESTREAM_SPEED / DIAMETER)
    ticks = np.arange(0, end * FREESTREAM_SPEED / DIAMETER + 1, 20)
    axes[1].set_xticks(ticks, [f"{tick:g}" for tick in ticks])
    axes[1].set_xlabel(r"$tU_\infty/D$")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([], [], color=COLORS["reference"], linestyle=":", linewidth=LINE_WIDTH))
    labels.append("steady mean")
    axes[0].legend(handles, labels, loc="upper right", ncol=3, frameon=True, framealpha=1.0,
                   edgecolor="0.8", handlelength=1.5, columnspacing=0.8, handletextpad=0.4)
    save(figure, "forces", figure_dir, fit_margins=False)


def profile_statistics(path, quantity, window, scale=1.0):
    """Time mean and rms of a sampled quantity at each transverse position."""
    data = read_columns(path, ("time", "position_y", quantity))
    inside = (data["time"] >= window[0]) & (data["time"] <= window[1])
    positions = np.round(data["position_y"][inside], 6)
    values = data[quantity][inside] / scale
    y = np.unique(positions)
    samples = np.array([values[positions == level] for level in y])
    return y, samples.mean(axis=1), samples.std(axis=1)


def nice_step(value):
    """Smallest step of 1, 2 or 5 times a power of ten that is at least a value."""
    exponent = np.floor(np.log10(value))
    mantissa = min(m for m in (1, 2, 5, 10) if m * 10**exponent >= value)
    return mantissa * 10**exponent


def figure_profiles(run, quantity, name, labels, difference_scale, profile_ticks, figure_dir, steady_start):
    """1. Mean and rms of a field across the wake at x = 2D and 4D, and the difference from the FVM.

    Statistics over the whole periods from the steady start to the end of the shorter run. FVM
    velocities and vorticities are divided by the blockage speed-up U'/U.
    """
    reference_dir = run["reference"]["directory"]
    reference_forces, coupled_forces = read_forces(reference_dir), run["forces"]
    end = min(reference_forces["time"][-1], coupled_forces["time"][-1])
    reference_window = whole_periods(reference_forces, steady_start, end)[:2]
    coupled_window = whole_periods(coupled_forces, steady_start, end)[:2]
    factor = blockage_factor(shedding_statistics(reference_forces, steady_start, end)["CD"])

    set_thesis_style()
    figure, axes = plt.subplots(2, 4, sharey=True, sharex="col", figsize=(WIDTH, 10.5 * CM),
                                gridspec_kw={"width_ratios": (1.25, 1, 1.25, 1)})
    centered_subplots_adjust(figure, outer=0.11, bottom=0.14, top=0.88, wspace=0.12, hspace=0.12)
    for row, station in enumerate((2, 4)):
        y_ref, mean_ref, rms_ref = profile_statistics(
            reference_dir / f"transverse_x{station}.csv", quantity, reference_window, factor)
        y, mean, rms = profile_statistics(
            run["directory"] / f"samples/vpm_transverse_x{station}.csv", quantity, coupled_window)
        for column, (ref_values, values) in enumerate(((mean_ref, mean), (rms_ref, rms))):
            profile_axis, difference_axis = axes[row, 2 * column], axes[row, 2 * column + 1]
            profile_axis.plot(ref_values, y_ref, label="FVM", **REFERENCE_STYLE)
            profile_axis.plot(values, y, label="FVM-VPM", **COUPLED_STYLE)
            difference = values - np.interp(y, y_ref, ref_values)
            difference_axis.axvline(0, color=COLORS["reference"], linewidth=LINE_WIDTH / 2)
            difference_axis.plot(difference * difference_scale, y, **COUPLED_STYLE)
        axes[row, 0].set_ylabel(f"$y/D$ at $x={station}D$")
        axes[row, 0].set_ylim(-2, 2)
    for column, label in enumerate(labels):
        axes[1, column].set_xlabel(label)
    for column, ticks in zip((0, 2), profile_ticks):
        values = np.concatenate([line.get_xdata() for row in (0, 1) for line in axes[row, column].get_lines()])
        low, high = min(values.min(), ticks[0]), max(values.max(), ticks[-1])
        span = ticks[-1] - ticks[0]
        axes[0, column].set_xlim(min(low, ticks[0] - 0.25 * span), max(high, ticks[-1] + 0.25 * span))
        axes[0, column].set_xticks(ticks)
    for column in (1, 3):
        largest = max(abs(line.get_xdata()).max() for row in (0, 1)
                      for line in axes[row, column].get_lines()[1:])
        step = nice_step(0.6 * largest)
        axes[0, column].set_xlim(-max(1.6 * step, 1.1 * largest), max(1.6 * step, 1.1 * largest))
        axes[0, column].set_xticks([-step, step])
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=True, edgecolor="0.8",
                  bbox_to_anchor=(0.5, 0.995))
    save(figure, name, figure_dir, fit_margins=False)


def error_point(axis, x, run, key, uncertainty, color=None, scale=100.0):
    """One error with its +-2 sigma bar; hollow when the bar contains zero (not resolved)."""
    style = STATISTIC_STYLES[key]
    value = scale * run["error"][key]
    bar = 2 * scale * uncertainty["sigma"][key]
    filled = np.isnan(bar) or abs(value) > bar
    color = color or style["color"]
    if not np.isnan(bar):
        axis.errorbar(x, value, yerr=bar, color=color, linewidth=LINE_WIDTH / 1.5, capsize=1.5,
                      linestyle="none", zorder=2)
    axis.plot(x, value, marker=style["marker"], linestyle="none", markersize=3.5, color=color,
              markerfacecolor=color if filled else "white", zorder=3)


def statistic_legend(figure, keys=STATISTICS, extra=()):
    handles = [Line2D([], [], linestyle="none", markersize=3.5, label=STATISTIC_SYMBOLS[key],
                      **STATISTIC_STYLES[key]) for key in keys]
    handles.append(Line2D([], [], linestyle="none", marker="o", markersize=3.5, color=COLORS["gray"],
                          markerfacecolor="white", label=r"inside $\pm2\sigma$"))
    figure.legend(handles=[*handles, *extra], loc="upper center", ncol=len(handles) + len(extra),
                  frameon=True, edgecolor="0.8", bbox_to_anchor=(0.5, 0.995), handlelength=1.2,
                  columnspacing=0.8, handletextpad=0.3)


GROUP_ORDER = ("baseline", "T", "M", "D", "I", "S", "P", "other")


def ordered(runs):
    def key(run):
        s = run["settings"]
        return (GROUP_ORDER.index(run["group"]), s["interval"], s["max_iterations"], -s["tolerance"],
                s["extrapolation"], s["margin"], s["inset"], s["cell"], s["particle_spacing"])
    return sorted(runs, key=key)


def figure_overview(runs, uncertainty, figure_dir):
    """2. Error of every run against its FVM-only reference, with the window uncertainty."""
    runs = ordered(runs)
    set_thesis_style()
    figure, axes = plt.subplots(len(STATISTICS), 1, sharex=True, figsize=(WIDTH, 15.0 * CM))
    centered_subplots_adjust(figure, outer=0.12, bottom=0.3, top=0.93, hspace=0.1)
    x = np.arange(len(runs))
    for axis, key in zip(axes, STATISTICS):
        bar = 200 * uncertainty["sigma"][key]
        if not np.isnan(bar):
            axis.axhspan(-bar, bar, color=COLORS["background_light"], alpha=0.4, linewidth=0, zorder=0)
        axis.axhline(0, color=COLORS["black"], linewidth=LINE_WIDTH / 2)
        for position, run in zip(x, runs):
            error_point(axis, position, run, key, uncertainty)
        axis.set_ylabel(f"{STATISTIC_SYMBOLS[key]} error ($\\%$)")
    for number in range(1, len(runs)):
        if runs[number]["group"] != runs[number - 1]["group"]:
            for axis in axes:
                axis.axvline(number - 0.5, color=COLORS["light_gray"], linewidth=LINE_WIDTH / 2)
    axes[-1].set_xticks(x, [run["label"] for run in runs], rotation=90)
    axes[-1].set_xlim(-0.6, len(runs) - 0.4)
    statistic_legend(figure)
    save(figure, "error_overview", figure_dir)


def fitted_order(x, error, bar):
    """Least-squares order p of |e| = C x^p over the points outside +-2 sigma (all if < 2)."""
    x, error = np.asarray(x, float), np.abs(np.asarray(error, float))
    use = error > bar if not np.isnan(bar) else np.ones_like(error, bool)
    note = ""
    if use.sum() < 2:
        use, note = error > 0, " (unresolved points)"
    if use.sum() < 2:
        return None
    slope, intercept = np.polyfit(np.log(x[use]), np.log(error[use]), 1)
    return slope, intercept, note


def convergence_panel(axis, series, variable, uncertainty, xlabel):
    """|error| against a step size, log-log, with the fitted order of each statistic."""
    for key in STATISTICS:
        style = STATISTIC_STYLES[key]
        x = np.array([run["settings"][variable] for run in series])
        error = np.array([run["error"][key] for run in series])
        bar = 2 * uncertainty["sigma"][key]
        for xi, ei in zip(x, error):
            filled = np.isnan(bar) or abs(ei) > bar
            axis.plot(xi, 100 * abs(ei), linestyle="none", marker=style["marker"], markersize=3.5,
                      color=style["color"], markerfacecolor=style["color"] if filled else "white")
        fit = fitted_order(x, error, bar)
        if fit:
            slope, intercept, note = fit
            line = np.geomspace(x.min(), x.max(), 20)
            axis.plot(line, 100 * np.exp(intercept) * line**slope, color=style["color"],
                      linewidth=LINE_WIDTH / 1.5, label=rf"{STATISTIC_SYMBOLS[key]}: $p={slope:.2f}${note}")
        if not np.isnan(bar):
            axis.axhline(100 * bar, color=style["color"], linestyle=":", linewidth=LINE_WIDTH / 2)
    axis.set_xscale("log")
    axis.set_yscale("log")
    levels = []
    for level in sorted({run["settings"][variable] for run in series}):
        if not levels or level / levels[-1] >= 1.3:  # labelled levels stay apart on the log axis
            levels.append(level)
    axis.set_xticks(levels, [f"{level:g}" for level in levels])
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.set_xlabel(xlabel)
    axis.legend(loc="best", frameon=True, framealpha=0.9, edgecolor="0.8", handlelength=1.2)


def figure_interval(runs, uncertainty, figure_dir):
    """3. Temporal convergence: |error| against the coupling interval N dt."""
    series = sorted((r for r in runs if r["group"] in ("baseline", "T") and r["error"]),
                    key=lambda r: r["settings"]["interval"])
    if len(series) < 2:
        return f"needs the baseline and at least one N run, has {len(series)}"
    set_thesis_style()
    figure, axis = plt.subplots(figsize=(WIDTH, 9.5 * CM))
    centered_subplots_adjust(figure, outer=0.14, bottom=0.15, top=0.74)
    convergence_panel(axis, series, "interval", uncertainty, r"coupling interval $N\Delta t\,U_\infty/D$")
    axis.set_ylabel(r"$|$error$|$ ($\%$)")
    step = BASELINE["fvm_step"]
    top = axis.secondary_xaxis("top", functions=(lambda t: t / step, lambda n: n * step))
    top.set_xticks([r["settings"]["N"] for r in series], [f"{r['settings']['N']}" for r in series])
    top.set_xlabel(r"$N$ at $\Delta t=0.008$ s")
    axis.tick_params(axis="x", which="both", top=False)
    statistic_legend(figure)
    save(figure, "error_coupling_interval", figure_dir)


def figure_cell_size(runs, uncertainty, figure_dir):
    """4. Spatial convergence at the fixed interval, and the particle-spacing pair."""
    series = sorted((r for r in runs if r["error"] and np.isclose(r["settings"]["interval"], FIXED_INTERVAL)
                     and r["group"] in ("S", "T")), key=lambda r: r["settings"]["cell"])
    pair = [r for r in runs if r["error"] and r["group"] in ("baseline", "P")]
    if len(series) < 2 and len(pair) < 2:
        return f"needs two cell sizes at {FIXED_INTERVAL} s ({len(series)} done) or the spacing pair ({len(pair)} done)"
    set_thesis_style()
    figure, axes = plt.subplots(1, 2, figsize=(WIDTH, 8.5 * CM), gridspec_kw={"width_ratios": (1.6, 1)})
    centered_subplots_adjust(figure, outer=0.13, bottom=0.17, top=0.86, wspace=0.35)
    if len(series) >= 2:
        convergence_panel(axes[0], series, "cell", uncertainty,
                          rf"cell size $h/D$, $N\Delta t={FIXED_INTERVAL}$ s")
    else:
        axes[0].text(0.5, 0.5, f"waits for the cell-size runs\n({len(series)} of 3 done)",
                     transform=axes[0].transAxes, ha="center", va="center", color=COLORS["gray"])
    axes[0].set_ylabel(r"$|$error$|$ ($\%$)")
    pair = sorted(pair, key=lambda r: r["settings"]["particle_spacing"])
    for position, run in enumerate(pair):
        for offset, key in zip((-0.15, 0, 0.15), STATISTICS):
            error_point(axes[1], position + offset, run, key, uncertainty)
    axes[1].axhline(0, color=COLORS["black"], linewidth=LINE_WIDTH / 2)
    axes[1].set_xticks(range(len(pair)), [f"{r['settings']['particle_spacing']:g}" for r in pair])
    axes[1].set_xlabel(r"particle spacing $h_p/D$")
    axes[1].set_xlim(-0.6, max(len(pair), 2) - 0.4)
    axes[1].set_ylabel(r"error ($\%$)")
    statistic_legend(figure)
    save(figure, "error_cell_size", figure_dir)


def figure_domain(runs, uncertainty, figure_dir):
    """5. Error against the box margin and against the transfer inset."""
    margin = sorted((r for r in runs if r["error"] and r["group"] in ("baseline", "D")),
                    key=lambda r: r["settings"]["margin"])
    inset = sorted((r for r in runs if r["error"] and r["group"] in ("baseline", "I")),
                   key=lambda r: r["settings"]["inset"])
    if len(margin) < 2 and len(inset) < 2:
        return f"needs a margin or inset run next to the baseline (margin {len(margin)}, inset {len(inset)} done)"
    set_thesis_style()
    figure, axes = plt.subplots(1, 2, sharey=True, figsize=(WIDTH, 8.0 * CM))
    centered_subplots_adjust(figure, outer=0.12, bottom=0.18, top=0.86, wspace=0.1)
    for axis, series, variable, label, step in (
        (axes[0], margin, "margin", r"box margin $/D$", 0.06),
        (axes[1], inset, "inset", r"transfer inset $/h_p$", 0.25),
    ):
        for run in series:
            for offset, key in zip((-step, 0, step), STATISTICS):
                error_point(axis, run["settings"][variable] + offset, run, key, uncertainty)
        axis.axhline(0, color=COLORS["black"], linewidth=LINE_WIDTH / 2)
        axis.set_xlabel(label)
    axes[0].set_xticks([1.6, 2.0, 2.4, 3.2, 4.0])
    axes[1].set_xticks([4, 8.75, 14], ["4", "8.75", "14"])
    axes[0].set_ylabel(r"error ($\%$)")
    statistic_legend(figure)
    save(figure, "error_domain", figure_dir)


def figure_iterations(runs, uncertainty, figure_dir):
    """6. Error and mean coupling-iteration count against the tolerance."""
    series = [r for r in ordered(runs) if r["error"] and r["group"] in ("baseline", "M")]
    if len(series) < 2:
        return f"needs an iteration run next to the baseline ({len(series)} done)"
    set_thesis_style()
    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(WIDTH, 12.0 * CM),
                                gridspec_kw={"height_ratios": (1.6, 1)})
    centered_subplots_adjust(figure, outer=0.12, bottom=0.33, top=0.9, hspace=0.1)
    for position, run in enumerate(series):
        for offset, key in zip((-0.15, 0, 0.15), STATISTICS):
            error_point(axes[0], position + offset, run, key, uncertainty)
        axes[1].bar(position, run["iterations"], width=0.5, color=COLORS["hybrid"], linewidth=0)
    axes[0].axhline(0, color=COLORS["black"], linewidth=LINE_WIDTH / 2)
    axes[0].set_ylabel(r"error ($\%$)")
    axes[1].set_ylabel("mean iterations")
    axes[1].set_ylim(0, max(2.0, 1.15 * max(r["iterations"] for r in series)))
    labels = [r"tol. $10^{-5}$ (baseline)" if r["group"] == "baseline" else r["label"] for r in series]
    axes[1].set_xticks(range(len(series)), labels, rotation=90)
    axes[1].set_xlim(-0.6, len(series) - 0.4)
    statistic_legend(figure)
    save(figure, "error_coupling_iterations", figure_dir)


def figure_interface(runs, figure_dir):
    """7. Interface measures of every run over the window."""
    runs = [r for r in ordered(runs) if not np.isnan(r["budget_error_median"])]
    if not runs:
        return "no run has reached the end of the window"
    set_thesis_style()
    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(WIDTH, 15.0 * CM))
    centered_subplots_adjust(figure, outer=0.12, bottom=0.3, top=0.93, hspace=0.1)
    x = np.arange(len(runs))
    markers = {"x-": "<", "x+": ">", "y-": "v", "y+": "^"}
    colors = {"x-": COLORS["teal"], "x+": COLORS["red"], "y-": COLORS["green"], "y+": COLORS["blue"]}
    for side in SIDES:
        axes[0].plot(x, [1e3 * r[f"mismatch_rms_{side}"] for r in runs], linestyle="none", marker=markers[side],
                     markersize=3.5, color=colors[side], label=SIDE_NAMES[side])
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axes[0].yaxis.set_minor_formatter(NullFormatter())
    axes[0].set_ylabel(r"rms $|\Delta u|/U_\infty$ $\times10^3$")
    axes[0].legend(loc="upper center", ncol=4, frameon=True, edgecolor="0.8",
                   bbox_to_anchor=(0.5, 1.32), handlelength=1.0)
    axes[1].plot(x, [1e4 * r["budget_error_median"] for r in runs], "o", markersize=3.5,
                 color=COLORS["teal"], label="median")
    axes[1].plot(x, [1e4 * r["budget_error_max"] for r in runs], "_", markersize=6, color=COLORS["teal"],
                 label="maximum")
    axes[1].set_ylim(0, 1.6e4 * np.nanmax([r["budget_error_max"] for r in runs]))  # room for the legend
    axes[1].set_ylabel(r"budget error $\times10^4$")
    axes[1].legend(loc="upper right", frameon=True, edgecolor="0.8", ncol=2)
    median = np.array([r["wall_ratio_median"] for r in runs])
    low, high = np.array([r["wall_ratio_p01"] for r in runs]), np.array([r["wall_ratio_p99"] for r in runs])
    axes[2].axhspan(*CURRENT_TRANSFER_BOUNDS, color=COLORS["background_light"], alpha=0.4, linewidth=0)
    axes[2].errorbar(x, median, yerr=(median - low, high - median), fmt="o", markersize=3.5,
                     color=COLORS["hybrid"], linewidth=LINE_WIDTH / 1.5, capsize=1.5)
    axes[2].axhline(1, color=COLORS["black"], linewidth=LINE_WIDTH / 2)
    axes[2].set_ylabel("near-wall ratio")
    for number in range(1, len(runs)):
        if runs[number]["group"] != runs[number - 1]["group"]:
            for axis in axes:
                axis.axvline(number - 0.5, color=COLORS["light_gray"], linewidth=LINE_WIDTH / 2)
    axes[-1].set_xticks(x, [r["label"] for r in runs], rotation=90)
    axes[-1].set_xlim(-0.6, len(runs) - 0.4)
    save(figure, "interface_measures", figure_dir)


def figure_cost(runs, uncertainty, figure_dir):
    """8. Wall time per flow time against the error of each statistic, one point per run."""
    runs = [r for r in runs if r["error"] and not np.isnan(r["cost"])]
    if not runs:
        return "no run with errors and timing"
    set_thesis_style()
    figure, axes = plt.subplots(1, len(STATISTICS), sharey=True, figsize=(WIDTH, 9.5 * CM))
    centered_subplots_adjust(figure, outer=0.12, bottom=0.16, top=0.72, wspace=0.08)
    references = {r["reference"]["name"]: r["reference"]["cost"] for r in runs}
    for axis, key in zip(axes, STATISTICS):
        bar = 200 * uncertainty["sigma"][key]
        if not np.isnan(bar):
            axis.axvspan(0, bar, color=COLORS["background_light"], alpha=0.4, linewidth=0)
        for run in runs:
            axis.plot(100 * abs(run["error"][key]), run["cost"], marker=STATISTIC_STYLES[key]["marker"],
                      linestyle="none", markersize=3.5, color=GROUPS[run["group"]][1])
        for name, cost in references.items():
            if not np.isnan(cost):
                axis.axhline(cost, color=COLORS["reference"], linestyle=":", linewidth=LINE_WIDTH / 2)
        axis.set_xlabel(f"{STATISTIC_SYMBOLS[key]} $|$error$|$ ($\\%$)")
        largest = max(100 * abs(run["error"][key]) for run in runs)
        axis.set_xlim(0, 1.15 * max(largest, 0.0 if np.isnan(bar) else bar))
    costs = [run["cost"] for run in runs] + [c for c in references.values() if not np.isnan(c)]
    axes[0].set_yscale("log")
    axes[0].set_ylim(0.8 * min(costs), 1.25 * max(costs))
    axes[0].yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axes[0].yaxis.set_minor_formatter(NullFormatter())
    axes[0].set_ylabel("wall seconds per flow second")
    present = [g for g in GROUP_ORDER if any(r["group"] == g for r in runs)]
    handles = [Line2D([], [], linestyle="none", marker="o", markersize=3.5, color=GROUPS[g][1],
                      label=GROUPS[g][0]) for g in present]
    handles.append(Line2D([], [], color=COLORS["reference"], linestyle=":", linewidth=LINE_WIDTH / 2,
                          label="FVM only"))
    figure.legend(handles=handles, loc="upper center", ncol=3, frameon=True, edgecolor="0.8",
                  bbox_to_anchor=(0.5, 0.995), handlelength=1.2, columnspacing=0.8)
    save(figure, "cost_accuracy", figure_dir)


# ------------------------------------------------------------------------------------------------


def main():
    from assets.window_uncertainty import window_uncertainty

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=RUNS_DIR, help="folder of the coupled runs")
    parser.add_argument("--figures", type=Path, default=FIGURE_DIR, help="output folder")
    parser.add_argument("--window", type=float, nargs=2, default=STUDY_WINDOW, help="statistics window [s]")
    parser.add_argument("--steady-start", type=float, default=STEADY_START,
                        help="start of the baseline force and profile statistics [s]")
    arguments = parser.parse_args()
    window, steady_start, figure_dir = tuple(arguments.window), arguments.steady_start, arguments.figures

    references = read_references(window)
    runs, skipped = read_runs(arguments.runs, references, window)
    uncertainty = window_uncertainty(long_runs(arguments.runs), window, steady_start,
                                     shedding_statistics, ALL_STATISTICS)

    print(f"Statistics window {window[0]:g}-{window[1]:g} s; runs in {arguments.runs}")
    for name, reason in skipped:
        print(f"  skipped {name}: {reason}")
    for run in ordered(runs):
        state = run.get("waiting") or ", ".join(
            f"{key} {100 * run['error'][key]:+.2f} %" for key in STATISTICS)
        print(f"  {run['group']:8s} {run['run']:40s} t = {run['end']:6.1f} s  {state}")
    print("Window uncertainty (2 sigma) from " + (", ".join(uncertainty["runs"]) or "no run reaching 95 s")
          + ": " + ", ".join(f"{key} {200 * uncertainty['sigma'][key]:.2f} %" for key in STATISTICS))
    if not uncertainty["runs"] or np.isnan(uncertainty["s_d"]["CD"]):
        print("  (drift term s_d needs two runs reaching 95 s)")

    write_summary(runs, references, uncertainty, window, figure_dir)
    print(f"Saved: {figure_dir / 'study_summary.csv'}")

    done = [r for r in runs if r["error"]]
    baselines = [r for r in runs if r["group"] == "baseline"]
    baseline = max(baselines, key=lambda r: r["end"]) if baselines else None
    if baseline is None or baseline["reference"] is None:
        print("figure 1 (forces, profiles): waits for the baseline run")
    elif baseline["end"] < steady_start + MINIMUM_STEADY_LENGTH:
        print(f"figure 1 (forces, profiles): waits for the baseline to reach "
              f"{steady_start + MINIMUM_STEADY_LENGTH:g} s (t = {baseline['end']:.1f} s)")
    else:
        figure_forces(baseline, figure_dir, steady_start)
        figure_profiles(baseline, "velocity_x", "velocity_profiles",
                        (r"$\bar u/U_\infty$", r"$\Delta\bar u/U_\infty$ ($\%$)",
                         r"$u'_\mathrm{rms}/U_\infty$", r"$\Delta u'_\mathrm{rms}/U_\infty$ ($\%$)"),
                        100, ([0.5, 1.0], [0.0, 0.2]), figure_dir, steady_start)
        figure_profiles(baseline, "vorticity_z", "vorticity_profiles",
                        (r"$\bar\omega_z D/U_\infty$", r"$\Delta\bar\omega_z D/U_\infty$",
                         r"$\omega'_\mathrm{rms}D/U_\infty$", r"$\Delta\omega'_\mathrm{rms}D/U_\infty$"),
                        1, ([-1.0, 0.0, 1.0], [0.0, 1.0, 2.0]), figure_dir, steady_start)
    if done:
        figure_overview(done, uncertainty, figure_dir)
    else:
        print("figure 2 (error overview): waits for a run to reach the end of the window")
    for number, name, make in (
        (3, "temporal convergence", lambda: figure_interval(done, uncertainty, figure_dir)),
        (4, "spatial convergence", lambda: figure_cell_size(done, uncertainty, figure_dir)),
        (5, "domain", lambda: figure_domain(done, uncertainty, figure_dir)),
        (6, "coupling iterations", lambda: figure_iterations(done, uncertainty, figure_dir)),
        (7, "interface measures", lambda: figure_interface(
            [r for r in runs if r["end"] >= window[1] - 1e-6], figure_dir)),
        (8, "cost against accuracy", lambda: figure_cost(done, uncertainty, figure_dir)),
    ):
        reason = make()
        if reason:
            print(f"figure {number} ({name}): waits, {reason}")


if __name__ == "__main__":
    main()
