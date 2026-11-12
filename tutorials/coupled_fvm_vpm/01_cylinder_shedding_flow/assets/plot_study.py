#!/usr/bin/env python3
"""Coupling-interface study of the cylinder at Re = 150: statistics, errors and figures.

Reads the samples of every coupled run in solution/sensitivity_analysis/ and of the four FVM-only
grids in reference_flow/samples/, and writes to figures/:

    forces.png                   drag and lift of the recommended setup and of the FVM
    velocity_profiles.png        mean and rms velocity across the wake, with their differences
    vorticity_profiles.png       mean and rms vorticity across the wake, with their differences
    error_convergence.png        FVM grid error and coupling error against the cell size
    error_coupling_interval.png  coupling error and boundary mismatch against the coupling interval
    error_sensitivity.png        change of each statistic for every parameter of the study
    study_summary.csv            every statistic and error behind the figures

The coupling error of a run is its difference from the FVM on the same cells and step,
Q_coupled / Q_FVM(h, dt) - 1. The FVM domain is bounded by slip walls, the coupled one is not, so
every FVM value is first corrected for the wall blockage (README, "How the error is measured").

Usage, from the case directory:  python -m assets.plot_study
"""

import csv
from pathlib import Path
import re

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np

from openonda.plotting import CM, COLORS, LINE_WIDTH, MAX_FIGURE_WIDTH_CM, REFERENCE_STYLE
from openonda.plotting import centered_subplots_adjust, save_validation_figure, set_thesis_style
from openonda.plotting import validation_subplots

CASE_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = CASE_DIR / "solution/sensitivity_analysis"
REFERENCE_DIR = CASE_DIR / "reference_flow/samples"
FIGURE_DIR = CASE_DIR / "figures"

DIAMETER = 1.0  # m
FREESTREAM_SPEED = 1.0  # m/s
REFERENCE_HEIGHT = 32.0  # m, distance between the slip walls of the FVM-only domain
BASE_CELL, BASE_FVM_STEP = 0.04, 0.008  # m, s; the FVM step scales with the cell
PARTICLE_STRAIN_RATE = 19.4  # 1/s, largest particle strain rate of the baseline
# Recommended setup, the baseline of the study: h = 0.04 m, dt_FVM = 0.008 s, N = 5 (particle CFL
# number 0.78), box margin 2.4D, inset 8.75 cells, iterations to tolerance. Its
# forces and profiles are the main results.
BASELINE = "box2.4_sub5_inset8.75"

# Every coupled run reaches 55 s, so the study compares runs over 40-55 s. Steady shedding (lift
# amplitude constant to 0.01 %) starts at 55 s; the forces and profiles use all whole periods
# from there to the end of the run.
STUDY_WINDOW = (40.0, 55.0)
STEADY_START = 55.0

STATISTICS = ("CD", "CL", "St")
STATISTIC_NAMES = {"CD": "mean drag", "CL": "lift amplitude", "St": "Strouhal number"}
STATISTIC_STYLES = {
    "CD": dict(color=COLORS["teal"], marker="o"),
    "CL": dict(color=COLORS["red"], marker="s"),
    "St": dict(color=COLORS["green"], marker="^"),
}
COUPLED_STYLE = dict(color=COLORS["hybrid"], linewidth=LINE_WIDTH)
WIDTH = MAX_FIGURE_WIDTH_CM * CM


# ------------------------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------------------------


def parse_run(name):
    """Settings of a coupled run from its directory name (see allrun.sh)."""
    match = re.match(r"box([\d.]+)_sub(\d+)_inset([\d.]+)(.*)", name)
    margin, substeps, inset, suffix = float(match[1]), int(match[2]), float(match[3]), match[4]
    options = dict(re.findall(r"_([a-z]+)([\d.]*)", suffix))
    cell = float(options.get("h", BASE_CELL))
    fvm_step = BASE_FVM_STEP * cell / BASE_CELL
    return {
        "run": name,
        "margin": margin,
        "N": substeps,
        "inset": inset,
        "cell": cell,
        "fvm_step": fvm_step,
        "vpm_step": substeps * fvm_step,
        "iterations": options.get("k", "tol"),
        "particle_spacing": float(options.get("hp", cell)),
        "integrator": "RK4" if "rk" in options else "RK2",
        "vorticity_floor": float(options.get("vf", 0.01)),
    }


def read_columns(path, columns):
    data = np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8")
    return {column: data[column] for column in columns}


def read_forces(directory):
    """Force-coefficient history of a coupled run or an FVM-only grid."""
    path = next(directory.glob("samples/forces_history.csv"), directory / "forces_history.csv")
    return read_columns(path, ("time", "drag_coefficient", "lift_coefficient"))


def coupler_indicators(run_dir):
    """Mean coupling iterations and mean FVM boundary-velocity mismatch, from coupler.log."""
    log = next(run_dir.glob("solution/coupler.log"), None)
    if log is None:
        return np.nan, np.nan
    text = log.read_text(errors="ignore")
    iterations = [int(n) for n in re.findall(r"coupling iterations \| iterations=(\d+)", text)]
    mismatch = [float(v) for v in re.findall(r"Velocity mismatch, mean\s+([\d.eE+-]+)", text)]
    return (np.mean(iterations) if iterations else 1.0), (np.mean(mismatch) if mismatch else np.nan)


# ------------------------------------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------------------------------------


def blockage_factor(drag):
    """Freestream speed-up U'/U between slip walls (Allen and Vincenti)."""
    ratio = DIAMETER / REFERENCE_HEIGHT
    return 1 + 0.25 * drag * ratio + 0.82 * ratio**2


def corrected(stats):
    """Unbounded-flow statistics of an FVM-only run: forces scale with U^2, St with U."""
    factor = blockage_factor(stats["CD"])
    return {
        "CD": stats["CD"] / factor**2,
        "CL": stats["CL"] / factor**2,
        "St": stats["St"] / factor,
    }


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
    """Mean drag, lift amplitude and Strouhal number over the whole periods in [start, end]."""
    first, last, periods = whole_periods(forces, start, end)
    inside = (forces["time"] >= first) & (forces["time"] <= last)
    time, drag, lift = (
        forces[key][inside] for key in ("time", "drag_coefficient", "lift_coefficient")
    )
    return {
        "CD": np.trapezoid(drag, time) / (time[-1] - time[0]),
        "CL": 0.5 * (vertex(time, lift, np.argmax(lift)) - vertex(time, lift, np.argmin(lift))),
        "St": DIAMETER * periods / ((last - first) * FREESTREAM_SPEED),
    }


def resolution():
    """Smallest difference between runs that the 40-55 s statistics resolve.

    The lift amplitude still grows in that window; the offset of the baseline's 40-55 s value
    from its steady value bounds how much two runs can differ by their growth alone.
    """
    forces = read_forces(RUNS_DIR / BASELINE)
    early = shedding_statistics(forces, *STUDY_WINDOW)
    steady = shedding_statistics(forces, STEADY_START)
    return {key: abs(early[key] / steady[key] - 1) for key in STATISTICS}


def collect():
    """Statistics and errors of every FVM-only grid and coupled run over the study window.

    The error of an FVM grid is its change from the finest grid; the error of a coupled run is its
    coupling error against the FVM grid with its cells and step.
    """
    references = []
    for directory in sorted(REFERENCE_DIR.glob("h*_dt*")):
        stats = corrected(shedding_statistics(read_forces(directory), *STUDY_WINDOW))
        references.append(
            {
                "run": f"FVM {directory.name}",
                "cell": float(directory.name[1:5]),
                "directory": directory,
                "stats": stats,
            }
        )
    references.sort(key=lambda row: row["cell"])
    for row in references:
        row["error"] = {
            key: row["stats"][key] / references[0]["stats"][key] - 1 for key in STATISTICS
        }
    matched = {row["cell"]: row for row in references}

    runs = []
    for path in sorted(RUNS_DIR.glob("*/samples/forces_history.csv")):
        directory = path.parents[1]
        run = parse_run(directory.name)
        forces = read_forces(directory)
        run["directory"], run["reference"] = directory, matched[run["cell"]]
        run["stats"] = shedding_statistics(forces, *STUDY_WINDOW)
        run["error"] = {
            key: run["stats"][key] / run["reference"]["stats"][key] - 1 for key in STATISTICS
        }
        run["mean_iterations"], run["mismatch"] = coupler_indicators(directory)
        runs.append(run)
    return references, runs


def write_summary(references, runs, floor):
    FIGURE_DIR.mkdir(exist_ok=True)
    columns = [
        "run",
        "cell",
        "N",
        "vpm_step",
        "margin",
        "inset",
        "iterations",
        "particle_spacing",
        "integrator",
        "vorticity_floor",
    ]
    with open(FIGURE_DIR / "study_summary.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                f"# statistics over {STUDY_WINDOW[0]:g}-{STUDY_WINDOW[1]:g} s; "
                "FVM values corrected for blockage"
            ]
        )
        writer.writerow(
            [
                "# error: FVM, change from h = 0.03 m; FVM-VPM, against the FVM with "
                "the same cells and step"
            ]
        )
        writer.writerow(["# resolution"] + [f"{key} {100 * floor[key]:.2f}%" for key in STATISTICS])
        writer.writerow([f"# recommended setup: {BASELINE}"])
        writer.writerow(
            columns
            + [
                "mean_iterations",
                "mismatch_mean_U",
                *STATISTICS,
                *(f"error_{key}_%" for key in STATISTICS),
            ]
        )
        for row in references:
            writer.writerow(
                [row["run"], row["cell"]]
                + [""] * 11
                + [f"{row['stats'][key]:.5f}" for key in STATISTICS]
                + [f"{100 * row['error'][key]:.3f}" for key in STATISTICS]
            )
        for row in runs:
            writer.writerow(
                [row[key] for key in columns]
                + [f"{row['mean_iterations']:.2f}", f"{row['mismatch']:.2e}"]
                + [f"{row['stats'][key]:.5f}" for key in STATISTICS]
                + [f"{100 * row['error'][key]:.3f}" for key in STATISTICS]
            )


# ------------------------------------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------------------------------------


def save(figure, name):
    save_validation_figure(figure, FIGURE_DIR / name, figure_format="png")


def figure_forces(run):
    """Drag and lift histories; dotted lines are the means over the steady periods."""
    reference_dir = run["reference"]["directory"]
    histories = (
        ("FVM", read_forces(reference_dir), REFERENCE_STYLE, True),
        ("FVM-VPM", read_forces(run["directory"]), COUPLED_STYLE, False),
    )
    figure, axes = validation_subplots(2, sharex=True, height_cm=9.5, outer=0.13)
    end = min(forces["time"][-1] for _, forces, _, _ in histories)
    for label, forces, style, bounded in histories:
        first, last, _ = whole_periods(forces, STEADY_START, end)
        stats = shedding_statistics(forces, STEADY_START, end)
        factor = blockage_factor(stats["CD"]) if bounded else 1.0
        time = forces["time"] * FREESTREAM_SPEED / DIAMETER
        for axis, key in zip(axes, ("drag_coefficient", "lift_coefficient")):
            axis.plot(time, forces[key] / factor**2, label=label, **style)
        mean = stats["CD"] / factor**2
        axes[0].plot(
            [first, last], [mean, mean], color=style["color"], linestyle=":", linewidth=LINE_WIDTH
        )
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
    axes[0].legend(
        handles,
        labels,
        loc="upper right",
        ncol=3,
        frameon=True,
        framealpha=1.0,
        edgecolor="0.8",
        handlelength=1.5,
        columnspacing=0.8,
        handletextpad=0.4,
    )
    save(figure, "forces")


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


def figure_profiles(run, quantity, name, labels, difference_scale, profile_ticks):
    """Mean and rms of a field across the wake at x = 2D and 4D, and the FVM-VPM minus FVM difference.

    Statistics over the whole periods from the steady start to the end of the shorter run. FVM
    velocities and vorticities are divided by the blockage speed-up U'/U.
    """
    reference_dir = run["reference"]["directory"]
    reference_forces, coupled_forces = read_forces(reference_dir), read_forces(run["directory"])
    end = min(reference_forces["time"][-1], coupled_forces["time"][-1])
    reference_window = whole_periods(reference_forces, STEADY_START, end)[:2]
    coupled_window = whole_periods(coupled_forces, STEADY_START, end)[:2]
    factor = blockage_factor(shedding_statistics(reference_forces, STEADY_START, end)["CD"])

    set_thesis_style()
    figure, axes = plt.subplots(
        2,
        4,
        sharey=True,
        sharex="col",
        figsize=(WIDTH, 10.5 * CM),
        gridspec_kw={"width_ratios": (1.25, 1, 1.25, 1)},
    )
    centered_subplots_adjust(figure, outer=0.11, bottom=0.14, top=0.88, wspace=0.12, hspace=0.12)
    for row, station in enumerate((2, 4)):
        y_ref, mean_ref, rms_ref = profile_statistics(
            reference_dir / f"transverse_x{station}.csv", quantity, reference_window, factor
        )
        y, mean, rms = profile_statistics(
            run["directory"] / f"samples/vpm_transverse_x{station}.csv", quantity, coupled_window
        )
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
    # Ticks well inside each panel, so that neighbouring tick labels stay apart
    for column, ticks in zip((0, 2), profile_ticks):
        values = np.concatenate(
            [line.get_xdata() for row in (0, 1) for line in axes[row, column].get_lines()]
        )
        low, high = min(values.min(), ticks[0]), max(values.max(), ticks[-1])
        span = ticks[-1] - ticks[0]
        axes[0, column].set_xlim(
            min(low, ticks[0] - 0.25 * span), max(high, ticks[-1] + 0.25 * span)
        )
        axes[0, column].set_xticks(ticks)
    for column in (1, 3):
        largest = max(
            abs(line.get_xdata()).max()
            for row in (0, 1)
            for line in axes[row, column].get_lines()[1:]
        )
        step = nice_step(0.6 * largest)
        axes[0, column].set_xlim(-max(1.6 * step, 1.1 * largest), max(1.6 * step, 1.1 * largest))
        axes[0, column].set_xticks([-step, step])
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=2,
        frameon=True,
        edgecolor="0.8",
        bbox_to_anchor=(0.5, 0.995),
    )
    save(figure, name)


def figure_convergence(references, runs):
    """FVM grid error and FVM-VPM coupling error against the cell size.

    The coupled series keeps the coupling interval at 0.048 s; cells, particles and the FVM step
    are refined together.
    """
    series = sorted(
        (run for run in runs if np.isclose(run["vpm_step"], 0.048)),
        key=lambda run: run["cell"],
    )
    set_thesis_style()
    figure, axes = plt.subplots(1, 2, sharey=True, figsize=(WIDTH, 8.5 * CM))
    centered_subplots_adjust(figure, outer=0.13, bottom=0.3, top=0.9, wspace=0.08)
    for axis, rows, title in (
        (axes[0], references, "FVM grid error"),
        (axes[1], series, "FVM-VPM coupling error"),
    ):
        h = [100 * row["cell"] for row in rows]
        for key in STATISTICS:
            axis.plot(
                h,
                [100 * row["error"][key] for row in rows],
                label=STATISTIC_NAMES[key],
                linewidth=LINE_WIDTH,
                **STATISTIC_STYLES[key],
            )
        axis.axhline(0, color=COLORS["reference"], linewidth=LINE_WIDTH / 2)
        axis.set_xlim(2, 9)
        axis.set_xticks([3, 4, 6, 8])
        axis.set_xlabel(r"$100\,h/D$")
        axis.set_title(title)
    axes[0].set_ylabel(r"error ($\%$)")
    axes[0].set_ylim(-7, 3)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=True,
        edgecolor="0.8",
        handlelength=1.2,
        columnspacing=0.8,
        handletextpad=0.4,
        bbox_to_anchor=(0.5, 0.0),
    )
    save(figure, "error_convergence")


def study_family(runs):
    """Runs that differ from the baseline only in the coupling interval."""
    return [
        run
        for run in runs
        if run["cell"] == BASE_CELL
        and run["margin"] == 2.4
        and run["inset"] == 8.75
        and run["iterations"] == "tol"
        and run["particle_spacing"] == BASE_CELL
        and run["integrator"] == "RK2"
        and run["vorticity_floor"] == 0.01
    ]


def figure_interval(runs, floor):
    """Coupling error and FVM boundary mismatch against the coupling interval."""
    family = study_family(runs)
    set_thesis_style()
    figure, axes = plt.subplots(2, 2, sharex=True, figsize=(WIDTH, 11.5 * CM))
    centered_subplots_adjust(figure, outer=0.12, bottom=0.12, top=0.86, wspace=0.42, hspace=0.12)
    panels = (
        ("CD", "drag error"),
        ("CL", "lift-amplitude error"),
        ("St", "Strouhal error"),
        ("mismatch", "boundary mismatch"),
    )
    series = sorted(family, key=lambda run: run["N"])
    interval = [run["vpm_step"] * FREESTREAM_SPEED / DIAMETER for run in series]
    for axis, (key, _) in zip(axes.flat, panels):
        values = [
            1e3 * run["mismatch"] if key == "mismatch" else 100 * run["error"][key]
            for run in series
        ]
        axis.plot(
            interval,
            values,
            linewidth=LINE_WIDTH,
            marker="o",
            linestyle="-",
            color=COLORS["hybrid"],
        )
    for axis, (key, label) in zip(axes.flat, panels):
        if key == "mismatch":
            axis.set_ylabel(r"$10^3\,\overline{|\Delta u|}/U_\infty$")
        else:
            axis.axhspan(
                -100 * floor[key],
                100 * floor[key],
                color=COLORS["background_light"],
                alpha=0.5,
                linewidth=0,
                zorder=0,
            )
            axis.set_ylabel(rf"{label} ($\%$)")
    for axis in axes[0]:
        top = axis.secondary_xaxis(
            "top",
            functions=(lambda t: t * PARTICLE_STRAIN_RATE, lambda c: c / PARTICLE_STRAIN_RATE),
        )
        top.set_xticks([0, 0.5, 1])
        top.set_xlabel("particle CFL number")
        axis.tick_params(axis="x", which="both", top=False)
    for axis in axes[1]:
        axis.set_xlabel(r"$\Delta t_\mathrm{VPM}U_\infty/D$")
        axis.set_xlim(0, 0.052)
        axis.set_xticks([0, 0.02, 0.04])
    axes[1, 1].set_ylim(0, 9.5)
    save(figure, "error_coupling_interval")


def sensitivity_entries(runs):
    """(label, run) of every run that changes one setting of the baseline, grouped by setting."""
    family = study_family(runs)
    base_cell = [run for run in runs if run["cell"] == BASE_CELL and run["run"] != BASELINE]

    def only(**changes):
        reference = {
            "N": 5,
            "margin": 2.4,
            "inset": 8.75,
            "iterations": "tol",
            "particle_spacing": BASE_CELL,
            "integrator": "RK2",
            "vorticity_floor": 0.01,
        }
        reference.update(changes)
        return [
            run for run in base_cell if all(run[key] == value for key, value in reference.items())
        ]

    interval = lambda run: rf"$\Delta t_\mathrm{{VPM}}={run['vpm_step']:.3f}$ s"
    groups = [
        [
            (f"{n} coupling iteration{'s' if n > 1 else ''}", run)
            for n in (1, 2, 4)
            for run in only(iterations=str(n))
        ],
        [
            (interval(run), run)
            for run in sorted(
                (run for run in family if run["N"] != 5),
                key=lambda run: run["N"],
            )
        ],
        [
            (f"box margin ${margin:g}D$", run)
            for margin in (1.6, 2.0, 3.2, 4.0)
            for run in only(margin=margin)
        ],
        [
            (f"transfer inset ${inset:g}h$", run)
            for inset in (4.0, 14.0)
            for run in only(inset=inset)
        ],
        [("particle spacing $2h$", run) for run in only(particle_spacing=2 * BASE_CELL)]
        + [("RK4 particle integration", run) for run in only(integrator="RK4")]
        + [(r"vorticity floor $10^{-3}$ s$^{-1}$", run) for run in only(vorticity_floor=0.001)],
    ]
    return [group for group in groups if group]


def figure_sensitivity(runs, floor):
    """Change of each statistic from the baseline, one setting at a time."""
    baseline = next(run for run in runs if run["run"] == BASELINE)
    set_thesis_style()
    figure, axis = plt.subplots(figsize=(WIDTH, 13.0 * CM))
    centered_subplots_adjust(figure, outer=0.1, bottom=0.42, top=0.86)
    position, ticks, labels = 0.0, [], []
    for number, group in enumerate(sensitivity_entries(runs)):
        if number:
            axis.axvline(
                position - 0.65, color=COLORS["background_light"], linewidth=LINE_WIDTH / 2
            )
        for label, run in group:
            for offset, key in zip((-0.27, 0.0, 0.27), STATISTICS):
                change = 100 * (run["stats"][key] / baseline["stats"][key] - 1)
                axis.bar(
                    position + offset,
                    change,
                    width=0.25,
                    color=STATISTIC_STYLES[key]["color"],
                    linewidth=0,
                    zorder=2,
                )
            ticks.append(position)
            labels.append(label)
            position += 1
        position += 0.3
    largest = 100 * max(floor.values())
    axis.axhspan(
        -largest, largest, color=COLORS["background_light"], alpha=0.5, linewidth=0, zorder=0
    )
    axis.axhline(0, color="black", linewidth=LINE_WIDTH / 2)
    axis.set_xticks(ticks, labels, rotation=90)
    axis.tick_params(axis="x", which="both", bottom=False, top=False)
    axis.set_xlim(-0.6, position - 0.7)
    axis.set_ylim(-7, 4)
    axis.set_ylabel(r"change from the baseline ($\%$)")
    handles = [
        Patch(color=STATISTIC_STYLES[key]["color"], label=STATISTIC_NAMES[key])
        for key in STATISTICS
    ]
    handles.append(Patch(color=COLORS["background_light"], alpha=0.5, label="resolution"))
    figure.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=True,
        edgecolor="0.8",
        bbox_to_anchor=(0.5, 0.995),
        handlelength=1.2,
    )
    save(figure, "error_sensitivity")


def main():
    floor = resolution()
    references, runs = collect()
    recommended = next(run for run in runs if run["run"] == BASELINE)
    write_summary(references, runs, floor)
    print(
        "Resolution of the 40-55 s statistics:",
        ", ".join(f"{key} {100 * floor[key]:.2f}%" for key in STATISTICS),
    )
    figure_forces(recommended)
    figure_profiles(
        recommended,
        "velocity_x",
        "velocity_profiles",
        (
            r"$\bar u/U_\infty$",
            r"$\Delta\bar u/U_\infty$ ($\%$)",
            r"$u'_\mathrm{rms}/U_\infty$",
            r"$\Delta u'_\mathrm{rms}/U_\infty$ ($\%$)",
        ),
        100,
        ([0.5, 1.0], [0.0, 0.2]),
    )
    figure_profiles(
        recommended,
        "vorticity_z",
        "vorticity_profiles",
        (
            r"$\bar\omega_z D/U_\infty$",
            r"$\Delta\bar\omega_z D/U_\infty$",
            r"$\omega'_\mathrm{rms}D/U_\infty$",
            r"$\Delta\omega'_\mathrm{rms}D/U_\infty$",
        ),
        1,
        ([-1.0, 0.0, 1.0], [0.0, 1.0, 2.0]),
    )
    figure_convergence(references, runs)
    figure_interval(runs, floor)
    figure_sensitivity(runs, floor)


if __name__ == "__main__":
    main()
