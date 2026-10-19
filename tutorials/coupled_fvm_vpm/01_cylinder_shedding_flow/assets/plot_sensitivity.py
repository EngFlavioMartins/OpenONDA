#!/usr/bin/env python3
"""Result quality and cost of the coupling sensitivity study against the FVM reference.

Force errors, the wake velocity difference at equal shedding phase and the wall time per second
of flow time, as each coupling parameter is varied about its baseline. Each run is compared with
the reference of its own Reynolds number, cells and FVM step. The convergence runs refine the
cells, the steps and the particles together; their observed orders and the split of the coupled
error into the FVM discretization and the coupling are measured against the reference
extrapolated to zero cell size. The studies of the coupling-error plan
(docs/plans/2026-10-19-coupling-error-sources.md) are drawn in their own figures, and the
exchange-ratio and particle-resolution runs again against the dimensionless groups
C_f = U dt_f / h_f, C_p = U dt_p / h_p and Lambda = max|S| dt_p.
"""

import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter
import numpy as np

from openonda.plotting import BACKGROUND_STRONG, CM, COLORS, LINE_WIDTH, MARKER_SIZE
from openonda.plotting import MAX_FIGURE_WIDTH_CM, REFERENCE_STYLE
from openonda.plotting import centered_subplots_adjust, save_validation_figure, set_thesis_style
from openonda.plotting import validation_legend

from assets.plot_forces import DEVELOPED_SHEDDING, read_forces, shedding_statistics
from assets.shedding_phase import phase_averaged_difference, shedding_phase
from setup import BLEND_WIDTH, BOX_MARGIN, CELL_SIZE, DIAMETER, FREESTREAM_VELOCITY, FVM_SUBSTEPS
from setup import FVM_TIME_STEP, REYNOLDS_NUMBER, TRANSFER_INSET, VPM_ONLY_WIDTH

CASE_DIR = Path(__file__).resolve().parents[1]
ERROR_BAND = 2.0  # %, accuracy targeted for the coupled forces
STATISTICS_LENGTH = 10.0  # s, about two shedding periods of developed shedding at least
WALL_STRAIN_RATE = 19.4  # 1/s, max|S| at the wall, Re = 150, h_p = h_f (8-step stability check)
CONVERGENCE_SUBSTEPS = 2  # FVM steps per exchange of the convergence runs

# Study parameters with their baselines, and the force statistics compared
PARAMETERS = {
    "box_margin": ("Box margin [$D$]", BOX_MARGIN),
    "fvm_substeps": ("FVM substeps", FVM_SUBSTEPS),
    "transfer_inset": ("Inset [cells]", TRANSFER_INSET),
}
BASELINE = {
    **{name: value for name, (_, value) in PARAMETERS.items()},
    "cell_size": CELL_SIZE,
    "fvm_time_step": FVM_TIME_STEP,
    "particle_spacing": CELL_SIZE,
    "interface_sweeps": 0,  # 0: sweeps until the boundary trace converges
    "reynolds_number": REYNOLDS_NUMBER,
}
STATISTICS = {
    "mean drag": r"$\bar C_D$ error [\%]",
    "lift amplitude": r"$\hat C_L$ error [\%]",
    "Strouhal number": r"$St$ error [\%]",
}
WAKE_STATIONS = (2, 4)  # x/D of the transverse velocity probes
ERRORS = {**STATISTICS, "wake velocity": r"$\|\Delta\mathbf{u}\|/U_\infty$ [\%]"}
ROWS = {**ERRORS, "cost": "Wall time per\nflow time [s/s]"}

# Run names: box{margin}_sub{n}_inset{cells}, with a suffix for each other parameter changed
RUN_NAME = re.compile(
    r"box(?P<box_margin>[^_]+)_sub(?P<fvm_substeps>\d+)_inset(?P<transfer_inset>[^_]+)"
    r"(?:_h(?P<cell_size>[\d.]+))?(?:_dtf(?P<fvm_time_step>[^_]+))?"
    r"(?:_hp(?P<particle_spacing>[^_]+))?"
    r"(?:_k(?P<interface_sweeps>\d+))?(?:_re(?P<reynolds_number>[^_]+))?"
)
REFERENCE_NAME = re.compile(
    r"reference(?:_h(?P<cell_size>[^_]+))?(?:_dt(?P<time_step>[^_]+))?"
    r"(?:_re(?P<reynolds_number>[^_]+))?"
)


def jsonl_records(path):
    """Records of a JSON-lines file; a running case may still be writing the last line."""
    return [json.loads(line) for line in path.read_text().split("\n")[:-1]]


def developed(forces):
    """Whether a force history covers the statistics window."""
    return forces["time"][-1] >= DEVELOPED_SHEDDING + STATISTICS_LENGTH


def read_reference(directory, cell_size, time_step, reynolds_number):
    """Forces, shedding phase, wake probes and cost of one FVM reference."""
    steps = jsonl_records(directory / "solution/performance.jsonl")
    return {
        "name": directory.name,
        "cell_size": cell_size,
        "time_step": time_step,
        "reynolds_number": reynolds_number,
        "forces": read_forces(directory / "samples/forces_history.csv"),
        "phase": shedding_phase(directory / "samples/forces_history.csv"),
        "wake": [
            np.genfromtxt(directory / f"samples/transverse_x{x}.csv", delimiter=",", names=True)
            for x in WAKE_STATIONS
        ],
        "cost": np.mean([step["step_seconds"]["max"] for step in steps]) / time_step,
    }


def read_references():
    """Every developed reference, keyed by Reynolds number, cell size and FVM step."""
    references = {}
    for path in [
        CASE_DIR / "reference_flow/samples/forces_history.csv",
        *sorted(CASE_DIR.glob("sensitivity/reference_*/samples/forces_history.csv")),
    ]:
        directory = path.parents[1]
        if not developed(np.atleast_1d(read_forces(path))):
            continue  # still developing
        match = REFERENCE_NAME.fullmatch(directory.name.replace("reference_flow", "reference"))
        # The step follows the cells at constant Courant number unless it is given
        cell_size = float(match["cell_size"] or CELL_SIZE)
        time_step = float(match["time_step"] or cell_size / CELL_SIZE * FVM_TIME_STEP)
        reynolds_number = float(match["reynolds_number"] or REYNOLDS_NUMBER)
        references[reynolds_number, cell_size, time_step] = read_reference(
            directory, cell_size, time_step, reynolds_number
        )
    return references


def errors_against(forces, phase, wake, reference):
    """Signed relative force errors and the rms wake velocity difference, in %."""
    end_time = min(forces["time"][-1], reference["forces"]["time"][-1])
    expected = shedding_statistics(reference["forces"][reference["forces"]["time"] <= end_time])
    measured = shedding_statistics(forces[forces["time"] <= end_time])
    differences = [
        phase_averaged_difference(
            samples, phase, reference_samples, reference["phase"], DEVELOPED_SHEDDING
        )[1]
        for samples, reference_samples in zip(wake, reference["wake"])
    ]
    return {
        **{name: 100.0 * (measured[name] / expected[name] - 1.0) for name in STATISTICS},
        "wake velocity": 100.0
        * np.sqrt(np.mean(np.concatenate([d.ravel() for d in differences]) ** 2)),
    }


def no_errors():
    return dict.fromkeys(ERRORS, np.nan)


def exchange_statistics(path):
    """Wall time per flow time and the interface sweeps and residuals of developed exchanges."""
    records = jsonl_records(path)
    exchange_step = records[1]["time"] - records[0]["time"]
    records = [record for record in records if record["time"] >= DEVELOPED_SHEDDING]
    sweeps = [
        record.get("interface_iteration", {"sweeps": 1, "residuals": []}) for record in records
    ]
    final = [entry["residuals"][-1] for entry in sweeps if entry["residuals"]]
    return {
        "cost": np.mean([record["timing_seconds"]["total"] for record in records]) / exchange_step,
        "mean sweeps": np.mean([entry["sweeps"] for entry in sweeps]),
        "normal residual": np.mean([row["normal_residual_rms"] for row in final])
        if final
        else np.nan,
        "gradient residual": np.mean([row["gradient_residual_rms"] for row in final])
        if final
        else np.nan,
    }


def read_runs(references):
    """Parameters, errors and cost of every developed run in sensitivity/."""
    # The finest-step reference on the baseline cells, for each Reynolds number
    finest = {}
    for reynolds_number, cell_size, time_step in sorted(references, reverse=True):
        if cell_size == CELL_SIZE:
            finest[reynolds_number] = references[reynolds_number, cell_size, time_step]
    runs = []
    for path in sorted(CASE_DIR.glob("sensitivity/box*/samples/forces_history.csv")):
        directory = path.parents[1]
        forces = np.atleast_1d(read_forces(path))
        if not developed(forces):
            continue  # still developing, or stopped early

        given = RUN_NAME.fullmatch(directory.name).groupdict()
        parameters = {
            name: type(BASELINE[name])(value) if value else BASELINE[name]
            for name, value in given.items()
        }
        # Refined cells refine the step at constant Courant number, and the particles with them
        refinement = parameters["cell_size"] / CELL_SIZE
        if not given["fvm_time_step"]:
            parameters["fvm_time_step"] = refinement * FVM_TIME_STEP
        if not given["particle_spacing"]:
            parameters["particle_spacing"] = parameters["cell_size"]
        particle_time_step = parameters["fvm_substeps"] * parameters["fvm_time_step"]
        reference = references.get(
            (parameters["reynolds_number"], parameters["cell_size"], parameters["fvm_time_step"])
        )
        finest_reference = finest.get(parameters["reynolds_number"])
        phase = shedding_phase(path)
        wake = [
            np.genfromtxt(directory / f"samples/vpm_transverse_x{x}.csv", delimiter=",", names=True)
            for x in WAKE_STATIONS
        ]
        runs.append(
            {
                "name": directory.name,
                **parameters,
                "particle_time_step": particle_time_step,
                "fvm_courant": FREESTREAM_VELOCITY[0]
                * parameters["fvm_time_step"]
                / parameters["cell_size"],
                "particle_courant": FREESTREAM_VELOCITY[0]
                * particle_time_step
                / parameters["particle_spacing"],
                "stability_number": WALL_STRAIN_RATE * particle_time_step,
                "end time": forces["time"][-1],
                "forces": forces,
                "phase": phase,
                "wake": wake,
                "reference": "none" if reference is None else reference["name"],
                **(
                    no_errors()
                    if reference is None
                    else errors_against(forces, phase, wake, reference)
                ),
                "finest reference": "none"
                if finest_reference is None
                else finest_reference["name"],
                "finest": no_errors()
                if finest_reference is None
                else errors_against(forces, phase, wake, finest_reference),
                **exchange_statistics(directory / "solution/coupler_diagnostics.jsonl"),
            }
        )
    return runs


def varied(runs, *names):
    """The runs that keep every parameter except ``names`` at its baseline."""
    return [
        run
        for run in runs
        if all(
            np.isclose(run[name], value) for name, value in BASELINE.items() if name not in names
        )
    ]


def study_figure(path, columns, rows, series, log_rows=(), lines=True):
    """One column per x quantity and one row per measured quantity, for each series of runs."""
    set_thesis_style()
    figure, axes = plt.subplots(
        len(rows),
        len(columns),
        squeeze=False,
        sharex="col",
        sharey="row",
        figsize=(MAX_FIGURE_WIDTH_CM * CM, (2.0 + 3.0 * len(rows)) * CM),
    )
    centered_subplots_adjust(
        figure,
        outer=0.14 if len(columns) > 1 else 0.25,
        bottom=1.3 / (2.0 + 3.0 * len(rows)),
        top=1.0 - 0.5 / (2.0 + 3.0 * len(rows)),
        wspace=0.15,
        hspace=0.12,
    )
    for column, (x, x_label) in enumerate(columns.items()):
        for row, (quantity, row_label) in enumerate(rows.items()):
            axis = axes[row, column]
            if quantity in STATISTICS:
                axis.axhline(ERROR_BAND, **REFERENCE_STYLE)  # accuracy target of the forces
            for label, members, color, marker in series:
                members = sorted(members, key=lambda run: run[x])
                axis.plot(
                    [run[x] for run in members],
                    [abs(run[quantity]) for run in members],
                    color=color,
                    linewidth=LINE_WIDTH if lines else 0.0,
                    marker=marker,
                    markersize=MARKER_SIZE,
                    label=label,
                )
            axis.set_yscale("log" if quantity in log_rows else "linear")
            if column == 0:
                axis.set_ylabel(row_label)
        axes[-1, column].set_xlabel(x_label)

        # Every plotted value, with a margin also when the runs share one value
        values = [run[x] for _, members, _, _ in series for run in members]
        low, high = min(values), max(values)
        margin = 0.08 * (high - low) if high > low else 0.25 * abs(high)
        axes[-1, column].set_xlim(low - margin, high + margin)
        axes[-1, column].xaxis.set_major_locator(MaxNLocator(4, prune="lower"))  # clear of y

    # Each linear row spans zero to its largest value and always shows the 2% target
    for row, quantity in enumerate(rows):
        values = [abs(run[quantity]) for _, members, _, _ in series for run in members]
        largest = np.nanmax([ERROR_BAND if quantity in STATISTICS else 0.0, *values])
        if quantity not in log_rows and largest > 0.0:
            axes[row, 0].set_ylim(0.0, 1.15 * largest)
            axes[row, 0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    validation_legend(figure, axes[0, -1], ncol=1)
    save_validation_figure(figure, path)


def print_runs(runs, references):
    print(
        f"Errors [%] against the reference of equal Re and FVM step, t >= {DEVELOPED_SHEDDING:g} s"
    )
    print(
        f"  {'run':40s} {'end [s]':>8s} {'FVM core [D]':>13s}"
        + "".join(f" {name:>16s}" for name in ROWS)
    )
    for run in runs:
        # Half-width of the region where FVM vorticity alone replaces the particles
        core = (
            run["box_margin"]
            - (run["transfer_inset"] + BLEND_WIDTH + VPM_ONLY_WIDTH) * CELL_SIZE / DIAMETER
        )
        values = "".join(f" {run[name]:16.2f}" for name in ROWS)
        print(f"  {run['name']:40s} {run['end time']:8.1f} {core:13.2f}{values}")
    for reference in references.values():
        print(
            f"  Reference {reference['name']} wall time per flow time: {reference['cost']:.1f} s/s"
        )
    print("  An FVM core below 0.5 D lies inside the cylinder: the wall vorticity is blended.")
    print(
        f"  Lambda = max|S| dt_p with max|S| = {WALL_STRAIN_RATE} 1/s, measured at Re = 150 with"
        " h_p = h_f; the runs do not record their own strain rate."
    )


def write_tables(runs, references):
    """Every run and every reference against the finest one, in figures/auxiliary/."""
    names = (
        [name for name in runs[0] if name not in ("finest", "forces", "phase", "wake")]
        if runs
        else []
    )
    lines = [",".join(names + [f"{name} vs finest" for name in ERRORS])]
    lines += [
        ",".join(
            f"{run[name]:.6g}" if isinstance(run[name], float) else f"{run[name]}" for name in names
        )
        + "".join(f",{run['finest'][name]:.6g}" for name in ERRORS)
        for run in runs
    ]
    (CASE_DIR / "figures/auxiliary").mkdir(parents=True, exist_ok=True)
    (CASE_DIR / "figures/auxiliary/sensitivity_runs.csv").write_text("\n".join(lines) + "\n")

    lines = [
        "reference,reynolds_number,cell_size,time_step,cost,"
        + ",".join(f"{name} vs finest" for name in ERRORS)
    ]
    for run in reference_changes(references):
        lines.append(
            f"{run['name']},{run['reynolds_number']},{run['cell_size']},{run['time_step']},"
            f"{run['cost']:.6g}," + ",".join(f"{run[name]:.6g}" for name in ERRORS)
        )
    (CASE_DIR / "figures/auxiliary/sensitivity_references.csv").write_text("\n".join(lines) + "\n")


def reference_changes(references):
    """Each reference on the baseline cells against the finest-step one of its Reynolds number."""
    changes = []
    for (reynolds_number, cell_size, time_step), reference in sorted(references.items()):
        if cell_size != CELL_SIZE:
            continue
        finest = references[
            reynolds_number,
            cell_size,
            min(step for re, h, step in references if (re, h) == (reynolds_number, cell_size)),
        ]
        errors = errors_against(reference["forces"], reference["phase"], reference["wake"], finest)
        changes.append(
            {
                **reference,
                **errors,
                # The exchange ratio a 0.04 s particle step gives at this FVM step
                "fvm_substeps": 0.04 / time_step,
                "fvm_courant": FREESTREAM_VELOCITY[0] * time_step / CELL_SIZE,
            }
        )
    return changes


def parameter_figure(runs, reference_cost):
    """The box, exchange-step and inset study about the baseline."""
    # One column per parameter, varied while the other two keep their baselines
    set_thesis_style()
    figure, axes = plt.subplots(
        len(ROWS),
        len(PARAMETERS),
        sharex="col",
        sharey="row",
        figsize=(MAX_FIGURE_WIDTH_CM * CM, 17.0 * CM),
    )
    centered_subplots_adjust(figure, outer=0.14, bottom=0.08, top=0.98, wspace=0.15, hspace=0.12)
    for column, (parameter, (label, baseline)) in enumerate(PARAMETERS.items()):
        members = sorted(varied(runs, parameter), key=lambda run: run[parameter])
        for row, (quantity, row_label) in enumerate(ROWS.items()):
            axis = axes[row, column]
            if quantity in STATISTICS:
                axis.axhline(ERROR_BAND, **REFERENCE_STYLE)  # accuracy target of the forces
            if quantity == "cost":
                axis.axhline(reference_cost, **REFERENCE_STYLE)  # the reference FVM
            axis.axvline(baseline, color=BACKGROUND_STRONG, linewidth=LINE_WIDTH, linestyle=":")
            axis.plot(
                [run[parameter] for run in members],
                [abs(run[quantity]) for run in members],
                color=COLORS["hybrid"],
                linewidth=LINE_WIDTH,
                marker="o",
                markersize=MARKER_SIZE,
            )
            if column == 0:
                axis.set_ylabel(row_label)
        axes[-1, column].set_xlabel(label)
        # Every studied value and the baseline, also before any run has finished
        values = [baseline, *(run[parameter] for run in members)]
        low, high = min(values), max(values)
        margin = 0.08 * (high - low) if high > low else 0.5 * baseline
        axes[-1, column].set_xlim(low - margin, high + margin)

    axes[-1, 1].xaxis.set_major_locator(MaxNLocator(integer=True))  # whole FVM steps
    axes[-1, 0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))  # no offset

    # Each row spans zero to its largest value and always shows its reference line
    plotted = [run for parameter in PARAMETERS for run in varied(runs, parameter)]
    for row, quantity in enumerate(ROWS):
        floor = reference_cost if quantity == "cost" else ERROR_BAND
        largest = max([floor, *(abs(run[quantity]) for run in plotted)])
        axes[row, 0].set_ylim(0.0, 1.15 * largest)
    save_validation_figure(figure, CASE_DIR / "figures/sensitivity")


def richardson(cell_sizes, values):
    """Observed order and zero-cell-size value from the three finest levels of a constant ratio."""
    (h1, h2, _), (f1, f2, f3) = cell_sizes[:3], values[:3]
    ratio = h2 / h1
    order = np.log(abs((f3 - f2) / (f2 - f1))) / np.log(ratio)
    return order, f1 + (f1 - f2) / (ratio**order - 1.0)


def observed_order(cell_sizes, errors):
    """Least-squares slope of log error against log cell size."""
    return np.polyfit(np.log(cell_sizes), np.log(np.abs(errors)), 1)[0]


def convergence(runs, references):
    """Observed orders of the reference and the coupled solution, and the split of the coupled error.

    Cells, steps and particles are refined together at constant Courant number. The error-free
    statistics are the reference ones extrapolated to zero cell size from its three finest
    levels; the wake velocity, which has no scalar to extrapolate, is measured against the
    finest reference. The coupled error then splits into the FVM discretization error of the
    reference on the same cells and the coupling error, the difference between the two runs.
    """
    levels = sorted(
        (reference for (re, h, dt), reference in references.items() if re == REYNOLDS_NUMBER),
        key=lambda reference: reference["cell_size"],
    )
    levels = [
        reference
        for reference in levels
        if np.isclose(reference["time_step"], reference["cell_size"] / CELL_SIZE * FVM_TIME_STEP)
    ]
    coupled = {
        run["cell_size"]: run
        for run in varied(runs, "cell_size", "fvm_time_step", "fvm_substeps", "particle_spacing")
        if run["fvm_substeps"] == CONVERGENCE_SUBSTEPS
        and np.isclose(run["fvm_time_step"], run["cell_size"] / CELL_SIZE * FVM_TIME_STEP)
        and np.isclose(run["particle_spacing"], run["cell_size"])
    }
    levels = [reference for reference in levels if reference["cell_size"] in coupled]
    if len(levels) < 3:
        return  # the extrapolation needs three levels with both runs
    cell_sizes = np.array([reference["cell_size"] for reference in levels])
    finest = levels[0]

    # Error-free statistics from the reference, and each run's statistics on its cells
    reference_statistics = [shedding_statistics(reference["forces"]) for reference in levels]
    coupled_statistics = [shedding_statistics(coupled[h]["forces"]) for h in cell_sizes]
    rows = []
    print(f"Convergence at constant Courant number, {CONVERGENCE_SUBSTEPS} FVM steps per exchange")
    for name in STATISTICS:
        reference_values = [statistics[name] for statistics in reference_statistics]
        order, exact = richardson(cell_sizes, reference_values)
        coupled_values = np.array([statistics[name] for statistics in coupled_statistics])
        reference_values = np.array(reference_values)
        rows.append(
            (
                name,
                100.0 * (reference_values - exact) / exact,
                100.0 * (coupled_values - exact) / exact,
                100.0 * (coupled_values - reference_values) / exact,
            )
        )
        print(f"  {name}: reference order {order:.2f}, zero-cell-size value {exact:.4f}")

    # Wake velocity against the finest reference, at equal shedding phase
    reference_wake = [
        errors_against(reference["forces"], reference["phase"], reference["wake"], finest)
        for reference in levels
    ]
    coupled_wake = [
        errors_against(coupled[h]["forces"], coupled[h]["phase"], coupled[h]["wake"], finest)
        for h in cell_sizes
    ]
    same_cells_wake = [
        errors_against(coupled[h]["forces"], coupled[h]["phase"], coupled[h]["wake"], reference)
        for h, reference in zip(cell_sizes, levels)
    ]
    rows.append(
        (
            "wake velocity",
            np.array([errors["wake velocity"] for errors in reference_wake]),
            np.array([errors["wake velocity"] for errors in coupled_wake]),
            np.array([errors["wake velocity"] for errors in same_cells_wake]),
        )
    )

    # Error against cell size: total coupled, FVM discretization and coupling
    set_thesis_style()
    figure, axes = plt.subplots(
        len(rows) + 1, 1, sharex=True, figsize=(MAX_FIGURE_WIDTH_CM * CM, 22.0 * CM)
    )
    centered_subplots_adjust(figure, outer=0.14, bottom=0.12, top=0.99, hspace=0.12)
    series = (
        ("Coupled", COLORS["hybrid"], "o"),
        ("Reference FVM", COLORS["reference"], "s"),
        ("Coupled $-$ reference", COLORS["vpm"], "^"),
    )
    table = ["quantity,series,observed_order," + ",".join(f"h={h:g}" for h in cell_sizes)]
    for axis, (name, *errors) in zip(axes, rows):
        reference_errors, coupled_errors, coupling_errors = errors
        for (label, color, marker), values in zip(
            series, (coupled_errors, reference_errors, coupling_errors)
        ):
            shown = np.abs(values) > 0.0  # the finest reference has no wake error against itself
            order = observed_order(cell_sizes[shown], values[shown])
            axis.loglog(
                cell_sizes[shown],
                np.abs(values[shown]),
                color=color,
                marker=marker,
                markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH,
                label=label,
            )
            table.append(
                f"{name},{label.replace('$-$', '-')},{order:.4g},"
                + ",".join(f"{value:.6g}" for value in values)
            )
            print(f"  {name}, {label.replace('$-$', '-')}: observed order {order:.2f}")
        # First- and second-order slopes through the coarsest coupled error
        anchor = np.abs(coupled_errors[-1])
        for order, style in ((1, ":"), (2, "--")):
            axis.loglog(
                cell_sizes,
                anchor * (cell_sizes / cell_sizes[-1]) ** order,
                color=BACKGROUND_STRONG,
                linewidth=LINE_WIDTH,
                linestyle=style,
                label=f"Order {order}",
            )
        axis.set_ylabel(ERRORS[name])

    # Wall time per flow time of both methods
    axes[-1].loglog(
        cell_sizes,
        [coupled[h]["cost"] for h in cell_sizes],
        color=COLORS["hybrid"],
        marker="o",
        markersize=MARKER_SIZE,
        linewidth=LINE_WIDTH,
    )
    axes[-1].loglog(
        cell_sizes,
        [reference["cost"] for reference in levels],
        color=COLORS["reference"],
        marker="s",
        markersize=MARKER_SIZE,
        linewidth=LINE_WIDTH,
    )
    axes[-1].set_ylabel(ROWS["cost"])
    axes[-1].set_xlabel("Cell size near the wall $h_f=h_p$ [m]")

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False, handlelength=1.5)

    # Decades only on the error axes, and the refinement levels on the cell-size axis
    for axis in axes:
        axis.yaxis.set_minor_formatter(NullFormatter())
    axes[-1].xaxis.set_minor_formatter(NullFormatter())
    axes[-1].set_xticks(cell_sizes, [f"{h:.3g}" for h in cell_sizes])
    save_validation_figure(figure, CASE_DIR / "figures/sensitivity_convergence")
    (CASE_DIR / "figures/auxiliary/convergence.csv").write_text("\n".join(table) + "\n")


def main():
    references = read_references()
    runs = read_runs(references)
    print_runs(runs, references)
    write_tables(runs, references)
    parameter_figure(runs, references[REYNOLDS_NUMBER, CELL_SIZE, FVM_TIME_STEP]["cost"])
    convergence(runs, references)
    figures = CASE_DIR / "figures"
    hybrid, teal, gray = COLORS["hybrid"], COLORS["fvm"], COLORS["reference"]

    # S1b: particle step fixed at 0.04 s, so the exchange ratio grows as the FVM step falls
    s1b = [
        run
        for run in varied(runs, "fvm_substeps", "fvm_time_step")
        if np.isclose(run["particle_time_step"], 0.04)
    ]
    if len(s1b) > 1:
        changes = [
            change
            for change in reference_changes(references)
            if change["reynolds_number"] == REYNOLDS_NUMBER
        ]
        study_figure(
            figures / "sensitivity_exchange_ratio",
            {
                "fvm_substeps": r"Exchange ratio $n$, $\Delta t_p=0.04$ s",
                "fvm_courant": r"$C_f=U_\infty\Delta t_f/h_f$",
            },
            ROWS,
            [
                (
                    "Equal-step reference",
                    [run for run in s1b if run["reference"] != "none"],
                    hybrid,
                    "o",
                ),
                ("Finest reference", [{**run, **run["finest"]} for run in s1b], teal, "s"),
                ("Reference vs finest", changes, gray, "^"),
            ],
        )

    # S2: particle spacing h_f and h_f/2 at a fixed FVM step, against n and C_p
    s2 = varied(runs, "fvm_substeps", "particle_spacing")
    if any(run["particle_spacing"] != CELL_SIZE for run in s2):
        study_figure(
            figures / "sensitivity_particle_spacing",
            {
                "fvm_substeps": r"Exchange ratio $n$",
                "particle_courant": r"$C_p=U_\infty\Delta t_p/h_p$",
            },
            ROWS,
            [
                (
                    f"$h_p={spacing:g}$ m",
                    [run for run in s2 if run["particle_spacing"] == spacing],
                    color,
                    marker,
                )
                for spacing, color, marker in (
                    (CELL_SIZE, hybrid, "o"),
                    (0.5 * CELL_SIZE, teal, "s"),
                )
            ],
        )

    # S4: forced interface sweeps per exchange, and the converged baseline at its mean sweeps
    s4 = varied(runs, "interface_sweeps")
    if any(run["interface_sweeps"] for run in s4):
        sweeps = [{**run, "sweeps": run["interface_sweeps"] or run["mean sweeps"]} for run in s4]
        study_figure(
            figures / "sensitivity_interface_sweeps",
            {"sweeps": "Interface sweeps per exchange $K$"},
            {
                **ERRORS,
                "gradient residual": "Last-sweep\n" + r"$\partial u_t/\partial n$ change [1/s]",
                "cost": ROWS["cost"],
            },
            [
                ("Exactly $K$", [run for run in sweeps if run["interface_sweeps"]], hybrid, "o"),
                (
                    "Tolerance $10^{-5}$",
                    [run for run in sweeps if not run["interface_sweeps"]],
                    teal,
                    "s",
                ),
            ],
            log_rows=("gradient residual",),
        )

    # S3: the baseline coupling at other Reynolds numbers, each against its own reference
    s3 = varied(runs, "reynolds_number")
    if len(s3) > 1:
        study_figure(
            figures / "sensitivity_reynolds_number",
            {"reynolds_number": "$Re$"},
            ERRORS,
            [("Baseline coupling", s3, hybrid, "o")],
        )

    # S1a, S1b and S2 against the dimensionless groups of the time steps and spacings
    s1a = varied(runs, "fvm_substeps")
    if not s1a:
        return
    study_figure(
        figures / "sensitivity_dimensionless",
        {
            "fvm_courant": r"$C_f$",
            "particle_courant": r"$C_p$",
            "stability_number": r"$\Lambda=\max|S|\,\Delta t_p$",
        },
        ERRORS,
        [
            (r"S1a, $\Delta t_f=0.008$ s", s1a, hybrid, "o"),
            (r"S1b, $\Delta t_p=0.04$ s", s1b, teal, "s"),
            (
                r"S2, $h_p=h_f/2$",
                [run for run in s2 if run["particle_spacing"] != CELL_SIZE],
                COLORS["vpm"],
                "^",
            ),
        ],
        lines=False,  # S1a holds C_f and S1b holds C_p and Lambda fixed
    )


if __name__ == "__main__":
    main()
