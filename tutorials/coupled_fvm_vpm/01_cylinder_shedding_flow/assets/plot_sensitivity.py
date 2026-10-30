#!/usr/bin/env python3
"""Plot force error, wake difference and cost for the cylinder sensitivity study.

Each coupled run is compared with a reference at matching Reynolds number,
cell size and FVM time step.
"""

import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np

from openonda.plotting import fit_thesis_y_label_margins
from openonda.plotting import BACKGROUND_STRONG, CM, COLORS, LINE_WIDTH, MARKER_SIZE
from openonda.plotting import MAX_FIGURE_WIDTH_CM, REFERENCE_STYLE
from openonda.plotting import centered_subplots_adjust, save_validation_figure, set_thesis_style
from openonda.plotting import validation_legend

from assets.plot_forces import DEVELOPED_SHEDDING, read_forces, shedding_statistics
from assets.shedding_phase import phase_averaged_difference, shedding_phase
from setup import BLEND_END, BLEND_START, BOX_MARGIN, CELL_SIZE, DIAMETER, FREESTREAM_VELOCITY
from setup import FVM_SUBSTEPS
from setup import FVM_TIME_STEP, MAX_STRENGTH_RATIO, REYNOLDS_NUMBER, TRANSFER_INSET

CASE_DIR = Path(__file__).resolve().parents[1]
ERROR_BAND = 2.0  # %
STATISTICS_LENGTH = 10.0  # s
# Statistics window of the study, the same in every run and reference. Runs continued beyond
# its end (section h) are compared over LATE_WINDOW separately, never over a longer window.
STATISTICS_WINDOW = (DEVELOPED_SHEDDING, 55.0)  # s
LATE_WINDOW = (55.0, 100.0)  # s
WALL_STRAIN_RATE = 19.4  # 1/s

# Baseline parameters and compared force statistics.
PARAMETERS = {
    "box_margin": ("Box margin [$D$]", BOX_MARGIN),
    "fvm_substeps": ("FVM substeps", FVM_SUBSTEPS),
    "transfer_inset": ("Inset [cells]", TRANSFER_INSET),
}
BASELINE = {
    **{name: value for name, (_, value) in PARAMETERS.items()},
    "box_downstream": BOX_MARGIN + 2.0,  # D
    "cell_size": CELL_SIZE,
    "fvm_time_step": FVM_TIME_STEP,
    "particle_spacing": CELL_SIZE,
    "interface_cell_size": CELL_SIZE,  # m
    "max_strength_ratio": MAX_STRENGTH_RATIO,
    "blend_start": float(BLEND_START),  # cells below the transfer surface
    "blend_end": float(BLEND_END),  # cells below the transfer surface
    "coupling_iterations": 0,  # tolerance-driven coupling iterations
    "repeat": 0,
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

# Run names encode the changed physical or numerical parameters.
RUN_NAME = re.compile(
    r"box(?P<box_margin>[^_]+)_sub(?P<fvm_substeps>\d+)_inset(?P<transfer_inset>[^_]+)"
    r"(?:_xd(?P<box_downstream>[^_]+))?"
    r"(?:_h(?P<cell_size>[\d.]+))?(?:_dtf(?P<fvm_time_step>[^_]+))?"
    r"(?:_ic(?P<interface_cell_size>[^_]+))?"
    r"(?:_hp(?P<particle_spacing>[^_]+))?"
    r"(?:_bs(?P<blend_start>[\d.]+))?(?:_be(?P<blend_end>[\d.]+))?"
    r"(?:_kappa(?P<max_strength_ratio>[\d.]+))?"
    r"(?:_k(?P<coupling_iterations>\d+))?(?:_re(?P<reynolds_number>[\d.]+))?"
    r"(?:_rep(?P<repeat>\d+))?"
)
REFERENCE_NAME = re.compile(
    r"reference(?:_h(?P<cell_size>[^_]+))?(?:_dt(?P<time_step>[^_]+))?"
    r"(?:_re(?P<reynolds_number>[\d.]+))?(?:_fw(?P<fine_wake_end>[\d.]+))?"
)


def jsonl_records(path):
    """Records from a JSON-lines file."""
    return [json.loads(line) for line in path.read_text().splitlines()]


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
        if match is None or match["fine_wake_end"]:
            continue  # the fine-wake reference is compared separately (read_fine_wake_reference)
        # The step follows the cells at constant Courant number unless it is given
        cell_size = float(match["cell_size"] or CELL_SIZE)
        time_step = float(match["time_step"] or cell_size / CELL_SIZE * FVM_TIME_STEP)
        reynolds_number = float(match["reynolds_number"] or REYNOLDS_NUMBER)
        references[reynolds_number, cell_size, time_step] = read_reference(
            directory, cell_size, time_step, reynolds_number
        )
    return references


def in_window(samples, start, end):
    """The rows of a time series with start <= time <= end."""
    time = samples["time"]
    return samples[(time >= start) & (time <= end)]


def errors_against(forces, phase, wake, reference, window=STATISTICS_WINDOW):
    """Signed relative force errors and the rms wake velocity difference, in %.

    Both solutions are measured over the same window, cut at the earlier of their end times.
    """
    start = window[0]
    end = min(window[1], forces["time"][-1], reference["forces"]["time"][-1])
    expected = shedding_statistics(in_window(reference["forces"], start, end))
    measured = shedding_statistics(in_window(forces, start, end))
    differences = [
        phase_averaged_difference(
            in_window(samples, start, end),
            phase,
            in_window(reference_samples, start, end),
            reference["phase"],
            start,
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


def coupling_step_statistics(path):
    """Wall time per flow time, coupling iterations, residuals and transfer limits of developed steps.

    The strength ratio kappa is the largest limited lattice strength over the peak target strength
    of a transfer; a transfer at its limit kappa_max (within 0.5%) has had its strength correction
    cut back near the wall.
    """
    records = jsonl_records(path)
    coupling_interval = records[1]["time"] - records[0]["time"]
    records = [
        record
        for record in records
        if STATISTICS_WINDOW[0] <= record["time"] <= STATISTICS_WINDOW[1]
    ]
    metadata = path.parent / "coupler_metadata.json"
    kappa_max = (
        json.loads(metadata.read_text())["configuration"].get("max_strength_ratio", np.nan)
        if metadata.exists()
        else np.nan
    )
    kappa = np.array(
        [record["transfer"]["strength_ratio"] for record in records if "transfer" in record]
    )
    mismatch = [
        record["fvm_boundary_data"]["mean_velocity_mismatch"]
        for record in records
        if "fvm_boundary_data" in record
    ]
    iterations = [
        record.get("coupling_iteration", {"iterations": 1, "residuals": []}) for record in records
    ]
    final = [entry["residuals"][-1] for entry in iterations if entry["residuals"]]
    return {
        "cost": np.mean([record["timing_seconds"]["total"] for record in records])
        / coupling_interval,
        "mean coupling iterations": np.mean([entry["iterations"] for entry in iterations]),
        "normal residual": np.mean([row["normal_velocity_residual"] for row in final])
        if final
        else np.nan,
        "gradient residual": np.mean([row["normal_gradient_residual"] for row in final])
        if final
        else np.nan,
        "strength ratio median": np.median(kappa) if kappa.size else np.nan,
        "strength ratio max": kappa.max() if kappa.size else np.nan,
        "strength-limit fraction": np.mean(kappa >= 0.995 * kappa_max) if kappa.size else np.nan,
        "boundary mismatch mean": np.mean(mismatch) / FREESTREAM_VELOCITY[0]
        if mismatch
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
        if not given["interface_cell_size"]:
            parameters["interface_cell_size"] = parameters["cell_size"]
        if not given["box_downstream"]:
            parameters["box_downstream"] = parameters["box_margin"] + 2.0  # setup.py default
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
                "lagrangian_cfl": WALL_STRAIN_RATE * particle_time_step,
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
                **coupling_step_statistics(directory / "solution/coupler_diagnostics.jsonl"),
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
    fit_thesis_y_label_margins(figure, figure.axes)
    save_validation_figure(figure, path)


def print_runs(runs, references):
    print(
        f"Errors [%] against the reference of equal Re and FVM step, t >= {DEVELOPED_SHEDDING:g} s"
    )
    print(
        f"  {'run':40s} {'end [s]':>8s} {'replacement [D]':>15s}"
        + "".join(f" {name:>16s}" for name in ROWS)
    )
    for run in runs:
        # Half-width of the replacement region, where FVM vorticity alone replaces the particle
        # vorticity: the blend end depth is measured from the transfer surface
        replacement = (
            run["box_margin"]
            - (run["transfer_inset"] + run["blend_end"]) * run["interface_cell_size"] / DIAMETER
        )
        values = "".join(f" {run[name]:16.2f}" for name in ROWS)
        print(f"  {run['name']:40s} {run['end time']:8.1f} {replacement:15.2f}{values}")
    for reference in references.values():
        print(
            f"  Reference {reference['name']} wall time per flow time: {reference['cost']:.1f} s/s"
        )
    print(
        "  A replacement region below 0.5 D lies inside the cylinder: the wall vorticity is blended."
    )
    print(
        f"  Lagrangian CFL number Lambda = max|S| dt_VPM with max|S| = {WALL_STRAIN_RATE} 1/s,"
        " measured at Re = 150 with h_VPM = h_FVM; the runs do not record their own strain rate."
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


def read_fine_wake_reference():
    """The FVM-only reference with h cells over the coupled FVM domain (section g), or None."""
    for directory in sorted(CASE_DIR.glob("sensitivity/reference_fw*")):
        match = REFERENCE_NAME.fullmatch(directory.name)
        path = directory / "samples/forces_history.csv"
        if match and path.exists() and developed(np.atleast_1d(read_forces(path))):
            return read_reference(directory, CELL_SIZE, FVM_TIME_STEP, REYNOLDS_NUMBER)
    return None


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
                # The FVM substeps N a 0.04 s particle step gives at this FVM step
                "fvm_substeps": 0.04 / time_step,
                "fvm_courant": FREESTREAM_VELOCITY[0] * time_step / CELL_SIZE,
            }
        )
    return changes


def parameter_figure(runs, reference_cost):
    """The domain-margin, FVM-substep and inset study about the baseline."""
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
    fit_thesis_y_label_margins(figure, figure.axes)
    save_validation_figure(figure, CASE_DIR / "figures/sensitivity")


def main():
    references = read_references()
    runs = read_runs(references)
    print_runs(runs, references)
    write_tables(runs, references)
    parameter_figure(runs, references[REYNOLDS_NUMBER, CELL_SIZE, FVM_TIME_STEP]["cost"])
    figures = CASE_DIR / "figures"
    hybrid, teal, gray = COLORS["hybrid"], COLORS["fvm"], COLORS["reference"]

    # S1b: particle step fixed at 0.04 s, so the FVM substeps N grow as the FVM step falls
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
            figures / "sensitivity_fvm_substeps",
            {
                "fvm_substeps": r"FVM substeps $N$, $\Delta t_{\rm VPM}=0.04$ s",
                "fvm_courant": r"$C_{\rm FVM}=U_\infty\Delta t_{\rm FVM}/h_{\rm FVM}$",
            },
            ROWS,
            [
                (
                    "FVM only, equal step",
                    [run for run in s1b if run["reference"] != "none"],
                    hybrid,
                    "o",
                ),
                ("FVM only, finest step", [{**run, **run["finest"]} for run in s1b], teal, "s"),
                ("FVM only vs finest step", changes, gray, "^"),
            ],
        )

    # S2: particle spacing h_FVM and h_FVM/2 at a fixed FVM step, against N and C_VPM
    s2 = varied(runs, "fvm_substeps", "particle_spacing")
    if any(run["particle_spacing"] != CELL_SIZE for run in s2):
        study_figure(
            figures / "sensitivity_particle_spacing",
            {
                "fvm_substeps": r"FVM substeps $N$",
                "particle_courant": r"$C_{\rm VPM}=U_\infty\Delta t_{\rm VPM}/h$",
            },
            ROWS,
            [
                (
                    f"$h={spacing:g}$ m",
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

    # S4: forced coupling iterations per interval, and the converged baseline at its mean count
    s4 = varied(runs, "coupling_iterations")
    if any(run["coupling_iterations"] for run in s4):
        iterations = [
            {**run, "iterations": run["coupling_iterations"] or run["mean coupling iterations"]}
            for run in s4
        ]
        study_figure(
            figures / "sensitivity_coupling_iterations",
            {"iterations": "Coupling iterations per interval $M$"},
            {
                **ERRORS,
                "gradient residual": "Last-iteration\n" + r"$r_{g_t}$ [1/s]",
                "cost": ROWS["cost"],
            },
            [
                (
                    "Exactly $M$",
                    [run for run in iterations if run["coupling_iterations"]],
                    hybrid,
                    "o",
                ),
                (
                    r"Tolerance $\varepsilon_{u_n}=\varepsilon_{g_t}=10^{-5}$",
                    [run for run in iterations if not run["coupling_iterations"]],
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
    if len(s1a) < 2:
        return
    study_figure(
        figures / "sensitivity_dimensionless",
        {
            "fvm_courant": r"$C_{\rm FVM}$",
            "particle_courant": r"$C_{\rm VPM}$",
            "lagrangian_cfl": r"$\Lambda=\max|S|\,\Delta t_{\rm VPM}$",
        },
        ERRORS,
        [
            (r"$\Delta t_{\rm FVM}=8$ ms", s1a, hybrid, "o"),
            (r"$\Delta t_{\rm VPM}=40$ ms", s1b, teal, "s"),
            (
                r"$h_{\rm VPM}=h_{\rm FVM}/2$",
                [run for run in s2 if run["particle_spacing"] != CELL_SIZE],
                COLORS["vpm"],
                "^",
            ),
        ],
        lines=False,  # S1a holds C_FVM and S1b holds C_VPM and Lambda fixed
    )


if __name__ == "__main__":
    main()
