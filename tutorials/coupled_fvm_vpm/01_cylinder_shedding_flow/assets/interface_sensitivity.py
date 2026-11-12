#!/usr/bin/env python3
"""Error sources and one-at-a-time sensitivity of the coupling interface.

Every coupled run in solution/sensitivity_analysis/box* (one solver version) is split into

    e_total = e_FVM + e_coupling,
    e_FVM      = Q_FVM / Q_ext - 1,             FVM only on the same cells and step,
    e_coupling = (Q_coupled - Q_FVM) / Q_ext,   what the particle wake and the interface add,

with Q_ext the FVM-only value at zero cell size and step (Eca-Hoekstra least squares over the
references at a constant Courant number, assets/error_analysis.py). Q_FVM is the FVM-only
reference on the run's cells and FVM step (reference_flow/samples/<name>,
assets/plot_sensitivity.reference_key).

Statistics are those of assets/plot_forces.py over 40 <= t <= 55 s: the mean drag over whole
shedding periods, the lift amplitude as half the peak-to-peak value (its per-period mean as a
check), and the Strouhal number from the mean upward-crossing interval.

The sampling floor of a difference between two runs over the same window comes from the runs
that reach 100 s (section h and the 100 s reference): the spread of the statistic between
sliding windows of the developed flow (sigma_w, in each run) and the spread of the transient
drift of the 40-55 s window between those runs (s_d). For a difference of two runs
sigma = sqrt(2 sigma_w^2 + s_d^2). An effect counts as resolved when it exceeds 2 sigma.

Run from the case directory: python -m assets.interface_sensitivity
Writes figures/auxiliary/interface_*.csv and figures/interface_*.pdf/png.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np

from openonda.plotting import CM, COLORS, LINE_WIDTH, MARKER_SIZE, MAX_FIGURE_WIDTH_CM
from openonda.plotting import BACKGROUND_LIGHT, REFERENCE_STYLE
from openonda.plotting import centered_subplots_adjust, fit_thesis_y_label_margins
from openonda.plotting import save_validation_figure, set_thesis_style

from assets.error_analysis import MINIMUM_LEVELS, discretization_uncertainty, space_time_fit
from assets.plot_forces import read_forces, shedding_statistics
from assets.plot_sensitivity import BASELINE, REFERENCE_NAME, RUN_NAME, jsonl_records
from assets.plot_sensitivity import REFERENCE_DIR, reference_key, reference_paths, run_paths
from setup import CELL_SIZE, DIAMETER, FREESTREAM_VELOCITY, FVM_SUBSTEPS, FVM_TIME_STEP
from setup import REYNOLDS_NUMBER

CASE_DIR = Path(__file__).resolve().parents[1]
FIGURES = CASE_DIR / "figures"
TABLES = FIGURES / "auxiliary"
SPEED = FREESTREAM_VELOCITY[0]
WINDOW = (40.0, 55.0)  # s, the statistics window of every run and reference
SLIDE = 2.5  # s, offset of the sliding windows of the floor
FLOOR_RUN_END = 95.0  # s, runs reaching this time give the sampling floor
RESOLVED = 2.0  # an effect counts when it exceeds this many floor sigmas

STATISTICS = {
    "mean drag": r"$\bar C_D$",
    "lift amplitude": r"$\hat C_L$",
    "Strouhal number": r"$St$",
}
CHECKS = ("lift amplitude per period", "mean drag, whole window")


# ---------------------------------------------------------------------------------------------
# Statistics and their sampling floor
# ---------------------------------------------------------------------------------------------


def sampling_floor(paths):
    """Floor sigma [%] of the difference of two runs, per statistic, and its parts.

    sigma_w: the largest sample deviation between sliding windows of the developed flow (the
    second half of each run's windows); s_d: the sample deviation of the drift of the study window
    from the developed mean between the runs. sigma = sqrt(2 sigma_w^2 + s_d^2).
    """
    names = (*STATISTICS, *CHECKS)
    rows = []
    for path in paths:
        forces = np.atleast_1d(read_forces(path))
        if forces["time"][-1] < FLOOR_RUN_END:
            continue
        starts = np.arange(WINDOW[0], forces["time"][-1] - (WINDOW[1] - WINDOW[0]) + 1e-9, SLIDE)
        windows = [shedding_statistics(forces, s, s + WINDOW[1] - WINDOW[0]) for s in starts]
        windows = [w for w in windows if w is not None]
        study = shedding_statistics(forces, *WINDOW)
        # reference_flow/samples/<name>/ or solution/sensitivity_analysis/<name>/samples/
        name = path.parents[1].name if path.parent.name == "samples" else path.parent.name
        row = {"run": name, "end time": forces["time"][-1], "windows": len(windows)}
        for name in names:
            late = np.array([w[name] for w in windows[len(windows) // 2 :]])
            row[f"{name} sigma_w [%]"] = 100.0 * late.std(ddof=1) / late.mean()
            row[f"{name} drift [%]"] = 100.0 * (study[name] / late.mean() - 1.0)
        rows.append(row)
    floor = {}
    for name in names:
        sigma_w = max((row[f"{name} sigma_w [%]"] for row in rows), default=np.nan)
        drifts = [row[f"{name} drift [%]"] for row in rows]
        s_d = np.std(drifts, ddof=1) if len(drifts) > 1 else np.nan
        floor[name] = {
            "sigma_w": sigma_w,
            "s_d": s_d,
            "sigma": np.sqrt(2.0 * sigma_w**2 + (s_d if np.isfinite(s_d) else 0.0) ** 2),
            "runs": len(rows),
        }
    return floor, rows


# ---------------------------------------------------------------------------------------------
# Runs, references and the extrapolated values
# ---------------------------------------------------------------------------------------------


def read_references():
    """FVM-only references through the statistics window, keyed by (Re, h, dt)."""
    references = {}
    for path in reference_paths():
        name = path.parent.name
        match = REFERENCE_NAME.fullmatch(name)
        forces = np.atleast_1d(read_forces(path))
        if match is None or forces["time"][-1] < WINDOW[1] - 0.1:
            continue  # not a reference of the study, or not yet through the statistics window
        steps = jsonl_records(REFERENCE_DIR / "solution" / name / "performance.jsonl")
        time_step = round(steps[0]["time_step_size"], 9)  # fixed: the step the reference ran with
        references[reference_key(float(match["cell_size"]), time_step)] = {
            "name": name,
            "cell_size": float(match["cell_size"]),
            "time_step": time_step,
            "reynolds_number": REYNOLDS_NUMBER,
            "forces": forces,
            "statistics": shedding_statistics(forces, *WINDOW),
        }
    return references


def extrapolated_values(references):
    """Q_ext and its uncertainty per statistic from the references at Re = 150.

    The family at a constant Courant number gives Q_ext with the Eca-Hoekstra uncertainty of its
    finest level; the space-time fit Q = Q_ext + a h^p + b dt^q over every reference
    replaces it when its standard error is smaller (a statistic that barely changes with the cells,
    such as the mean drag, converges oscillatorily within its sampling floor), as in
    assets/error_analysis.py.
    """
    standard = list(references.values())
    family = sorted(
        (ref for ref in standard if np.isclose(ref["time_step"], ref["cell_size"] / CELL_SIZE * FVM_TIME_STEP)),
        key=lambda ref: ref["cell_size"],
    )
    exact = {}
    for name in STATISTICS:
        if len(family) < MINIMUM_LEVELS:  # references still running: no Q_ext yet
            exact[name] = {"Q_ext": np.nan, "U_ext": np.nan, "method": "too few levels",
                           "class": "", "order": np.nan, "levels": ";".join(r["name"] for r in family)}
            continue
        fit = discretization_uncertainty(
            [ref["cell_size"] / DIAMETER for ref in family],
            [ref["statistics"][name] for ref in family],
        )
        finest = int(np.argmin(fit["x"]))
        exact[name] = {
            "Q_ext": fit["Q_ext"],
            "U_ext": fit["uncertainty"][finest],
            "method": "family C",
            "class": fit["class"],
            "order": fit["order"],
            "levels": ";".join(ref["name"] for ref in family),
        }
        if len(standard) > 5:
            combined = space_time_fit(
                [ref["cell_size"] for ref in standard],
                [ref["time_step"] for ref in standard],
                [ref["statistics"][name] for ref in standard],
            )
            if combined["standard errors"]["Q_ext"] < exact[name]["U_ext"]:
                exact[name].update(
                    Q_ext=combined["Q_ext"],
                    U_ext=combined["standard errors"]["Q_ext"],
                    method="space-time fit",
                    order=combined["p"],
                    levels=";".join(ref["name"] for ref in standard),
                )
    return exact


def run_parameters(name):
    """Every study parameter of a coupled run from its directory name."""
    given = RUN_NAME.fullmatch(name).groupdict()
    parameters = {
        key: type(BASELINE[key])(value) if value else BASELINE[key] for key, value in given.items()
    }
    refinement = parameters["cell_size"] / CELL_SIZE
    if not given["fvm_time_step"]:
        parameters["fvm_time_step"] = refinement * FVM_TIME_STEP
    if not given["particle_spacing"]:
        parameters["particle_spacing"] = parameters["cell_size"]
    if not given["interface_cell_size"]:
        parameters["interface_cell_size"] = parameters["cell_size"]
    if not given["box_downstream"]:
        parameters["box_downstream"] = parameters["box_margin"] + 2.0
    parameters["particle_time_step"] = parameters["fvm_substeps"] * parameters["fvm_time_step"]
    return parameters


def transfer_indicators(directory):
    """Developed-window means of the per-transfer indicators, the cost and the backend check."""
    path = directory / "solution/coupler_diagnostics.jsonl"
    records = [r for r in jsonl_records(path) if WINDOW[0] <= r["time"] <= WINDOW[1]]
    iterations = [r.get("coupling_iteration", {}) for r in records]
    backends = {
        r["last_induction_image_call"]["mesh"]["finite_evaluation"]["execution_backend"]
        for r in records
        if "last_induction_image_call" in r
    }
    fallbacks = sum(
        r["last_induction_image_call"]["mesh"]["finite_evaluation"]["memory_fallback"] is not None
        for r in records
        if "last_induction_image_call" in r
    )
    times = [r["time"] for r in records]
    return {
        "boundary mismatch mean [U]": np.mean(
            [r["fvm_boundary_data"]["mean_velocity_mismatch"] for r in records]
        )
        / SPEED,
        "removed strength fraction": np.mean(
            [r["transfer"]["removed_vortex_strength_fraction"] for r in records if "transfer" in r]
        ),
        "moment correction fraction": np.mean(
            [r["gbd_moment_correction"]["correction_fraction"] for r in records if "gbd_moment_correction" in r]
        ),
        "coupling iterations": np.mean([it.get("iterations", 1) for it in iterations]),
        "converged fraction": np.mean([it.get("converged", True) for it in iterations]),
        "particles": np.mean([r["transfer"]["n_particles_after"] for r in records if "transfer" in r]),
        "cost [s/s]": np.mean([r["timing_seconds"]["total"] for r in records])
        / (np.diff(times).mean() if len(times) > 1 else np.nan),
        "image backend": "+".join(sorted(backends)),
        "memory fallbacks": int(fallbacks),
    }


def comparison_reference(parameters, references):
    """The FVM-only reference on the cells and step of a run (or None)."""
    return references.get(reference_key(parameters["cell_size"], parameters["fvm_time_step"]))


def read_runs(references, exact):
    """Parameters, statistics, errors and indicators of every coupled run through the window.

    "e_coupling" is against the reference on the run's cells and step.
    """
    runs = []
    for path in run_paths():
        directory = path.parents[1]
        forces = np.atleast_1d(read_forces(path))
        if forces["time"][-1] < WINDOW[1] - 0.1:
            continue  # still developing, or stopped early
        parameters = run_parameters(directory.name)
        statistics = shedding_statistics(forces, *WINDOW)
        reference = comparison_reference(parameters, references)
        run = {"name": directory.name, **parameters, "end time": forces["time"][-1]}
        run["reference"] = reference["name"] if reference else "none"
        at_re = parameters["reynolds_number"] == REYNOLDS_NUMBER
        for name in (*STATISTICS, *CHECKS):
            value = statistics[name]
            expected = reference["statistics"][name] if reference else np.nan
            # Q_ext exists at Re = 150 only; other Re scale with their own reference
            scale = exact[name]["Q_ext"] if at_re and name in exact else expected
            run[name] = value
            run[f"{name} e_coupling [%]"] = 100.0 * (value - expected) / scale
            if name in STATISTICS:
                phi0 = exact[name]["Q_ext"] if at_re else np.nan
                run[f"{name} e_FVM [%]"] = 100.0 * (expected - phi0) / scale
                run[f"{name} e_total [%]"] = 100.0 * (value - phi0) / scale
        run.update(transfer_indicators(directory))
        runs.append(run)
    return runs


# ---------------------------------------------------------------------------------------------
# One-at-a-time series of the five groups
# ---------------------------------------------------------------------------------------------


def at_baseline(run, *varied):
    return all(
        run[key] == value if isinstance(value, str) else np.isclose(run[key], value)
        for key, value in BASELINE.items()
        if key not in varied
    )


def panel_definitions():
    """Figure, x label, parameter, log2 axis and (series label, run filter) of every panel.

    Each series varies one parameter about the baseline; a panel holds the series of one parameter
    (for instance tau_transfer at N = 5 and at N = 2).
    """
    n2 = lambda r: r["fvm_substeps"] == 2 and np.isclose(r["fvm_time_step"], FVM_TIME_STEP)
    return [
        # D: the FVM domain; R: the same coupling at other Reynolds numbers
        ("D", r"margin [$D$]", "box_margin", True, [
            ("downstream face at 4.4 $D$", lambda r: at_baseline(r, "box_margin", "box_downstream")
             and np.isclose(r["box_downstream"], 4.4)),
            ("downstream face at $m+2D$", lambda r: at_baseline(r, "box_margin", "box_downstream")
             and np.isclose(r["box_downstream"], r["box_margin"] + 2.0)),
        ]),
        ("D", r"face $x_d$ [$D$]", "box_downstream", True, [
            ("margin 2.4 $D$", lambda r: at_baseline(r, "box_downstream")),
        ]),
        ("D", r"$Re$", "reynolds_number", False, [
            ("own reference", lambda r: at_baseline(r, "reynolds_number")),
        ]),
        # I: the interface at a fixed FVM, particle set and interval
        ("I_transfer", r"$\tau_{\rm transfer}$ [1/s]", "transfer_removal_threshold", True, [
            ("$N=5$", lambda r: at_baseline(r, "transfer_removal_threshold")),
            ("$N=2$", lambda r: at_baseline(r, "transfer_removal_threshold", "fvm_substeps") and n2(r)),
        ]),
        ("I_transfer", "moment cap", "moment_correction_fraction", True, [
            ("$N=5$", lambda r: at_baseline(r, "moment_correction_fraction")),
        ]),
        ("I_transfer", "iterations", "coupling_iterations", False, [
            ("$N=5$", lambda r: at_baseline(r, "coupling_iterations")),
        ]),
        ("I_geometry", r"$\delta_0$ [cells]", "blend_start", False, [
            ("$N=5$", lambda r: at_baseline(r, "blend_start")),
        ]),
        ("I_geometry", r"$\delta_1$ [cells]", "blend_end", True, [
            ("$N=5$", lambda r: at_baseline(r, "blend_end")),
        ]),
        ("I_geometry", "inset [cells]", "transfer_inset", True, [
            ("$N=5$", lambda r: at_baseline(r, "transfer_inset")),
        ]),
        # T: the coupling interval at the baseline FVM step
        ("T", r"$\Delta t_{\rm VPM}$ [s]", "particle_time_step", True, [
            (r"$\Delta t_{\rm FVM}=0.008$ s", lambda r: at_baseline(r, "fvm_substeps")),
        ]),
        # V: the particle side at a fixed FVM and interface
        ("V", r"$h_{\rm VPM}$ [m]", "particle_spacing", True, [
            ("$N=5$", lambda r: at_baseline(r, "particle_spacing")),
        ]),
        ("V", "integrator", "vpm_integrator", False, [
            ("$N=5$", lambda r: at_baseline(r, "vpm_integrator")),
        ]),
        ("V", r"$\omega_{\min}$ [1/s]", "vorticity_floor", True, [
            ("$N=5$", lambda r: at_baseline(r, "vorticity_floor")),
        ]),
        # F: the FVM side at a fixed interface and particle set
        ("F", "outer correctors", "fvm_outer_correctors", False, [
            ("$N=5$", lambda r: at_baseline(r, "fvm_outer_correctors")),
        ]),
        ("F", r"$\Delta t_{\rm FVM}$ [s]", "fvm_time_step", True, [
            (r"$\Delta t_{\rm VPM}=0.04$ s", lambda r: at_baseline(r, "fvm_time_step", "fvm_substeps")
             and np.isclose(r["particle_time_step"], 0.04)),
        ]),
        ("F", r"$h_{\rm FVM}$ [m]", "cell_size", True, [
            (r"$\Delta t_{\rm VPM}=0.032$ s", lambda r: at_baseline(
                r, "cell_size", "fvm_time_step", "fvm_substeps", "particle_spacing", "interface_cell_size")
             and np.isclose(r["fvm_time_step"], 0.004) and np.isclose(r["particle_time_step"], 0.032)
             and np.isclose(r["interface_cell_size"], CELL_SIZE)
             and np.isclose(r["particle_spacing"], r["cell_size"])),
        ]),
        ("F", r"$Co_{\max}$", "max_courant", False, [
            ("$N=2$", lambda r: at_baseline(r, "max_courant", "fvm_substeps", "fvm_time_step") and (
                (np.isclose(r["fvm_time_step"], 0.02) and r["fvm_substeps"] == 2) or (r["max_courant"] == 0.0 and n2(r)))),
        ]),
    ]


SHORT_NAMES = {
    "box_margin": "margin", "box_downstream": "downstream face", "reynolds_number": "$Re$",
    "transfer_removal_threshold": r"$\tau_{\rm transfer}$",
    "moment_correction_fraction": "moment cap", "coupling_iterations": "iterations",
    "blend_start": r"$\delta_0$", "blend_end": r"$\delta_1$", "transfer_inset": "inset",
    "particle_time_step": r"$\Delta t_{\rm VPM}$", "particle_spacing": r"$h_{\rm VPM}$",
    "vpm_integrator": "integrator", "vorticity_floor": r"$\omega_{\min}$",
    "fvm_outer_correctors": "correctors", "fvm_time_step": r"$\Delta t_{\rm FVM}$",
    "cell_size": r"$h_{\rm FVM}$", "max_courant": r"$Co_{\max}$",
}
SHORT_SERIES = {"downstream face at 4.4 $D$": "(face 4.4)", "downstream face at $m+2D$": "(face $m$+2)",
                "$N=5$": "($N$=5)", "$N=2$": "($N$=2)"}
GROUP_OF_FIGURE = {"D": "D", "I_transfer": "I", "I_geometry": "I", "T": "T", "V": "V", "F": "F"}
SERIES_STYLES = ((COLORS["hybrid"], "o"), (COLORS["fvm"], "s"), (COLORS["vpm"], "^"))
TICK_LABELS = {
    "coupling_iterations": lambda v: "tol." if v == 0 else f"{v:g}",
    "max_courant": lambda v: "fixed" if v == 0 else f"{v:g}",
    "vpm_integrator": lambda v: v.upper(),
}


def position(parameter, value):
    """Plot position of a level: numbers as they are, the integrators by their order."""
    if isinstance(value, str):
        return {"rk2": 2, "ssprk3": 3, "rk4": 4}.get(value, 0)
    return value


def series_summary(members, parameter, log_scale, floor):
    """Range of e_coupling over a series and its slope per factor of two, per statistic."""
    summary = {}
    x = np.array([position(parameter, run[parameter]) for run in members], float)
    for name in STATISTICS:
        error = np.array([run[f"{name} e_coupling [%]"] for run in members], float)
        span = np.nanmax(error) - np.nanmin(error) if len(error) > 1 else np.nan
        slope = np.nan
        if len(members) > 1 and log_scale and np.all(x > 0) and np.ptp(np.log2(x)) > 0:
            slope = np.polyfit(np.log2(x), error, 1)[0]
        threshold = RESOLVED * floor[name]["sigma"]
        summary[name] = {
            "range": span,
            "slope per factor 2": slope,
            "resolved": bool(np.isfinite(span) and span > threshold),
            "threshold": threshold,
        }
    return summary


# ---------------------------------------------------------------------------------------------
# Tables and figures
# ---------------------------------------------------------------------------------------------


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: (f"{v:.6g}" if isinstance(v, (float, np.floating)) else v) for k, v in row.items()}
            )


def group_figure(name, panels, floor):
    """One column per parameter, one row per statistic: e_coupling of each series, +-1 sigma bars,
    and the 2-sigma band about the first series' baseline run."""
    panels = [panel for panel in panels if any(len(members) for _, members in panel["series"])]
    if not panels:
        return
    set_thesis_style()
    rows = len(STATISTICS)
    labels = {
        label for panel in panels for label, members in panel["series"]
        if members and sum(bool(m) for _, m in panel["series"]) > 1
    }
    legend_rows = int(np.ceil(len(labels) / 2)) if len(labels) > 1 else 0
    height = 2.0 + 0.7 * legend_rows + 3.2 * rows
    figure, axes = plt.subplots(
        rows, len(panels), squeeze=False, sharex="col", sharey="row",
        figsize=(MAX_FIGURE_WIDTH_CM * CM, height * CM),
    )
    centered_subplots_adjust(
        figure, outer=0.14, bottom=1.5 / height, top=1.0 - (0.4 + 0.7 * legend_rows) / height,
        wspace=0.12, hspace=0.12,
    )
    handles = {}
    for column, panel in enumerate(panels):
        parameter = panel["parameter"]
        levels = sorted(
            {position(parameter, run[parameter]) for _, members in panel["series"] for run in members}
        )
        baseline = position(parameter, BASELINE.get(parameter))
        key = "e_coupling [%]"
        for row, statistic in enumerate(STATISTICS):
            axis = axes[row, column]
            axis.axhline(0.0, **REFERENCE_STYLE)
            first = panel["series"][0][1]
            anchor = [run for run in first if position(parameter, run[parameter]) == baseline]
            if anchor:
                centre = anchor[0][f"{statistic} {key}"]
                band = RESOLVED * floor[statistic]["sigma"]
                axis.axhspan(centre - band, centre + band, color=BACKGROUND_LIGHT, zorder=0)
            for (label, members), (color, marker) in zip(panel["series"], SERIES_STYLES):
                members = sorted(members, key=lambda run: position(parameter, run[parameter]))
                if not members:
                    continue
                line = axis.errorbar(
                    [position(parameter, run[parameter]) for run in members],
                    [run[f"{statistic} {key}"] for run in members],
                    yerr=floor[statistic]["sigma"], color=color, marker=marker,
                    markerfacecolor=color,
                    markersize=MARKER_SIZE, linewidth=LINE_WIDTH, capsize=2.0,
                )
                if sum(bool(m) for _, m in panel["series"]) > 1:
                    handles.setdefault(label, line)  # only panels that hold several series
            if column == 0:
                axis.set_ylabel(STATISTICS[statistic] + "\n" + r"$e_{\rm coupling}$ [\%]")
        bottom = axes[-1, column]
        if panel["log scale"]:
            bottom.set_xscale("log", base=2)
            bottom.minorticks_off()
        if len(levels) == 1 or not panel["log scale"]:
            low, high = min(levels), max(levels)
            margin = 0.15 * (high - low) if high > low else 0.5
            bottom.set_xlim(low - margin, high + margin)
        else:
            bottom.set_xlim(min(levels) / 1.3, max(levels) * 1.3)
        bottom.set_xticks(levels)
        bottom.set_xticks([], minor=True)
        formatter = TICK_LABELS.get(parameter, lambda v: f"{v:g}")
        values = {position(parameter, run[parameter]): run[parameter]
                  for _, members in panel["series"] for run in members}
        bottom.set_xticklabels(
            [formatter(values[level]) if level in clear_labels(bottom, levels, baseline) else ""
             for level in levels]
        )
        bottom.set_xlabel(panel["x label"])
    for axis in axes.ravel():
        axis.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    if len(handles) > 1:
        figure.legend(list(handles.values()), list(handles), loc="upper center", ncol=2, frameon=False)
    fit_thesis_y_label_margins(figure, figure.axes)
    save_validation_figure(figure, FIGURES / f"interface_group_{name}")


def clear_labels(axis, levels, baseline, spacing=0.27):
    """The levels whose tick labels keep ``spacing`` (axis fraction) apart, the baseline first."""
    low, high = axis.get_xlim()
    scale = np.log2 if axis.get_xscale() == "log" else (lambda v: v)
    fraction = {level: (scale(level) - scale(low)) / (scale(high) - scale(low)) for level in levels}
    chosen = []
    for level in sorted(levels, key=lambda level: (level != baseline, abs(fraction[level] - fraction.get(baseline, 0.0)))):
        if all(abs(fraction[level] - fraction[other]) >= spacing for other in chosen):
            chosen.append(level)
    return set(chosen)


def ranking_figure(ranking, floor):
    """Range of e_coupling of every series with two or more levels against the 2-sigma floor."""
    ranking = [row for row in ranking if row["levels"] > 1]
    if not ranking:
        return
    set_thesis_style()
    order = sorted(ranking, key=lambda row: -row["lift amplitude range"])
    height = 3.0 + 0.6 * len(order)
    figure, axes = plt.subplots(1, len(STATISTICS), sharey=True, figsize=(MAX_FIGURE_WIDTH_CM * CM, height * CM))
    centered_subplots_adjust(figure, outer=0.25, bottom=2.0 / height, top=1.0 - 0.3 / height, wspace=0.12)
    y = np.arange(len(order))
    colors = {"D": COLORS["hybrid"], "I": COLORS["fvm"], "T": COLORS["vpm"], "V": COLORS["reference"], "F": COLORS["hybrid"]}
    for axis, statistic in zip(axes, STATISTICS):
        axis.barh(y, [row[f"{statistic} range"] for row in order], height=0.6,
                  color=[colors.get(row["group"], COLORS["reference"]) for row in order])
        axis.axvline(RESOLVED * floor[statistic]["sigma"], **REFERENCE_STYLE)
        axis.set_xlabel(STATISTICS[statistic] + r" [\%]")
        axis.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        axis.xaxis.set_major_locator(MaxNLocator(3))
    axes[0].set_yticks(y, [f"{row['group']} {row['label']}" for row in order])
    figure.text(0.5, 0.25 / height, r"range of $e_{\rm coupling}$ over the series", ha="center", va="bottom")
    axes[0].invert_yaxis()
    fit_thesis_y_label_margins(figure, figure.axes)
    save_validation_figure(figure, FIGURES / "interface_ranking")


def main():
    references = read_references()
    exact = extrapolated_values(references)
    floor_paths = [*reference_paths(), *run_paths()]  # those that reach FLOOR_RUN_END
    floor, floor_rows = sampling_floor(floor_paths)
    runs = read_runs(references, exact)

    print(f"Statistics over whole periods in {WINDOW[0]:g} <= t <= {WINDOW[1]:g} s")
    print("Q_ext (FVM only, zero cell size and step):")
    for name, value in exact.items():
        print(f"  {name:16s} {value['Q_ext']:.5f} +- {value['U_ext']:.5f} ({value['method']}, {value['class']})")
    print(f"Sampling floor from {floor[next(iter(floor))]['runs']} run(s) to >= {FLOOR_RUN_END:g} s:")
    for name, value in floor.items():
        print(
            f"  {name:26s} sigma_w {value['sigma_w']:.3f} %  s_d {value['s_d']:.3f} %"
            f"  sigma of a difference {value['sigma']:.3f} %  (resolved above {RESOLVED * value['sigma']:.3f} %)"
        )
    print(f"\n{len(runs)} coupled runs through the window (e_coupling [%]: drag, lift amplitude, St)")
    for run in runs:
        flag = "" if run["image backend"] == "cupy_cuda" and run["memory fallbacks"] == 0 else "  CHECK BACKEND"
        print(
            f"  {run['name']:52s} " + " ".join(f"{run[f'{n} e_coupling [%]']:+7.3f}" for n in STATISTICS)
            + f"  mismatch {run['boundary mismatch mean [U]']:.2e}  vs {run['reference']}{flag}"
        )

    figures, ranking = {}, []
    for figure, x_label, parameter, log_scale, series in panel_definitions():
        members = [(label, [run for run in runs if keep(run)]) for label, keep in series]
        figures.setdefault(figure, []).append(
            {"x label": x_label, "parameter": parameter, "log scale": log_scale, "series": members}
        )
        for label, chosen in members:
            summary = series_summary(chosen, parameter, log_scale, floor) if chosen else None
            ranking.append({
                "group": GROUP_OF_FIGURE[figure],
                "label": SHORT_NAMES.get(parameter, parameter) + (f" {SHORT_SERIES.get(label, label)}" if len(series) > 1 else ""),
                "parameter": parameter,
                "series": label,
                "levels": len(chosen),
                "runs": ";".join(run["name"] for run in sorted(chosen, key=lambda r: position(parameter, r[parameter]))),
                **{
                    f"{name} {key}": (summary[name][key] if summary else np.nan)
                    for name in STATISTICS
                    for key in ("range", "slope per factor 2", "resolved", "threshold")
                },
            })

    TABLES.mkdir(parents=True, exist_ok=True)
    write_csv(TABLES / "interface_runs.csv", runs)
    write_csv(TABLES / "interface_series.csv", ranking)
    write_csv(TABLES / "interface_floor.csv",
              [{"statistic": name, **value} for name, value in floor.items()] + floor_rows)
    write_csv(
        TABLES / "interface_reference_values.csv",
        [{"statistic": name, **value} for name, value in exact.items()]
        + [{"statistic": ref["name"], **ref["statistics"]} for ref in references.values()],
    )
    print(f"\nwrote {', '.join(p.name for p in sorted(TABLES.glob('interface_*.csv')))}")

    print("\nSeries (range of e_coupling in points; * = above 2 sigma of the floor):")
    for row in ranking:
        if row["levels"] < 2:
            print(f"  {row['group']} {row['label']:44s} {row['levels']} level(s): not yet comparable")
            continue
        print(f"  {row['group']} {row['label']:44s} " + "  ".join(
            f"{name.split()[0]} {row[f'{name} range']:.3f}{'*' if row[f'{name} resolved'] else ' '}"
            for name in STATISTICS))

    for name, panels in figures.items():
        if name != "T":
            group_figure(name, panels, floor)
    ranking_figure(ranking, floor)


if __name__ == "__main__":
    main()
