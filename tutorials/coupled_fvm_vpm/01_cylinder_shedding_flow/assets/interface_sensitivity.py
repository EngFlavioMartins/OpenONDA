#!/usr/bin/env python3
"""Error sources and one-at-a-time sensitivity of the coupling interface.

Every coupled run in sensitivity/box* (one solver version; older runs are archived by
COMPLETE.sh) is split into

    e_total = e_FVM + e_coupling,
    e_FVM      = Q_FVM / Q_ext - 1,             FVM only on the same cells and step,
    e_coupling = (Q_coupled - Q_FVM) / Q_ext,   what the particle wake and the interface add,

with Q_ext the FVM-only value at zero cell size and step (Eca-Hoekstra least squares over the
references at a constant Courant number, assets/error_analysis.py). Q_FVM is the fine-wake
reference (section g: h cells over the whole coupled FVM domain) where it exists for the run's
cells and step, so that e_coupling no longer holds the coarser wake cells of the standard
reference; otherwise the standard reference, and the row says so.

Statistics are taken over whole shedding periods in 40 <= t <= 55 s: the mean drag between the
first and the last upward lift crossing (whole periods, so the drag oscillation averages out),
the lift amplitude as half the peak-to-peak value, its per-period mean as a check, and the
Strouhal number from the mean crossing interval.

The sampling floor of a difference between two runs over the same window comes from the runs
that reach 100 s (section h and the 100 s reference): the spread of the statistic between
sliding windows of the developed flow (sigma_w, in each run) and the spread of the transient
drift of the 40-55 s window between those runs (s_d). For a difference of two runs
sigma = sqrt(2 sigma_w^2 + s_d^2). An effect counts as resolved when it exceeds 2 sigma.

Run from the case directory: python -m assets.interface_sensitivity
Writes figures/auxiliary/interface_*.csv and figures/interface_*.pdf/png.
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np

from openonda.plotting import CM, COLORS, LINE_WIDTH, MARKER_SIZE, MAX_FIGURE_WIDTH_CM
from openonda.plotting import BACKGROUND_LIGHT, REFERENCE_STYLE
from openonda.plotting import centered_subplots_adjust, fit_thesis_y_label_margins
from openonda.plotting import save_validation_figure, set_thesis_style

from assets.error_analysis import discretization_uncertainty
from assets.plot_forces import read_forces
from assets.plot_sensitivity import BASELINE, REFERENCE_NAME, RUN_NAME, jsonl_records
from setup import CELL_SIZE, DIAMETER, FREESTREAM_VELOCITY, FVM_SUBSTEPS, FVM_TIME_STEP
from setup import REYNOLDS_NUMBER

CASE_DIR = Path(__file__).resolve().parents[1]
FIGURES = CASE_DIR / "figures"
TABLES = FIGURES / "auxiliary"
SPEED = FREESTREAM_VELOCITY[0]
WINDOW = (40.0, 55.0)  # s, the statistics window of every run and reference
SLIDE = 2.5  # s, offset of the sliding windows of the floor
FLOOR_RUN_END = 95.0  # s, runs reaching this time give the sampling floor
CAP_FRACTION = 0.995  # a transfer within 0.5 % of kappa_max sits at the strength limit
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


def shedding_statistics(forces, start=WINDOW[0], end=WINDOW[1]):
    """Mean drag over whole periods, lift amplitudes and Strouhal number in [start, end]."""
    window = (forces["time"] >= start) & (forces["time"] <= end)
    time, drag = forces["time"][window], forces["drag_coefficient"][window]
    lift = forces["lift_coefficient"][window]
    fluctuation = lift - lift.mean()
    up = np.flatnonzero((fluctuation[:-1] < 0.0) & (fluctuation[1:] >= 0.0))
    crossing = time[up] - fluctuation[up] * np.diff(time)[up] / np.diff(fluctuation)[up]
    if crossing.size < 2:
        return None
    # Linear interpolation between samples: whole periods from the first to the last crossing
    grid = np.linspace(crossing[0], crossing[-1], 20 * len(time) + 1)
    drag_grid, lift_grid = np.interp(grid, time, drag), np.interp(grid, time, lift)
    per_period = [
        0.5 * np.ptp(lift_grid[(grid >= a) & (grid <= b)])
        for a, b in zip(crossing[:-1], crossing[1:])
    ]
    return {
        "mean drag": np.trapezoid(drag_grid, grid) / (grid[-1] - grid[0]),
        "lift amplitude": 0.5 * np.ptp(lift),
        "Strouhal number": DIAMETER * (crossing.size - 1) / ((crossing[-1] - crossing[0]) * SPEED),
        "lift amplitude per period": float(np.mean(per_period)),
        "mean drag, whole window": drag.mean(),
        "periods": crossing.size - 1,
    }


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
        study = shedding_statistics(forces)
        row = {"run": path.parents[1].name, "end time": forces["time"][-1], "windows": len(windows)}
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
    """FVM-only references keyed by (Re, h, dt, fine wake); the base reference is reference_flow."""
    references = {}
    for path in [
        CASE_DIR / "reference_flow/samples/forces_history.csv",
        *sorted(CASE_DIR.glob("sensitivity/reference_*/samples/forces_history.csv")),
    ]:
        directory = path.parents[1]
        forces = np.atleast_1d(read_forces(path))
        if forces["time"][-1] < WINDOW[1] - 0.1:
            continue  # not yet through the statistics window
        match = REFERENCE_NAME.fullmatch(directory.name.replace("reference_flow", "reference"))
        if match is None:
            continue
        cell_size = float(match["cell_size"] or CELL_SIZE)
        time_step = float(match["time_step"] or cell_size / CELL_SIZE * FVM_TIME_STEP)
        reynolds_number = float(match["reynolds_number"] or REYNOLDS_NUMBER)
        key = (reynolds_number, cell_size, time_step, bool(match["fine_wake_end"]))
        references[key] = {
            "name": directory.name,
            "cell_size": cell_size,
            "time_step": time_step,
            "reynolds_number": reynolds_number,
            "fine wake": bool(match["fine_wake_end"]),
            "forces": forces,
            "statistics": shedding_statistics(forces),
        }
    return references


def extrapolated_values(references):
    """Q_ext and U_ext of each statistic from the standard references at a constant Courant number."""
    family = sorted(
        (
            ref
            for (re, h, dt, fine), ref in references.items()
            if re == REYNOLDS_NUMBER and not fine and np.isclose(dt, h / CELL_SIZE * FVM_TIME_STEP)
        ),
        key=lambda ref: ref["cell_size"],
    )
    exact = {}
    for name in STATISTICS:
        fit = discretization_uncertainty(
            [ref["cell_size"] / DIAMETER for ref in family],
            [ref["statistics"][name] for ref in family],
        )
        finest = int(np.argmin(fit["x"]))
        exact[name] = {
            "Q_ext": fit["Q_ext"],
            "U_finest": fit["uncertainty"][finest],
            "class": fit["class"],
            "order": fit["order"],
            "levels": ";".join(ref["name"] for ref in family),
        }
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
    configuration = json.loads((directory / "solution/coupler_metadata.json").read_text())
    kappa_max = configuration["configuration"]["max_strength_ratio"]
    kappa = np.array([r["transfer"]["strength_ratio"] for r in records if "transfer" in r])
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
        "kappa_max": kappa_max,
        "kappa median": np.median(kappa) if kappa.size else np.nan,
        "kappa max observed": kappa.max() if kappa.size else np.nan,
        "strength-limit fraction": np.mean(kappa >= CAP_FRACTION * kappa_max)
        if kappa.size
        else np.nan,
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


def comparison_references(parameters, references):
    """The fine-wake and the standard FVM-only reference of the run's Re, cells and step (or None)."""
    key = (parameters["reynolds_number"], parameters["cell_size"], parameters["fvm_time_step"])
    return references.get((*key, True)), references.get((*key, False))


def read_runs(references, exact):
    """Parameters, statistics, errors and indicators of every coupled run through the window.

    "e_coupling" is against the fine-wake reference where it exists, else the standard one;
    "e_coupling std" is always against the standard reference, for series that mix FVM steps or
    cells only some of which have a fine-wake reference.
    """
    runs = []
    for path in sorted(CASE_DIR.glob("sensitivity/box*/samples/forces_history.csv")):
        directory = path.parents[1]
        forces = np.atleast_1d(read_forces(path))
        if forces["time"][-1] < WINDOW[1] - 0.1:
            continue  # still developing, or stopped early
        parameters = run_parameters(directory.name)
        statistics = shedding_statistics(forces)
        fine, standard = comparison_references(parameters, references)
        reference = fine or standard
        run = {"name": directory.name, **parameters, "end time": forces["time"][-1]}
        run["reference"] = reference["name"] if reference else "none"
        run["fine-wake reference"] = fine is not None
        run["standard reference"] = standard["name"] if standard else "none"
        at_re = parameters["reynolds_number"] == REYNOLDS_NUMBER
        for name in (*STATISTICS, *CHECKS):
            value = statistics[name]
            expected = reference["statistics"][name] if reference else np.nan
            expected_standard = standard["statistics"][name] if standard else np.nan
            # Q_ext exists at Re = 150 only; other Re scale with their own reference
            scale = exact[name]["Q_ext"] if at_re and name in exact else expected
            run[name] = value
            run[f"{name} e_coupling [%]"] = 100.0 * (value - expected) / scale
            run[f"{name} e_coupling std [%]"] = 100.0 * (value - expected_standard) / scale
            if name in STATISTICS:
                phi0 = exact[name]["Q_ext"] if at_re else np.nan
                run[f"{name} e_FVM [%]"] = 100.0 * (expected - phi0) / scale
                run[f"{name} e_total [%]"] = 100.0 * (value - phi0) / scale
        run.update(transfer_indicators(directory))
        runs.append(run)
    return runs


def error_key(members):
    """The coupling error a series is drawn with: fine-wake when every member has that reference."""
    if members and all(run["fine-wake reference"] for run in members):
        return "e_coupling [%]"
    return "e_coupling std [%]"


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
    (for instance kappa_max at N = 5 and at N = 2).
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
        ("I_transfer", r"$\kappa_{\max}$", "max_strength_ratio", True, [
            ("$N=5$", lambda r: at_baseline(r, "max_strength_ratio")),
            ("$N=2$", lambda r: at_baseline(r, "max_strength_ratio", "fvm_substeps") and n2(r)),
        ]),
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
            (r"$\kappa_{\max}=1.8$", lambda r: at_baseline(r, "fvm_substeps")),
            (r"$\kappa_{\max}=3$", lambda r: at_baseline(r, "fvm_substeps", "max_strength_ratio")
             and np.isclose(r["max_strength_ratio"], 3.0)),
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
    "max_strength_ratio": r"$\kappa_{\max}$", "transfer_removal_threshold": r"$\tau_{\rm transfer}$",
    "moment_correction_fraction": "moment cap", "coupling_iterations": "iterations",
    "blend_start": r"$\delta_0$", "blend_end": r"$\delta_1$", "transfer_inset": "inset",
    "particle_time_step": r"$\Delta t_{\rm VPM}$", "particle_spacing": r"$h_{\rm VPM}$",
    "vpm_integrator": "integrator", "vorticity_floor": r"$\omega_{\min}$",
    "fvm_outer_correctors": "correctors", "fvm_time_step": r"$\Delta t_{\rm FVM}$",
    "cell_size": r"$h_{\rm FVM}$", "max_courant": r"$Co_{\max}$",
}
SHORT_SERIES = {"downstream face at 4.4 $D$": "(face 4.4)", "downstream face at $m+2D$": "(face $m$+2)",
                r"$\kappa_{\max}=1.8$": r"($\kappa_{\max}$ 1.8)", r"$\kappa_{\max}=3$": r"($\kappa_{\max}$ 3)",
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
    summary = {"reference": "fine wake" if error_key(members) == "e_coupling [%]" else "standard"}
    x = np.array([position(parameter, run[parameter]) for run in members], float)
    for name in STATISTICS:
        error = np.array([run[f"{name} {error_key(members)}"] for run in members], float)
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
        # One reference type per panel; open markers: the standard reference (see error_key)
        key = error_key([run for _, members in panel["series"] for run in members])
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
                    markerfacecolor=color if key == "e_coupling [%]" else "white",
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


def interval_figure(runs, floor):
    """e_coupling and median kappa against the coupling interval at kappa_max = 1.8 and 3, with the
    runs at other FVM steps as open markers: the mechanism test of the interval error."""
    series = [
        (kappa, [run for run in runs
                 if at_baseline(run, "fvm_substeps", "fvm_time_step", "max_strength_ratio")
                 and np.isclose(run["max_strength_ratio"], kappa)], color, marker)
        for kappa, (color, marker) in zip((1.8, 3.0), SERIES_STYLES)
    ]
    if sum(len(members) for _, members, _, _ in series) < 3:
        return
    key = error_key([run for _, members, _, _ in series for run in members])
    set_thesis_style()
    quantities = (*STATISTICS, "kappa median")
    labels = sum(len({run["fvm_time_step"] for run in members}) for _, members, _, _ in series)
    legend_rows = int(np.ceil(labels / 2))
    height = 2.2 + 0.7 * legend_rows + 2 * 4.0
    figure, grid = plt.subplots(2, 2, sharex=True, figsize=(MAX_FIGURE_WIDTH_CM * CM, height * CM))
    centered_subplots_adjust(figure, outer=0.12, bottom=1.5 / height,
                             top=1.0 - (0.4 + 0.7 * legend_rows) / height, wspace=0.42, hspace=0.12)
    axes = grid.ravel()
    handles = {}
    for kappa, members, color, marker in series:
        for step in sorted({run["fvm_time_step"] for run in members}, reverse=True):
            chosen = sorted([run for run in members if np.isclose(run["fvm_time_step"], step)],
                            key=lambda run: run["particle_time_step"])
            main = np.isclose(step, FVM_TIME_STEP)
            for axis, quantity in zip(axes, quantities):
                line = axis.errorbar(
                    [run["particle_time_step"] for run in chosen],
                    [run[f"{quantity} {key}"] if quantity in STATISTICS else run[quantity] for run in chosen],
                    yerr=floor[quantity]["sigma"] if quantity in STATISTICS else None,
                    color=color, marker=marker, markersize=MARKER_SIZE,
                    markerfacecolor=color if main else "white", linewidth=LINE_WIDTH if main else 0.0, capsize=2.0,
                )
            label = rf"$\kappa_{{\max}}={kappa:g}$, $\Delta t_{{\rm FVM}}={step:g}$ s"
            handles.setdefault(label, line)
        axes[-1].axhline(kappa, color=color, linewidth=0.5 * LINE_WIDTH, linestyle=":")
    for axis, quantity in zip(axes, STATISTICS):
        axis.axhline(0.0, **REFERENCE_STYLE)
        axis.set_ylabel(STATISTICS[quantity] + " " + r"$e_{\rm coupling}$ [\%]")
    axes[-1].set_ylabel(r"median $\kappa$")
    steps = sorted({run["particle_time_step"] for _, members, _, _ in series for run in members})
    for axis in grid[-1]:
        axis.set_xscale("log", base=2)
        axis.set_xlim(min(steps) / 1.3, max(steps) * 1.3)
        axis.set_xticks(steps)
        axis.set_xticks([], minor=True)
        shown = clear_labels(axis, steps, FVM_SUBSTEPS * FVM_TIME_STEP, spacing=0.12)
        axis.set_xticklabels([f"{step:g}" if step in shown else "" for step in steps])
        axis.set_xlabel(r"coupling interval $\Delta t_{\rm VPM}$ [s]")
    for axis in axes:
        axis.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    figure.legend(list(handles.values()), list(handles), loc="upper center", ncol=2, frameon=False)
    fit_thesis_y_label_margins(figure, figure.axes)
    save_validation_figure(figure, FIGURES / "interface_interval_mechanism")


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
    floor_paths = [
        CASE_DIR / "reference_flow/samples/forces_history.csv",
        *sorted(CASE_DIR.glob("sensitivity/box*/samples/forces_history.csv")),
    ]
    floor, floor_rows = sampling_floor(floor_paths)
    runs = read_runs(references, exact)

    print(f"Statistics over whole periods in {WINDOW[0]:g} <= t <= {WINDOW[1]:g} s")
    print("Q_ext (FVM only, zero cell size and step):")
    for name, value in exact.items():
        print(f"  {name:16s} {value['Q_ext']:.5f} +- {value['U_finest']:.5f} (finest level), {value['class']}")
    print(f"Sampling floor from {floor[next(iter(floor))]['runs']} run(s) to >= {FLOOR_RUN_END:g} s:")
    for name, value in floor.items():
        print(
            f"  {name:26s} sigma_w {value['sigma_w']:.3f} %  s_d {value['s_d']:.3f} %"
            f"  sigma of a difference {value['sigma']:.3f} %  (resolved above {RESOLVED * value['sigma']:.3f} %)"
        )
    wake = {}
    for (re, h, dt, fine), ref in references.items():
        if fine and (re, h, dt, False) in references:
            standard = references[re, h, dt, False]
            wake[dt] = {
                name: 100.0 * (ref["statistics"][name] - standard["statistics"][name]) / exact[name]["Q_ext"]
                for name in STATISTICS
            }
            print(f"Fine-wake minus standard reference at dt = {dt:g} s: "
                  + ", ".join(f"{name} {value:+.3f} %" for name, value in wake[dt].items()))

    print(f"\n{len(runs)} coupled runs through the window (e_coupling [%]: drag, lift amplitude, St)")
    for run in runs:
        flag = "" if run["image backend"] == "cupy_cuda" and run["memory fallbacks"] == 0 else "  CHECK BACKEND"
        print(
            f"  {run['name']:52s} " + " ".join(f"{run[f'{n} e_coupling [%]']:+7.3f}" for n in STATISTICS)
            + f"  kappa {run['kappa median']:.2f} ({100 * run['strength-limit fraction']:.0f}% at limit)"
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
                "reference": summary["reference"] if summary else "",
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
        + [{"statistic": f"fine minus standard wake, dt {dt:g} s", **values} for dt, values in wake.items()]
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
    interval_figure(runs, floor)
    ranking_figure(ranking, floor)


if __name__ == "__main__":
    main()
