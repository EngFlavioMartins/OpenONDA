#!/usr/bin/env python3
"""Plot discretization error, coupling error and diagnostic indicators.

Reference refinements provide FVM error estimates. The difference between a
coupled run and its matched reference is the coupling contribution.
"""

from itertools import combinations

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
from scipy.stats import pearsonr, spearmanr

from openonda.plotting import fit_thesis_y_label_margins
from openonda.plotting import BACKGROUND_STRONG, CM, COLORS, LINE_WIDTH, MARKER_SIZE
from openonda.plotting import MAX_FIGURE_WIDTH_CM, centered_subplots_adjust
from openonda.plotting import save_validation_figure, set_thesis_style

from assets.error_analysis import FACTORIAL_LEVELS, INDICATORS, MINIMUM_LEVELS
from assets.error_analysis import PARTICLE_STRAIN_RATE, additivity_defect, box_scales
from assets.error_analysis import coupled_families, decompose, discretization_uncertainty
from assets.error_analysis import driver_fit, drivers, exact_values, factorial_effects
from assets.error_analysis import interface_indicators, logged_keys, reference_families
from assets.error_analysis import space_time_fit, statistics_of, term_model, wake_scales
from assets.plot_sensitivity import BASELINE, CASE_DIR, ERRORS, LATE_WINDOW, STATISTICS
from assets.plot_sensitivity import STATISTICS_WINDOW, errors_against, read_fine_wake_reference
from assets.plot_sensitivity import read_references, read_runs
from setup import DIAMETER, FREESTREAM_VELOCITY, REYNOLDS_NUMBER

SPEED = FREESTREAM_VELOCITY[0]
FIGURES = CASE_DIR / "figures"
BASELINE_NAME = "box2.4_sub5_inset8.75"
TABLES = CASE_DIR / "figures/auxiliary"

# Refinement variables in flow scales.
VARIABLES = {
    "cell_size": ("$h/D$", 1.0 / DIAMETER),
    "time_step": (r"$U\Delta t/D$", SPEED / DIAMETER),
    "fvm_time_step": (r"$U\Delta t_{\rm FVM}/D$", SPEED / DIAMETER),
    "particle_time_step": (r"$U\Delta t_{\rm VPM}/D$", SPEED / DIAMETER),
}
CONVERGENCE_FAMILIES = {"C": "cell_size", "X": "cell_size", "T2": "fvm_time_step"}
REFINEMENT_FAMILIES = {
    **CONVERGENCE_FAMILIES,
    "A": "cell_size",
    "XP": "cell_size",
    "P": "fvm_time_step",
    "E": "particle_time_step",
}
DRIVER_FAMILIES = ("T2", "P", "E")
PLOTTED_DRIVERS = {
    "particle step": r"$U\Delta t_{\rm VPM}/D$",
    "coupling intervals per convective time": r"$D/(U\Delta t_{\rm VPM})$",
}
PLOTTED_INDICATORS = {
    "boundary velocity mismatch, mean": r"$\Delta u_{\Sigma_{\rm FVM}}/U$",
}
SERIES = {
    "total": ("Coupled solver, total", COLORS["hybrid"], "o"),
    "FVM": ("FVM discretization", COLORS["reference"], "s"),
    "coupling": ("Coupling", COLORS["vpm"], "^"),
}
# Panel titles and legend names for the study families.
FAMILY_NAMES = {
    "C": "Space and time",
    "X": "Space",
    "T": "Time",
    "T2": "Time",
    "P": "FVM step",
    "E": "Particle step",
    "K": "Coupling iterations",
    "B": "Box",
    "R": "Reynolds number",
    "S": "Particle spacing",
    "A": "Space, fixed interval 0.032",
    "XP": "Space, fixed interval 0.04",
    "KAPPA": "Strength-ratio limit",
    "BL": "Blending depths",
}
# Draw each run in its first matching family.
FAMILY_STYLE = {
    "T2": (COLORS["fvm"], "s"),
    "P": (COLORS["vpm"], "^"),
    "E": (COLORS["red"], "D"),
    "C": (COLORS["hybrid"], "o"),
    "K": (COLORS["green"], "v"),
    "X": (COLORS["dark"], "P"),
    "B": (COLORS["gray"], "X"),
    "R": (COLORS["amber"], "*"),
    "S": (COLORS["teal"], "h"),
    "A": (COLORS["dark"], "p"),
    "XP": (COLORS["dark"], "d"),
    "KAPPA": (COLORS["red"], "<"),
    "BL": (COLORS["amber"], ">"),
}


def flow_scaled(variable, value):
    return value * VARIABLES[variable][1]


def drawn_family(entry):
    """The family a run or table row is drawn in."""
    return next(f for f in [*FAMILY_STYLE, "-"] if f in [*entry["families"].split(), "-"])


def panel_grid(n_rows, n_columns, top_cm=1.2, bottom_cm=3.2):
    """Rows of quantities and columns of families or drivers at the thesis width.

    The legend goes below the panels, clear of the data.
    """
    set_thesis_style()
    height = bottom_cm + top_cm + 3.0 * n_rows
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        squeeze=False,
        sharex="col",
        sharey="row",
        figsize=(MAX_FIGURE_WIDTH_CM * CM, height * CM),
    )
    centered_subplots_adjust(
        figure,
        outer=0.15 if n_columns > 1 else 0.25,
        bottom=bottom_cm / height,
        top=1.0 - top_cm / height,
        wspace=0.12,
        hspace=0.12,
    )
    return figure, axes


def bottom_legend(figure, axes):
    """One legend entry per label over all panels, in a row below them."""
    entries = {}
    for axis in axes.flat:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            entries.setdefault(label, handle)
    figure.legend(
        entries.values(),
        entries.keys(),
        loc="lower center",
        ncol=min(len(entries), 2),
        frameon=False,
        handlelength=1.5,
        columnspacing=1.0,
    )


def fit_row(kind, family, variable, quantity, fit):
    """One line of the order table, with the uncertainty of the finest level also in % of Q_ext."""
    finest = np.argmin(fit["x"])
    return {
        "kind": kind,
        "family": family,
        "variable": variable,
        "quantity": quantity,
        "levels": len(fit["x"]),
        "class": fit["class"],
        "fit": fit["fit"],
        "order": fit["order"],
        "Q_ext": fit["Q_ext"],
        "finest value": fit["phi"][finest],
        "uncertainty": fit["uncertainty"][finest],
        # The coupling error is already in % of Q_ext
        "uncertainty [%]": fit["uncertainty"][finest]
        if kind == "coupled coupling"
        else 100.0 * fit["uncertainty"][finest] / abs(fit["Q_ext"]),
        "safety factor": fit["safety factor"],
        "sigma": fit["sigma"],
    }


def write_table(path, rows):
    """Rows of dictionaries with the same keys, as CSV."""
    names = list(rows[0])
    # Column names keep their words; a comma inside a name would split the column
    lines = [",".join(name.replace(", ", " ") for name in names)]
    for row in rows:
        lines.append(
            ",".join(f"{row[n]:.6g}" if isinstance(row[n], float) else f"{row[n]}" for n in names)
        )
    path.write_text("\n".join(lines) + "\n")


def reference_orders(references):
    """Eça–Hoekstra fit of each reference family and statistic."""
    fits = {}
    for family, (variable, members) in reference_families(references).items():
        if len(members) < MINIMUM_LEVELS:
            continue
        members = sorted(members, key=lambda ref: ref[variable])
        for quantity in STATISTICS:
            fits[family, quantity] = discretization_uncertainty(
                [flow_scaled(variable, ref[variable]) for ref in members],
                [statistics_of(ref["forces"])[quantity] for ref in members],
            )
            fits[family, quantity]["variable"] = variable
    return fits


def space_time_orders(references):
    """Q = Q_ext + a h^p + b dt^q over every reference at Re = 150, once it has freedom."""
    levels = [ref for (re, h, dt), ref in references.items() if re == REYNOLDS_NUMBER]
    if len(levels) < 6:
        print(f"Space-time fit of the references: {len(levels)} levels, needs 6 (5 coefficients)")
        return []
    rows = []
    print(f"Space-time fit Q = Q_ext + a (h/D)^p + b (U dt/D)^q over {len(levels)} references")
    for quantity in STATISTICS:
        fit = space_time_fit(
            [flow_scaled("cell_size", ref["cell_size"]) for ref in levels],
            [flow_scaled("time_step", ref["time_step"]) for ref in levels],
            [statistics_of(ref["forces"])[quantity] for ref in levels],
        )
        errors = fit["standard errors"]
        print(
            f"  {quantity:16s} Q_ext {fit['Q_ext']:.4f} +- {errors['Q_ext']:.4f}, "
            f"p {fit['p']:.2f} +- {errors['p']:.2f}, q {fit['q']:.2f} +- {errors['q']:.2f}, "
            f"rms residual {100.0 * fit['rms residual'] / fit['Q_ext']:.3f}%"
        )
        rows.append(
            {
                "quantity": quantity,
                "levels": len(levels),
                **{name: fit[name] for name in ("Q_ext", "a", "p", "b", "q")},
                **{f"{name} standard error": value for name, value in errors.items()},
                "rms residual": fit["rms residual"],
            }
        )
    return rows


def coupled_orders(families, decomposition):
    """Eça–Hoekstra fit of the coupled statistic and of its coupling error, per family."""
    fits = {}
    for family, variable in REFINEMENT_FAMILIES.items():
        members = sorted(families[family][1], key=lambda run: run[variable])
        if len(members) < MINIMUM_LEVELS:
            continue
        x = [flow_scaled(variable, run[variable]) for run in members]
        for quantity in STATISTICS:
            fits[family, quantity, "total"] = discretization_uncertainty(
                x, [statistics_of(run["forces"])[quantity] for run in members]
            )
            fits[family, quantity, "coupling"] = discretization_uncertainty(
                x, [decomposition[run["name"]][quantity]["coupling"] for run in members]
            )
            for kind in ("total", "coupling"):
                fits[family, quantity, kind]["variable"] = variable
    return fits


def print_orders(title, fits):
    print(title)
    for key, fit in fits.items():
        finest = np.argmin(fit["x"])
        order = f"p = {fit['order']:.2f}" if np.isfinite(fit["order"]) else "no order"
        print(
            f"  {' '.join(key):34s} {fit['class']:11s} {order:10s} Q_ext = {fit['Q_ext']:9.4f}  "
            f"U = {fit['uncertainty'][finest]:.4f} at the finest level ({fit['fit']}, "
            f"F_s = {fit['safety factor']:.2f}, {len(fit['x'])} levels)"
        )


def reference_figure(fits, exact):
    """Reference error against h/D and U dt/D, with the fitted curves and the uncertainties."""
    columns = [family for family in ("C", "X", "T") if (family, "mean drag") in fits]
    if not columns:
        return
    figure, axes = panel_grid(len(STATISTICS), len(columns))
    for column, family in enumerate(columns):
        for row, quantity in enumerate(STATISTICS):
            axis, fit = axes[row, column], fits[family, quantity]
            phi0 = exact[quantity][0]
            curve = np.linspace(0.0, 1.05 * fit["x"].max(), 100)
            axis.plot(
                curve,
                100.0 * (fit["curve"](curve) - phi0) / phi0,
                color=COLORS["reference"],
                linewidth=LINE_WIDTH,
            )
            axis.errorbar(
                fit["x"],
                100.0 * (fit["phi"] - phi0) / phi0,
                yerr=100.0 * fit["uncertainty"] / abs(phi0),
                color=COLORS["reference"],
                marker="s",
                markersize=MARKER_SIZE,
                linestyle="none",
                elinewidth=LINE_WIDTH,
                capsize=2.0,
                label="FVM only",
            )
            if column == 0:
                axis.set_ylabel(STATISTICS[quantity])
        axes[0, column].set_title(FAMILY_NAMES[family])
        axes[-1, column].set_xlabel(VARIABLES[fits[family, "mean drag"]["variable"]][0])
        axes[-1, column].set_xlim(left=0.0)
    bottom_legend(figure, axes)
    fit_thesis_y_label_margins(figure, figure.axes)
    save_validation_figure(figure, FIGURES / "error_references_C_X_T")


def decomposition_figure(families, decomposition, fits, exact):
    """e, e_FVM and e_coupling of the C, X and T2 families against their refinement variable."""
    columns = [f for f, variable in CONVERGENCE_FAMILIES.items() if len(families[f][1]) > 1]
    if not columns:
        return
    figure, axes = panel_grid(len(ERRORS), len(columns))
    for column, family in enumerate(columns):
        variable = CONVERGENCE_FAMILIES[family]
        members = sorted(families[family][1], key=lambda run: run[variable])
        x = np.array([flow_scaled(variable, run[variable]) for run in members])
        curve = np.linspace(0.0, 1.05 * x.max(), 100)
        for row, quantity in enumerate(ERRORS):
            axis = axes[row, column]
            for kind, (label, color, marker) in SERIES.items():
                axis.plot(
                    x,
                    [decomposition[run["name"]][quantity][kind] for run in members],
                    color=color,
                    marker=marker,
                    markersize=MARKER_SIZE,
                    linestyle="none",
                    label=label,
                )
            # Fitted coupled statistic and coupling error, once the family has enough levels
            if (family, quantity, "total") in fits:
                phi0 = exact[quantity][0]
                total = fits[family, quantity, "total"]["curve"](curve)
                coupling = fits[family, quantity, "coupling"]["curve"](curve)
                axis.plot(
                    curve,
                    100.0 * (total - phi0) / phi0,
                    color=COLORS["hybrid"],
                    linewidth=LINE_WIDTH,
                )
                axis.plot(curve, coupling, color=COLORS["vpm"], linewidth=LINE_WIDTH)
            if column == 0:
                axis.set_ylabel(ERRORS[quantity])
        axes[0, column].set_title(FAMILY_NAMES[family])
        axes[-1, column].set_xlabel(VARIABLES[variable][0])
        axes[-1, column].set_xlim(left=0.0)
    bottom_legend(figure, axes)
    fit_thesis_y_label_margins(figure, figure.axes)
    save_validation_figure(figure, FIGURES / "error_decomposition_C_X_T2")


def driver_analysis(families, decomposition):
    """Fits of the coupling error over the T2, P and E runs on each driver, and both time terms.

    A coupling error added at each transfer accumulates with the coupling intervals per
    convective time D/(U dt_VPM): it grows as the particle step falls. A temporal error of the
    particle step falls with it. The two-term model
    e_coupling = c + a (U dt_VPM/D) + b D/(U dt_VPM) + d (U dt_FVM/D) weighs both against an FVM-step
    contribution; P holds dt_VPM while dt_FVM changes, E holds dt_FVM while dt_VPM changes, and
    T2 changes both at a fixed ratio.
    """
    pool = {run["name"]: run for family in DRIVER_FAMILIES for run in families[family][1]}
    pool = [
        run
        for run in pool.values()
        if np.isfinite(decomposition[run["name"]]["mean drag"]["coupling"])
    ]
    rows = []
    print(f"Coupling-error drivers over {len(pool)} runs of {', '.join(DRIVER_FAMILIES)}")
    if len(pool) < 4:
        print("  needs 4 runs at least (c_0, a, q and one residual freedom)")
        return rows
    values = [drivers(run) for run in pool]
    for quantity in ERRORS:
        error = np.array([decomposition[run["name"]][quantity]["coupling"] for run in pool])
        for driver in values[0]:
            x = np.array([value[driver] for value in values])
            if np.ptp(x) == 0.0:
                continue  # held fixed over these runs
            fit = driver_fit(x, error)
            rows.append({"quantity": quantity, "driver": driver, "runs": len(pool), **fit})
            print(
                f"  {quantity:16s} {driver:30s} e_coupling = {fit['c0']:+.3f} {fit['a']:+.3g} x^{fit['q']:+.2f}"
                f"  R^2 = {fit['R2']:.3f}, rms {fit['rms']:.3f}%"
            )

        # Temporal against accumulated, with the FVM step, as many terms as the runs allow
        particle_step = np.array([value["particle step"] for value in values])
        terms = {"U dt_VPM/D": particle_step, "D/(U dt_VPM)": 1.0 / particle_step}
        fvm_step = np.array([value["FVM step"] for value in values])
        if len(pool) > 4 and np.ptp(fvm_step / particle_step) > 0.0:
            terms["U dt_FVM/D"] = fvm_step
        model = term_model(terms, error)
        verdict = ", ".join(
            f"{name} {model['coefficients'][name]:+.3g} +- {model['standard errors'][name]:.2g}"
            + (
                " (significant)"
                if abs(model["coefficients"][name]) > 2.0 * model["standard errors"][name]
                else ""
            )
            for name in terms
        )
        print(f"  {quantity:16s} two-term model: {verdict}; R^2 = {model['R2']:.3f}")
    return rows


def driver_figure(families, decomposition, fits):
    """e_coupling against the particle step, the coupling intervals per convective time and N, by family."""
    # Only the runs on the baseline cells with converged iterations, so the particle step is the cause
    runs = list({run["name"]: run for f in DRIVER_FAMILIES for run in families[f][1]}.values())
    if len(runs) < 3:
        return
    figure, axes = panel_grid(len(ERRORS), len(PLOTTED_DRIVERS), top_cm=2.0)

    # Lagrangian CFL number on top of the particle-step column, from the particle-field strain
    to_stability = PARTICLE_STRAIN_RATE * DIAMETER / SPEED
    stability = axes[0, 0].secondary_xaxis(
        "top", functions=(lambda x: to_stability * x, lambda x: x / to_stability)
    )
    stability.set_xlabel(r"$\Lambda$")
    stability.xaxis.set_major_locator(MaxNLocator(3))
    figure.text(
        0.5,
        0.99,
        rf"$\Lambda=|S|\,\Delta t_{{\rm VPM}}$, particle-field $|S|={PARTICLE_STRAIN_RATE:g}$ s$^{{-1}}$",
        ha="center",
        va="top",
    )
    for column, (driver, label) in enumerate(PLOTTED_DRIVERS.items()):
        for row, quantity in enumerate(ERRORS):
            axis = axes[row, column]
            for family in DRIVER_FAMILIES:
                color, marker = FAMILY_STYLE[family]
                members = [run for run in runs if drawn_family(run) == family]
                if not members:
                    continue
                axis.plot(
                    [drivers(run)[driver] for run in members],
                    [decomposition[run["name"]][quantity]["coupling"] for run in members],
                    color=color,
                    marker=marker,
                    markersize=MARKER_SIZE,
                    linestyle="none",
                    label=FAMILY_NAMES[family],
                )
            fit = fits.get((quantity, driver))
            if fit:
                pool = [run for run in runs if drawn_family(run) in DRIVER_FAMILIES]
                x = np.array([drivers(run)[driver] for run in pool])
                curve = np.linspace(x.min(), x.max(), 100)
                axis.plot(
                    curve,
                    fit["c0"] + fit["a"] * curve ** fit["q"],
                    color=COLORS["text"],
                    linewidth=LINE_WIDTH,
                )
            if column == 0:
                axis.set_ylabel(r"$e_{\rm coupling}$" + "\n" + ERRORS[quantity])
        axes[-1, column].set_xlabel(label)
        axes[-1, column].margins(x=0.12)
        axes[-1, column].xaxis.set_major_locator(MaxNLocator(3))
        axes[-1, column].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    bottom_legend(figure, axes)
    fit_thesis_y_label_margins(figure, figure.axes)
    save_validation_figure(figure, FIGURES / "error_drivers_T2_P_E")


def indicator_analysis(runs, decomposition):
    """Developed means of the logged mismatch of every run, and their correlation with |e_coupling|."""
    rows = []
    for run in runs:
        means = interface_indicators(
            CASE_DIR / f"sensitivity/{run['name']}/solution/coupler_diagnostics.jsonl"
        )
        rows.append(
            {
                "run": run["name"],
                "families": run["families"] or "-",
                **means,
                **{
                    f"|e_coupling| {q} [%]": abs(decomposition[run["name"]][q]["coupling"])
                    for q in ERRORS
                },
            }
        )
    print(f"Indicator: developed means of the logged interface mismatch over {len(rows)} runs")
    correlations = {}
    for name in INDICATORS:
        for quantity in ERRORS:
            pairs = np.array([(row[name], row[f"|e_coupling| {quantity} [%]"]) for row in rows])
            pairs = pairs[np.all(np.isfinite(pairs), axis=1)]
            if len(pairs) < 3:
                continue
            correlations[name, quantity] = (
                pearsonr(*pairs.T).statistic,
                spearmanr(*pairs.T).statistic,
                len(pairs),
            )
            r, rho, count = correlations[name, quantity]
            print(
                f"  {name:31s} vs |e_coupling| {quantity:16s} r = {r:+.2f}, rho = {rho:+.2f} ({count} runs)"
            )
    return rows, correlations


def indicator_figure(rows, correlations):
    """|e_coupling| of every run against the developed mean of each plotted indicator."""
    if not any(key[0] in PLOTTED_INDICATORS for key in correlations):
        return
    figure, axes = panel_grid(len(ERRORS), len(PLOTTED_INDICATORS), bottom_cm=5.2)
    for column, (name, label) in enumerate(PLOTTED_INDICATORS.items()):
        # The decade of the indicator goes in the label, clear of the axis
        exponent = int(np.floor(np.log10(np.nanmax([entry[name] for entry in rows]))))
        for row, quantity in enumerate(ERRORS):
            axis = axes[row, column]
            for family, (color, marker) in FAMILY_STYLE.items():
                members = [entry for entry in rows if drawn_family(entry) == family]
                axis.plot(
                    [entry[name] / 10.0**exponent for entry in members],
                    [entry[f"|e_coupling| {quantity} [%]"] for entry in members],
                    color=color,
                    marker=marker,
                    markersize=MARKER_SIZE,
                    linestyle="none",
                    label=FAMILY_NAMES[family] if members else None,
                )
            if (name, quantity) in correlations:
                axis.text(
                    0.05,
                    0.82,
                    f"$r={correlations[name, quantity][0]:+.2f}$",
                    transform=axis.transAxes,
                )
            if column == 0:
                axis.set_ylabel(r"$|e_{\rm coupling}|$" + "\n" + ERRORS[quantity])
        axes[-1, column].set_xlabel(f"{label} [$10^{{{exponent}}}$]")
        axes[-1, column].margins(x=0.12)
        axes[-1, column].xaxis.set_major_locator(MaxNLocator(3))
    bottom_legend(figure, axes)
    fit_thesis_y_label_margins(figure, figure.axes)
    save_validation_figure(figure, FIGURES / "error_indicator")


def transfer_limit_summary(runs, decomposition):
    """Coupling error against the strength-ratio limit, the blending depths and N.

    The strength ratio kappa of every transfer is logged; when it sits at kappa_max in most
    transfers the strength correction is cut back near the wall at every coupling interval, and
    a series in N then changes the transfer count and the action of the limit together.
    """
    members = [
        run
        for run in runs
        if {"E", "KAPPA", "BL"} & set(run["families"].split()) or run["name"] == BASELINE_NAME
    ]
    rows = []
    print("Transfer limits and blending depths, 40 <= t <= 55 s (e_coupling in % of Q_ext)")
    for run in sorted(members, key=lambda r: (r["max_strength_ratio"], r["fvm_substeps"])):
        row = {
            "run": run["name"],
            "N": run["fvm_substeps"],
            "coupling intervals per convective time": DIAMETER / (SPEED * run["particle_time_step"]),
            "kappa_max": run["max_strength_ratio"],
            "delta_0 [cells]": run["blend_start"],
            "delta_1 [cells]": run["blend_end"],
            "kappa median": run["strength ratio median"],
            "fraction of transfers at kappa_max": run["strength-limit fraction"],
            "boundary mismatch mean [U]": run["boundary mismatch mean"],
        }
        row.update(
            {f"e_coupling {q} [%]": decomposition[run["name"]][q]["coupling"] for q in STATISTICS}
        )
        rows.append(row)
        print(
            f"  {run['name']:36s} N {row['N']:2d}  kappa_max {row['kappa_max']:.1f}"
            f"  delta {row['delta_0 [cells]']:g}/{row['delta_1 [cells]']:g}"
            f"  kappa {row['kappa median']:.3f}  at limit {row['fraction of transfers at kappa_max']:.2f}"
            f"  mismatch {row['boundary mismatch mean [U]']:.2e}"
            + "".join(f"  {q[:6]} {row[f'e_coupling {q} [%]']:+6.2f}" for q in STATISTICS)
        )
    return rows


def window_and_reference_checks(runs, references):
    """The 40-55 s window against 55-100 s, and the coarse against the fine-wake reference.

    Runs continued to 100 s (section h) are compared with the reference of their cells and step
    over both windows. The baseline is also compared with the reference that keeps the coupled
    cells over the whole coupled FVM domain (section g), which removes the near-wake resolution
    difference from e_coupling.
    """
    rows = []
    for run in runs:
        reference = references.get(
            (run["reynolds_number"], run["cell_size"], run["fvm_time_step"])
        )
        if reference is None or run["forces"]["time"][-1] < LATE_WINDOW[1] - 0.1:  # within a sample
            continue
        for window in (STATISTICS_WINDOW, LATE_WINDOW):
            errors = errors_against(run["forces"], run["phase"], run["wake"], reference, window)
            rows.append({"run": run["name"], "reference": reference["name"], "window": window, **errors})
    fine = read_fine_wake_reference()
    if fine is not None:
        for run in runs:
            if run["name"] in (BASELINE_NAME, BASELINE_NAME.replace("sub5", "sub2")):
                errors = errors_against(run["forces"], run["phase"], run["wake"], fine)
                rows.append(
                    {"run": run["name"], "reference": fine["name"], "window": STATISTICS_WINDOW, **errors}
                )
    if rows:
        print("Statistics window and near-wake reference: e_coupling [% of the same-window reference]")
        for row in rows:
            print(
                f"  {row['run']:36s} vs {row['reference']:22s} {row['window'][0]:g}-{row['window'][1]:g} s"
                + "".join(f"  {q[:6]} {row[q]:+6.2f}" for q in STATISTICS)
            )
    return [
        {**row, "window": f"{row['window'][0]:g}-{row['window'][1]:g}"} for row in rows
    ]


def factorial_summary(runs, decomposition):
    """Main effects and interactions of the F family once all its corners have run."""
    factors = list(FACTORIAL_LEVELS)
    corners = [run for run in runs if "F" in run["families"].split()]
    for quantity in ERRORS:
        effects = factorial_effects(
            corners, factors, lambda run: decomposition[run["name"]][quantity]["coupling"]
        )
        if effects is None:
            print(f"Factorial F: {len(corners)} of {2 ** len(factors)} corners have run")
            return
        print(
            f"Factorial F, effects on e_coupling {quantity}: "
            + ", ".join(f"{k} {v:+.3f}" for k, v in effects.items())
        )

        # Additivity of each pair of changes, any other factor at its first level
        def error_at(levels):
            corner = next(run for run in corners if all(run[n] == v for n, v in levels.items()))
            return decomposition[corner["name"]][quantity]["coupling"]

        for first, second in combinations(factors, 2):
            low = {name: FACTORIAL_LEVELS[name][0] for name in factors}
            defect = additivity_defect(
                error_at(low),
                error_at({**low, first: FACTORIAL_LEVELS[first][1]}),
                error_at({**low, second: FACTORIAL_LEVELS[second][1]}),
                error_at(
                    {**low, first: FACTORIAL_LEVELS[first][1], second: FACTORIAL_LEVELS[second][1]}
                ),
            )
            print(f"  {first} with {second}: additivity defect {defect:+.3f}%")


def box_figure(families, decomposition):
    """Coupling error of family B against the wake reach and the lateral clearance.

    l_wake/L_f = (x_t - L_f)/L_f uses the u_rms-peak formation length, uncertain by about 0.15 D at
    Re = 150; the v_rms peak, which converges better, gives the second column. If the box scales
    with the wake, both Reynolds numbers fall on one curve.
    """
    scales = wake_scales()
    members = [run for run in families["B"][1] if run["reynolds_number"] in scales]
    if len(members) < 3 or not scales:
        return
    rows = []
    for run in members:
        rows.append(
            {
                "run": run["name"],
                "Re": run["reynolds_number"],
                "box margin": run["box_margin"],
                "box downstream": run["box_downstream"],
                **box_scales(run, scales),
                **{
                    f"e_coupling {q} [%]": decomposition[run["name"]][q]["coupling"] for q in ERRORS
                },
            }
        )
    write_table(TABLES / "error_box.csv", rows)
    print("Family B: coupling error against the box in wake scales")
    for row in rows:
        print(
            f"  {row['run']:42s} l_wake/L_f {row['reach']:5.2f} (v_rms {row['reach, v_rms']:5.2f}) "
            f"l_side/delta_w {row['clearance']:5.2f}"
            + "".join(f"  {q[:6]} {row[f'e_coupling {q} [%]']:+6.2f}" for q in ERRORS)
        )

    # The downstream series keeps the side margin, the side series the downstream boundary
    columns = {
        "reach": (
            r"$\ell_{\rm wake}/L_f$, $L_f$ at" + "\n" + r"$u_\mathrm{rms}$ peak",
            "box margin",
        ),
        "reach, v_rms": (
            r"$\ell_{\rm wake}/L_f$, $L_f$ at" + "\n" + r"$v_\mathrm{rms}$ peak",
            "box margin",
        ),
        "clearance": ("Lateral clearance\n" + r"$\ell_{\rm side}/\delta_\omega$", "box downstream"),
    }
    colors = {100.0: COLORS["green"], 150.0: COLORS["fvm"], 250.0: COLORS["hybrid"]}
    figure, axes = panel_grid(len(ERRORS), len(columns))
    for column, (scale, (label, held)) in enumerate(columns.items()):
        for row, quantity in enumerate(ERRORS):
            axis = axes[row, column]
            for reynolds_number, color in colors.items():
                baseline = (
                    BASELINE["box_margin"] if held == "box margin" else BASELINE["box_downstream"]
                )
                series = sorted(
                    (
                        entry
                        for entry in rows
                        if entry["Re"] == reynolds_number and np.isclose(entry[held], baseline)
                    ),
                    key=lambda entry: entry[scale],
                )
                if not series:
                    continue
                axis.plot(
                    [entry[scale] for entry in series],
                    [entry[f"e_coupling {quantity} [%]"] for entry in series],
                    color=color,
                    marker="o",
                    markersize=MARKER_SIZE,
                    linewidth=LINE_WIDTH,
                    label=f"$Re={reynolds_number:g}$",
                )
            if column == 0:
                axis.set_ylabel(r"$e_{\rm coupling}$" + "\n" + ERRORS[quantity])
        axes[-1, column].set_xlabel(label)
        axes[-1, column].margins(x=0.12)
        axes[-1, column].xaxis.set_major_locator(MaxNLocator(3))
    bottom_legend(figure, axes)
    fit_thesis_y_label_margins(figure, figure.axes)
    save_validation_figure(figure, FIGURES / "error_box_B")


def main():
    references = read_references()
    runs = read_runs(references)
    families = coupled_families(runs)
    print(
        f"Lambda = max|S| dt_VPM with the particle-field max|S| = {PARTICLE_STRAIN_RATE} 1/s at"
        " Re = 150, scaled with sqrt(Re)"
    )

    # Orders and uncertainties of the references, and the zero-cell-size, zero-step values
    exact = exact_values(references)
    for quantity, (value, origin) in exact.items():
        print(f"Q_ext of the {quantity}: {value:.5f} ({origin})")
    reference_fits = reference_orders(references)
    print_orders("Reference families (Eça–Hoekstra least squares)", reference_fits)
    space_time_rows = space_time_orders(references)

    # Total, FVM and coupling error of every coupled run
    levels = [ref for (re, h, dt), ref in references.items() if re == REYNOLDS_NUMBER]
    finest = min(levels, key=lambda ref: (ref["cell_size"], ref["time_step"]))
    decomposition = {run["name"]: decompose(run, references, exact, finest) for run in runs}
    for run in runs:
        run["families"] = " ".join(
            f for f, (_, members) in families.items() if run["name"] in {m["name"] for m in members}
        )
    coupled_fits = coupled_orders(families, decomposition)
    print_orders("Coupled families: statistic (total) and coupling error", coupled_fits)

    print(f"Error decomposition [% of Q_ext]; wake against {finest['name']}")
    decomposition_rows = []
    for run in runs:
        row = {"run": run["name"], "families": run["families"] or "-"}
        row.update(drivers(run))
        row.update({"h/D": run["cell_size"] / DIAMETER, "Re": run["reynolds_number"]})
        for quantity in ERRORS:
            for kind in SERIES:
                row[f"{quantity} {kind}"] = decomposition[run["name"]][quantity][kind]
        decomposition_rows.append(row)
        print(
            f"  {run['name']:42s} {row['families']:6s}"
            + "".join(
                f"  {q[:6]} e {row[f'{q} total']:+6.2f} e_FVM {row[f'{q} FVM']:+6.2f} e_coupling {row[f'{q} coupling']:+6.2f}"
                for q in ERRORS
            )
        )

    driver_rows = driver_analysis(families, decomposition)
    print(
        "Logged interface keys:",
        ", ".join(
            logged_keys(
                CASE_DIR / f"sensitivity/{runs[0]['name']}/solution/coupler_diagnostics.jsonl"
            )
        ),
    )
    indicator_rows, correlations = indicator_analysis(runs, decomposition)
    factorial_summary(runs, decomposition)
    limit_rows = transfer_limit_summary(runs, decomposition)
    check_rows = window_and_reference_checks(runs, references)

    # Tables
    TABLES.mkdir(parents=True, exist_ok=True)
    order_rows = [
        fit_row("reference", f, fit["variable"], q, fit) for (f, q), fit in reference_fits.items()
    ]
    order_rows += [
        fit_row(f"coupled {k}", f, fit["variable"], q, fit)
        for (f, q, k), fit in coupled_fits.items()
    ]
    write_table(TABLES / "error_orders.csv", order_rows)
    if space_time_rows:
        write_table(TABLES / "error_space_time.csv", space_time_rows)
    write_table(TABLES / "error_decomposition.csv", decomposition_rows)
    if driver_rows:
        write_table(TABLES / "error_drivers.csv", driver_rows)
    write_table(TABLES / "error_indicators.csv", indicator_rows)
    if limit_rows:
        write_table(TABLES / "error_transfer_limits.csv", limit_rows)
    if check_rows:
        write_table(TABLES / "error_window_reference.csv", check_rows)

    # Figures
    reference_figure(reference_fits, exact)
    decomposition_figure(families, decomposition, coupled_fits, exact)
    driver_figure(
        families,
        decomposition,
        {(r["quantity"], r["driver"]): r for r in driver_rows},
    )
    indicator_figure(indicator_rows, correlations)
    box_figure(families, decomposition)


if __name__ == "__main__":
    main()
