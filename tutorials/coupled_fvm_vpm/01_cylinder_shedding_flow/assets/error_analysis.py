"""Fit discretization error and coupling-error indicators for the sensitivity study.

Reference refinements estimate the FVM contribution at matched cell size and
time step. The remaining difference from the coupled solution is reported as
the coupling contribution.
"""

from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

from assets.plot_forces import DEVELOPED_SHEDDING, shedding_statistics
from assets.plot_sensitivity import BASELINE, STATISTICS, errors_against, jsonl_records
from setup import CELL_SIZE, DIAMETER, FREESTREAM_VELOCITY, FVM_TIME_STEP
from setup import REYNOLDS_NUMBER

CASE_DIR = Path(__file__).resolve().parents[1]
SPEED = FREESTREAM_VELOCITY[0]
STATISTICS_END = 55.0  # s
MINIMUM_LEVELS = 3
ORDER_RANGE = (0.05, 8.0)
PARTICLE_STRAIN_RATE = 19.4  # 1/s, maximum particle strain rate at the baseline

# Factorial levels: downstream FVM extent [D] and FVM substeps at the baseline inset.
FACTORIAL_LEVELS = {
    "box_downstream": (3.36, 4.4),
    "fvm_substeps": (2, 5),
}

# Fixed orders considered alongside the free-order fit.
FIXED_FITS = {"order 1": (1.0,), "order 2": (2.0,), "orders 1 and 2": (1.0, 2.0)}

# Coupling diagnostics and their physical scales.
INDICATORS = {
    "boundary velocity mismatch, mean": ("fvm_boundary_data", "mean_velocity_mismatch", SPEED),
    "boundary velocity mismatch, max": ("fvm_boundary_data", "maximum_velocity_mismatch", SPEED),
    "outflow mismatch, mean": ("fvm_boundary_data", "mean_outflow_velocity_mismatch", SPEED),
    "first-iteration normal-velocity residual": ("residuals", 0, "normal_velocity_residual", SPEED),
    "first-iteration normal-gradient residual": (
        "residuals",
        0,
        "normal_gradient_residual",
        SPEED / DIAMETER,
    ),
    "last-iteration normal-gradient residual": (
        "residuals",
        -1,
        "normal_gradient_residual",
        SPEED / DIAMETER,
    ),
}


# Least-squares fits of Q_i = Q_ext + sum_k alpha_k x_i^p_k.


def linear_fit(x, phi, orders, weights):
    """Q_ext and the alpha_k of fixed orders, the fitted values and the standard deviation.

    The weighted fit uses w_i proportional to 1/x_i, normalized to sum to one, so that the
    finest levels count most; its standard deviation carries the factor n_g to compare with the
    unweighted one.
    """
    columns = np.column_stack([np.ones_like(x), *(x**order for order in orders)])
    root = np.sqrt(weights)
    coefficients = np.linalg.lstsq(columns * root[:, None], phi * root, rcond=None)[0]
    fitted = columns @ coefficients
    freedom = len(x) - columns.shape[1]
    sigma = np.sqrt(len(x) * np.sum(weights * (phi - fitted) ** 2) / freedom) if freedom else 0.0
    return {"Q_ext": coefficients[0], "alpha": coefficients[1:], "fitted": fitted, "sigma": sigma}


def power_fit(x, phi, weights):
    """Q = Q_ext + alpha x^p with the order free: the order minimizing the residual."""

    def residual(order):
        return np.sum(weights * (phi - linear_fit(x, phi, (order,), weights)["fitted"]) ** 2)

    # Locate the best interval before the bounded minimization.
    orders = np.geomspace(*ORDER_RANGE, 200)
    best = orders[np.argmin([residual(order) for order in orders])]
    low, high = best / 1.05, min(best * 1.05, ORDER_RANGE[1])
    order = minimize_scalar(residual, bounds=(low, high), method="bounded").x
    fit = linear_fit(x, phi, (order,), weights)
    freedom = len(x) - 3
    fit["sigma"] = np.sqrt(len(x) * residual(order) / freedom) if freedom > 0 else 0.0
    fit["order"] = order
    return fit


def best_of_weightings(x, phi, fit_of):
    """The unweighted or the 1/x-weighted fit, whichever has the smaller standard deviation."""
    unweighted = fit_of(np.full(len(x), 1.0 / len(x)))
    weighted = fit_of((1.0 / x) / np.sum(1.0 / x))
    return weighted if weighted["sigma"] < unweighted["sigma"] else unweighted


def convergence_class(x, phi, order):
    """Monotone, oscillatory or anomalous, from the successive differences towards x -> 0.

    Monotone data whose free-order fit cannot converge (order at the lower search bound) are
    anomalous, as are differences that change sign irregularly.
    """
    signs = np.sign(np.diff(phi[np.argsort(x)]))
    if np.all(signs == signs[0]) and signs[0] != 0.0:
        return "monotone" if order > 1.01 * ORDER_RANGE[0] else "anomalous"
    alternating = np.all(signs[1:] * signs[:-1] < 0.0)
    return "oscillatory" if alternating else "anomalous"


def discretization_uncertainty(x, phi):
    """Extrapolated value, observed order, convergence class and uncertainty of every level.

    Monotone data with 0.5 <= p < 2.1 use the free-order fit and a safety factor of 1.25. Other
    orders, and data that do not converge monotonically, use the fixed-order fit with the smallest
    standard deviation and a safety factor of 3; their order is not reported. When the fit
    standard deviation sigma reaches the data range Delta = (phi_max - phi_min)/(n_g - 1), the
    data are not well behaved and U = 3 sigma/Delta (|eps| + sigma + |phi - phi_fit|) instead of
    U = F_s |eps| + sigma + |phi - phi_fit|, eps being the fitted error of each level.
    """
    x, phi = np.asarray(x, float), np.asarray(phi, float)
    free = best_of_weightings(x, phi, lambda weights: power_fit(x, phi, weights))
    category = convergence_class(x, phi, free["order"])

    if category == "monotone" and 0.5 <= free["order"] < 2.1:
        chosen, safety, name = free, 1.25, "free order"
    else:
        # Fits with fewer coefficients than levels; above p = 2.1 the first-order fit is left out
        steep = category == "monotone" and free["order"] >= 2.1
        candidates = {
            fit: best_of_weightings(x, phi, lambda weights: linear_fit(x, phi, orders, weights))
            for fit, orders in FIXED_FITS.items()
            if len(orders) + 1 < len(x) and not (steep and fit == "order 1")
        }
        name = min(candidates, key=lambda fit: candidates[fit]["sigma"])
        chosen, safety = candidates[name], 3.0

    data_range = np.ptp(phi) / (len(x) - 1)
    error = phi - chosen["Q_ext"]
    deviation = np.abs(phi - chosen["fitted"])
    sigma = chosen["sigma"]
    well_behaved = sigma < data_range
    if well_behaved:
        uncertainty = safety * np.abs(error) + sigma + deviation
    else:
        uncertainty = 3.0 * sigma / data_range * (np.abs(error) + sigma + deviation)
    return {
        "x": x,
        "phi": phi,
        "Q_ext": chosen["Q_ext"],
        "order": free["order"] if category == "monotone" else np.nan,
        "class": category,
        "fit": name,
        "safety factor": safety if well_behaved else 3.0 * sigma / data_range,
        "sigma": sigma,
        "uncertainty": uncertainty,  # of each level
        "spread": np.ptp(phi),
        "curve": fit_curve(chosen, free["order"] if name == "free order" else FIXED_FITS[name]),
    }


def fit_curve(fit, orders):
    """The fitted phi(x) of a chosen fit, for plotting."""
    orders = np.atleast_1d(orders)
    return lambda x: (
        fit["Q_ext"]
        + sum(alpha * np.asarray(x) ** order for alpha, order in zip(fit["alpha"], orders))
    )


def space_time_fit(cell_sizes, time_steps, phi):
    """Q = Q_ext + a h^p + b dt^q over every reference level, with the standard errors.

    The orders are found on a grid first, the linear coefficients solved for each pair, then all
    five refined together; the covariance follows from the Jacobian and the residual variance.
    """
    h, dt, phi = (np.asarray(values, float) for values in (cell_sizes, time_steps, phi))

    def model(parameters):
        phi0, a, p, b, q = parameters
        return phi0 + a * h**p + b * dt**q

    orders = np.linspace(0.5, 4.0, 36)
    best = None
    for p in orders:
        for q in orders:
            columns = np.column_stack([np.ones_like(h), h**p, dt**q])
            coefficients, residual = np.linalg.lstsq(columns, phi, rcond=None)[:2]
            value = residual[0] if len(residual) else np.sum((columns @ coefficients - phi) ** 2)
            if best is None or value < best[0]:
                best = (value, [coefficients[0], coefficients[1], p, coefficients[2], q])
    solution = least_squares(lambda parameters: model(parameters) - phi, best[1])
    freedom = len(phi) - 5
    variance = np.sum(solution.fun**2) / freedom
    covariance = variance * np.linalg.inv(solution.jac.T @ solution.jac)
    phi0, a, p, b, q = solution.x
    return {
        "Q_ext": phi0,
        "a": a,
        "p": p,
        "b": b,
        "q": q,
        "standard errors": dict(zip(("Q_ext", "a", "p", "b", "q"), np.sqrt(np.diag(covariance)))),
        "rms residual": np.sqrt(np.mean(solution.fun**2)),
        "freedom": freedom,
    }


# ---------------------------------------------------------------------------------------------
# Run families
# ---------------------------------------------------------------------------------------------


def constant_courant_step(cell_size):
    return cell_size / CELL_SIZE * FVM_TIME_STEP


def reference_families(references):
    """Reference levels at Re = 150 of each family: (x name, [(x, reference), ...])."""
    levels = [ref for (re, h, dt), ref in references.items() if re == REYNOLDS_NUMBER]
    return {
        "C": (
            "cell_size",
            [
                ref
                for ref in levels
                if np.isclose(ref["time_step"], constant_courant_step(ref["cell_size"]))
            ],
        ),
        "X": ("cell_size", [ref for ref in levels if np.isclose(ref["time_step"], 0.004)]),
        "T": ("time_step", [ref for ref in levels if np.isclose(ref["cell_size"], CELL_SIZE)]),
    }


def at_baseline(run, *varied):
    """Whether a run keeps every parameter except ``varied`` at its baseline."""
    return all(
        run[name] == value if isinstance(value, str) else np.isclose(run[name], value)
        for name, value in BASELINE.items()
        if name not in varied
    )


def coupled_families(runs):
    """Coupled runs of each family of the plan, and the variable each refines or varies.

    Family C counts the interface depths in its own cells, as a user refining the whole method
    would; family X keeps the interface in place (depths in 0.04 m cells) so that only the
    discretization is refined.
    """
    spatial = ("cell_size", "fvm_time_step", "fvm_substeps", "particle_spacing")
    return {
        "C": (
            "cell_size",
            [
                run
                for run in runs
                if at_baseline(run, *spatial, "interface_cell_size")
                and np.isclose(run["interface_cell_size"], run["cell_size"])
                and run["fvm_substeps"] == 2
                and np.isclose(run["fvm_time_step"], constant_courant_step(run["cell_size"]))
                and np.isclose(run["particle_spacing"], run["cell_size"])
            ],
        ),
        "X": (
            "cell_size",
            [
                run
                for run in runs
                if at_baseline(run, *spatial)
                and run["fvm_substeps"] == 4
                and np.isclose(run["fvm_time_step"], 0.004)
                and np.isclose(run["particle_spacing"], run["cell_size"])
            ],
        ),
        "T2": (
            "fvm_time_step",
            [
                run
                for run in runs
                if at_baseline(run, "fvm_time_step", "fvm_substeps") and run["fvm_substeps"] == 2
            ],
        ),
        "P": (
            "fvm_time_step",
            [
                run
                for run in runs
                if at_baseline(run, "fvm_time_step", "fvm_substeps")
                and np.isclose(run["particle_time_step"], 0.04)
            ],
        ),
        "E": (
            "fvm_substeps",
            [
                run
                for run in runs
                if at_baseline(run, "fvm_substeps") and np.isclose(run["fvm_time_step"], 0.008)
            ],
        ),
        # A: cells refined at a coupling interval admissible on the finest cells (0.032 s), and XP:
        # the same at the baseline interval 0.04 s; both with the interface fixed in 0.04 m cells
        **{
            name: (
                "cell_size",
                [
                    run
                    for run in runs
                    if at_baseline(run, *spatial, "interface_cell_size")
                    and np.isclose(run["particle_time_step"], particle_step)
                    and np.isclose(run["fvm_time_step"], 0.004)
                    and np.isclose(run["interface_cell_size"], CELL_SIZE)
                    and np.isclose(run["particle_spacing"], run["cell_size"])
                ],
            )
            for name, particle_step in (("A", 0.032), ("XP", 0.04))
        },
        # Strength-ratio limit kappa_max at each number of FVM substeps
        "KAPPA": (
            "max_strength_ratio",
            [
                run
                for run in runs
                if at_baseline(run, "max_strength_ratio", "fvm_substeps")
                and np.isclose(run["fvm_time_step"], FVM_TIME_STEP)
            ],
        ),
        # Blending depths delta_0 and delta_1 below the transfer surface
        "BL": ("blend_end", [run for run in runs if at_baseline(run, "blend_start", "blend_end")]),
        "K": (
            "coupling_iterations",
            [run for run in runs if at_baseline(run, "coupling_iterations")],
        ),
        "S": (
            "particle_spacing",
            [run for run in runs if at_baseline(run, "fvm_substeps", "particle_spacing")],
        ),
        "R": ("reynolds_number", [run for run in runs if at_baseline(run, "reynolds_number")]),
        # Downstream outer FVM boundary and side margin of the FVM domain, at each Reynolds number
        "B": (
            "box_downstream",
            [
                run
                for run in runs
                if at_baseline(run, "box_margin", "box_downstream", "reynolds_number")
            ],
        ),
        "F": (
            "box_downstream",
            [
                run
                for run in runs
                if at_baseline(run, *FACTORIAL_LEVELS)
                and all(run[name] in levels for name, levels in FACTORIAL_LEVELS.items())
            ],
        ),
    }


# ---------------------------------------------------------------------------------------------
# Error decomposition
# ---------------------------------------------------------------------------------------------


def statistics_of(forces):
    """Developed-shedding statistics over the common window of the study."""
    return shedding_statistics(forces[forces["time"] <= STATISTICS_END])


def exact_values(references):
    """Q_ext of each statistic: the space-time fit when it determines Q_ext more tightly than the
    family-C uncertainty, else family C.

    A statistic that barely changes with the cells or the step, such as the mean drag, leaves the
    orders of the space-time fit undetermined and its Q_ext meaningless.
    """
    levels = [ref for (re, h, dt), ref in references.items() if re == REYNOLDS_NUMBER]
    family = reference_families(references)["C"][1]
    exact = {}
    for quantity in STATISTICS:
        fit = discretization_uncertainty(
            [ref["cell_size"] / DIAMETER for ref in family],
            [statistics_of(ref["forces"])[quantity] for ref in family],
        )
        exact[quantity] = (fit["Q_ext"], "family C")
        if len(levels) > 5:
            combined = space_time_fit(
                [ref["cell_size"] for ref in levels],
                [ref["time_step"] for ref in levels],
                [statistics_of(ref["forces"])[quantity] for ref in levels],
            )
            if combined["standard errors"]["Q_ext"] < np.min(fit["uncertainty"]):
                exact[quantity] = (combined["Q_ext"], "space-time fit")
    return exact


def decompose(run, references, exact, finest):
    """Total, FVM and coupling errors of one coupled run, in % of Q_ext.

    The wake velocity has no scalar to extrapolate: its total and FVM errors are distances to the
    finest reference, the coupling error the distance to the reference on the same cells and step.
    """
    reference = references.get((run["reynolds_number"], run["cell_size"], run["fvm_time_step"]))
    measured = statistics_of(run["forces"])
    expected = (
        statistics_of(reference["forces"]) if reference else dict.fromkeys(STATISTICS, np.nan)
    )
    errors = {}
    for quantity in STATISTICS:
        # Other Reynolds numbers have no extrapolated value: their errors scale with the reference
        if run["reynolds_number"] == REYNOLDS_NUMBER:
            phi0 = exact[quantity][0]
            scale = phi0
        else:
            phi0, scale = np.nan, expected[quantity]
        errors[quantity] = {
            "total": 100.0 * (measured[quantity] - phi0) / scale,
            "FVM": 100.0 * (expected[quantity] - phi0) / scale,
            "coupling": 100.0 * (measured[quantity] - expected[quantity]) / scale,
        }

    def wake_distance(target):
        return errors_against(run["forces"], run["phase"], run["wake"], target)["wake velocity"]

    errors["wake velocity"] = {
        "total": wake_distance(finest) if run["reynolds_number"] == REYNOLDS_NUMBER else np.nan,
        "FVM": errors_against(reference["forces"], reference["phase"], reference["wake"], finest)[
            "wake velocity"
        ]
        if reference and run["reynolds_number"] == REYNOLDS_NUMBER
        else np.nan,
        "coupling": wake_distance(reference) if reference else np.nan,
    }
    return errors


# ---------------------------------------------------------------------------------------------
# Drivers of the coupling error
# ---------------------------------------------------------------------------------------------


def drivers(run):
    """Possible drivers of the coupling error, each in flow scales."""
    particle_step = run["particle_time_step"]
    strain_rate = PARTICLE_STRAIN_RATE * np.sqrt(run["reynolds_number"] / REYNOLDS_NUMBER)
    return {
        "particle step": SPEED * particle_step / DIAMETER,
        "Lagrangian CFL number": strain_rate * particle_step,
        "coupling intervals per convective time": DIAMETER / (SPEED * particle_step),
        "FVM substeps per coupling interval": float(run["fvm_substeps"]),
        "FVM step": SPEED * run["fvm_time_step"] / DIAMETER,
        "particle Courant number": run["particle_courant"],
        "spacing ratio": run["particle_spacing"] / run["cell_size"],
        "coupling iterations": float(run["coupling_iterations"] or run["mean coupling iterations"]),
    }


def wake_scales():
    """Formation lengths and shear-layer thickness [D] of the references on the baseline cells.

    Keyed by Reynolds number, from the flow-scale table that flow_scales.py writes first in
    allplot.sh.
    L_f is the centreline peak of u_rms, L_f,v that of v_rms, delta_w the vorticity thickness.
    """
    table = np.atleast_1d(
        np.genfromtxt(
            CASE_DIR / "figures/auxiliary/flow_scales.csv", delimiter=",", names=True, dtype=None
        )
    )
    return {
        row["reynolds_number"]: {
            "L_f": row["formation_length"],
            "L_f,v": row["transverse_rms_peak"],
            "delta_w": row["vorticity_thickness"],
        }
        for row in table[np.isclose(table["cell_size"], CELL_SIZE)]
    }


def box_scales(run, scales):
    """The FVM domain in wake scales: wake reach l_wake/L_f and lateral clearance l_side/delta_w.

    l_wake/L_f = (x_t - L_f)/L_f, with x_t the downstream transfer surface, is how far past the
    formation point the FVM holds the wake, given for both formation lengths;
    l_side/delta_w = (m - w - D/2)/delta_w is the clearance of the shear layer from the side faces
    of the replacement region, a depth w inside the outer FVM boundary.
    """
    scale = scales[run["reynolds_number"]]
    interface_cell = run["interface_cell_size"] / DIAMETER
    transfer_face = run["box_downstream"] - run["transfer_inset"] * interface_cell
    # The replacement region starts the inset plus the blend end depth inside the outer FVM boundary
    bands = (run["transfer_inset"] + run["blend_end"]) * interface_cell
    return {
        "reach": (transfer_face - scale["L_f"]) / scale["L_f"],
        "reach, v_rms": (transfer_face - scale["L_f,v"]) / scale["L_f,v"],
        "clearance": (run["box_margin"] - bands - 0.5) / scale["delta_w"],
    }


def driver_fit(x, error):
    """e = c_0 + a x^q with q free in [-3, 3]: the collapse of the error on one driver.

    Returns c_0, a, q, the coefficient of determination R^2 and the rms residual.
    """
    x, error = np.asarray(x, float), np.asarray(error, float)

    def residual(order):
        columns = np.column_stack([np.ones_like(x), x**order])
        coefficients = np.linalg.lstsq(columns, error, rcond=None)[0]
        return np.sum((columns @ coefficients - error) ** 2), coefficients

    orders = [order for order in np.linspace(-3.0, 3.0, 241) if abs(order) > 1.0e-6]
    order = min(orders, key=lambda order: residual(order)[0])
    value, (offset, amplitude) = residual(order)
    total = np.sum((error - error.mean()) ** 2)
    return {
        "c0": offset,
        "a": amplitude,
        "q": order,
        "R2": 1.0 - value / total if total > 0.0 else np.nan,
        "rms": np.sqrt(value / len(x)),
    }


def term_model(columns, error):
    """Linear least squares of the error on named terms, with standard errors of the coefficients.

    ``columns`` maps term names to their values at each run; a constant term is added.
    """
    matrix = np.column_stack([np.ones(len(error)), *columns.values()])
    coefficients, *_ = np.linalg.lstsq(matrix, error, rcond=None)
    residual = error - matrix @ coefficients
    freedom = len(error) - matrix.shape[1]
    covariance = np.sum(residual**2) / freedom * np.linalg.inv(matrix.T @ matrix)
    total = np.sum((error - error.mean()) ** 2)
    names = ["constant", *columns]
    return {
        "coefficients": dict(zip(names, coefficients)),
        "standard errors": dict(zip(names, np.sqrt(np.diag(covariance)))),
        "R2": 1.0 - np.sum(residual**2) / total,
        "freedom": freedom,
    }


# ---------------------------------------------------------------------------------------------
# In-run indicator
# ---------------------------------------------------------------------------------------------


def interface_indicators(path):
    """Developed-flow means of the logged boundary-data mismatch, in flow scales.

    Coupling steps with a single coupling iteration log no residuals; their residual indicators
    are NaN.
    """
    records = [record for record in jsonl_records(path) if record["time"] >= DEVELOPED_SHEDDING]
    means = {}
    for name, (*keys, scale) in INDICATORS.items():
        if keys[0] == "residuals":
            values = [
                record["coupling_iteration"]["residuals"][keys[1]][keys[2]]
                for record in records
                if record.get("coupling_iteration", {}).get("residuals")
            ]
        else:
            values = [record[keys[0]][keys[1]] for record in records if keys[0] in record]
        means[name] = np.mean(values) / scale if values else np.nan
    return means


def logged_keys(path):
    """Every mismatch, residual and error key logged in the last coupling-step record."""
    keys = []

    def walk(record, prefix):
        for key, value in record.items():
            if isinstance(value, dict):
                walk(value, f"{prefix}{key}.")
            elif any(word in key for word in ("mismatch", "residual", "error")):
                keys.append(f"{prefix}{key}")
            elif key == "residuals" and value:
                keys.extend(f"{prefix}residuals[].{name}" for name in value[-1])

    walk(jsonl_records(path)[-1], "")
    return keys


# ---------------------------------------------------------------------------------------------
# Two-level factorial designs
# ---------------------------------------------------------------------------------------------


def factorial_effects(runs, factors, response):
    """Main effects and interactions of a full 2^k design, one run per corner.

    ``factors`` names the run parameters at two levels each; ``response`` gives a run's error.
    The effect of a set of factors is the mean response where the product of their coded levels
    (-1 low, +1 high) is +1 minus the mean where it is -1. Returns None until every corner has run.
    """
    levels = {factor: sorted({run[factor] for run in runs}) for factor in factors}
    if any(len(values) != 2 for values in levels.values()) or len(runs) != 2 ** len(factors):
        return None
    coded = np.array(
        [[1.0 if run[factor] == levels[factor][1] else -1.0 for factor in factors] for run in runs]
    )
    values = np.array([response(run) for run in runs])
    effects = {}
    for size in range(1, len(factors) + 1):
        for subset in combinations(range(len(factors)), size):
            sign = np.prod(coded[:, subset], axis=1)
            effects[" x ".join(factors[i] for i in subset)] = 2.0 * np.mean(sign * values)
    return effects


def additivity_defect(base, first, second, both):
    """e(A, B) - [e(A) + e(B) - e(0)]: zero when the two changes add their errors independently."""
    return both - (first + second - base)
