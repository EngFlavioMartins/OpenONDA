#!/usr/bin/env python3
r"""Gate B.0: exact-solution verification of the research spectral LES path.

The test fields are a shear eigenmode and an ABC Beltrami field.  Both satisfy

    u(t) = u(0) exp(-nu k^2 t),

and have identically zero exact SGS vorticity torque.  This script checks the
spectral operators, Heun temporal order, closure null response, energy budget,
and the analytic initial identities of the filtered Taylor--Green vortex.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/openonda_stage4b0_matplotlib")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_4a_formulation import SpectralGrid, model_torques, norm  # noqa: E402
from stage_4b_spectral_pilot import (  # noqa: E402
    VorticitySolver,
    coarse_reference,
    taylor_green,
)

COLORS = {"theory": "#20252a", "no_sgs": "#8a99a8", "sensed": "#286f9b"}
LABELS = {"theory": "Theory", "no_sgs": "No SGS", "sensed": "Sensed DIAD"}


def coordinates(grid: SpectralGrid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = 2.0 * np.pi * np.arange(grid.n) / grid.n
    return np.meshgrid(x, x, x, indexing="ij")


def shear_velocity(grid: SpectralGrid, wave: int = 2) -> np.ndarray:
    xx, yy, _ = coordinates(grid)
    return np.array((np.sin(wave * yy), np.zeros_like(xx), np.zeros_like(xx)))


def abc_velocity(grid: SpectralGrid) -> np.ndarray:
    xx, yy, zz = coordinates(grid)
    return np.array(
        (
            np.sin(zz) + np.cos(yy),
            np.sin(xx) + np.cos(zz),
            np.sin(yy) + np.cos(xx),
        )
    )


def energy(solver: VorticitySolver, vorticity: np.ndarray) -> float:
    velocity = solver.velocity(vorticity)
    return 0.5 * float(np.mean(np.sum(velocity * velocity, axis=0)))


def enstrophy(vorticity: np.ndarray) -> float:
    return 0.5 * float(np.mean(np.sum(vorticity * vorticity, axis=0)))


def divergence_error(solver: VorticitySolver, vorticity: np.ndarray) -> float:
    divergence = sum(solver.grid.derivative(vorticity[i], i) for i in range(3))
    return norm(divergence) / max(norm(vorticity), np.finfo(float).tiny)


def relative_error(numerical: np.ndarray, exact: np.ndarray) -> float:
    return norm(numerical - exact) / max(norm(exact), np.finfo(float).tiny)


def initial_state(
    solver: VorticitySolver,
    velocity_function,
    gaussian_delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    filtered_velocity = solver.grid.gaussian(velocity_function(solver.grid), gaussian_delta)
    vorticity = solver.project(solver.grid.curl(filtered_velocity))
    return filtered_velocity, vorticity


def integrate_case(
    solver: VorticitySolver,
    initial_vorticity: np.ndarray,
    eigenvalue: float,
    gaussian_delta: float,
    model: str,
    dt: float,
    end_time: float,
) -> dict[str, object]:
    steps = int(round(end_time / dt))
    if abs(steps * dt - end_time) > 1.0e-13:
        raise ValueError("end_time must be an integer multiple of dt")
    state = initial_vorticity.copy()
    e0 = energy(solver, state)
    z0 = enstrophy(state)
    records: list[dict[str, float]] = []
    for step in range(steps + 1):
        time = step * dt
        exact_factor = np.exp(-solver.viscosity * eigenvalue * time)
        exact_state = initial_vorticity * exact_factor
        records.append(
            {
                "time": time,
                "energy_ratio": energy(solver, state) / e0,
                "enstrophy_ratio": enstrophy(state) / z0,
                "theory_ratio": exact_factor**2,
                "field_relative_error": relative_error(state, exact_state),
                "divergence_relative": divergence_error(solver, state),
            }
        )
        if step < steps:
            state = solver.heun_step(state, dt, model, gaussian_delta)
    time = np.asarray([record["time"] for record in records])
    viscous_power = np.asarray(
        [-2.0 * solver.viscosity * z0 * record["enstrophy_ratio"] for record in records]
    )
    predicted_energy = e0 + float(np.trapezoid(viscous_power, time))
    energy_change = max(abs(records[-1]["energy_ratio"] * e0 - e0), 1.0e-15)
    budget_residual = abs(records[-1]["energy_ratio"] * e0 - predicted_energy) / energy_change
    return {
        "dt": dt,
        "model": model,
        "final_field_relative_error": records[-1]["field_relative_error"],
        "energy_budget_relative_residual": budget_residual,
        "max_divergence_relative": max(record["divergence_relative"] for record in records),
        "records": records,
    }


def convergence_order(dt: list[float], error: list[float]) -> float:
    return float(np.polyfit(np.log(dt), np.log(error), 1)[0])


def exact_case_checks(
    name: str,
    velocity_function,
    eigenvalue: float,
    viscosity: float,
    n: int,
    dt_values: list[float],
    end_time: float,
) -> dict[str, object]:
    solver = VorticitySolver(n, viscosity)
    h = 2.0 * np.pi / n
    gaussian_delta = 2.0 * h / np.sqrt(6.0)
    filtered_velocity, initial_vorticity = initial_state(solver, velocity_function, gaussian_delta)
    rhs = solver.rhs(initial_vorticity, "no_sgs", gaussian_delta)
    operator_error = relative_error(rhs, -viscosity * eigenvalue * initial_vorticity)
    scale = max(
        norm(initial_vorticity) * norm(filtered_velocity) * np.sqrt(eigenvalue),
        np.finfo(float).tiny,
    )
    torques, torque_diagnostics = model_torques(solver.grid, filtered_velocity, gaussian_delta)
    torque_null = {model: norm(torque) / scale for model, torque in torques.items()}
    runs: dict[str, list[dict[str, object]]] = {}
    for model in ("no_sgs", "sensed"):
        runs[model] = [
            integrate_case(
                solver,
                initial_vorticity,
                eigenvalue,
                gaussian_delta,
                model,
                dt,
                end_time,
            )
            for dt in dt_values
        ]
    orders = {
        model: convergence_order(
            dt_values,
            [float(run["final_field_relative_error"]) for run in model_runs],
        )
        for model, model_runs in runs.items()
    }
    return {
        "name": name,
        "n": n,
        "viscosity": viscosity,
        "eigenvalue": eigenvalue,
        "gaussian_delta": gaussian_delta,
        "operator_relative_error": operator_error,
        "initial_divergence_relative": divergence_error(solver, initial_vorticity),
        "sgs_torque_nondimensional": torque_null,
        "diad": torque_diagnostics,
        "temporal_order": orders,
        "runs": runs,
    }


def spatial_operator_screen(
    viscosity: float, resolutions: list[int]
) -> dict[str, dict[str, list[float]]]:
    result: dict[str, dict[str, list[float]]] = {}
    cases = (
        ("shear", shear_velocity, 4.0),
        ("ABC", abc_velocity, 1.0),
    )
    for name, function, eigenvalue in cases:
        errors = []
        torque_errors = []
        for n in resolutions:
            solver = VorticitySolver(n, viscosity)
            delta = 2.0 * (2.0 * np.pi / n) / np.sqrt(6.0)
            velocity, vorticity = initial_state(solver, function, delta)
            rhs = solver.rhs(vorticity, "no_sgs", delta)
            errors.append(relative_error(rhs, -viscosity * eigenvalue * vorticity))
            scale = max(
                norm(vorticity) * norm(velocity) * np.sqrt(eigenvalue),
                np.finfo(float).tiny,
            )
            sensed = model_torques(solver.grid, velocity, delta)[0]["sensed"]
            torque_errors.append(norm(sensed) / scale)
        result[name] = {
            "n": resolutions,
            "operator_relative_error": errors,
            "sensed_torque_nondimensional": torque_errors,
        }
    return result


def taylor_green_checks(viscosity: float, n: int) -> dict[str, float]:
    solver = VorticitySolver(n, viscosity)
    h = 2.0 * np.pi / n
    delta = 2.0 * h / np.sqrt(6.0)
    unfiltered = taylor_green(solver.grid)
    filtered = solver.grid.gaussian(unfiltered, delta)
    vorticity = solver.project(solver.grid.curl(filtered))
    numerical_energy = energy(solver, vorticity)
    numerical_enstrophy = enstrophy(vorticity)
    transfer = np.exp(-(delta**2) * 3.0 / 4.0)
    theoretical_energy = 0.125 * transfer**2
    theoretical_enstrophy = 3.0 * theoretical_energy
    rhs_w = solver.rhs(vorticity, "no_sgs", delta)
    rhs_u = solver.velocity(rhs_w)
    numerical_energy_rate = float(np.mean(np.sum(filtered * rhs_u, axis=0)))
    theoretical_energy_rate = -2.0 * viscosity * theoretical_enstrophy

    dns = VorticitySolver(2 * n, viscosity)
    dns_vorticity = dns.project(dns.grid.curl(taylor_green(dns.grid)))
    paired = coarse_reference(dns, dns_vorticity, n, delta)
    pairing_error = relative_error(paired, vorticity)
    return {
        "energy_theory": theoretical_energy,
        "energy_numerical": numerical_energy,
        "energy_relative_error": abs(numerical_energy / theoretical_energy - 1.0),
        "enstrophy_theory": theoretical_enstrophy,
        "enstrophy_numerical": numerical_enstrophy,
        "enstrophy_relative_error": abs(numerical_enstrophy / theoretical_enstrophy - 1.0),
        "energy_rate_theory": theoretical_energy_rate,
        "energy_rate_numerical": numerical_energy_rate,
        "energy_rate_relative_error": abs(numerical_energy_rate / theoretical_energy_rate - 1.0),
        "dns_les_pairing_relative_error": pairing_error,
    }


def gate_decision(result: dict[str, object]) -> tuple[bool, dict[str, float]]:
    exact_cases = result["exact_cases"]
    finest_errors = [
        float(case["runs"][model][-1]["final_field_relative_error"])
        for case in exact_cases
        for model in ("no_sgs", "sensed")
    ]
    orders = [
        float(case["temporal_order"][model])
        for case in exact_cases
        for model in ("no_sgs", "sensed")
    ]
    budgets = [
        float(case["runs"][model][-1]["energy_budget_relative_residual"])
        for case in exact_cases
        for model in ("no_sgs", "sensed")
    ]
    spatial = result["spatial_screen"]
    operator_errors = [
        value for case in spatial.values() for value in case["operator_relative_error"]
    ]
    torque_errors = [
        value for case in spatial.values() for value in case["sensed_torque_nondimensional"]
    ]
    divergences = [
        float(run["max_divergence_relative"])
        for case in exact_cases
        for model in ("no_sgs", "sensed")
        for run in case["runs"][model]
    ]
    tgv = result["taylor_green_initial"]
    observed = {
        "minimum_temporal_order": min(orders),
        "maximum_finest_field_error": max(finest_errors),
        "maximum_budget_residual": max(budgets),
        "maximum_operator_error": max(operator_errors),
        "maximum_null_torque": max(torque_errors),
        "maximum_divergence": max(divergences),
        "maximum_tgv_identity_error": max(
            tgv["energy_relative_error"],
            tgv["enstrophy_relative_error"],
            tgv["energy_rate_relative_error"],
            tgv["dns_les_pairing_relative_error"],
        ),
    }
    passed = (
        observed["minimum_temporal_order"] >= 1.95
        and observed["maximum_finest_field_error"] <= 5.0e-5
        and observed["maximum_budget_residual"] <= 5.0e-4
        and observed["maximum_operator_error"] <= 1.0e-11
        and observed["maximum_null_torque"] <= 1.0e-10
        and observed["maximum_divergence"] <= 1.0e-12
        and observed["maximum_tgv_identity_error"] <= 1.0e-12
    )
    return bool(passed), observed


def plot_exact_overlays(result: dict[str, object], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2), constrained_layout=True)
    for row, case in enumerate(result["exact_cases"]):
        for column, (quantity, title) in enumerate(
            (("energy_ratio", r"$E(t)/E_0$"), ("enstrophy_ratio", r"$Z(t)/Z_0$"))
        ):
            axis = axes[row, column]
            theory_records = case["runs"]["no_sgs"][-1]["records"]
            axis.plot(
                [record["time"] for record in theory_records],
                [record["theory_ratio"] for record in theory_records],
                color=COLORS["theory"],
                linestyle="--",
                linewidth=2.2,
                label=LABELS["theory"],
            )
            for model, style in (("no_sgs", "o"), ("sensed", "s")):
                records = case["runs"][model][-1]["records"]
                axis.plot(
                    [record["time"] for record in records],
                    [record[quantity] for record in records],
                    color=COLORS[model],
                    marker=style,
                    markevery=max(1, len(records) // 8),
                    markersize=3.5,
                    linewidth=1.3,
                    label=LABELS[model],
                )
            axis.set_title(f"{case['name']}: {title}", fontsize=11)
            axis.set_xlabel(r"$t$")
            axis.grid(color="#d8dde2", linewidth=0.7)
            axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(r"Exact viscous decay: numerical--theoretical overlay", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_temporal_convergence(result: dict[str, object], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), constrained_layout=True)
    markers = {"no_sgs": "o", "sensed": "s"}
    for axis, case in zip(axes, result["exact_cases"], strict=True):
        for model in ("no_sgs", "sensed"):
            runs = case["runs"][model]
            dt = np.asarray([run["dt"] for run in runs])
            error = np.asarray([run["final_field_relative_error"] for run in runs])
            axis.loglog(
                dt,
                error,
                color=COLORS[model],
                marker=markers[model],
                linewidth=1.5,
                label=f"{LABELS[model]} ($p={case['temporal_order'][model]:.3f}$)",
            )
        anchor_dt = np.asarray([case["runs"]["no_sgs"][0]["dt"], case["runs"]["no_sgs"][-1]["dt"]])
        anchor_error = float(case["runs"]["no_sgs"][0]["final_field_relative_error"])
        axis.loglog(
            anchor_dt,
            anchor_error * (anchor_dt / anchor_dt[0]) ** 2,
            color=COLORS["theory"],
            linestyle="--",
            linewidth=1.2,
            label=r"reference $\mathcal{O}(\Delta t^2)$",
        )
        axis.set_title(case["name"])
        axis.set_xlabel(r"$\Delta t$")
        axis.set_ylabel(r"$\|\omega_h-\omega\|_2/\|\omega\|_2$")
        axis.grid(color="#d8dde2", linewidth=0.7, which="both")
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Temporal convergence against exact solutions", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_tgv_references(result: dict[str, object], output: Path) -> None:
    tgv = result["taylor_green_initial"]
    labels = (r"$E(0)$", r"$Z(0)$", r"$-\dot E(0)$")
    deviations = (
        1.0e15 * (tgv["energy_numerical"] / tgv["energy_theory"] - 1.0),
        1.0e15 * (tgv["enstrophy_numerical"] / tgv["enstrophy_theory"] - 1.0),
        1.0e15 * (tgv["energy_rate_numerical"] / tgv["energy_rate_theory"] - 1.0),
    )
    fig, axis = plt.subplots(figsize=(7.6, 4.3), constrained_layout=True)
    positions = np.arange(len(labels))
    axis.axhline(0.0, color=COLORS["theory"], linestyle="--", linewidth=1.7, label="Theory")
    axis.scatter(
        positions,
        deviations,
        color=COLORS["sensed"],
        edgecolor="#174864",
        marker="o",
        s=70,
        zorder=3,
        label="Spectral evaluation",
    )
    axis.set_xticks(positions, labels)
    span = max(2.0, 1.4 * max(abs(value) for value in deviations))
    axis.set_ylim(-span, span)
    axis.set_ylabel(r"$(\mathrm{numerical}/\mathrm{theory}-1)\times10^{15}$")
    axis.set_title("Filtered Taylor--Green initial identities")
    axis.grid(axis="y", color="#d8dde2", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_gate_residuals(result: dict[str, object], output: Path) -> None:
    observed = result["gate_observed"]
    tolerances = {
        "maximum_finest_field_error": 5.0e-5,
        "maximum_budget_residual": 5.0e-4,
        "maximum_operator_error": 1.0e-11,
        "maximum_null_torque": 1.0e-10,
        "maximum_divergence": 1.0e-12,
        "maximum_tgv_identity_error": 1.0e-12,
    }
    labels = [key.replace("maximum_", "").replace("_", " ") for key in tolerances]
    normalized = [observed[key] / tolerance for key, tolerance in tolerances.items()]
    fig, axis = plt.subplots(figsize=(8.6, 4.7), constrained_layout=True)
    positions = np.arange(len(labels))
    axis.barh(positions, normalized, color=COLORS["sensed"], edgecolor="#174864")
    axis.axvline(1.0, color="#d9973b", linestyle="--", linewidth=1.5, label="Gate limit")
    axis.set_xscale("log")
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel("observed residual / gate limit")
    axis.set_title("Gate B.0 normalized residuals")
    axis.grid(axis="x", color="#d8dde2", linewidth=0.7, which="both")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--viscosity", type=float, default=0.2)
    parser.add_argument("--end-time", type=float, default=1.0)
    args = parser.parse_args()
    dt_values = [0.1, 0.05, 0.025, 0.0125]
    result: dict[str, object] = {
        "gate": "B.0 exact-solution verification",
        "equations": {
            "exact_field": "u(t) = u(0) exp(-nu k^2 t)",
            "energy": "E(t) = E(0) exp(-2 nu k^2 t)",
            "enstrophy": "Z(t) = Z(0) exp(-2 nu k^2 t)",
            "exact_sgs_torque": "g_SGS = 0",
            "tgv_initial": "E(0)=G_delta(sqrt(3))^2/8; Z(0)=3E(0); dE/dt=-2nu Z(0)",
        },
        "exact_cases": [
            exact_case_checks(
                "Shear eigenmode",
                shear_velocity,
                4.0,
                args.viscosity,
                args.n,
                dt_values,
                args.end_time,
            ),
            exact_case_checks(
                "ABC Beltrami field",
                abc_velocity,
                1.0,
                args.viscosity,
                args.n,
                dt_values,
                args.end_time,
            ),
        ],
        "spatial_screen": spatial_operator_screen(args.viscosity, [12, 16, 24, 32]),
        "taylor_green_initial": taylor_green_checks(args.viscosity, args.n),
    }
    passed, observed = gate_decision(result)
    result["gate_pass"] = passed
    result["gate_observed"] = observed
    result["gate_requirements"] = {
        "minimum_temporal_order": 1.95,
        "maximum_finest_field_error": 5.0e-5,
        "maximum_budget_residual": 5.0e-4,
        "maximum_operator_error": 1.0e-11,
        "maximum_null_torque": 1.0e-10,
        "maximum_divergence": 1.0e-12,
        "maximum_tgv_identity_error": 1.0e-12,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    plot_exact_overlays(result, args.figure_dir / "stage_4b0_exact_overlays.png")
    plot_temporal_convergence(result, args.figure_dir / "stage_4b0_temporal_convergence.png")
    plot_tgv_references(result, args.figure_dir / "stage_4b0_tgv_references.png")
    plot_gate_residuals(result, args.figure_dir / "stage_4b0_gate_residuals.png")
    if not passed:
        raise SystemExit("GATE B.0 FAIL")


if __name__ == "__main__":
    main()
