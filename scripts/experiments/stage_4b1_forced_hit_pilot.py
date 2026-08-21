#!/usr/bin/env python3
"""Gate B.1b pilot: transient forced homogeneous-turbulence comparison.

This reduced 48^3/24^3 calculation is a screen for forcing/filter consistency,
energy budgets, stability, and model separation.  It is not a statistically
stationary or publication-resolution Gate-B result.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/openonda_stage4b1_hit_matplotlib")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_4b1_forcing_verification import (  # noqa: E402
    ForcingHistory,
    cumulative_trapezoid,
    forced_heun_step,
    random_isotropic_velocity,
)
from stage_4b_spectral_pilot import (  # noqa: E402
    COLORS,
    LABELS,
    MODELS,
    VorticitySolver,
    coarse_reference,
    diagnostics,
    energy_spectrum,
)

DISPLAY_LABELS = {**LABELS, "filtered_dns": "Filtered reference"}
BLUE = COLORS["sensed"]


def relative_error(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right) / np.linalg.norm(right))


def forcing_power(
    solver: VorticitySolver,
    vorticity: np.ndarray,
    acceleration: np.ndarray,
) -> float:
    velocity = solver.velocity(vorticity)
    return float(np.mean(np.sum(velocity * acceleration, axis=0)))


def reynolds_lambda(energy: float, enstrophy: float, viscosity: float) -> float:
    component_variance = 2.0 * energy / 3.0
    dissipation = 2.0 * viscosity * enstrophy
    if dissipation <= np.finfo(float).tiny:
        return 0.0
    microscale = np.sqrt(15.0 * viscosity * component_variance / dissipation)
    return float(np.sqrt(component_variance) * microscale / viscosity)


def integral_scale(spectrum: np.ndarray, energy: float) -> float:
    component_variance = 2.0 * energy / 3.0
    wave = np.arange(len(spectrum), dtype=float)
    positive = wave > 0.0
    if component_variance <= np.finfo(float).tiny:
        return 0.0
    return float(np.pi * np.sum(spectrum[positive] / wave[positive]) / (2.0 * component_variance))


def add_reference_diagnostics(
    solver: VorticitySolver,
    vorticity: np.ndarray,
    acceleration: np.ndarray,
    gaussian_delta: float,
    time: float,
) -> dict[str, float]:
    record = diagnostics(solver, vorticity, "no_sgs", gaussian_delta)
    record.update(
        {
            "time": time,
            "forcing_power": forcing_power(solver, vorticity, acceleration),
            "relative_vorticity_error": 0.0,
            "spectral_relative_l2": 0.0,
            "reynolds_lambda": reynolds_lambda(
                record["energy"], record["enstrophy"], solver.viscosity
            ),
        }
    )
    return record


def add_model_diagnostics(
    solver: VorticitySolver,
    vorticity: np.ndarray,
    reference: np.ndarray,
    reference_spectrum: np.ndarray,
    acceleration: np.ndarray,
    gaussian_delta: float,
    model: str,
    time: float,
) -> dict[str, float]:
    record = diagnostics(solver, vorticity, model, gaussian_delta)
    spectrum = energy_spectrum(solver, vorticity)
    record.update(
        {
            "time": time,
            "forcing_power": forcing_power(solver, vorticity, acceleration),
            "relative_vorticity_error": relative_error(vorticity, reference),
            "spectral_relative_l2": relative_error(spectrum, reference_spectrum),
            "reynolds_lambda": reynolds_lambda(
                record["energy"], record["enstrophy"], solver.viscosity
            ),
        }
    )
    return record


def budget_summary(records: list[dict[str, float]]) -> dict[str, object]:
    time = np.asarray([record["time"] for record in records])
    energy = np.asarray([record["energy"] for record in records])
    forcing = np.asarray([record["forcing_power"] for record in records])
    viscous = np.asarray([record["viscous_power"] for record in records])
    sgs = np.asarray([record["sgs_power"] for record in records])
    predicted = cumulative_trapezoid(forcing + viscous + sgs, time)
    actual = energy - energy[0]
    scale = float(np.trapezoid(np.abs(forcing) + np.abs(viscous) + np.abs(sgs), time))
    return {
        "actual_energy_change": actual.tolist(),
        "predicted_energy_change": predicted.tolist(),
        "relative_residual": float(abs(actual[-1] - predicted[-1]) / max(scale, 1.0e-15)),
    }


def mean_time_error(records: list[dict[str, float]], quantity: str) -> float:
    return float(np.mean([record[quantity] for record in records]))


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.reference_n != 2 * args.les_n:
        raise ValueError("pilot requires reference_n = 2 * les_n")
    reference_solver = VorticitySolver(args.reference_n, args.viscosity)
    les_solver = VorticitySolver(args.les_n, args.viscosity)
    gaussian_delta = 2.0 * (2.0 * np.pi / args.les_n) / np.sqrt(6.0)
    forcing = ForcingHistory(
        args.les_n,
        args.time_step_size,
        args.end_time,
        args.correlation_time,
        args.forcing_rms,
        args.seed,
    )

    reference_velocity = random_isotropic_velocity(args.reference_n, args.seed + 1)
    reference_vorticity = reference_solver.project(reference_solver.grid.curl(reference_velocity))
    initial_filtered = coarse_reference(
        reference_solver,
        reference_vorticity,
        args.les_n,
        gaussian_delta,
    )
    states = {model: initial_filtered.copy() for model in MODELS}
    steps = int(round(args.end_time / args.time_step_size))
    save_every = max(1, int(round(args.save_interval / args.time_step_size)))
    histories: dict[str, list[dict[str, float]]] = {
        model: [] for model in ("filtered_dns", *MODELS)
    }
    fine_reference_history: list[dict[str, float]] = []
    final_reference = initial_filtered

    for step in range(steps + 1):
        time = step * args.time_step_size
        les_acceleration = forcing.at(time, args.les_n)
        if step % save_every == 0 or step == steps:
            fine_record = diagnostics(
                reference_solver,
                reference_vorticity,
                "no_sgs",
                gaussian_delta,
            )
            fine_record.update(
                {
                    "time": time,
                    "forcing_power": forcing_power(
                        reference_solver,
                        reference_vorticity,
                        forcing.reference_at(
                            time,
                            args.reference_n,
                            gaussian_delta,
                        ),
                    ),
                    "reynolds_lambda": reynolds_lambda(
                        fine_record["energy"],
                        fine_record["enstrophy"],
                        reference_solver.viscosity,
                    ),
                }
            )
            fine_reference_history.append(fine_record)
            final_reference = coarse_reference(
                reference_solver,
                reference_vorticity,
                args.les_n,
                gaussian_delta,
            )
            reference_spectrum = energy_spectrum(les_solver, final_reference)
            histories["filtered_dns"].append(
                add_reference_diagnostics(
                    les_solver,
                    final_reference,
                    les_acceleration,
                    gaussian_delta,
                    time,
                )
            )
            for model, state in states.items():
                histories[model].append(
                    add_model_diagnostics(
                        les_solver,
                        state,
                        final_reference,
                        reference_spectrum,
                        les_acceleration,
                        gaussian_delta,
                        model,
                        time,
                    )
                )
        if step == steps:
            break
        reference_vorticity = forced_heun_step(
            reference_solver,
            reference_vorticity,
            gaussian_delta,
            args.time_step_size,
            forcing.reference_at(time, args.reference_n, gaussian_delta),
            forcing.reference_at(time + args.time_step_size, args.reference_n, gaussian_delta),
            "no_sgs",
        )
        for model in MODELS:
            states[model] = forced_heun_step(
                les_solver,
                states[model],
                gaussian_delta,
                args.time_step_size,
                les_acceleration,
                forcing.at(time + args.time_step_size, args.les_n),
                model,
            )
            if not np.all(np.isfinite(states[model])):
                raise FloatingPointError(f"non-finite {model} state at step {step + 1}")

    final_spectra = {"filtered_dns": energy_spectrum(les_solver, final_reference).tolist()}
    final_spectra.update(
        {model: energy_spectrum(les_solver, state).tolist() for model, state in states.items()}
    )
    reference_final = histories["filtered_dns"][-1]
    summary: dict[str, object] = {}
    budgets: dict[str, object] = {}
    for model in MODELS:
        records = histories[model]
        final = records[-1]
        spectrum = np.asarray(final_spectra[model])
        scale = integral_scale(spectrum, final["energy"])
        turnover = scale / np.sqrt(2.0 * final["energy"] / 3.0)
        budgets[model] = budget_summary(records)
        summary[model] = {
            "final_energy_relative_error": abs(final["energy"] - reference_final["energy"])
            / reference_final["energy"],
            "final_enstrophy_relative_error": abs(final["enstrophy"] - reference_final["enstrophy"])
            / reference_final["enstrophy"],
            "final_vorticity_relative_l2": final["relative_vorticity_error"],
            "time_mean_spectral_relative_l2": mean_time_error(records, "spectral_relative_l2"),
            "maximum_high_k_energy_fraction": max(
                record["high_k_energy_fraction"] for record in records
            ),
            "maximum_divergence_relative": max(record["divergence_relative"] for record in records),
            "mean_ssev_activation": mean_time_error(records, "activation"),
            "maximum_kkt_condition": max(record["kkt_condition"] for record in records),
            "final_reynolds_lambda": final["reynolds_lambda"],
            "final_integral_scale": scale,
            "final_turnover_time": turnover,
            "duration_in_final_turnover_times": args.end_time / turnover,
            "energy_budget_relative_residual": budgets[model]["relative_residual"],
        }
    reference_resolution = {
        "maximum_high_k_energy_fraction": max(
            record["high_k_energy_fraction"] for record in fine_reference_history
        ),
        "final_high_k_energy_fraction": fine_reference_history[-1]["high_k_energy_fraction"],
        "maximum_divergence_relative": max(
            record["divergence_relative"] for record in fine_reference_history
        ),
        "final_reynolds_lambda": fine_reference_history[-1]["reynolds_lambda"],
        "high_k_gate": 0.01,
        "final_energy_spectrum": energy_spectrum(reference_solver, reference_vorticity).tolist(),
    }
    pilot_pass = bool(
        all(np.isfinite(value["final_vorticity_relative_l2"]) for value in summary.values())
        and max(value["maximum_divergence_relative"] for value in summary.values()) < 1.0e-12
        and max(value["energy_budget_relative_residual"] for value in summary.values()) < 2.0e-3
        and reference_resolution["maximum_high_k_energy_fraction"]
        < reference_resolution["high_k_gate"]
        and summary["sensed"]["time_mean_spectral_relative_l2"]
        < summary["no_sgs"]["time_mean_spectral_relative_l2"]
    )
    return {
        "gate": "B.1b transient forced-HIT pilot",
        "qualification_status": "NOT A GATE-B PASS: transient reduced-resolution pilot",
        "pilot_screen_pass": pilot_pass,
        "configuration": {
            "reference_n": args.reference_n,
            "les_n": args.les_n,
            "viscosity": args.viscosity,
            "dt": args.time_step_size,
            "end_time": args.end_time,
            "save_interval": args.save_interval,
            "forcing_rms": args.forcing_rms,
            "forcing_correlation_time": args.correlation_time,
            "forcing_seed": args.seed,
            "forcing_relation": "G_delta f_reference = f_LES",
            "paper_filter_width_over_h": 2.0,
            "gaussian_delta": gaussian_delta,
        },
        "theoretical_model_energy_balance": "dE/dt = P_f - 2 nu Z + P_SGS",
        "histories": histories,
        "fine_reference_history": fine_reference_history,
        "reference_resolution": reference_resolution,
        "budgets": budgets,
        "final_spectra": final_spectra,
        "summary": summary,
    }


def plot_histories(result: dict[str, object], output: Path) -> None:
    histories = result["histories"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), constrained_layout=True)
    quantities = (
        ("energy", "Resolved kinetic energy"),
        ("enstrophy", "Resolved enstrophy"),
        ("high_k_energy_fraction", "High-wavenumber energy fraction"),
    )
    for axis, (quantity, title) in zip(axes, quantities, strict=True):
        for model in ("filtered_dns", *MODELS):
            records = histories[model]
            axis.plot(
                [record["time"] for record in records],
                [record[quantity] for record in records],
                color=COLORS[model],
                label=DISPLAY_LABELS[model],
                linewidth=1.8 if model in ("filtered_dns", "sensed") else 1.25,
            )
        axis.set_title(title)
        axis.set_xlabel(r"$t$")
        axis.grid(color="#d8dde2", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Transient forced homogeneous-turbulence pilot", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_spectra(result: dict[str, object], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    for model in ("filtered_dns", *MODELS):
        values = np.asarray(result["final_spectra"][model])
        wave = np.arange(len(values))
        positive = (wave > 0) & (values > 0.0)
        axis.loglog(
            wave[positive],
            values[positive],
            color=COLORS[model],
            marker="o",
            markersize=3,
            linewidth=1.8 if model in ("filtered_dns", "sensed") else 1.25,
            label=DISPLAY_LABELS[model],
        )
    axis.set_xlabel(r"wavenumber shell $k$")
    axis.set_ylabel(r"$E(k)$")
    axis.set_title("Final energy-spectrum reference overlay")
    axis.grid(color="#d8dde2", linewidth=0.7, which="both")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_budgets(result: dict[str, object], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.7, 7.2), constrained_layout=True)
    for axis, model in zip(axes.flat, MODELS, strict=True):
        records = result["histories"][model]
        budget = result["budgets"][model]
        time = [record["time"] for record in records]
        axis.plot(
            time,
            budget["actual_energy_change"],
            color=COLORS[model],
            linewidth=1.8,
            label=r"$E(t)-E(0)$",
        )
        axis.plot(
            time,
            budget["predicted_energy_change"],
            color="#20252a",
            linestyle="--",
            linewidth=1.5,
            label=r"$\int(P_f-2\nu Z+P_{SGS})\,dt$",
        )
        axis.set_title(f"{DISPLAY_LABELS[model]}: residual {budget['relative_residual']:.2e}")
        axis.set_xlabel(r"$t$")
        axis.set_ylabel("energy change")
        axis.grid(color="#d8dde2", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Model energy balances: numerical and theoretical", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_errors(result: dict[str, object], output: Path) -> None:
    histories = result["histories"]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), constrained_layout=True)
    for model in MODELS:
        records = histories[model]
        time = [record["time"] for record in records]
        axes[0].plot(
            time,
            [record["relative_vorticity_error"] for record in records],
            color=COLORS[model],
            label=DISPLAY_LABELS[model],
        )
        axes[1].plot(
            time,
            [record["spectral_relative_l2"] for record in records],
            color=COLORS[model],
            label=DISPLAY_LABELS[model],
        )
        axes[2].plot(
            time,
            [record["activation"] for record in records],
            color=COLORS[model],
            label=DISPLAY_LABELS[model],
        )
    axes[0].set_title("Vorticity-field error")
    axes[0].set_ylabel("relative $L_2$ error")
    axes[1].set_title("Energy-spectrum error")
    axes[1].set_ylabel("relative $L_2$ error")
    axes[2].set_title("Eddy-viscosity activation")
    axes[2].set_ylabel("activation fraction")
    for axis in axes:
        axis.set_xlabel(r"$t$")
        axis.grid(color="#d8dde2", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Model error against filtered reference", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_reference_resolution(result: dict[str, object], output: Path) -> None:
    resolution = result["reference_resolution"]
    history = result["fine_reference_history"]
    spectrum = np.asarray(resolution["final_energy_spectrum"])
    wave = np.arange(len(spectrum))
    positive = (wave > 0) & (spectrum > 0.0)
    cutoff = result["configuration"]["reference_n"] // 3
    high_start = 0.7 * cutoff
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    axes[0].loglog(wave[positive], spectrum[positive], color=BLUE, marker="o", markersize=3)
    axes[0].axvspan(high_start, cutoff, color="#d9973b", alpha=0.18, label="resolution-check band")
    axes[0].set_xlabel(r"wavenumber shell $k$")
    axes[0].set_ylabel(r"$E(k)$")
    axes[0].set_title("Fine-grid final energy spectrum")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(
        [record["time"] for record in history],
        [record["high_k_energy_fraction"] for record in history],
        color=BLUE,
        linewidth=1.8,
    )
    axes[1].axhline(
        resolution["high_k_gate"],
        color="#d9973b",
        linestyle="--",
        label="1% reference-resolution gate",
    )
    axes[1].set_xlabel(r"$t$")
    axes[1].set_ylabel("fine-grid high-wavenumber energy fraction")
    axes[1].set_title("Reference resolution over time")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(color="#d8dde2", linewidth=0.7, which="both")
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Reference-grid resolution check", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-n", type=int, default=48)
    parser.add_argument("--les-n", type=int, default=24)
    parser.add_argument("--viscosity", type=float, default=0.01)
    parser.add_argument("--time-step-size", dest="time_step_size", type=float, default=0.02)
    parser.add_argument("--end-time", type=float, default=2.0)
    parser.add_argument("--save-interval", type=float, default=0.1)
    parser.add_argument("--forcing-rms", type=float, default=0.25)
    parser.add_argument("--correlation-time", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    plot_histories(result, args.figure_dir / "stage_4b1_forced_hit_histories.png")
    plot_spectra(result, args.figure_dir / "stage_4b1_forced_hit_spectra.png")
    plot_budgets(result, args.figure_dir / "stage_4b1_forced_hit_budgets.png")
    plot_errors(result, args.figure_dir / "stage_4b1_forced_hit_errors.png")
    plot_reference_resolution(
        result, args.figure_dir / "stage_4b1_forced_hit_reference_resolution.png"
    )
    if not result["pilot_screen_pass"]:
        raise SystemExit("FORCED-HIT PILOT SCREEN FAIL")


if __name__ == "__main__":
    main()
