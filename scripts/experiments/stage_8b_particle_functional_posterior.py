#!/usr/bin/env python3
"""One-turnover posterior screen of particle-filter functional closures.

The run branches from the qualified stationary 64^3 reference backup at
t=60.  A 32^3 LES is initialized from the same reference after applying the
actual Gaussian-particle plus M4' filter at core_radius/particle_spacing=2.5.  Three models are
compared: no SGS, the continuous analogue of current OpenONDA Smagorinsky+GBD,
and the filter-adjusted Mansfield vorticity eddy-diffusivity operator.

This is a bounded spectral feasibility screen, not VPM validation and not a
production-code implementation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/openonda_stage8b_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/openonda_stage8b_cache")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stage_4a_formulation import SpectralGrid, norm  # noqa: E402
from stage_4b1_forced_hit_pilot import budget_summary  # noqa: E402
from stage_4b2_stationary_reference import (  # noqa: E402
    StreamingOUForcing,
    curl_hat,
    rotational_reference_rhs,
    rotational_reference_step,
)
from stage_4b_spectral_pilot import (  # noqa: E402
    VorticitySolver,
    kinetic_energy_spectrum,
    projected_force_from_torque,
)
from stage_6a_composite_filter_gate import PHASES  # noqa: E402
from stage_7a_composite_sgs_audit import apply_symbol, particle_symbol  # noqa: E402
from stage_8a_particle_functional_gate import (  # noqa: E402
    gaussian_energy_width_ratio,
    mansfield_gaussian_coefficient,
    mansfield_torque,
    openonda_current_torque,
)

MODELS = ("no_sgs", "openonda_current", "mansfield")
LABELS = {
    "filtered_dns": "Particle-filtered reference",
    "no_sgs": "No SGS",
    "openonda_current": "Current OpenONDA",
    "mansfield": "Mansfield particle-filter closure",
}
COLORS = {
    "filtered_dns": "#20252a",
    "no_sgs": "#9b6a45",
    "openonda_current": "#8a99a8",
    "mansfield": "#286f9b",
}
GOLD = "#d9973b"
RED = "#b54a4a"
GRID = "#d8dde2"


def coarse_particle_filtered(
    reference_solver: VorticitySolver,
    field: np.ndarray,
    les_n: int,
    symbol: np.ndarray,
) -> np.ndarray:
    if reference_solver.grid.n % les_n != 0:
        raise ValueError("reference grid must be an integer multiple of LES grid")
    ratio = reference_solver.grid.n // les_n
    cutoff = les_n // 3
    mask = (
        (np.abs(reference_solver.grid.kx) < cutoff)
        & (np.abs(reference_solver.grid.ky) < cutoff)
        & (np.abs(reference_solver.grid.kz) < cutoff)
    )
    filtered = apply_symbol(reference_solver.grid, field, symbol * mask)
    return filtered[:, ::ratio, ::ratio, ::ratio]


def inverse_particle_embed(
    field: np.ndarray,
    target_n: int,
    target_symbol: np.ndarray,
) -> np.ndarray:
    """Embed low LES modes while undoing the reference-grid particle filter."""
    source_n = field.shape[-1]
    if target_n % source_n != 0:
        raise ValueError("target grid must be an integer multiple of source grid")
    source_grid = SpectralGrid(source_n)
    source_hat = source_grid.fft(field)
    target_hat = np.zeros((3, target_n, target_n, target_n), dtype=complex)
    radius = np.sqrt(source_grid.k2)
    for i, j, k in np.argwhere((radius >= 1.0) & (radius <= 2.0)):
        ki = int(source_grid.kx[i, j, k])
        kj = int(source_grid.ky[i, j, k])
        kk = int(source_grid.kz[i, j, k])
        transfer = float(target_symbol[ki % target_n, kj % target_n, kk % target_n])
        if transfer <= np.finfo(float).tiny:
            raise FloatingPointError("particle-filter forcing inversion is singular")
        target_hat[:, ki % target_n, kj % target_n, kk % target_n] = (
            source_hat[:, i, j, k] * (target_n / source_n) ** 3 / transfer
        )
    return SpectralGrid.ifft(target_hat)


def model_torque(
    solver: VorticitySolver,
    vorticity: np.ndarray,
    model: str,
    particle_spacing: float,
    filter_width: float,
    mansfield_coefficient: float,
) -> np.ndarray:
    if model == "no_sgs":
        return np.zeros_like(vorticity)
    velocity = solver.velocity(vorticity)
    if model == "openonda_current":
        return openonda_current_torque(
            solver.grid, velocity, vorticity, cs=0.17, particle_spacing=particle_spacing
        )[0]
    if model == "mansfield":
        return mansfield_torque(
            solver.grid,
            velocity,
            vorticity,
            coefficient=mansfield_coefficient,
            filter_width=filter_width,
        )[0]
    raise ValueError(model)


def model_rhs(
    solver: VorticitySolver,
    vorticity: np.ndarray,
    model: str,
    acceleration_curl_hat: np.ndarray,
    particle_spacing: float,
    filter_width: float,
    mansfield_coefficient: float,
) -> np.ndarray:
    base = rotational_reference_rhs(solver, vorticity, acceleration_curl_hat)
    torque = model_torque(
        solver, vorticity, model, particle_spacing, filter_width, mansfield_coefficient
    )
    return solver.grid.ifft(solver.grid.fft(base + torque) * solver.mask)


def model_step(
    solver: VorticitySolver,
    vorticity: np.ndarray,
    model: str,
    time_step_size: float,
    acceleration_start_curl_hat: np.ndarray,
    acceleration_end_curl_hat: np.ndarray,
    particle_spacing: float,
    filter_width: float,
    mansfield_coefficient: float,
) -> np.ndarray:
    first = model_rhs(
        solver,
        vorticity,
        model,
        acceleration_start_curl_hat,
        particle_spacing,
        filter_width,
        mansfield_coefficient,
    )
    predictor = solver.grid.ifft(solver.grid.fft(vorticity + time_step_size * first) * solver.mask)
    second = model_rhs(
        solver,
        predictor,
        model,
        acceleration_end_curl_hat,
        particle_spacing,
        filter_width,
        mansfield_coefficient,
    )
    return solver.grid.ifft(
        solver.grid.fft(vorticity + 0.5 * time_step_size * (first + second)) * solver.mask
    )


def diagnostics(
    solver: VorticitySolver,
    vorticity: np.ndarray,
    acceleration: np.ndarray,
    model: str,
    reference: np.ndarray,
    reference_spectrum: np.ndarray,
    particle_spacing: float,
    filter_width: float,
    mansfield_coefficient: float,
    time: float,
) -> dict[str, Any]:
    velocity = solver.velocity(vorticity)
    spectrum = kinetic_energy_spectrum(solver, vorticity)
    total_kinetic_energy = 0.5 * float(np.mean(np.sum(velocity * velocity, axis=0)))
    total_enstrophy = 0.5 * float(np.mean(np.sum(vorticity * vorticity, axis=0)))
    torque = model_torque(
        solver, vorticity, model, particle_spacing, filter_width, mansfield_coefficient
    )
    force = projected_force_from_torque(solver, torque)
    divergence = sum(solver.grid.derivative(vorticity[i], i) for i in range(3))
    wave = np.arange(len(spectrum))
    high_k = wave >= max(1, int(np.floor(0.7 * (solver.grid.n // 3))))
    return {
        "time": time,
        "total_kinetic_energy": total_kinetic_energy,
        "total_enstrophy": total_enstrophy,
        "kinetic_energy_spectrum": spectrum.tolist(),
        "spectral_relative_l2": float(
            np.linalg.norm(spectrum - reference_spectrum)
            / max(np.linalg.norm(reference_spectrum), np.finfo(float).tiny)
        ),
        "total_kinetic_energy_relative_error": abs(
            total_kinetic_energy
            - 0.5 * float(np.mean(np.sum(solver.velocity(reference) ** 2, axis=0)))
        )
        / max(
            0.5 * float(np.mean(np.sum(solver.velocity(reference) ** 2, axis=0))),
            np.finfo(float).tiny,
        ),
        "total_enstrophy_relative_error": abs(
            total_enstrophy - 0.5 * float(np.mean(np.sum(reference * reference, axis=0)))
        )
        / max(0.5 * float(np.mean(np.sum(reference * reference, axis=0))), np.finfo(float).tiny),
        "high_k_energy_fraction": float(
            np.sum(spectrum[high_k]) / max(np.sum(spectrum), np.finfo(float).tiny)
        ),
        "divergence_relative": norm(divergence) / max(norm(vorticity), np.finfo(float).tiny),
        "forcing_power": float(np.mean(np.sum(velocity * acceleration, axis=0))),
        "viscous_kinetic_energy_rate": -2.0 * solver.kinematic_viscosity * total_enstrophy,
        "sgs_power": float(np.mean(np.sum(velocity * force, axis=0))),
        "enstrophy_transfer": float(np.mean(np.sum(vorticity * torque, axis=0))),
    }


def reference_diagnostics(
    solver: VorticitySolver,
    vorticity: np.ndarray,
    acceleration: np.ndarray,
    time: float,
) -> dict[str, Any]:
    velocity = solver.velocity(vorticity)
    spectrum = kinetic_energy_spectrum(solver, vorticity)
    total_kinetic_energy = 0.5 * float(np.mean(np.sum(velocity * velocity, axis=0)))
    total_enstrophy = 0.5 * float(np.mean(np.sum(vorticity * vorticity, axis=0)))
    divergence = sum(solver.grid.derivative(vorticity[i], i) for i in range(3))
    wave = np.arange(len(spectrum))
    high_k = wave >= max(1, int(np.floor(0.7 * (solver.grid.n // 3))))
    return {
        "time": time,
        "total_kinetic_energy": total_kinetic_energy,
        "total_enstrophy": total_enstrophy,
        "kinetic_energy_spectrum": spectrum.tolist(),
        "spectral_relative_l2": 0.0,
        "total_kinetic_energy_relative_error": 0.0,
        "total_enstrophy_relative_error": 0.0,
        "high_k_energy_fraction": float(
            np.sum(spectrum[high_k]) / max(np.sum(spectrum), np.finfo(float).tiny)
        ),
        "divergence_relative": norm(divergence) / max(norm(vorticity), np.finfo(float).tiny),
        "forcing_power": float(np.mean(np.sum(velocity * acceleration, axis=0))),
        "viscous_kinetic_energy_rate": -2.0 * solver.kinematic_viscosity * total_enstrophy,
        "sgs_power": 0.0,
        "enstrophy_transfer": 0.0,
    }


def mean(records: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([record[key] for record in records]))


def run(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metadata_path = args.backup.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    arrays = np.load(args.backup)
    reference_vorticity = arrays["reference_vorticity"].copy()
    reference_solver = VorticitySolver(args.reference_n, args.kinematic_viscosity)
    les_solver = VorticitySolver(args.les_n, args.kinematic_viscosity)
    particle_spacing = 2.0 * np.pi / args.les_n
    core_radius = args.sigma_over_h * particle_spacing
    filter_width = gaussian_energy_width_ratio() * core_radius
    coefficient = mansfield_gaussian_coefficient()
    p_reference = particle_symbol(reference_solver.grid, particle_spacing, core_radius, PHASES)

    forcing = StreamingOUForcing(
        args.les_n,
        args.time_step_size,
        args.correlation_time,
        args.forcing_rms,
        args.seed,
    )
    forcing.field = arrays["forcing_field"].copy()
    forcing.rng.bit_generator.state = metadata["rng_state"]

    initial = coarse_particle_filtered(
        reference_solver, reference_vorticity, args.les_n, p_reference
    )
    states = {model: initial.copy() for model in MODELS}
    histories: dict[str, list[dict[str, Any]]] = {name: [] for name in ("filtered_dns", *MODELS)}
    fine_high_k: list[float] = []
    steps = int(round(args.duration / args.time_step_size))
    save_every = max(1, int(round(args.save_interval / args.time_step_size)))

    # Exact forcing relation at the start of the branched trajectory.
    reference_force = inverse_particle_embed(forcing.field, args.reference_n, p_reference)
    filtered_force = coarse_particle_filtered(
        reference_solver, reference_force, args.les_n, p_reference
    )
    forcing_relation_error = norm(filtered_force - forcing.field) / norm(forcing.field)

    final_reference = initial
    for step in range(steps + 1):
        time = step * args.time_step_size
        if step % max(1, steps // 10) == 0:
            print(f"functional-posterior progress: {100.0 * step / steps:5.1f}%", flush=True)
        les_force = forcing.field
        if step % save_every == 0 or step == steps:
            final_reference = coarse_particle_filtered(
                reference_solver, reference_vorticity, args.les_n, p_reference
            )
            reference_spectrum = kinetic_energy_spectrum(les_solver, final_reference)
            histories["filtered_dns"].append(
                reference_diagnostics(les_solver, final_reference, les_force, time)
            )
            fine_spectrum = kinetic_energy_spectrum(reference_solver, reference_vorticity)
            fine_wave = np.arange(len(fine_spectrum))
            fine_high_k.append(
                float(
                    np.sum(fine_spectrum[fine_wave >= int(0.7 * (args.reference_n // 3))])
                    / max(np.sum(fine_spectrum), np.finfo(float).tiny)
                )
            )
            for model, state in states.items():
                histories[model].append(
                    diagnostics(
                        les_solver,
                        state,
                        les_force,
                        model,
                        final_reference,
                        reference_spectrum,
                        particle_spacing,
                        filter_width,
                        coefficient,
                        time,
                    )
                )
        if step == steps:
            break

        reference_start = inverse_particle_embed(forcing.field, args.reference_n, p_reference)
        forcing.advance()
        reference_end = inverse_particle_embed(forcing.field, args.reference_n, p_reference)
        reference_vorticity = rotational_reference_step(
            reference_solver,
            reference_vorticity,
            args.time_step_size,
            curl_hat(reference_solver, reference_start),
            curl_hat(reference_solver, reference_end),
        )
        les_start_curl = curl_hat(les_solver, les_force)
        les_end_curl = curl_hat(les_solver, forcing.field)
        states = {
            model: model_step(
                les_solver,
                state,
                model,
                args.time_step_size,
                les_start_curl,
                les_end_curl,
                particle_spacing,
                filter_width,
                coefficient,
            )
            for model, state in states.items()
        }
        if not np.all(np.isfinite(reference_vorticity)) or not all(
            np.all(np.isfinite(state)) for state in states.values()
        ):
            raise FloatingPointError(f"non-finite field at step {step + 1}")

    summaries: dict[str, Any] = {}
    budgets: dict[str, Any] = {}
    for model in MODELS:
        records = histories[model]
        budgets[model] = budget_summary(records)
        summaries[model] = {
            "mean_total_kinetic_energy_relative_error": mean(
                records, "total_kinetic_energy_relative_error"
            ),
            "final_total_kinetic_energy_relative_error": records[-1][
                "total_kinetic_energy_relative_error"
            ],
            "mean_total_enstrophy_relative_error": mean(records, "total_enstrophy_relative_error"),
            "final_total_enstrophy_relative_error": records[-1]["total_enstrophy_relative_error"],
            "mean_spectral_relative_l2": mean(records, "spectral_relative_l2"),
            "final_spectral_relative_l2": records[-1]["spectral_relative_l2"],
            "max_high_k_energy_fraction": max(
                record["high_k_energy_fraction"] for record in records
            ),
            "max_divergence_relative": max(record["divergence_relative"] for record in records),
            "mean_sgs_power": mean(records, "sgs_power"),
            "mean_total_enstrophy_transfer": mean(records, "enstrophy_transfer"),
            "energy_budget_relative_residual": budgets[model]["relative_residual"],
            "final_kinetic_energy_spectrum": records[-1]["kinetic_energy_spectrum"],
        }

    no_sgs_error = summaries["no_sgs"]["mean_spectral_relative_l2"]
    improvement = 1.0 - summaries["mansfield"]["mean_spectral_relative_l2"] / max(
        no_sgs_error, np.finfo(float).tiny
    )
    mansfield_better_count = sum(
        summaries["mansfield"][key] < summaries["openonda_current"][key]
        for key in (
            "mean_total_kinetic_energy_relative_error",
            "mean_total_enstrophy_relative_error",
            "mean_spectral_relative_l2",
        )
    )
    checks = {
        "particle_filtered_forcing_relation": forcing_relation_error < 1.0e-12,
        "reference_resolved": max(fine_high_k) < 0.01,
        "all_energy_budgets_close": max(
            summary["energy_budget_relative_residual"] for summary in summaries.values()
        )
        < 5.0e-3,
        "mansfield_remains_solenoidal": summaries["mansfield"]["max_divergence_relative"] < 1.0e-10,
        "mansfield_mean_total_kinetic_energy_error_below_10_percent": summaries["mansfield"][
            "mean_total_kinetic_energy_relative_error"
        ]
        < 0.10,
        "mansfield_mean_total_enstrophy_error_below_10_percent": summaries["mansfield"][
            "mean_total_enstrophy_relative_error"
        ]
        < 0.10,
        "mansfield_spectrum_improves_over_no_sgs_by_20_percent": improvement > 0.20,
        "mansfield_has_no_high_k_pileup": summaries["mansfield"]["max_high_k_energy_fraction"]
        < 0.01,
        "mansfield_beats_current_openonda_on_two_of_three_statistics": mansfield_better_count >= 2,
    }
    result: dict[str, Any] = {
        "stage": "8B — bounded particle-filter functional posterior screen",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "qualification": "one-turnover spectral screen; not VPM or journal validation",
        "configuration": {
            "backup": str(args.backup.relative_to(ROOT)),
            "backup_time": metadata["time"],
            "reference_n": args.reference_n,
            "les_n": args.les_n,
            "kinematic_viscosity": args.kinematic_viscosity,
            "time_step_size": args.time_step_size,
            "duration": args.duration,
            "sigma_over_h": args.sigma_over_h,
            "particle_filter_energy_equivalent_width_over_h": filter_width / particle_spacing,
            "mansfield_gaussian_coefficient": coefficient,
            "models": list(MODELS),
        },
        "forcing_relation_relative_l2": forcing_relation_error,
        "max_reference_high_k_energy_fraction": max(fine_high_k),
        "models": summaries,
        "mansfield_spectral_improvement_over_no_sgs": improvement,
        "mansfield_better_statistics_count_vs_current": mansfield_better_count,
        "predeclared_checks": checks,
        "budgets": budgets,
        "histories": histories,
    }
    raw = {
        "reference_vorticity": reference_vorticity,
        **{f"state_{model}": state for model, state in states.items()},
        "forcing_field": forcing.field,
    }
    return result, raw


def plot(result: dict[str, Any], figure_dir: Path) -> None:
    histories = result["histories"]
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.4), constrained_layout=True)
    quantities = (
        ("total_kinetic_energy", "Resolved kinetic total_kinetic_energy", None),
        ("total_enstrophy", "Resolved total_enstrophy", None),
        ("spectral_relative_l2", "Instantaneous spectrum error", "relative $L_2$"),
        (
            "high_k_energy_fraction",
            "High-wavenumber total_kinetic_energy",
            "total_kinetic_energy fraction",
        ),
    )
    for axis, (key, title, ylabel) in zip(axes.flat, quantities, strict=True):
        names = MODELS if key == "spectral_relative_l2" else ("filtered_dns", *MODELS)
        for name in names:
            axis.plot(
                [record["time"] for record in histories[name]],
                [record[key] for record in histories[name]],
                color=COLORS[name],
                linewidth=2.0 if name in ("filtered_dns", "mansfield") else 1.4,
                linestyle="--" if name == "openonda_current" else "-",
                label=LABELS[name],
            )
        if key == "high_k_energy_fraction":
            axis.axhline(0.01, color=GOLD, linestyle=":", label="1% limit")
        axis.set_title(title)
        axis.set_xlabel("time after stationary backup")
        if ylabel:
            axis.set_ylabel(ylabel)
        axis.grid(color=GRID, linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=7)
    axes[1, 1].legend(frameon=False, fontsize=7)
    fig.suptitle("Particle-filter functional closure: one-turnover posterior screen")
    fig.savefig(figure_dir / "stage_8b_functional_histories.png", dpi=180, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5), constrained_layout=True)
    for name in ("filtered_dns", *MODELS):
        values = np.asarray(histories[name][-1]["kinetic_energy_spectrum"])
        wave = np.arange(len(values))
        positive = (wave > 0) & (values > 0.0)
        axes[0].loglog(
            wave[positive],
            values[positive],
            color=COLORS[name],
            linewidth=2.0 if name in ("filtered_dns", "mansfield") else 1.4,
            linestyle="--" if name == "openonda_current" else "-",
            marker="o" if name == "filtered_dns" else None,
            markersize=3,
            label=LABELS[name],
        )
    # A slope guide is a theoretical reference, not a fitted claim.
    guide_k = np.asarray([2.0, 7.0])
    reference_values = np.asarray(histories["filtered_dns"][-1]["kinetic_energy_spectrum"])
    guide_amplitude = reference_values[2] * 2.0 ** (5.0 / 3.0)
    axes[0].loglog(
        guide_k,
        guide_amplitude * guide_k ** (-5.0 / 3.0),
        color=GOLD,
        linestyle=":",
        label=r"$k^{-5/3}$ guide",
    )
    axes[0].set_xlabel("wavenumber shell $k$")
    axes[0].set_ylabel("final $E(k)$")
    axes[0].set_title("Final spectrum against filtered reference")
    axes[0].grid(color=GRID, linewidth=0.7, which="both")
    axes[0].legend(frameon=False, fontsize=7)

    keys = (
        "mean_total_kinetic_energy_relative_error",
        "mean_total_enstrophy_relative_error",
        "mean_spectral_relative_l2",
    )
    labels = ("total_kinetic_energy", "total_enstrophy", "spectrum")
    x = np.arange(3)
    width = 0.24
    for offset, model in zip((-width, 0.0, width), MODELS, strict=True):
        axes[1].bar(
            x + offset,
            [result["models"][model][key] for key in keys],
            width,
            color=COLORS[model],
            label=LABELS[model],
        )
    axes[1].axhline(0.10, color=GOLD, linestyle=":", label="10% limit")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("mean relative error")
    axes[1].set_title("Objective errors over the complete screen")
    axes[1].grid(axis="y", color=GRID, linewidth=0.7)
    axes[1].legend(frameon=False, fontsize=7)
    fig.savefig(
        figure_dir / "stage_8b_functional_reference_overlay.png", dpi=180, facecolor="white"
    )
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
    for model in MODELS:
        budget = result["budgets"][model]
        time = np.asarray([record["time"] for record in histories[model]])
        axis.plot(
            time,
            budget["actual_energy_change"],
            color=COLORS[model],
            linewidth=2.0,
            label=f"{LABELS[model]}: measured",
        )
        axis.plot(
            time,
            budget["predicted_energy_change"],
            color=COLORS[model],
            linestyle="--",
            linewidth=1.2,
            label=f"{LABELS[model]}: budget",
        )
    axis.axhline(0.0, color="#20252a", linewidth=0.8)
    axis.set_xlabel("time after stationary backup")
    axis.set_ylabel("change in resolved kinetic total_kinetic_energy")
    axis.set_title("Measured total_kinetic_energy change against the theoretical budget")
    axis.grid(color=GRID, linewidth=0.7)
    axis.legend(frameon=False, fontsize=7, ncol=2)
    fig.savefig(figure_dir / "stage_8b_functional_energy_budget.png", dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backup",
        type=Path,
        default=ROOT / "artifacts/vpm_les/stage_4b3_seed20260817/backup_0003000.npz",
    )
    parser.add_argument("--reference-n", type=int, default=64)
    parser.add_argument("--les-n", type=int, default=32)
    parser.add_argument("--kinematic-viscosity", type=float, default=0.02)
    parser.add_argument("--time-step-size", dest="time_step_size", type=float, default=0.02)
    parser.add_argument("--duration", type=float, default=4.0)
    # Power must be sampled at every Heun step; the earlier stage-4 budget
    # audit showed that 0.2-wide sampling aliases the rapidly varying OU force.
    parser.add_argument("--save-interval", type=float, default=0.02)
    parser.add_argument("--core_radius-over-particle_spacing", type=float, default=2.5)
    parser.add_argument("--forcing-rms", type=float, default=0.5)
    parser.add_argument("--correlation-time", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "scripts/experiments/stage_8b_particle_functional_results.json",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=ROOT / "artifacts/vpm_les/stage_8b_particle_functional_final.npz",
    )
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "docs/figures/vpm_les")
    args = parser.parse_args()
    result, raw = run(args)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.raw_output, **raw)
    plot(result, args.figure_dir)
    print(json.dumps({key: result[key] for key in result if key != "histories"}, indent=2))


if __name__ == "__main__":
    main()
