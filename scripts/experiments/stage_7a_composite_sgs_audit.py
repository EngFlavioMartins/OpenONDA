#!/usr/bin/env python3
"""A-priori audit of the exact particle-plus-LES subgrid torque.

This test distinguishes the stress hidden by the particle representation from
the additional stress hidden by the explicit LES filter.  It uses the AGARD
128^3 turbulence field and the same Gaussian-particle/M4'/Gaussian-LES symbol
audited in stage 6a.  No production solver code is changed.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/openonda_stage7a_matplotlib")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stage_4a_formulation import (  # noqa: E402
    DIAD_RATIO,
    DIAD_STENCIL,
    DIAD_UPDATES,
    SpectralGrid,
    _apply_stencil,
    _weights,
    diad,
    exact_sgs,
    load_agard,
    metrics,
    nonlinear_parts,
    norm,
    shell_transfer,
    stress_torque,
    structural_stress,
)
from stage_6a_composite_filter_gate import PHASES, m4_symbol  # noqa: E402

Filter = Callable[[np.ndarray], np.ndarray]
INK = "#20252a"
BLUE = "#286f9b"
GOLD = "#d9973b"
RED = "#b54a4a"
GREY = "#8a99a8"


def apply_symbol(grid: SpectralGrid, field: np.ndarray, symbol: np.ndarray) -> np.ndarray:
    return grid.ifft(grid.fft(field) * symbol)


def particle_symbol(
    grid: SpectralGrid,
    particle_spacing: float,
    core_radius: float,
    phases: tuple[float, float, float],
) -> np.ndarray:
    wave = np.fft.fftfreq(grid.n, d=1.0 / grid.n)
    one_dimensional = [np.abs(m4_symbol(wave * particle_spacing, phase)) ** 2 for phase in phases]
    m4 = (
        one_dimensional[0][:, None, None]
        * one_dimensional[1][None, :, None]
        * one_dimensional[2][None, None, :]
    )
    return np.exp(-(core_radius**2) * grid.k2 / 4.0) * m4


def exact_sgs_for_filter(
    grid: SpectralGrid, velocity: np.ndarray, filter_field: Filter
) -> dict[str, np.ndarray]:
    vorticity = grid.curl(velocity)
    convection, stretching = nonlinear_parts(grid, velocity, vorticity)
    u_bar = filter_field(velocity)
    w_bar = filter_field(vorticity)
    convection_bar, stretching_bar = nonlinear_parts(grid, u_bar, w_bar)
    g_c = -filter_field(convection) + convection_bar
    g_s = filter_field(stretching) - stretching_bar
    return {"velocity": u_bar, "vorticity": w_bar, "subgrid_torque": g_c + g_s}


def generalized_diad(
    grid: SpectralGrid,
    resolved: np.ndarray,
    filter_field: Filter,
    equivalent_width: float,
) -> tuple[np.ndarray, dict[str, object]]:
    spacing = equivalent_width / DIAD_RATIO
    current = resolved.copy()
    history: list[dict[str, float]] = []
    transfer = np.ones_like(grid.k2, dtype=complex)
    for update in range(1, DIAD_UPDATES + 1):
        filtered_current = filter_field(current)
        weights, solve = _weights(grid, current, filtered_current, DIAD_STENCIL, spacing)
        following, transfer = _apply_stencil(grid, resolved, weights, DIAD_STENCIL, spacing)
        filtered_following = filter_field(following)
        consistency = float(
            np.mean(
                [
                    np.mean(np.abs(filtered_following[i] - resolved[i]))
                    / max(np.mean(np.abs(resolved[i])), np.finfo(float).tiny)
                    for i in range(3)
                ]
            )
        )
        history.append(
            {
                "update": update,
                "consistency_error": consistency,
                "weight_sum_error": float(abs(np.sum(weights) - 1.0)),
                "max_abs_weight": float(np.max(np.abs(weights))),
                "transfer_gain": float(np.max(np.abs(transfer))),
                **solve,
            }
        )
        current = following
    return current, {
        "history": history,
        "consistency_error": history[-1]["consistency_error"],
        "high_k_amplification": float(np.max(np.abs(transfer))),
    }


def generalized_structural_stress(
    grid: SpectralGrid, reconstructed: np.ndarray, filter_field: Filter
) -> np.ndarray:
    reconstructed_bar = filter_field(reconstructed)
    stress = np.empty((3, 3, grid.n, grid.n, grid.n), dtype=float)
    for i in range(3):
        for j in range(3):
            stress[i, j] = (
                filter_field(reconstructed[i] * reconstructed[j])
                - reconstructed_bar[i] * reconstructed_bar[j]
            )
    return stress


def transfer_value(vorticity: np.ndarray, torque: np.ndarray) -> float:
    return float(np.mean(np.sum(vorticity * torque, axis=0)))


def evaluate(
    agard_path: Path,
    les_n: int = 32,
    sigma_over_h: float = 2.5,
    delta_over_h: float = 2.0,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    velocity = load_agard(agard_path)
    grid = SpectralGrid(velocity.shape[-1])
    particle_spacing = 2.0 * np.pi / les_n
    core_radius = sigma_over_h * particle_spacing
    paper_delta = delta_over_h * particle_spacing
    gaussian_delta = paper_delta / np.sqrt(6.0)
    delta_effective = float(np.sqrt(paper_delta**2 + 6.0 * core_radius**2))

    p_symbol = particle_symbol(grid, particle_spacing, core_radius, PHASES)
    g_symbol = np.exp(-(gaussian_delta**2) * grid.k2 / 4.0)
    h_symbol = p_symbol * g_symbol

    def particle_filter(field):
        return apply_symbol(grid, field, p_symbol)

    def total_filter(field):
        return apply_symbol(grid, field, h_symbol)

    # Exact nested-filter decomposition: g_H = G(g_P) + g_{G|P}.
    exact_particle = exact_sgs_for_filter(grid, velocity, particle_filter)
    particle_filtered_torque = grid.gaussian(exact_particle["subgrid_torque"], gaussian_delta)
    tilde_velocity = exact_particle["velocity"]
    exact_added = exact_sgs(grid, tilde_velocity, gaussian_delta)
    exact_total = exact_sgs_for_filter(grid, velocity, total_filter)
    decomposition_residual = norm(
        exact_total["subgrid_torque"] - particle_filtered_torque - exact_added["subgrid_torque"]
    ) / norm(exact_total["subgrid_torque"])
    resolved_residual = norm(exact_total["velocity"] - exact_added["velocity"]) / norm(
        exact_total["velocity"]
    )

    # Current model: invert only G and close only the added-filter stress.
    reconstructed_g, diagnostics_g = diad(grid, exact_total["velocity"], gaussian_delta)
    torque_g = stress_torque(grid, structural_stress(grid, reconstructed_g, gaussian_delta))

    # Candidate correction: invert and close the actual complete symbol H.
    reconstructed_h, diagnostics_h = generalized_diad(
        grid, exact_total["velocity"], total_filter, delta_effective
    )
    torque_h = stress_torque(
        grid,
        generalized_structural_stress(grid, reconstructed_h, total_filter),
    )

    w_bar = exact_total["vorticity"]
    metric_g_added = metrics(grid, w_bar, exact_added["subgrid_torque"], torque_g)
    metric_g_total = metrics(grid, w_bar, exact_total["subgrid_torque"], torque_g)
    metric_h_total = metrics(grid, w_bar, exact_total["subgrid_torque"], torque_h)

    total_norm = norm(exact_total["subgrid_torque"])
    transfers = {
        "exact_total": transfer_value(w_bar, exact_total["subgrid_torque"]),
        "exact_particle_filtered": transfer_value(w_bar, particle_filtered_torque),
        "exact_added_filter": transfer_value(w_bar, exact_added["subgrid_torque"]),
        "model_G_only": transfer_value(w_bar, torque_g),
        "model_complete_H": transfer_value(w_bar, torque_h),
    }
    component_shares = {
        "particle_torque_rms_over_total": norm(particle_filtered_torque) / total_norm,
        "added_filter_torque_rms_over_total": norm(exact_added["subgrid_torque"]) / total_norm,
        "particle_transfer_over_total": transfers["exact_particle_filtered"]
        / transfers["exact_total"],
        "added_filter_transfer_over_total": transfers["exact_added_filter"]
        / transfers["exact_total"],
    }

    checks = {
        "exact_nested_filter_identity": decomposition_residual < 1.0e-10,
        "resolved_fields_identical": resolved_residual < 1.0e-12,
        "current_G_model_resolves_its_declared_subproblem": (
            metric_g_added["correlation"] >= 0.75 and metric_g_added["relative_l2"] <= 0.75
        ),
        "complete_H_model_has_useful_local_structure": (
            metric_h_total["correlation"] >= 0.75 and metric_h_total["relative_l2"] <= 0.75
        ),
        "complete_H_model_has_useful_spectral_transfer": (
            metric_h_total["shell_error"] <= 0.75
            and 0.25 <= metric_h_total["transfer_ratio"] <= 2.0
        ),
        "complete_H_reconstruction_gain_below_100": (
            float(diagnostics_h["high_k_amplification"]) <= 100.0
        ),
    }
    total_model_pass = all(
        checks[key]
        for key in (
            "exact_nested_filter_identity",
            "resolved_fields_identical",
            "complete_H_model_has_useful_local_structure",
            "complete_H_model_has_useful_spectral_transfer",
            "complete_H_reconstruction_gain_below_100",
        )
    )

    result: dict[str, object] = {
        "stage": "7A — exact composite SGS a-priori audit",
        "status": "PASS" if total_model_pass else "FAIL",
        "source_field": str(agard_path.relative_to(ROOT)),
        "configuration": {
            "dns_n": grid.n,
            "nominal_les_n": les_n,
            "sigma_over_h": sigma_over_h,
            "delta_over_h": delta_over_h,
            "delta_effective_over_h": delta_effective / particle_spacing,
            "particle_grid_phase": list(PHASES),
            "particle_operator": "Gaussian core followed by M4' P2M/M2P symbol",
            "added_filter": "Gaussian G_delta",
        },
        "model_scope": {
            "current_G_only": (
                "reconstructs particle-filtered velocity and models only the "
                "additional G-filter stress"
            ),
            "candidate_complete_H": (
                "reconstructs the unfiltered velocity using the complete P*G "
                "symbol and models the total composite stress"
            ),
        },
        "identity_residuals": {
            "torque_decomposition_relative_l2": decomposition_residual,
            "resolved_velocity_relative_l2": resolved_residual,
        },
        "component_shares": component_shares,
        "transfers": transfers,
        "metrics": {
            "current_G_model_vs_exact_added_filter": metric_g_added,
            "current_G_model_vs_exact_total": metric_g_total,
            "complete_H_model_vs_exact_total": metric_h_total,
        },
        "diagnostics": {
            "current_G_model": diagnostics_g,
            "complete_H_model": diagnostics_h,
        },
        "predeclared_checks": checks,
        "interpretation": (
            "PASS means the complete-filter reconstruction is accurate enough "
            "to justify one inexpensive posterior spectral screen. It does not "
            "validate VPM LES."
        ),
    }
    arrays = {
        "shell_exact_total": shell_transfer(grid, w_bar, exact_total["subgrid_torque"]),
        "shell_exact_particle": shell_transfer(grid, w_bar, particle_filtered_torque),
        "shell_exact_added": shell_transfer(grid, w_bar, exact_added["subgrid_torque"]),
        "shell_model_g": shell_transfer(grid, w_bar, torque_g),
        "shell_model_h": shell_transfer(grid, w_bar, torque_h),
    }
    return result, arrays


def plot(result: dict[str, object], arrays: dict[str, np.ndarray], output: Path) -> None:
    metrics_by_model = result["metrics"]
    component_shares = result["component_shares"]
    transfers = result["transfers"]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.3), constrained_layout=True)

    labels = ["particle\nterm", "added $G$\nterm"]
    rms = [
        component_shares["particle_torque_rms_over_total"],
        component_shares["added_filter_torque_rms_over_total"],
    ]
    transfer = [
        component_shares["particle_transfer_over_total"],
        component_shares["added_filter_transfer_over_total"],
    ]
    x = np.arange(2)
    axes[0].bar(x - 0.18, rms, 0.36, color=BLUE, label="torque RMS / total")
    axes[0].bar(x + 0.18, transfer, 0.36, color=GOLD, label="transfer / total")
    axes[0].axhline(0.0, color=INK, linewidth=0.8)
    axes[0].axhline(1.0, color=GREY, linestyle="--", linewidth=1.0)
    axes[0].set_xticks(x, labels)
    axes[0].set_title("What the composite stress contains")
    axes[0].set_ylabel("fraction of exact total")
    axes[0].legend(frameon=False, fontsize=8)

    names = ["$G$ model\nvs added", "$G$ model\nvs total", "$H$ model\nvs total"]
    records = [
        metrics_by_model["current_G_model_vs_exact_added_filter"],
        metrics_by_model["current_G_model_vs_exact_total"],
        metrics_by_model["complete_H_model_vs_exact_total"],
    ]
    correlations = [record["correlation"] for record in records]
    errors = [record["relative_l2"] for record in records]
    x = np.arange(3)
    axes[1].bar(x - 0.18, correlations, 0.36, color=BLUE, label="correlation (1 ideal)")
    axes[1].bar(x + 0.18, errors, 0.36, color=RED, label=r"relative $L_2$ (0 ideal)")
    axes[1].axhline(
        0.75, color=GOLD, linestyle="--", linewidth=1.0, label="declared structure gate"
    )
    axes[1].set_xticks(x, names)
    axes[1].set_ylim(0.0, max(1.05, 1.05 * max(errors)))
    axes[1].set_title("Model against the correct target")
    axes[1].legend(frameon=False, fontsize=8)

    k = np.arange(len(arrays["shell_exact_total"]))
    scale = max(abs(transfers["exact_total"]), np.finfo(float).tiny)
    axes[2].plot(
        k, arrays["shell_exact_total"] / scale, color=INK, linewidth=2.0, label="exact total"
    )
    axes[2].plot(
        k,
        arrays["shell_exact_particle"] / scale,
        color=GREY,
        linestyle="--",
        label="exact particle term",
    )
    axes[2].plot(
        k,
        arrays["shell_exact_added"] / scale,
        color=GOLD,
        linestyle=":",
        label="exact added-$G$ term",
    )
    axes[2].plot(
        k, arrays["shell_model_g"] / scale, color=BLUE, linestyle="--", label="$G$-only model"
    )
    axes[2].plot(
        k, arrays["shell_model_h"] / scale, color=RED, linestyle="-.", label="complete-$H$ model"
    )
    axes[2].axhline(0.0, color=GREY, linewidth=0.8)
    axes[2].set_xlim(0, min(30, len(k) - 1))
    axes[2].set_title("Enstrophy transfer by scale")
    axes[2].set_xlabel("wavenumber shell $k$")
    axes[2].set_ylabel("shell transfer / total mean transfer")
    axes[2].legend(frameon=False, fontsize=7)

    fig.suptitle(
        r"Composite SGS audit: $\core_radius/particle_spacing=2.5$, $\Delta/particle_spacing=2$, "
        + rf"$\Delta_{{\rm eff}}/particle_spacing={result['configuration']['delta_effective_over_h']:.2f}$",
        fontsize=12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agard",
        type=Path,
        default=ROOT / "docs/dns/agard_hom02/CB128_9.bin",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "scripts/experiments/stage_7a_composite_sgs_results.json",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=ROOT / "docs/figures/vpm_les/stage_7a_composite_sgs_audit.png",
    )
    args = parser.parse_args()
    result, arrays = evaluate(args.agard)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    plot(result, arrays, args.figure)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
