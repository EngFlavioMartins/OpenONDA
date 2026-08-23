#!/usr/bin/env python3
"""Audit the short, unperturbed Gaussian-ring relaxation preflight.

The particle moments follow Archer, Thomas & Coleman (JFM 598, 2008),
equations (2.2)--(2.4).  Centreline Fourier amplitudes are recomputed from the
raw HDF5 states, so results do not depend on legacy azimuthal diagnostic bins.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/openonda_relaxed_ring_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/openonda_relaxed_ring_cache")

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RING_ASSETS = ROOT / "tutorials" / "vpm" / "vortex_ring" / "assets"
sys.path.insert(0, str(RING_ASSETS))

from ring_diagnostics import RingModeDiagnosticsSampler  # noqa: E402

RADIUS = 1.0
CIRCULATION = 1.0
CORE_RADIUS = 0.4131
VISCOSITY = CIRCULATION / 3000.0

# These are engineering preflight thresholds, selected after the exploratory
# short runs.  They authorize the next, prospectively gated relaxation; they
# are not presented as an independent confirmation of the solver.
PREFLIGHT_LIMITS = {
    "max_energy_balance_relative_residual": 0.05,
    "max_impulse_relative_drift": 1.0e-3,
    "max_axisymmetry_mode_amplitude": 1.0e-4,
    "max_time_pair_relative_difference": 5.0e-3,
    "max_time_pair_mode_absolute_difference": 1.0e-6,
    "max_finest_pair_speed_relative_difference": 1.0e-2,
    "max_finest_pair_energy_relative_difference": 2.0e-2,
    "max_finest_initial_core_radius_relative_error": 1.0e-2,
}

INK = "#20252a"
BLUE = "#286f9b"
GOLD = "#d9973b"
GREEN = "#448a62"
RED = "#b44c43"
GREY = "#7d8993"
GRID = "#d8dde2"


@dataclass(frozen=True)
class RunSpec:
    name: str
    directory: Path
    spacing: float
    time_step_size: float
    diffusion: str


RUNS = (
    RunSpec(
        "CS h=0.15",
        ROOT / "tutorials/vpm/vortex_ring/solution/relaxed_reference_cs_h015_tstar02",
        0.15,
        0.02,
        "Core Spreading",
    ),
    RunSpec(
        "CS h=0.12",
        ROOT / "tutorials/vpm/vortex_ring/solution/relaxed_reference_cs_h012_dt002_tstar02",
        0.12,
        0.02,
        "Core Spreading",
    ),
    RunSpec(
        "CS h=0.10",
        ROOT / "tutorials/vpm/vortex_ring/solution/relaxed_reference_cs_h010_dt002_tstar02",
        0.10,
        0.02,
        "Core Spreading",
    ),
    RunSpec(
        "CS h=0.08",
        ROOT / "tutorials/vpm/vortex_ring/solution/relaxed_reference_cs_h008_dt002_tstar02",
        0.08,
        0.02,
        "Core Spreading",
    ),
    RunSpec(
        "CS h=0.10, dt=0.01",
        ROOT / "tutorials/vpm/vortex_ring/solution/relaxed_reference_cs_h010_tstar02",
        0.10,
        0.01,
        "Core Spreading",
    ),
    RunSpec(
        "GBD h=0.15",
        ROOT / "tutorials/vpm/vortex_ring/solution/relaxed_reference_gbd_h015_tstar02",
        0.15,
        0.02,
        "Grid-Based Diffusion",
    ),
)


def state_paths(run: RunSpec) -> tuple[Path, Path]:
    manifest_path = next(run.directory.glob("run_manifest_*.json"))
    label = json.loads(manifest_path.read_text(encoding="utf-8"))["output_label"]
    return (
        run.directory / f"vpm_{label}_000000.h5",
        run.directory / f"vpm_{label}_final.h5",
    )


def load_state(path: Path) -> dict[str, np.ndarray | float | int]:
    with h5py.File(path, "r") as handle:
        particles = handle["particles"]
        attrs = handle["solver"].attrs
        return {
            "position": np.asarray(particles["position"], dtype=np.float64),
            "vortex_strength": np.asarray(particles["vortex_strength"], dtype=np.float64),
            "core_radius": np.asarray(particles["core_radius"], dtype=np.float64),
            "group_id": np.asarray(particles["group_id"], dtype=np.int32),
            "time": float(attrs["time"]),
            "n_particles_total": int(attrs["n_particles_total"]),
        }


def archer_moments(state: dict[str, np.ndarray | float | int]) -> dict[str, float]:
    position = np.asarray(state["position"])
    vortex_strength = np.asarray(state["vortex_strength"])
    core_radius = np.asarray(state["core_radius"])
    cylindrical_radius = np.hypot(position[:, 1], position[:, 2])
    theta = np.arctan2(position[:, 2], position[:, 1])
    tangent = np.column_stack((np.zeros_like(theta), -np.sin(theta), np.cos(theta)))
    alpha_theta = np.einsum("ij,ij->i", vortex_strength, tangent)
    orientation = np.sign(alpha_theta.sum()) or 1.0
    weight = orientation * alpha_theta / np.maximum(cylindrical_radius, np.finfo(float).eps)
    total_weight = float(weight.sum())
    ring_radius = float(np.sum(weight * cylindrical_radius) / total_weight)
    second_radius = float(np.sum(weight * cylindrical_radius**2) / total_weight)
    # The anti-diffused initializer and Core Spreading use additive squared
    # Gaussian widths.  The particle-centre moment therefore receives the
    # circulation-weighted blob-width contribution below.
    blob_width_sq = float(np.sum(weight * core_radius**2) / total_weight)
    core_radius_sq = max(
        2.0 * (second_radius - ring_radius**2) + blob_width_sq,
        0.0,
    )
    axial_centroid = float(np.sum(weight * position[:, 0]) / total_weight)
    tube_circulation = total_weight / (2.0 * np.pi)
    impulse = 0.5 * np.sum(np.cross(position, vortex_strength), axis=0)
    return {
        "axial_centroid": axial_centroid,
        "ring_radius_theta": ring_radius,
        "ring_radius_second_moment": float(np.sqrt(max(second_radius, 0.0))),
        "core_radius_theta": float(np.sqrt(core_radius_sq)),
        "tube_circulation": tube_circulation,
        "impulse_x": float(impulse[0]),
        "impulse_norm": float(np.linalg.norm(impulse)),
    }


def modal_metrics(state: dict[str, np.ndarray | float | int]) -> dict[str, float]:
    sampler = RingModeDiagnosticsSampler(
        max_mode=16,
        azimuthal_bins=96,
        reference_radius=RADIUS,
        transverse_origin=(0.0, 0.0),
    )
    rows = np.asarray(
        sampler._sample_group(
            np.asarray(state["position"]),
            np.asarray(state["vortex_strength"]),
        ),
        dtype=float,
    )
    if rows.shape != (16, 8):
        raise RuntimeError(f"failed to recover all 16 centreline modes: {rows.shape}")
    combined = rows[:, 3]
    return {
        "max_mode_amplitude": float(combined.max()),
        "rms_mode_amplitude": float(np.sqrt(np.mean(combined**2))),
        "dominant_mode": int(rows[np.argmax(combined), 0]),
        "diagnostic_ring_radius": float(rows[0, 6]),
        "azimuthal_coverage": float(rows[0, 7]),
    }


def gaussian_speed(circulation: float, radius: float, core_radius: float) -> float:
    epsilon = core_radius / radius
    return circulation / (4.0 * np.pi * radius) * (np.log(8.0 / epsilon) - 0.558)


def relaxed_empirical_speed(circulation: float, radius: float, core_radius: float) -> float:
    epsilon = core_radius / radius
    correction = -0.558 - 1.12 * epsilon**2 - 5.0 * epsilon**4
    return circulation / (4.0 * np.pi * radius) * (np.log(8.0 / epsilon) + correction)


def analyze_run(run: RunSpec) -> dict[str, object]:
    initial_path, final_path = state_paths(run)
    initial_state = load_state(initial_path)
    final_state = load_state(final_path)
    initial = archer_moments(initial_state)
    final = archer_moments(final_state)
    modes = modal_metrics(final_state)
    flow = pd.read_csv(run.directory / "samples" / "flow_integrals.csv")
    first = flow.iloc[0]
    last = flow.iloc[-1]
    duration = float(final_state["time"]) - float(initial_state["time"])
    energy_change = float(last["total_kinetic_energy"] - first["total_kinetic_energy"])
    molecular_energy_change = (
        duration
        * 0.5
        * float(first["viscous_kinetic_energy_rate"] + last["viscous_kinetic_energy_rate"])
    )
    mean_gaussian_speed = 0.5 * (
        gaussian_speed(
            initial["tube_circulation"],
            initial["ring_radius_theta"],
            initial["core_radius_theta"],
        )
        + gaussian_speed(
            final["tube_circulation"],
            final["ring_radius_theta"],
            final["core_radius_theta"],
        )
    )
    mean_relaxed_speed = 0.5 * (
        relaxed_empirical_speed(
            initial["tube_circulation"],
            initial["ring_radius_theta"],
            initial["core_radius_theta"],
        )
        + relaxed_empirical_speed(
            final["tube_circulation"],
            final["ring_radius_theta"],
            final["core_radius_theta"],
        )
    )
    predicted_core = np.sqrt(CORE_RADIUS**2 + 4.0 * VISCOSITY * duration)
    return {
        "name": run.name,
        "diffusion": run.diffusion,
        "spacing": run.spacing,
        "time_step_size": run.time_step_size,
        "initial_n_particles_total": int(initial_state["n_particles_total"]),
        "final_n_particles_total": int(final_state["n_particles_total"]),
        "duration": duration,
        "measured_speed": (final["axial_centroid"] - initial["axial_centroid"]) / duration,
        "gaussian_speed_reference": mean_gaussian_speed,
        "relaxed_empirical_speed_reference": mean_relaxed_speed,
        "initial": initial,
        "final": final,
        "predicted_gaussian_core_radius": float(predicted_core),
        "core_diffusion_relative_error": abs(final["core_radius_theta"] / predicted_core - 1.0),
        "initial_total_kinetic_energy": float(first["total_kinetic_energy"]),
        "final_total_kinetic_energy": float(last["total_kinetic_energy"]),
        "measured_energy_decay_rate": -energy_change / duration,
        "molecular_dissipation_rate": -molecular_energy_change / duration,
        "energy_balance_relative_residual": abs(
            (energy_change - molecular_energy_change) / molecular_energy_change
        ),
        "impulse_relative_drift": abs(final["impulse_norm"] / initial["impulse_norm"] - 1.0),
        "circulation_relative_drift": abs(
            final["tube_circulation"] / initial["tube_circulation"] - 1.0
        ),
        "final_vorticity_divergence_error": float(last["vorticity_divergence_error"]),
        "final_vortex_strength_misalignment_degrees": float(
            last["vortex_strength_misalignment_degrees"]
        ),
        **modes,
    }


def relative_difference(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), np.finfo(float).tiny)


def evaluate_gates(results: list[dict[str, object]]) -> dict[str, object]:
    by_name = {str(result["name"]): result for result in results}
    cs_primary = [by_name[f"CS h={spacing:.2f}"] for spacing in (0.15, 0.12, 0.10, 0.08)]
    time_coarse = by_name["CS h=0.10"]
    time_fine = by_name["CS h=0.10, dt=0.01"]
    spatial_coarse = by_name["CS h=0.10"]
    spatial_fine = by_name["CS h=0.08"]

    time_pair_metrics = {
        key: relative_difference(float(time_coarse[key]), float(time_fine[key]))
        for key in (
            "measured_speed",
            "final_total_kinetic_energy",
        )
    }
    time_pair_metrics["ring_radius_theta"] = relative_difference(
        float(time_coarse["final"]["ring_radius_theta"]),
        float(time_fine["final"]["ring_radius_theta"]),
    )
    time_pair_metrics["core_radius_theta"] = relative_difference(
        float(time_coarse["final"]["core_radius_theta"]),
        float(time_fine["final"]["core_radius_theta"]),
    )
    time_pair_mode_absolute_difference = abs(
        float(time_coarse["max_mode_amplitude"]) - float(time_fine["max_mode_amplitude"])
    )
    speed_difference = relative_difference(
        float(spatial_coarse["measured_speed"]),
        float(spatial_fine["measured_speed"]),
    )
    energy_difference = relative_difference(
        float(spatial_coarse["final_total_kinetic_energy"]),
        float(spatial_fine["final_total_kinetic_energy"]),
    )
    finest_initial_core_error = abs(
        float(spatial_fine["initial"]["core_radius_theta"]) / CORE_RADIUS - 1.0
    )
    checks = {
        "energy_balance": max(
            float(result["energy_balance_relative_residual"]) for result in cs_primary
        )
        <= PREFLIGHT_LIMITS["max_energy_balance_relative_residual"],
        "impulse": max(float(result["impulse_relative_drift"]) for result in cs_primary)
        <= PREFLIGHT_LIMITS["max_impulse_relative_drift"],
        "axisymmetry": max(float(result["max_mode_amplitude"]) for result in cs_primary)
        <= PREFLIGHT_LIMITS["max_axisymmetry_mode_amplitude"],
        "time_step_convergence": (
            max(time_pair_metrics.values()) <= PREFLIGHT_LIMITS["max_time_pair_relative_difference"]
            and time_pair_mode_absolute_difference
            <= PREFLIGHT_LIMITS["max_time_pair_mode_absolute_difference"]
        ),
        "finest_pair_speed": speed_difference
        <= PREFLIGHT_LIMITS["max_finest_pair_speed_relative_difference"],
        "finest_pair_energy": energy_difference
        <= PREFLIGHT_LIMITS["max_finest_pair_energy_relative_difference"],
        "benchmark_core_definition": finest_initial_core_error
        <= PREFLIGHT_LIMITS["max_finest_initial_core_radius_relative_error"],
    }
    return {
        "threshold_provenance": (
            "Engineering preflight limits selected after exploratory short runs; "
            "the next relaxation gate is frozen prospectively."
        ),
        "limits": PREFLIGHT_LIMITS,
        "observed": {
            "max_energy_balance_relative_residual": max(
                float(result["energy_balance_relative_residual"]) for result in cs_primary
            ),
            "max_impulse_relative_drift": max(
                float(result["impulse_relative_drift"]) for result in cs_primary
            ),
            "max_axisymmetry_mode_amplitude": max(
                float(result["max_mode_amplitude"]) for result in cs_primary
            ),
            "time_pair_relative_differences": time_pair_metrics,
            "time_pair_mode_absolute_difference": time_pair_mode_absolute_difference,
            "max_time_pair_relative_difference": max(time_pair_metrics.values()),
            "finest_pair_speed_relative_difference": speed_difference,
            "finest_pair_energy_relative_difference": energy_difference,
            "finest_initial_core_radius_relative_error": finest_initial_core_error,
        },
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def style_axis(axis: plt.Axes) -> None:
    axis.grid(True, color=GRID, linewidth=0.7, alpha=0.75)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(colors=INK)


def plot(results: list[dict[str, object]], output: Path) -> None:
    cs = [
        result
        for result in results
        if result["diffusion"] == "Core Spreading" and result["time_step_size"] == 0.02
    ]
    cs.sort(key=lambda item: float(item["spacing"]), reverse=True)
    gbd = next(result for result in results if result["diffusion"] == "Grid-Based Diffusion")
    h = np.array([float(result["spacing"]) for result in cs])

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4), constrained_layout=True)
    fig.suptitle(
        r"Gaussian vortex-ring relaxation preflight: $R=\Gamma=1$, "
        r"$\delta_0=0.4131$, $Re_\Gamma=3000$, $t^*=0.2$",
        color=INK,
        fontsize=13,
        fontweight="bold",
    )

    axis = axes[0, 0]
    speeds = [float(result["measured_speed"]) for result in cs]
    gaussian = [float(result["gaussian_speed_reference"]) for result in cs]
    relaxed = [float(result["relaxed_empirical_speed_reference"]) for result in cs]
    axis.plot(h, speeds, "o-", color=BLUE, label="VPM, first 0.2 time units")
    axis.plot(h, gaussian, "--", color=INK, label="Gaussian-core theory")
    axis.plot(h, relaxed, ":", color=GOLD, label="relaxed-core empirical target")
    axis.set_xlabel(r"particle spacing $h/R$")
    axis.set_ylabel(r"translation speed $UR/\Gamma$")
    axis.set_title("Translation has not yet reached the relaxed-ring value")
    axis.legend(frameon=False, fontsize=8)
    axis.invert_xaxis()
    style_axis(axis)

    axis = axes[0, 1]
    measured_core = [float(result["final"]["core_radius_theta"]) for result in cs]
    predicted_core = [float(result["predicted_gaussian_core_radius"]) for result in cs]
    ring_radius_change_ppm = [
        1.0e6 * (float(result["final"]["ring_radius_theta"]) / RADIUS - 1.0) for result in cs
    ]
    axis.plot(h, measured_core, "o-", color=GREEN, label=r"measured $\delta_\theta$")
    axis.plot(h, predicted_core, "--", color=INK, label=r"$\sqrt{\delta_0^2+4\nu t}$")
    twin = axis.twinx()
    twin.plot(h, ring_radius_change_ppm, "s-", color=GOLD, label=r"$R_\theta-R$")
    twin.axhline(0.0, color=GREY, linestyle=":", label=r"initial $R_\theta$")
    axis.set_xlabel(r"particle spacing $h/R$")
    axis.set_ylabel(r"core thickness $\delta_\theta/R$")
    twin.set_ylabel(r"ring-radius change $(R_\theta/R-1)$ [ppm]")
    axis.set_title("The truncated cloud misses the prescribed core thickness")
    handles, labels = axis.get_legend_handles_labels()
    twin_handles, twin_labels = twin.get_legend_handles_labels()
    axis.legend(handles + twin_handles, labels + twin_labels, frameon=False, fontsize=8)
    axis.invert_xaxis()
    style_axis(axis)
    twin.spines["top"].set_visible(False)

    axis = axes[1, 0]
    all_balance = cs + [gbd]
    colors = [BLUE] * len(cs) + [RED]
    annotation_offsets = ((6, -13), (6, 5), (6, 15), (6, 25), (6, 5))
    for result, color, offset in zip(all_balance, colors, annotation_offsets, strict=True):
        axis.scatter(
            float(result["molecular_dissipation_rate"]),
            float(result["measured_energy_decay_rate"]),
            color=color,
            s=42,
            zorder=3,
        )
        axis.annotate(
            str(result["name"]).replace(", dt=0.01", ""),
            (
                float(result["molecular_dissipation_rate"]),
                float(result["measured_energy_decay_rate"]),
            ),
            xytext=offset,
            textcoords="offset points",
            fontsize=7,
            color=INK,
        )
    values = np.array(
        [
            [
                float(result["molecular_dissipation_rate"]),
                float(result["measured_energy_decay_rate"]),
            ]
            for result in all_balance
        ]
    )
    lower = 0.9 * values.min()
    upper = 1.06 * values.max()
    axis.plot(
        [lower, upper], [lower, upper], "--", color=INK, label=r"$-dE/dt=\nu\int|\omega|^2dV$"
    )
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_xlabel("molecular dissipation rate")
    axis.set_ylabel("measured energy-decay rate")
    axis.set_title("Exact Navier-Stokes energy identity")
    axis.legend(frameon=False, fontsize=8)
    style_axis(axis)

    axis = axes[1, 1]
    cs_modes = [float(result["max_mode_amplitude"]) for result in cs]
    axis.semilogy(h, cs_modes, "o-", color=BLUE, label="Core Spreading")
    axis.scatter(
        [float(gbd["spacing"])],
        [float(gbd["max_mode_amplitude"])],
        marker="s",
        color=RED,
        label="Grid-Based Diffusion",
        zorder=3,
    )
    axis.axhline(
        PREFLIGHT_LIMITS["max_axisymmetry_mode_amplitude"],
        color=INK,
        linestyle="--",
        label="preflight limit",
    )
    axis.set_xlabel(r"particle spacing $h/R$")
    axis.set_ylabel(r"largest artificial mode $A_m/R$")
    axis.set_title("Axisymmetry from direct particle Fourier quadrature")
    axis.legend(frameon=False, fontsize=8)
    axis.invert_xaxis()
    style_axis(axis)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    results = [analyze_run(run) for run in RUNS]
    gate = evaluate_gates(results)
    payload = {
        "stage": "5B relaxed Gaussian-ring short preflight",
        "status": gate["status"],
        "claim_scope": (
            "Short numerical preflight only. It does not demonstrate a relaxed base "
            "state, Widnall growth, or LES performance."
        ),
        "primary_references": [
            {
                "authors": "Archer, Thomas & Coleman",
                "journal": "Journal of Fluid Mechanics 598 (2008), 201-226",
                "doi": "10.1017/S0022112007009883",
                "equations_used": ["2.1", "2.2", "2.3", "2.4", "2.6", "3.2"],
            },
            {
                "authors": "Verzicco & Shariff",
                "publication": "CTR Annual Research Briefs 1994, 221-228",
                "purpose": "relax-before-perturb protocol and mode-6 benchmark",
            },
        ],
        "gate": gate,
        "runs": results,
    }
    result_path = ROOT / "scripts/experiments/stage_5b_relaxed_ring_preflight_results.json"
    figure_path = ROOT / "docs/figures/vpm_les/stage_5b_relaxed_ring_preflight.png"
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    plot(results, figure_path)
    print(
        json.dumps({"status": gate["status"], "gate": gate, "figure": str(figure_path)}, indent=2)
    )


if __name__ == "__main__":
    main()
