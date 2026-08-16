#!/usr/bin/env python3
"""Evaluate the corrected short Gaussian-ring relaxation gate."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/openonda_relaxed_corrected_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/openonda_relaxed_corrected_cache")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.experiments.stage_5b_relaxed_ring_analysis import (
    BLUE,
    CORE_RADIUS,
    GOLD,
    GREEN,
    GREY,
    INK,
    RunSpec,
    analyze_run,
    relative_difference,
    style_axis,
)

LIMITS = {
    "maximum_initial_core_radius_relative_error": 1.0e-2,
    "maximum_energy_balance_relative_residual": 5.0e-2,
    "maximum_impulse_relative_drift": 1.0e-3,
    "maximum_circulation_relative_drift": 1.0e-3,
    "maximum_axisymmetry_mode_amplitude": 1.0e-4,
    "maximum_time_pair_relative_difference": 5.0e-3,
    "maximum_time_pair_mode_absolute_difference": 1.0e-6,
    "maximum_spatial_pair_speed_relative_difference": 1.0e-2,
    "maximum_spatial_pair_energy_relative_difference": 2.0e-2,
    "maximum_finest_gaussian_speed_relative_error": 1.0e-2,
}

SPECS = (
    RunSpec(
        "h=0.15, dt=0.02",
        ROOT / "tutorials/VPM/vortexRing/solution/relaxed_reference_tail002_cs_h015_dt002_tstar02",
        0.15,
        0.02,
        "Core Spreading",
    ),
    RunSpec(
        "h=0.12, dt=0.02",
        ROOT / "tutorials/VPM/vortexRing/solution/relaxed_reference_tail002_cs_h012_dt002_tstar02",
        0.12,
        0.02,
        "Core Spreading",
    ),
    RunSpec(
        "h=0.10, dt=0.02",
        ROOT / "tutorials/VPM/vortexRing/solution/relaxed_reference_tail002_cs_h010_dt002_tstar02",
        0.10,
        0.02,
        "Core Spreading",
    ),
    RunSpec(
        "h=0.12, dt=0.01",
        ROOT / "tutorials/VPM/vortexRing/solution/relaxed_reference_tail002_cs_h012_dt001_tstar02",
        0.12,
        0.01,
        "Core Spreading",
    ),
)


def evaluate(results: list[dict[str, object]]) -> dict[str, object]:
    by_name = {str(result["name"]): result for result in results}
    primary = [
        by_name["h=0.15, dt=0.02"],
        by_name["h=0.12, dt=0.02"],
        by_name["h=0.10, dt=0.02"],
    ]
    time_coarse = by_name["h=0.12, dt=0.02"]
    time_fine = by_name["h=0.12, dt=0.01"]
    spatial_coarse = by_name["h=0.12, dt=0.02"]
    spatial_fine = by_name["h=0.10, dt=0.02"]

    time_relative = {
        "speed": relative_difference(
            float(time_coarse["measured_speed"]), float(time_fine["measured_speed"])
        ),
        "energy": relative_difference(
            float(time_coarse["energy_final"]), float(time_fine["energy_final"])
        ),
        "ring_radius": relative_difference(
            float(time_coarse["final"]["ring_radius_theta"]),
            float(time_fine["final"]["ring_radius_theta"]),
        ),
        "core_radius": relative_difference(
            float(time_coarse["final"]["core_radius_theta"]),
            float(time_fine["final"]["core_radius_theta"]),
        ),
    }
    mode_absolute = abs(
        float(time_coarse["maximum_mode_amplitude"]) - float(time_fine["maximum_mode_amplitude"])
    )
    spatial_speed = relative_difference(
        float(spatial_coarse["measured_speed"]),
        float(spatial_fine["measured_speed"]),
    )
    spatial_energy = relative_difference(
        float(spatial_coarse["energy_final"]), float(spatial_fine["energy_final"])
    )
    finest_speed_theory = abs(
        float(spatial_fine["measured_speed"]) / float(spatial_fine["gaussian_speed_reference"])
        - 1.0
    )
    observed = {
        "maximum_initial_core_radius_relative_error": max(
            abs(float(result["initial"]["core_radius_theta"]) / CORE_RADIUS - 1.0)
            for result in primary
        ),
        "maximum_energy_balance_relative_residual": max(
            float(result["energy_balance_relative_residual"]) for result in primary
        ),
        "maximum_impulse_relative_drift": max(
            float(result["impulse_relative_drift"]) for result in primary
        ),
        "maximum_circulation_relative_drift": max(
            float(result["circulation_relative_drift"]) for result in primary
        ),
        "maximum_axisymmetry_mode_amplitude": max(
            float(result["maximum_mode_amplitude"]) for result in primary
        ),
        "time_pair_relative_differences": time_relative,
        "maximum_time_pair_relative_difference": max(time_relative.values()),
        "time_pair_mode_absolute_difference": mode_absolute,
        "spatial_pair_speed_relative_difference": spatial_speed,
        "spatial_pair_energy_relative_difference": spatial_energy,
        "finest_gaussian_speed_relative_error": finest_speed_theory,
    }
    checks = {
        "benchmark_core": observed["maximum_initial_core_radius_relative_error"]
        <= LIMITS["maximum_initial_core_radius_relative_error"],
        "energy_balance": observed["maximum_energy_balance_relative_residual"]
        <= LIMITS["maximum_energy_balance_relative_residual"],
        "impulse": observed["maximum_impulse_relative_drift"]
        <= LIMITS["maximum_impulse_relative_drift"],
        "circulation": observed["maximum_circulation_relative_drift"]
        <= LIMITS["maximum_circulation_relative_drift"],
        "axisymmetry": observed["maximum_axisymmetry_mode_amplitude"]
        <= LIMITS["maximum_axisymmetry_mode_amplitude"],
        "time_step": (
            observed["maximum_time_pair_relative_difference"]
            <= LIMITS["maximum_time_pair_relative_difference"]
            and observed["time_pair_mode_absolute_difference"]
            <= LIMITS["maximum_time_pair_mode_absolute_difference"]
        ),
        "spatial_speed": observed["spatial_pair_speed_relative_difference"]
        <= LIMITS["maximum_spatial_pair_speed_relative_difference"],
        "spatial_energy": observed["spatial_pair_energy_relative_difference"]
        <= LIMITS["maximum_spatial_pair_energy_relative_difference"],
        "gaussian_speed": observed["finest_gaussian_speed_relative_error"]
        <= LIMITS["maximum_finest_gaussian_speed_relative_error"],
    }
    return {
        "limits": LIMITS,
        "observed": observed,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def plot(results: list[dict[str, object]], output: Path) -> None:
    primary = sorted(
        (result for result in results if float(result["time_step"]) == 0.02),
        key=lambda result: float(result["spacing"]),
        reverse=True,
    )
    h = np.array([float(result["spacing"]) for result in primary])
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4), constrained_layout=True)
    fig.suptitle(
        r"Corrected Gaussian-ring preflight: $R=\Gamma=1$, "
        r"$\delta_0=0.4131$, $Re_\Gamma=3000$, $t^*=0.2$",
        color=INK,
        fontsize=13,
        fontweight="bold",
    )

    axis = axes[0, 0]
    axis.plot(
        h,
        [float(result["measured_speed"]) for result in primary],
        "o-",
        color=BLUE,
        label="VPM",
    )
    axis.plot(
        h,
        [float(result["gaussian_speed_reference"]) for result in primary],
        "--",
        color=INK,
        label="Gaussian-core theory",
    )
    axis.plot(
        h,
        [float(result["relaxed_empirical_speed_reference"]) for result in primary],
        ":",
        color=GOLD,
        label="relaxed-core target",
    )
    axis.set_xlabel(r"particle spacing $h/R$")
    axis.set_ylabel(r"translation speed $UR/\Gamma$")
    axis.set_title("Short-time motion overlaps Gaussian-core theory")
    axis.invert_xaxis()
    axis.legend(frameon=False, fontsize=8)
    style_axis(axis)

    axis = axes[0, 1]
    axis.plot(
        h,
        [float(result["initial"]["core_radius_theta"]) for result in primary],
        "o-",
        color=GREY,
        label=r"initial $\delta_\theta$",
    )
    axis.plot(
        h,
        [float(result["final"]["core_radius_theta"]) for result in primary],
        "s-",
        color=GREEN,
        label=r"measured final $\delta_\theta$",
    )
    axis.plot(
        h,
        [float(result["predicted_gaussian_core_radius"]) for result in primary],
        "--",
        color=INK,
        label=r"$\sqrt{\delta_0^2+4\nu t}$",
    )
    axis.set_xlabel(r"particle spacing $h/R$")
    axis.set_ylabel(r"integral core thickness $\delta_\theta/R$")
    axis.set_title("Prescribed core and viscous broadening are resolved")
    axis.invert_xaxis()
    axis.legend(frameon=False, fontsize=8)
    style_axis(axis)

    axis = axes[1, 0]
    for result in primary:
        axis.scatter(
            float(result["molecular_dissipation_rate"]),
            float(result["measured_energy_decay_rate"]),
            color=BLUE,
            s=44,
            zorder=3,
        )
        axis.annotate(
            f"h={float(result['spacing']):.2f}",
            (
                float(result["molecular_dissipation_rate"]),
                float(result["measured_energy_decay_rate"]),
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color=INK,
        )
    values = np.array(
        [
            [
                float(result["molecular_dissipation_rate"]),
                float(result["measured_energy_decay_rate"]),
            ]
            for result in primary
        ]
    )
    lower = 0.98 * values.min()
    upper = 1.02 * values.max()
    axis.plot([lower, upper], [lower, upper], "--", color=INK, label="exact identity")
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.xaxis.set_major_locator(MaxNLocator(5))
    axis.yaxis.set_major_locator(MaxNLocator(5))
    axis.ticklabel_format(axis="both", style="sci", scilimits=(-3, -3))
    axis.set_xlabel("molecular dissipation rate")
    axis.set_ylabel("measured energy-decay rate")
    axis.set_title("Energy decay is molecular and quantitatively closed")
    axis.legend(frameon=False, fontsize=8)
    style_axis(axis)

    axis = axes[1, 1]
    axis.semilogy(
        h,
        [float(result["maximum_mode_amplitude"]) for result in primary],
        "o-",
        color=BLUE,
        label="largest artificial mode",
    )
    axis.semilogy(
        h,
        [float(result["impulse_relative_drift"]) for result in primary],
        "s-",
        color=GREEN,
        label="impulse drift",
    )
    axis.semilogy(
        h,
        [float(result["circulation_relative_drift"]) for result in primary],
        "^-",
        color=GOLD,
        label="circulation drift",
    )
    axis.axhline(1.0e-4, color=INK, linestyle="--", label="strictest relevant limit")
    axis.set_xlabel(r"particle spacing $h/R$")
    axis.set_ylabel("dimensionless error")
    axis.set_title("Symmetry and invariants remain far below their limits")
    axis.invert_xaxis()
    axis.legend(frameon=False, fontsize=8)
    style_axis(axis)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def main() -> None:
    results = [analyze_run(spec) for spec in SPECS]
    gate = evaluate(results)
    payload = {
        "stage": "5B corrected relaxed Gaussian-ring short gate",
        "status": gate["status"],
        "claim_scope": (
            "The corrected unperturbed ring is qualified only for a longer "
            "axisymmetric relaxation. No Widnall or LES claim is made."
        ),
        "tail_fraction": 2.0e-3,
        "gate": gate,
        "runs": results,
    }
    result_path = ROOT / "scripts/experiments/stage_5b_relaxed_ring_corrected_results.json"
    figure_path = ROOT / "docs/figures/vpm_les/stage_5b_relaxed_ring_corrected_gate.png"
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    plot(results, figure_path)
    print(json.dumps({"status": gate["status"], "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
