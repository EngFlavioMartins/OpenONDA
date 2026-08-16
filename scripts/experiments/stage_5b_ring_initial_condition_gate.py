#!/usr/bin/env python3
"""Select a Gaussian-tail cutoff that reproduces the benchmark ring moments."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/openonda_ring_ic_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/openonda_ring_ic_cache")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from openonda.vpm import ParticleDistributor, VortexRingVPM  # noqa: E402
from scripts.experiments.stage_5b_relaxed_ring_analysis import (  # noqa: E402
    CIRCULATION,
    CORE_RADIUS,
    RADIUS,
    VISCOSITY,
    archer_moments,
    modal_metrics,
)

SPACINGS = (0.15, 0.12, 0.10, 0.08)
TAIL_FRACTIONS = (5.0e-2, 1.0e-2, 5.0e-3, 2.0e-3, 1.0e-3, 1.0e-4)
SELECTED_TAIL_FRACTION = 2.0e-3
LIMITS = {
    "maximum_core_radius_relative_error": 1.0e-2,
    "maximum_ring_radius_relative_error": 1.0e-3,
    "maximum_tube_circulation_relative_error": 1.0e-6,
    "maximum_axisymmetry_mode_amplitude": 1.0e-4,
    "maximum_omitted_circulation_fraction": 2.0e-3,
}

INK = "#20252a"
BLUE = "#286f9b"
GOLD = "#d9973b"
GREY = "#7d8993"
GRID = "#d8dde2"


def continuum_truncated_core_radius(spacing: float, tail_fraction: float) -> float:
    particle_radius = 2.0 * spacing
    represented_sq = CORE_RADIUS**2 - particle_radius**2
    cutoff = -np.log(tail_fraction)
    retained_second_moment = (1.0 - (cutoff + 1.0) * tail_fraction) / (1.0 - tail_fraction)
    return float(np.sqrt(particle_radius**2 + represented_sq * retained_second_moment))


def make_case(spacing: float, tail_fraction: float) -> dict[str, object]:
    particle_radius = 2.0 * spacing
    represented_sq = CORE_RADIUS**2 - particle_radius**2
    tube_radius = np.sqrt(represented_sq) * np.sqrt(-np.log(tail_fraction))
    if tube_radius >= RADIUS:
        return {
            "spacing": spacing,
            "particle_radius": particle_radius,
            "tail_fraction": tail_fraction,
            "particle_count": None,
            "tube_radius": float(tube_radius),
            "feasible": False,
            "infeasibility_reason": "toroidal cloud reaches or crosses the symmetry axis",
        }
    position, volume, radius = ParticleDistributor.toroidal_distribution(
        RADIUS,
        tube_radius,
        spacing,
        epsilon_w=0.0,
    )
    radius.fill(particle_radius)
    _, _, circulation = VortexRingVPM(
        viscosity=VISCOSITY,
        ring_center=[0.0, 0.0, 0.0],
        ring_radius=RADIUS,
        ring_strength=CIRCULATION,
        ring_thickness=CORE_RADIUS,
        avg_particle_radius=particle_radius,
        positions=position,
        volumes=volume,
        epsilon_W=0.0,
        max_modes=1,
        anti_diffuse_flag=True,
        normalize_circulation=True,
    )
    state: dict[str, np.ndarray | float | int] = {
        "position": position,
        "circulation": circulation,
        "radius": radius,
        "group_id": np.zeros(len(position), dtype=np.int32),
        "time": 0.0,
        "particles": len(position),
    }
    moments = archer_moments(state)
    modes = modal_metrics(state)
    continuum_core = continuum_truncated_core_radius(spacing, tail_fraction)
    return {
        "spacing": spacing,
        "particle_radius": particle_radius,
        "tail_fraction": tail_fraction,
        "particle_count": len(position),
        "tube_radius": float(tube_radius),
        "feasible": True,
        "core_radius_theta": moments["core_radius_theta"],
        "core_radius_relative_error": abs(moments["core_radius_theta"] / CORE_RADIUS - 1.0),
        "continuum_truncated_core_radius": continuum_core,
        "quadrature_error_against_truncated_continuum": abs(
            moments["core_radius_theta"] / continuum_core - 1.0
        ),
        "ring_radius_theta": moments["ring_radius_theta"],
        "ring_radius_relative_error": abs(moments["ring_radius_theta"] / RADIUS - 1.0),
        "tube_circulation": moments["tube_circulation"],
        "tube_circulation_relative_error": abs(moments["tube_circulation"] / CIRCULATION - 1.0),
        "maximum_axisymmetry_mode_amplitude": modes["maximum_mode_amplitude"],
    }


def evaluate(cases: list[dict[str, object]]) -> dict[str, object]:
    selected = [case for case in cases if float(case["tail_fraction"]) == SELECTED_TAIL_FRACTION]
    if not all(bool(case["feasible"]) for case in selected):
        return {
            "selected_tail_fraction": SELECTED_TAIL_FRACTION,
            "limits": LIMITS,
            "status": "FAIL",
            "reason": "selected cutoff is not geometrically feasible at every spacing",
        }
    observed = {
        "maximum_core_radius_relative_error": max(
            float(case["core_radius_relative_error"]) for case in selected
        ),
        "maximum_ring_radius_relative_error": max(
            float(case["ring_radius_relative_error"]) for case in selected
        ),
        "maximum_tube_circulation_relative_error": max(
            float(case["tube_circulation_relative_error"]) for case in selected
        ),
        "maximum_axisymmetry_mode_amplitude": max(
            float(case["maximum_axisymmetry_mode_amplitude"]) for case in selected
        ),
        "maximum_omitted_circulation_fraction": SELECTED_TAIL_FRACTION,
    }
    checks = {key: observed[key] <= limit for key, limit in LIMITS.items()}
    return {
        "selected_tail_fraction": SELECTED_TAIL_FRACTION,
        "limits": LIMITS,
        "observed": observed,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def plot(cases: list[dict[str, object]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), constrained_layout=True)
    fig.suptitle(
        "Gaussian-ring initial-condition gate",
        color=INK,
        fontsize=13,
        fontweight="bold",
    )
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(SPACINGS)))

    axis = axes[0]
    for spacing, color in zip(SPACINGS, colors, strict=True):
        subset = sorted(
            (
                case
                for case in cases
                if float(case["spacing"]) == spacing and bool(case["feasible"])
            ),
            key=lambda case: float(case["tail_fraction"]),
        )
        axis.loglog(
            [float(case["tail_fraction"]) for case in subset],
            [float(case["core_radius_relative_error"]) for case in subset],
            "o-",
            color=color,
            label=rf"$h/R={spacing:.2f}$",
        )
    axis.axhline(
        LIMITS["maximum_core_radius_relative_error"],
        color=INK,
        linestyle="--",
        label="1% benchmark limit",
    )
    axis.axvline(SELECTED_TAIL_FRACTION, color=GOLD, linestyle=":", label="selected cutoff")
    axis.set_xlabel("vorticity fraction at cloud boundary")
    axis.set_ylabel(r"integral core-radius error $|\delta_\theta/0.4131-1|$")
    axis.set_title("The 5% cutoff fails the prescribed core definition")
    axis.grid(True, which="both", color=GRID, linewidth=0.7, alpha=0.75)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1]
    for spacing, color in zip(SPACINGS, colors, strict=True):
        subset = sorted(
            (
                case
                for case in cases
                if float(case["spacing"]) == spacing and bool(case["feasible"])
            ),
            key=lambda case: float(case["tail_fraction"]),
        )
        axis.loglog(
            [float(case["tail_fraction"]) for case in subset],
            [int(case["particle_count"]) for case in subset],
            "o-",
            color=color,
            label=rf"$h/R={spacing:.2f}$",
        )
    axis.axvline(SELECTED_TAIL_FRACTION, color=GOLD, linestyle=":")
    axis.set_xlabel("vorticity fraction at cloud boundary")
    axis.set_ylabel("initial particle count")
    axis.set_title("Cost of retaining the Gaussian tail")
    axis.grid(True, which="both", color=GRID, linewidth=0.7, alpha=0.75)
    axis.spines[["top", "right"]].set_visible(False)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def main() -> None:
    cases = [
        make_case(spacing, tail_fraction)
        for spacing in SPACINGS
        for tail_fraction in TAIL_FRACTIONS
    ]
    gate = evaluate(cases)
    payload = {
        "stage": "5B Gaussian-ring initial-condition gate",
        "status": gate["status"],
        "benchmark": {
            "ring_radius": RADIUS,
            "tube_circulation": CIRCULATION,
            "integral_core_radius": CORE_RADIUS,
        },
        "definition": (
            "Archer et al. (2008), equations (2.2)-(2.4), including the "
            "Gaussian particle-blob second moment."
        ),
        "gate": gate,
        "cases": cases,
    }
    result_path = ROOT / "scripts/experiments/stage_5b_ring_initial_condition_results.json"
    figure_path = ROOT / "docs/figures/vpm_les/stage_5b_ring_initial_condition_gate.png"
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    plot(cases, figure_path)
    print(json.dumps({"status": gate["status"], "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
