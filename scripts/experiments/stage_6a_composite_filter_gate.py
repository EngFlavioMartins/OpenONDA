#!/usr/bin/env python3
"""Reduced-campaign Gate 1: audit the complete VPM-to-LES filter symbol.

The resolved field seen by a grid closure is not filtered by ``Delta`` alone.
For Gaussian vortex blobs on a uniform particle lattice, followed by M4-prime
particle-to-mesh scatter, the LES Gaussian, spectral differentiation, and the
adjoint M4-prime mesh-to-particle gather, a Fourier mode is multiplied by

    H(k) = exp(-sigma^2 |k|^2 / 4)
           exp(-Delta^2 |k|^2 / 24)
           prod_d |M4'(k_d h; xi_d)|^2.

The first exponential is the exact Gaussian-blob regularisation used by the
VPM.  The second uses Yuan's filter-width convention.  The final factor is
measured from the production M4-prime kernel.  Spectral differentiation does
not change the normalized transfer of non-zero modes.

This is research/audit code and does not modify the production solver.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/openonda_stage6a_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/openonda_stage6a_cache")

import matplotlib.pyplot as plt
import numpy as np

from source.solvers.vpm.physics.diffusion.grid import _m4_prime_1d

INK = "#20252a"
BLUE = "#286f9b"
GOLD = "#d9973b"
GREY = "#8a99a8"
GRID = "#d8dde2"
DELTA_OVER_H = 2.0
PHASES = (0.0, 0.21, 0.37)
PHASE_SWEEP = (
    (0.0, 0.0, 0.0),
    (0.10, 0.10, 0.10),
    (0.25, 0.25, 0.25),
    (0.37, 0.37, 0.37),
    (0.49, 0.49, 0.49),
    PHASES,
)
OPERATING_SIGMA_VALUES = (2.25, 2.50, 2.75)
STRESS_SIGMA_VALUES = (1.00, 1.50, 2.00)
DIRECTIONS = {
    "axis": np.array((1.0, 0.0, 0.0)),
    "face diagonal": np.array((1.0, 1.0, 0.0)) / np.sqrt(2.0),
    "body diagonal": np.array((1.0, 1.0, 1.0)) / np.sqrt(3.0),
}


def m4_symbol(theta: np.ndarray, phase: float) -> np.ndarray:
    """Exact 1-D P2M M4' symbol for a uniformly shifted particle lattice."""
    theta = np.asarray(theta, dtype=float)
    # q = particle index - grid index; four terms can be non-zero.
    q = np.arange(-3, 4, dtype=float)
    distance = q + phase
    weight = _m4_prime_1d(np.abs(distance))
    return np.sum(weight[:, None] * np.exp(1j * distance[:, None] * theta.reshape(1, -1)), axis=0)


def transfer(
    theta_vector: np.ndarray,
    sigma_over_h: float,
    phases: tuple[float, float, float] = PHASES,
) -> np.ndarray:
    """Composite normalized transfer for vectors of nondimensional kh."""
    theta_vector = np.asarray(theta_vector, dtype=float)
    theta_sq = np.sum(theta_vector * theta_vector, axis=-1)
    gaussian_blob = np.exp(-0.25 * sigma_over_h**2 * theta_sq)
    gaussian_les = np.exp(-(DELTA_OVER_H**2) * theta_sq / 24.0)
    m4 = np.ones(len(theta_vector), dtype=float)
    for component, phase in enumerate(phases):
        m4 *= np.abs(m4_symbol(theta_vector[:, component], phase)) ** 2
    return gaussian_blob * gaussian_les * m4


def continuous_reference(theta: np.ndarray, sigma_over_h: float) -> np.ndarray:
    """Particle Gaussian times LES Gaussian, without mesh transfer."""
    return np.exp(-(0.25 * sigma_over_h**2 + DELTA_OVER_H**2 / 24.0) * np.asarray(theta) ** 2)


def direct_m4_spot_check(n: int, mode: int, phase: float) -> float:
    """Compare the symbol with an explicit periodic P2M/M2P calculation."""
    theta = 2.0 * np.pi * mode / n
    particle = np.exp(1j * theta * (np.arange(n) + phase))
    grid = np.zeros(n, dtype=complex)
    for p in range(n):
        for j in range(n):
            q = p + phase - j
            # Periodic image with the smallest grid distance.
            q -= np.rint(q / n) * n
            grid[j] += _m4_prime_1d(np.array([abs(q)]))[0] * particle[p]
    gathered = np.zeros(n, dtype=complex)
    for p in range(n):
        for j in range(n):
            q = p + phase - j
            q -= np.rint(q / n) * n
            gathered[p] += _m4_prime_1d(np.array([abs(q)]))[0] * grid[j]
    measured = np.vdot(particle, gathered) / np.vdot(particle, particle)
    predicted = abs(m4_symbol(np.array([theta]), phase)[0]) ** 2
    return float(abs(measured - predicted))


def evaluate(
    sigma_values: tuple[float, ...],
    operating_sigma_values: tuple[float, ...] = OPERATING_SIGMA_VALUES,
) -> dict[str, object]:
    radial_theta = np.linspace(0.0, np.pi, 501)
    cases: list[dict[str, object]] = []
    for sigma_over_h in sigma_values:
        directional: dict[str, list[float]] = {}
        for label, direction in DIRECTIONS.items():
            theta_vector = radial_theta[:, None] * direction[None, :]
            directional[label] = transfer(theta_vector, sigma_over_h).tolist()
        stacked = np.asarray(list(directional.values()))
        passband = np.mean(stacked, axis=0) >= 0.10
        anisotropy = np.ptp(stacked[:, passband], axis=0) / np.maximum(
            np.mean(stacked[:, passband], axis=0), np.finfo(float).tiny
        )
        monotone_violation = max(
            float(np.max(np.maximum(np.diff(values), 0.0))) for values in stacked
        )
        gain_max = float(np.max(stacked))
        phase_transfer = np.asarray(
            [
                transfer(
                    radial_theta[:, None] * DIRECTIONS["body diagonal"][None, :],
                    sigma_over_h,
                    phase,
                )
                for phase in PHASE_SWEEP
            ]
        )
        phase_passband = np.mean(phase_transfer, axis=0) >= 0.10
        phase_sensitivity = np.ptp(phase_transfer[:, phase_passband], axis=0) / np.maximum(
            np.mean(phase_transfer[:, phase_passband], axis=0), np.finfo(float).tiny
        )

        # Fit the low-k Gaussian slope. M4' reproduces quadratics, so its
        # leading contribution is O((kh)^4), not an artificial filter width.
        fit = radial_theta <= 0.25
        mean_transfer = np.mean(stacked, axis=0)
        slope = float(
            np.polyfit(radial_theta[fit] ** 2, np.log(np.maximum(mean_transfer[fit], 1e-300)), 1)[0]
        )
        measured_delta_eff_over_h = float(np.sqrt(max(0.0, -24.0 * slope)))
        theoretical_delta_eff_over_h = float(np.sqrt(DELTA_OVER_H**2 + 6.0 * sigma_over_h**2))
        width_relative_error = (
            abs(measured_delta_eff_over_h - theoretical_delta_eff_over_h)
            / theoretical_delta_eff_over_h
        )
        cases.append(
            {
                "sigma_over_h": sigma_over_h,
                "delta_over_h": DELTA_OVER_H,
                "theta": radial_theta.tolist(),
                "directional_transfer": directional,
                "continuous_reference": continuous_reference(radial_theta, sigma_over_h).tolist(),
                "max_gain": gain_max,
                "max_monotonicity_violation": monotone_violation,
                "max_passband_anisotropy": float(np.max(anisotropy)),
                "max_passband_phase_sensitivity": float(np.max(phase_sensitivity)),
                "measured_delta_effective_over_h": measured_delta_eff_over_h,
                "theoretical_delta_effective_over_h": theoretical_delta_eff_over_h,
                "effective_width_relative_error": width_relative_error,
            }
        )

    spot_errors = [
        direct_m4_spot_check(n, mode, phase)
        for n in (16, 24, 32)
        for mode in (1, max(2, n // 5), n // 2 - 1)
        for phase in PHASES
    ]
    operating_cases = [
        case
        for case in cases
        if any(np.isclose(float(case["sigma_over_h"]), value) for value in operating_sigma_values)
    ]
    if len(operating_cases) != len(operating_sigma_values):
        raise ValueError("every declared operating sigma/h must be included in the sweep")
    checks = {
        "no_amplification": max(float(case["max_gain"]) for case in operating_cases) <= 1.0 + 1e-12,
        "monotone_along_three_directions": max(
            float(case["max_monotonicity_violation"]) for case in operating_cases
        )
        < 1e-10,
        "passband_anisotropy_below_5_percent": max(
            float(case["max_passband_anisotropy"]) for case in operating_cases
        )
        < 0.05,
        "passband_phase_sensitivity_below_5_percent": max(
            float(case["max_passband_phase_sensitivity"]) for case in operating_cases
        )
        < 0.05,
        "low_k_width_matches_gaussian_composition_within_1_percent": max(
            float(case["effective_width_relative_error"]) for case in operating_cases
        )
        < 0.01,
        "symbol_matches_explicit_periodic_pipeline": max(spot_errors) < 1e-12,
    }
    return {
        "gate": "reduced feasibility 1 — composite filter",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "operating_envelope_sigma_over_h": list(operating_sigma_values),
        "stress_cases_not_used_for_gate": [
            float(case["sigma_over_h"]) for case in cases if case not in operating_cases
        ],
        "operator": (
            "Gaussian particle regularisation -> production M4' P2M -> "
            "Gaussian LES filter -> spectral derivative -> adjoint M4' M2P"
        ),
        "width_convention": ("particle exp(-sigma^2 k^2/4); LES exp(-Delta^2 k^2/24)"),
        "checks": checks,
        "max_explicit_symbol_error": max(spot_errors),
        "cases": cases,
        "interpretation": (
            "At low wavenumber the comparison filter is Delta_eff^2 = "
            "Delta^2 + 6 sigma^2. M4' adds non-Gaussian high-k attenuation."
        ),
    }


def plot(result: dict[str, object], output: Path) -> None:
    cases = result["cases"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    styles = {
        "axis": (BLUE, "-"),
        "face diagonal": (GOLD, "--"),
        "body diagonal": (GREY, ":"),
    }

    selected = cases[len(cases) // 2]
    theta = np.asarray(selected["theta"])
    for label, values in selected["directional_transfer"].items():
        color, linestyle = styles[label]
        axes[0].plot(theta / np.pi, values, color=color, linestyle=linestyle, label=label)
    axes[0].plot(
        theta / np.pi,
        selected["continuous_reference"],
        color=INK,
        linestyle="-.",
        label="continuous Gaussian reference",
    )
    axes[0].set_title(rf"Composite transfer, $\sigma/h={selected['sigma_over_h']:.2f}$")
    axes[0].set_xlabel(r"normalized wavenumber $|k|h/\pi$")
    axes[0].set_ylabel(r"gain $|H|$")
    axes[0].set_ylim(-0.02, 1.04)
    axes[0].legend(frameon=False, fontsize=8)

    sigma = np.asarray([case["sigma_over_h"] for case in cases])
    measured = np.asarray([case["measured_delta_effective_over_h"] for case in cases])
    theory = np.asarray([case["theoretical_delta_effective_over_h"] for case in cases])
    axes[1].plot(
        sigma, theory, color=INK, linestyle="--", label=r"theory $\sqrt{\Delta^2+6\sigma^2}/h$"
    )
    axes[1].plot(sigma, measured, color=BLUE, marker="o", label="measured low-$k$ width")
    axes[1].set_title("Equivalent resolved-filter width")
    axes[1].set_xlabel(r"particle core ratio $\sigma/h$")
    axes[1].set_ylabel(r"$\Delta_{\rm eff}/h$")
    axes[1].legend(frameon=False, fontsize=8)

    anisotropy = 100.0 * np.asarray([case["max_passband_anisotropy"] for case in cases])
    phase_sensitivity = 100.0 * np.asarray(
        [case["max_passband_phase_sensitivity"] for case in cases]
    )
    axes[2].plot(sigma, anisotropy, color=BLUE, marker="o", label="direction")
    axes[2].plot(
        sigma,
        phase_sensitivity,
        color=GREY,
        marker="s",
        linestyle="--",
        label="particle/grid phase",
    )
    axes[2].axhline(5.0, color=GOLD, linestyle="--", label="5% gate")
    axes[2].set_title("Passband sensitivity")
    axes[2].set_xlabel(r"particle core ratio $\sigma/h$")
    axes[2].set_ylabel("maximum spread (%)")
    axes[2].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.grid(color=GRID, linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Composite VPM–LES filter gate", fontsize=14, color=INK)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sigma-over-h",
        type=float,
        nargs="+",
        default=(*STRESS_SIGMA_VALUES, *OPERATING_SIGMA_VALUES),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(tuple(args.sigma_over_h))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    plot(result, args.figure)
    if result["status"] != "PASS":
        raise SystemExit("COMPOSITE FILTER GATE FAIL")


if __name__ == "__main__":
    main()
