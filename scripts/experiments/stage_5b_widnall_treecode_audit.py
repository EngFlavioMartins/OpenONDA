#!/usr/bin/env python3
"""Compare Widnall-ring treecode fields with direct pairwise evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/openonda_stage5b_tree_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/openonda_stage5b_tree_cache")

import matplotlib.pyplot as plt
import numpy as np

from source.solvers.VPM import VPMSolver

MODES = np.arange(20, 25)
INK = "#20252a"
BLUE = "#286f9b"
GOLD = "#d9973b"
GREY = "#8a99a8"
GRID = "#d8dde2"


def relative_l2(value: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(value - reference) / np.linalg.norm(reference))


def ring_field_coefficients(
    position: np.ndarray,
    circulation: np.ndarray,
    field: np.ndarray,
    *,
    azimuthal_bins: int = 128,
) -> np.ndarray:
    """Return complex radial/axial coefficients for modes 20...24."""
    theta = np.mod(np.arctan2(position[:, 2], position[:, 1]), 2.0 * np.pi)
    rho = np.hypot(position[:, 1], position[:, 2])
    tangent = np.column_stack((np.zeros_like(theta), -np.sin(theta), np.cos(theta)))
    weight = np.abs(np.einsum("ij,ij->i", circulation, tangent)) / np.maximum(
        rho, np.finfo(float).eps
    )
    radial_field = field[:, 1] * np.cos(theta) + field[:, 2] * np.sin(theta)
    axial_field = field[:, 0]
    bin_index = np.minimum(
        np.floor(theta * azimuthal_bins / (2.0 * np.pi)).astype(int),
        azimuthal_bins - 1,
    )
    bin_weight = np.bincount(bin_index, weights=weight, minlength=azimuthal_bins)
    occupied = bin_weight > np.finfo(float).tiny
    if not np.all(occupied):
        raise ValueError("every azimuthal bin must be occupied for the modal audit")
    radial = (
        np.bincount(bin_index, weights=weight * radial_field, minlength=azimuthal_bins) / bin_weight
    )
    axial = (
        np.bincount(bin_index, weights=weight * axial_field, minlength=azimuthal_bins) / bin_weight
    )
    angles = (np.arange(azimuthal_bins) + 0.5) * 2.0 * np.pi / azimuthal_bins
    coefficients: list[complex] = []
    for component in (radial, axial):
        for mode in MODES:
            transfer = np.sinc(mode / azimuthal_bins)
            coefficients.append(np.mean(component * np.exp(-1j * mode * angles)) / transfer)
    return np.asarray(coefficients)


def transposed_rate(gradient: np.ndarray, circulation: np.ndarray) -> np.ndarray:
    """Contract J-transpose with particle circulation."""
    return np.einsum("nij,ni->nj", gradient, circulation)


def audit_backup(backup: Path, theta_values: list[float]) -> dict[str, object]:
    solver = VPMSolver.continue_from_checkpoint(str(backup))
    if solver is None:
        raise RuntimeError(f"could not restore {backup}")
    particles = solver.particles
    position = particles.position_cpu(use_cache=False).astype(np.float64)
    circulation = particles.vortex_strength_cpu(use_cache=False).astype(np.float64)

    started = time.perf_counter()
    solver.physics.compute_velocity_and_gradient(particles)
    direct_seconds = time.perf_counter() - started
    direct_velocity = particles.velocity_cpu(use_cache=False).astype(np.float64)
    direct_gradient = particles.velocity_gradient_cpu(use_cache=False).astype(np.float64)
    direct_rate = transposed_rate(direct_gradient, circulation)
    direct_velocity_modes = ring_field_coefficients(position, circulation, direct_velocity)
    direct_rate_modes = ring_field_coefficients(position, circulation, direct_rate)

    rows: list[dict[str, float]] = []
    for theta in theta_values:
        started = time.perf_counter()
        solver.physics.compute_velocity_and_gradient_hierarchical(particles, theta=theta)
        elapsed = time.perf_counter() - started
        velocity = particles.velocity_cpu(use_cache=False).astype(np.float64)
        gradient = particles.velocity_gradient_cpu(use_cache=False).astype(np.float64)
        rate = transposed_rate(gradient, circulation)
        rows.append(
            {
                "theta": theta,
                "seconds": elapsed,
                "velocity_relative_l2": relative_l2(velocity, direct_velocity),
                "gradient_relative_l2": relative_l2(gradient, direct_gradient),
                "transposed_rate_relative_l2": relative_l2(rate, direct_rate),
                "velocity_widnall_band_relative_l2": relative_l2(
                    ring_field_coefficients(position, circulation, velocity),
                    direct_velocity_modes,
                ),
                "rate_widnall_band_relative_l2": relative_l2(
                    ring_field_coefficients(position, circulation, rate), direct_rate_modes
                ),
            }
        )
    result = {
        "backup": str(backup),
        "flow_time": float(solver.time),
        "time_step": int(solver.step),
        "particles": len(particles),
        "direct_seconds": direct_seconds,
        "theta_results": rows,
    }
    solver.reset_gpu()
    return result


def plot(results: list[dict[str, object]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    colors = (BLUE, GOLD, GREY)
    for result, color in zip(results, colors, strict=False):
        rows = result["theta_results"]
        theta = np.asarray([row["theta"] for row in rows], dtype=float)
        order = np.argsort(theta)
        label = rf"$t={result['flow_time']:.3g}$"
        axes[0].loglog(
            theta[order],
            np.asarray([row["velocity_relative_l2"] for row in rows])[order],
            marker="o",
            color=color,
            label=label,
        )
        axes[1].loglog(
            theta[order],
            np.asarray([row["velocity_widnall_band_relative_l2"] for row in rows])[order],
            marker="o",
            color=color,
            label=label,
        )
    axes[0].set_title("All-particle velocity")
    axes[1].set_title(r"Velocity modes $m=20\ldots24$")
    for axis in axes:
        axis.axhline(0.05, color=INK, linestyle="--", label="5% accuracy target")
        axis.set_xlabel(r"tree opening angle $\theta$")
        axis.set_ylabel("relative $L_2$ error")
        axis.grid(color=GRID, linewidth=0.6, which="both")
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
    fig.suptitle("Widnall-ring treecode audit against direct summation", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94), pad=1.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup", type=Path, action="append", required=True)
    parser.add_argument("--theta", type=float, nargs="+", default=[0.3, 0.2, 0.15, 0.1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    results = [audit_backup(path, args.theta) for path in args.backup]
    payload = {"status": "AUDIT", "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plot(results, args.figure)


if __name__ == "__main__":
    main()
