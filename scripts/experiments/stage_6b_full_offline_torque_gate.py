#!/usr/bin/env python3
"""Reduced-campaign Gate 2: complete offline DIAD-to-particle torque audit.

The manufactured divergence-free velocity is passed through Gaussian particle
regularisation, the production M4-prime P2M symbol, the LES Gaussian, DIAD,
spectral derivatives, and the adjoint M4-prime M2P symbol.  Particle torque is
compared with an oversampled continuous reference at three resolutions and in
single/double precision.  This does not advance the VPM equations.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/openonda_stage6b_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/openonda_stage6b_cache")

import matplotlib.pyplot as plt
import numpy as np
from scipy import fft as spfft

from source.solvers.VPM.physics.diffusion.grid import _m4_prime_1d

STENCIL = 5
UPDATES = 2
DELTA_OVER_H = 2.0
SIGMA_OVER_H = 2.5
PHASE = (0.21, 0.37, 0.13)
INK = "#20252a"
BLUE = "#286f9b"
GOLD = "#d9973b"
GREY = "#8a99a8"
GRID_COLOR = "#d8dde2"


class Grid:
    def __init__(self, n: int, dtype: np.dtype) -> None:
        self.n = n
        self.dtype = np.dtype(dtype)
        wave = spfft.fftfreq(n, d=1.0 / n).astype(self.dtype)
        self.kx, self.ky, self.kz = np.meshgrid(wave, wave, wave, indexing="ij")
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2
        self.complex_dtype = np.dtype(np.complex64 if self.dtype == np.float32 else np.complex128)

    def fft(self, field: np.ndarray) -> np.ndarray:
        return spfft.fftn(np.asarray(field, dtype=self.dtype), axes=(-3, -2, -1)).astype(
            self.complex_dtype, copy=False
        )

    def ifft(self, field_hat: np.ndarray) -> np.ndarray:
        return spfft.ifftn(field_hat, axes=(-3, -2, -1)).real.astype(self.dtype, copy=False)

    def derivative(self, field: np.ndarray, axis: int) -> np.ndarray:
        wave = (self.kx, self.ky, self.kz)[axis]
        return self.ifft(1j * wave * self.fft(field))

    def gradient(self, vector: np.ndarray) -> np.ndarray:
        return np.stack(
            [np.stack([self.derivative(vector[i], j) for j in range(3)]) for i in range(3)]
        ).astype(self.dtype)

    def gaussian_les(self, field: np.ndarray, delta: float) -> np.ndarray:
        transfer = np.exp(-(delta**2) * self.k2 / 24.0).astype(self.dtype)
        return self.ifft(self.fft(field) * transfer)

    def gaussian_particle(self, field: np.ndarray, sigma: float) -> np.ndarray:
        transfer = np.exp(-0.25 * sigma**2 * self.k2).astype(self.dtype)
        return self.ifft(self.fft(field) * transfer)


def analytic_velocity(grid: Grid) -> np.ndarray:
    x = (2.0 * np.pi * np.arange(grid.n) / grid.n).astype(grid.dtype)
    xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
    # Each component is independent of its own coordinate: div(u)=0 exactly.
    return np.asarray(
        (
            np.sin(yy) + 0.5 * np.cos(2.0 * zz) + 0.25 * np.sin(2.0 * yy + zz),
            0.7 * np.sin(zz) + 0.3 * np.cos(3.0 * xx) + 0.2 * np.cos(xx + 2.0 * zz),
            0.5 * np.sin(2.0 * xx) + 0.4 * np.cos(2.0 * yy) + 0.3 * np.sin(2.0 * xx + yy),
        ),
        dtype=grid.dtype,
    )


def m4_scatter_symbol(grid: Grid, phase: tuple[float, float, float]) -> np.ndarray:
    result = np.ones_like(grid.k2, dtype=grid.complex_dtype)
    q = np.arange(-3, 4, dtype=grid.dtype)
    for wave, offset in zip((grid.kx, grid.ky, grid.kz), phase, strict=True):
        weights = _m4_prime_1d(np.abs(q + offset)).astype(grid.dtype)
        theta = 2.0 * np.pi * wave / grid.n
        factor = np.zeros_like(result)
        for index in range(len(q)):
            factor += weights[index] * np.exp(1j * theta * q[index]).astype(grid.complex_dtype)
        result *= factor
    return result


def apply_symbol(grid: Grid, field: np.ndarray, symbol: np.ndarray) -> np.ndarray:
    return grid.ifft(grid.fft(field) * symbol)


def offsets() -> np.ndarray:
    half = STENCIL // 2
    return np.asarray(
        [
            (i, j, k)
            for i in range(-half, half + 1)
            for j in range(-half, half + 1)
            for k in range(-half, half + 1)
        ],
        dtype=int,
    )


def correlation_lattice(
    grid: Grid, left: np.ndarray, right: np.ndarray, half_width: int, spacing: float
) -> np.ndarray:
    cross = np.sum(np.conj(grid.fft(left)) * grid.fft(right), axis=0) / grid.dtype.type(grid.n**6)
    positions = np.arange(-half_width, half_width + 1, dtype=grid.dtype) * spacing
    complex_type = grid.complex_dtype.type
    px = np.exp(complex_type(1j) * np.outer(positions, grid.kx[:, 0, 0])).astype(grid.complex_dtype)
    py = np.exp(complex_type(1j) * np.outer(positions, grid.ky[0, :, 0])).astype(grid.complex_dtype)
    pz = np.exp(complex_type(1j) * np.outer(positions, grid.kz[0, 0, :])).astype(grid.complex_dtype)
    along_x = np.einsum("ax,xyz->ayz", px, cross, optimize=True)
    along_y = np.einsum("by,ayz->abz", py, along_x, optimize=True)
    result = np.einsum("cz,abz->abc", pz, along_y, optimize=True)
    return result.real.astype(grid.dtype)


def weights(
    grid: Grid, current: np.ndarray, filtered: np.ndarray, spacing: float
) -> tuple[np.ndarray, dict[str, float]]:
    points = offsets()
    half = STENCIL // 2
    auto = correlation_lattice(grid, filtered, filtered, 2 * half, spacing)
    cross = correlation_lattice(grid, current, filtered, half, spacing)
    differences = points[None, :, :] - points[:, None, :]
    matrix = auto[
        differences[..., 0] + 2 * half,
        differences[..., 1] + 2 * half,
        differences[..., 2] + 2 * half,
    ]
    vector = cross[
        points[:, 0] + half,
        points[:, 1] + half,
        points[:, 2] + half,
    ]
    size = len(points)
    kkt = np.empty((size + 1, size + 1), dtype=grid.dtype)
    kkt[:size, :size] = 0.5 * (matrix + matrix.T)
    kkt[:size, size] = 1.0
    kkt[size, :size] = 1.0
    kkt[size, size] = 0.0
    rhs = np.concatenate((vector, np.ones(1, dtype=grid.dtype)))
    # Use the same rank rule relative to the arithmetic precision. This tests
    # reconstructed physics, rather than asking f32 to retain f64-only modes.
    rcond = float(np.finfo(grid.dtype).eps * (size + 1))
    solution, _, rank, singular = np.linalg.lstsq(kkt, rhs, rcond=rcond)
    residual = np.linalg.norm(
        kkt.astype(np.float64) @ solution.astype(np.float64) - rhs
    ) / np.linalg.norm(rhs)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf")
    return solution[:size].astype(grid.dtype), {
        "rank": int(rank),
        "rcond": rcond,
        "condition": condition,
        "relative_residual": float(residual),
        "weight_sum_error": float(abs(np.sum(solution[:size], dtype=np.float64) - 1.0)),
        "max_abs_weight": float(np.max(np.abs(solution[:size]))),
    }


def apply_stencil(
    grid: Grid, field: np.ndarray, stencil_weights: np.ndarray, spacing: float
) -> np.ndarray:
    half = STENCIL // 2
    indices = np.arange(-half, half + 1, dtype=grid.dtype)
    complex_type = grid.complex_dtype.type
    px = np.exp(complex_type(1j) * np.outer(indices * spacing, grid.kx[:, 0, 0])).astype(
        grid.complex_dtype
    )
    py = np.exp(complex_type(1j) * np.outer(indices * spacing, grid.ky[0, :, 0])).astype(
        grid.complex_dtype
    )
    pz = np.exp(complex_type(1j) * np.outer(indices * spacing, grid.kz[0, 0, :])).astype(
        grid.complex_dtype
    )
    transfer = np.einsum(
        "abc,ax,by,cz->xyz",
        stencil_weights.reshape(STENCIL, STENCIL, STENCIL),
        px,
        py,
        pz,
        optimize=True,
    ).astype(grid.complex_dtype)
    return apply_symbol(grid, field, transfer)


def diad(
    grid: Grid, resolved: np.ndarray, delta: float
) -> tuple[np.ndarray, list[dict[str, float]]]:
    spacing = delta / 2.0
    current = resolved.copy()
    history = []
    for _ in range(UPDATES):
        filtered = grid.gaussian_les(current, delta)
        stencil_weights, diagnostics = weights(grid, current, filtered, spacing)
        current = apply_stencil(grid, resolved, stencil_weights, spacing)
        consistency = np.mean(
            [
                np.mean(np.abs(grid.gaussian_les(current, delta)[i] - resolved[i]))
                / max(float(np.mean(np.abs(resolved[i]))), np.finfo(grid.dtype).tiny)
                for i in range(3)
            ]
        )
        history.append({**diagnostics, "consistency_error": float(consistency)})
    return current, history


def structural_torque(
    grid: Grid, resolved: np.ndarray, delta: float
) -> tuple[np.ndarray, list[dict[str, float]]]:
    reconstructed, history = diad(grid, resolved, delta)
    reconstructed_bar = grid.gaussian_les(reconstructed, delta)
    stress = np.empty((3, 3, grid.n, grid.n, grid.n), dtype=grid.dtype)
    for i in range(3):
        for j in range(3):
            stress[i, j] = (
                grid.gaussian_les(reconstructed[i] * reconstructed[j], delta)
                - reconstructed_bar[i] * reconstructed_bar[j]
            )
    force = np.zeros((3, grid.n, grid.n, grid.n), dtype=grid.dtype)
    for i in range(3):
        for j in range(3):
            force[i] -= grid.derivative(stress[i, j], j)
    gradient = grid.gradient(force)
    torque = np.asarray(
        (
            gradient[2, 1] - gradient[1, 2],
            gradient[0, 2] - gradient[2, 0],
            gradient[1, 0] - gradient[0, 1],
        ),
        dtype=grid.dtype,
    )
    return torque, history


def shifted_subsample(
    fine: Grid, field: np.ndarray, ratio: int, phase: tuple[float, float, float]
) -> np.ndarray:
    phase_factor = np.exp(
        1j
        * 2.0
        * np.pi
        / fine.n
        * (fine.kx * ratio * phase[0] + fine.ky * ratio * phase[1] + fine.kz * ratio * phase[2])
    ).astype(fine.complex_dtype)
    shifted = apply_symbol(fine, field, phase_factor)
    return shifted[:, ::ratio, ::ratio, ::ratio]


def metrics(model: np.ndarray, exact: np.ndarray) -> dict[str, float]:
    left = np.asarray(model, dtype=np.float64).reshape(-1)
    right = np.asarray(exact, dtype=np.float64).reshape(-1)
    left -= np.mean(left)
    right -= np.mean(right)
    return {
        "correlation": float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))),
        "relative_l2": float(np.linalg.norm(left - right) / np.linalg.norm(right)),
    }


def divergence_free_noise(grid: Grid, relative_amplitude: float, seed: int) -> np.ndarray:
    """Deterministic solenoidal perturbation with prescribed component RMS."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((3, grid.n, grid.n, grid.n)).astype(grid.dtype)
    noise_hat = grid.fft(noise)
    dot = grid.kx * noise_hat[0] + grid.ky * noise_hat[1] + grid.kz * noise_hat[2]
    safe_k2 = np.where(grid.k2 > 0.0, grid.k2, 1.0).astype(grid.dtype)
    for component, wave in enumerate((grid.kx, grid.ky, grid.kz)):
        noise_hat[component] -= wave * dot / safe_k2
    noise_hat[:, 0, 0, 0] = 0.0
    projected = grid.ifft(noise_hat)
    rms = float(np.sqrt(np.mean(projected.astype(np.float64) ** 2)))
    return projected * grid.dtype.type(relative_amplitude / max(rms, np.finfo(float).tiny))


def reference(n: int) -> np.ndarray:
    ratio = 2
    grid = Grid(ratio * n, np.float64)
    h = 2.0 * np.pi / n
    delta = DELTA_OVER_H * h
    sigma = SIGMA_OVER_H * h
    velocity = analytic_velocity(grid)
    resolved = grid.gaussian_les(grid.gaussian_particle(velocity, sigma), delta)
    torque, _ = structural_torque(grid, resolved, delta)
    return shifted_subsample(grid, torque, ratio, PHASE)


def production_case(
    n: int,
    dtype: np.dtype,
    exact: np.ndarray,
    perturbation_relative: float = 0.0,
) -> tuple[dict[str, object], np.ndarray]:
    grid = Grid(n, dtype)
    h = 2.0 * np.pi / n
    delta = DELTA_OVER_H * h
    sigma = SIGMA_OVER_H * h
    velocity = analytic_velocity(grid)
    regularized = grid.gaussian_particle(velocity, sigma)
    scatter = m4_scatter_symbol(grid, PHASE)
    scattered = apply_symbol(grid, regularized, scatter)
    resolved = grid.gaussian_les(scattered, delta)
    if perturbation_relative > 0.0:
        resolved_rms = float(np.sqrt(np.mean(resolved.astype(np.float64) ** 2)))
        resolved += divergence_free_noise(grid, perturbation_relative, 3100 + n) * resolved_rms
    torque_grid, history = structural_torque(grid, resolved, delta)
    gathered = apply_symbol(grid, torque_grid, np.conj(scatter))
    # Arrays indexed by particle number lack the physical phase carried by the
    # reference samples; shift the gathered grid representation accordingly.
    physical_shift = np.exp(
        1j * 2.0 * np.pi / n * (grid.kx * PHASE[0] + grid.ky * PHASE[1] + grid.kz * PHASE[2])
    ).astype(grid.complex_dtype)
    particle_torque = apply_symbol(grid, gathered, physical_shift)
    circulation_rate = h**3 * particle_torque
    exact_rate = h**3 * exact
    return {
        "n": n,
        "dtype": np.dtype(dtype).name,
        "h": h,
        "sigma_over_h": SIGMA_OVER_H,
        "delta_over_h": DELTA_OVER_H,
        "metrics": metrics(circulation_rate, exact_rate),
        "input_perturbation_relative": perturbation_relative,
        "updates": history,
    }, particle_torque


def evaluate(resolutions: tuple[int, ...]) -> dict[str, object]:
    cases = []
    precision = []
    perturbation = []
    for n in resolutions:
        exact = reference(n)
        f64_case, f64 = production_case(n, np.float64, exact)
        f32_case, f32 = production_case(n, np.float32, exact)
        cases.extend((f64_case, f32_case))
        precision.append({"n": n, **metrics(f32, f64)})
        for dtype, base in ((np.float64, f64), (np.float32, f32)):
            relative_input = float(32.0 * np.finfo(dtype).eps)
            _, perturbed = production_case(n, dtype, exact, relative_input)
            comparison = metrics(perturbed, base)
            perturbation.append(
                {
                    "n": n,
                    "dtype": np.dtype(dtype).name,
                    "input_relative_rms": relative_input,
                    "torque_correlation": comparison["correlation"],
                    "torque_relative_change": comparison["relative_l2"],
                    "normalized_amplification": comparison["relative_l2"] / relative_input,
                }
            )

    f64_cases = [case for case in cases if case["dtype"] == "float64"]
    f32_cases = [case for case in cases if case["dtype"] == "float32"]
    f64_error = [case["metrics"]["relative_l2"] for case in f64_cases]
    f32_error = [case["metrics"]["relative_l2"] for case in f32_cases]
    convergence_orders = {
        "float64": float(
            np.polyfit(np.log([case["h"] for case in f64_cases]), np.log(f64_error), 1)[0]
        ),
        "float32": float(
            np.polyfit(np.log([case["h"] for case in f32_cases]), np.log(f32_error), 1)[0]
        ),
    }
    finest = max(resolutions)
    finest_cases = [case for case in cases if case["n"] == finest]
    checks = {
        "finest_correlation_above_0p95": min(
            case["metrics"]["correlation"] for case in finest_cases
        )
        > 0.95,
        "finest_relative_l2_below_0p10": max(
            case["metrics"]["relative_l2"] for case in finest_cases
        )
        < 0.10,
        "monotonic_convergence_f64": bool(np.all(np.diff(f64_error) < 0.0)),
        "monotonic_convergence_f32": bool(np.all(np.diff(f32_error) < 0.0)),
        "f32_f64_correlation_above_0p999": min(item["correlation"] for item in precision) > 0.999,
        "f32_f64_relative_difference_below_0p02": max(item["relative_l2"] for item in precision)
        < 0.02,
        "roundoff_perturbation_changes_f32_torque_below_0p5_percent": max(
            item["torque_relative_change"] for item in perturbation if item["dtype"] == "float32"
        )
        < 0.005,
        "roundoff_perturbation_changes_f64_torque_below_1e_minus_8": max(
            item["torque_relative_change"] for item in perturbation if item["dtype"] == "float64"
        )
        < 1e-8,
    }
    return {
        "gate": "reduced feasibility 2 — complete offline particle torque",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "configuration": {
            "sigma_over_h": SIGMA_OVER_H,
            "delta_over_h": DELTA_OVER_H,
            "stencil": STENCIL,
            "updates": UPDATES,
            "particle_grid_phase": list(PHASE),
            "reference_oversampling": 2,
            "svd_rank_rule": "eps(dtype) * (KKT size)",
        },
        "cases": cases,
        "convergence_orders": convergence_orders,
        "precision_comparisons": precision,
        "roundoff_perturbation_comparisons": perturbation,
    }


def plot(result: dict[str, object], output: Path) -> None:
    cases = result["cases"]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.1), constrained_layout=True)
    for dtype, color, marker in (("float64", BLUE, "o"), ("float32", GOLD, "s")):
        selected = [case for case in cases if case["dtype"] == dtype]
        h = np.asarray([case["h"] for case in selected])
        error = np.asarray([case["metrics"]["relative_l2"] for case in selected])
        order = result["convergence_orders"][dtype]
        axes[0].loglog(h, error, color=color, marker=marker, label=f"{dtype}, slope {order:.2f}")
        correlation = np.asarray([case["metrics"]["correlation"] for case in selected])
        axes[1].plot(
            [case["n"] for case in selected], correlation, color=color, marker=marker, label=dtype
        )
    axes[0].axhline(0.10, color=INK, linestyle="--", label="10% gate")
    guide_h = np.asarray((min(h), max(h)))
    axes[0].loglog(
        guide_h,
        0.8 * error[-1] * (guide_h / h[-1]) ** 2,
        color=GREY,
        linestyle=":",
        label=r"$O(h^2)$ reference",
    )
    axes[0].set_title("Particle torque error")
    axes[0].set_xlabel(r"spacing $h$")
    axes[0].set_ylabel(r"relative $L_2$ error")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].axhline(0.95, color=INK, linestyle="--", label="0.95 gate")
    axes[1].set_title("Torque structure")
    axes[1].set_xlabel("particles per direction")
    axes[1].set_ylabel("correlation")
    axes[1].legend(frameon=False, fontsize=8)

    precision = result["precision_comparisons"]
    n = [item["n"] for item in precision]
    difference = 100.0 * np.asarray([item["relative_l2"] for item in precision])
    axes[2].plot(n, difference, color=BLUE, marker="o")
    axes[2].axhline(2.0, color=GOLD, linestyle="--", label="2% gate")
    axes[2].set_title("Single vs double precision")
    axes[2].set_xlabel("particles per direction")
    axes[2].set_ylabel("torque difference (%)")
    axes[2].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.grid(color=GRID_COLOR, linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Complete offline DIAD–VPM torque gate", color=INK, fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", type=int, nargs="+", default=(16, 24, 32))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(tuple(args.resolutions))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    plot(result, args.figure)
    if result["status"] != "PASS":
        raise SystemExit("FULL OFFLINE TORQUE GATE FAIL")


if __name__ == "__main__":
    main()
