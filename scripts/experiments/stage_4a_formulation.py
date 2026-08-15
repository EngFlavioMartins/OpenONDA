#!/usr/bin/env python3
"""Gate A: frozen DIAD/SSEV formulation checks and evidence regeneration.

This is research code. It does not import or modify the production VPM solver.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/openonda_stage4a_matplotlib")

import matplotlib.pyplot as plt
import numpy as np

DIAD_STENCIL = 5
DIAD_RATIO = 2.0
DIAD_UPDATES = 2
DIAD_TOLERANCE = 0.01
DIAD_RCOND = np.finfo(np.float64).eps * (DIAD_STENCIL**3 + 1)


class SpectralGrid:
    """Periodic Fourier operators on [0, 2 pi)^3."""

    def __init__(self, n: int) -> None:
        self.n = n
        wave = np.fft.fftfreq(n, d=1.0 / n)
        self.kx, self.ky, self.kz = np.meshgrid(wave, wave, wave, indexing="ij")
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2

    @staticmethod
    def fft(field: np.ndarray) -> np.ndarray:
        return np.fft.fftn(field, axes=(-3, -2, -1))

    @staticmethod
    def ifft(field_hat: np.ndarray) -> np.ndarray:
        return np.fft.ifftn(field_hat, axes=(-3, -2, -1)).real

    def derivative(self, field: np.ndarray, axis: int) -> np.ndarray:
        wave = (self.kx, self.ky, self.kz)[axis]
        return self.ifft(1j * wave * self.fft(field))

    def gradient(self, vector: np.ndarray) -> np.ndarray:
        return np.stack(
            [np.stack([self.derivative(vector[i], j) for j in range(3)]) for i in range(3)]
        )

    def curl(self, vector: np.ndarray) -> np.ndarray:
        gradient = self.gradient(vector)
        return np.array(
            (
                gradient[2, 1] - gradient[1, 2],
                gradient[0, 2] - gradient[2, 0],
                gradient[1, 0] - gradient[0, 1],
            )
        )

    def gaussian(self, field: np.ndarray, delta: float) -> np.ndarray:
        transfer = np.exp(-(delta**2) * self.k2 / 4.0)
        return self.ifft(self.fft(field) * transfer)


def norm(field: np.ndarray) -> float:
    return float(np.sqrt(np.mean(field * field)))


def load_agard(path: Path) -> np.ndarray:
    prefix = path.read_bytes()[:4096]
    match = re.search(rb"HEADERLENGTH=(\d+)", prefix)
    if match is None:
        raise ValueError("AGARD header length is absent")
    header_length = int(match.group(1))
    header = prefix[:header_length].decode("ascii")
    if "128x128x128" not in header or "(u,v,w)" not in header:
        raise ValueError("unexpected AGARD field metadata")
    data = np.memmap(
        path,
        dtype=">f4",
        mode="r",
        offset=header_length,
        shape=(128, 128, 128, 3),
    )
    return data.transpose(3, 2, 1, 0).astype(np.float64)


def truncate(grid: SpectralGrid, velocity: np.ndarray, retained: int) -> np.ndarray:
    mask = (
        (np.abs(grid.kx) <= retained)
        & (np.abs(grid.ky) <= retained)
        & (np.abs(grid.kz) <= retained)
    )
    return grid.ifft(grid.fft(velocity) * mask)


def nonlinear_parts(
    grid: SpectralGrid, velocity: np.ndarray, vorticity: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    convection = np.einsum("j...,ij...->i...", velocity, grid.gradient(vorticity))
    stretching = np.einsum("j...,ij...->i...", vorticity, grid.gradient(velocity))
    return convection, stretching


def exact_sgs(grid: SpectralGrid, velocity: np.ndarray, delta: float) -> dict[str, np.ndarray]:
    vorticity = grid.curl(velocity)
    convection, stretching = nonlinear_parts(grid, velocity, vorticity)
    u_bar = grid.gaussian(velocity, delta)
    w_bar = grid.gaussian(vorticity, delta)
    convection_bar, stretching_bar = nonlinear_parts(grid, u_bar, w_bar)
    g_c = -grid.gaussian(convection, delta) + convection_bar
    g_s = grid.gaussian(stretching, delta) - stretching_bar
    return {"u": u_bar, "w": w_bar, "g": g_c + g_s}


def _correlation_lattice(
    grid: SpectralGrid,
    left: np.ndarray,
    right: np.ndarray,
    half_width: int,
    spacing: float,
) -> np.ndarray:
    cross = np.sum(np.conj(grid.fft(left)) * grid.fft(right), axis=0) / grid.n**6
    offsets = np.arange(-half_width, half_width + 1, dtype=float) * spacing
    px = np.exp(1j * np.outer(offsets, grid.kx[:, 0, 0]))
    py = np.exp(1j * np.outer(offsets, grid.ky[0, :, 0]))
    pz = np.exp(1j * np.outer(offsets, grid.kz[0, 0, :]))
    along_x = np.einsum("ax,xyz->ayz", px, cross, optimize=True)
    along_y = np.einsum("by,ayz->abz", py, along_x, optimize=True)
    result = np.einsum("cz,abz->abc", pz, along_y, optimize=True)
    return np.real_if_close(result, tol=1000).real


def _offsets(ns: int) -> np.ndarray:
    half = ns // 2
    return np.array(
        [
            (i, j, k)
            for i in range(-half, half + 1)
            for j in range(-half, half + 1)
            for k in range(-half, half + 1)
        ],
        dtype=int,
    )


def _weights(
    grid: SpectralGrid,
    current: np.ndarray,
    filtered_current: np.ndarray,
    ns: int,
    spacing: float,
) -> tuple[np.ndarray, dict[str, float]]:
    offsets = _offsets(ns)
    half = ns // 2
    auto = _correlation_lattice(grid, filtered_current, filtered_current, 2 * half, spacing)
    cross = _correlation_lattice(grid, current, filtered_current, half, spacing)
    differences = offsets[None, :, :] - offsets[:, None, :]
    matrix = auto[
        differences[..., 0] + 2 * half,
        differences[..., 1] + 2 * half,
        differences[..., 2] + 2 * half,
    ]
    vector = cross[
        offsets[:, 0] + half,
        offsets[:, 1] + half,
        offsets[:, 2] + half,
    ]
    size = len(offsets)
    kkt = np.empty((size + 1, size + 1), dtype=float)
    kkt[:size, :size] = 0.5 * (matrix + matrix.T)
    kkt[:size, size] = 1.0
    kkt[size, :size] = 1.0
    kkt[size, size] = 0.0
    rhs = np.concatenate((vector, [1.0]))
    solution, _, rank, singular = np.linalg.lstsq(kkt, rhs, rcond=DIAD_RCOND)
    return solution[:size], {
        "rank": float(rank),
        "condition": float(singular[0] / singular[-1]) if singular[-1] > 0.0 else float("inf"),
        "relative_residual": float(np.linalg.norm(kkt @ solution - rhs) / np.linalg.norm(rhs)),
    }


def _apply_stencil(
    grid: SpectralGrid,
    field: np.ndarray,
    weights: np.ndarray,
    ns: int,
    spacing: float,
) -> tuple[np.ndarray, np.ndarray]:
    half = ns // 2
    indices = np.arange(-half, half + 1, dtype=float)
    px = np.exp(1j * np.outer(indices * spacing, grid.kx[:, 0, 0]))
    py = np.exp(1j * np.outer(indices * spacing, grid.ky[0, :, 0]))
    pz = np.exp(1j * np.outer(indices * spacing, grid.kz[0, 0, :]))
    transfer = np.einsum(
        "abc,ax,by,cz->xyz", weights.reshape(ns, ns, ns), px, py, pz, optimize=True
    )
    return grid.ifft(grid.fft(field) * transfer), transfer


def diad(
    grid: SpectralGrid, resolved: np.ndarray, delta: float
) -> tuple[np.ndarray, dict[str, object]]:
    paper_width = np.sqrt(6.0) * delta
    spacing = paper_width / DIAD_RATIO
    current = resolved.copy()
    history: list[dict[str, float]] = []
    transfer = np.ones_like(grid.k2, dtype=complex)
    for update in range(1, DIAD_UPDATES + 1):
        filtered_current = grid.gaussian(current, delta)
        weights, solve = _weights(grid, current, filtered_current, DIAD_STENCIL, spacing)
        following, transfer = _apply_stencil(grid, resolved, weights, DIAD_STENCIL, spacing)
        consistency = float(
            np.mean(
                [
                    np.mean(np.abs(grid.gaussian(following, delta)[i] - resolved[i]))
                    / max(np.mean(np.abs(resolved[i])), np.finfo(float).tiny)
                    for i in range(3)
                ]
            )
        )
        history.append(
            {
                "update": float(update),
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


def structural_stress(grid: SpectralGrid, reconstructed: np.ndarray, delta: float) -> np.ndarray:
    reconstructed_bar = grid.gaussian(reconstructed, delta)
    stress = np.empty((3, 3, grid.n, grid.n, grid.n), dtype=float)
    for i in range(3):
        for j in range(3):
            stress[i, j] = (
                grid.gaussian(reconstructed[i] * reconstructed[j], delta)
                - reconstructed_bar[i] * reconstructed_bar[j]
            )
    return stress


def stress_torque(grid: SpectralGrid, stress: np.ndarray) -> np.ndarray:
    force = np.zeros((3, grid.n, grid.n, grid.n), dtype=float)
    for i in range(3):
        for j in range(3):
            force[i] -= grid.derivative(stress[i, j], j)
    return grid.curl(force)


def model_torques(
    grid: SpectralGrid, resolved: np.ndarray, delta: float
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    reconstructed, diad_diagnostics = diad(grid, resolved, delta)
    structural = structural_stress(grid, reconstructed, delta)
    trace = sum(structural[i, i] for i in range(3))
    deviatoric = structural.copy()
    for i in range(3):
        deviatoric[i, i] -= trace / 3.0

    gradient = grid.gradient(resolved)
    strain = 0.5 * (gradient + np.swapaxes(gradient, 0, 1))
    strain_magnitude = np.sqrt(2.0 * np.sum(strain * strain, axis=(0, 1)))
    paper_width = np.sqrt(6.0) * delta
    basis = -2.0 * paper_width**2 * strain_magnitude[None, None, ...] * strain
    numerator = float(np.mean(np.sum(deviatoric * basis, axis=(0, 1))))
    denominator = float(np.mean(np.sum(basis * basis, axis=(0, 1))))
    coefficient = numerator / denominator if denominator > np.finfo(float).tiny else 0.0
    ssev = coefficient * (basis - grid.gaussian(basis, delta))
    error = float(diad_diagnostics["consistency_error"])
    activation = max(0.0, 1.0 - DIAD_TOLERANCE / error) if error > 0.0 else 0.0
    torques = {
        "structural": stress_torque(grid, structural),
        "full_ssev": stress_torque(grid, structural + ssev),
        "sensed": stress_torque(grid, structural + activation * ssev),
    }
    diagnostics = {
        "coefficient": coefficient,
        "activation": activation,
        "consistency_error": error,
        "high_k_amplification": float(diad_diagnostics["high_k_amplification"]),
        "weight_sum_error": float(diad_diagnostics["history"][-1]["weight_sum_error"]),
        "kkt_condition": float(diad_diagnostics["history"][-1]["condition"]),
    }
    return torques, diagnostics


def shell_transfer(grid: SpectralGrid, vorticity: np.ndarray, torque: np.ndarray) -> np.ndarray:
    shell = np.rint(np.sqrt(grid.k2)).astype(int)
    density = np.real(np.sum(np.conj(grid.fft(vorticity)) * grid.fft(torque), axis=0)) / grid.n**6
    return np.array([np.sum(density[shell == k]) for k in range(grid.n // 3)])


def metrics(
    grid: SpectralGrid, vorticity: np.ndarray, exact: np.ndarray, model: np.ndarray
) -> dict[str, float]:
    exact_flat = exact.reshape(-1)
    model_flat = model.reshape(-1)
    exact_centered = exact_flat - np.mean(exact_flat)
    model_centered = model_flat - np.mean(model_flat)
    correlation = float(
        np.dot(exact_centered, model_centered)
        / (np.linalg.norm(exact_centered) * np.linalg.norm(model_centered))
    )
    exact_transfer = float(np.mean(np.sum(vorticity * exact, axis=0)))
    model_transfer = float(np.mean(np.sum(vorticity * model, axis=0)))
    exact_shell = shell_transfer(grid, vorticity, exact)
    model_shell = shell_transfer(grid, vorticity, model)
    return {
        "correlation": correlation,
        "relative_l2": float(np.linalg.norm(model - exact) / np.linalg.norm(exact)),
        "transfer_ratio": model_transfer / exact_transfer,
        "shell_error": float(
            np.linalg.norm(model_shell - exact_shell) / np.linalg.norm(exact_shell)
        ),
    }


def analyze_case(
    label: str,
    group: str,
    grid: SpectralGrid,
    velocity: np.ndarray,
    delta: float,
) -> dict[str, object]:
    exact = exact_sgs(grid, velocity, delta)
    torques, diagnostics = model_torques(grid, exact["u"], delta)
    return {
        "label": label,
        "group": group,
        "n": grid.n,
        "delta": delta,
        "delta_over_dns_h": delta / (2.0 * np.pi / grid.n),
        "diagnostics": diagnostics,
        "models": {
            name: metrics(grid, exact["w"], exact["g"], torque) for name, torque in torques.items()
        },
    }


def analytic_velocity(grid: SpectralGrid) -> np.ndarray:
    x = 2.0 * np.pi * np.arange(grid.n) / grid.n
    xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
    return np.array(
        (
            np.sin(yy) + 0.5 * np.cos(2.0 * zz) + 0.25 * np.sin(2.0 * yy + zz),
            0.7 * np.sin(zz) + 0.3 * np.cos(3.0 * xx) + 0.2 * np.cos(xx + 2.0 * zz),
            0.5 * np.sin(2.0 * xx) + 0.4 * np.cos(2.0 * yy) + 0.3 * np.sin(2.0 * xx + yy),
        )
    )


def formulation_checks() -> dict[str, float]:
    grid = SpectralGrid(24)
    h = 2.0 * np.pi / grid.n
    delta = 2.0 * h / np.sqrt(6.0)
    velocity = analytic_velocity(grid)
    resolved = grid.gaussian(velocity, delta)

    constant = np.full((3, grid.n, grid.n, grid.n), 1.25)
    constant_filter = norm(grid.gaussian(constant, delta) - constant) / norm(constant)
    composed = grid.gaussian(grid.gaussian(velocity, 0.12), 0.17)
    direct = grid.gaussian(velocity, np.sqrt(0.12**2 + 0.17**2))
    composition = norm(composed - direct) / norm(direct)
    mapping = float(
        np.max(
            np.abs(
                np.exp(-((np.sqrt(6.0) * delta) ** 2) * grid.k2 / 24.0)
                - np.exp(-(delta**2) * grid.k2 / 4.0)
            )
        )
    )

    reconstructed, _ = diad(grid, resolved, delta)
    shift = np.array((0.71, -0.37, 0.19))[:, None, None, None]
    reconstructed_shifted, _ = diad(grid, resolved + shift, delta)
    galilean_velocity = norm(reconstructed_shifted - reconstructed - shift) / norm(reconstructed)
    stress = structural_stress(grid, reconstructed, delta)
    stress_shifted = structural_stress(grid, reconstructed_shifted, delta)
    galilean_stress = norm(stress_shifted - stress) / norm(stress)
    divergence = sum(grid.derivative(reconstructed[i], i) for i in range(3))
    divergence_relative = norm(divergence) / norm(grid.curl(reconstructed))

    w_star = grid.curl(reconstructed)
    u_hat = grid.gaussian(reconstructed, delta)
    w_hat = grid.gaussian(w_star, delta)
    c_star, s_star = nonlinear_parts(grid, reconstructed, w_star)
    c_hat, s_hat = nonlinear_parts(grid, u_hat, w_hat)
    vorticity_form = -grid.gaussian(c_star, delta) + c_hat + grid.gaussian(s_star, delta) - s_hat
    stress_form = stress_torque(grid, stress)
    stress_vorticity_identity = norm(stress_form - vorticity_form) / norm(stress_form)

    base_torque = model_torques(grid, resolved, delta)[0]["sensed"]
    scale = 2.75
    scaled_torque = model_torques(grid, scale * resolved, delta)[0]["sensed"]
    amplitude_scaling = norm(scaled_torque - scale**2 * base_torque) / norm(scale**2 * base_torque)
    uniform_torque = norm(model_torques(grid, constant, delta)[0]["sensed"])
    return {
        "constant_filter_relative": constant_filter,
        "gaussian_composition_relative": composition,
        "yuan_width_mapping_absolute": mapping,
        "galilean_velocity_relative": galilean_velocity,
        "galilean_stress_relative": galilean_stress,
        "divergence_relative": divergence_relative,
        "stress_vorticity_identity_relative": stress_vorticity_identity,
        "quadratic_amplitude_scaling_relative": amplitude_scaling,
        "uniform_torque_absolute": uniform_torque,
    }


def checkpoint_cases(path: Path, group: str) -> list[dict[str, object]]:
    with np.load(path, allow_pickle=False) as saved:
        snapshots = np.asarray(saved["snapshots"], dtype=float)
        steps = np.asarray(saved["snapshot_steps"], dtype=int)
    grid = SpectralGrid(int(snapshots.shape[-1]))
    h = 2.0 * np.pi / grid.n
    cases = []
    for snapshot, step in zip(snapshots, steps, strict=True):
        for multiplier in (3.5, 5.0, 7.0):
            cases.append(
                analyze_case(
                    f"{group}_step_{step}_delta_{multiplier:g}h",
                    group,
                    grid,
                    snapshot,
                    multiplier * h,
                )
            )
    return cases


def summarize(cases: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    groups = sorted({str(case["group"]) for case in cases})
    for group in groups:
        selected = [case for case in cases if case["group"] == group]
        result[group] = {}
        for model in ("structural", "full_ssev", "sensed"):
            result[group][model] = {}
            for metric in ("correlation", "relative_l2", "transfer_ratio", "shell_error"):
                values = np.asarray([case["models"][model][metric] for case in selected])
                result[group][model][metric] = {
                    "mean": float(np.mean(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
    return result


def plot_evidence(cases: list[dict[str, object]], output: Path) -> None:
    groups = ["AGARD development", "stationary HIT development", "transient HIT holdout"]
    models = ["structural", "full_ssev", "sensed"]
    labels = {"structural": "Structural DIAD", "full_ssev": "Full SSEV", "sensed": "Sensed DIAD"}
    colors = {"structural": "#8a99a8", "full_ssev": "#d9973b", "sensed": "#286f9b"}
    metrics_to_plot = [
        ("correlation", "Torque correlation", (0.0, 1.05), None),
        ("transfer_ratio", "Mean transfer ratio", (0.0, 2.15), 1.0),
        ("shell_error", "Shell-transfer error", (0.0, 1.05), None),
    ]
    rng = np.random.default_rng(20260815)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    for axis, (metric, title, limits, reference) in zip(axes, metrics_to_plot, strict=True):
        for group_index, group in enumerate(groups):
            selected = [case for case in cases if case["group"] == group]
            for model_index, model in enumerate(models):
                values = np.asarray([case["models"][model][metric] for case in selected])
                x_center = group_index + (model_index - 1) * 0.22
                jitter = rng.uniform(-0.035, 0.035, size=len(values))
                axis.scatter(
                    x_center + jitter,
                    values,
                    s=18,
                    color=colors[model],
                    alpha=0.35,
                    linewidths=0,
                )
                axis.scatter(
                    [x_center],
                    [np.mean(values)],
                    s=70,
                    color=colors[model],
                    edgecolor="#20252a",
                    linewidth=0.7,
                    zorder=3,
                    label=labels[model] if group_index == 0 and metric == "correlation" else None,
                )
        if reference is not None:
            axis.axhline(reference, color="#30363b", linewidth=1.0, linestyle="--")
        axis.set_title(title, fontsize=11, color="#20252a")
        axis.set_ylim(*limits)
        axis.set_xticks(
            range(len(groups)), ["AGARD\ndev.", "Stationary HIT\ndev.", "Transient HIT\nholdout"]
        )
        axis.grid(axis="y", color="#d8dde2", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    fig.legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(
        "A-priori SGS performance by validation group", fontsize=14, color="#20252a", y=1.16
    )
    fig.text(
        0.5,
        -0.04,
        "Small points are individual cases; outlined points are group means. Transfer-ratio reference = 1.",
        ha="center",
        fontsize=9,
        color="#59636d",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_checks(checks: dict[str, float], output: Path) -> None:
    labels = [key.replace("_", " ") for key in checks]
    values = np.maximum(np.asarray(list(checks.values())), 1.0e-18)
    fig, axis = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    positions = np.arange(len(labels))
    axis.barh(positions, values, color="#286f9b", edgecolor="#174864", linewidth=0.7)
    axis.axvline(1.0e-10, color="#d9973b", linestyle="--", linewidth=1.2, label="Gate A tolerance")
    axis.set_xscale("log")
    axis.set_xlim(1.0e-18, max(1.0e-8, float(np.max(values)) * 10.0))
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Relative or absolute residual (log scale)")
    axis.set_title("Gate A formulation residuals", fontsize=13, color="#20252a")
    axis.grid(axis="x", color="#d8dde2", linewidth=0.7, which="both")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="lower right")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agard", type=Path, required=True)
    parser.add_argument("--stationary-checkpoint", type=Path, required=True)
    parser.add_argument("--holdout-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args()

    checks = formulation_checks()
    agard_grid = SpectralGrid(128)
    agard = truncate(agard_grid, load_agard(args.agard), 63)
    cases = [
        analyze_case(f"AGARD_delta_{delta:g}", "AGARD development", agard_grid, agard, delta)
        for delta in (0.15, 0.20)
    ]
    cases.extend(checkpoint_cases(args.stationary_checkpoint, "stationary HIT development"))
    cases.extend(checkpoint_cases(args.holdout_checkpoint, "transient HIT holdout"))

    result = {
        "gate": "A",
        "frozen_model": {
            "stencil": DIAD_STENCIL,
            "filter_to_spacing_ratio": DIAD_RATIO,
            "updates": DIAD_UPDATES,
            "consistency_tolerance": DIAD_TOLERANCE,
            "svd_rcond": DIAD_RCOND,
            "activation": "max(0, 1 - epsilon/e)",
        },
        "formulation_checks": checks,
        "formulation_gate_tolerance": 1.0e-10,
        "formulation_gate_pass": bool(max(checks.values()) < 1.0e-10),
        "cases": cases,
        "summary": summarize(cases),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    plot_evidence(cases, args.figure_dir / "stage_4a_apriori_metrics.png")
    plot_checks(checks, args.figure_dir / "stage_4a_formulation_residuals.png")


if __name__ == "__main__":
    main()
