#!/usr/bin/env python3
"""Gate B pilot: paired dealiased DNS/LES Taylor--Green evolution.

This is research-only code.  It exercises the frozen Gate-A closure in a
periodic a-posteriori calculation without importing or modifying OpenONDA's
production solvers.  The default 48/24 run is a numerical-method pilot, not a
publication-resolution Gate-B result.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/openonda_stage4b_matplotlib")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_4a_formulation import SpectralGrid, model_torques  # noqa: E402

MODELS = ("no_sgs", "structural", "full_ssev", "sensed")
COLORS = {
    "filtered_dns": "#20252a",
    "no_sgs": "#9b6a45",
    "structural": "#8a99a8",
    "full_ssev": "#d9973b",
    "sensed": "#286f9b",
}
LABELS = {
    "filtered_dns": "Filtered DNS",
    "no_sgs": "No SGS",
    "structural": "Structural DIAD",
    "full_ssev": "Full SSEV",
    "sensed": "Sensed DIAD",
}


class VorticitySolver:
    """Two-thirds dealiased Fourier discretization on [0, 2 pi)^3."""

    def __init__(self, n: int, kinematic_viscosity: float) -> None:
        self.grid = SpectralGrid(n)
        self.kinematic_viscosity = kinematic_viscosity
        cutoff = n // 3
        self.mask = (
            (np.abs(self.grid.kx) < cutoff)
            & (np.abs(self.grid.ky) < cutoff)
            & (np.abs(self.grid.kz) < cutoff)
        )
        self.nonzero = self.grid.k2 > 0.0

    def project(self, vector: np.ndarray) -> np.ndarray:
        hat = self.grid.fft(vector)
        dot = self.grid.kx * hat[0] + self.grid.ky * hat[1] + self.grid.kz * hat[2]
        for i, wave in enumerate((self.grid.kx, self.grid.ky, self.grid.kz)):
            correction = np.zeros_like(dot)
            correction[self.nonzero] = (
                wave[self.nonzero] * dot[self.nonzero] / self.grid.k2[self.nonzero]
            )
            hat[i] -= correction
        hat *= self.mask
        return self.grid.ifft(hat)

    def velocity(self, vorticity: np.ndarray) -> np.ndarray:
        hat = self.grid.fft(vorticity)
        cross = np.array(
            (
                self.grid.ky * hat[2] - self.grid.kz * hat[1],
                self.grid.kz * hat[0] - self.grid.kx * hat[2],
                self.grid.kx * hat[1] - self.grid.ky * hat[0],
            )
        )
        velocity_hat = np.zeros_like(cross)
        velocity_hat[:, self.nonzero] = 1j * cross[:, self.nonzero] / self.grid.k2[self.nonzero]
        velocity_hat *= self.mask
        return self.grid.ifft(velocity_hat)

    def dealias(self, vector: np.ndarray) -> np.ndarray:
        return self.grid.ifft(self.grid.fft(vector) * self.mask)

    def rhs(
        self,
        vorticity: np.ndarray,
        model: str,
        gaussian_delta: float,
    ) -> np.ndarray:
        velocity = self.velocity(vorticity)
        gradient_w = self.grid.gradient(vorticity)
        gradient_u = self.grid.gradient(velocity)
        convection = np.einsum("j...,ij...->i...", velocity, gradient_w)
        stretching = np.einsum("j...,ij...->i...", vorticity, gradient_u)
        laplacian = self.grid.ifft(-self.grid.k2 * self.grid.fft(vorticity))
        result = -convection + stretching + self.kinematic_viscosity * laplacian
        if model != "no_sgs":
            result += model_torques(self.grid, velocity, gaussian_delta)[0][model]
        return self.project(self.dealias(result))

    def heun_step(
        self,
        vorticity: np.ndarray,
        time_step_size: float,
        model: str,
        gaussian_delta: float,
    ) -> np.ndarray:
        first = self.rhs(vorticity, model, gaussian_delta)
        predictor = self.project(vorticity + time_step_size * first)
        second = self.rhs(predictor, model, gaussian_delta)
        return self.project(vorticity + 0.5 * time_step_size * (first + second))


def taylor_green(grid: SpectralGrid) -> np.ndarray:
    x = 2.0 * np.pi * np.arange(grid.n) / grid.n
    xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
    return np.array(
        (
            np.sin(xx) * np.cos(yy) * np.cos(zz),
            -np.cos(xx) * np.sin(yy) * np.cos(zz),
            np.zeros_like(xx),
        )
    )


def coarse_reference(
    dns_solver: VorticitySolver,
    dns_vorticity: np.ndarray,
    les_n: int,
    gaussian_delta: float,
) -> np.ndarray:
    if dns_solver.grid.n % les_n != 0:
        raise ValueError("DNS grid must be an integer multiple of the LES grid")
    ratio = dns_solver.grid.n // les_n
    transfer = np.exp(-(gaussian_delta**2) * dns_solver.grid.k2 / 4.0)
    cutoff = les_n // 3
    mask = (
        (np.abs(dns_solver.grid.kx) < cutoff)
        & (np.abs(dns_solver.grid.ky) < cutoff)
        & (np.abs(dns_solver.grid.kz) < cutoff)
    )
    filtered = dns_solver.grid.ifft(dns_solver.grid.fft(dns_vorticity) * transfer * mask)
    return filtered[:, ::ratio, ::ratio, ::ratio]


def projected_force_from_torque(solver: VorticitySolver, torque: np.ndarray) -> np.ndarray:
    """Recover the divergence-free momentum force whose curl is ``torque``."""
    return solver.velocity(torque)


def kinetic_energy_spectrum(solver: VorticitySolver, vorticity: np.ndarray) -> np.ndarray:
    velocity_hat = solver.grid.fft(solver.velocity(vorticity))
    density = 0.5 * np.sum(np.abs(velocity_hat) ** 2, axis=0) / solver.grid.n**6
    shell = np.rint(np.sqrt(solver.grid.k2)).astype(int)
    return np.array([np.sum(density[shell == k]) for k in range(solver.grid.n // 3)])


def diagnostics(
    solver: VorticitySolver,
    vorticity: np.ndarray,
    model: str,
    gaussian_delta: float,
) -> dict[str, float]:
    velocity = solver.velocity(vorticity)
    total_kinetic_energy = 0.5 * float(np.mean(np.sum(velocity * velocity, axis=0)))
    total_enstrophy = 0.5 * float(np.mean(np.sum(vorticity * vorticity, axis=0)))
    velocity_hat = solver.grid.fft(velocity)
    modal_energy = 0.5 * np.sum(np.abs(velocity_hat) ** 2, axis=0) / solver.grid.n**6
    radius = np.sqrt(solver.grid.k2)
    high_k = float(np.sum(modal_energy[radius >= 0.7 * (solver.grid.n // 3)]))
    total = float(np.sum(modal_energy))
    divergence = sum(solver.grid.derivative(vorticity[i], i) for i in range(3))
    sgs_power = 0.0
    enstrophy_transfer = 0.0
    activation = 0.0
    condition = 0.0
    if model != "no_sgs":
        torques, model_diagnostics = model_torques(solver.grid, velocity, gaussian_delta)
        torque = torques[model]
        force = projected_force_from_torque(solver, torque)
        sgs_power = float(np.mean(np.sum(velocity * force, axis=0)))
        enstrophy_transfer = float(np.mean(np.sum(vorticity * torque, axis=0)))
        activation = {
            "structural": 0.0,
            "full_ssev": 1.0,
            "sensed": float(model_diagnostics["activation"]),
        }[model]
        condition = float(model_diagnostics["kkt_condition"])
    return {
        "total_kinetic_energy": total_kinetic_energy,
        "total_enstrophy": total_enstrophy,
        "viscous_kinetic_energy_rate": -2.0 * solver.kinematic_viscosity * total_enstrophy,
        "sgs_power": sgs_power,
        "enstrophy_transfer": enstrophy_transfer,
        "high_k_energy_fraction": high_k / max(total, np.finfo(float).tiny),
        "divergence_relative": float(
            np.sqrt(np.mean(divergence * divergence))
            / max(np.sqrt(np.mean(vorticity * vorticity)), np.finfo(float).tiny)
        ),
        "activation": activation,
        "kkt_condition": condition,
    }


def field_error(model: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(model - reference) / np.linalg.norm(reference))


def integrate_budget(records: list[dict[str, float]]) -> float:
    time = np.asarray([record["time"] for record in records])
    power = np.asarray(
        [record["viscous_kinetic_energy_rate"] + record["sgs_power"] for record in records]
    )
    predicted = records[0]["total_kinetic_energy"] + float(np.trapezoid(power, time))
    scale = max(
        abs(records[0]["total_kinetic_energy"] - records[-1]["total_kinetic_energy"]), 1.0e-14
    )
    return abs(records[-1]["total_kinetic_energy"] - predicted) / scale


def integrated_relative_error(
    records: list[dict[str, float]],
    reference: list[dict[str, float]],
    quantity: str,
) -> float:
    time = np.asarray([record["time"] for record in records])
    model_values = np.asarray([record[quantity] for record in records])
    reference_values = np.asarray([record[quantity] for record in reference])
    numerator = float(np.trapezoid(np.abs(model_values - reference_values), time))
    denominator = float(np.trapezoid(np.abs(reference_values), time))
    return numerator / max(denominator, np.finfo(float).tiny)


def plot_histories(result: dict[str, object], output: Path) -> None:
    histories = result["histories"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), constrained_layout=True)
    quantities = (
        ("total_kinetic_energy", "Resolved kinetic total_kinetic_energy"),
        ("total_enstrophy", "Resolved total_enstrophy"),
        ("high_k_energy_fraction", "High-k total_kinetic_energy fraction"),
    )
    for axis, (quantity, title) in zip(axes, quantities, strict=True):
        for model in ("filtered_dns", *MODELS):
            records = histories[model]
            axis.plot(
                [record["time"] for record in records],
                [record[quantity] for record in records],
                color=COLORS[model],
                label=LABELS[model],
                linewidth=1.8 if model in ("filtered_dns", "sensed") else 1.25,
            )
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("Time")
        axis.grid(color="#d8dde2", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Gate B Taylor--Green pilot", fontsize=14, color="#20252a")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_spectra(result: dict[str, object], output: Path) -> None:
    spectra = result["final_spectra"]
    fig, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for model in ("filtered_dns", *MODELS):
        values = np.asarray(spectra[model])
        wave = np.arange(len(values))
        positive = (wave > 0) & (values > 0.0)
        axis.loglog(
            wave[positive],
            values[positive],
            color=COLORS[model],
            label=LABELS[model],
            linewidth=1.9 if model in ("filtered_dns", "sensed") else 1.3,
            marker="o",
            markersize=3,
        )
    axis.set_xlabel("Wavenumber shell k")
    axis.set_ylabel("E(k)")
    axis.set_title("Final resolved total_kinetic_energy spectra")
    axis.grid(color="#d8dde2", linewidth=0.7, which="both")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_model_diagnostics(result: dict[str, object], output: Path) -> None:
    histories = result["histories"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    quantities = (
        ("relative_vorticity_error", "Vorticity relative L2 error"),
        ("spectral_relative_l2", "Energy-spectrum relative L2 error"),
        ("sgs_power", "SGS contribution to dE/dt"),
        ("activation", "SSEV activation"),
    )
    for axis, (quantity, title) in zip(axes.flat, quantities, strict=True):
        for model in MODELS:
            records = histories[model]
            axis.plot(
                [record["time"] for record in records],
                [record[quantity] for record in records],
                color=COLORS[model],
                label=LABELS[model],
                linewidth=1.8 if model == "sensed" else 1.25,
            )
        if quantity == "sgs_power":
            axis.axhline(0.0, color="#59636d", linewidth=0.8, linestyle="--")
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("Time")
        axis.grid(color="#d8dde2", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Gate B pilot model diagnostics", fontsize=14, color="#20252a")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.dns_n != 2 * args.les_n:
        raise ValueError("the pilot currently requires dns_n = 2 * les_n")
    dns = VorticitySolver(args.dns_n, args.kinematic_viscosity)
    les = VorticitySolver(args.les_n, args.kinematic_viscosity)
    paper_width = 2.0 * (2.0 * np.pi / args.les_n)
    gaussian_delta = paper_width / np.sqrt(6.0)
    dns_vorticity = dns.project(dns.grid.curl(taylor_green(dns.grid)))
    initial_reference = coarse_reference(dns, dns_vorticity, args.les_n, gaussian_delta)
    states = {model: les.project(initial_reference.copy()) for model in MODELS}
    save_every = max(1, int(round(args.save_interval / args.time_step_size)))
    steps = int(round(args.end_time / args.time_step_size))
    if abs(steps * args.time_step_size - args.end_time) > 1.0e-12:
        raise ValueError("end_time must be an integer multiple of dt")

    histories: dict[str, list[dict[str, float]]] = {
        model: [] for model in ("filtered_dns", *MODELS)
    }
    final_reference = initial_reference
    for step in range(steps + 1):
        if step % save_every == 0 or step == steps:
            time = step * args.time_step_size
            final_reference = coarse_reference(dns, dns_vorticity, args.les_n, gaussian_delta)
            reference_record = diagnostics(les, final_reference, "no_sgs", gaussian_delta)
            reference_spectrum = kinetic_energy_spectrum(les, final_reference)
            reference_record.update(
                {
                    "time": time,
                    "relative_vorticity_error": 0.0,
                    "spectral_relative_l2": 0.0,
                }
            )
            histories["filtered_dns"].append(reference_record)
            for model, state in states.items():
                record = diagnostics(les, state, model, gaussian_delta)
                model_spectrum = kinetic_energy_spectrum(les, state)
                record.update(
                    {
                        "time": time,
                        "relative_vorticity_error": field_error(state, final_reference),
                        "spectral_relative_l2": field_error(model_spectrum, reference_spectrum),
                    }
                )
                histories[model].append(record)
        if step == steps:
            break
        dns_vorticity = dns.heun_step(dns_vorticity, args.time_step_size, "no_sgs", gaussian_delta)
        for model in MODELS:
            states[model] = les.heun_step(states[model], args.time_step_size, model, gaussian_delta)
            if not np.all(np.isfinite(states[model])):
                raise FloatingPointError(f"non-finite state in {model} at step {step + 1}")

    final_spectra = {"filtered_dns": kinetic_energy_spectrum(les, final_reference).tolist()}
    final_spectra.update(
        {model: kinetic_energy_spectrum(les, state).tolist() for model, state in states.items()}
    )
    summary = {}
    reference_final = histories["filtered_dns"][-1]
    for model in MODELS:
        final = histories[model][-1]
        summary[model] = {
            "final_total_kinetic_energy_relative_error": abs(
                final["total_kinetic_energy"] - reference_final["total_kinetic_energy"]
            )
            / reference_final["total_kinetic_energy"],
            "final_total_enstrophy_relative_error": abs(
                final["total_enstrophy"] - reference_final["total_enstrophy"]
            )
            / reference_final["total_enstrophy"],
            "final_vorticity_relative_l2": final["relative_vorticity_error"],
            "max_high_k_energy_fraction": max(
                record["high_k_energy_fraction"] for record in histories[model]
            ),
            "max_divergence_relative": max(
                record["divergence_relative"] for record in histories[model]
            ),
            "energy_budget_relative_residual": integrate_budget(histories[model]),
            "time_integrated_total_kinetic_energy_relative_error": integrated_relative_error(
                histories[model], histories["filtered_dns"], "total_kinetic_energy"
            ),
            "time_integrated_total_enstrophy_relative_error": integrated_relative_error(
                histories[model], histories["filtered_dns"], "total_enstrophy"
            ),
            "time_mean_spectral_relative_l2": float(
                np.mean([record["spectral_relative_l2"] for record in histories[model]])
            ),
            "mean_activation": float(
                np.mean([record["activation"] for record in histories[model]])
            ),
            "max_kkt_condition": max(record["kkt_condition"] for record in histories[model]),
        }
    return {
        "gate": "B pilot",
        "qualification_status": "NOT A GATE-B PASS: reduced-resolution method pilot",
        "configuration": {
            "flow": "Taylor-Green vortex",
            "dns_n": args.dns_n,
            "les_n": args.les_n,
            "kinematic_viscosity": args.kinematic_viscosity,
            "time_step_size": args.time_step_size,
            "end_time": args.end_time,
            "integrator": "Heun RK2",
            "dealiasing": "strict two-thirds component cutoff",
            "paper_filter_width_over_les_h": 2.0,
            "gaussian_delta": gaussian_delta,
        },
        "histories": histories,
        "final_spectra": final_spectra,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dns-n", type=int, default=48)
    parser.add_argument("--les-n", type=int, default=24)
    parser.add_argument("--kinematic-viscosity", type=float, default=1.0 / 200.0)
    parser.add_argument("--time-step-size", dest="time_step_size", type=float, default=0.01)
    parser.add_argument("--end-time", type=float, default=0.5)
    parser.add_argument("--save-interval", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    plot_histories(result, args.figure_dir / "stage_4b_pilot_histories.png")
    plot_spectra(result, args.figure_dir / "stage_4b_pilot_spectra.png")
    plot_model_diagnostics(result, args.figure_dir / "stage_4b_pilot_diagnostics.png")


if __name__ == "__main__":
    main()
