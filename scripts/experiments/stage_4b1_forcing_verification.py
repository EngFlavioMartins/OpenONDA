#!/usr/bin/env python3
r"""Gate B.1a: verify deterministic nested-grid forcing and its energy balance.

The forcing is a divergence-free Ornstein--Uhlenbeck process supported only on
the shared low Fourier modes 1 <= |k| <= 2.  One realization is generated on
the LES grid and embedded exactly on the reference grid, so every paired run
receives the same external acceleration history.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/openonda_stage4b1_matplotlib")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_4a_formulation import SpectralGrid, norm  # noqa: E402
from stage_4b_spectral_pilot import VorticitySolver  # noqa: E402

INK = "#20252a"
BLUE = "#286f9b"
GOLD = "#d9973b"
GREY = "#8a99a8"


def divergence_free_band_noise(
    grid: SpectralGrid,
    rng: np.random.Generator,
    target_rms: float,
) -> np.ndarray:
    noise = rng.standard_normal((3, grid.n, grid.n, grid.n))
    hat = grid.fft(noise)
    radius = np.sqrt(grid.k2)
    band = (radius >= 1.0) & (radius <= 2.0)
    hat *= band
    nonzero = grid.k2 > 0.0
    dot = grid.kx * hat[0] + grid.ky * hat[1] + grid.kz * hat[2]
    for component, wave in enumerate((grid.kx, grid.ky, grid.kz)):
        correction = np.zeros_like(dot)
        correction[nonzero] = wave[nonzero] * dot[nonzero] / grid.k2[nonzero]
        hat[component] -= correction
    field = grid.ifft(hat)
    rms = float(np.sqrt(np.mean(np.sum(field * field, axis=0))))
    return field * (target_rms / rms)


def embed_periodic_field(
    field: np.ndarray,
    target_n: int,
    inverse_gaussian_delta: float | None = None,
) -> np.ndarray:
    """Evaluate a low-mode field on a finer grid, optionally undoing a filter."""
    source_n = field.shape[-1]
    if target_n % source_n != 0:
        raise ValueError("target grid must be an integer multiple of source grid")
    source_grid = SpectralGrid(source_n)
    source_hat = source_grid.fft(field)
    target_hat = np.zeros((3, target_n, target_n, target_n), dtype=complex)
    radius = np.sqrt(source_grid.k2)
    for i, j, k in np.argwhere((radius >= 1.0) & (radius <= 2.0)):
        wave_i = int(source_grid.kx[i, j, k])
        wave_j = int(source_grid.ky[i, j, k])
        wave_k = int(source_grid.kz[i, j, k])
        inverse_transfer = 1.0
        if inverse_gaussian_delta is not None:
            wave_squared = wave_i**2 + wave_j**2 + wave_k**2
            inverse_transfer = np.exp(inverse_gaussian_delta**2 * wave_squared / 4.0)
        target_hat[:, wave_i % target_n, wave_j % target_n, wave_k % target_n] = (
            source_hat[:, i, j, k] * (target_n / source_n) ** 3 * inverse_transfer
        )
    return SpectralGrid.ifft(target_hat)


class ForcingHistory:
    """Piecewise-linear realization of a low-mode OU acceleration process."""

    def __init__(
        self,
        n: int,
        anchor_time_step_size: float,
        end_time: float,
        correlation_time: float,
        target_rms: float,
        seed: int,
    ) -> None:
        self.n = n
        self.anchor_time_step_size = anchor_time_step_size
        self.end_time = end_time
        self.correlation_time = correlation_time
        self.grid = SpectralGrid(n)
        rng = np.random.default_rng(seed)
        count = int(round(end_time / anchor_time_step_size)) + 1
        rho = np.exp(-anchor_time_step_size / correlation_time)
        fields = [divergence_free_band_noise(self.grid, rng, target_rms)]
        for _ in range(1, count):
            innovation = divergence_free_band_noise(self.grid, rng, target_rms)
            fields.append(rho * fields[-1] + np.sqrt(1.0 - rho**2) * innovation)
        self.fields = np.asarray(fields)
        self._grid_cache: dict[int, np.ndarray] = {n: self.fields}
        self._reference_cache: dict[tuple[int, float], np.ndarray] = {}

    def on_grid(self, n: int) -> np.ndarray:
        if n not in self._grid_cache:
            self._grid_cache[n] = np.asarray(
                [embed_periodic_field(field, n) for field in self.fields]
            )
        return self._grid_cache[n]

    def at(self, time: float, n: int) -> np.ndarray:
        fields = self.on_grid(n)
        position = min(max(time / self.anchor_time_step_size, 0.0), len(fields) - 1.0)
        lower = min(int(np.floor(position)), len(fields) - 1)
        upper = min(lower + 1, len(fields) - 1)
        fraction = position - lower
        return (1.0 - fraction) * fields[lower] + fraction * fields[upper]

    def reference_on_grid(self, n: int, gaussian_delta: float) -> np.ndarray:
        """Raw reference force whose Gaussian-filtered value is the LES force."""
        key = (n, gaussian_delta)
        if key not in self._reference_cache:
            self._reference_cache[key] = np.asarray(
                [embed_periodic_field(field, n, gaussian_delta) for field in self.fields]
            )
        return self._reference_cache[key]

    def reference_at(self, time: float, n: int, gaussian_delta: float) -> np.ndarray:
        fields = self.reference_on_grid(n, gaussian_delta)
        position = min(max(time / self.anchor_time_step_size, 0.0), len(fields) - 1.0)
        lower = min(int(np.floor(position)), len(fields) - 1)
        upper = min(lower + 1, len(fields) - 1)
        fraction = position - lower
        return (1.0 - fraction) * fields[lower] + fraction * fields[upper]


def random_isotropic_velocity(n: int, seed: int, target_rms: float = 1.0) -> np.ndarray:
    grid = SpectralGrid(n)
    rng = np.random.default_rng(seed)
    hat = grid.fft(rng.standard_normal((3, n, n, n)))
    radius = np.sqrt(grid.k2)
    cutoff = n // 3
    envelope = radius**2 * np.exp(-0.5 * (radius / 3.0) ** 2)
    envelope[0, 0, 0] = 0.0
    hat *= envelope * (
        (np.abs(grid.kx) < cutoff) & (np.abs(grid.ky) < cutoff) & (np.abs(grid.kz) < cutoff)
    )
    nonzero = grid.k2 > 0.0
    dot = grid.kx * hat[0] + grid.ky * hat[1] + grid.kz * hat[2]
    for component, wave in enumerate((grid.kx, grid.ky, grid.kz)):
        correction = np.zeros_like(dot)
        correction[nonzero] = wave[nonzero] * dot[nonzero] / grid.k2[nonzero]
        hat[component] -= correction
    velocity = grid.ifft(hat)
    rms = float(np.sqrt(np.mean(np.sum(velocity * velocity, axis=0))))
    return velocity * (target_rms / rms)


def forced_rhs(
    solver: VorticitySolver,
    vorticity: np.ndarray,
    gaussian_delta: float,
    acceleration: np.ndarray,
    model: str = "no_sgs",
) -> np.ndarray:
    base = solver.rhs(vorticity, model, gaussian_delta)
    return solver.project(base + solver.grid.curl(acceleration))


def forced_heun_step(
    solver: VorticitySolver,
    vorticity: np.ndarray,
    gaussian_delta: float,
    time_step_size: float,
    acceleration_start: np.ndarray,
    acceleration_end: np.ndarray,
    model: str = "no_sgs",
) -> np.ndarray:
    first = forced_rhs(solver, vorticity, gaussian_delta, acceleration_start, model)
    predictor = solver.project(vorticity + time_step_size * first)
    second = forced_rhs(solver, predictor, gaussian_delta, acceleration_end, model)
    return solver.project(vorticity + 0.5 * time_step_size * (first + second))


def state_diagnostics(
    solver: VorticitySolver,
    vorticity: np.ndarray,
    acceleration: np.ndarray,
) -> dict[str, float]:
    velocity = solver.velocity(vorticity)
    energy = 0.5 * float(np.mean(np.sum(velocity * velocity, axis=0)))
    enstrophy = 0.5 * float(np.mean(np.sum(vorticity * vorticity, axis=0)))
    forcing_power = float(np.mean(np.sum(velocity * acceleration, axis=0)))
    return {
        "energy": energy,
        "enstrophy": enstrophy,
        "forcing_power": forcing_power,
        "viscous_power": -2.0 * solver.viscosity * enstrophy,
    }


def cumulative_trapezoid(values: np.ndarray, time: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values)
    result[1:] = np.cumsum(0.5 * (values[1:] + values[:-1]) * np.diff(time))
    return result


def run_budget_case(
    n: int,
    viscosity: float,
    time_step_size: float,
    end_time: float,
    initial_velocity: np.ndarray,
    forcing: ForcingHistory,
) -> dict[str, object]:
    solver = VorticitySolver(n, viscosity)
    delta = 2.0 * (2.0 * np.pi / n) / np.sqrt(6.0)
    vorticity = solver.project(solver.grid.curl(initial_velocity))
    steps = int(round(end_time / time_step_size))
    records: list[dict[str, float]] = []
    for step in range(steps + 1):
        time = step * time_step_size
        acceleration = forcing.at(time, n)
        record = state_diagnostics(solver, vorticity, acceleration)
        record["time"] = time
        records.append(record)
        if step < steps:
            vorticity = forced_heun_step(
                solver,
                vorticity,
                delta,
                time_step_size,
                acceleration,
                forcing.at(time + time_step_size, n),
            )
    time = np.asarray([record["time"] for record in records])
    energy = np.asarray([record["energy"] for record in records])
    forcing_power = np.asarray([record["forcing_power"] for record in records])
    viscous_power = np.asarray([record["viscous_power"] for record in records])
    predicted_change = cumulative_trapezoid(forcing_power + viscous_power, time)
    actual_change = energy - energy[0]
    scale = float(np.trapezoid(np.abs(forcing_power) + np.abs(viscous_power), time))
    residual = abs(actual_change[-1] - predicted_change[-1]) / max(scale, 1.0e-15)
    return {
        "dt": time_step_size,
        "budget_relative_residual": residual,
        "records": records,
        "actual_energy_change": actual_change.tolist(),
        "predicted_energy_change": predicted_change.tolist(),
    }


def autocorrelation(fields: np.ndarray, max_lag: int) -> np.ndarray:
    result = []
    for lag in range(max_lag + 1):
        products = [
            float(np.mean(np.sum(fields[index] * fields[index + lag], axis=0)))
            for index in range(len(fields) - lag)
        ]
        result.append(float(np.mean(products)))
    return np.asarray(result) / result[0]


def forcing_checks(
    forcing: ForcingHistory, fine_n: int, gaussian_delta: float
) -> dict[str, object]:
    coarse = forcing.on_grid(forcing.n)
    fine_raw = forcing.reference_on_grid(fine_n, gaussian_delta)
    fine_grid = SpectralGrid(fine_n)
    fine = np.asarray([fine_grid.gaussian(field, gaussian_delta) for field in fine_raw])
    ratio = fine_n // forcing.n
    pairing = max(
        norm(fine[index, :, ::ratio, ::ratio, ::ratio] - coarse[index])
        / max(norm(coarse[index]), np.finfo(float).tiny)
        for index in range(len(coarse))
    )
    divergence = []
    leakage = []
    means = []
    component_energy = np.zeros(3)
    for field in coarse:
        gradient_divergence = sum(
            forcing.grid.derivative(field[component], component) for component in range(3)
        )
        divergence.append(norm(gradient_divergence) / norm(field))
        hat = forcing.grid.fft(field)
        radius = np.sqrt(forcing.grid.k2)
        energy_hat = np.sum(np.abs(hat) ** 2, axis=0)
        outside = (radius < 1.0) | (radius > 2.0)
        leakage.append(float(np.sum(energy_hat[outside]) / np.sum(energy_hat)))
        means.append(float(np.max(np.abs(np.mean(field, axis=(-3, -2, -1)))) / norm(field)))
        component_energy += np.mean(field * field, axis=(-3, -2, -1))
    component_fraction = component_energy / np.sum(component_energy)
    max_lag = min(15, len(coarse) // 3)
    empirical = autocorrelation(coarse, max_lag)
    lag_time = np.arange(max_lag + 1) * forcing.anchor_time_step_size
    theoretical = np.exp(-lag_time / forcing.correlation_time)
    return {
        "maximum_nested_grid_pairing_error": pairing,
        "maximum_divergence_relative": max(divergence),
        "maximum_out_of_band_energy_fraction": max(leakage),
        "maximum_spatial_mean_relative": max(means),
        "component_energy_fraction": component_fraction.tolist(),
        "maximum_component_isotropy_error": float(np.max(np.abs(component_fraction - 1.0 / 3.0))),
        "autocorrelation_lag_time": lag_time.tolist(),
        "autocorrelation_empirical": empirical.tolist(),
        "autocorrelation_theory": theoretical.tolist(),
        "autocorrelation_rmse": float(np.sqrt(np.mean((empirical - theoretical) ** 2))),
    }


def plot_forcing(result: dict[str, object], output: Path) -> None:
    checks = result["forcing_checks"]
    forcing = result["forcing_sample"]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.1), constrained_layout=True)
    axes[0].plot(
        checks["autocorrelation_lag_time"],
        checks["autocorrelation_theory"],
        color=INK,
        linestyle="--",
        linewidth=2.0,
        label=r"OU theory $e^{-\tau/T_f}$",
    )
    axes[0].plot(
        checks["autocorrelation_lag_time"],
        checks["autocorrelation_empirical"],
        color=BLUE,
        marker="o",
        markersize=3.5,
        label="Generated history",
    )
    axes[0].set_title("Temporal autocorrelation")
    axes[0].set_xlabel(r"lag $\tau$")
    axes[0].set_ylabel(r"$R_f(\tau)/R_f(0)$")
    axes[0].legend(frameon=False, fontsize=8)

    positions = np.arange(3)
    axes[1].axhline(1.0 / 3.0, color=INK, linestyle="--", label="Isotropic value")
    axes[1].bar(positions, checks["component_energy_fraction"], color=BLUE, edgecolor="#174864")
    axes[1].set_xticks(positions, (r"$f_x$", r"$f_y$", r"$f_z$"))
    axes[1].set_ylim(0.0, 0.5)
    axes[1].set_title("Long-time component fractions")
    axes[1].set_ylabel("fraction of forcing variance")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].plot(
        forcing["fine_x"], forcing["fine_value"], color=GREY, linewidth=1.5, label="Reference grid"
    )
    axes[2].scatter(
        forcing["coarse_x"],
        forcing["coarse_value"],
        color=BLUE,
        s=28,
        marker="s",
        label="LES grid",
        zorder=3,
    )
    axes[2].set_title("Nested-grid forcing overlay")
    axes[2].set_xlabel(r"$x$")
    axes[2].set_ylabel(r"$f_x(x,0,0)$")
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(color="#d8dde2", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Gate B.1a forcing verification", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_budget(result: dict[str, object], output: Path) -> None:
    runs = result["budget_runs"]
    finest = runs[-1]
    time = [record["time"] for record in finest["records"]]
    time_step_size = np.asarray([run["dt"] for run in runs])
    residual = np.asarray([run["budget_relative_residual"] for run in runs])
    order = result["budget_convergence_order"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    axes[0].plot(
        time, finest["actual_energy_change"], color=BLUE, linewidth=1.8, label=r"$E(t)-E(0)$"
    )
    axes[0].plot(
        time,
        finest["predicted_energy_change"],
        color=INK,
        linestyle="--",
        linewidth=1.8,
        label=r"$\int_0^t(P_f-2\nu Z)\,dt$",
    )
    axes[0].set_title("Energy-balance overlay")
    axes[0].set_xlabel(r"$t$")
    axes[0].set_ylabel("energy change")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].loglog(
        time_step_size,
        residual,
        color=BLUE,
        marker="o",
        linewidth=1.6,
        label=f"Measured ($p={order:.3f}$)",
    )
    reference = residual[0] * (time_step_size / time_step_size[0]) ** 2
    axes[1].loglog(
        time_step_size,
        reference,
        color=INK,
        linestyle="--",
        label=r"reference $\mathcal{O}(\Delta t^2)$",
    )
    axes[1].set_title("Energy-balance convergence")
    axes[1].set_xlabel(r"$\Delta t$")
    axes[1].set_ylabel("relative balance residual")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(color="#d8dde2", linewidth=0.7, which="both")
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Forced-flow theoretical balance", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--les-n", type=int, default=16)
    parser.add_argument("--reference-n", type=int, default=32)
    parser.add_argument("--end-time", type=float, default=0.4)
    parser.add_argument("--anchor-dt", type=float, default=0.02)
    parser.add_argument("--correlation-time", type=float, default=0.2)
    parser.add_argument("--forcing-rms", type=float, default=0.15)
    parser.add_argument("--viscosity", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()

    forcing = ForcingHistory(
        args.les_n,
        args.anchor_time_step_size,
        args.end_time,
        args.correlation_time,
        args.forcing_rms,
        args.seed,
    )
    gaussian_delta = 2.0 * (2.0 * np.pi / args.les_n) / np.sqrt(6.0)
    checks = forcing_checks(forcing, args.reference_n, gaussian_delta)
    initial_velocity = random_isotropic_velocity(args.les_n, args.seed + 1)
    time_step_size_values = [
        args.anchor_time_step_size,
        args.anchor_time_step_size / 2.0,
        args.anchor_time_step_size / 4.0,
    ]
    budget_runs = [
        run_budget_case(
            args.les_n,
            args.viscosity,
            time_step_size,
            args.end_time,
            initial_velocity,
            forcing,
        )
        for time_step_size in time_step_size_values
    ]
    budget_order = float(
        np.polyfit(
            np.log(time_step_size_values),
            np.log([run["budget_relative_residual"] for run in budget_runs]),
            1,
        )[0]
    )
    coarse_sample = forcing.on_grid(args.les_n)[-1]
    fine_raw_sample = forcing.reference_on_grid(args.reference_n, gaussian_delta)[-1]
    fine_sample = SpectralGrid(args.reference_n).gaussian(fine_raw_sample, gaussian_delta)
    x_coarse = 2.0 * np.pi * np.arange(args.les_n) / args.les_n
    x_fine = 2.0 * np.pi * np.arange(args.reference_n) / args.reference_n
    result: dict[str, object] = {
        "gate": "B.1a forcing verification",
        "configuration": {
            "les_n": args.les_n,
            "reference_n": args.reference_n,
            "end_time": args.end_time,
            "anchor_dt": args.anchor_time_step_size,
            "correlation_time": args.correlation_time,
            "forcing_rms": args.forcing_rms,
            "viscosity": args.viscosity,
            "seed": args.seed,
            "forced_band": "1 <= |k| <= 2",
            "temporal_interpolation": "piecewise linear",
            "reference_forcing_relation": "G_delta f_reference = f_LES",
            "gaussian_delta": gaussian_delta,
        },
        "theory": {
            "forcing_autocorrelation": "R_f(tau)/R_f(0) = exp(-tau/T_f)",
            "energy_balance": "dE/dt = P_f - 2 nu Z (no SGS)",
        },
        "forcing_checks": checks,
        "budget_runs": budget_runs,
        "budget_convergence_order": budget_order,
        "forcing_sample": {
            "coarse_x": x_coarse.tolist(),
            "coarse_value": coarse_sample[0, :, 0, 0].tolist(),
            "fine_x": x_fine.tolist(),
            "fine_value": fine_sample[0, :, 0, 0].tolist(),
        },
    }
    result["gate_requirements"] = {
        "maximum_nested_grid_pairing_error": 1.0e-12,
        "maximum_divergence_relative": 1.0e-12,
        "maximum_out_of_band_energy_fraction": 1.0e-12,
        "maximum_spatial_mean_relative": 1.0e-12,
        "maximum_component_isotropy_error": 0.08,
        "autocorrelation_rmse": 0.15,
        "minimum_budget_convergence_order": 1.8,
        "maximum_finest_budget_residual": 1.0e-4,
    }
    result["gate_pass"] = bool(
        checks["maximum_nested_grid_pairing_error"] <= 1.0e-12
        and checks["maximum_divergence_relative"] <= 1.0e-12
        and checks["maximum_out_of_band_energy_fraction"] <= 1.0e-12
        and checks["maximum_spatial_mean_relative"] <= 1.0e-12
        and checks["maximum_component_isotropy_error"] <= 0.08
        and checks["autocorrelation_rmse"] <= 0.15
        and budget_order >= 1.8
        and budget_runs[-1]["budget_relative_residual"] <= 1.0e-4
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    plot_forcing(result, args.figure_dir / "stage_4b1_forcing_verification.png")
    plot_budget(result, args.figure_dir / "stage_4b1_forcing_budget.png")
    if not result["gate_pass"]:
        raise SystemExit("GATE B.1a FAIL")


if __name__ == "__main__":
    main()
