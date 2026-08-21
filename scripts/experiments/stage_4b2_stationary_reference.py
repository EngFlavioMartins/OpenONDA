#!/usr/bin/env python3
r"""Gate B.1c: prepare and qualify a stationary forced-HIT reference.

The external acceleration is the already verified divergence-free, low-mode
Ornstein--Uhlenbeck process.  It is advanced in streaming form so a long
spin-up does not store one three-dimensional force field per time step.

This script qualifies only the reference calculation and the statistical
protocol.  It does not evaluate an SGS model or modify production OpenONDA.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/openonda_stage4b2_matplotlib")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_4b1_forced_hit_pilot import (  # noqa: E402
    integral_scale,
    reynolds_lambda,
)
from stage_4b1_forcing_verification import (  # noqa: E402
    divergence_free_band_noise,
    embed_periodic_field,
    forced_rhs,
    random_isotropic_velocity,
)
from stage_4b_spectral_pilot import (  # noqa: E402
    VorticitySolver,
    diagnostics,
    energy_spectrum,
)

INK = "#20252a"
BLUE = "#286f9b"
GOLD = "#d9973b"
GREY = "#8a99a8"
GRID = "#d8dde2"


class StreamingOUForcing:
    """Reproducible OU acceleration on the modes common to reference and LES."""

    def __init__(
        self,
        les_n: int,
        time_step_size: float,
        correlation_time: float,
        target_rms: float,
        seed: int,
    ) -> None:
        self.les_n = les_n
        self.time_step_size = time_step_size
        self.target_rms = target_rms
        self.rng = np.random.default_rng(seed)
        self.rho = float(np.exp(-time_step_size / correlation_time))
        self.grid = VorticitySolver(les_n, 0.0).grid
        self.field = divergence_free_band_noise(self.grid, self.rng, target_rms)

    def advance(self) -> np.ndarray:
        innovation = divergence_free_band_noise(self.grid, self.rng, self.target_rms)
        self.field = self.rho * self.field + np.sqrt(1.0 - self.rho**2) * innovation
        return self.field

    def reference_field(self, reference_n: int, gaussian_delta: float) -> np.ndarray:
        return embed_periodic_field(self.field, reference_n, gaussian_delta)


def component_anisotropy(velocity: np.ndarray) -> float:
    variances = component_variances(velocity)
    target = float(np.mean(variances))
    return float(np.max(np.abs(variances / target - 1.0)))


def component_variances(velocity: np.ndarray) -> np.ndarray:
    return np.mean(velocity * velocity, axis=(1, 2, 3))


def curl_hat(solver: VorticitySolver, vector: np.ndarray) -> np.ndarray:
    hat = solver.grid.fft(vector)
    return (
        1j
        * np.asarray(
            (
                solver.grid.ky * hat[2] - solver.grid.kz * hat[1],
                solver.grid.kz * hat[0] - solver.grid.kx * hat[2],
                solver.grid.kx * hat[1] - solver.grid.ky * hat[0],
            )
        )
        * solver.mask
    )


def rotational_reference_rhs(
    solver: VorticitySolver,
    vorticity: np.ndarray,
    acceleration_curl_hat: np.ndarray,
) -> np.ndarray:
    """Dealiased curl(u cross omega) form of the unmodeled reference RHS."""
    vorticity_hat = solver.grid.fft(vorticity) * solver.mask
    wave_cross_vorticity = np.asarray(
        (
            solver.grid.ky * vorticity_hat[2] - solver.grid.kz * vorticity_hat[1],
            solver.grid.kz * vorticity_hat[0] - solver.grid.kx * vorticity_hat[2],
            solver.grid.kx * vorticity_hat[1] - solver.grid.ky * vorticity_hat[0],
        )
    )
    velocity_hat = np.zeros_like(wave_cross_vorticity)
    velocity_hat[:, solver.nonzero] = (
        1j * wave_cross_vorticity[:, solver.nonzero] / solver.grid.k2[solver.nonzero]
    )
    velocity = solver.grid.ifft(velocity_hat)
    velocity_cross_vorticity = np.cross(velocity, vorticity, axisa=0, axisb=0, axisc=0)
    nonlinear_curl_hat = curl_hat(solver, velocity_cross_vorticity)
    rhs_hat = (
        nonlinear_curl_hat
        - solver.viscosity * solver.grid.k2 * vorticity_hat
        + acceleration_curl_hat
    ) * solver.mask
    return solver.grid.ifft(rhs_hat)


def rotational_reference_step(
    solver: VorticitySolver,
    vorticity: np.ndarray,
    time_step_size: float,
    acceleration_start_curl_hat: np.ndarray,
    acceleration_end_curl_hat: np.ndarray,
) -> np.ndarray:
    first = rotational_reference_rhs(solver, vorticity, acceleration_start_curl_hat)
    predictor = solver.grid.ifft(solver.grid.fft(vorticity + time_step_size * first) * solver.mask)
    second = rotational_reference_rhs(solver, predictor, acceleration_end_curl_hat)
    return solver.grid.ifft(
        solver.grid.fft(vorticity + 0.5 * time_step_size * (first + second)) * solver.mask
    )


def verify_rotational_rhs(
    solver: VorticitySolver,
    vorticity: np.ndarray,
    acceleration: np.ndarray,
    gaussian_delta: float,
) -> float:
    baseline = forced_rhs(solver, vorticity, gaussian_delta, acceleration, "no_sgs")
    rotational = rotational_reference_rhs(solver, vorticity, curl_hat(solver, acceleration))
    return float(
        np.linalg.norm(rotational - baseline) / max(np.linalg.norm(baseline), np.finfo(float).tiny)
    )


def record_state(
    solver: VorticitySolver,
    vorticity: np.ndarray,
    acceleration: np.ndarray,
    gaussian_delta: float,
    time: float,
) -> dict[str, float]:
    record = diagnostics(solver, vorticity, "no_sgs", gaussian_delta)
    velocity = solver.velocity(vorticity)
    spectrum = energy_spectrum(solver, vorticity)
    energy = float(record["energy"])
    enstrophy = float(record["enstrophy"])
    dissipation = 2.0 * solver.viscosity * enstrophy
    scale = integral_scale(spectrum, energy)
    component_rms = np.sqrt(2.0 * energy / 3.0)
    variances = component_variances(velocity)
    turnover = scale / max(component_rms, np.finfo(float).tiny)
    eta = (solver.viscosity**3 / max(dissipation, np.finfo(float).tiny)) ** 0.25
    conservative_kmax = solver.grid.n // 3 - 1
    record.update(
        {
            "time": time,
            "forcing_power": float(np.mean(np.sum(velocity * acceleration, axis=0))),
            "dissipation": dissipation,
            "integral_scale": scale,
            "turnover_time": turnover,
            "reynolds_lambda": reynolds_lambda(energy, enstrophy, solver.viscosity),
            "kmax_eta": conservative_kmax * eta,
            "component_anisotropy": component_anisotropy(velocity),
            "component_variance_x": float(variances[0]),
            "component_variance_y": float(variances[1]),
            "component_variance_z": float(variances[2]),
        }
    )
    return record


def add_turnover_coordinate(records: list[dict[str, float]]) -> None:
    records[0]["turnovers"] = 0.0
    for previous, current in zip(records[:-1], records[1:], strict=True):
        time_step_size = current["time"] - previous["time"]
        inverse_turnover = 0.5 * (1.0 / previous["turnover_time"] + 1.0 / current["turnover_time"])
        current["turnovers"] = previous["turnovers"] + time_step_size * inverse_turnover


def relative_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.ptp(x) <= np.finfo(float).eps:
        return float("inf")
    return float(abs(np.polyfit(x, y, 1)[0]) / max(abs(np.mean(y)), 1.0e-15))


def half_window_change(values: np.ndarray) -> float:
    midpoint = len(values) // 2
    if midpoint < 2:
        return float("inf")
    first = float(np.mean(values[:midpoint]))
    second = float(np.mean(values[midpoint:]))
    return abs(second - first) / max(abs(0.5 * (first + second)), 1.0e-15)


def integral_correlation_time(x: np.ndarray, values: np.ndarray) -> float:
    centered = values - np.mean(values)
    variance = float(np.mean(centered * centered))
    numerical_floor = 100.0 * np.finfo(float).eps * float(np.mean(values * values))
    if variance <= max(numerical_floor, np.finfo(float).tiny):
        return 0.0
    correlation = np.correlate(centered, centered, mode="full")[len(centered) - 1 :]
    correlation /= np.arange(len(centered), 0, -1) * variance
    nonpositive = np.flatnonzero(correlation[1:] <= 0.0)
    stop = int(nonpositive[0] + 1) if len(nonpositive) else len(correlation)
    spacing = float(np.mean(np.diff(x)))
    return float(spacing * (0.5 + np.sum(correlation[1:stop])))


def block_mean_test(x: np.ndarray, values: np.ndarray) -> dict[str, float]:
    midpoint = len(values) // 2
    if midpoint < 3 or len(values) - midpoint < 3:
        return {"relative_change": float("inf"), "z_score": float("inf")}
    blocks = ((x[:midpoint], values[:midpoint]), (x[midpoint:], values[midpoint:]))
    means = [float(np.mean(block_values)) for _, block_values in blocks]
    standard_errors = []
    for block_x, block_values in blocks:
        duration = max(float(np.ptp(block_x)), np.finfo(float).tiny)
        integral_time = integral_correlation_time(block_x, block_values)
        standard_errors.append(
            np.sqrt(max(2.0 * np.var(block_values) * integral_time / duration, 0.0))
        )
    difference = abs(means[1] - means[0])
    difference_error = float(np.hypot(*standard_errors))
    if difference_error <= np.finfo(float).tiny:
        z_score = 0.0 if difference <= np.finfo(float).tiny else float("inf")
    else:
        z_score = difference / difference_error
    return {
        "relative_change": difference / max(abs(0.5 * (means[0] + means[1])), 1.0e-15),
        "z_score": float(z_score),
    }


def assess_stationarity(
    records: list[dict[str, float]],
    window_turnovers: float = 10.0,
) -> dict[str, object]:
    add_turnover_coordinate(records)
    total_turnovers = float(records[-1]["turnovers"])
    start = total_turnovers - window_turnovers
    window = [record for record in records if record["turnovers"] >= start]
    coordinate = np.asarray([record["turnovers"] for record in window])
    energy = np.asarray([record["energy"] for record in window])
    dissipation = np.asarray([record["dissipation"] for record in window])
    forcing = np.asarray([record["forcing_power"] for record in window])
    high_k = np.asarray([record["high_k_energy_fraction"] for record in window])
    kmax_eta = np.asarray([record["kmax_eta"] for record in window])
    mean_variances = np.asarray(
        [
            np.mean([record[f"component_variance_{axis}"] for record in window])
            for axis in ("x", "y", "z")
        ]
    )
    averaged_anisotropy = float(np.max(np.abs(mean_variances / np.mean(mean_variances) - 1.0)))
    energy_blocks = block_mean_test(coordinate, energy)
    dissipation_blocks = block_mean_test(coordinate, dissipation)
    energy_integral_time = integral_correlation_time(coordinate, energy)
    dissipation_integral_time = integral_correlation_time(coordinate, dissipation)
    longest_integral_time = max(energy_integral_time, dissipation_integral_time)
    correlation_times_covered = (
        float("inf")
        if longest_integral_time == 0.0
        else float(np.ptp(coordinate) / longest_integral_time)
    )
    thresholds = {
        "minimum_development_turnovers": 5.0,
        "minimum_total_turnovers": 5.0 + window_turnovers,
        "minimum_correlation_times_covered": 5.0,
        "energy_slope_per_turnover": 0.02,
        "dissipation_slope_per_turnover": 0.05,
        "maximum_block_relative_change": 0.20,
        "maximum_block_z_score": 1.96,
        "mean_power_imbalance": 0.10,
        "maximum_high_k_energy_fraction": 0.01,
        "minimum_kmax_eta": 1.0,
        "time_averaged_component_anisotropy": 0.10,
    }
    values = {
        "total_turnovers": total_turnovers,
        "window_turnovers": float(coordinate[-1] - coordinate[0]),
        "energy_slope_per_turnover": relative_slope(coordinate, energy),
        "dissipation_slope_per_turnover": relative_slope(coordinate, dissipation),
        "energy_half_window_change": energy_blocks["relative_change"],
        "dissipation_half_window_change": dissipation_blocks["relative_change"],
        "energy_block_z_score": energy_blocks["z_score"],
        "dissipation_block_z_score": dissipation_blocks["z_score"],
        "energy_integral_correlation_time": energy_integral_time,
        "dissipation_integral_correlation_time": dissipation_integral_time,
        "correlation_times_covered": correlation_times_covered,
        "mean_power_imbalance": float(
            abs(np.mean(forcing) - np.mean(dissipation)) / max(abs(np.mean(dissipation)), 1.0e-15)
        ),
        "maximum_high_k_energy_fraction": float(np.max(high_k)),
        "minimum_kmax_eta": float(np.min(kmax_eta)),
        "time_averaged_component_anisotropy": averaged_anisotropy,
        "mean_reynolds_lambda": float(np.mean([record["reynolds_lambda"] for record in window])),
        "window_start_time": float(window[0]["time"]),
        "window_end_time": float(window[-1]["time"]),
    }
    checks = {
        "enough_spinup_and_window": total_turnovers >= thresholds["minimum_total_turnovers"],
        "enough_effective_samples": correlation_times_covered
        >= thresholds["minimum_correlation_times_covered"],
        "energy_drift": values["energy_slope_per_turnover"]
        <= thresholds["energy_slope_per_turnover"],
        "dissipation_drift": values["dissipation_slope_per_turnover"]
        <= thresholds["dissipation_slope_per_turnover"],
        "energy_block_agreement": values["energy_half_window_change"]
        <= thresholds["maximum_block_relative_change"]
        and values["energy_block_z_score"] <= thresholds["maximum_block_z_score"],
        "dissipation_block_agreement": values["dissipation_half_window_change"]
        <= thresholds["maximum_block_relative_change"]
        and values["dissipation_block_z_score"] <= thresholds["maximum_block_z_score"],
        "stationary_energy_balance": values["mean_power_imbalance"]
        <= thresholds["mean_power_imbalance"],
        "spectral_resolution": values["maximum_high_k_energy_fraction"]
        <= thresholds["maximum_high_k_energy_fraction"],
        "kolmogorov_resolution": values["minimum_kmax_eta"] >= thresholds["minimum_kmax_eta"],
        "isotropy": values["time_averaged_component_anisotropy"]
        <= thresholds["time_averaged_component_anisotropy"],
    }
    return {
        "pass": bool(all(checks.values())),
        "thresholds": thresholds,
        "values": values,
        "checks": checks,
    }


def verify_assessment_logic() -> dict[str, object]:
    coordinate = np.linspace(0.0, 20.0, 201)

    def synthetic(energy_drift: float, imbalance: float) -> list[dict[str, float]]:
        records = []
        for value in coordinate:
            energy = 1.0 + energy_drift * value
            dissipation = 0.1
            records.append(
                {
                    "time": value,
                    "turnover_time": 1.0,
                    "energy": energy,
                    "dissipation": dissipation,
                    "forcing_power": dissipation * (1.0 + imbalance),
                    "high_k_energy_fraction": 0.001,
                    "kmax_eta": 1.5,
                    "component_anisotropy": 0.01,
                    "component_variance_x": 1.0,
                    "component_variance_y": 1.0,
                    "component_variance_z": 1.0,
                    "reynolds_lambda": 40.0,
                }
            )
        return records

    steady = assess_stationarity(synthetic(0.0, 0.0), 10.0)
    drifting = assess_stationarity(synthetic(0.05, 0.0), 10.0)
    unbalanced = assess_stationarity(synthetic(0.0, 0.2), 10.0)
    passed = bool(steady["pass"] and not drifting["pass"] and not unbalanced["pass"])
    return {
        "pass": passed,
        "steady_series_passes": steady["pass"],
        "five_percent_per_turnover_drift_fails": not drifting["pass"],
        "twenty_percent_power_imbalance_fails": not unbalanced["pass"],
    }


def run(args: argparse.Namespace) -> tuple[dict[str, object], np.ndarray]:
    if args.reference_n != 2 * args.les_n:
        raise ValueError("reference_n must equal 2 * les_n")
    solver = VorticitySolver(args.reference_n, args.viscosity)
    gaussian_delta = 2.0 * (2.0 * np.pi / args.les_n) / np.sqrt(6.0)
    forcing = StreamingOUForcing(
        args.les_n,
        args.time_step_size,
        args.correlation_time,
        args.forcing_rms,
        args.seed,
    )
    velocity = random_isotropic_velocity(args.reference_n, args.seed + 1, args.initial_rms)
    vorticity = solver.project(solver.grid.curl(velocity))
    steps = int(round(args.end_time / args.time_step_size))
    save_every = max(1, int(round(args.save_interval / args.time_step_size)))
    reference_force = forcing.reference_field(args.reference_n, gaussian_delta)
    rhs_difference = verify_rotational_rhs(solver, vorticity, reference_force, gaussian_delta)
    if rhs_difference > 1.0e-12:
        raise RuntimeError(f"rotational reference RHS mismatch: {rhs_difference:.3e}")
    reference_force_curl_hat = curl_hat(solver, reference_force)
    records: list[dict[str, float]] = []
    for step in range(steps + 1):
        time = step * args.time_step_size
        if step % max(1, steps // 10) == 0:
            print(f"stationary-reference progress: {100.0 * step / steps:5.1f}%", flush=True)
        if step % save_every == 0 or step == steps:
            records.append(record_state(solver, vorticity, reference_force, gaussian_delta, time))
        if step == steps:
            break
        next_les_force = forcing.advance()
        next_reference_force = embed_periodic_field(
            next_les_force, args.reference_n, gaussian_delta
        )
        next_reference_force_curl_hat = curl_hat(solver, next_reference_force)
        vorticity = rotational_reference_step(
            solver,
            vorticity,
            args.time_step_size,
            reference_force_curl_hat,
            next_reference_force_curl_hat,
        )
        reference_force = next_reference_force
        reference_force_curl_hat = next_reference_force_curl_hat
        if not np.all(np.isfinite(vorticity)):
            raise FloatingPointError(f"non-finite reference state at step {step + 1}")

    assessment = assess_stationarity(records, args.window_turnovers)
    spectrum = energy_spectrum(solver, vorticity)
    return (
        {
            "gate": "B.1c stationary forced-HIT reference qualification",
            "status": "PASS" if assessment["pass"] else "FAIL",
            "configuration": {
                "reference_n": args.reference_n,
                "les_n": args.les_n,
                "viscosity": args.viscosity,
                "dt": args.time_step_size,
                "end_time": args.end_time,
                "save_interval": args.save_interval,
                "forcing_rms": args.forcing_rms,
                "forcing_correlation_time": args.correlation_time,
                "forcing_seed": args.seed,
                "initial_rms": args.initial_rms,
                "forcing_relation": "G_delta f_reference = f_LES",
                "paper_filter_width_over_h": 2.0,
                "gaussian_delta": gaussian_delta,
            },
            "theory": {
                "energy_balance": "dE/dt = P_f - epsilon",
                "stationary_limit": "<P_f> = <epsilon>",
                "kolmogorov_scale": "eta = (nu^3/epsilon)^(1/4)",
                "literature_spinup": "approximately 3-5 large-eddy turnover times",
            },
            "assessment_verification": verify_assessment_logic(),
            "rotational_rhs_relative_difference": rhs_difference,
            "assessment": assessment,
            "records": records,
            "final_energy_spectrum": spectrum.tolist(),
        },
        vorticity,
    )


def plot_stationarity(result: dict[str, object], output: Path) -> None:
    records = result["records"]
    assessment = result["assessment"]
    values = assessment["values"]
    x = np.asarray([record["turnovers"] for record in records])
    energy = np.asarray([record["energy"] for record in records])
    power = np.asarray([record["forcing_power"] for record in records])
    dissipation = np.asarray([record["dissipation"] for record in records])
    window = np.asarray([record["time"] >= values["window_start_time"] for record in records])
    mean_energy = float(np.mean(energy[window]))
    mean_dissipation = float(np.mean(dissipation[window]))

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.5), constrained_layout=True)
    axes[0, 0].plot(x, energy / mean_energy, color=BLUE, linewidth=1.7)
    axes[0, 0].axhline(1.0, color=INK, linestyle="--", label="stationary mean")
    axes[0, 0].set_title("Kinetic energy")
    axes[0, 0].set_ylabel(r"$E/\langle E\rangle_{window}$")

    axes[0, 1].plot(
        x,
        power / mean_dissipation,
        color=GOLD,
        linewidth=1.3,
        label=r"$P_f/\langle\varepsilon\rangle$",
    )
    axes[0, 1].plot(
        x,
        dissipation / mean_dissipation,
        color=BLUE,
        linewidth=1.7,
        label=r"$\varepsilon/\langle\varepsilon\rangle$",
    )
    axes[0, 1].axhline(1.0, color=INK, linestyle="--", label="stationary balance")
    axes[0, 1].set_title("Power input and dissipation")
    axes[0, 1].legend(frameon=False, fontsize=8)

    high_k = [record["high_k_energy_fraction"] for record in records]
    axes[1, 0].plot(x, high_k, color=BLUE, linewidth=1.7)
    axes[1, 0].axhline(0.01, color=GOLD, linestyle="--", label="1% spectral-tail gate")
    axes[1, 0].set_title("Fine-grid spectral tail")
    axes[1, 0].set_ylabel("high-wavenumber energy fraction")
    axes[1, 0].legend(frameon=False, fontsize=8)

    axes[1, 1].plot(
        x,
        [record["kmax_eta"] for record in records],
        color=BLUE,
        linewidth=1.7,
        label=r"$k_{max}\eta$",
    )
    window_start = int(np.flatnonzero(window)[0])
    running_components = {}
    for axis_name in ("x", "y", "z"):
        raw = np.asarray([record[f"component_variance_{axis_name}"] for record in records])
        running = np.full_like(raw, np.nan)
        running[window_start:] = np.cumsum(raw[window_start:]) / np.arange(
            1, len(raw) - window_start + 1
        )
        running_components[axis_name] = running
    running_target = sum(running_components.values()) / 3.0
    for axis_name, style in zip(("x", "y", "z"), ("-", "--", ":"), strict=True):
        ratios = running_components[axis_name] / running_target
        axes[1, 1].plot(
            x,
            ratios,
            color=GREY,
            linestyle=style,
            linewidth=1.0,
            label=rf"running $\langle u_{axis_name}^2\rangle/(2E/3)$",
        )
    axes[1, 1].axhline(1.0, color=INK, linestyle="--", label="theoretical target")
    axes[1, 1].axhspan(0.9, 1.1, color=GOLD, alpha=0.08, label="10% isotropy band")
    axes[1, 1].set_title("Resolution and time-averaged isotropy")
    axes[1, 1].legend(frameon=False, fontsize=8)

    for axis in axes.flat:
        axis.axvspan(x[window][0], x[-1], color=BLUE, alpha=0.06, label="assessment window")
        axis.set_xlabel("elapsed large-eddy turnover times")
        axis.grid(color=GRID, linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Stationary-reference qualification: {result['status']}", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_spectrum(result: dict[str, object], output: Path) -> None:
    spectrum = np.asarray(result["final_energy_spectrum"])
    wave = np.arange(len(spectrum), dtype=float)
    positive = (wave > 0.0) & (spectrum > 0.0)
    fig, axis = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    axis.loglog(
        wave[positive],
        spectrum[positive],
        color=BLUE,
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="reference",
    )
    available = wave[positive]
    anchor_k = available[min(3, len(available) - 1)]
    anchor_e = spectrum[int(anchor_k)]
    guide_k = np.asarray([anchor_k, min(float(available[-1]), 4.0 * anchor_k)])
    guide_e = anchor_e * (guide_k / anchor_k) ** (-5.0 / 3.0)
    axis.loglog(
        guide_k, guide_e, color=INK, linestyle="--", linewidth=1.4, label=r"$k^{-5/3}$ slope guide"
    )
    cutoff = result["configuration"]["reference_n"] // 3
    axis.axvspan(0.7 * cutoff, cutoff, color=GOLD, alpha=0.15, label="resolution-check band")
    axis.set_xlabel(r"wavenumber shell $k$")
    axis.set_ylabel(r"$E(k)$")
    axis.set_title("Final reference energy spectrum")
    axis.grid(color=GRID, linewidth=0.7, which="both")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-n", type=int, default=64)
    parser.add_argument("--les-n", type=int, default=32)
    parser.add_argument("--viscosity", type=float, default=0.01)
    parser.add_argument("--time-step-size", dest="time_step_size", type=float, default=0.02)
    parser.add_argument("--end-time", type=float, default=60.0)
    parser.add_argument("--save-interval", type=float, default=0.1)
    parser.add_argument("--forcing-rms", type=float, default=0.5)
    parser.add_argument("--correlation-time", type=float, default=0.2)
    parser.add_argument("--initial-rms", type=float, default=1.0)
    parser.add_argument("--window-turnovers", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-output", type=Path)
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args()
    result, final_vorticity = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.state_output is not None:
        args.state_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.state_output, vorticity=final_vorticity)
    plot_stationarity(result, args.figure_dir / "stage_4b2_stationary_reference.png")
    plot_spectrum(result, args.figure_dir / "stage_4b2_stationary_spectrum.png")
    if not result["assessment_verification"]["pass"]:
        raise SystemExit("STATIONARITY ASSESSMENT VERIFICATION FAIL")
    if not result["assessment"]["pass"]:
        raise SystemExit("STATIONARY REFERENCE QUALIFICATION FAIL")


if __name__ == "__main__":
    main()
