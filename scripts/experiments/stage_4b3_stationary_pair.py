#!/usr/bin/env python3
r"""Gate B.1d: checkpointed stationary reference/LES paired screen.

This research-only driver compares the frozen SGS candidates with the same
filtered external force and a resolved reference.  It archives restartable raw
states and verifies restart reproducibility before a long run is permitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/openonda_stage4b3_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/openonda_stage4b3_cache")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_4a_formulation import model_torques  # noqa: E402
from stage_4b1_forced_hit_pilot import (  # noqa: E402
    add_model_diagnostics,
    add_reference_diagnostics,
    budget_summary,
)
from stage_4b1_forcing_verification import (  # noqa: E402
    forced_rhs,
    random_isotropic_velocity,
)
from stage_4b2_stationary_reference import (  # noqa: E402
    StreamingOUForcing,
    assess_stationarity,
    curl_hat,
    record_state,
    rotational_reference_rhs,
    rotational_reference_step,
)
from stage_4b_spectral_pilot import (  # noqa: E402
    COLORS,
    LABELS,
    MODELS,
    VorticitySolver,
    coarse_reference,
    energy_spectrum,
)

DISPLAY_LABELS = {**LABELS, "filtered_dns": "Filtered reference"}
INK = "#20252a"
BLUE = "#286f9b"
GOLD = "#d9973b"
GREY = "#8a99a8"
GRID = "#d8dde2"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "reference_n": args.reference_n,
        "les_n": args.les_n,
        "viscosity": args.viscosity,
        "dt": args.dt,
        "end_time": args.end_time,
        "save_interval": args.save_interval,
        "checkpoint_interval": args.checkpoint_interval,
        "forcing_rms": args.forcing_rms,
        "forcing_correlation_time": args.correlation_time,
        "forcing_seed": args.seed,
        "initial_rms": args.initial_rms,
        "forcing_relation": "G_delta f_reference = f_LES",
        "paper_filter_width_over_h": 2.0,
        "models": list(MODELS),
    }


def configuration_fingerprint(config: dict[str, Any]) -> str:
    restart_config = dict(config)
    restart_config.pop("end_time")
    restart_config.pop("checkpoint_interval")
    return hashlib.sha256(canonical_json(restart_config).encode()).hexdigest()


def checkpoint_paths(directory: Path, step: int) -> tuple[Path, Path, Path]:
    stem = directory / f"checkpoint_{step:07d}"
    return stem.with_suffix(".npz"), stem.with_suffix(".json"), stem.with_suffix(".sha256")


def write_checkpoint(
    directory: Path,
    step: int,
    time: float,
    config: dict[str, Any],
    reference_vorticity: np.ndarray,
    states: dict[str, np.ndarray],
    forcing: StreamingOUForcing,
    histories: dict[str, list[dict[str, Any]]],
    fine_reference_history: list[dict[str, Any]],
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    npz_path, json_path, checksum_path = checkpoint_paths(directory, step)
    npz_tmp = npz_path.with_suffix(".npz.tmp")
    json_tmp = json_path.with_suffix(".json.tmp")
    with npz_tmp.open("wb") as stream:
        np.savez_compressed(
            stream,
            reference_vorticity=reference_vorticity,
            forcing_field=forcing.field,
            **{f"state_{model}": state for model, state in states.items()},
        )
    metadata = {
        "schema": "openonda-vpm-les-checkpoint-v1",
        "step": step,
        "time": time,
        "configuration": config,
        "configuration_fingerprint": configuration_fingerprint(config),
        "rng_state": forcing.rng.bit_generator.state,
        "histories": histories,
        "fine_reference_history": fine_reference_history,
    }
    json_tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    npz_tmp.replace(npz_path)
    json_tmp.replace(json_path)
    checksums = {
        npz_path.name: sha256(npz_path),
        json_path.name: sha256(json_path),
    }
    checksum_path.write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n")
    return npz_path


def verify_checkpoint_files(npz_path: Path) -> tuple[Path, Path]:
    json_path = npz_path.with_suffix(".json")
    checksum_path = npz_path.with_suffix(".sha256")
    checksums = json.loads(checksum_path.read_text())
    for path in (npz_path, json_path):
        actual = sha256(path)
        expected = checksums[path.name]
        if actual != expected:
            raise ValueError(f"checkpoint checksum mismatch for {path.name}")
    return json_path, checksum_path


def load_checkpoint(
    npz_path: Path,
    args: argparse.Namespace,
) -> tuple[
    int,
    float,
    np.ndarray,
    dict[str, np.ndarray],
    StreamingOUForcing,
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
]:
    json_path, _ = verify_checkpoint_files(npz_path)
    metadata = json.loads(json_path.read_text())
    config = configuration(args)
    if metadata["configuration_fingerprint"] != configuration_fingerprint(config):
        raise ValueError("checkpoint configuration fingerprint mismatch")
    arrays = np.load(npz_path)
    forcing = StreamingOUForcing(
        args.les_n,
        args.dt,
        args.correlation_time,
        args.forcing_rms,
        args.seed,
    )
    forcing.field = arrays["forcing_field"].copy()
    forcing.rng.bit_generator.state = metadata["rng_state"]
    states = {model: arrays[f"state_{model}"].copy() for model in MODELS}
    return (
        int(metadata["step"]),
        float(metadata["time"]),
        arrays["reference_vorticity"].copy(),
        states,
        forcing,
        metadata["histories"],
        metadata["fine_reference_history"],
    )


def optimized_model_rhs(
    solver: VorticitySolver,
    vorticity: np.ndarray,
    model: str,
    gaussian_delta: float,
    acceleration_curl_hat: np.ndarray,
) -> np.ndarray:
    base = rotational_reference_rhs(solver, vorticity, acceleration_curl_hat)
    if model == "no_sgs":
        return base
    velocity = solver.velocity(vorticity)
    torque = model_torques(solver.grid, velocity, gaussian_delta)[0][model]
    return solver.grid.ifft(solver.grid.fft(base + torque) * solver.mask)


def optimized_model_step(
    solver: VorticitySolver,
    vorticity: np.ndarray,
    model: str,
    gaussian_delta: float,
    dt: float,
    acceleration_start_curl_hat: np.ndarray,
    acceleration_end_curl_hat: np.ndarray,
) -> np.ndarray:
    first = optimized_model_rhs(
        solver,
        vorticity,
        model,
        gaussian_delta,
        acceleration_start_curl_hat,
    )
    predictor = solver.grid.ifft(solver.grid.fft(vorticity + dt * first) * solver.mask)
    second = optimized_model_rhs(
        solver,
        predictor,
        model,
        gaussian_delta,
        acceleration_end_curl_hat,
    )
    return solver.grid.ifft(solver.grid.fft(vorticity + 0.5 * dt * (first + second)) * solver.mask)


def verify_optimized_rhs(
    solver: VorticitySolver,
    vorticity: np.ndarray,
    acceleration: np.ndarray,
    gaussian_delta: float,
) -> dict[str, float]:
    acceleration_curl_hat = curl_hat(solver, acceleration)
    differences = {}
    for model in MODELS:
        baseline = forced_rhs(solver, vorticity, gaussian_delta, acceleration, model)
        optimized = optimized_model_rhs(
            solver,
            vorticity,
            model,
            gaussian_delta,
            acceleration_curl_hat,
        )
        differences[model] = float(
            np.linalg.norm(optimized - baseline)
            / max(np.linalg.norm(baseline), np.finfo(float).tiny)
        )
    return differences


def initialize(
    args: argparse.Namespace,
) -> tuple[
    VorticitySolver,
    VorticitySolver,
    float,
    int,
    np.ndarray,
    dict[str, np.ndarray],
    StreamingOUForcing,
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
]:
    reference_solver = VorticitySolver(args.reference_n, args.viscosity)
    les_solver = VorticitySolver(args.les_n, args.viscosity)
    gaussian_delta = 2.0 * (2.0 * np.pi / args.les_n) / np.sqrt(6.0)
    if args.restart is not None:
        (
            step,
            _,
            reference_vorticity,
            states,
            forcing,
            histories,
            fine_reference_history,
        ) = load_checkpoint(args.restart, args)
    else:
        step = 0
        forcing = StreamingOUForcing(
            args.les_n,
            args.dt,
            args.correlation_time,
            args.forcing_rms,
            args.seed,
        )
        reference_velocity = random_isotropic_velocity(
            args.reference_n, args.seed + 1, args.initial_rms
        )
        reference_vorticity = reference_solver.project(
            reference_solver.grid.curl(reference_velocity)
        )
        initial_filtered = coarse_reference(
            reference_solver,
            reference_vorticity,
            args.les_n,
            gaussian_delta,
        )
        states = {model: initial_filtered.copy() for model in MODELS}
        histories = {model: [] for model in ("filtered_dns", *MODELS)}
        fine_reference_history = []
    return (
        reference_solver,
        les_solver,
        gaussian_delta,
        step,
        reference_vorticity,
        states,
        forcing,
        histories,
        fine_reference_history,
    )


def append_diagnostics(
    reference_solver: VorticitySolver,
    les_solver: VorticitySolver,
    reference_vorticity: np.ndarray,
    states: dict[str, np.ndarray],
    reference_acceleration: np.ndarray,
    les_acceleration: np.ndarray,
    gaussian_delta: float,
    time: float,
    histories: dict[str, list[dict[str, Any]]],
    fine_reference_history: list[dict[str, Any]],
) -> None:
    fine_record = record_state(
        reference_solver,
        reference_vorticity,
        reference_acceleration,
        gaussian_delta,
        time,
    )
    fine_record["energy_spectrum"] = energy_spectrum(reference_solver, reference_vorticity).tolist()
    fine_reference_history.append(fine_record)
    filtered = coarse_reference(
        reference_solver,
        reference_vorticity,
        les_solver.grid.n,
        gaussian_delta,
    )
    filtered_spectrum = energy_spectrum(les_solver, filtered)
    filtered_record = add_reference_diagnostics(
        les_solver,
        filtered,
        les_acceleration,
        gaussian_delta,
        time,
    )
    filtered_record["energy_spectrum"] = filtered_spectrum.tolist()
    histories["filtered_dns"].append(filtered_record)
    for model, state in states.items():
        record = add_model_diagnostics(
            les_solver,
            state,
            filtered,
            filtered_spectrum,
            les_acceleration,
            gaussian_delta,
            model,
            time,
        )
        record["energy_spectrum"] = energy_spectrum(les_solver, state).tolist()
        histories[model].append(record)


def advance_one_step(
    args: argparse.Namespace,
    reference_solver: VorticitySolver,
    les_solver: VorticitySolver,
    gaussian_delta: float,
    reference_vorticity: np.ndarray,
    states: dict[str, np.ndarray],
    forcing: StreamingOUForcing,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    reference_start = forcing.reference_field(args.reference_n, gaussian_delta)
    les_start = forcing.field
    forcing.advance()
    reference_end = forcing.reference_field(args.reference_n, gaussian_delta)
    les_end = forcing.field
    reference_vorticity = rotational_reference_step(
        reference_solver,
        reference_vorticity,
        args.dt,
        curl_hat(reference_solver, reference_start),
        curl_hat(reference_solver, reference_end),
    )
    les_start_curl_hat = curl_hat(les_solver, les_start)
    les_end_curl_hat = curl_hat(les_solver, les_end)
    states = {
        model: optimized_model_step(
            les_solver,
            state,
            model,
            gaussian_delta,
            args.dt,
            les_start_curl_hat,
            les_end_curl_hat,
        )
        for model, state in states.items()
    }
    return reference_vorticity, states


def restart_verification(args: argparse.Namespace, directory: Path) -> dict[str, Any]:
    verify_args = argparse.Namespace(**vars(args))
    verify_args.reference_n = 16
    verify_args.les_n = 8
    verify_args.end_time = 0.2
    verify_args.restart = None
    (
        reference_solver,
        les_solver,
        gaussian_delta,
        step,
        reference_vorticity,
        states,
        forcing,
        histories,
        fine_history,
    ) = initialize(verify_args)
    rhs_differences = verify_optimized_rhs(
        les_solver,
        states["structural"],
        forcing.field,
        gaussian_delta,
    )
    for _ in range(2):
        reference_vorticity, states = advance_one_step(
            verify_args,
            reference_solver,
            les_solver,
            gaussian_delta,
            reference_vorticity,
            states,
            forcing,
        )
        step += 1
    checkpoint = write_checkpoint(
        directory,
        step,
        step * verify_args.dt,
        configuration(verify_args),
        reference_vorticity,
        states,
        forcing,
        histories,
        fine_history,
    )
    verify_args.restart = checkpoint
    (
        _,
        _,
        _,
        loaded_step,
        loaded_reference,
        loaded_states,
        loaded_forcing,
        _,
        _,
    ) = initialize(verify_args)
    load_differences = {
        "reference": float(np.max(np.abs(loaded_reference - reference_vorticity))),
        "forcing": float(np.max(np.abs(loaded_forcing.field - forcing.field))),
        **{model: float(np.max(np.abs(loaded_states[model] - states[model]))) for model in MODELS},
    }
    for _ in range(2):
        reference_vorticity, states = advance_one_step(
            verify_args,
            reference_solver,
            les_solver,
            gaussian_delta,
            reference_vorticity,
            states,
            forcing,
        )
        loaded_reference, loaded_states = advance_one_step(
            verify_args,
            reference_solver,
            les_solver,
            gaussian_delta,
            loaded_reference,
            loaded_states,
            loaded_forcing,
        )
    continuation_differences = {
        "reference": float(np.max(np.abs(loaded_reference - reference_vorticity))),
        "forcing": float(np.max(np.abs(loaded_forcing.field - forcing.field))),
        **{model: float(np.max(np.abs(loaded_states[model] - states[model]))) for model in MODELS},
    }
    passed = bool(
        loaded_step == step
        and max(rhs_differences.values()) < 1.0e-12
        and max(load_differences.values()) == 0.0
        and max(continuation_differences.values()) == 0.0
    )
    return {
        "pass": passed,
        "checkpoint": str(checkpoint),
        "optimized_rhs_relative_differences": rhs_differences,
        "load_maximum_absolute_differences": load_differences,
        "continued_maximum_absolute_differences": continuation_differences,
    }


def window_records(
    histories: dict[str, list[dict[str, Any]]],
    start_time: float,
) -> dict[str, list[dict[str, Any]]]:
    return {
        model: [record for record in records if record["time"] >= start_time]
        for model, records in histories.items()
    }


def time_mean(records: list[dict[str, Any]], quantity: str) -> float:
    return float(np.mean([record[quantity] for record in records]))


def time_mean_spectrum(records: list[dict[str, Any]]) -> np.ndarray:
    return np.mean([record["energy_spectrum"] for record in records], axis=0)


def summarize(
    args: argparse.Namespace,
    histories: dict[str, list[dict[str, Any]]],
    fine_reference_history: list[dict[str, Any]],
) -> dict[str, Any]:
    stationarity = assess_stationarity(fine_reference_history, 10.0)
    start_time = float(stationarity["values"]["window_start_time"])
    selected = window_records(histories, start_time)
    reference = selected["filtered_dns"]
    reference_energy = time_mean(reference, "energy")
    reference_enstrophy = time_mean(reference, "enstrophy")
    reference_spectrum = time_mean_spectrum(reference)
    models = {}
    budgets = {}
    for model in MODELS:
        records = selected[model]
        spectrum = time_mean_spectrum(records)
        budgets[model] = budget_summary(histories[model])
        models[model] = {
            "mean_energy_relative_error": abs(time_mean(records, "energy") - reference_energy)
            / reference_energy,
            "mean_enstrophy_relative_error": abs(
                time_mean(records, "enstrophy") - reference_enstrophy
            )
            / reference_enstrophy,
            "mean_spectral_relative_l2": float(
                np.linalg.norm(spectrum - reference_spectrum) / np.linalg.norm(reference_spectrum)
            ),
            "mean_instantaneous_spectral_relative_l2": time_mean(records, "spectral_relative_l2"),
            "mean_high_k_energy_fraction": time_mean(records, "high_k_energy_fraction"),
            "maximum_high_k_energy_fraction": max(
                record["high_k_energy_fraction"] for record in records
            ),
            "mean_sgs_power": time_mean(records, "sgs_power"),
            "mean_ssev_activation": time_mean(records, "activation"),
            "maximum_kkt_condition": max(record["kkt_condition"] for record in records),
            "maximum_divergence_relative": max(record["divergence_relative"] for record in records),
            "energy_budget_relative_residual": budgets[model]["relative_residual"],
            "time_mean_energy_spectrum": spectrum.tolist(),
        }
    structural_improvement = 1.0 - (
        models["structural"]["mean_spectral_relative_l2"]
        / models["no_sgs"]["mean_spectral_relative_l2"]
    )
    screen_checks = {
        "reference_stationary_and_resolved": stationarity["pass"],
        "all_models_finite_and_solenoidal": max(
            model["maximum_divergence_relative"] for model in models.values()
        )
        < 1.0e-12,
        "all_energy_budgets_close": max(
            model["energy_budget_relative_residual"] for model in models.values()
        )
        < 2.0e-3,
        "structural_energy_error_below_10_percent": models["structural"][
            "mean_energy_relative_error"
        ]
        < 0.10,
        "structural_enstrophy_error_below_10_percent": models["structural"][
            "mean_enstrophy_relative_error"
        ]
        < 0.10,
        "structural_spectrum_improves_by_25_percent": structural_improvement > 0.25,
        "structural_no_high_k_pileup": models["structural"]["maximum_high_k_energy_fraction"]
        < 0.01,
    }
    return {
        "gate": "B.1d stationary 64^3/32^3 paired screen",
        "status": "PASS" if all(screen_checks.values()) else "FAIL",
        "qualification": "reduced-resolution one-seed screen, not journal qualification",
        "configuration": configuration(args),
        "stationarity": stationarity,
        "measurement_window_start_time": start_time,
        "measurement_sample_count": len(reference),
        "models": models,
        "structural_spectral_improvement": structural_improvement,
        "screen_checks": screen_checks,
        "budgets": budgets,
        "histories": histories,
        "fine_reference_history": fine_reference_history,
        "reference_time_mean_energy_spectrum": reference_spectrum.tolist(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.reference_n != 2 * args.les_n:
        raise ValueError("reference_n must equal 2 * les_n")
    archive_verification = restart_verification(args, args.archive_dir / "restart_verification")
    if not archive_verification["pass"]:
        raise RuntimeError("raw checkpoint/restart verification failed")
    (
        reference_solver,
        les_solver,
        gaussian_delta,
        start_step,
        reference_vorticity,
        states,
        forcing,
        histories,
        fine_reference_history,
    ) = initialize(args)
    rhs_differences = verify_optimized_rhs(
        les_solver,
        states["structural"],
        forcing.field,
        gaussian_delta,
    )
    if max(rhs_differences.values()) > 1.0e-12:
        raise RuntimeError(f"optimized model RHS mismatch: {rhs_differences}")
    total_steps = int(round(args.end_time / args.dt))
    save_every = max(1, int(round(args.save_interval / args.dt)))
    checkpoint_every = max(1, int(round(args.checkpoint_interval / args.dt)))
    config = configuration(args)
    for step in range(start_step, total_steps + 1):
        time = step * args.dt
        if step % max(1, total_steps // 20) == 0:
            print(f"stationary-pair progress: {100.0 * step / total_steps:5.1f}%", flush=True)
        already_recorded = bool(
            histories["filtered_dns"]
            and abs(histories["filtered_dns"][-1]["time"] - time) < 0.5 * args.dt
        )
        if (step % save_every == 0 or step == total_steps) and not already_recorded:
            append_diagnostics(
                reference_solver,
                les_solver,
                reference_vorticity,
                states,
                forcing.reference_field(args.reference_n, gaussian_delta),
                forcing.field,
                gaussian_delta,
                time,
                histories,
                fine_reference_history,
            )
        if step % checkpoint_every == 0 or step == total_steps:
            write_checkpoint(
                args.archive_dir,
                step,
                time,
                config,
                reference_vorticity,
                states,
                forcing,
                histories,
                fine_reference_history,
            )
        if step == total_steps:
            break
        reference_vorticity, states = advance_one_step(
            args,
            reference_solver,
            les_solver,
            gaussian_delta,
            reference_vorticity,
            states,
            forcing,
        )
        if not np.all(np.isfinite(reference_vorticity)) or not all(
            np.all(np.isfinite(state)) for state in states.values()
        ):
            raise FloatingPointError(f"non-finite state at step {step + 1}")
    result = summarize(args, histories, fine_reference_history)
    result["archive_verification"] = archive_verification
    result["optimized_rhs_relative_differences"] = rhs_differences
    result["archive_directory"] = str(args.archive_dir)
    return result


def plot_histories(result: dict[str, Any], output: Path) -> None:
    histories = result["histories"]
    start = result["measurement_window_start_time"]
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.5), constrained_layout=True)
    quantities = (
        ("energy", "Resolved kinetic energy", None),
        ("enstrophy", "Resolved enstrophy", None),
        ("spectral_relative_l2", "Instantaneous spectrum error", "relative $L_2$ error"),
        ("high_k_energy_fraction", "High-wavenumber energy", "energy fraction"),
    )
    for axis, (quantity, title, ylabel) in zip(axes.flat, quantities, strict=True):
        models = MODELS if quantity == "spectral_relative_l2" else ("filtered_dns", *MODELS)
        for model in models:
            records = histories[model]
            axis.plot(
                [record["time"] for record in records],
                [record[quantity] for record in records],
                color=COLORS[model],
                linestyle="--" if model == "structural" else "-",
                linewidth=1.8 if model in ("filtered_dns", "sensed") else 1.25,
                label=DISPLAY_LABELS[model],
            )
        axis.axvspan(start, histories["filtered_dns"][-1]["time"], color=BLUE, alpha=0.06)
        if quantity == "high_k_energy_fraction":
            axis.axhline(0.01, color=GOLD, linestyle=":", label="1% gate")
        axis.set_title(title)
        axis.set_xlabel(r"$t$")
        if ylabel:
            axis.set_ylabel(ylabel)
        axis.grid(color=GRID, linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
    axes[1, 1].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Stationary forced-turbulence paired screen", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_spectra(result: dict[str, Any], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.5, 4.9), constrained_layout=True)
    spectra = {
        "filtered_dns": result["reference_time_mean_energy_spectrum"],
        **{model: result["models"][model]["time_mean_energy_spectrum"] for model in MODELS},
    }
    for model, values_list in spectra.items():
        values = np.asarray(values_list)
        wave = np.arange(len(values))
        positive = (wave > 0) & (values > 0.0)
        axis.loglog(
            wave[positive],
            values[positive],
            color=COLORS[model],
            linestyle="--" if model == "structural" else "-",
            marker="o" if model in ("filtered_dns", "sensed") else None,
            markersize=3,
            linewidth=1.8 if model in ("filtered_dns", "sensed") else 1.25,
            label=DISPLAY_LABELS[model],
        )
    axis.set_xlabel(r"wavenumber shell $k$")
    axis.set_ylabel(r"time-mean $E(k)$")
    axis.set_title("Stationary energy-spectrum reference overlay")
    axis.grid(color=GRID, linewidth=0.7, which="both")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_model_summary(result: dict[str, Any], output: Path) -> None:
    metrics = (
        ("mean_energy_relative_error", "Mean energy error"),
        ("mean_enstrophy_relative_error", "Mean enstrophy error"),
        ("mean_spectral_relative_l2", "Time-mean spectrum error"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.3, 3.9), constrained_layout=True)
    labels = [DISPLAY_LABELS[model] for model in MODELS]
    positions = np.arange(len(MODELS))
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        values = [result["models"][model][metric] for model in MODELS]
        bars = axis.bar(
            positions,
            values,
            color=[COLORS[model] for model in MODELS],
            edgecolor=INK,
            linewidth=0.6,
        )
        axis.bar_label(bars, fmt="%.3f", fontsize=8)
        axis.axhline(0.10, color=GOLD, linestyle=":", label="10% reference")
        axis.set_xticks(positions, labels, rotation=28, ha="right")
        axis.set_title(title)
        axis.set_ylabel("relative error")
        axis.grid(axis="y", color=GRID, linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Stationary model errors against filtered reference", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-n", type=int, default=64)
    parser.add_argument("--les-n", type=int, default=32)
    parser.add_argument("--viscosity", type=float, default=0.02)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--end-time", type=float, default=60.0)
    parser.add_argument("--save-interval", type=float, default=0.5)
    parser.add_argument("--checkpoint-interval", type=float, default=5.0)
    parser.add_argument("--forcing-rms", type=float, default=0.5)
    parser.add_argument("--correlation-time", type=float, default=0.2)
    parser.add_argument("--initial-rms", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--restart", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    plot_histories(result, args.figure_dir / "stage_4b3_stationary_pair_histories.png")
    plot_spectra(result, args.figure_dir / "stage_4b3_stationary_pair_spectra.png")
    plot_model_summary(result, args.figure_dir / "stage_4b3_stationary_pair_errors.png")
    if result["status"] != "PASS":
        raise SystemExit("STATIONARY PAIRED SCREEN FAIL")


if __name__ == "__main__":
    main()
