#!/usr/bin/env python3
"""Track the physical quasi-steady criterion through a raw ring trajectory."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/openonda_relax_history_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/openonda_relax_history_cache")

import h5py
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.experiments.stage_5b_relaxed_ring_analysis import (  # noqa: E402
    archer_moments,
    load_state,
    modal_metrics,
)
from scripts.experiments.stage_5b_ring_quasi_steady import (  # noqa: E402
    GAUSSIAN_SPEED,
    PRIMARY_CORE_FRACTION,
    RELAXED_EMPIRICAL_SPEED,
    sample_solver,
    serializable_metrics,
)
from source.solvers.VPM import VPMSolver  # noqa: E402
from source.solvers.VPM.io import CheckpointManager  # noqa: E402

INK = "#20252a"
BLUE = "#286f9b"
GOLD = "#d9973b"
GREEN = "#35845d"
GREY = "#8a99a8"
GRID = "#d8dde2"

# These operational limits are fixed before examining the long trajectory.
# They demand a clear approach to the exact zero-residual theory, agreement of
# two independent propagation-speed estimates, and a settled time history.
LIMITS = {
    "collapse_residual": 0.05,
    "advective_residual": 0.05,
    "fitted_speed_relative_difference": 0.02,
    "last_three_relative_range": 0.10,
    "circulation_relative_drift": 1.0e-3,
    "impulse_relative_drift": 1.0e-3,
    "maximum_axisymmetry_mode_amplitude": 1.0e-4,
    "energy_balance_relative_residual": 5.0e-2,
    "maximum_invariant_projection_correction_ratio": 1.0e-2,
}


def checkpoint_time(path: Path) -> float:
    with h5py.File(path, "r") as handle:
        attributes = handle["solver"].attrs
        if "time" in attributes:
            return float(attributes["time"])
        if "flow_time" in attributes:
            return float(attributes["flow_time"])
        raise KeyError("checkpoint has neither time nor legacy flow_time")


def trajectory_files(run_directory: Path, label: str) -> list[Path]:
    candidates = list(run_directory.glob(f"vpm_{label}*.h5"))
    by_time: dict[float, Path] = {}
    for path in candidates:
        time = checkpoint_time(path)
        current = by_time.get(time)
        if current is None or path.stem.endswith("_final"):
            by_time[time] = path
    return [by_time[time] for time in sorted(by_time)]


def local_speeds(times: np.ndarray, centers: np.ndarray) -> np.ndarray:
    if len(times) < 3:
        return np.gradient(centers, times, edge_order=1)
    return np.gradient(centers, times, edge_order=2)


def relative_range(values: np.ndarray) -> float:
    return float(np.ptp(values) / max(abs(float(np.mean(values))), np.finfo(float).tiny))


def cumulative_energy_residuals(run_directory: Path, target_times: np.ndarray) -> np.ndarray:
    """Compare measured energy loss with the integrated exact molecular sink."""
    path = run_directory / "samples/flow_integrals.csv"
    history = np.genfromtxt(path, delimiter=",", names=True)
    time = np.atleast_1d(history["time"])
    energy = np.atleast_1d(history["kinetic_energy"])
    dissipation = np.atleast_1d(history["neg_nu_enstrophy"])
    order = np.argsort(time)
    time, energy, dissipation = time[order], energy[order], dissipation[order]
    residuals: list[float] = []
    for target in target_times:
        if target <= time[0]:
            residuals.append(0.0)
            continue
        inside = time < target
        integration_time = np.append(time[inside], target)
        integration_dissipation = np.append(
            dissipation[inside], np.interp(target, time, dissipation)
        )
        predicted = float(np.trapezoid(integration_dissipation, integration_time))
        measured = float(np.interp(target, time, energy) - energy[0])
        residuals.append(abs((measured - predicted) / predicted))
    return np.asarray(residuals)


def projection_correction_history(run_directory: Path, target_times: np.ndarray) -> np.ndarray:
    path = run_directory / "samples/flow_integrals.csv"
    history = np.genfromtxt(path, delimiter=",", names=True)
    time = np.atleast_1d(history["time"])
    correction = np.atleast_1d(history["invariant_projection_correction_ratio"])
    order = np.argsort(time)
    return np.interp(target_times, time[order], correction[order])


def evaluate(rows: list[dict[str, object]]) -> dict[str, object]:
    if len(rows) < 3:
        return {"status": "INCOMPLETE", "reason": "at least three saved times are required"}
    initial = rows[0]
    final = rows[-1]
    last = rows[-3:]
    plateau = {
        key: relative_range(np.asarray([float(row[key]) for row in last]))
        for key in ("collapse_residual", "advective_residual", "fitted_translation_speed")
    }
    circulation_drift = abs(
        float(final["tube_circulation"]) / float(initial["tube_circulation"]) - 1.0
    )
    impulse_drift = abs(float(final["impulse_x"]) / float(initial["impulse_x"]) - 1.0)
    observed = {
        "final_collapse_residual": float(final["collapse_residual"]),
        "final_advective_residual": float(final["advective_residual"]),
        "final_fitted_speed_relative_difference": float(final["fitted_speed_relative_difference"]),
        "last_three_relative_ranges": plateau,
        "circulation_relative_drift": circulation_drift,
        "impulse_relative_drift": impulse_drift,
        "maximum_axisymmetry_mode_amplitude": max(
            float(row["maximum_mode_amplitude"]) for row in rows
        ),
        "energy_balance_relative_residual": float(final["energy_balance_relative_residual"]),
        "maximum_invariant_projection_correction_ratio": max(
            float(row["invariant_projection_correction_ratio"]) for row in rows
        ),
    }
    checks = {
        "single_valued_relation": observed["final_collapse_residual"]
        <= LIMITS["collapse_residual"],
        "small_material_residual": observed["final_advective_residual"]
        <= LIMITS["advective_residual"],
        "independent_speed_agreement": observed["final_fitted_speed_relative_difference"]
        <= LIMITS["fitted_speed_relative_difference"],
        "time_plateau": max(plateau.values()) <= LIMITS["last_three_relative_range"],
        "circulation": circulation_drift <= LIMITS["circulation_relative_drift"],
        "impulse": impulse_drift <= LIMITS["impulse_relative_drift"],
        "axisymmetry": observed["maximum_axisymmetry_mode_amplitude"]
        <= LIMITS["maximum_axisymmetry_mode_amplitude"],
        "energy_balance": observed["energy_balance_relative_residual"]
        <= LIMITS["energy_balance_relative_residual"],
        "small_invariant_projection": observed["maximum_invariant_projection_correction_ratio"]
        <= LIMITS["maximum_invariant_projection_correction_ratio"],
    }
    if all(checks.values()):
        status = "QUASI_STEADY"
    elif checks["time_plateau"] and not (
        checks["single_valued_relation"]
        and checks["small_material_residual"]
        and checks["independent_speed_agreement"]
    ):
        status = "PLATEAUED_ABOVE_TARGET"
    else:
        status = "CONTINUE_RELAXATION"
    return {
        "status": status,
        "limits": LIMITS,
        "observed": observed,
        "checks": checks,
    }


def analyze(run_directory: Path, label: str, grid_size: int) -> tuple[list[dict], dict]:
    paths = trajectory_files(run_directory, label)
    if not paths:
        raise FileNotFoundError(f"no raw states found for {label} in {run_directory}")
    states = [load_state(path) for path in paths]
    moments = [archer_moments(state) for state in states]
    times = np.asarray([float(state["time"]) for state in states])
    centers = np.asarray([item["axial_centroid"] for item in moments])
    speeds = local_speeds(times, centers)
    energy_residuals = cumulative_energy_residuals(run_directory, times)
    projection_ratios = projection_correction_history(run_directory, times)

    final_base = run_directory / f"vpm_{label}_final"
    solver = VPMSolver.continue_from_backup(str(final_base))
    if solver is None:
        raise RuntimeError(f"could not restore {final_base}")

    rows: list[dict] = []
    for path, state, moment, speed, energy_residual, projection_ratio in zip(
        paths,
        states,
        moments,
        speeds,
        energy_residuals,
        projection_ratios,
        strict=True,
    ):
        CheckpointManager.load_numerical_state(solver, path)
        sample = sample_solver(
            solver,
            grid_size,
            axial_center=float(moment["axial_centroid"]),
            translation_speed=float(speed),
        )
        metrics = sample["sensitivity"][f"{PRIMARY_CORE_FRACTION:.2f}"]
        modes = modal_metrics(state)
        rows.append(
            {
                "raw_state": str(path.relative_to(ROOT)),
                "time_star": float(state["time"]),
                "measured_translation_speed": float(speed),
                "energy_balance_relative_residual": float(energy_residual),
                "invariant_projection_correction_ratio": float(projection_ratio),
                **moment,
                **modes,
                **serializable_metrics(metrics),
            }
        )
    return rows, evaluate(rows)


def plot(rows: list[dict], gate: dict, output: Path) -> None:
    time = np.asarray([float(row["time_star"]) for row in rows])
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)
    fig.suptitle(
        r"Axisymmetric ring relaxation toward $\omega_\phi/r=F(\psi)$",
        color=INK,
        fontsize=13,
        fontweight="bold",
    )

    axis = axes[0]
    axis.plot(
        time,
        [row["collapse_residual"] for row in rows],
        "o-",
        color=BLUE,
        label=r"$q(\psi)$ scatter",
    )
    axis.plot(
        time,
        [row["advective_residual"] for row in rows],
        "s-",
        color=GOLD,
        label="material residual",
    )
    axis.axhline(LIMITS["collapse_residual"], color=INK, linestyle="--", label="frozen limit")
    axis.axhline(0.0, color=GREY, linestyle=":", label="exact steady value")
    axis.set_xlabel(r"time $t^*=t\Gamma/R^2$")
    axis.set_ylabel("dimensionless residual")
    axis.set_title("Quasi-steady residuals")
    axis.legend(frameon=False, fontsize=7)

    axis = axes[1]
    axis.plot(
        time,
        [row["measured_translation_speed"] for row in rows],
        "o-",
        color=BLUE,
        label="measured VPM",
    )
    axis.plot(
        time,
        [row["fitted_translation_speed"] for row in rows],
        "s-",
        color=GREEN,
        label="best steady frame",
    )
    axis.axhline(GAUSSIAN_SPEED, color=INK, linestyle="--", label="Gaussian theory")
    axis.axhline(RELAXED_EMPIRICAL_SPEED, color=GOLD, linestyle=":", label="relaxed-core formula")
    axis.set_xlabel(r"time $t^*$")
    axis.set_ylabel(r"$UR/\Gamma$")
    axis.set_title("Translation-speed consistency")
    axis.legend(frameon=False, fontsize=7)

    axis = axes[2]
    circulation_0 = float(rows[0]["tube_circulation"])
    impulse_0 = float(rows[0]["impulse_x"])
    axis.semilogy(
        time,
        [abs(float(row["tube_circulation"]) / circulation_0 - 1.0) + 1.0e-16 for row in rows],
        "o-",
        color=BLUE,
        label="circulation drift",
    )
    axis.semilogy(
        time,
        [abs(float(row["impulse_x"]) / impulse_0 - 1.0) + 1.0e-16 for row in rows],
        "s-",
        color=GREEN,
        label="impulse drift",
    )
    axis.semilogy(
        time,
        [row["maximum_mode_amplitude"] for row in rows],
        "^-",
        color=GOLD,
        label="largest artificial mode",
    )
    axis.semilogy(
        time[1:],
        [row["energy_balance_relative_residual"] for row in rows[1:]],
        "d-",
        color="#7a5195",
        label="energy-balance residual",
    )
    if max(float(row["invariant_projection_correction_ratio"]) for row in rows) > 0.0:
        axis.semilogy(
            time,
            [row["invariant_projection_correction_ratio"] for row in rows],
            "v-",
            color="#ef5675",
            label="invariant-projection correction",
        )
    axis.axhline(1.0e-3, color=INK, linestyle="--", label="circulation/impulse limit")
    axis.axhline(1.0e-4, color=GREY, linestyle=":", label="axisymmetry limit")
    axis.axhline(5.0e-2, color="#7a5195", linestyle="--", label="energy-balance limit")
    axis.axhline(1.0e-2, color="#ef5675", linestyle=":", label="projection limit")
    axis.set_xlabel(r"time $t^*$")
    axis.set_ylabel("relative magnitude")
    axis.set_title(f"Health: {gate['status'].replace('_', ' ').lower()}")
    axis.legend(frameon=False, fontsize=7)

    for axis in axes:
        axis.grid(True, color=GRID, linewidth=0.7, alpha=0.75)
        axis.spines[["top", "right"]].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--grid-size", type=int, default=97)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    rows, gate = analyze(args.run_directory.resolve(), args.label, args.grid_size)
    payload = {
        "stage": "5B axisymmetric relaxation history",
        "status": gate["status"],
        "criterion": "omega_phi/r must become a single-valued function of translating-frame psi",
        "grid_size": args.grid_size,
        "gate": gate,
        "trajectory": rows,
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    plot(rows, gate, args.figure)
    print(json.dumps({"status": gate["status"], "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
