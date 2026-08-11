"""Physics and comparative-stability gate for all six vortex-ring cases."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

CASE_DIR = Path(__file__).resolve().parents[1]
SOLUTION_DIR = CASE_DIR / "solution"
FAMILIES = ("leapfrog", "collide")
VARIANTS = ("dns", "les", "les_stabilized")
EXPECTED_CASES = tuple(f"{family}_{variant}" for family in FAMILIES for variant in VARIANTS)
RING_INVARIANT_SCALE = 2.0 * np.pi**2


def column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    try:
        return np.asarray([float(row[name]) for row in rows])
    except KeyError as error:
        raise ValueError(f"missing diagnostic column {name!r}") from error


def vectors(rows: list[dict[str, str]], prefix: str) -> np.ndarray:
    return np.column_stack([column(rows, f"{prefix}_{axis}") for axis in "xyz"])


def relative_vector_drift(values: np.ndarray, scale: float) -> float:
    return float(np.linalg.norm(values - values[0], axis=1).max() / scale)


def inspect_case(case_name: str) -> dict[str, float]:
    case_dir = SOLUTION_DIR / case_name
    manifest_path = case_dir / "run_manifest.json"
    csv_path = case_dir / "samples" / "flow_integrals.csv"
    if not manifest_path.is_file() or not csv_path.is_file():
        raise ValueError("run manifest or flow-integral CSV is missing")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    variant = case_name.removeprefix("leapfrog_").removeprefix("collide_")
    status = manifest.get("status")
    completed_steps = manifest.get("completed_steps")
    requested_steps = manifest.get("requested_steps")
    if variant == "les_stabilized":
        if status != "completed" or completed_steps != requested_steps:
            raise ValueError("stabilized simulation did not reach its requested end time")
    elif status == "resolution_lost":
        if not manifest.get("termination_reason"):
            raise ValueError("resolution-loss termination has no recorded reason")
        if not isinstance(completed_steps, int) or completed_steps >= requested_steps:
            raise ValueError("invalid resolution-loss termination step")
    elif status != "completed" or completed_steps != requested_steps:
        raise ValueError(f"baseline run has invalid status {status!r}")

    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 3:
        raise ValueError("fewer than three diagnostic samples were written")

    time = column(rows, "time")
    energy = column(rows, "kinetic_energy")
    energy_rate = column(rows, "dEdt")
    sink = column(rows, "neg_nu_enstrophy")
    enstrophy = column(rows, "enstrophy")
    strength_magnitude = column(rows, "strength_magnitude")
    max_gamma = column(rows, "max_gamma")
    circulation = vectors(rows, "strength")
    impulse = vectors(rows, "impulse")
    angular_impulse = vectors(rows, "angular_impulse")
    particle_count = column(rows, "n_particles")
    overlap = column(rows, "overlap_ratio_max")
    divergence = column(rows, "vorticity_divergence_error")
    misalignment = column(rows, "strength_misalignment_deg")
    turbulent_viscosity = column(rows, "turbulent_viscosity_max")
    stabilization_viscosity = column(rows, "stabilization_viscosity_max")
    stabilization_active = column(rows, "stabilization_viscosity_active_fraction")
    projection_correction = column(rows, "invariant_projection_correction_ratio")
    regularization_events = column(rows, "regularization_events")
    regularization_transfer = column(rows, "regularization_cumulative_energy_transfer")
    regularization_energy_dissipation = column(
        rows, "regularization_max_energy_dissipation"
    )
    regularization_enstrophy_injection = column(
        rows, "regularization_max_enstrophy_injection"
    )
    regularization_enstrophy_dissipation = column(
        rows, "regularization_max_enstrophy_dissipation"
    )
    regularization_correction = column(rows, "regularization_max_correction_relative")
    regularization_projection_correction = column(
        rows, "regularization_max_projection_correction_relative"
    )
    regularization_circulation = column(
        rows, "regularization_max_circulation_error_relative"
    )
    regularization_impulse = column(
        rows, "regularization_max_linear_impulse_error_relative"
    )
    regularization_angular = column(
        rows, "regularization_max_angular_impulse_error_relative"
    )

    values = np.column_stack(
        (
            time,
            energy,
            energy_rate,
            sink,
            enstrophy,
            strength_magnitude,
            max_gamma,
            circulation,
            impulse,
            angular_impulse,
            particle_count,
            overlap,
            divergence,
            misalignment,
            turbulent_viscosity,
            stabilization_viscosity,
            stabilization_active,
            projection_correction,
            regularization_events,
            regularization_transfer,
            regularization_energy_dissipation,
            regularization_enstrophy_injection,
            regularization_enstrophy_dissipation,
            regularization_correction,
            regularization_projection_correction,
            regularization_circulation,
            regularization_impulse,
            regularization_angular,
        )
    )
    if not np.isfinite(values).all():
        raise ValueError("non-finite flow diagnostics were recorded")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("diagnostic time is not strictly increasing")
    if variant != "les_stabilized" and np.ptp(particle_count) != 0.0:
        raise ValueError("fixed-particle baseline changed population")
    if np.any(energy <= 0.0) or np.any(enstrophy < 0.0) or np.any(sink > 1.0e-8):
        raise ValueError("energy, enstrophy, or modeled dissipation has a nonphysical sign")

    interval_rate = np.diff(energy) / np.diff(time)
    rate_scale = max(float(np.max(np.abs(interval_rate))), energy[0] / (time[-1] - time[0]))
    rate_mismatch = float(np.max(np.abs(energy_rate[1:] - interval_rate)) / rate_scale)
    energy_injection = max(0.0, float(np.max(np.diff(energy)))) / energy[0]
    if np.any(np.diff(regularization_transfer) > 1.0e-8):
        raise ValueError("conservative regularization injected kinetic energy")
    filter_decay = -float(regularization_transfer[-1] - regularization_transfer[0])
    modeled_decay = -float(np.trapezoid(sink, time)) + filter_decay
    observed_decay = float(energy[0] - energy[-1])
    budget_error = abs(observed_decay - modeled_decay) / energy[0]

    metrics = {
        "end_time": float(time[-1]),
        "completed_fraction": float(completed_steps / requested_steps),
        "energy_ratio": float(energy[-1] / energy[0]),
        "energy_injection": energy_injection,
        "rate_mismatch": rate_mismatch,
        "budget_error": budget_error,
        "circulation_drift": relative_vector_drift(circulation, strength_magnitude[0]),
        "impulse_drift": relative_vector_drift(impulse, RING_INVARIANT_SCALE),
        "angular_drift": relative_vector_drift(angular_impulse, RING_INVARIANT_SCALE),
        "max_overlap": float(np.max(overlap)),
        "max_divergence": float(np.max(divergence)),
        "max_misalignment": float(np.max(misalignment)),
        "strength_growth": float(np.max(strength_magnitude) / strength_magnitude[0]),
        "max_gamma_growth": float(np.max(max_gamma) / max_gamma[0]),
        "projection_correction": float(np.max(projection_correction)),
        "turbulent_viscosity": float(np.max(turbulent_viscosity)),
        "stabilization_viscosity": float(np.max(stabilization_viscosity)),
        "stabilization_active": float(np.max(stabilization_active)),
        "regularization_events": float(np.max(regularization_events)),
        "regularization_filter_decay": filter_decay / energy[0],
        "regularization_energy_dissipation": float(
            np.max(regularization_energy_dissipation)
        ),
        "regularization_enstrophy_injection": float(
            np.max(regularization_enstrophy_injection)
        ),
        "regularization_enstrophy_dissipation": float(
            np.max(regularization_enstrophy_dissipation)
        ),
        "regularization_correction": float(np.max(regularization_correction)),
        "regularization_projection_correction": float(
            np.max(regularization_projection_correction)
        ),
        "regularization_circulation": float(np.max(regularization_circulation)),
        "regularization_impulse": float(np.max(regularization_impulse)),
        "regularization_angular": float(np.max(regularization_angular)),
    }

    limits = {
        "energy_injection": 2.0e-5,
        "rate_mismatch": 2.0e-6,
        "budget_error": 0.02,
        "circulation_drift": 5.0e-5,
        "impulse_drift": 5.0e-5,
        "angular_drift": 5.0e-4,
        "max_overlap": 1.25,
        "max_divergence": 0.25,
        "max_misalignment": 55.0 if variant != "les_stabilized" else 45.0,
        "projection_correction": 0.25,
        "regularization_enstrophy_injection": 5.0e-6,
        "regularization_energy_dissipation": 0.30,
        "regularization_enstrophy_dissipation": 0.15,
        "regularization_correction": 0.5,
        "regularization_projection_correction": 0.20,
        "regularization_circulation": 1.0e-5,
        "regularization_impulse": 1.0e-5,
        "regularization_angular": 1.0e-5,
    }
    failures = [
        f"{name}={metrics[name]:.3g} > {limit:.3g}"
        for name, limit in limits.items()
        if metrics[name] > limit
    ]

    viscosity_epsilon = 1.0e-12
    if variant == "dns":
        if metrics["turbulent_viscosity"] > viscosity_epsilon:
            failures.append("DNS has nonzero turbulent viscosity")
        if metrics["stabilization_viscosity"] > viscosity_epsilon:
            failures.append("DNS has nonzero stabilization viscosity")
        if metrics["regularization_events"] > 0.0:
            failures.append("DNS performed conservative regularization")
    elif variant == "les":
        if metrics["turbulent_viscosity"] <= viscosity_epsilon:
            failures.append("LES never activated its eddy viscosity")
        if metrics["stabilization_viscosity"] > viscosity_epsilon:
            failures.append("plain LES has nonzero stabilization viscosity")
        if metrics["regularization_events"] > 0.0:
            failures.append("plain LES performed conservative regularization")
    else:
        if metrics["turbulent_viscosity"] <= viscosity_epsilon:
            failures.append("stabilized LES never activated its eddy viscosity")
        if metrics["stabilization_viscosity"] <= viscosity_epsilon:
            failures.append("stabilized LES never activated residual viscosity")
        if metrics["stabilization_active"] <= 0.0:
            failures.append("stabilized LES has no active residual-viscosity particles")
        if metrics["regularization_events"] <= 0.0:
            failures.append("stabilized LES never regularized its particle cloud")
        if metrics["regularization_filter_decay"] <= 0.0:
            failures.append("stabilized LES recorded no dissipative filter transfer")

    if failures:
        raise ValueError("; ".join(failures))
    return metrics


def comparative_failures(metrics: dict[str, dict[str, float]]) -> list[str]:
    failures: list[str] = []
    indicators = ("strength_growth", "max_divergence", "max_misalignment")
    for family in FAMILIES:
        dns = metrics[f"{family}_dns"]
        les = metrics[f"{family}_les"]
        stabilized = metrics[f"{family}_les_stabilized"]
        if les["completed_fraction"] < 1.05 * dns["completed_fraction"]:
            failures.append(f"{family}: LES does not outlive DNS by at least 5%")
        les_improvements = [les[name] <= 0.95 * dns[name] for name in indicators]
        stabilized_improvements = [
            stabilized[name] <= 0.90 * les[name] for name in indicators
        ]
        if sum(les_improvements) < 2:
            failures.append(f"{family}: LES does not improve at least two stability indicators")
        if sum(stabilized_improvements) < 2:
            failures.append(
                f"{family}: stabilization does not improve at least two indicators by 10%"
            )
    return failures


def main() -> None:
    requested = tuple(sys.argv[1:]) or EXPECTED_CASES
    unknown = [name for name in requested if name not in EXPECTED_CASES]
    if unknown:
        raise SystemExit("Unknown case(s): " + ", ".join(unknown))
    metrics: dict[str, dict[str, float]] = {}
    failures: list[str] = []
    for case_name in requested:
        try:
            metrics[case_name] = inspect_case(case_name)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            failures.append(f"{case_name}: {error}")

    if not failures and requested == EXPECTED_CASES:
        failures.extend(comparative_failures(metrics))
    if failures:
        raise SystemExit("FAIL:\n  - " + "\n  - ".join(failures))

    scope = "all six cases" if requested == EXPECTED_CASES else f"{len(requested)} selected case(s)"
    print(f"PASS: {scope} ended with physical budgets and invariant conservation.")
    for name in requested:
        item = metrics[name]
        print(
            f"  {name:28s} duration={item['completed_fraction']:.1%}, "
            f"budget={item['budget_error']:.2%}, "
            f"max|Gamma|/initial={item['max_gamma_growth']:.2f}, "
            f"max div(omega)={item['max_divergence']:.3f}, "
            f"projection={item['projection_correction']:.2%}"
        )


if __name__ == "__main__":
    main()
