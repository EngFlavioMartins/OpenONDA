"""Physics and comparative-stability gate for all six vortex-ring cases."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import sys

import numpy as np

CASE_DIR = Path(__file__).resolve().parents[1]
SOLUTION_DIR = CASE_DIR / "solution"
SAMPLES_DIR = CASE_DIR / "samples"
FAMILIES = ("leapfrog", "collide")
VARIANTS = ("dns", "les", "les_stabilized")
EXPECTED_CASES = tuple(f"{family}_{variant}" for family in FAMILIES for variant in VARIANTS)
RING_INVARIANT_SCALE = 2.0 * np.pi**2
EXPECTED_SMAGORINSKY = {"leapfrog": 0.16, "collide": 0.32}


def column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    try:
        return np.asarray([float(row[name]) for row in rows])
    except KeyError as error:
        raise ValueError(f"missing diagnostic column {name!r}") from error


def vectors(rows: list[dict[str, str]], prefix: str) -> np.ndarray:
    return np.column_stack([column(rows, f"{prefix}_{axis}") for axis in "xyz"])


def relative_vector_drift(values: np.ndarray, scale: float) -> float:
    return float(np.linalg.norm(values - values[0], axis=1).max() / scale)


def stability_indicators(case_name: str, through_time: float) -> dict[str, float]:
    """Return peak instability measures over a common physical-time window."""
    csv_path = SAMPLES_DIR / case_name / "flow_integrals.csv"
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    time = column(rows, "time")
    selected = time <= through_time + 32.0 * np.finfo(float).eps * max(1.0, through_time)
    if not np.any(selected):
        raise ValueError(f"{case_name} has no diagnostic sample through t={through_time:.6g}")
    strength = column(rows, "vortex_strength_magnitude_sum")[selected]
    return {
        "vortex_strength_magnitude_sum_growth": float(np.max(strength) / strength[0]),
        "max_vorticity_divergence_error": float(
            np.max(column(rows, "vorticity_divergence_error")[selected])
        ),
        "max_vortex_strength_misalignment_degrees": float(
            np.max(column(rows, "vortex_strength_misalignment_degrees")[selected])
        ),
    }


def initial_state(case_name: str) -> np.ndarray:
    """Return the shared particle-state diagnostics before model evolution begins."""
    csv_path = SAMPLES_DIR / case_name / "flow_integrals.csv"
    with csv_path.open(newline="") as stream:
        row = next(csv.DictReader(stream))
    names = (
        "total_kinetic_energy",
        "total_enstrophy",
        "vortex_strength_magnitude_sum",
        "net_vortex_strength_x",
        "net_vortex_strength_y",
        "net_vortex_strength_z",
        "linear_impulse_x",
        "linear_impulse_y",
        "linear_impulse_z",
        "angular_impulse_x",
        "angular_impulse_y",
        "angular_impulse_z",
        "n_particles_total",
        "max_overlap_ratio",
        "vorticity_divergence_error",
        "vortex_strength_misalignment_degrees",
    )
    return np.asarray([float(row[name]) for name in names])


def inspect_case(case_name: str) -> dict[str, float]:
    case_dir = SOLUTION_DIR / case_name
    manifest_path = case_dir / "run_manifest.json"
    csv_path = SAMPLES_DIR / case_name / "flow_integrals.csv"
    ring_csv_path = SAMPLES_DIR / case_name / "ring_diagnostics.csv"
    if not all(path.is_file() for path in (manifest_path, csv_path, ring_csv_path)):
        raise ValueError("run manifest or built-in diagnostic CSV is missing")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    family = case_name.split("_", maxsplit=1)[0]
    variant = case_name.removeprefix("leapfrog_").removeprefix("collide_")
    if manifest.get("case") != case_name or manifest.get("model") != variant:
        raise ValueError("run manifest identifies a different case or model")
    status = manifest.get("status")
    completed_steps = manifest.get("completed_steps")
    requested_steps = manifest.get("requested_steps")
    diagnostic_frequency = manifest.get("diagnostic_frequency")
    snapshot_frequency = manifest.get("snapshot_frequency")
    if not isinstance(diagnostic_frequency, int) or not 1 <= diagnostic_frequency <= 5:
        raise ValueError("flow diagnostics are not sampled at least every five steps")
    if not isinstance(snapshot_frequency, int) or not 1 <= snapshot_frequency <= 10:
        raise ValueError("particle states are not saved at least every ten steps")
    if float(manifest.get("particle_spacing", np.inf)) > 0.04:
        raise ValueError("initial vortex-ring particle spacing is too coarse")
    if int(manifest.get("initial_n_particles_total", 0)) < 6_000:
        raise ValueError("initial vortex rings contain too few particles")
    expected_cs = 0.0 if variant == "dns" else EXPECTED_SMAGORINSKY[family]
    if not np.isclose(float(manifest.get("smagorinsky_coefficient", np.nan)), expected_cs):
        raise ValueError("run used the wrong family-specific Smagorinsky coefficient")
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
    step = column(rows, "step")
    total_kinetic_energy = column(rows, "total_kinetic_energy")
    kinetic_energy_rate = column(rows, "kinetic_energy_rate")
    viscous_kinetic_energy_rate = column(rows, "viscous_kinetic_energy_rate")
    total_enstrophy = column(rows, "total_enstrophy")
    test_filtered_enstrophy = column(rows, "test_filtered_enstrophy")
    vortex_strength_magnitude_sum = column(rows, "vortex_strength_magnitude_sum")
    max_vortex_strength_magnitude = column(rows, "max_vortex_strength_magnitude")
    net_vortex_strength = vectors(rows, "net_vortex_strength")
    linear_impulse = vectors(rows, "linear_impulse")
    angular_impulse = vectors(rows, "angular_impulse")
    n_particles_total = column(rows, "n_particles_total")
    mean_core_radius = column(rows, "mean_core_radius")
    mean_overlap_ratio = column(rows, "mean_overlap_ratio")
    max_overlap_ratio = column(rows, "max_overlap_ratio")
    divergence = column(rows, "vorticity_divergence_error")
    misalignment = column(rows, "vortex_strength_misalignment_degrees")
    mean_eddy_viscosity = column(rows, "mean_eddy_viscosity")
    max_eddy_viscosity = column(rows, "max_eddy_viscosity")
    mean_effective_viscosity = column(rows, "mean_effective_viscosity")
    max_effective_viscosity = column(rows, "max_effective_viscosity")
    stabilization_kinematic_viscosity = column(rows, "max_stabilization_kinematic_viscosity")
    stabilization_active = column(rows, "stabilization_kinematic_viscosity_active_fraction")
    projection_correction = column(rows, "invariant_projection_correction_ratio")
    core_spreading_events = column(rows, "core_spreading_moment_projection_events")
    core_spreading_correction = column(rows, "core_spreading_max_moment_correction_relative")
    core_spreading_vortex_strength_error_relative = column(
        rows, "core_spreading_max_vortex_strength_error_relative"
    )
    core_spreading_impulse = column(rows, "core_spreading_max_linear_impulse_error_relative")
    core_spreading_angular = column(rows, "core_spreading_max_angular_impulse_error_relative")
    n_regularization_events = column(rows, "n_regularization_events")
    regularization_transfer = column(
        rows, "regularization_cumulative_total_kinetic_energy_transfer"
    )
    regularization_total_kinetic_energy_dissipation = column(
        rows, "regularization_max_total_kinetic_energy_dissipation"
    )
    regularization_total_enstrophy_injection = column(
        rows, "regularization_max_total_enstrophy_injection"
    )
    regularization_total_enstrophy_dissipation = column(
        rows, "regularization_max_total_enstrophy_dissipation"
    )
    regularization_correction = column(rows, "regularization_max_correction_relative")
    regularization_projection_correction = column(
        rows, "regularization_max_projection_correction_relative"
    )
    n_regularization_adaptive_core_events = column(rows, "n_regularization_adaptive_core_events")
    regularization_max_core_radius = column(rows, "regularization_max_core_radius")
    regularization_vortex_strength_error_relative = column(
        rows, "regularization_max_vortex_strength_error_relative"
    )
    regularization_impulse = column(rows, "regularization_max_linear_impulse_error_relative")
    regularization_angular = column(rows, "regularization_max_angular_impulse_error_relative")

    try:
        all_diagnostics = np.asarray(
            [[float(value) for value in row.values()] for row in rows],
            dtype=float,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("flow diagnostics contain a non-numeric value") from error
    if not np.isfinite(all_diagnostics).all():
        raise ValueError("a non-finite value appears in the exported flow diagnostics")

    required_values = np.column_stack(
        (
            time,
            step,
            total_kinetic_energy,
            kinetic_energy_rate,
            viscous_kinetic_energy_rate,
            total_enstrophy,
            vortex_strength_magnitude_sum,
            max_vortex_strength_magnitude,
            net_vortex_strength,
            linear_impulse,
            angular_impulse,
            n_particles_total,
            max_overlap_ratio,
            divergence,
            misalignment,
            max_eddy_viscosity,
            stabilization_kinematic_viscosity,
            stabilization_active,
            projection_correction,
            core_spreading_events,
            core_spreading_correction,
            core_spreading_vortex_strength_error_relative,
            core_spreading_impulse,
            core_spreading_angular,
            n_regularization_events,
            regularization_transfer,
            regularization_total_kinetic_energy_dissipation,
            regularization_total_enstrophy_injection,
            regularization_total_enstrophy_dissipation,
            regularization_correction,
            regularization_projection_correction,
            n_regularization_adaptive_core_events,
            regularization_max_core_radius,
            regularization_vortex_strength_error_relative,
            regularization_impulse,
            regularization_angular,
        )
    )
    if not np.isfinite(required_values).all():
        raise ValueError("a required flow diagnostic is non-finite")
    expected_steps = np.arange(0, completed_steps + 1, diagnostic_frequency, dtype=float)
    if expected_steps[-1] != completed_steps:
        expected_steps = np.append(expected_steps, completed_steps)
    if not np.array_equal(step, expected_steps):
        raise ValueError("diagnostic samples are missing, duplicated, or at the wrong steps")
    expected_time_step_size = float(manifest["requested_end_time"]) / requested_steps
    if not np.allclose(time, step * expected_time_step_size, rtol=0.0, atol=1.0e-9):
        raise ValueError("diagnostic time does not match step times the configured time step")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("diagnostic time is not strictly increasing")

    with ring_csv_path.open(newline="") as stream:
        ring_rows = list(csv.DictReader(stream))
    try:
        ring_values = np.asarray(
            [[float(value) for value in row.values()] for row in ring_rows],
            dtype=float,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("ring sampler contains a non-numeric value") from error
    if not np.isfinite(ring_values).all():
        raise ValueError("ring sampler contains a non-finite value")
    if (
        np.any(column(ring_rows, "major_radius") <= 0.0)
        or np.any(column(ring_rows, "tube_circulation") <= 0.0)
        or np.any(column(ring_rows, "vortex_strength_magnitude_sum") <= 0.0)
        or np.any(column(ring_rows, "max_vortex_strength_magnitude") <= 0.0)
        or np.any(column(ring_rows, "linear_impulse_magnitude") < 0.0)
        or np.any(column(ring_rows, "impulse_radius") < 0.0)
    ):
        raise ValueError(
            "ring sampler contains a nonphysical radius, net_vortex_strength, or linear_impulse"
        )
    initial_ring_rows = ring_rows[:2]
    expected_centres = (-0.5, 0.5) if case_name.startswith("leapfrog_") else (-2.5, 2.5)
    if not np.allclose(
        column(initial_ring_rows, "vortex_centroid_x"),
        expected_centres,
        rtol=0.0,
        atol=2.0e-3,
    ):
        raise ValueError("sampled initial ring centres do not match the prescribed geometry")
    if not np.allclose(
        column(initial_ring_rows, "major_radius"),
        1.0,
        rtol=2.0e-2,
        atol=0.0,
    ) or not np.allclose(
        column(initial_ring_rows, "tube_circulation"),
        np.pi,
        rtol=1.0e-2,
        atol=0.0,
    ):
        raise ValueError("sampled initial radius or tube net_vortex_strength is under-resolved")
    ring_steps = np.asarray([int(row["step"]) for row in ring_rows])
    ring_groups = np.asarray([int(row["group_id"]) for row in ring_rows])
    expected_ring_steps = np.repeat(expected_steps.astype(int), 2)
    expected_ring_groups = np.tile((0, 1), len(expected_steps))
    if not np.array_equal(ring_steps, expected_ring_steps) or not np.array_equal(
        ring_groups, expected_ring_groups
    ):
        raise ValueError("grouped ring diagnostics are missing, duplicated, or out of order")
    ring_time = np.asarray([float(row["time"]) for row in ring_rows])
    if not np.allclose(
        ring_time,
        np.repeat(time, 2),
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise ValueError("ring-sampler times do not match the global diagnostics")

    numbered_snapshots = {
        int(match.group(1))
        for path in case_dir.glob(f"vpm_{case_name}_*.h5")
        if (match := re.search(r"_(\d{6})\.h5$", path.name))
    }
    expected_snapshots = set(range(0, completed_steps + 1, snapshot_frequency))
    if numbered_snapshots != expected_snapshots:
        raise ValueError("scheduled particle-state snapshots are missing or duplicated")
    if any(
        not (case_dir / f"vpm_{case_name}_{snapshot:06d}.xdmf").is_file()
        for snapshot in expected_snapshots
    ):
        raise ValueError("a scheduled particle snapshot has no XDMF descriptor")
    if not all(
        (case_dir / f"vpm_{case_name}_final{suffix}").is_file() for suffix in (".h5", ".xdmf")
    ):
        raise ValueError("final particle state or its XDMF descriptor is missing")
    if variant != "les_stabilized" and np.ptp(n_particles_total) != 0.0:
        raise ValueError("fixed-particle baseline changed population")
    if (
        np.any(total_kinetic_energy <= 0.0)
        or np.any(total_enstrophy < 0.0)
        or np.any(test_filtered_enstrophy < 0.0)
        or np.any(viscous_kinetic_energy_rate > 1.0e-8)
    ):
        raise ValueError(
            "total_kinetic_energy, total_enstrophy, or modeled dissipation has a nonphysical sign"
        )
    if (
        np.any(n_particles_total <= 0.0)
        or np.any(n_particles_total != np.floor(n_particles_total))
        or np.any(vortex_strength_magnitude_sum <= 0.0)
        or np.any(max_vortex_strength_magnitude <= 0.0)
        or np.any(mean_core_radius <= 0.0)
    ):
        raise ValueError("particle population, net_vortex_strength, or core size is nonphysical")
    nonnegative = np.column_stack(
        (
            mean_eddy_viscosity,
            max_eddy_viscosity,
            mean_effective_viscosity,
            max_effective_viscosity,
            mean_overlap_ratio,
            max_overlap_ratio,
            divergence,
        )
    )
    if np.any(nonnegative < -1.0e-14) or np.any((misalignment < 0.0) | (misalignment > 180.0)):
        raise ValueError("viscosity, resolution, or alignment diagnostics are nonphysical")

    interval_rate = np.diff(total_kinetic_energy) / np.diff(time)
    rate_scale = max(
        float(np.max(np.abs(interval_rate))), total_kinetic_energy[0] / (time[-1] - time[0])
    )
    rate_mismatch = float(np.max(np.abs(kinetic_energy_rate[1:] - interval_rate)) / rate_scale)
    filter_rate = np.diff(regularization_transfer) / np.diff(time)
    modeled_rate = (
        0.5 * (viscous_kinetic_energy_rate[1:] + viscous_kinetic_energy_rate[:-1]) + filter_rate
    )
    modeled_rate_mismatch = float(
        np.sqrt(np.mean(np.square(interval_rate - modeled_rate)))
        / max(float(np.sqrt(np.mean(np.square(modeled_rate)))), 1.0e-30)
    )
    positive_kinetic_energy_rate = max(0.0, float(np.max(kinetic_energy_rate[1:]))) / rate_scale
    total_kinetic_energy_injection = (
        max(0.0, float(np.max(np.diff(total_kinetic_energy)))) / total_kinetic_energy[0]
    )
    if np.any(np.diff(regularization_transfer) > 1.0e-8):
        raise ValueError("conservative regularization injected total kinetic energy")
    filter_decay = -float(regularization_transfer[-1] - regularization_transfer[0])
    modeled_decay = -float(np.trapezoid(viscous_kinetic_energy_rate, time)) + filter_decay
    observed_decay = float(total_kinetic_energy[0] - total_kinetic_energy[-1])
    kinetic_energy_budget_error = abs(observed_decay - modeled_decay) / total_kinetic_energy[0]

    metrics = {
        "end_time": float(time[-1]),
        "completed_fraction": float(completed_steps / requested_steps),
        "total_kinetic_energy_ratio": float(total_kinetic_energy[-1] / total_kinetic_energy[0]),
        "total_kinetic_energy_injection": total_kinetic_energy_injection,
        "positive_kinetic_energy_rate": positive_kinetic_energy_rate,
        "rate_mismatch": rate_mismatch,
        "modeled_rate_mismatch": modeled_rate_mismatch,
        "kinetic_energy_budget_error": kinetic_energy_budget_error,
        "net_vortex_strength_drift_relative": relative_vector_drift(
            net_vortex_strength, vortex_strength_magnitude_sum[0]
        ),
        "linear_impulse_drift_relative": relative_vector_drift(
            linear_impulse, RING_INVARIANT_SCALE
        ),
        "angular_impulse_drift_relative": relative_vector_drift(
            angular_impulse, RING_INVARIANT_SCALE
        ),
        "max_overlap_ratio": float(np.max(max_overlap_ratio)),
        "max_vorticity_divergence_error": float(np.max(divergence)),
        "max_vortex_strength_misalignment_degrees": float(np.max(misalignment)),
        "vortex_strength_magnitude_sum_growth": float(
            np.max(vortex_strength_magnitude_sum) / vortex_strength_magnitude_sum[0]
        ),
        "max_vortex_strength_magnitude_growth": float(
            np.max(max_vortex_strength_magnitude) / max_vortex_strength_magnitude[0]
        ),
        "projection_correction": float(np.max(projection_correction)),
        "core_spreading_events": float(np.max(core_spreading_events)),
        "core_spreading_correction": float(np.max(core_spreading_correction)),
        "core_spreading_vortex_strength_error_relative": float(
            np.max(core_spreading_vortex_strength_error_relative)
        ),
        "core_spreading_impulse": float(np.max(core_spreading_impulse)),
        "core_spreading_angular": float(np.max(core_spreading_angular)),
        "max_eddy_viscosity": float(np.max(max_eddy_viscosity)),
        "stabilization_kinematic_viscosity": float(np.max(stabilization_kinematic_viscosity)),
        "stabilization_active": float(np.max(stabilization_active)),
        "n_regularization_events": float(np.max(n_regularization_events)),
        "regularization_filter_decay": filter_decay / total_kinetic_energy[0],
        "regularization_total_kinetic_energy_dissipation": float(
            np.max(regularization_total_kinetic_energy_dissipation)
        ),
        "regularization_total_enstrophy_injection": float(
            np.max(regularization_total_enstrophy_injection)
        ),
        "regularization_total_enstrophy_dissipation": float(
            np.max(regularization_total_enstrophy_dissipation)
        ),
        "regularization_correction": float(np.max(regularization_correction)),
        "regularization_projection_correction": float(np.max(regularization_projection_correction)),
        "n_regularization_adaptive_core_events": float(
            np.max(n_regularization_adaptive_core_events)
        ),
        "regularization_max_core_radius": float(np.max(regularization_max_core_radius)),
        "regularization_vortex_strength_error_relative": float(
            np.max(regularization_vortex_strength_error_relative)
        ),
        "regularization_impulse": float(np.max(regularization_impulse)),
        "regularization_angular": float(np.max(regularization_angular)),
        "diagnostic_samples": float(len(rows)),
        "state_snapshots": float(len(numbered_snapshots)),
    }

    limits = {
        "total_kinetic_energy_injection": 2.0e-5,
        "positive_kinetic_energy_rate": 1.0e-6,
        "rate_mismatch": 2.0e-6,
        "modeled_rate_mismatch": 0.20,
        "kinetic_energy_budget_error": 0.02,
        "net_vortex_strength_drift_relative": 5.0e-5,
        "linear_impulse_drift_relative": 5.0e-5,
        "angular_impulse_drift_relative": 5.0e-4,
        "max_overlap_ratio": 1.25,
        "max_vorticity_divergence_error": 0.25,
        "max_vortex_strength_misalignment_degrees": 55.0 if variant != "les_stabilized" else 45.0,
        "projection_correction": 0.25,
        "core_spreading_correction": 1.0e-3,
        "core_spreading_vortex_strength_error_relative": 1.0e-10,
        "core_spreading_impulse": 1.0e-10,
        "core_spreading_angular": 1.0e-10,
        "regularization_total_enstrophy_injection": 5.0e-6,
        "regularization_total_kinetic_energy_dissipation": 0.20,
        "regularization_total_enstrophy_dissipation": 0.15,
        "regularization_correction": 0.5,
        "regularization_projection_correction": (
            0.051 if case_name.startswith("leapfrog_") else 0.101
        ),
        "regularization_vortex_strength_error_relative": 1.0e-5,
        "regularization_impulse": 1.0e-5,
        "regularization_angular": 1.0e-5,
    }
    failures = [
        f"{name}={metrics[name]:.3g} > {limit:.3g}"
        for name, limit in limits.items()
        if metrics[name] > limit
    ]

    viscosity_epsilon = 1.0e-12
    if metrics["core_spreading_events"] <= 0.0:
        failures.append("core spreading never applied its moment-conserving correction")
    if variant == "dns":
        if metrics["max_eddy_viscosity"] > viscosity_epsilon:
            failures.append("DNS has nonzero turbulent viscosity")
        if metrics["stabilization_kinematic_viscosity"] > viscosity_epsilon:
            failures.append("DNS has nonzero stabilization viscosity")
        if metrics["n_regularization_events"] > 0.0:
            failures.append("DNS performed conservative regularization")
    elif variant == "les":
        if metrics["max_eddy_viscosity"] <= viscosity_epsilon:
            failures.append("LES never activated its eddy viscosity")
        if metrics["stabilization_kinematic_viscosity"] > viscosity_epsilon:
            failures.append("plain LES has nonzero stabilization viscosity")
        if metrics["n_regularization_events"] > 0.0:
            failures.append("plain LES performed conservative regularization")
    else:
        if metrics["max_eddy_viscosity"] <= viscosity_epsilon:
            failures.append("stabilized LES never activated its eddy viscosity")
        if metrics["stabilization_kinematic_viscosity"] <= viscosity_epsilon:
            failures.append("stabilized LES never activated residual viscosity")
        if metrics["stabilization_active"] <= 0.0:
            failures.append("stabilized LES has no active residual-viscosity particles")
        if metrics["n_regularization_events"] <= 0.0:
            failures.append("stabilized LES never regularized its particle cloud")
        if metrics["regularization_filter_decay"] <= 0.0:
            failures.append("stabilized LES recorded no dissipative filter transfer")

    if failures:
        raise ValueError("; ".join(failures))
    return metrics


def comparative_failures(metrics: dict[str, dict[str, float]]) -> list[str]:
    failures: list[str] = []
    indicators = (
        "vortex_strength_magnitude_sum_growth",
        "max_vorticity_divergence_error",
        "max_vortex_strength_misalignment_degrees",
    )
    for family in FAMILIES:
        reference_state = initial_state(f"{family}_dns")
        for variant in VARIANTS[1:]:
            if not np.allclose(
                initial_state(f"{family}_{variant}"),
                reference_state,
                rtol=1.0e-12,
                atol=1.0e-12,
            ):
                failures.append(f"{family}: model variants do not share one initial state")
                break
        dns = metrics[f"{family}_dns"]
        les = metrics[f"{family}_les"]
        if les["completed_fraction"] < 1.05 * dns["completed_fraction"]:
            failures.append(f"{family}: LES does not outlive DNS by at least 5%")
        dns_common = stability_indicators(f"{family}_dns", dns["end_time"])
        les_at_dns_end = stability_indicators(f"{family}_les", dns["end_time"])
        les_common = stability_indicators(f"{family}_les", les["end_time"])
        stabilized_at_les_end = stability_indicators(f"{family}_les_stabilized", les["end_time"])
        les_improvements = [les_at_dns_end[name] <= 0.95 * dns_common[name] for name in indicators]
        stabilized_improvements = [
            stabilized_at_les_end[name] <= 0.90 * les_common[name] for name in indicators
        ]
        if sum(les_improvements) < 2:
            failures.append(
                f"{family}: LES does not improve at least two stability indicators "
                "over the DNS lifetime"
            )
        if sum(stabilized_improvements) < 2:
            failures.append(
                f"{family}: stabilization does not improve at least two indicators by 10% "
                "over the plain-LES lifetime"
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
            f"E/E0={item['total_kinetic_energy_ratio']:.3f}, budget={item['kinetic_energy_budget_error']:.2%}, "
            f"rate RMS={item['modeled_rate_mismatch']:.2%}"
        )
        print(
            "  "
            f"{'':28s} drift(G,I,A)=({item['net_vortex_strength_drift_relative']:.2e}, "
            f"{item['linear_impulse_drift_relative']:.2e}, {item['angular_impulse_drift_relative']:.2e}), "
            f"max|vortex_strength|/initial={item['max_vortex_strength_magnitude_growth']:.2f}, "
            f"max div(omega)={item['max_vorticity_divergence_error']:.3f}, "
            f"max angle={item['max_vortex_strength_misalignment_degrees']:.1f} deg, "
            f"samples/frames={int(item['diagnostic_samples'])}/"
            f"{int(item['state_snapshots'])}"
        )
        if name.endswith("_les_stabilized"):
            print(
                "  "
                f"{'':28s} filter events={int(item['n_regularization_events'])}, "
                f"adaptive cores={int(item['n_regularization_adaptive_core_events'])}, "
                f"filter total_kinetic_energy={-item['regularization_filter_decay']:.2%} E0, "
                f"max regenerated core={item['regularization_max_core_radius']:.3f}"
            )


if __name__ == "__main__":
    main()
