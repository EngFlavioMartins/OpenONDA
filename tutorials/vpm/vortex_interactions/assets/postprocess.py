#!/usr/bin/env python3
"""Postprocess vortex-interaction runs: pre-plot validation and physics gate.

Usage:
    python postprocess.py                    # full post-run physics validation
    python postprocess.py --pre-plot         # lightweight pre-plot completeness check
    python postprocess.py --pre-plot --allow-partial   # accept partial result sets
    python postprocess.py --cases leapfrog_dns collide_dns  # validate specific cases
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys

import numpy as np

ASSETS_DIR = Path(__file__).resolve().parent
CASE_DIR = ASSETS_DIR.parent
SOLUTION_DIR = CASE_DIR / "solution"
SAMPLES_DIR = CASE_DIR / "samples"

FAMILIES = ("leapfrog", "collide")
VARIANTS = ("dns", "les", "les_stabilized")
EXPECTED_CASES = tuple(f"{family}_{variant}" for family in FAMILIES for variant in VARIANTS)

RING_INVARIANT_SCALE = 2.0 * np.pi**2
EXPECTED_SMAGORINSKY = {"leapfrog": 0.20, "collide": 0.20}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    try:
        return np.asarray([float(row[name]) for row in rows])
    except KeyError as error:
        raise ValueError(f"missing diagnostic column {name!r}") from error


def _vectors(rows: list[dict[str, str]], prefix: str) -> np.ndarray:
    return np.column_stack([_column(rows, f"{prefix}_{axis}") for axis in "xyz"])


def _relative_vector_drift(values: np.ndarray, scale: float) -> float:
    return float(np.linalg.norm(values - values[0], axis=1).max() / scale)


def _discover(solution_dir: Path, *, allow_partial: bool) -> list[Path]:
    if not solution_dir.is_dir():
        return []
    cases = []
    for case_dir in sorted(solution_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        if (case_dir / "run_manifest.json").exists():
            cases.append(case_dir)
    if not allow_partial:
        missing = [name for name in EXPECTED_CASES if name not in {c.name for c in cases}]
        if missing:
            raise SystemExit(f"missing cases: {', '.join(missing)}")
    return cases


# ---------------------------------------------------------------------------
# pre-plot validation (lightweight completeness check)
# ---------------------------------------------------------------------------


def _validate_pre_plot(solution_dir: Path, *, allow_partial: bool) -> list[str]:
    failures: list[str] = []
    cases = _discover(solution_dir, allow_partial=allow_partial)
    if not cases:
        return [f"no recognized cases found under {solution_dir}"]

    for case_dir in cases:
        name = case_dir.name
        manifest_path = case_dir / "run_manifest.json"
        if not manifest_path.exists():
            failures.append(f"{name}: missing run_manifest.json")
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            failures.append(f"{name}: unreadable manifest ({error})")
            continue

        status = manifest.get("status")
        stabilized = name.endswith("_les_stabilized")
        valid_status = status == "completed" or (not stabilized and status == "resolution_lost")
        if not valid_status:
            failures.append(f"{name}: invalid run status {status!r}")

        samples_dir = SAMPLES_DIR / name
        integrals = samples_dir / "flow_integrals.csv"
        if not integrals.is_file() or len(integrals.read_text().splitlines()) < 2:
            failures.append(f"{name}: fewer than two flow-integral samples")

        ring_csv = samples_dir / "ring_diagnostics.csv"
        if not ring_csv.is_file():
            failures.append(f"{name}: missing grouped ring diagnostics")

        numbered_steps = {
            int(m.group(1))
            for p in case_dir.glob("vpm_*_*.h5")
            if (m := re.search(r"_(\d{6})\.h5$", p.name))
        }
        if isinstance(manifest.get("completed_steps"), int):
            ci = int(manifest.get("checkpoint_interval_steps", 0))
            if ci > 0:
                expected = set(range(ci, manifest["completed_steps"] + 1, ci))
                if expected - numbered_steps:
                    failures.append(f"{name}: missing scheduled state snapshots")

        if not all((case_dir / f"vpm_{name}_final{s}").is_file() for s in (".h5", ".xdmf")):
            failures.append(f"{name}: final VPM state or XDMF descriptor is missing")

    return failures


# ---------------------------------------------------------------------------
# stability indicators (used by comparative gate)
# ---------------------------------------------------------------------------


def _stability_indicators(case_name: str, through_time: float) -> dict[str, float]:
    csv_path = SAMPLES_DIR / case_name / "flow_integrals.csv"
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    time = _column(rows, "time")
    selected = time <= through_time + 32.0 * np.finfo(float).eps * max(1.0, through_time)
    if not np.any(selected):
        raise ValueError(f"{case_name} has no diagnostic sample through t={through_time:.6g}")
    vortex_strength_magnitude_sum = _column(rows, "vortex_strength_magnitude_sum")[selected]
    return {
        "vortex_strength_magnitude_sum_growth": float(
            np.max(vortex_strength_magnitude_sum) / vortex_strength_magnitude_sum[0]
        ),
        "max_vorticity_divergence_error": float(
            np.max(_column(rows, "vorticity_divergence_error")[selected])
        ),
        "max_vortex_strength_misalignment_degrees": float(
            np.max(_column(rows, "vortex_strength_misalignment_degrees")[selected])
        ),
    }


def _initial_state(case_name: str) -> np.ndarray:
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


# ---------------------------------------------------------------------------
# physics gate (post-run validation)
# ---------------------------------------------------------------------------


def _validate_physics(cases: list[Path]) -> tuple[list[str], dict[str, dict[str, float]]]:
    failures: list[str] = []
    metrics_all: dict[str, dict[str, float]] = {}

    for case_dir in cases:
        name = case_dir.name
        variant = name.removeprefix("leapfrog_").removeprefix("collide_")
        family = name.split("_", maxsplit=1)[0]

        manifest_path = case_dir / "run_manifest.json"
        csv_path = SAMPLES_DIR / name / "flow_integrals.csv"
        ring_csv_path = SAMPLES_DIR / name / "ring_diagnostics.csv"
        if not all(p.is_file() for p in (manifest_path, csv_path, ring_csv_path)):
            failures.append(f"{name}: run manifest or diagnostic CSV missing")
            continue

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            failures.append(f"{name}: unreadable manifest ({error})")
            continue

        status = manifest.get("status")
        completed_steps = manifest.get("completed_steps")
        requested_steps = manifest.get("requested_steps")
        diagnostic_interval_steps = manifest.get("diagnostic_interval_steps")
        checkpoint_interval_steps = manifest.get("checkpoint_interval_steps")

        if (
            not isinstance(diagnostic_interval_steps, int)
            or not 1 <= diagnostic_interval_steps <= 5
        ):
            failures.append(f"{name}: diagnostic interval out of range")
            continue
        if (
            not isinstance(checkpoint_interval_steps, int)
            or not 1 <= checkpoint_interval_steps <= 50
        ):
            failures.append(f"{name}: checkpoint interval out of range")
            continue

        with csv_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) < 3:
            failures.append(f"{name}: fewer than three diagnostic samples")
            continue

        time = _column(rows, "time")
        step = _column(rows, "step")
        total_kinetic_energy = _column(rows, "total_kinetic_energy")
        kinetic_energy_rate = _column(rows, "kinetic_energy_rate")
        viscous_kinetic_energy_rate = _column(rows, "viscous_kinetic_energy_rate")
        total_enstrophy = _column(rows, "total_enstrophy")
        test_filtered_enstrophy = _column(rows, "test_filtered_enstrophy")
        vortex_strength_magnitude_sum = _column(rows, "vortex_strength_magnitude_sum")
        max_vortex_strength_magnitude = _column(rows, "max_vortex_strength_magnitude")
        net_vortex_strength = _vectors(rows, "net_vortex_strength")
        linear_impulse = _vectors(rows, "linear_impulse")
        angular_impulse = _vectors(rows, "angular_impulse")
        n_particles_total = _column(rows, "n_particles_total")
        mean_core_radius = _column(rows, "mean_core_radius")
        max_overlap_ratio = _column(rows, "max_overlap_ratio")
        divergence = _column(rows, "vorticity_divergence_error")
        misalignment = _column(rows, "vortex_strength_misalignment_degrees")
        mean_eddy_viscosity = _column(rows, "mean_eddy_viscosity")
        max_eddy_viscosity = _column(rows, "max_eddy_viscosity")
        stabilization_kinematic_viscosity = _column(rows, "max_stabilization_kinematic_viscosity")
        projection_correction = _column(rows, "invariant_projection_correction_ratio")
        core_spreading_events = _column(rows, "core_spreading_moment_projection_events")
        core_spreading_correction = _column(rows, "core_spreading_max_moment_correction_relative")
        core_spreading_vortex_strength_error_relative = _column(
            rows, "core_spreading_max_vortex_strength_error_relative"
        )
        core_spreading_impulse = _column(rows, "core_spreading_max_linear_impulse_error_relative")
        core_spreading_angular = _column(rows, "core_spreading_max_angular_impulse_error_relative")
        n_regularization_events = _column(rows, "n_regularization_events")
        regularization_transfer = _column(
            rows, "regularization_cumulative_total_kinetic_energy_transfer"
        )
        regularization_total_kinetic_energy_dissipation = _column(
            rows, "regularization_max_total_kinetic_energy_dissipation"
        )
        regularization_total_enstrophy_injection = _column(
            rows, "regularization_max_total_enstrophy_injection"
        )
        regularization_total_enstrophy_dissipation = _column(
            rows, "regularization_max_total_enstrophy_dissipation"
        )
        regularization_correction = _column(rows, "regularization_max_correction_relative")
        regularization_projection_correction = _column(
            rows, "regularization_max_projection_correction_relative"
        )
        regularization_vortex_strength_error_relative = _column(
            rows, "regularization_max_vortex_strength_error_relative"
        )
        regularization_impulse = _column(rows, "regularization_max_linear_impulse_error_relative")
        regularization_angular = _column(rows, "regularization_max_angular_impulse_error_relative")
        n_stabilization_events = _column(rows, "n_stabilization_events")
        max_stabilization_kinematic_viscosity = _column(rows, "max_stabilization_kinetic_viscosity")

        # --- basic sanity ---
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
                projection_correction,
                core_spreading_events,
                core_spreading_correction,
                n_regularization_events,
                regularization_transfer,
                regularization_correction,
                n_stabilization_events,
            )
        )
        if not np.isfinite(required_values).all():
            failures.append(f"{name}: non-finite required diagnostic")
            continue

        expected_steps = np.arange(0, completed_steps + 1, diagnostic_interval_steps, dtype=float)
        if expected_steps[-1] != completed_steps:
            expected_steps = np.append(expected_steps, completed_steps)
        if not np.array_equal(step, expected_steps):
            failures.append(f"{name}: diagnostic samples at wrong steps")
            continue

        expected_time_step_size = (
            float(manifest.get("requested_end_time", time[-1])) / requested_steps
        )
        if not np.allclose(time, step * expected_time_step_size, rtol=0.0, atol=1.0e-9):
            failures.append(f"{name}: diagnostic time mismatch")
            continue
        if np.any(np.diff(time) <= 0.0):
            failures.append(f"{name}: diagnostic time not strictly increasing")
            continue

        # --- ring sampler ---
        with ring_csv_path.open(newline="") as stream:
            ring_rows = list(csv.DictReader(stream))
        try:
            ring_values = np.asarray(
                [[float(v) for v in row.values()] for row in ring_rows], dtype=float
            )
        except (TypeError, ValueError) as error:
            failures.append(f"{name}: non-numeric ring sampler value ({error})")
            continue
        if not np.isfinite(ring_values).all():
            failures.append(f"{name}: non-finite ring sampler value")
            continue
        if (
            np.any(_column(ring_rows, "major_radius") <= 0.0)
            or np.any(_column(ring_rows, "tube_circulation") <= 0.0)
            or np.any(_column(ring_rows, "vortex_strength_magnitude_sum") <= 0.0)
        ):
            failures.append(f"{name}: nonphysical ring sampler values")
            continue

        ring_steps = np.asarray([int(r["step"]) for r in ring_rows])
        ring_groups = np.asarray([int(r["group_id"]) for r in ring_rows])
        expected_ring_steps = np.repeat(expected_steps.astype(int), 2)
        expected_ring_groups = np.tile((0, 1), len(expected_steps))
        if not np.array_equal(ring_steps, expected_ring_steps) or not np.array_equal(
            ring_groups, expected_ring_groups
        ):
            failures.append(f"{name}: ring diagnostics out of order")
            continue

        ring_time = np.asarray([float(r["time"]) for r in ring_rows])
        if not np.allclose(ring_time, np.repeat(time, 2), rtol=0.0, atol=1.0e-9):
            failures.append(f"{name}: ring-sampler times mismatch")
            continue

        # --- snapshot check ---
        numbered_snapshots = {
            int(m.group(1))
            for p in case_dir.glob(f"vpm_{name}_*.h5")
            if (m := re.search(r"_(\d{6})\.h5$", p.name))
        }
        expected_snapshots = set(
            range(checkpoint_interval_steps, completed_steps + 1, checkpoint_interval_steps)
        )
        if numbered_snapshots != expected_snapshots:
            failures.append(f"{name}: scheduled snapshots missing or duplicated")
            continue
        if any(not (case_dir / f"vpm_{name}_{s:06d}.xdmf").is_file() for s in expected_snapshots):
            failures.append(f"{name}: snapshot missing XDMF descriptor")
            continue
        if not all((case_dir / f"vpm_{name}_final{s}").is_file() for s in (".h5", ".xdmf")):
            failures.append(f"{name}: final state missing")
            continue

        # --- physical sign checks ---
        if (
            np.any(total_kinetic_energy <= 0.0)
            or np.any(total_enstrophy < 0.0)
            or np.any(test_filtered_enstrophy < 0.0)
            or np.any(viscous_kinetic_energy_rate > 1.0e-8)
        ):
            failures.append(f"{name}: nonphysical energy/enstrophy sign")
            continue
        if (
            np.any(n_particles_total <= 0.0)
            or np.any(n_particles_total != np.floor(n_particles_total))
            or np.any(vortex_strength_magnitude_sum <= 0.0)
            or np.any(max_vortex_strength_magnitude <= 0.0)
            or np.any(mean_core_radius <= 0.0)
        ):
            failures.append(f"{name}: nonphysical particle/field values")
            continue
        nonnegative = np.column_stack(
            (
                mean_eddy_viscosity,
                max_eddy_viscosity,
                _column(rows, "mean_effective_viscosity"),
                _column(rows, "max_effective_viscosity"),
                _column(rows, "mean_overlap_ratio"),
                max_overlap_ratio,
                divergence,
            )
        )
        if np.any(nonnegative < -1.0e-14) or np.any((misalignment < 0.0) | (misalignment > 180.0)):
            failures.append(f"{name}: nonphysical viscosity/resolution/alignment values")
            continue
        if variant != "les_stabilized" and np.ptp(n_particles_total) != 0.0:
            failures.append(f"{name}: fixed-particle baseline changed population")
            continue

        # --- energy budget ---
        interval_rate = np.diff(total_kinetic_energy) / np.diff(time)
        rate_scale = max(
            float(np.max(np.abs(interval_rate))),
            total_kinetic_energy[0] / (time[-1] - time[0]),
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
            failures.append(f"{name}: conservative regularization injected energy")
            continue
        filter_decay = -float(regularization_transfer[-1] - regularization_transfer[0])
        modeled_decay = -float(np.trapezoid(viscous_kinetic_energy_rate, time)) + filter_decay
        observed_decay = float(total_kinetic_energy[0] - total_kinetic_energy[-1])
        kinetic_energy_budget_error = abs(observed_decay - modeled_decay) / total_kinetic_energy[0]

        # --- build metrics ---
        completed_fraction = float(completed_steps / requested_steps) if requested_steps else 0.0
        metrics = {
            "end_time": float(time[-1]),
            "completed_fraction": completed_fraction,
            "total_kinetic_energy_ratio": float(total_kinetic_energy[-1] / total_kinetic_energy[0]),
            "total_kinetic_energy_injection": total_kinetic_energy_injection,
            "positive_kinetic_energy_rate": positive_kinetic_energy_rate,
            "rate_mismatch": rate_mismatch,
            "modeled_rate_mismatch": modeled_rate_mismatch,
            "kinetic_energy_budget_error": kinetic_energy_budget_error,
            "net_vortex_strength_drift_relative": _relative_vector_drift(
                net_vortex_strength, vortex_strength_magnitude_sum[0]
            ),
            "linear_impulse_drift_relative": _relative_vector_drift(
                linear_impulse, RING_INVARIANT_SCALE
            ),
            "angular_impulse_drift_relative": _relative_vector_drift(
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
            "n_regularization_events": float(np.max(n_regularization_events)),
            "n_stabilization_events": float(np.max(n_stabilization_events)),
            "max_particle_count": float(np.max(n_particles_total)),
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
            "regularization_projection_correction": float(
                np.max(regularization_projection_correction)
            ),
            "regularization_vortex_strength_error_relative": float(
                np.max(regularization_vortex_strength_error_relative)
            ),
            "regularization_impulse": float(np.max(regularization_impulse)),
            "regularization_angular": float(np.max(regularization_angular)),
            "diagnostic_samples": float(len(rows)),
            "state_snapshots": float(len(numbered_snapshots)),
        }
        metrics_all[name] = metrics

        # --- limit checks ---
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
            "max_vortex_strength_misalignment_degrees": 55.0
            if variant != "les_stabilized"
            else 45.0,
            "projection_correction": 0.25,
            "core_spreading_correction": 1.0e-3,
            "core_spreading_vortex_strength_error_relative": 1.0e-10,
            "core_spreading_impulse": 1.0e-10,
            "core_spreading_angular": 1.0e-10,
            "regularization_total_enstrophy_injection": 5.0e-6,
            "regularization_total_kinetic_energy_dissipation": 0.20,
            "regularization_total_enstrophy_dissipation": 0.15,
            "regularization_correction": 0.5,
            "regularization_projection_correction": (0.051 if family == "leapfrog" else 0.101),
            "regularization_vortex_strength_error_relative": 1.0e-5,
            "regularization_impulse": 1.0e-5,
            "regularization_angular": 1.0e-5,
        }
        for metric_name, limit in limits.items():
            if metrics[metric_name] > limit:
                failures.append(f"{name}: {metric_name}={metrics[metric_name]:.3g} > {limit:.3g}")

        # --- variant-specific physics checks ---
        viscosity_epsilon = 1.0e-12
        if metrics["core_spreading_events"] <= 0.0:
            failures.append(f"{name}: core spreading never applied correction")
        if variant == "dns":
            if metrics["max_eddy_viscosity"] > viscosity_epsilon:
                failures.append(f"{name}: DNS has nonzero turbulent viscosity")
            if metrics["n_regularization_events"] > 0.0:
                failures.append(f"{name}: DNS performed regularization")
            if metrics["n_stabilization_events"] > 0.0:
                failures.append(f"{name}: DNS performed stabilization")
        elif variant == "les":
            if metrics["max_eddy_viscosity"] <= viscosity_epsilon:
                failures.append(f"{name}: LES never activated eddy viscosity")
            if metrics["n_regularization_events"] > 0.0:
                failures.append(f"{name}: plain LES performed regularization")
            if metrics["n_stabilization_events"] > 0.0:
                failures.append(f"{name}: plain LES performed stabilization")
        else:
            if metrics["max_eddy_viscosity"] <= viscosity_epsilon:
                failures.append(f"{name}: stabilized LES never activated eddy viscosity")
            if metrics["n_regularization_events"] > 0.0:
                failures.append(f"{name}: stabilized LES unexpectedly remeshed")
            if metrics["n_stabilization_events"] <= 0.0:
                failures.append(f"{name}: stabilized LES never encountered strength overshoot")
            if metrics["max_particle_count"] <= n_particles_total[0]:
                failures.append(f"{name}: stabilized LES recorded no filament splits")

    # --- comparative stability gate (only when all 6 cases present) ---
    if not failures and len(metrics_all) == len(EXPECTED_CASES):
        indicators = (
            "vortex_strength_magnitude_sum_growth",
            "max_vorticity_divergence_error",
            "max_vortex_strength_misalignment_degrees",
        )
        for family in FAMILIES:
            dns_name = f"{family}_dns"
            les_name = f"{family}_les"
            stab_name = f"{family}_les_stabilized"
            ref_state = _initial_state(dns_name)
            for variant in VARIANTS[1:]:
                case_initial = _initial_state(f"{family}_{variant}")
                if not np.allclose(case_initial, ref_state, rtol=1.0e-12, atol=1.0e-12):
                    failures.append(f"{family}: model variants do not share one initial state")
                    break
            dns = metrics_all[dns_name]
            les = metrics_all[les_name]
            if les["completed_fraction"] < 1.05 * dns["completed_fraction"]:
                failures.append(f"{family}: LES does not outlive DNS by at least 5%")
            dns_common = _stability_indicators(dns_name, dns["end_time"])
            les_at_dns_end = _stability_indicators(les_name, dns["end_time"])
            les_common = _stability_indicators(les_name, les["end_time"])
            stab_at_les_end = _stability_indicators(stab_name, les["end_time"])
            les_improvements = [les_at_dns_end[k] <= 0.95 * dns_common[k] for k in indicators]
            stab_improvements = [stab_at_les_end[k] <= 0.90 * les_common[k] for k in indicators]
            if sum(les_improvements) < 2:
                failures.append(
                    f"{family}: LES does not improve at least two stability indicators "
                    "over the DNS lifetime"
                )
            if sum(stab_improvements) < 2:
                failures.append(
                    f"{family}: stabilization does not improve at least two indicators "
                    "by 10% over the plain-LES lifetime"
                )

    return failures, metrics_all


# ---------------------------------------------------------------------------
# manifest display (--manifest)
# ---------------------------------------------------------------------------


def _show_manifest(solution_dir: Path) -> None:
    for case_dir in sorted(solution_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        manifest_path = case_dir / "run_manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        status = manifest.get("status", "unknown")
        requested = manifest.get("requested_steps", "?")
        completed = manifest.get("completed_steps", "?")
        model = manifest.get("model", "?")
        print(
            f"  {case_dir.name:28s} status={status}, model={model}, steps={completed}/{requested}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-plot", action="store_true", help="lightweight pre-plot validation")
    parser.add_argument("--allow-partial", action="store_true", help="accept partial result sets")
    parser.add_argument("--manifest", action="store_true", help="display manifest summaries")
    parser.add_argument("--solution-dir", default=str(SOLUTION_DIR))
    parser.add_argument("--cases", nargs="*", help="validate specific cases (post-run only)")
    args = parser.parse_args()

    solution_dir = Path(args.solution_dir)

    if args.manifest:
        _show_manifest(solution_dir)
        return

    if args.pre_plot:
        failures = _validate_pre_plot(solution_dir, allow_partial=args.allow_partial)
        if failures:
            print("Pre-plot validation failed:")
            for f in failures:
                print(f"  - {f}")
            raise SystemExit(1)
        count = len(_discover(solution_dir, allow_partial=args.allow_partial))
        scope = "partial result set" if args.allow_partial else "complete tutorial run"
        print(f"Plot inputs validated: {count} cases ({scope}).")
        return

    # full post-run physics validation
    if args.cases:
        cases = [solution_dir / c for c in args.cases]
        missing = [str(c) for c in cases if not c.is_dir()]
        if missing:
            raise SystemExit(f"missing solution dirs: {', '.join(missing)}")
    else:
        cases = [solution_dir / c for c in EXPECTED_CASES if (solution_dir / c).is_dir()]
        if len(cases) < len(EXPECTED_CASES):
            missing = [c for c in EXPECTED_CASES if (solution_dir / c).is_dir()]
            raise SystemExit(f"expected {len(EXPECTED_CASES)} cases, found {len(cases)}")

    failures, metrics_all = _validate_physics(cases)

    if failures:
        raise SystemExit("FAIL:\n  - " + "\n  - ".join(failures))

    scope = (
        f"all {len(cases)} cases"
        if len(cases) == len(EXPECTED_CASES)
        else f"{len(cases)} selected case(s)"
    )
    print(f"PASS: {scope} ended with physical budgets and invariant conservation.")
    for name in sorted(metrics_all):
        m = metrics_all[name]
        print(
            f"  {name:28s} duration={m['completed_fraction']:.1%}, "
            f"E/E0={m['total_kinetic_energy_ratio']:.3f}, "
            f"budget={m['kinetic_energy_budget_error']:.2%}, "
            f"rate RMS={m['modeled_rate_mismatch']:.2%}"
        )
        print(
            "  "
            f"{'':28s} drift(G,I,A)=({m['net_vortex_strength_drift_relative']:.2e}, "
            f"{m['linear_impulse_drift_relative']:.2e}, "
            f"{m['angular_impulse_drift_relative']:.2e}), "
            f"max|Γ|/initial={m['max_vortex_strength_magnitude_growth']:.2f}, "
            f"max div(ω)={m['max_vorticity_divergence_error']:.3f}, "
            f"max angle={m['max_vortex_strength_misalignment_degrees']:.1f} deg, "
            f"samples/frames={int(m['diagnostic_samples'])}/"
            f"{int(m['state_snapshots'])}"
        )
        if name.endswith("_les_stabilized"):
            print(
                "  "
                f"{'':28s} splitting events={int(m['n_stabilization_events'])}, "
                f"max particles={int(m['max_particle_count'])}, "
                f"remeshing events={int(m['n_regularization_events'])}"
            )


if __name__ == "__main__":
    main()
