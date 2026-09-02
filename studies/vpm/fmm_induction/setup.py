#!/usr/bin/env python3
"""Process-isolated production FMM qualification runner.

Usage is intentionally positional and small::

    python setup.py init
    python setup.py accuracy GAUSSIAN CPU 512 uniform
    python setup.py scaling FMM VULKAN 14080 elongated
    python setup.py comparison VULKAN 14080 10 leapfrog
    python setup.py evolution VULKAN 512 200 two_rings
    python setup.py plot
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np
import taichi as ti

from source.solvers.vpm.kernels.base import make_vortex_kernel
from source.solvers.vpm.numerics.rk_tableaux import SSPRK3
from source.solvers.vpm.numerics.runge_kutta import RungeKutta
from source.solvers.vpm.physics.base import PhysicsBase
from source.solvers.vpm.physics.induction.direct import DirectInduction
from source.solvers.vpm.physics.induction.fmm import FMMInduction
from source.solvers.vpm.physics.induction.treecode import TreecodeInduction
from source.solvers.vpm.physics.stage_rhs import StageRHS

STUDY_DIR = Path(__file__).resolve().parent
RESULTS_DIR = STUDY_DIR / "results"
FIGURES_DIR = STUDY_DIR / "figures"
SEED = 20260902
REFERENCE_TARGETS = 128
REFERENCE_SOURCE_CHUNK = 512
KERNELS = ("GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS")
DISTRIBUTIONS = (
    "uniform",
    "clustered",
    "elongated",
    "ring",
    "two_rings",
    "leapfrog",
    "rotor",
)


def _revision() -> tuple[str, bool]:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=STUDY_DIR,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=STUDY_DIR,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    return sha, dirty


def _hardware_model(backend_name: str) -> str:
    if backend_name == "CPU":
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
        return platform.processor() or platform.machine()
    try:
        output = subprocess.run(
            ["lspci"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout
        displays = [
            line.split(": ", 1)[-1]
            for line in output.splitlines()
            if "VGA compatible controller" in line or "Display controller" in line
        ]
        if displays:
            return "; ".join(displays)
    except OSError:
        pass
    return f"{platform.machine()} {backend_name} device"


def initialize_results() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sha, dirty = _revision()
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "commit": sha,
        "dirty": dirty,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "taichi": ti.__version__,
        "platform": platform.platform(),
        "seed": SEED,
        "kernels": KERNELS,
        "distributions": DISTRIBUTIONS,
        "accuracy_gates": {
            "velocity_relative_l2": 5.0e-3,
            "gradient_relative_l2": 1.0e-2,
            "rate_relative_l2": 1.5e-2,
            "rate_particle_p95": 3.0e-2,
            "raw_rate_defect": 1.0e-3,
        },
    }
    (RESULTS_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def _cloud(name: str, count: int):
    if name not in DISTRIBUTIONS:
        raise ValueError(f"unknown distribution {name!r}")
    rng = np.random.default_rng(SEED + count + DISTRIBUTIONS.index(name))
    strength = rng.normal(scale=0.01, size=(count, 3))
    radius = rng.uniform(0.008, 0.016, size=count)
    if name == "uniform":
        position = rng.uniform(-1.0, 1.0, size=(count, 3))
    elif name == "clustered":
        centres = np.array([[-0.6, -0.6, 0.0], [0.6, -0.6, 0.0], [-0.6, 0.6, 0.0], [0.6, 0.6, 0.0]])
        position = centres[np.arange(count) % len(centres)] + rng.normal(
            scale=0.08, size=(count, 3)
        )
    elif name == "elongated":
        position = np.column_stack(
            (
                rng.uniform(-8.0, 8.0, count),
                rng.normal(scale=0.12, size=count),
                rng.normal(scale=0.12, size=count),
            )
        )
    else:
        theta = 2.0 * np.pi * (np.arange(count) + 0.25) / count
        ring_radius = 1.0
        centre_x = np.zeros(count)
        if name in {"two_rings", "leapfrog"}:
            split = count // 2
            centre_x[:split] = -0.45
            centre_x[split:] = 0.45
            theta[split:] = 2.0 * np.pi * (np.arange(count - split) + 0.25) / (count - split)
            if name == "leapfrog":
                ring_radius = np.where(np.arange(count) < split, 0.8, 1.2)
        if name == "rotor":
            axial = np.linspace(0.0, 12.0, count)
            theta = 5.0 * axial
            centre_x = axial
        position = np.column_stack(
            (
                centre_x + rng.normal(scale=0.01, size=count),
                ring_radius * np.cos(theta) + rng.normal(scale=0.01, size=count),
                ring_radius * np.sin(theta) + rng.normal(scale=0.01, size=count),
            )
        )
        strength = np.column_stack(
            (
                np.zeros(count),
                -0.01 * np.sin(theta),
                0.01 * np.cos(theta),
            )
        )
    return (
        np.asarray(position, dtype=np.float32),
        np.asarray(strength, dtype=np.float32),
        np.asarray(radius, dtype=np.float32),
    )


def _exact_subset(position, strength, radius, kernel_name: str, target_indices):
    kernel = make_vortex_kernel(kernel_name)
    target_position = position[target_indices]
    target_radius = radius[target_indices]
    velocity = np.zeros((len(target_indices), 3), dtype=np.float64)
    gradient = np.zeros((len(target_indices), 3, 3), dtype=np.float64)
    for start in range(0, len(position), REFERENCE_SOURCE_CHUNK):
        stop = min(start + REFERENCE_SOURCE_CHUNK, len(position))
        displacement = target_position[:, None, :] - position[None, start:stop, :]
        velocity += kernel.velocity_pair(
            displacement,
            strength[None, start:stop, :],
            target_radius[:, None],
            radius[None, start:stop],
        ).sum(axis=1)
        gradient += kernel.gradient_pair(
            displacement,
            strength[None, start:stop, :],
            target_radius[:, None],
            radius[None, start:stop],
        ).sum(axis=1)
    rate = np.einsum("nji,nj->ni", gradient, strength[target_indices])
    return velocity, gradient, rate


def _relative(actual, reference) -> float:
    return float(np.linalg.norm(actual - reference) / max(np.linalg.norm(reference), 1.0e-30))


def _append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(row), lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _upsert_csv(path: Path, row: dict, *, key_fields: tuple[str, ...]) -> None:
    """Replace one deterministic study row without duplicating rerun evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as stream:
            existing = list(csv.DictReader(stream))
    key = tuple(str(row[field]) for field in key_fields)
    retained = [
        candidate
        for candidate in existing
        if tuple(candidate.get(field, "") for field in key_fields) != key
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(row), lineterminator="\n")
        writer.writeheader()
        writer.writerows(retained)
        writer.writerow(row)


def run_case(method: str, backend_name: str, count: int, distribution: str, kernel_name: str):
    method = method.upper()
    if method in {"TREE", "TREECODE"}:
        method = "TREECODE"
    elif method != "FMM":
        raise ValueError(f"unknown induction method {method!r}")
    backend_name = backend_name.upper()
    kernel_name = kernel_name.upper()
    if kernel_name not in KERNELS:
        raise ValueError(f"unknown kernel {kernel_name!r}")
    if method == "TREECODE" and kernel_name not in TreecodeInduction.supported_kernels:
        raise ValueError(f"TreecodeInduction does not support {kernel_name}")
    position_np, strength_np, radius_np = _cloud(distribution, count)
    backend = ti.vulkan if backend_name == "VULKAN" else ti.cpu
    ti.init(
        arch=backend,
        offline_cache=False,
        cpu_max_num_threads=min(4, max(1, os.cpu_count() or 1)),
    )
    physics = PhysicsBase(
        particle_kernel=kernel_name,
        max_n_particles=count,
        accumulator_dtype=ti.f32,
        max_evaluation_points=count,
    )
    position = ti.Vector.field(3, dtype=ti.f32, shape=count)
    strength = ti.Vector.field(3, dtype=ti.f32, shape=count)
    radius = ti.field(dtype=ti.f32, shape=count)
    velocity = ti.Vector.field(3, dtype=ti.f32, shape=count)
    gradient = ti.Matrix.field(3, 3, dtype=ti.f32, shape=count)
    rate = ti.Vector.field(3, dtype=ti.f32, shape=count)
    position.from_numpy(position_np)
    strength.from_numpy(strength_np)
    radius.from_numpy(radius_np)
    induction = (FMMInduction() if method == "FMM" else TreecodeInduction()).bind(
        physics,
        kernel=make_vortex_kernel(kernel_name),
    )
    induction.evaluate_stage(
        position=position,
        vortex_strength=strength,
        core_radius=radius,
        count=count,
        velocity_out=velocity,
        vortex_strength_rate_out=rate,
        velocity_gradient_out=gradient,
    )
    ti.sync()
    start = time.perf_counter()
    induction.evaluate_stage(
        position=position,
        vortex_strength=strength,
        core_radius=radius,
        count=count,
        velocity_out=velocity,
        vortex_strength_rate_out=rate,
        velocity_gradient_out=gradient,
    )
    ti.sync()
    stage_seconds = time.perf_counter() - start
    actual_velocity = velocity.to_numpy().astype(np.float64)
    actual_gradient = gradient.to_numpy().astype(np.float64)
    actual_rate = rate.to_numpy().astype(np.float64)
    phase_seconds = {
        "tree_build_seconds": 0.0,
        "upward_pass_seconds": 0.0,
        "interaction_list_seconds": 0.0,
        "m2l_seconds": 0.0,
        "downward_pass_seconds": 0.0,
        "near_field_seconds": 0.0,
        "strength_rate_seconds": 0.0,
    }
    if method == "FMM":
        induction.workspace.profile_passes = True
        induction.evaluate_stage(
            position=position,
            vortex_strength=strength,
            core_radius=radius,
            count=count,
            velocity_out=velocity,
            vortex_strength_rate_out=rate,
            velocity_gradient_out=gradient,
        )
        induction.workspace.profile_passes = False
        phase_seconds = {
            "tree_build_seconds": induction.diagnostics.last_tree_build_seconds,
            "upward_pass_seconds": induction.diagnostics.last_upward_pass_seconds,
            "interaction_list_seconds": induction.diagnostics.last_interaction_list_seconds,
            "m2l_seconds": induction.diagnostics.last_m2l_seconds,
            "downward_pass_seconds": induction.diagnostics.last_downward_pass_seconds,
            "near_field_seconds": induction.diagnostics.last_near_field_seconds,
            "strength_rate_seconds": induction.diagnostics.last_strength_rate_seconds,
        }
    target_count = min(REFERENCE_TARGETS, count)
    target_indices = np.linspace(0, count - 1, target_count, dtype=int)
    reference_velocity, reference_gradient, reference_rate = _exact_subset(
        position_np,
        strength_np,
        radius_np,
        kernel_name,
        target_indices,
    )
    actual_subset_rate = actual_rate[target_indices]
    reference_norm = np.linalg.norm(reference_rate, axis=1)
    denominator_floor = max(float(np.sqrt(np.mean(reference_norm**2))) * 1.0e-3, 1.0e-14)
    particle_rate_error = np.linalg.norm(actual_subset_rate - reference_rate, axis=1) / np.maximum(
        reference_norm,
        denominator_floor,
    )
    diagnostics = induction.diagnostics
    m2l = int(induction.workspace._m2l_count[None]) if method == "FMM" else 0
    near_node_pairs = int(induction.workspace._near_count[None]) if method == "FMM" else 0
    p2p_particle_pairs = (
        int(induction.workspace._p2p_particle_count[None]) if method == "FMM" else 0
    )
    raw_defect = (
        diagnostics.last_relative_rate_defect
        if method == "FMM"
        else float(
            np.linalg.norm(actual_rate.sum(axis=0))
            / max(np.linalg.norm(actual_rate, axis=1).sum(), np.finfo(float).eps)
        )
    )
    sha, dirty = _revision()
    integrator = RungeKutta(SSPRK3(), max_n_particles=count, dtype=ti.f32)
    right_hand_side = StageRHS(induction)
    integrator.advance(
        position=position,
        vortex_strength=strength,
        core_radius=radius,
        count=count,
        time=0.0,
        time_step_size=1.0e-4,
        right_hand_side=right_hand_side,
    )
    ti.sync()
    position.from_numpy(position_np)
    strength.from_numpy(strength_np)
    radius.from_numpy(radius_np)
    ti.sync()
    accepted_step_start = time.perf_counter()
    integrator.advance(
        position=position,
        vortex_strength=strength,
        core_radius=radius,
        count=count,
        time=0.0,
        time_step_size=1.0e-4,
        right_hand_side=right_hand_side,
    )
    ti.sync()
    accepted_step_seconds = time.perf_counter() - accepted_step_start
    row = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "commit": sha,
        "dirty": dirty,
        "method": method,
        "backend": backend_name,
        "hardware_model": _hardware_model(backend_name),
        "precision": "f32",
        "kernel": kernel_name,
        "distribution": distribution,
        "count": count,
        "reference_targets": target_count,
        "velocity_relative_l2": _relative(actual_velocity[target_indices], reference_velocity),
        "gradient_relative_l2": _relative(actual_gradient[target_indices], reference_gradient),
        "rate_relative_l2": _relative(actual_subset_rate, reference_rate),
        "rate_particle_p50": float(np.percentile(particle_rate_error, 50.0)),
        "rate_particle_p95": float(np.percentile(particle_rate_error, 95.0)),
        "rate_particle_max": float(np.max(particle_rate_error)),
        "raw_rate_defect": raw_defect,
        "m2l_pairs": m2l,
        "near_node_pairs": near_node_pairs,
        "p2p_particle_pairs": p2p_particle_pairs,
        "interaction_work": m2l + p2p_particle_pairs,
        **phase_seconds,
        "stage_seconds": stage_seconds,
        "ssprk3_accepted_step_seconds": accepted_step_seconds,
        "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "estimated_device_bytes": (
            diagnostics.device_memory_estimate_bytes if method == "FMM" else 0
        ),
        "host_particle_transfers": diagnostics.host_particle_transfers if method == "FMM" else 0,
    }
    _append_csv(RESULTS_DIR / "accuracy.csv", row)
    _append_csv(RESULTS_DIR / "scaling.csv", row)
    _append_csv(
        RESULTS_DIR / "conservation.csv",
        {
            "timestamp_utc": row["timestamp_utc"],
            "commit": row["commit"],
            "method": method,
            "backend": backend_name,
            "kernel": kernel_name,
            "distribution": distribution,
            "count": count,
            "raw_rate_defect": raw_defect,
        },
    )
    if distribution in {"leapfrog", "rotor", "two_rings"}:
        _append_csv(RESULTS_DIR / "checkpoints.csv", row)
    print(json.dumps(row, indent=2))
    ti.reset()


def _gaussian_kinetic_energy(
    position: np.ndarray,
    strength: np.ndarray,
    core_radius: np.ndarray,
) -> float:
    """Return the direct unbounded Gaussian-blob energy for a bounded cloud."""
    energy = 0.0
    source_position = position.astype(np.float64)
    source_strength = strength.astype(np.float64)
    source_radius = core_radius.astype(np.float64)
    for start in range(0, len(position), REFERENCE_SOURCE_CHUNK):
        stop = min(start + REFERENCE_SOURCE_CHUNK, len(position))
        displacement = source_position[start:stop, None, :] - source_position[None, :, :]
        distance = np.linalg.norm(displacement, axis=2)
        convolved_radius = np.sqrt(
            source_radius[start:stop, None] ** 2 + source_radius[None, :] ** 2
        )
        density = distance / convolved_radius
        scaled_green = np.empty_like(distance)
        at_origin = distance <= np.finfo(float).eps
        scaled_green[at_origin] = 1.0 / (2.0 * math.pi**1.5 * convolved_radius[at_origin])
        nonzero = ~at_origin
        scaled_green[nonzero] = np.vectorize(math.erf, otypes=[float])(density[nonzero]) / (
            4.0 * math.pi * distance[nonzero]
        )
        pair_dot = source_strength[start:stop] @ source_strength.T
        energy += 0.5 * float(np.sum(scaled_green * pair_dot, dtype=np.float64))
    return energy


def _invariants(
    position: np.ndarray,
    strength: np.ndarray,
    core_radius: np.ndarray,
    *,
    include_direct_energy: bool,
) -> dict[str, object]:
    total_strength = strength.astype(np.float64).sum(axis=0)
    linear_impulse = 0.5 * np.cross(position.astype(np.float64), strength).sum(axis=0)
    particle_radius = np.linalg.norm(position.astype(np.float64), axis=1)
    cloud_radius = float(particle_radius.max(initial=0.0))
    cloud_rms_radius = float(np.sqrt(np.mean(particle_radius**2))) if len(position) else 0.0
    return {
        "total_strength": total_strength,
        "linear_impulse": linear_impulse,
        "maximum_strength": float(np.linalg.norm(strength, axis=1).max(initial=0.0)),
        "cloud_radius": cloud_radius,
        "cloud_rms_radius": cloud_rms_radius,
        "cloud_radius_to_rms": cloud_radius / max(cloud_rms_radius, np.finfo(float).eps),
        "total_kinetic_energy": (
            _gaussian_kinetic_energy(position, strength, core_radius)
            if include_direct_energy
            else None
        ),
    }


def run_evolution(backend_name: str, count: int, steps: int, distribution: str) -> None:
    """Run a process-isolated SSPRK3 FMM trajectory and sample exact targets."""
    backend_name = backend_name.upper()
    position_np, strength_np, radius_np = _cloud(distribution, count)
    record_direct_energy = count <= 512
    initial = _invariants(
        position_np,
        strength_np,
        radius_np,
        include_direct_energy=record_direct_energy,
    )
    backend = ti.vulkan if backend_name == "VULKAN" else ti.cpu
    ti.init(
        arch=backend,
        offline_cache=False,
        cpu_max_num_threads=min(4, max(1, os.cpu_count() or 1)),
    )
    physics = PhysicsBase(
        particle_kernel="GAUSSIAN",
        max_n_particles=count,
        accumulator_dtype=ti.f32,
        max_evaluation_points=count,
    )
    position = ti.Vector.field(3, dtype=ti.f32, shape=count)
    strength = ti.Vector.field(3, dtype=ti.f32, shape=count)
    radius = ti.field(dtype=ti.f32, shape=count)
    position.from_numpy(position_np)
    strength.from_numpy(strength_np)
    radius.from_numpy(radius_np)
    induction = FMMInduction().bind(physics, kernel=make_vortex_kernel("GAUSSIAN"))
    integrator = RungeKutta(SSPRK3(), max_n_particles=count, dtype=ti.f32)
    right_hand_side = StageRHS(induction)
    time_step_size = 1.0e-4
    record_interval = max(1, steps // 10)
    records = []
    maximum_rate_defect = 0.0
    integrator.advance(
        position=position,
        vortex_strength=strength,
        core_radius=radius,
        count=count,
        time=0.0,
        time_step_size=time_step_size,
        right_hand_side=right_hand_side,
    )
    ti.sync()
    position.from_numpy(position_np)
    strength.from_numpy(strength_np)
    radius.from_numpy(radius_np)
    ti.sync()
    start = time.perf_counter()
    for step in range(1, steps + 1):
        integrator.advance(
            position=position,
            vortex_strength=strength,
            core_radius=radius,
            count=count,
            time=(step - 1) * time_step_size,
            time_step_size=time_step_size,
            right_hand_side=right_hand_side,
        )
        maximum_rate_defect = max(
            maximum_rate_defect,
            induction.diagnostics.last_relative_rate_defect,
        )
        if step % record_interval == 0 or step == steps:
            ti.sync()
            current_position = position.to_numpy()
            current_strength = strength.to_numpy()
            if not np.all(np.isfinite(current_position)) or not np.all(
                np.isfinite(current_strength)
            ):
                raise RuntimeError(f"non-finite FMM state at accepted step {step}")
            records.append(
                {
                    "step": step,
                    "time": step * time_step_size,
                    **{
                        key: value.tolist() if isinstance(value, np.ndarray) else value
                        for key, value in _invariants(
                            current_position,
                            current_strength,
                            radius_np,
                            include_direct_energy=record_direct_energy,
                        ).items()
                    },
                    "raw_rate_defect": induction.diagnostics.last_relative_rate_defect,
                }
            )
    ti.sync()
    elapsed = time.perf_counter() - start
    final_position = position.to_numpy()
    final_strength = strength.to_numpy()
    final = _invariants(
        final_position,
        final_strength,
        radius_np,
        include_direct_energy=record_direct_energy,
    )
    target_count = min(REFERENCE_TARGETS, count)
    target_indices = np.linspace(0, count - 1, target_count, dtype=int)
    velocity = ti.Vector.field(3, dtype=ti.f32, shape=count)
    gradient = ti.Matrix.field(3, 3, dtype=ti.f32, shape=count)
    rate = ti.Vector.field(3, dtype=ti.f32, shape=count)
    induction.evaluate_stage(
        position=position,
        vortex_strength=strength,
        core_radius=radius,
        count=count,
        velocity_out=velocity,
        vortex_strength_rate_out=rate,
        velocity_gradient_out=gradient,
    )
    actual_velocity = velocity.to_numpy()[target_indices].astype(np.float64)
    actual_gradient = gradient.to_numpy()[target_indices].astype(np.float64)
    actual_rate = rate.to_numpy()[target_indices].astype(np.float64)
    reference_velocity, reference_gradient, reference_rate = _exact_subset(
        final_position,
        final_strength,
        radius_np,
        "GAUSSIAN",
        target_indices,
    )
    initial_strength_norm = max(
        float(np.linalg.norm(initial["total_strength"])),
        float(np.linalg.norm(strength_np.astype(np.float64), axis=1).sum()),
        np.finfo(float).eps,
    )
    checksum = hashlib.sha256()
    checksum.update(np.ascontiguousarray(final_position).tobytes())
    checksum.update(np.ascontiguousarray(final_strength).tobytes())
    checksum.update(np.ascontiguousarray(radius_np).tobytes())
    sha, dirty = _revision()
    result = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "commit": sha,
        "dirty": dirty,
        "backend": backend_name,
        "hardware_model": _hardware_model(backend_name),
        "precision": "f32",
        "kernel": "GAUSSIAN",
        "distribution": distribution,
        "count": count,
        "integrator": "SSPRK3",
        "steps": steps,
        "time_step_size": time_step_size,
        "elapsed_seconds": elapsed,
        "mean_accepted_step_seconds": elapsed / steps,
        "velocity_relative_l2": _relative(actual_velocity, reference_velocity),
        "gradient_relative_l2": _relative(actual_gradient, reference_gradient),
        "rate_relative_l2": _relative(actual_rate, reference_rate),
        "maximum_raw_rate_defect": maximum_rate_defect,
        "total_strength_relative_drift": float(
            np.linalg.norm(final["total_strength"] - initial["total_strength"])
            / initial_strength_norm
        ),
        "linear_impulse_relative_drift": float(
            np.linalg.norm(final["linear_impulse"] - initial["linear_impulse"])
            / max(float(np.linalg.norm(initial["linear_impulse"])), np.finfo(float).eps)
        ),
        "initial_maximum_strength": initial["maximum_strength"],
        "final_maximum_strength": final["maximum_strength"],
        "initial_cloud_radius": initial["cloud_radius"],
        "final_cloud_radius": final["cloud_radius"],
        "initial_cloud_rms_radius": initial["cloud_rms_radius"],
        "final_cloud_rms_radius": final["cloud_rms_radius"],
        "initial_cloud_radius_to_rms": initial["cloud_radius_to_rms"],
        "final_cloud_radius_to_rms": final["cloud_radius_to_rms"],
        "initial_total_kinetic_energy": initial["total_kinetic_energy"],
        "final_total_kinetic_energy": final["total_kinetic_energy"],
        "total_kinetic_energy_relative_drift": (
            abs(final["total_kinetic_energy"] - initial["total_kinetic_energy"])
            / max(abs(initial["total_kinetic_energy"]), np.finfo(float).eps)
            if record_direct_energy
            else None
        ),
        "host_particle_transfers": induction.diagnostics.host_particle_transfers,
        "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "estimated_device_bytes": induction.diagnostics.device_memory_estimate_bytes,
        "state_sha256": checksum.hexdigest(),
        "records": records,
    }
    output = RESULTS_DIR / f"evolution_{count}_{distribution}_{backend_name.lower()}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _upsert_csv(
        RESULTS_DIR / "long_horizon.csv",
        {key: value for key, value in result.items() if key != "records"},
        key_fields=("backend", "count", "distribution"),
    )
    print(json.dumps(result, indent=2))
    ti.reset()


def _run_short_trajectory(
    method: str,
    backend_name: str,
    position_np: np.ndarray,
    strength_np: np.ndarray,
    radius_np: np.ndarray,
    steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Run one full-cloud trajectory in a fresh Taichi runtime."""
    count = len(position_np)
    backend = ti.vulkan if backend_name == "VULKAN" else ti.cpu
    ti.init(
        arch=backend,
        offline_cache=False,
        cpu_max_num_threads=min(4, max(1, os.cpu_count() or 1)),
    )
    physics = PhysicsBase(
        particle_kernel="GAUSSIAN",
        max_n_particles=count,
        accumulator_dtype=ti.f32,
        max_evaluation_points=count,
    )
    position = ti.Vector.field(3, dtype=ti.f32, shape=count)
    strength = ti.Vector.field(3, dtype=ti.f32, shape=count)
    radius = ti.field(dtype=ti.f32, shape=count)
    position.from_numpy(position_np)
    strength.from_numpy(strength_np)
    radius.from_numpy(radius_np)
    constructor = DirectInduction if method == "DIRECT" else FMMInduction
    induction = constructor().bind(physics, kernel=make_vortex_kernel("GAUSSIAN"))
    integrator = RungeKutta(SSPRK3(), max_n_particles=count, dtype=ti.f32)
    right_hand_side = StageRHS(induction)
    maximum_rate_defect = 0.0
    start = time.perf_counter()
    for step in range(steps):
        integrator.advance(
            position=position,
            vortex_strength=strength,
            core_radius=radius,
            count=count,
            time=step * 1.0e-4,
            time_step_size=1.0e-4,
            right_hand_side=right_hand_side,
        )
        if method == "FMM":
            maximum_rate_defect = max(
                maximum_rate_defect,
                induction.diagnostics.last_relative_rate_defect,
            )
    ti.sync()
    metadata: dict[str, object] = {
        "elapsed_seconds": time.perf_counter() - start,
        "stage_evaluations": steps * 3,
        "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    if method == "FMM":
        metadata.update(
            {
                "host_particle_transfers": induction.diagnostics.host_particle_transfers,
                "direct_strength_rate_fallbacks": (
                    induction.diagnostics.direct_strength_rate_fallbacks
                ),
                "maximum_raw_rate_defect": maximum_rate_defect,
            }
        )
    final = (position.to_numpy(), strength.to_numpy(), radius.to_numpy(), metadata)
    ti.reset()
    return final


def run_short_comparison(
    backend_name: str,
    count: int,
    steps: int,
    distribution: str,
) -> None:
    """Compare full Direct and FMM trajectories over accepted SSPRK3 steps."""
    backend_name = backend_name.upper()
    position, strength, radius = _cloud(distribution, count)
    direct = _run_short_trajectory("DIRECT", backend_name, position, strength, radius, steps)
    fmm = _run_short_trajectory("FMM", backend_name, position, strength, radius, steps)
    position_difference = _relative(fmm[0], direct[0])
    strength_difference = _relative(fmm[1], direct[1])
    result = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "commit": _revision()[0],
        "dirty": _revision()[1],
        "backend": backend_name,
        "hardware_model": _hardware_model(backend_name),
        "precision": "f32",
        "kernel": "GAUSSIAN",
        "distribution": distribution,
        "count": count,
        "integrator": "SSPRK3",
        "steps": steps,
        "direct_elapsed_seconds": direct[3]["elapsed_seconds"],
        "fmm_elapsed_seconds": fmm[3]["elapsed_seconds"],
        "position_relative_difference": position_difference,
        "strength_relative_difference": strength_difference,
        "core_radius_maximum_absolute_difference": float(np.max(np.abs(fmm[2] - direct[2]))),
        "fmm_stage_evaluations": fmm[3]["stage_evaluations"],
        "fmm_host_particle_transfers": fmm[3]["host_particle_transfers"],
        "fmm_direct_strength_rate_fallbacks": fmm[3]["direct_strength_rate_fallbacks"],
        "fmm_maximum_raw_rate_defect": fmm[3]["maximum_raw_rate_defect"],
        "state_difference_gate": 0.02,
        "comparison_gate_passed": bool(
            position_difference <= 0.02
            and strength_difference <= 0.02
            and float(np.max(np.abs(fmm[2] - direct[2]))) == 0.0
            and fmm[3]["host_particle_transfers"] == 0
            and fmm[3]["direct_strength_rate_fallbacks"] == 0
        ),
    }
    output = RESULTS_DIR / f"direct_fmm_{count}_{steps}_{distribution}.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def plot_results() -> None:
    import matplotlib.pyplot as plt

    path = RESULTS_DIR / "scaling.csv"
    if not path.exists():
        raise FileNotFoundError("run scaling cases before plotting")
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    for method in sorted(set(np.atleast_1d(data["method"]))):
        selected = np.atleast_1d(data)[np.atleast_1d(data["method"]) == method]
        order = np.argsort(selected["count"])
        axis.loglog(selected["count"][order], selected["stage_seconds"][order], "o-", label=method)
    axis.set(xlabel="particle count", ylabel="complete stage [s]")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "scaling.png", dpi=180)
    plt.close(figure)


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "init":
        initialize_results()
        return
    if command == "plot":
        plot_results()
        return
    if command == "accuracy" and len(sys.argv) == 7:
        _, _, kernel, backend, count, distribution, method = sys.argv
        run_case(method, backend, int(count), distribution, kernel)
        return
    if command == "scaling" and len(sys.argv) == 6:
        _, _, method, backend, count, distribution = sys.argv
        run_case(method, backend, int(count), distribution, "GAUSSIAN")
        return
    if command == "evolution" and len(sys.argv) == 6:
        _, _, backend, count, steps, distribution = sys.argv
        run_evolution(backend, int(count), int(steps), distribution)
        return
    if command == "comparison" and len(sys.argv) == 6:
        _, _, backend, count, steps, distribution = sys.argv
        run_short_comparison(backend, int(count), int(steps), distribution)
        return
    raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
