#!/usr/bin/env python3
"""Run the maintained rotor with device-resident FMM and record readiness evidence."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import taichi as ti

from source.solvers.vpm.kernels.base import make_vortex_kernel
from source.solvers.vpm.physics.induction.treecode import TreecodeInduction

STUDY_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = STUDY_DIR / "results"
TUTORIAL_DIR = STUDY_DIR.parents[2] / "tutorials" / "vpm" / "rotor_flow"
RUN_LENGTHS = (10, 100, 256)
CHECKPOINTS = (1, 10, 25, 50, 100, 128, 192, 256)
DIRECT_CHECKPOINTS = {10, 100, 256}
REFERENCE_CHUNK = 512

sys.path.insert(0, str(TUTORIAL_DIR))
import setup as rotor_setup

RELEASE_INTERVAL = rotor_setup.RELEASE_INTERVAL
TEMPORAL_SUBSTEPS = 1


def _load_tutorial_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _selected_indices(strength: np.ndarray, position: np.ndarray) -> np.ndarray:
    strongest = np.argsort(np.linalg.norm(strength, axis=1))[-64:]
    distributed = np.linspace(0, len(position) - 1, min(128, len(position)), dtype=int)
    selected = list(strongest.astype(int))
    for index in distributed:
        if index not in selected and len(selected) < 128:
            selected.append(int(index))
    if len(selected) < min(128, len(position)):
        selected.extend(index for index in range(len(position)) if index not in selected)
    return np.asarray(selected[: min(128, len(position))], dtype=np.int32)


def _exact_values(
    position: np.ndarray,
    strength: np.ndarray,
    radius: np.ndarray,
    target_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kernel = make_vortex_kernel("WINCKELMANS")
    targets = position[target_indices]
    target_radius = radius[target_indices]
    velocity = np.zeros((len(target_indices), 3), dtype=np.float64)
    gradient = np.zeros((len(target_indices), 3, 3), dtype=np.float64)
    for start in range(0, len(position), REFERENCE_CHUNK):
        stop = min(start + REFERENCE_CHUNK, len(position))
        displacement = targets[:, None, :] - position[None, start:stop, :]
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


def _relative(actual: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(actual - reference) / max(np.linalg.norm(reference), 1.0e-30))


def _fmm_outputs(solver, count: int):
    dtype = solver.compute_dtype
    velocity = ti.Vector.field(3, dtype=dtype, shape=solver.particles.capacity)
    gradient = ti.Matrix.field(3, 3, dtype=dtype, shape=solver.particles.capacity)
    rate = ti.Vector.field(3, dtype=dtype, shape=solver.particles.capacity)
    solver.induction.evaluate_stage(
        position=solver.particles.position,
        vortex_strength=solver.particles.vortex_strength,
        core_radius=solver.particles.core_radius,
        count=count,
        velocity_out=velocity,
        vortex_strength_rate_out=rate,
        velocity_gradient_out=gradient,
    )
    return (
        velocity.to_numpy()[:count].astype(np.float64),
        gradient.to_numpy()[:count].astype(np.float64),
        rate.to_numpy()[:count].astype(np.float64),
    )


def _direct_sample(solver, step: int) -> dict[str, object]:
    count = solver.particles.n_particles_total
    position = solver.particle_position
    strength = solver.particle_vortex_strength
    radius = solver.particle_core_radius
    indices = _selected_indices(strength, position)
    fmm_velocity, fmm_gradient, fmm_rate = _fmm_outputs(solver, count)
    reference_velocity, reference_gradient, reference_rate = _exact_values(
        position, strength, radius, indices
    )
    particle_rate_norm = np.linalg.norm(reference_rate, axis=1)
    floor = max(float(np.sqrt(np.mean(particle_rate_norm**2))) * 1.0e-3, 1.0e-14)
    particle_error = np.linalg.norm(fmm_rate[indices] - reference_rate, axis=1) / np.maximum(
        particle_rate_norm, floor
    )
    worst = int(np.argmax(particle_error))
    result = {
        "step": step,
        "targets": len(indices),
        "velocity_relative_l2": _relative(fmm_velocity[indices], reference_velocity),
        "gradient_relative_l2": _relative(fmm_gradient[indices], reference_gradient),
        "rate_relative_l2": _relative(fmm_rate[indices], reference_rate),
        "rate_particle_p50": float(np.percentile(particle_error, 50.0)),
        "rate_particle_p95": float(np.percentile(particle_error, 95.0)),
        "rate_particle_max": float(np.max(particle_error)),
        "worst_particle_index": int(indices[worst]),
        "worst_particle_position": position[indices[worst]].astype(float).tolist(),
        "relative_total_rate_defect": float(solver.induction.diagnostics.last_relative_rate_defect),
    }
    if result["velocity_relative_l2"] > 5.0e-3:
        raise RuntimeError(f"rotor step {step}: velocity direct-sample gate failed: {result}")
    if result["gradient_relative_l2"] > 1.0e-2:
        raise RuntimeError(f"rotor step {step}: gradient direct-sample gate failed: {result}")
    if result["rate_relative_l2"] > 1.5e-2 or result["rate_particle_p95"] > 3.0e-2:
        raise RuntimeError(f"rotor step {step}: strength-rate direct-sample gate failed: {result}")
    if result["relative_total_rate_defect"] > 1.0e-3:
        raise RuntimeError(f"rotor step {step}: total-rate defect gate failed: {result}")
    return result


def _finite_state(solver) -> bool:
    values = [solver.particle_position, solver.particle_vortex_strength]
    if solver.vlm_solver is not None:
        values.append(
            solver.vlm_solver.lattice.circulation.to_numpy()[: solver.vlm_solver.lattice.n_panels]
        )
        values.extend(
            np.asarray(value)
            for value in (solver.vlm_solver._last_forces or {}).values()
            if np.isscalar(value)
        )
    return all(np.isfinite(value).all() for value in values)


def _force_coefficients(solver) -> list[float]:
    forces = solver.vlm_solver._last_forces if solver.vlm_solver is not None else {}
    return [
        float(forces.get("force_coefficient_x", 0.0)),
        float(forces.get("force_coefficient_y", 0.0)),
        float(forces.get("force_coefficient_z", 0.0)),
    ]


def _record(solver, run_steps: int, step: int, method: str) -> dict[str, object]:
    count = solver.particles.n_particles_total
    capacity = solver.particles.capacity
    position = solver.particle_position
    strength = solver.particle_vortex_strength
    gradient = solver.particle_velocity_gradient
    symmetric = 0.5 * (gradient + np.swapaxes(gradient, 1, 2))
    gamma_norm = np.linalg.norm(strength, axis=1)
    rate = np.einsum("nji,nj->ni", gradient, strength)
    chi_s = solver.time_step_size * np.linalg.norm(symmetric, axis=(1, 2))
    chi_gamma = (
        solver.time_step_size * np.linalg.norm(rate, axis=1) / np.maximum(gamma_norm, 1.0e-30)
    )
    workspace = getattr(solver.induction, "workspace", None)
    m2l_count = int(workspace._m2l_count[None]) if workspace is not None else 0
    p2p_count = int(workspace._p2p_particle_count[None]) if workspace is not None else 0
    near_count = int(workspace._near_count[None]) if workspace is not None else 0
    interaction_count = m2l_count + near_count
    interaction_capacity = int(workspace.max_pairs) if workspace is not None else 0
    estimated_bytes = (
        solver.induction.estimated_workspace_bytes(capacity)
        if hasattr(solver.induction, "estimated_workspace_bytes")
        else 0
    )
    output_dir = Path(solver.case_dir)
    vlm_circulation = (
        solver.vlm_solver.lattice.circulation.to_numpy()[: solver.vlm_solver.lattice.n_panels]
        if solver.vlm_solver is not None
        else np.zeros(0, dtype=np.float32)
    )
    row = {
        "run_steps": run_steps,
        "method": method,
        "accepted_step": step,
        "physical_time": float(solver.time),
        "particle_count": count,
        "particle_capacity": capacity,
        "capacity_fraction": count / capacity,
        "maximum_vortex_strength_magnitude": float(gamma_norm.max(initial=0.0)),
        "total_vector_vortex_strength": strength.sum(axis=0).astype(float).tolist(),
        "relative_fmm_total_rate_defect": float(
            getattr(solver.induction.diagnostics, "last_relative_rate_defect", 0.0)
        ),
        "m2l_interaction_count": m2l_count,
        "exact_p2p_particle_pair_count": p2p_count,
        "total_interaction_list_count": interaction_count,
        "interaction_list_capacity": interaction_capacity,
        "interaction_list_occupancy_fraction": (
            interaction_count / interaction_capacity if interaction_capacity else 0.0
        ),
        "estimated_fmm_workspace_bytes": estimated_bytes,
        "estimated_fmm_workspace_mib": estimated_bytes / 1024**2,
        "maximum_chi_s": float(chi_s.max(initial=0.0)),
        "maximum_chi_Gamma": float(chi_gamma.max(initial=0.0)),
        "wake_centroid": position.mean(axis=0).astype(float).tolist(),
        "vlm_circulation": vlm_circulation.astype(float).tolist(),
        "vlm_bound_circulation_norm": float(
            np.linalg.norm(
                solver.vlm_solver.lattice.circulation.to_numpy()[
                    : solver.vlm_solver.lattice.n_panels
                ]
            )
            if solver.vlm_solver is not None
            else 0.0
        ),
        "force_coefficients": _force_coefficients(solver),
        "finite_state": _finite_state(solver),
        "backup_present": any(output_dir.glob("solution/vpm_*.h5")),
        "output_present": any(output_dir.glob("samples/**/*.csv"))
        or any(output_dir.glob("samples/**/*.vtp")),
        "host_particle_transfers": int(
            getattr(solver.induction.diagnostics, "host_particle_transfers", 0)
        ),
        "direct_strength_rate_fallbacks": int(
            getattr(solver.induction.diagnostics, "direct_strength_rate_fallbacks", 0)
        ),
    }
    if not row["finite_state"]:
        raise RuntimeError(f"rotor step {step}: non-finite state")
    if interaction_capacity and row["interaction_list_occupancy_fraction"] > 1.0:
        raise RuntimeError(f"rotor step {step}: interaction-list capacity exceeded")
    if row["capacity_fraction"] > 1.0:
        raise RuntimeError(f"rotor step {step}: particle capacity exceeded")
    if method == "FMM" and row["relative_fmm_total_rate_defect"] > 1.0e-3:
        raise RuntimeError(f"rotor step {step}: FMM rate defect exceeded")
    return row


def _vlm_velocity(solver, positions: np.ndarray) -> np.ndarray:
    target_position = ti.Vector.field(3, dtype=solver.compute_dtype, shape=len(positions))
    target_velocity = ti.Vector.field(3, dtype=solver.compute_dtype, shape=len(positions))
    target_position.from_numpy(positions.astype(solver.np_dtype))
    target_velocity.fill(0.0)
    solver.vlm_solver.add_stage_velocity(
        target_position, target_velocity, len(positions), solver.time
    )
    return target_velocity.to_numpy().astype(np.float64)


def _measure_vlm_gradient(solver, step: int) -> list[dict[str, object]]:
    position = solver.particle_position
    strength = solver.particle_vortex_strength
    indices = _selected_indices(strength, position)
    targets = position[indices]
    high_strength = set(np.argsort(np.linalg.norm(strength, axis=1))[-64:].astype(int))
    distributed = set(np.linspace(0, len(position) - 1, min(128, len(position)), dtype=int))
    h = max(1.0e-5, 2.0e-3 * rotor_setup.ROTOR_RADIUS)
    gradients = []
    for delta in (h, h / 2.0, h / 4.0):
        gradient = np.zeros((len(targets), 3, 3), dtype=np.float64)
        for axis in range(3):
            offset = np.zeros(3)
            offset[axis] = delta
            gradient[:, :, axis] = (
                _vlm_velocity(solver, targets + offset) - _vlm_velocity(solver, targets - offset)
            ) / (2.0 * delta)
        gradients.append(gradient)
    gradient = gradients[1]
    gradient_half = gradients[2]
    relative_difference = np.linalg.norm(gradient - gradient_half, axis=(1, 2)) / np.maximum(
        np.linalg.norm(gradient_half, axis=(1, 2)), 1.0e-30
    )
    _, _, self_rate = _fmm_outputs(solver, solver.particles.n_particles_total)
    vlm_rate = np.einsum("nji,nj->ni", gradient, strength[indices])
    self_rate = self_rate[indices]
    ratio = np.linalg.norm(vlm_rate, axis=1) / np.maximum(
        np.linalg.norm(self_rate, axis=1), 1.0e-30
    )
    rows = []
    accepted_high_strength = 0
    accepted_distributed = 0
    for target_number, (index, difference, value) in enumerate(
        zip(indices, relative_difference, ratio, strict=True)
    ):
        if difference > 5.0e-3:
            continue
        accepted_high_strength += int(index in high_strength)
        accepted_distributed += int(index in distributed)
        rows.append(
            {
                "step": step,
                "particle_index": int(index),
                "finite_difference_relative_difference": float(difference),
                "vlm_rate_norm": float(np.linalg.norm(vlm_rate[target_number])),
                "self_rate_norm": float(np.linalg.norm(self_rate[target_number])),
                "rate_ratio": float(value),
                "h": h,
            }
        )
    if len(rows) < 96 or accepted_high_strength < 40 or accepted_distributed < 40:
        raise RuntimeError(
            f"rotor step {step}: insufficient VLM gradient coverage: "
            f"accepted={len(rows)}, high_strength={accepted_high_strength}, "
            f"distributed={accepted_distributed}"
        )
    return rows


def _run_case(
    run_steps: int,
    *,
    method: str = "FMM",
    physical_checkpoints: tuple[float, ...] = (),
    substeps: int = TEMPORAL_SUBSTEPS,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    case_dir = RESULTS_DIR / f"actual_rotor_{run_steps}"
    if method != "FMM":
        case_dir = RESULTS_DIR / f"actual_rotor_{method.lower()}_{run_steps}"
    if case_dir.exists():
        shutil.rmtree(case_dir)
    internal_step_size = RELEASE_INTERVAL / substeps
    case = rotor_setup.build_rotor_case(
        steps=run_steps * substeps,
        max_n_particles=rotor_setup.MAX_N_PARTICLES,
        directory=case_dir,
        induction=TreecodeInduction() if method == "TREECODE" else None,
        time_step_size=internal_step_size,
        wake_spacing=rotor_setup.FIXED_WAKE_SPACING,
    )
    solver = rotor_setup.vpm.VPMSolver(case)
    history = []
    comparisons = []
    vlm_rows = []
    checkpoint_steps = {
        max(1, round(physical_time / RELEASE_INTERVAL)) for physical_time in physical_checkpoints
    }
    try:
        solver._build_initial_conditions()
        for macro_step in range(1, run_steps + 1):
            macro_started = time.perf_counter()
            for substep in range(substeps):
                solver._release_wake_particles = substep == 0
                solver._release_interval = RELEASE_INTERVAL
                solver.advance()
            row = _record(solver, run_steps, macro_step, method)
            row["accepted_step_wall_seconds"] = time.perf_counter() - macro_started
            row["internal_substeps"] = substeps
            history.append(row)
            if method == "FMM" and (
                macro_step in DIRECT_CHECKPOINTS or macro_step in checkpoint_steps
            ):
                comparisons.append(_direct_sample(solver, macro_step))
                vlm_rows.extend(_measure_vlm_gradient(solver, macro_step))
        solver.save_backup()
        solver.execute_final_samples()
    finally:
        solver.close()
    return history, comparisons, vlm_rows


def _copy_state(solver) -> dict[str, object]:
    circulation = solver.vlm_solver.lattice.circulation.to_numpy()[
        : solver.vlm_solver.lattice.n_panels
    ]
    return {
        "position": solver.particle_position.copy(),
        "strength": solver.particle_vortex_strength.copy(),
        "radius": solver.particle_core_radius.copy(),
        "group_id": solver.particles.group_id_cpu().copy(),
        "zone_id": solver.particles.zone_id_cpu().copy(),
        "circulation": circulation.copy(),
        "particle_count": solver.particles.n_particles_total,
        "step": solver.step,
        "time": solver.time,
    }


def _run_restart_check() -> dict[str, object]:
    continuous_dir = RESULTS_DIR / "actual_rotor_restart_continuous"
    split_dir = RESULTS_DIR / "actual_rotor_restart_split"
    for path in (continuous_dir, split_dir):
        if path.exists():
            shutil.rmtree(path)
    continuous_solver = rotor_setup.vpm.VPMSolver(
        rotor_setup.build_rotor_case(
            steps=128, max_n_particles=rotor_setup.MAX_N_PARTICLES, directory=continuous_dir
        )
    )
    try:
        continuous_solver._build_initial_conditions()
        for _ in range(128):
            continuous_solver.advance()
        continuous = _copy_state(continuous_solver)
    finally:
        continuous_solver.close()

    first_solver = rotor_setup.vpm.VPMSolver(
        rotor_setup.build_rotor_case(
            steps=64, max_n_particles=rotor_setup.MAX_N_PARTICLES, directory=split_dir
        )
    )
    backup_path = split_dir / "solution" / "vpm.h5"
    try:
        first_solver._build_initial_conditions()
        for _ in range(64):
            first_solver.advance()
        first_solver.save_backup()
    finally:
        first_solver.close()

    resumed_solver = rotor_setup.vpm.VPMSolver(
        rotor_setup.build_rotor_case(
            steps=64, max_n_particles=rotor_setup.MAX_N_PARTICLES, directory=split_dir
        )
    )
    try:
        resumed_solver.load_backup(str(backup_path))
        for _ in range(64):
            resumed_solver.advance()
        resumed = _copy_state(resumed_solver)
    finally:
        resumed_solver.close()

    result = {
        "particle_count_identical": continuous["particle_count"] == resumed["particle_count"],
        "accepted_step_identical": continuous["step"] == resumed["step"],
        "accepted_time_identical": continuous["time"] == resumed["time"],
        "position_allclose": bool(
            np.allclose(continuous["position"], resumed["position"], rtol=1.0e-6, atol=1.0e-7)
        ),
        "strength_allclose": bool(
            np.allclose(continuous["strength"], resumed["strength"], rtol=1.0e-6, atol=1.0e-7)
        ),
        "radius_allclose": bool(
            np.allclose(continuous["radius"], resumed["radius"], rtol=1.0e-7, atol=1.0e-8)
        ),
        "group_id_identical": np.array_equal(continuous["group_id"], resumed["group_id"]),
        "zone_id_identical": np.array_equal(continuous["zone_id"], resumed["zone_id"]),
        "vlm_circulation_allclose": bool(
            np.allclose(continuous["circulation"], resumed["circulation"], rtol=1.0e-6, atol=1.0e-7)
        ),
    }
    result["comparison_gate_passed"] = all(result.values())
    if not result["comparison_gate_passed"]:
        raise RuntimeError(f"rotor restart equivalence failed: {result}")
    return result


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_history = []
    all_comparisons = []
    all_vlm_rows = []
    for run_steps in RUN_LENGTHS:
        history, comparisons, vlm_rows = _run_case(run_steps)
        all_history.extend(history)
        all_comparisons.extend(comparisons)
        all_vlm_rows.extend(vlm_rows)
    with (RESULTS_DIR / "actual_rotor_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        rows = all_history
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (RESULTS_DIR / "actual_rotor_vlm_gradient.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_vlm_rows[0]))
        writer.writeheader()
        writer.writerows(all_vlm_rows)
    summary = {
        "configuration": {
            "integrator": "SSPRK3",
            "induction": "FMMInduction",
            "particle_kernel": "WINCKELMANS",
            "compute_device": "VULKAN",
            "precision": "f32",
            "time_step_size": rotor_setup.TIME_STEP_SIZE,
            "particle_capacity": rotor_setup.MAX_N_PARTICLES,
        },
        "history_rows": len(all_history),
        "direct_comparisons": all_comparisons,
        "host_particle_transfers": max(row["host_particle_transfers"] for row in all_history),
        "direct_strength_rate_fallbacks": max(
            row["direct_strength_rate_fallbacks"] for row in all_history
        ),
        "maximum_rate_defect": max(row["relative_fmm_total_rate_defect"] for row in all_history),
        "maximum_chi_s": max(row["maximum_chi_s"] for row in all_history),
        "maximum_chi_Gamma": max(row["maximum_chi_Gamma"] for row in all_history),
    }
    restart = _run_restart_check()
    (RESULTS_DIR / "actual_rotor_restart.json").write_text(
        json.dumps(restart, indent=2) + "\n", encoding="utf-8"
    )
    fmm_256 = [row for row in all_history if row["run_steps"] == 256]
    interval_growth = []
    for end in (32, 64, 96, 128, 160, 192, 224, 256):
        start = end - 32
        start_count = (
            0
            if start == 0
            else next(row["particle_count"] for row in fmm_256 if row["accepted_step"] == start)
        )
        end_count = next(row["particle_count"] for row in fmm_256 if row["accepted_step"] == end)
        interval_growth.append((end_count - start_count) / 32.0)
    largest_growth = max(0.0, max(interval_growth))
    particles_at_256 = next(row["particle_count"] for row in fmm_256 if row["accepted_step"] == 256)
    projected_particles = particles_at_256 + largest_growth * (2400 - 256)
    final_capacity = max(
        rotor_setup.MAX_N_PARTICLES,
        int(np.ceil(1.25 * projected_particles / 10000.0) * 10000),
    )
    fmm_rows = [row for row in all_history if row["method"] == "FMM"]
    vlm_ratios = np.asarray([row["rate_ratio"] for row in all_vlm_rows])
    tree_history, _, _ = _run_case(100, method="TREECODE")
    fmm_100 = [row for row in all_history if row["run_steps"] == 100]
    tree_counts = np.asarray([row["particle_count"] for row in tree_history])
    fmm_counts = np.asarray([row["particle_count"] for row in fmm_100])
    circulation_difference = _relative(
        np.asarray([row["vlm_circulation"] for row in fmm_100]),
        np.asarray([row["vlm_circulation"] for row in tree_history]),
    )
    force_difference = _relative(
        np.asarray([row["force_coefficients"] for row in fmm_100]).sum(axis=0),
        np.asarray([row["force_coefficients"] for row in tree_history]).sum(axis=0),
    )
    centroid_difference = float(
        np.linalg.norm(
            np.asarray(fmm_100[-1]["wake_centroid"]) - np.asarray(tree_history[-1]["wake_centroid"])
        )
    )
    summary["treecode_comparison"] = {
        "fmm_particle_count_final": int(fmm_counts[-1]),
        "treecode_particle_count_final": int(tree_counts[-1]),
        "particle_count_history_identical": bool(np.array_equal(fmm_counts, tree_counts)),
        "bound_circulation_history_relative_difference": circulation_difference,
        "integrated_force_coefficients_relative_difference": force_difference,
        "wake_centroid_distance": centroid_difference,
        "comparison_gate_passed": bool(
            circulation_difference <= 0.02
            and force_difference <= 0.02
            and centroid_difference <= rotor_setup.ROTOR_RADIUS * 0.1
        ),
    }
    summary.update(
        {
            "vlm_gradient_ratio_median": float(np.median(vlm_ratios)),
            "vlm_gradient_ratio_p95": float(np.percentile(vlm_ratios, 95.0)),
            "vlm_gradient_ratio_max": float(np.max(vlm_ratios)),
            "vlm_advection_only_policy": "measured; see actual_rotor_vlm_gradient.csv",
            "restart": restart,
            "capacity_projection": {
                "particles_at_256": particles_at_256,
                "largest_growth_per_step": largest_growth,
                "projected_particles_at_2400": projected_particles,
                "final_capacity": final_capacity,
                "workspace_bytes": int(
                    rotor_setup.vpm.FMMInduction().estimated_workspace_bytes(final_capacity)
                ),
            },
            "maximum_capacity_fraction": max(row["capacity_fraction"] for row in fmm_rows),
        }
    )
    (RESULTS_DIR / "actual_rotor_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
