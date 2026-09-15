"""Bounded VPM replay with reference-driven curl residual transfer.

This trial retains the original evolution, GBD, particle spacing, cores and
health gates. Renewal first performs the coefficient-preserving control through
its normal cleanup. A compact curl correction is added after that cleanup and
is not clipped, pruned, or repaired back to the old impulse. Fine-reference
velocities are interpolated in time between the same native t=6 and t=7 states
used by the preceding controls. Only those endpoints are exact saved times.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

from cube_lattice_phase_study import distance, smooth
from cube_wake_oracle_replay import lattice_and_target
from cube_wake_particle_probe import load_case, rms
from numba import set_num_threads
import numpy as np
from threadpoolctl import threadpool_limits


def add_correction(solver, lattice, correction):
    """Merge the full correction into aligned particles without losing support."""
    p, g, sigma = (
        getattr(solver, name).astype(float)
        for name in ("particle_position", "particle_vortex_strength", "particle_core_radius")
    )
    relative = (p - lattice.origin) / lattice.particle_spacing
    index = np.rint(relative).astype(int)
    inside = np.all((index >= 0) & (index < np.asarray(lattice.shape)), axis=1)
    inside &= np.max(np.abs(relative - index), axis=1) < 1e-4
    np.testing.assert_allclose(sigma[inside], 0.066, rtol=0, atol=1e-8)
    updated = correction.copy()
    flat = np.ravel_multi_index(index[inside].T, lattice.shape)
    np.add.at(updated, flat, g[inside])
    active = np.linalg.norm(updated, axis=1) > 0
    position = np.vstack((lattice.positions[active], p[~inside]))
    strength = np.vstack((updated[active], g[~inside]))
    radii = np.r_[np.full(int(active.sum()), 0.066), sigma[~inside]]
    if np.any(np.all(np.abs(position) < 0.5, axis=1)):
        raise ValueError(
            "Curl correction produced a particle inside the solid; no clipping allowed"
        )
    n = len(position)
    if n > solver.particles.capacity:
        raise ValueError(
            f"Full correction needs {n} particles; capacity is {solver.particles.capacity}"
        )
    dtype = solver.np_dtype
    solver.replace_vortex_particles(
        position=np.asarray(position, dtype=dtype),
        velocity=np.zeros((n, 3), dtype=dtype),
        vortex_strength=np.asarray(strength, dtype=dtype),
        core_radius=np.asarray(radii, dtype=dtype),
        particle_volume=np.full(n, 0.06**3, dtype=dtype),
        kinematic_viscosity=np.full(n, 0.001, dtype=dtype),
        eddy_viscosity=np.zeros(n, dtype=dtype),
        group_id=np.zeros(n, dtype=np.int32),
        zone_id=np.zeros(n, dtype=np.int32),
        report_removal=False,
    )
    return {
        "particles": n,
        "correction_strength_l1": float(np.linalg.norm(correction, axis=1).sum()),
        "correction_net_strength": correction.sum(axis=0).tolist(),
        "correction_impulse": (0.5 * np.cross(lattice.positions, correction).sum(axis=0)).tolist(),
    }


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    case = load_case(args.source_tree)
    from cube_curl_residual_transfer import curl_residual_correction

    from source.coupler import stable_renewal as renewal
    from source.coupler.vorticity_transfer import replace_particles_from_buffered_m4_renewal
    from source.solvers.vpm import VPMSolver

    lattice, points, targets, references = lattice_and_target(args)
    with np.load(args.velocity_cache) as data:
        cache = {key: data[key].copy() for key in data.files}
    policy = replace(
        case.VPM_CASE,
        directory=args.output / "runtime",
        samplers=case.vpm.Samplers(),
        numerics=replace(
            case.VPM_CASE.numerics, max_n_particles=300000, max_evaluation_points=300000
        ),
    )
    masks = {
        "renewal_seam": (points[:, 0] >= 1.25) & (points[:, 0] <= 1.62),
        "outer_wake": points[:, 0] > 1.62,
    }
    original_blend = renewal.blend_represented_state

    def passive_blend(*positional, **keywords):
        result = original_blend(*positional, **keywords)
        return replace(result, vortex_strength=positional[0].copy())

    def lookup(query, values):
        fractional = (query - cache["origin"]) / float(cache["spacing"])
        index = np.rint(fractional).astype(int)
        if np.max(np.abs(fractional - index)) > 1e-8:
            raise ValueError("Curl requested a query outside the exact reference lattice")
        flat = np.ravel_multi_index(index.T, tuple(cache["shape"]))
        return values[flat]

    report = {
        "description": __doc__,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "velocity_cache_sha256": hashlib.sha256(args.velocity_cache.read_bytes()).hexdigest(),
        "body_velocity_residual_guard": [0.06, 0.12],
        "deconvolution_steps": 0,
        "steps_completed": 0,
        "status": "running",
        "observations": [],
        "step_records": [],
    }
    study_sources = [Path(__file__), Path(__file__).with_name("cube_curl_residual_transfer.py")]
    numerical_sources = [
        args.source_tree / "source/coupler/stable_renewal.py",
        args.source_tree / "source/coupler/vorticity_transfer.py",
        args.source_tree / "source/solvers/vpm/core/evolution.py",
    ]
    report["source_sha256"] = {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [*study_sources, *numerical_sources]
    }
    solver = VPMSolver(policy)
    try:
        solver.load_backup(args.checkpoint)
        assert abs(solver.time - 6) < 1e-8
        solver.physics.configure_body_box(np.array([-0.5, 0.5] * 3))
        solver.physics.configure_grid_lattice_anchor(np.array([-0.03] * 3), 0.06)
        solver.refresh_boundary_element_solution()
        started = time.perf_counter()
        for step in range(args.steps + 1):
            if step % 10 == 0 or step == args.steps:
                velocity = solver.compute_velocity_at_points(points)
                reference = (1 - step / 100) * references[0] + step / 100 * references[1]
                reflected = points * [1, 1, -1]
                asymmetry = 0.5 * (
                    velocity - solver.compute_velocity_at_points(reflected) * [1, 1, -1]
                )
                row = {
                    "time": solver.time,
                    "particles": len(solver.particles),
                    "elapsed_wall_seconds": time.perf_counter() - started,
                    "regions": {
                        name: {
                            "velocity_error_rms": rms((velocity - reference)[mask]),
                            "transverse_error_rms": rms((velocity - reference)[mask, 1:]),
                            "reflection_asymmetry_rms": rms(asymmetry[mask]),
                        }
                        for name, mask in masks.items()
                    },
                }
                report["observations"].append(row)
                np.savez_compressed(
                    args.output / f"fields_step{step:03d}.npz",
                    points=points,
                    velocity=velocity,
                    reference=reference,
                    asymmetry=asymmetry,
                )
                print(json.dumps(row), flush=True)
            if step == args.steps:
                break
            step_start = time.perf_counter()
            solver.advance(defer_output=True)
            after_evolution = time.perf_counter()
            fraction = (step + 1) / 100
            target = (1 - fraction) * targets[0] + fraction * targets[1]
            renewal.blend_represented_state = passive_blend
            try:
                replace_particles_from_buffered_m4_renewal(
                    solver,
                    lattice=lattice,
                    fvm_vortex_strength_at_node=lambda _, target=target: target,
                    particle_fluid_weight=lambda p: smooth(distance(p), -0.06, 0),
                    particle_in_solid=lambda p: np.all(np.abs(p) < 0.5, axis=1),
                    prune_threshold=0.05 * 0.06**3,
                    core_radius_ratio=1.1,
                    amplification_cap=1.8,
                    boundary_prune_multiplier=10,
                    kinematic_viscosity=0.001,
                    freestream_speed=1,
                    time_step_size=solver.time_step_size,
                    compute_diagnostics=False,
                )
            finally:
                renewal.blend_represented_state = original_blend
            solver.refresh_boundary_element_solution()
            after_remap = time.perf_counter()
            vpm_velocity = solver.compute_velocity_at_points(cache["position"])
            fvm_velocity = (1 - fraction) * cache["velocity_t6"] + fraction * cache["velocity_t7"]
            correction = curl_residual_correction(
                lattice.positions,
                lattice.shape,
                0.06,
                fvm_velocity_at=lambda query, values=fvm_velocity: lookup(query, values),
                vpm_velocity_at=lambda query, values=vpm_velocity: lookup(query, values),
                authority_at=lambda query: renewal.inward_cosine_authority(
                    query, np.array([-1.25, 1.25] * 3), 0.36
                ),
                fluid_weight_at=lambda query: smooth(distance(query), 0.06, 0.12),
            )
            budget = add_correction(solver, lattice, correction)
            after_correction = time.perf_counter()
            solver.refresh_boundary_element_solution()
            solver.execute_scheduled_samplers()
            record = {
                "step": step + 1,
                "evolution_seconds": after_evolution - step_start,
                "passive_remap_seconds": after_remap - after_evolution,
                "curl_correction_seconds": after_correction - after_remap,
                "health_and_body_seconds": time.perf_counter() - after_correction,
                **budget,
            }
            report["step_records"].append(record)
            report["steps_completed"] = step + 1
            (args.output / "replay.json").write_text(json.dumps(report, indent=2) + "\n")
        report["status"] = "completed"
        (args.output / "replay.json").write_text(json.dumps(report, indent=2) + "\n")
        np.savez_compressed(
            args.output / "final_particles.npz",
            position=solver.particle_position,
            vortex_strength=solver.particle_vortex_strength,
            core_radius=solver.particle_core_radius,
        )
    except Exception as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        (args.output / "replay.json").write_text(json.dumps(report, indent=2) + "\n")
        raise
    finally:
        renewal.blend_represented_state = original_blend
        solver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tree", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--target-cache", type=Path, required=True)
    parser.add_argument("--velocity-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.steps <= 100:
        parser.error("This bounded trial supports 1 through 100 steps only")
    set_num_threads(2)
    with threadpool_limits(limits=2):
        run(args)
