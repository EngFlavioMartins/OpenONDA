"""Replay VPM with prescribed 3D reference donors and no FVM boundary feedback.

Reference native velocities at t=6 and 7 supply the renewal target, linearly
interpolated in time. Only VPM evolves. This tests whether FVM mixed-boundary
feedback is necessary for the existing transverse disturbance to grow.
Optional controls change alignment, the nodal covector source, or the compact
conservative source on remeshed cells. They are mutually exclusive experiments.
An optional precomputed strength file screens a fixed-basis correction from the
same accepted particle checkpoint without changing donor or boundary inputs.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time as wall_clock

from cube_covector_flux_step import invariants
from cube_fixed_basis_runtime_fit import fit_outer_strength
from cube_lattice_phase_study import distance, reference_gradient, smooth
from cube_wake_drift_audit import frame, ordered_fields
from cube_wake_particle_probe import load_case, rms
from numba import set_num_threads
import numpy as np
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits


def lattice_and_target(args):
    from source.coupler.interpolation import FVMVelocityInterpolator
    from source.coupler.stable_renewal import (
        build_stable_renewal_lattice,
        vortex_strength_from_velocity_trace,
    )
    from source.solvers.fvm.io.mesh_storage import load_native_mesh
    from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
    from source.solvers.fvm.sampling.fields import _PointProbe

    lattice = build_stable_renewal_lattice(
        np.array([-1.25, 1.25] * 3),
        0.06,
        buffer_length=0.135,
        authority_ramp_width=0.36,
        lattice_anchor=np.array([-0.03] * 3),
        mesh_weight_at_node=lambda p: np.all(np.abs(p) <= 1.5 + 1e-12, axis=1).astype(float),
        fluid_weight_at_node=lambda p: smooth(distance(p), -0.06, 0),
        interior_at_node=lambda p: np.all(np.abs(p) < 0.5, axis=1),
    )
    x = np.array([0.9, 1.14, 1.26, 1.38, 1.5, 1.62, 1.86, 2.34])
    yz = np.linspace(-0.54, 0.54, 7)
    points = np.stack(np.meshgrid(x, yz, yz, indexing="ij"), axis=-1).reshape(-1, 3)
    # The same full 3D probe orbit as the isolated operator audit.
    if args.target_cache.exists():
        with np.load(args.target_cache) as data:
            np.testing.assert_array_equal(data["lattice_position"], lattice.positions)
            np.testing.assert_array_equal(data["points"], points)
            return (
                lattice,
                points,
                [data["target_0"].copy(), data["target_1"].copy()],
                [data["reference_0"].copy(), data["reference_1"].copy()],
            )
    mesh = load_native_mesh(args.reference / "mesh.npz")
    geometry = compute_mesh_geometry(mesh, compute_lsq=False)
    trace = FVMVelocityInterpolator(
        geometry["cell_centre"], cKDTree(geometry["cell_centre"]), neighbour_count=4
    )
    probe = _PointProbe(points, k=12, reconstruction="affine")
    targets, references = [], []
    for t in (6.0, 7.0):
        fields = ordered_fields(frame(args.reference, "fine", t), mesh["n_cells"])
        gradient = reference_gradient(fields["velocity"], mesh, geometry)

        def velocity_at(p, fields=fields, gradient=gradient):
            return (
                trace.sample(p, fields["velocity"], gradient)
                * smooth(distance(p), 0, 0.06)[:, None]
            )

        targets.append(vortex_strength_from_velocity_trace(lattice.positions, 0.06, velocity_at))
        references.append(probe._interpolate(fields["velocity"], geometry["cell_centre"]))
    args.target_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.target_cache,
        lattice_position=lattice.positions,
        points=points,
        target_0=targets[0],
        target_1=targets[1],
        reference_0=references[0],
        reference_1=references[1],
    )
    return lattice, points, targets, references


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    case = load_case(args.source_tree)
    from source.coupler.vorticity_transfer import replace_particles_from_buffered_m4_renewal
    from source.solvers.vpm import VPMSolver

    lattice, points, targets, references = lattice_and_target(args)
    policy = replace(
        case.VPM_CASE,
        directory=args.output / "runtime",
        samplers=case.vpm.Samplers(),
        numerics=replace(
            case.VPM_CASE.numerics,
            time_step_size=args.time_step_size,
            max_n_particles=300000,
            max_evaluation_points=300000,
        ),
    )
    report = {
        "description": __doc__,
        "alignment_rate_per_second": args.alignment_rate,
        "nodal_covector_control": args.covector_control,
        "compact_flux_control": args.flux_control,
        "outer_projection_interval_steps": args.projection_interval,
        "time_step_size": args.time_step_size,
        "renewal_interval": args.renewal_interval,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "initial_strength_file": (
            None if args.initial_strength is None else str(args.initial_strength.resolve())
        ),
        "initial_strength_sha256": (
            None
            if args.initial_strength is None
            else hashlib.sha256(args.initial_strength.read_bytes()).hexdigest()
        ),
        "target_cache_sha256": hashlib.sha256(args.target_cache.read_bytes()).hexdigest(),
        "time_interpolation": "Linear between native 3D reference fields at 6 and 7; endpoint errors use actual reference states",
        "observations": [],
        "step_budgets": [],
        "outer_projection_events": [],
        "steps_completed": 0,
        "steps_requested": args.steps,
        "status": "running",
        "source_sha256": {
            str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        },
    }
    masks = {
        "renewal_seam": (points[:, 0] >= 1.25) & (points[:, 0] <= 1.62),
        "outer_wake": points[:, 0] > 1.62,
    }
    solver = VPMSolver(policy)
    covector = None
    flux = None
    try:
        solver.load_backup(
            args.checkpoint,
            time_step_size=args.time_step_size if args.time_step_size != 0.01 else None,
        )
        assert abs(solver.time - 6) < 1e-8
        assert abs(solver.time_step_size - args.time_step_size) < 1e-12
        solver.physics.configure_body_box(np.array([-0.5, 0.5] * 3))
        solver.physics.configure_grid_lattice_anchor(np.array([-0.03] * 3), 0.06)
        if args.initial_strength is not None:
            with np.load(args.initial_strength) as stored:
                candidate = stored["vortex_strength"].astype(solver.np_dtype)
            if candidate.shape != (len(solver.particles), 3) or not np.all(np.isfinite(candidate)):
                raise ValueError("Initial fixed-basis strength is not a finite matched cloud")
            report["initial_strength_before"] = invariants(solver)
            solver.set_particles_properties(vortex_strength=candidate)
            solver.notify_external_particle_mutation()
            report["initial_strength_after"] = invariants(solver)
        if args.covector_control:
            from cube_covector_stage_source import NodalCovectorSource

            covector = NodalCovectorSource(solver)
            solver.stage_rhs.providers = (*solver.stage_rhs.providers, covector)
            solver.stepper._apply_viscous_diffusion = covector.observe_diffusion(
                solver.stepper._apply_viscous_diffusion
            )
            for name in ("cube_covector_stage_source.py", "cube_wake_measured_longitudinal.py"):
                path = Path(__file__).with_name(name)
                report["source_sha256"][str(path.resolve())] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
        if args.flux_control:
            from cube_covector_flux_step import CompactFluxStep

            flux = CompactFluxStep(solver)
            for name in ("cube_covector_flux_step.py", "cube_covector_flux_control.py"):
                path = Path(__file__).with_name(name)
                report["source_sha256"][str(path.resolve())] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
        if args.projection_interval:
            from source.solvers.vpm.stabilization.divergence_relaxation import (
                GaussianParticleGridOperator,
                _MomentNullspace,
                gaussian_invariant_rows,
            )
            from source.solvers.vpm.stabilization.filament_refinement import (
                gaussian_particle_moments,
            )

            path = Path(__file__).with_name("cube_fixed_basis_runtime_fit.py")
            report["source_sha256"][str(path.resolve())] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        if args.alignment_rate:
            stabilization = replace(
                solver.stabilization_config,
                pedrizzetti_relaxation_factor=args.alignment_rate * solver.time_step_size,
                pedrizzetti_relaxation_preserve_moments=True,
            )
            solver.stabilization_config = stabilization
            solver.stabilization.config = stabilization
            solver.stabilization.ctx = replace(solver.stabilization.ctx, config=stabilization)
            solver.setup = replace(solver.setup, stabilization=stabilization)
            solver.numerics = solver.setup
            solver.case = replace(solver.case, numerics=solver.setup)
        solver.refresh_boundary_element_solution()
        started = wall_clock.perf_counter()
        observation_stride = round(0.1 / solver.time_step_size)
        renewal_stride = round(args.renewal_interval / solver.time_step_size)
        for step in range(args.steps + 1):
            if step % observation_stride == 0 or step == args.steps:
                u = solver.compute_velocity_at_points(points)
                fraction = solver.time - 6.0
                reference = (1 - fraction) * references[0] + fraction * references[1]
                reflected = points.copy()
                reflected[:, 2] *= -1
                ur = solver.compute_velocity_at_points(reflected) * [1, 1, -1]
                asymmetry = 0.5 * (u - ur)
                row = {
                    "time": solver.time,
                    "particles": len(solver.particles),
                    "elapsed_wall_seconds": wall_clock.perf_counter() - started,
                    "regions": {
                        name: {
                            "velocity_error_rms": rms((u - reference)[m]),
                            "transverse_error_rms": rms((u - reference)[m, 1:]),
                            "reflection_asymmetry_rms": rms(asymmetry[m]),
                        }
                        for name, m in masks.items()
                    },
                }
                report["observations"].append(row)
                if covector is not None:
                    report["covector_source"] = covector.measurements()
                if flux is not None:
                    report["compact_flux_source"] = flux.measurements()
                print(json.dumps(row), flush=True)
                np.savez_compressed(
                    args.output / f"fields_step{step:03d}.npz",
                    points=points,
                    velocity=u,
                    reference=reference,
                    asymmetry=asymmetry,
                )
            if step == args.steps:
                break
            step_started = wall_clock.perf_counter()
            budget = {"start_time": solver.time, "before": invariants(solver)}
            if flux is not None:
                flux.advance(0.5 * solver.time_step_size, "before_native_evolution")
            budget["before_native_evolution"] = invariants(solver)
            native_started = wall_clock.perf_counter()
            solver.advance(defer_output=True)
            budget["native_evolution_wall_seconds"] = wall_clock.perf_counter() - native_started
            budget["before_transfer"] = invariants(solver)
            if covector is not None:
                covector.step_budgets[-1]["before_transfer"] = covector.invariants()
            fraction = solver.time - 6.0
            target = (1 - fraction) * targets[0] + fraction * targets[1]
            transfer_started = wall_clock.perf_counter()
            budget["transfer_due"] = (step + 1) % renewal_stride == 0
            if budget["transfer_due"]:
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
                    time_step_size=args.renewal_interval,
                    compute_diagnostics=False,
                )
            budget["transfer_wall_seconds"] = wall_clock.perf_counter() - transfer_started
            budget["after_transfer"] = invariants(solver)
            if (
                args.projection_interval
                and (step + 1) % args.projection_interval == 0
                and step + 1 < args.steps
            ):
                particles = solver.particles
                position = particles.position_cpu(use_cache=False).astype(np.float64)
                strength = particles.vortex_strength_cpu(use_cache=False).astype(np.float64)
                radius = particles.core_radius_cpu(use_cache=False).astype(np.float64)
                volume = particles.particle_volume_cpu(use_cache=False).astype(np.float64)
                before_moments = gaussian_particle_moments(position, strength, radius)
                projected, fit = fit_outer_strength(
                    position,
                    strength,
                    radius,
                    volume,
                    operator_type=GaussianParticleGridOperator,
                    moment_nullspace_type=_MomentNullspace,
                    gaussian_moment_rows=gaussian_invariant_rows,
                    renewal_bounds=lattice.renewal_bounds,
                )
                stored_strength = np.asarray(projected, dtype=solver.np_dtype)
                after_moments = gaussian_particle_moments(
                    position, stored_strength.astype(np.float64), radius
                )
                moment_errors = [
                    float(np.linalg.norm(after_moments[index] - before_moments[index]))
                    for index in (0, 2, 3)
                ]
                if max(moment_errors) > 1e-6:
                    raise RuntimeError(
                        f"Research projection exceeded its storage moment error: {moment_errors}"
                    )
                if np.any(np.all(np.abs(position) < 0.5, axis=1)):
                    raise RuntimeError("Research projection input has particles inside the cube")
                solver.set_particles_properties(vortex_strength=stored_strength)
                solver.notify_external_particle_mutation()
                fit["time_s"] = float(solver.time)
                fit["moment_errors_net_impulse_angular"] = moment_errors
                report["outer_projection_events"].append(fit)
                budget["after_projection"] = invariants(solver)
            if flux is not None:
                flux.advance(0.5 * solver.time_step_size, "after_native_evolution")
            solver.refresh_boundary_element_solution()
            solver.execute_scheduled_samplers()
            budget["after"] = invariants(solver)
            budget["wall_seconds"] = wall_clock.perf_counter() - step_started
            report["step_budgets"].append(budget)
            if covector is not None:
                covector.step_budgets[-1]["after_transfer"] = covector.invariants()
            report["steps_completed"] = step + 1
            if covector is not None:
                report["covector_source"] = covector.measurements()
            if flux is not None:
                report["compact_flux_source"] = flux.measurements()
            (args.output / "replay.json").write_text(json.dumps(report, indent=2) + "\n")
        np.savez_compressed(
            args.output / "final_particles.npz",
            position=solver.particle_position,
            vortex_strength=solver.particle_vortex_strength,
            core_radius=solver.particle_core_radius,
        )
        report["status"] = "completed"
        (args.output / "replay.json").write_text(json.dumps(report, indent=2) + "\n")
    except Exception as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        if covector is not None:
            report["covector_source"] = covector.measurements()
        if flux is not None:
            report["compact_flux_source"] = flux.measurements()
        (args.output / "replay.json").write_text(json.dumps(report, indent=2) + "\n")
        raise
    finally:
        solver.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tree", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--target-cache", type=Path, required=True)
    parser.add_argument("--initial-strength", type=Path)
    parser.add_argument("--projection-interval", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alignment-rate", type=float, default=0)
    parser.add_argument("--covector-control", action="store_true")
    parser.add_argument("--flux-control", action="store_true")
    parser.add_argument("--time-step-size", type=float, choices=(0.01, 0.005, 0.0025), default=0.01)
    parser.add_argument("--renewal-interval", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.steps <= round(1 / args.time_step_size):
        parser.error("This bounded replay must remain inside the physical interval [6, 7]")
    if args.projection_interval < 0:
        parser.error("Projection interval must be a non-negative number of steps")
    if (
        sum(
            (
                args.covector_control,
                args.flux_control,
                bool(args.alignment_rate),
                bool(args.projection_interval),
            )
        )
        > 1
    ):
        parser.error("Accuracy controls must be measured separately")
    ratio = args.renewal_interval / args.time_step_size
    if not np.isfinite(ratio) or ratio < 1 or abs(ratio - round(ratio)) > 1e-10:
        parser.error("Renewal interval must be a positive integer multiple of the time step")
    set_num_threads(2)
    with threadpool_limits(limits=2):
        run(args)


if __name__ == "__main__":
    main()
