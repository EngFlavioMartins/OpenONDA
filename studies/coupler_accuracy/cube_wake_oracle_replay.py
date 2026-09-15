"""Replay VPM with prescribed 3D reference donors and no FVM boundary feedback.

Reference native velocities at t=6 and 7 supply the renewal target, linearly
interpolated in time. Only VPM evolves. This tests whether FVM mixed-boundary
feedback is necessary for the existing transverse disturbance to grow.
Optional controls change either the existing alignment operator or the nodal
covector stage source. They are distinct, mutually exclusive experiments.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time as wall_clock

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
            case.VPM_CASE.numerics, max_n_particles=300000, max_evaluation_points=300000
        ),
    )
    report = {
        "description": __doc__,
        "alignment_rate_per_second": args.alignment_rate,
        "nodal_covector_control": args.covector_control,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "target_cache_sha256": hashlib.sha256(args.target_cache.read_bytes()).hexdigest(),
        "time_interpolation": "Linear between native 3D reference fields at 6 and 7; endpoint errors use actual reference states",
        "observations": [],
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
    try:
        solver.load_backup(args.checkpoint)
        assert abs(solver.time - 6) < 1e-8
        solver.physics.configure_body_box(np.array([-0.5, 0.5] * 3))
        solver.physics.configure_grid_lattice_anchor(np.array([-0.03] * 3), 0.06)
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
        for step in range(args.steps + 1):
            if step % 10 == 0 or step == args.steps:
                u = solver.compute_velocity_at_points(points)
                reference = (1 - step / 100) * references[0] + (step / 100) * references[1]
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
            solver.advance(defer_output=True)
            if covector is not None:
                covector.step_budgets[-1]["before_transfer"] = covector.invariants()
            fraction = (step + 1) / 100
            target = (1 - fraction) * targets[0] + fraction * targets[1]
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
            solver.refresh_boundary_element_solution()
            solver.execute_scheduled_samplers()
            if covector is not None:
                covector.step_budgets[-1]["after_transfer"] = covector.invariants()
            report["steps_completed"] = step + 1
            if covector is not None:
                report["covector_source"] = covector.measurements()
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alignment-rate", type=float, default=0)
    parser.add_argument("--covector-control", action="store_true")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.steps <= 100:
        parser.error("This bounded replay supports 1 through 100 steps")
    if args.covector_control and args.alignment_rate:
        parser.error("The two controls must be measured separately")
    set_num_threads(2)
    with threadpool_limits(limits=2):
        run(args)


if __name__ == "__main__":
    main()
