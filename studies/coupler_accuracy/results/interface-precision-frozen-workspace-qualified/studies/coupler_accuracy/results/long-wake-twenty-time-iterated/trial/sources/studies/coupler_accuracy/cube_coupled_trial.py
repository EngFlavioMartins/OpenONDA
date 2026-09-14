#!/usr/bin/env python3
"""Run the real 3D coupler beside an independent FVM on inherited cells.

The reference advances only for same-time measurement; none of its evolving
fields feed the hybrid solver. Both FVMs start from the oracle's same saved
cell fields. Initial VPM particles use the full seed vorticity times volume,
with an explicitly recorded weak-vorticity cutoff. This is a short diagnostic
initial-value experiment, not a continuation of deleted tutorial results.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import copy
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree

import openonda.coupler as coupling
import openonda.fvm as fvm
import openonda.vpm as vpm
from source.coupler.interpolation import FVMVelocityInterpolator
from source.solvers.fvm.fields.gradients import compute_lsq_geometry, compute_lsq_gradient
from source.solvers.fvm.io.mesh_storage import load_native_mesh
from studies.coupler_accuracy.cube_boundary_oracle import (
    CASE,
    ROOT,
    field_rms,
    hash_file,
    setup_for,
)
from studies.coupler_accuracy.experimental_mixed_convection import mixed_outflow_linear_upwind


class MatchedComparison:
    """Observe accepted coupled states without changing either hybrid solver."""

    name = "matched_3d_comparison"
    schedule = vpm.EverySteps(1)

    def __init__(self, full, small, ids, output, seed_time, *, audit_pressure=False):
        self.full, self.small, self.ids, self.output = full, small, ids, output
        self.seed_time = seed_time
        self.force = fvm.ForceSampler(patch_names=["cube"], reference_area=1, reference_length=1)
        self.records = []
        self.coupler = None
        self.audit_pressure = audit_pressure
        self.pressure_previous = None
        self.pressure_previous_time = None
        if audit_pressure:
            full_centres = full.get_cell_centre_coordinates()
            self.pressure_trace = FVMVelocityInterpolator(full_centres, cKDTree(full_centres))
            self.pressure_geometry = dict(full.geo_data)
            self.pressure_geometry.update(compute_lsq_geometry(full.mesh_data, full.geo_data))
            self.face = small.get_boundary_face_centre_coordinates("numericalBoundary")
            self.normal = small.get_boundary_face_normal("numericalBoundary")
            self.area = small.get_boundary_face_area("numericalBoundary")
        self.centres = small.get_cell_centre_coordinates().copy()
        self.volumes = small.get_cell_volume().copy()
        self.near = np.max(np.abs(self.centres), axis=1) < 1
        self.query = np.sort(
            np.random.default_rng(20260913).choice(len(ids), min(len(ids), 256), replace=False)
        )

    def write(self, context):
        self.measure(context.solver)

    def measure(self, particle_solver, *, renewal_iteration=None):
        while self.full.time < particle_solver.time - 1e-12:
            self.full.advance()
        np.testing.assert_allclose(
            [self.full.time, self.small.time], particle_solver.time, rtol=0, atol=1e-11
        )
        expected = self.full.get_velocity_field()[self.ids]
        error = self.small.get_velocity_field() - expected
        particle_velocity = particle_solver.compute_velocity_at_points(
            self.centres[self.query],
            include_freestream=True,
            include_body=True,
        )
        full_force = self.force.sample(self.full)["cube"]["coeffs"]
        small_force = self.force.sample(self.small)["cube"]["coeffs"]
        row = {
            "coupling_step": int(particle_solver.step),
            "elapsed_flow_time": float(particle_solver.time),
            "physical_time": self.seed_time + float(particle_solver.time),
            "fvm_velocity_rms_over_Uinf": field_rms(error, self.volumes),
            "fvm_near_body_velocity_rms_over_Uinf": field_rms(
                error[self.near], self.volumes[self.near]
            ),
            "vpm_sampled_velocity_rms_over_Uinf": field_rms(
                particle_velocity - expected[self.query], self.volumes[self.query]
            ),
            "full_drag_coefficient": float(full_force["drag_coefficient"]),
            "hybrid_drag_coefficient": float(small_force["drag_coefficient"]),
            "drag_coefficient_difference": float(
                small_force["drag_coefficient"] - full_force["drag_coefficient"]
            ),
            "particles": int(particle_solver.particles.n_particles_total),
        }
        if renewal_iteration is not None:
            row["frozen_renewal_iteration"] = renewal_iteration
            strength = np.asarray(particle_solver.particle_vortex_strength, dtype=np.float64)
            row["particle_strength_l1"] = float(np.linalg.norm(strength, axis=1).sum())
            row["particle_strength_net"] = strength.sum(axis=0).tolist()
        if self.audit_pressure and particle_solver.step > 0:
            row.update(self.measure_pressure(particle_solver))
        self.records.append(row)
        (self.output / "comparison-history.json").write_text(
            json.dumps(self.records, indent=2) + "\n"
        )
        np.savez_compressed(
            self.output / "latest-comparison-fields.npz",
            position=self.centres,
            full_velocity=expected,
            hybrid_velocity=self.small.get_velocity_field(),
            vpm_query_ids=self.query,
            vpm_velocity=particle_velocity,
            elapsed_time=float(particle_solver.time),
        )
        print(json.dumps({"event": "matched_comparison", **row}), flush=True)

    def measure_pressure(self, particle_solver):
        """Compare pressure traces observationally; never feed them to either FVM."""
        full_gradient = compute_lsq_gradient(
            self.full.kinematic_pressure, self.full.mesh_data, self.pressure_geometry
        )[: len(self.full.get_cell_volume()), :, 0]
        reference = self.pressure_trace.sample_cell_field(self.face, full_gradient)
        has_previous = self.pressure_previous is not None
        result, velocity = particle_solver.compute_pressure_gradient_at_points(
            self.face, density=1, kinematic_viscosity=0.001,
            include_viscous=True, include_temporal=has_previous,
            include_freestream=True, include_body=True,
            particle_spacing=particle_solver.setup.viscous.particle_spacing,
            temporal_method="eulerian", velocity_previous=self.pressure_previous,
            time_step_size=(
                float(particle_solver.time) - self.pressure_previous_time
                if has_previous else particle_solver.time_step_size
            ),
            return_velocity=True,
        )
        predicted = self.coupler._kinematic_pressure_gradient_boundary_condition

        def normal_rms(value):
            return float(np.sqrt(np.average(
                np.einsum("ij,ij->i", value, self.normal)**2, weights=self.area
            )))

        row = {
            "pressure_audit_temporal_valid": has_previous,
            "pressure_reference_normal_rms": normal_rms(reference),
            "pressure_accepted_normal_rms_error": normal_rms(result["pressure_gradient"] - reference),
        }
        if predicted is not None:
            row["pressure_predicted_normal_rms_error"] = normal_rms(predicted - reference)
        for name in ("convective", "temporal", "viscous"):
            row[f"pressure_accepted_{name}_normal_rms"] = normal_rms(result[f"{name}_pressure_gradient"])
        np.savez_compressed(
            self.output / "latest-pressure-audit.npz", face=self.face, normal=self.normal,
            area=self.area, reference=reference,
            predicted=np.empty((0, 3)) if predicted is None else predicted,
            accepted_velocity=velocity, elapsed_time=float(particle_solver.time), **result,
        )
        self.pressure_previous = np.asarray(velocity, dtype=np.float64).copy()
        self.pressure_previous_time = float(particle_solver.time)
        return row


@contextmanager
def experimental_hooks(args):
    """Scope explicitly requested numerical experiments to this study run."""
    originals = []
    try:
        if args.experimental_residual_blend:
            from source.coupler import stable_renewal
            from studies.coupler_accuracy.experimental_residual_blend import blend_represented_state

            originals.append((stable_renewal, "blend_represented_state", stable_renewal.blend_represented_state))
            stable_renewal.blend_represented_state = blend_represented_state
        if args.pressure_history == "predicted":
            # Compare like-phase predictions in the temporal derivative.
            # Other boundary histories retain production resynchronization.
            from source.coupler import solver as coupler_module

            original_update = coupler_module.update_boundary_history_after_replacement

            def preserve_predicted_pressure_history(coupler, *geometry):
                saved = coupler._pressure_velocity_snapshot
                original_update(coupler, *geometry)
                coupler._pressure_velocity_snapshot = saved

            originals.append((coupler_module, "update_boundary_history_after_replacement", original_update))
            coupler_module.update_boundary_history_after_replacement = preserve_predicted_pressure_history
        yield
    finally:
        for module, name, value in reversed(originals):
            setattr(module, name, value)


def run(args):
    with ExitStack() as stack:
        stack.enter_context(experimental_hooks(args))
        if getattr(args, "mixed_convection", "native") == "outflow_linear_upwind":
            stack.enter_context(mixed_outflow_linear_upwind())
        run_trial(args)


def run_trial(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    oracle = args.oracle.resolve()
    source_report = json.loads((oracle / "cube-boundary-oracle.json").read_text())
    if (
        not source_report.get("valid_for_comparison")
        or source_report.get("spatial_dimensions") != 3
    ):
        raise ValueError("A qualified fully 3D oracle is required")
    dt = float(source_report["dt"])
    h, macro_dt = args.particle_spacing, dt * args.substeps
    if args.steps * macro_dt > 20 + 1e-12 and not args.frozen_renewals:
        raise ValueError("This short diagnostic supports at most 20 elapsed flow-time units")
    numerics = source_report["numerics"]
    if numerics["sgs"] not in {"none", "equilibrium_smagorinsky"}:
        raise ValueError("Unsupported source SGS model")
    fvm_options = {
        "turbulence": numerics["sgs"] != "none",
        "outer_correctors": int(numerics["outer_correctors"]),
    }
    eta_width = 0.0 if args.transfer_method == "projected_renewal" else 0.375
    transfer_half_width = 1.0 if args.transfer_method == "projected_renewal" else 1.25
    full_mesh = load_native_mesh(oracle / "full-native-mesh.npz")
    small_mesh = load_native_mesh(oracle / "small-native-mesh.npz")
    with np.load(oracle / "cell-and-face-map.npz", allow_pickle=False) as data:
        ids = data["cell_ids"]
    with np.load(oracle / "initial-cell-fields.npz", allow_pickle=False) as data:
        centres, velocity, pressure = (data[k].copy() for k in ("centres", "velocity", "pressure"))
        seed_time = float(data["physical_time"])
    started = time.perf_counter()
    sources = [
        Path(__file__),
        ROOT / "source/coupler/boundary.py",
        ROOT / "source/coupler/stable_renewal.py",
        ROOT / "source/coupler/vorticity_transfer.py",
        ROOT / "source/solvers/fvm/assemble/convection.py",
        ROOT / "source/solvers/fvm/assemble/momentum.py",
        ROOT / "source/solvers/fvm/core/solver.py",
        ROOT / "source/solvers/fvm/coupling/coupler_interface.py",
        ROOT / "source/solvers/fvm/fields/mixed_velocity_boundary.py",
        ROOT / "source/solvers/vpm/core/solver.py",
        ROOT / "source/solvers/vpm/physics/pressure.py",
        ROOT / "source/solvers/vpm/physics/diffusion/grid.py",
        ROOT / "studies/coupler_accuracy/experimental_residual_blend.py",
        ROOT / "studies/coupler_accuracy/experimental_mixed_convection.py",
        ROOT / "studies/coupler_accuracy/cube_boundary_oracle.py",
        oracle / "cube-boundary-oracle.json",
        oracle / "initial-cell-fields.npz",
        oracle / "full-native-mesh.npz",
        oracle / "small-native-mesh.npz",
    ]
    report = {
        "schema": "openonda-cube-coupled-trial/1",
        "status": "running",
        "spatial_dimensions": 3,
        "source_seed_time": seed_time,
        "fvm_dt": dt,
        "vpm_dt": macro_dt,
        "particle_spacing": h,
        "core_radius_ratio": 1,
        "boundary_mode": args.boundary_mode,
        "transfer_method": args.transfer_method,
        "eta_blend_width": eta_width,
        "transfer_region_bounds": [-transfer_half_width, transfer_half_width] * 3,
        "initial_vorticity_cutoff": args.initial_cutoff,
        "transfer_vorticity_cutoff": args.transfer_cutoff,
        "transfer_amplification_cap": args.transfer_amplification,
        "frozen_renewals": args.frozen_renewals,
        "experimental_residual_blend": args.experimental_residual_blend,
        "audit_pressure": args.audit_pressure,
        "pressure_history": args.pressure_history,
        "mixed_boundary_convection": getattr(args, "mixed_convection", "native"),
        "requested_coupling_steps": args.steps,
        "full_cells": full_mesh["n_cells"],
        "small_cells": small_mesh["n_cells"],
        "requested_wall_spacing": source_report["requested_wall_spacing"],
        "small_actual_bounds": source_report["small_bounds"],
        "identical_native_shared_cells": True,
        "fvm_numerics": source_report["numerics"],
        "vpm_numerics": "FMM, RK2, GBD, f32; SGS=" + numerics["sgs"],
        "sources": [hash_file(p) for p in sources],
        "limitations": [
            "Fresh seeded experiment; VPM initial velocity is approximate due to vorticity representation and pruning.",
            "Full reference has finite far boundaries; VPM uses free-space induction.",
            "Matched native FVM cells; no long-time force statistics claimed.",
        ],
    }
    for path in sources:
        if path.suffix == ".py":
            destination = output / "sources" / path.resolve().relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
    report_path = output / "cube-coupled-trial.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    observer = None
    try:
        with ExitStack() as stack:
            full = stack.enter_context(
                fvm.create_fvm_solver(
                    setup_for(full_mesh, "full", dt, round(20 / dt), **fvm_options),
                    case_dir=output / "full",
                    mesh=copy.deepcopy(full_mesh),
                )
            )
            full.set_initial_state(velocity, pressure)
            small = stack.enter_context(
                fvm.create_fvm_solver(
                    setup_for(small_mesh, "hybrid", dt, round(20 / dt), **fvm_options),
                    case_dir=output / "hybrid",
                    mesh=copy.deepcopy(small_mesh),
                )
            )
            np.testing.assert_allclose(
                full.get_cell_centre_coordinates(), centres, rtol=0, atol=1e-13
            )
            np.testing.assert_allclose(
                small.get_cell_centre_coordinates(), centres[ids], rtol=0, atol=1e-13
            )
            np.testing.assert_allclose(
                small.get_cell_volume(), full.get_cell_volume()[ids], rtol=1e-13, atol=1e-14
            )
            small.set_initial_state(velocity[ids], pressure[ids])
            observer = MatchedComparison(
                full, small, ids, output, seed_time, audit_pressure=args.audit_pressure
            )
            domain = (-5.5, 10.5, -5.5, 5.5, -5.5, 5.5)
            panel = vpm.PanelSolver(
                max_n_panels=128,
                float_dtype="f32",
                linear_solver="SCIPY",
                boundary_condition_type="NEUMANN",
                density=1,
                freestream_velocity=np.array([1, 0, 0]),
                coupling_scope="vpm_boundary_condition",
            )
            particle_case = vpm.VPMCase(
                numerics=vpm.Numerics(
                    time_step_size=macro_dt,
                    freestream_velocity=(1, 0, 0),
                    viscous=vpm.ViscousConfig.gbd(
                        particle_spacing=h,
                        core_radius_ratio=1,
                        padding=5,
                        kinematic_viscosity=0.001,
                        threshold_mode="absolute",
                        threshold=0.02 * h**3,
                        max_nodes=500000,
                    ),
                    integrator=vpm.RK2(),
                    induction=vpm.FMMInduction(),
                    turbulence=vpm.TurbulenceConfig.equilibrium_smagorinsky(
                        subgrid_kinetic_energy_coefficient=0.094,
                        subgrid_dissipation_coefficient=1.048,
                    ) if fvm_options["turbulence"] else vpm.TurbulenceConfig.dns(),
                    stabilization=vpm.StabilizationConfig.bounded_domain(domain),
                    particle_kernel="GAUSSIAN",
                    precision="f32",
                    compute_device="CPU",
                    max_n_particles=500000,
                    max_evaluation_points=500000,
                    domain_bounds=domain,
                    write_precision="f32",
                    panel_solver=panel,
                    bodies=(
                        vpm.PanelBodySetup(
                            stl=CASE / "assets/cube.stl", uid="body", reference_area=1
                        ),
                    ),
                ),
                backup=vpm.Backup(interval_steps=0, directory="solution", log_directory="solution"),
                samplers=vpm.Samplers(samples=(observer,)),
                run=vpm.RunPlan(steps=round(20 / macro_dt)),
                directory=output / "hybrid",
            )
            particle_solver = vpm.VPMSolver(particle_case)
            stack.callback(particle_solver.close)
            omega, volume = full.get_vorticity_field(), full.get_cell_volume()
            retained = np.linalg.norm(omega, axis=1) >= args.initial_cutoff
            strength = omega * volume[:, None]
            report["seed_particles"] = int(retained.sum())
            report["seed_discarded_strength_l1_fraction"] = float(
                np.linalg.norm(strength[~retained], axis=1).sum()
                / np.linalg.norm(strength, axis=1).sum()
            )
            particle_solver.add_vortex_particles(
                position=centres[retained],
                velocity=velocity[retained],
                vortex_strength=strength[retained],
                core_radius=np.full(retained.sum(), h),
                particle_volume=volume[retained],
                kinematic_viscosity=np.full(retained.sum(), 0.001),
            )
            particle_solver.refresh_boundary_element_solution()
            observer.measure(
                particle_solver, renewal_iteration=0 if args.frozen_renewals else None
            )
            coupler = coupling.create_coupler(
                small,
                particle_solver,
                coupling.CouplerSetup(
                    transfer_method=args.transfer_method,
                    transfer_region_bounds=(-transfer_half_width, transfer_half_width) * 3,
                    eta_blend_width=eta_width,
                    transfer_vorticity_cutoff=args.transfer_cutoff,
                    transfer_amplification_cap=args.transfer_amplification,
                    boundary_condition_mode=args.boundary_mode,
                    fvm_consistency_width=0.0,
                    backup_interval_steps=0,
                ),
            )
            observer.coupler = coupler
            if args.frozen_renewals:
                # Freeze the *full* native donor state, including its gradients,
                # so the artificial small boundary cannot contaminate this test.
                # No pressure solve, advection, stretching, GBD or clock advance.
                coupler.initialize()
                donor_velocity = full.get_velocity_field()[ids].copy()
                donor_gradient = full.get_velocity_gradient_field()[ids].copy()
                for iteration in range(1, args.frozen_renewals + 1):
                    coupler.vorticity_transfer.transfer(
                        particle_solver, donor_velocity, donor_gradient
                    )
                    particle_solver.refresh_boundary_element_solution()
                    observer.measure(particle_solver, renewal_iteration=iteration)
                np.testing.assert_array_equal(full.get_velocity_field()[ids], donor_velocity)
                np.testing.assert_array_equal(
                    full.get_velocity_gradient_field()[ids], donor_gradient
                )
                assert full.time == small.time == particle_solver.time == 0
            else:
                coupler.run(max_coupling_steps=args.steps, backup_at_stop=True)
        report["status"] = "complete"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["wall_time_seconds"] = time.perf_counter() - started
        report["comparison"] = observer.records if observer is not None else []
        report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--substeps", type=int, default=5)
    parser.add_argument("--particle-spacing", type=float, default=0.125)
    parser.add_argument("--initial-cutoff", type=float, default=0.02)
    parser.add_argument("--audit-pressure", action="store_true")
    parser.add_argument("--mixed-convection", choices=("native", "outflow_linear_upwind"),
                        default="native", help="Scoped experimental mixed-boundary convection")
    parser.add_argument(
        "--pressure-history", choices=("accepted", "predicted"), default="accepted",
        help="Study-only choice of temporal history; production uses accepted fields.",
    )
    parser.add_argument("--transfer-cutoff", type=float, default=0.05)
    parser.add_argument(
        "--transfer-method", choices=("buffered_m4_renewal", "projected_renewal"),
        default="buffered_m4_renewal",
    )
    parser.add_argument("--transfer-amplification", type=float, default=1.8)
    parser.add_argument(
        "--frozen-renewals", type=int, default=0,
        help="Apply only transfer this many times to fixed 3D FVM fields; no time advance.",
    )
    parser.add_argument(
        "--experimental-residual-blend", action="store_true",
        help="Reproduce the unqualified residual-blend candidate; not a production setting.",
    )
    parser.add_argument(
        "--boundary-mode",
        choices=("vorticity_mixed", "vorticity_mixed_pressure_gradient"),
        default="vorticity_mixed",
    )
    args = parser.parse_args()
    if args.pressure_history == "predicted" and args.boundary_mode != "vorticity_mixed_pressure_gradient":
        parser.error("The predicted pressure-history experiment requires the pressure-gradient mode")
    if (
        args.steps <= 0
        or args.substeps <= 0
        or args.particle_spacing <= 0
        or args.initial_cutoff < 0
        or args.transfer_cutoff < 0
        or args.transfer_amplification < 1
        or args.frozen_renewals < 0
    ):
        parser.error(
            "steps, substeps and particle spacing must be positive; cutoff must be nonnegative"
        )
    run(args)
