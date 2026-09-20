"""Two-rank planar qualification helper, with no obstacle or cylinder flow.

Run from the repository with the MPI/PETSc environment's Python::

    python tests/coupler/_planar_mpi_smoke.py /tmp/unique-planar-smoke

The public factory launches two ranks. The output directory must be unused.
The companion pytest test applies a timeout and checks both shutdown markers.
"""

import json
from pathlib import Path
import sys

import numpy as np

from openonda import coupler, fvm, vpm
from source.solvers.fvm.mesh.rectilinear import coupling_box_mesh


def main(directory):
    h, span = 0.125, 2.0
    flow = fvm.FVMSetup(
        case_name="planar_mpi_smoke",
        cores=2,
        time=fvm.TimeConfig(
            time_step_size=0.01,
            end_time=0.02,
            output_schedule=fvm.RunSchedule(final_only=True),
        ),
        transport=fvm.TransportConfig(kinematic_viscosity=0.01),
        boundaries=[
            fvm.BoundaryConfig(
                name="numericalBoundary",
                velocity_type="fixedValue",
                velocity_value=[1, 0, 0],
                pressure_type="fixedFluxPressure",
            ),
            fvm.BoundaryConfig.slip("zmin"),
            fvm.BoundaryConfig.slip("zmax"),
        ],
        initial_velocity=[1, 0, 0],
    )
    particles = vpm.VPMCase(
        directory=directory,
        numerics=vpm.Numerics(
            induction=vpm.PlanarInduction(span=span),
            compute_device="CPU",
            time_step_size=0.01,
            freestream_velocity=(1, 0, 0),
            max_n_particles=5000,
            max_evaluation_points=5000,
            viscous=vpm.ViscousConfig.gbd(
                particle_spacing=h,
                gbd_grid_spacing=h,
                kinematic_viscosity=0.01,
                threshold_mode="absolute",
                threshold=1e-9,
                core_radius_ratio=1,
            ),
            verbose=False,
        ),
        run=vpm.RunPlan(steps=2, final_backup=False, initial_samples=False),
        backup=vpm.Backup(interval_steps=0),
    )
    policy = coupler.CouplerSetup(
        freestream_velocity=[1, 0, 0],
        transfer_method="buffered_m4_renewal",
        transfer_region_bounds=(-0.375, 0.375, -0.375, 0.375, -0.875, 0.875),
        eta_blend_width=0.25,
        vpm_only_width=0,
        transfer_vorticity_cutoff=0.001,
        boundary_condition_mode="vorticity_mixed",
        interface_iterations=1,
        backup_interval_steps=0,
    )

    def seed(solver):
        solver.add_vortex_particles(
            position=np.array([[0.8, -0.25, 0], [0.8, 0.25, 0]]),
            velocity=np.tile([1.0, 0, 0], (2, 1)),
            vortex_strength=np.array([[0, 0, 0.001 * span], [0, 0, -0.001 * span]]),
            core_radius=np.full(2, h),
            particle_volume=np.full(2, h * h * span),
            kinematic_viscosity=np.full(2, 0.01),
        )

    def inspect(solver, velocity, coordinates):
        # This callback is root-owned, but its fields were gathered collectively.
        assert velocity.shape == coordinates.shape == (128, 3)
        assert np.isfinite(velocity).all()
        xy, groups, counts = np.unique(
            np.round(coordinates[:, :2], 12), axis=0, return_inverse=True, return_counts=True
        )
        assert len(xy) == 16
        np.testing.assert_array_equal(counts, 8)
        means = np.column_stack(
            [np.bincount(groups, weights=velocity[:, axis]) / counts for axis in range(3)]
        )
        global_variation = float(np.max(np.abs(velocity - means[groups])))
        global_span_velocity = float(np.max(np.abs(velocity[:, 2])))
        assert global_variation < 1e-3
        assert global_span_velocity < 1e-3
        assert solver.particles.n_particles_total > 2
        assert np.isfinite(solver.particle_velocity).all()
        assert np.isfinite(solver.particle_vortex_strength).all()
        np.testing.assert_array_equal(solver.particle_position[:, 2], 0)
        np.testing.assert_allclose(solver.particle_volume, h * h * span)
        # Surviving nonzero exterior vorticity prevents a vacuous empty-wake pass.
        exterior = solver.particle_position[:, 0] > 0.5
        assert np.any(exterior)
        exterior_strength = float(np.sum(np.abs(solver.particle_vortex_strength[exterior, 2])))
        assert exterior_strength > 1e-4
        records = [
            json.loads(line)
            for line in (directory / "solution/coupler_diagnostics.jsonl").read_text().splitlines()
        ]
        assert len(records) == 2
        for record in records:
            metrics = record["planar_spanwise_consistency"]
            assert np.isfinite(list(metrics.values())).all()
            assert metrics["span_variation_max"] < 1e-3 * metrics["velocity_scale"]
            assert metrics["span_velocity_max"] < 1e-3 * metrics["velocity_scale"]
        return {
            "ranks": 2,
            "steps": 2,
            "time": 0.02,
            "global_cells": len(velocity),
            "xy_stacks": len(xy),
            "layers_per_stack": int(counts[0]),
            "particles": solver.particles.n_particles_total,
            "particle_volume": h * h * span,
            "exterior_absolute_strength": exterior_strength,
            "global_span_variation_max": global_variation,
            "global_span_velocity_max": global_span_velocity,
            "span_diagnostics": [record["planar_spanwise_consistency"] for record in records],
        }

    with coupler.create_coupler(
        flow,
        particles,
        policy,
        mesh=lambda: coupling_box_mesh(
            (-0.5, 0.5, -0.5, 0.5, -1, 1), 0.25, separate_outer=("zmin", "zmax")
        ),
        case_dir=directory,
        require_empty_output=True,
    ) as driver:
        driver.initialize()
        native = driver.fvm_solver
        assert native.parallel.size == 2
        assert native.parallel.mode == "petsc_partitioned"
        assert (driver.vpm_solver is not None) == native.parallel.is_root
        driver.apply_vpm(seed)
        assert driver.run(max_coupling_steps=2, backup_at_stop=False) == 2
        assert np.isclose(native.time, 0.02)
        velocity = native.get_velocity_field()
        coordinates = native.get_cell_centre_coordinates()
        report = driver.apply_vpm(inspect, velocity, coordinates)
        rank = native.parallel.rank
    assert driver._closed
    assert native._closed
    assert not native.algorithm._partitioned_linear_workspaces
    (directory / f"rank-{rank}.json").write_text(
        json.dumps({"rank": rank, "steps": 2, "time": 0.02, "closed": True}) + "\n"
    )
    if rank == 0:
        (directory / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
        print("PLANAR_MPI_QUALIFIED " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
