"""Isolated CPU one-layer periodic FVM/infinite-filament qualification."""

import json
from pathlib import Path
import sys

import numpy as np

from openonda import coupler, fvm, vpm
from source.solvers.fvm.mesh.rectilinear import box_mesh_3d


def main(directory, ranks):
    h = 0.125
    flow = fvm.FVMSetup(
        case_name="planar_single_layer",
        cores=ranks,
        time=fvm.TimeConfig(
            time_step_size=0.01, end_time=0.02, output_schedule=fvm.RunSchedule(final_only=True)
        ),
        transport=fvm.TransportConfig(kinematic_viscosity=0.01),
        boundaries=[
            fvm.BoundaryConfig(
                name="numericalBoundary",
                velocity_type="fixedValue",
                velocity_value=[1, 0, 0],
                pressure_type="fixedFluxPressure",
            ),
            fvm.BoundaryConfig.cyclic("zmin", "zmax"),
            fvm.BoundaryConfig.cyclic("zmax", "zmin"),
        ],
        initial_velocity=[1, 0, 0],
    )
    particles = vpm.VPMCase(
        directory=directory,
        numerics=vpm.Numerics(
            induction=vpm.PlanarInduction(span=1, plane_z=0.5),
            compute_device="CPU",
            time_step_size=0.01,
            freestream_velocity=(1, 0, 0),
            max_n_particles=5000,
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
    settings = coupler.CouplerSetup(
        freestream_velocity=[1, 0, 0],
        transfer_region_bounds=(-0.375, 0.375, -0.375, 0.375, 0, 1),
        eta_blend_width=0.25,
        vpm_only_width=0,
        transfer_vorticity_cutoff=0.0001,
        interface_iterations=1,
        backup_interval_steps=0,
    )

    def seed(solver):
        solver.add_vortex_particles(
            position=np.array([[0.8, -0.25, 0.5], [0.8, 0.25, 0.5]]),
            velocity=np.tile([1.0, 0, 0], (2, 1)),
            vortex_strength=np.array([[0, 0, 0.001], [0, 0, -0.001]]),
            core_radius=np.full(2, h),
            particle_volume=np.full(2, h * h),
            kinematic_viscosity=np.full(2, 0.01),
        )

    def inspect(solver, velocity, coordinates):
        assert velocity.shape == coordinates.shape == (16, 3)
        assert np.isfinite(velocity).all()
        _, groups, counts = np.unique(
            np.round(coordinates[:, :2], 11), axis=0, return_inverse=True, return_counts=True
        )
        np.testing.assert_array_equal(counts, 1)
        np.testing.assert_array_equal(coordinates[:, 2], 0.5)
        means = np.column_stack(
            [np.bincount(groups, weights=velocity[:, axis]) / counts for axis in range(3)]
        )
        np.testing.assert_array_equal(solver.particle_position[:, 2], 0.5)
        np.testing.assert_array_equal(solver.particle_vortex_strength[:, :2], 0)
        np.testing.assert_allclose(solver.particle_volume, h * h)
        assert np.isfinite(solver.particle_velocity).all()
        assert np.isfinite(solver.particle_vortex_strength).all()
        exterior = solver.particle_position[:, 0] > 0.5
        records = [
            json.loads(line)
            for line in (directory / "solution/coupler_diagnostics.jsonl").read_text().splitlines()
        ]
        assert len(records) == 2
        for record in records:
            metrics = record["planar_spanwise_consistency"]
            assert metrics["span_variation_max"] == 0
            assert metrics["span_velocity_max"] < 1e-3 * metrics["velocity_scale"]
        return {
            "ranks": ranks,
            "steps": 2,
            "time": 0.02,
            "global_cells": len(velocity),
            "layers_per_stack": int(counts[0]),
            "particle_volume": h * h,
            "particles": solver.particles.n_particles_total,
            "exterior_absolute_strength": float(
                np.abs(solver.particle_vortex_strength[exterior, 2]).sum()
            ),
            "span_velocity_max": float(np.max(np.abs(velocity[:, 2]))),
            "span_variation_max": float(np.max(np.abs(velocity - means[groups]))),
            "span_diagnostics": [record["planar_spanwise_consistency"] for record in records],
        }

    with coupler.create_coupler(
        flow,
        particles,
        settings,
        mesh=lambda: box_mesh_3d(
            np.linspace(-0.5, 0.5, 5),
            np.linspace(-0.5, 0.5, 5),
            np.array([0.0, 1.0]),
            merge_outer_patch="numericalBoundary",
            separate_outer=("zmin", "zmax"),
        ),
        case_dir=directory,
        require_empty_output=True,
    ) as driver:
        driver.initialize()
        native = driver.fvm_solver
        assert native.parallel.size == ranks
        assert (driver.vpm_solver is not None) == native.parallel.is_root
        driver.apply_vpm(seed)
        assert driver.run(max_coupling_steps=2, backup_at_stop=False) == 2
        velocity = native.get_velocity_field()
        coordinates = native.get_cell_centre_coordinates()
        report = driver.apply_vpm(inspect, velocity, coordinates)
        rank = native.parallel.rank
    assert driver._closed and native._closed
    assert not native.algorithm._partitioned_linear_workspaces
    (directory / f"rank-{rank}.json").write_text(
        json.dumps({"rank": rank, "steps": 2, "time": 0.02, "closed": True}) + "\n"
    )
    if rank == 0:
        (directory / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), int(sys.argv[2]))
