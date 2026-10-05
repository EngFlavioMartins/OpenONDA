"""Tiny native periodic FVM probe; invoked only by its temporary-case test."""

import json
from pathlib import Path
import sys

import numpy as np

from openonda import fvm
from openonda.runtime import RunConfig
from source.solvers.fvm.fields.gradients import compute_lsq_gradient
from source.solvers.fvm.mesh.rectilinear import box_mesh_3d


def main():
    directory, cores = Path(sys.argv[1]), int(sys.argv[2])
    RunConfig(cpu_cores=cores, parallel_mode="mpi").ensure_runtime(__file__)

    def mesh():
        result = box_mesh_3d(np.linspace(0, 1, 7), np.linspace(0, 1, 6), np.array([0.0, 1.0]))
        for patch, name in zip(
            result["boundary"], ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"), strict=True
        ):
            patch["name"] = name
        return result

    setup = fvm.FVMSetup(
        case_name="single_layer_cyclic",
        cores=cores,
        logging=fvm.LoggingConfig(console=False),
        backup=fvm.BackupConfig(schedule=None, write_at_end=False),
        time=fvm.TimeConfig(
            time_step_size=0.008, end_time=0.016, output_schedule=fvm.RunSchedule(every_n_steps=100)
        ),
        schemes=fvm.DiscretizationConfig(
            convection_scheme="limitedLinear", gradient_scheme="lsq", time_scheme="backward"
        ),
        linear=fvm.LinearSolverConfig(
            linear_solver="spsolve",
            pressure_solver="spsolve",
            pressure_tolerance=1e-11,
            momentum_tolerance=1e-11,
        ),
        transport=fvm.TransportConfig(density=1.0, kinematic_viscosity=1 / 150),
        boundaries=[
            fvm.BoundaryConfig(
                name=name,
                velocity_type="fixedValue",
                velocity_value=[1, 0, 0],
                pressure_type="fixedFluxPressure",
            )
            for name in ("xmin", "xmax", "ymin", "ymax")
        ]
        + [fvm.BoundaryConfig.cyclic("zmin", "zmax"), fvm.BoundaryConfig.cyclic("zmax", "zmin")],
        initial_velocity=[1, 0, 0],
    )

    def exact(points):
        return np.column_stack((1 + 0.1 * points[:, 1], np.zeros((len(points), 2))))

    with fvm.create_fvm_solver(setup, case_dir=directory, mesh=mesh) as solver:
        solver.auto_write = False
        n = solver.mesh_data["n_cells"]
        assert n == 30
        assert solver.parallel.size == cores
        if cores > 1:
            assert solver.parallel.mode == "petsc_replicated"
        for patch in solver.boundaries:
            if patch["name"].startswith("z"):
                continue
            faces = np.arange(patch["start_face"], patch["start_face"] + patch["n_faces"])
            solver.set_dirichlet_velocity_boundary_condition_vec(
                exact(solver.geo_data["face_centre"][faces]), patch["name"]
            )
        expected = exact(solver.geo_data["cell_centre"][:n])
        solver.set_initial_state(expected, np.zeros(n))
        gradient = compute_lsq_gradient(solver.velocity, solver.mesh_data, solver.geo_data)
        np.testing.assert_allclose(gradient[:n, 2], 0, atol=1e-13)
        faces = np.flatnonzero(solver.mesh_data["boundary_neighbour_cell"] >= 0)
        np.testing.assert_array_equal(
            solver.mesh_data["boundary_neighbour_cell"][faces], solver.mesh_data["owners"][faces]
        )
        for _ in range(2):
            solver.advance()
        error = float(np.max(np.abs(solver.velocity[:n] - expected)))
        continuity = float(solver.last_diagnostics.max_continuity_error)
        assert error < 1e-9, error
        assert continuity < 1e-9, continuity
        report = {
            "status": "passed",
            "rank": solver.parallel.rank,
            "cores": cores,
            "cells": n,
            "velocity_error": error,
            "continuity_error": continuity,
        }
        (directory / f"rank-{solver.parallel.rank}.json").write_text(json.dumps(report))


if __name__ == "__main__":
    main()
