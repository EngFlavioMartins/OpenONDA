"""Tiny native periodic FVM probe; invoked only by its temporary-case test."""

import json
from pathlib import Path
import sys

import numpy as np

from openonda import fvm
from openonda.fvm.fields.gradients import compute_lsq_gradient
from openonda.fvm.mesh.rectilinear import box_mesh_3d
from openonda.runtime import RunConfig


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

    case = fvm.FVMCase(
        name="single_layer_cyclic",
        directory=directory,
        numerics=fvm.Numerics(
            time_step_size=0.008,
            schemes=fvm.DiscretizationConfig(
                convection_scheme="limited_linear", gradient_scheme="lsq", time_scheme="backward"
            ),
            linear=fvm.LinearSolverConfig(
                linear_solver="spsolve",
                pressure_solver="spsolve",
                pressure_tolerance=1e-11,
                momentum_tolerance=1e-11,
            ),
            transport=fvm.TransportConfig(density=1.0, kinematic_viscosity=1 / 150),
        ),
        initial_conditions=fvm.InitialFields(velocity=(1, 0, 0)),
        run=fvm.RunPlan(end_time=0.016),
        backup=fvm.Backup(write_at_end=False),
        logging=fvm.LoggingConfig(console=False),
        mesh=mesh,
        boundaries=tuple(
            fvm.BoundaryConfig(
                name=name,
                velocity_type="fixed_value",
                velocity_value=(1, 0, 0),
                pressure_type="fixed_flux_pressure",
            )
            for name in ("xmin", "xmax", "ymin", "ymax")
        )
        + (fvm.BoundaryConfig.cyclic("zmin", "zmax"), fvm.BoundaryConfig.cyclic("zmax", "zmin")),
        cores=cores,
    )

    def exact(points):
        return np.column_stack((1 + 0.1 * points[:, 1], np.zeros((len(points), 2))))

    with fvm.FVMSolver(case) as solver:
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
            solver.set_dirichlet_velocity_boundary_condition(
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
