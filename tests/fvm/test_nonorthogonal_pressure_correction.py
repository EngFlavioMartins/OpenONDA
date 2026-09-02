"""Regression coverage for non-orthogonal Rhie--Chow pressure fluxes."""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from openonda import fvm
from source.solvers.fvm.core.solver import FVMSolver
from source.solvers.fvm.mesh.cartesian import structured_box
from source.solvers.fvm.solve.simple_solver import (
    _pressure_interior_flux_scalar,
    _pressure_interior_flux_vector,
    _process_boundary_faces_jit,
)


@pytest.mark.parametrize(
    ("coefficient", "velocity_h_over_a", "kernel"),
    [
        (
            np.ones(2),
            np.tile([2.0, 3.0, 0.0], (2, 1)),
            _pressure_interior_flux_scalar,
        ),
        (
            np.tile([1.0, 2.0, 3.0], (2, 1)),
            np.tile([2.0, 6.0, 0.0], (2, 1)),
            _pressure_interior_flux_vector,
        ),
    ],
)
def test_linear_pressure_field_cancels_rhie_chow_flux_on_a_nonorthogonal_face(
    coefficient: np.ndarray,
    velocity_h_over_a: np.ndarray,
    kernel,
) -> None:
    """H/A and pressure fluxes cancel exactly for a stationary linear field."""
    flux = np.zeros(1)
    kernel(
        np.array([0], dtype=np.int32),
        np.array([1], dtype=np.int32),
        np.array([0.5]),
        np.array([[1.0, 1.0, 0.0]]),
        np.array([[1.0, 0.0, 0.0]]),
        coefficient,
        velocity_h_over_a,
        np.tile([2.0, 3.0, 0.0], (2, 1)),
        np.array([0.0, 2.0]),
        np.array([1.0]),
        np.empty(0),
        flux,
    )

    assert flux[0] == pytest.approx(0.0, abs=1.0e-9)


def test_linear_pressure_field_cancels_boundary_rhie_chow_flux() -> None:
    _, _, flux = _process_boundary_faces_jit(
        1,
        0,
        1,
        np.array([0], dtype=np.int32),
        np.array([[1.0, 1.0, 0.0]]),
        np.array([[1.0, 0.0, 0.0]]),
        np.zeros((2, 3)),
        np.ones(1),
        np.ones(1),
        np.ones(1),
        np.array([[2.0, 3.0, 0.0]]),
        np.array([0.0, 2.0]),
        np.array([1], dtype=np.int32),
        np.array([2.0]),
        np.array([0], dtype=np.int32),
        np.array([1.0]),
    )

    assert flux[0] == pytest.approx(0.0, abs=1.0e-9)


def test_nonorthogonal_pressure_sweep_remains_bounded_on_a_sheared_mesh(tmp_path) -> None:
    mesh = structured_box(3, 3, 1)
    mesh["vertex_position"][:, 0] += 0.45 * mesh["vertex_position"][:, 1]
    setup = fvm.FVMSetup(
        case_name="nonorthogonal_pressure_regression",
        logging=fvm.LoggingConfig(console=False),
        backup=fvm.BackupConfig(schedule=None),
        time=fvm.TimeConfig(
            time_step_size=0.002,
            end_time=0.006,
            output_schedule=fvm.RunSchedule(every_n_steps=100),
        ),
        schemes=fvm.DiscretizationConfig(
            convection_scheme="upwind",
            gradient_scheme="lsq",
        ),
        linear=fvm.LinearSolverConfig(linear_solver="spsolve"),
        pimple=fvm.PimpleControl(
            n_correctors=2,
            n_outer_correctors=2,
            n_orthogonal_correctors=1,
            velocity_relaxation=0.7,
            pressure_relaxation=0.3,
        ),
        transport=fvm.TransportConfig(density=1.0, kinematic_viscosity=0.02),
        boundaries=[
            fvm.BoundaryConfig.inlet("xmin", [0.5, 0.0, 0.0]),
            fvm.BoundaryConfig.outlet("xmax"),
            fvm.BoundaryConfig.wall("ymin"),
            fvm.BoundaryConfig.wall("ymax"),
            fvm.BoundaryConfig.slip("zmin"),
            fvm.BoundaryConfig.slip("zmax"),
        ],
        initial_velocity=[0.2, 0.0, 0.0],
    )

    with contextlib.redirect_stdout(io.StringIO()):
        solver = FVMSolver(setup, str(tmp_path), mesh_data=mesh)
        solver.auto_write = False
        try:
            for _ in range(3):
                solver.advance()
        finally:
            solver.close()

    assert np.all(np.isfinite(solver.velocity))
    assert np.max(np.linalg.norm(solver.velocity, axis=1)) < 2.0
    assert solver.last_diagnostics.max_continuity_error < 1.0e-6
