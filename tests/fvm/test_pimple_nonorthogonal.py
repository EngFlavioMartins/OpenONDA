"""Integrated PIMPLE non-orthogonal-correction regression."""

import contextlib
import io

import numpy as np

from source.solvers.fvm import (
    BoundaryConfig,
    DiscretizationConfig,
    FVMSetup,
    FVMSolver,
    LinearSolverConfig,
    PimpleControl,
    TimeConfig,
    TransportConfig,
)

from ._structured_mesh import structured_box


def test_nonorthogonal_sweep_returns_equation_residuals(tmp_path):
    mesh = structured_box(4, 3, 2)
    mesh["vertex_position"][:, 0] += 0.25 * mesh["vertex_position"][:, 1]
    params_schemes = DiscretizationConfig(convection_scheme="upwind")
    params_linear = LinearSolverConfig(linear_solver="spsolve")
    params_pimple = PimpleControl(n_correctors=1, n_orthogonal_correctors=1)
    config = FVMSetup(
        case_name="skewed_pimple",
        time=TimeConfig.transient(time_step_size=0.01, duration=0.01, output_interval_steps=100),
        schemes=params_schemes,
        linear=params_linear,
        pimple=params_pimple,
        transport=TransportConfig(density=1.0, kinematic_viscosity=0.01),
        boundaries=[
            BoundaryConfig.inlet("xmin", [1.0, 0.0, 0.0]),
            BoundaryConfig.outlet("xmax", 0.0),
            BoundaryConfig.wall("ymin"),
            BoundaryConfig.wall("ymax"),
            BoundaryConfig.wall("zmin"),
            BoundaryConfig.wall("zmax"),
        ],
        initial_velocity=[1.0, 0.0, 0.0],
        initial_kinematic_pressure=0.0,
    )

    with contextlib.redirect_stdout(io.StringIO()):
        solver = FVMSolver(config, str(tmp_path), mesh_data=mesh)
        residuals = solver.solve_pimple(0.01)

    for key in (
        "kinematic_pressure",
        "velocity",
        "initial_kinematic_pressure",
        "velocity_increment",
        "velocity_x",
        "velocity_y",
        "velocity_z",
    ):
        assert key in residuals
        assert np.isfinite(residuals[key])
    assert residuals["kinematic_pressure"] < 1e-10
    assert residuals["velocity"] < 1e-10
    assert np.all(np.isfinite(solver.velocity[: mesh["n_cells"]]))
