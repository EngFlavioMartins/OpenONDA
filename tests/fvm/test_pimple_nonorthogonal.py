"""Integrated PIMPLE non-orthogonal-correction regression."""

import contextlib
import io

import numpy as np

from source.solvers.FVM import (
    BoundaryConfig,
    FVMSetup,
    LinearSolverConfig,
    PimpleControl,
    SchemesConfig,
    Solver,
    TimeConfig,
    TransportConfig,
)

from ._structured_mesh import structured_box


def test_nonorthogonal_sweep_returns_equation_residuals(tmp_path):
    mesh = structured_box(4, 3, 2)
    mesh["points"][:, 0] += 0.25 * mesh["points"][:, 1]
    params_schemes = SchemesConfig(convection_scheme="upwind")
    params_linear = LinearSolverConfig(linear_solver="spsolve")
    params_pimple = PimpleControl(n_correctors=1, n_orthogonal_correctors=1)
    config = FVMSetup(
        case_name="skewed_pimple",
        time=TimeConfig.transient(dt=0.01, duration=0.01, write_interval=100),
        schemes=params_schemes,
        linear=params_linear,
        pimple=params_pimple,
        transport=TransportConfig(density=1.0, nu=0.01),
        boundaries=[
            BoundaryConfig.inlet("xmin", [1.0, 0.0, 0.0]),
            BoundaryConfig.outlet("xmax", 0.0),
            BoundaryConfig.wall("ymin"),
            BoundaryConfig.wall("ymax"),
            BoundaryConfig.wall("zmin"),
            BoundaryConfig.wall("zmax"),
        ],
        initial_velocity=[1.0, 0.0, 0.0],
        initial_p=0.0,
    )

    with contextlib.redirect_stdout(io.StringIO()):
        solver = Solver(config, str(tmp_path), mesh_data=mesh)
        residuals = solver.solve_pimple(0.01)

    for key in ("p", "U", "p_initial", "U_increment", "U_x", "U_y", "U_z"):
        assert key in residuals
        assert np.isfinite(residuals[key])
    assert residuals["p"] < 1e-10
    assert residuals["U"] < 1e-10
    assert np.all(np.isfinite(solver.U[: mesh["n_elements"]]))
