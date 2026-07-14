"""Integrated PIMPLE non-orthogonal-correction regression."""

import contextlib
import io

import numpy as np

from source.solvers.FVM import (
    BoundaryConfig,
    FVMConfig,
    Solver,
    SolverParams,
    TimeConfig,
    TransportConfig,
)

from ._structured_mesh import structured_box


def test_nonorthogonal_sweep_returns_equation_residuals(tmp_path):
    mesh = structured_box(4, 3, 2)
    mesh["points"][:, 0] += 0.25 * mesh["points"][:, 1]
    params = SolverParams.pimple(
        n_correctors=1,
        n_non_orthogonal=1,
        linear_solver="spsolve",
        convection_scheme="upwind",
    )
    config = FVMConfig(
        case_name="skewed_pimple",
        time=TimeConfig.transient(dt=0.01, duration=0.01, write_interval=100),
        solver=params,
        transport=TransportConfig(density=1.0, nu=0.01),
        boundaries=[
            BoundaryConfig.inlet("xmin", [1.0, 0.0, 0.0]),
            BoundaryConfig.outlet("xmax", 0.0),
            BoundaryConfig.wall("ymin"),
            BoundaryConfig.wall("ymax"),
            BoundaryConfig.wall("zmin"),
            BoundaryConfig.wall("zmax"),
        ],
        initial_U=[1.0, 0.0, 0.0],
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
