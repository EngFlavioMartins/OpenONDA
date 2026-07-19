"""Slip / symmetry velocity boundary condition.

A uniform stream parallel to slip walls is an exact solution of the
incompressible Navier-Stokes equations: the slip planes must add no shear,
no convective flux, and no mass flux.  If any operator treats the slip patch
as a wall (or leaks flux through it), the uniform profile distorts within a
few steps — this is the production configuration used by the boundary-layer
and square-cylinder tutorials.
"""

import contextlib
import io

import numpy as np

from source.solvers.FVM import (
    BoundaryConfig,
    ForcesConfig,
    FVMConfig,
    LinearSolverConfig,
    PimpleControl,
    SchemesConfig,
    Solver,
    TimeConfig,
    TransportConfig,
)

from ._structured_mesh import structured_box

U_INF = 1.0


def _slip_channel_solver(tmp_path):
    mesh = structured_box(8, 6, 1, lx=2.0, ly=1.0, lz=0.1)
    sp_schemes = SchemesConfig(convection_scheme="central")
    sp_linear = LinearSolverConfig(linear_solver="spsolve")
    sp_pimple = PimpleControl(n_correctors=2)
    cfg = FVMConfig(
        case_name="slip_channel",
        time=TimeConfig(delta_t=0.05, end_time=1.0, write_interval=10**9),
        schemes=sp_schemes,
        linear=sp_linear,
        pimple=sp_pimple,
        forces=ForcesConfig(),
        transport=TransportConfig(density=1.0, nu=0.01),
        boundaries=[
            BoundaryConfig.inlet("xmin", [U_INF, 0.0, 0.0]),
            BoundaryConfig.outlet("xmax", p=0.0),
            BoundaryConfig(name="ymin", type_U="slip", type_p="zeroGradient"),
            BoundaryConfig(name="ymax", type_U="slip", type_p="zeroGradient"),
            BoundaryConfig.empty("zmin"),
            BoundaryConfig.empty("zmax"),
        ],
        initial_U=[U_INF, 0.0, 0.0],
        initial_p=0.0,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        solver = Solver(cfg, case_dir=str(tmp_path), mesh_data=mesh)
        solver.auto_write = False
    return solver, mesh


def test_uniform_stream_is_preserved_by_slip_walls(tmp_path):
    solver, mesh = _slip_channel_solver(tmp_path)
    n = mesh["n_elements"]
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(5):
            solver.evolve()
    U = solver.U[:n]
    assert np.allclose(U[:, 0], U_INF, atol=1e-8)
    assert np.allclose(U[:, 1], 0.0, atol=1e-8)
    assert solver.continuity_max < 1e-10


def test_slip_ghosts_are_tangential(tmp_path):
    solver, mesh = _slip_channel_solver(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        solver.evolve()
    n = mesh["n_elements"]
    n_interior = mesh["n_interior_faces"]
    for patch in solver.boundaries:
        if patch["name"] not in ("ymin", "ymax"):
            continue
        start = n + (patch["startFace"] - n_interior)
        ghosts = solver.U[start : start + patch["nFaces"]]
        # y is the slip-plane normal on these patches.
        assert np.allclose(ghosts[:, 1], 0.0, atol=1e-12)
