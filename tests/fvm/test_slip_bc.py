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
import pytest

from source.solvers.FVM import (
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

U_INF = 1.0


def _slip_channel_solver(tmp_path):
    mesh = structured_box(8, 6, 1, lx=2.0, ly=1.0, lz=0.1)
    sp_schemes = DiscretizationConfig(convection_scheme="central")
    sp_linear = LinearSolverConfig(linear_solver="spsolve")
    sp_pimple = PimpleControl(n_correctors=2)
    cfg = FVMSetup(
        case_name="slip_channel",
        time=TimeConfig(time_step_size=0.05, end_time=1.0, output_interval_steps=10**9),
        schemes=sp_schemes,
        linear=sp_linear,
        pimple=sp_pimple,
        transport=TransportConfig(density=1.0, kinematic_viscosity=0.01),
        boundaries=[
            BoundaryConfig.inlet("xmin", [U_INF, 0.0, 0.0]),
            BoundaryConfig.outlet("xmax", kinematic_pressure=0.0),
            BoundaryConfig(name="ymin", velocity_type="slip", pressure_type="zeroGradient"),
            BoundaryConfig(name="ymax", velocity_type="slip", pressure_type="zeroGradient"),
            BoundaryConfig.empty("zmin"),
            BoundaryConfig.empty("zmax"),
        ],
        initial_velocity=[U_INF, 0.0, 0.0],
        initial_kinematic_pressure=0.0,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        solver = FVMSolver(cfg, case_dir=str(tmp_path), mesh_data=mesh)
        solver.auto_write = False
    return solver, mesh


def test_uniform_stream_is_preserved_by_slip_walls(tmp_path):
    solver, mesh = _slip_channel_solver(tmp_path)
    n = mesh["n_cells"]
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(5):
            solver.advance()
    velocity = solver.velocity[:n]
    assert np.allclose(velocity[:, 0], U_INF, atol=1e-8)
    assert np.allclose(velocity[:, 1], 0.0, atol=1e-8)
    assert solver.continuity_max < 1e-10


def test_slip_ghosts_are_tangential(tmp_path):
    solver, mesh = _slip_channel_solver(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        solver.advance()
    n = mesh["n_cells"]
    n_interior = mesh["n_interior_faces"]
    for patch in solver.boundaries:
        if patch["name"] not in ("ymin", "ymax"):
            continue
        start = n + (patch["start_face"] - n_interior)
        ghosts = solver.velocity[start : start + patch["n_faces"]]
        # y is the slip-plane normal on these patches.
        assert np.allclose(ghosts[:, 1], 0.0, atol=1e-12)


@pytest.mark.parametrize(
    "face_sf",
    [
        np.array([[2.0, -1.0, 3.0], [-4.0, 2.0, 1.0], [0.0, 0.0, 0.0]]),
    ],
)
def test_vectorized_projection_preserves_tangential_velocity_and_degenerate_fallback(face_sf):
    """Both production projection paths retain the scalar BC contract."""
    from source.solvers.FVM.assemble.momentum import _apply_empty_bc_ustar
    from source.solvers.FVM.solve.simple_solver import _apply_slip_bc

    owners = np.array([1, 0, 1], dtype=np.int64)
    interior = np.array([[2.0, 3.0, -1.0], [-4.0, 1.0, 5.0]])
    velocity = np.vstack((interior, np.zeros((3, 3))))
    boundary_indices = np.array([2, 3, 4], dtype=np.int64)

    _apply_empty_bc_ustar(velocity, boundary_indices, owners, face_sf)
    expected = interior[owners].copy()
    valid = np.linalg.norm(face_sf, axis=1) > 1e-10
    normal = face_sf[valid] / np.linalg.norm(face_sf[valid], axis=1)[:, np.newaxis]
    expected[valid] -= np.sum(expected[valid] * normal, axis=1)[:, np.newaxis] * normal
    assert np.allclose(velocity[boundary_indices], expected, rtol=0.0, atol=1e-14)
    assert np.allclose(velocity[boundary_indices[~valid]], interior[owners[~valid]])
    assert np.allclose(np.sum(velocity[boundary_indices[valid]] * normal, axis=1), 0.0, atol=1e-14)

    boundary = {"start_face": 0, "n_faces": 3}
    geo = {"face_sf": face_sf}
    U_slip = np.vstack((interior, np.zeros((3, 3))))
    _apply_slip_bc(U_slip, boundary, owners, geo, n_cells=2, n_interior=0)
    assert np.allclose(U_slip[2:], expected, rtol=0.0, atol=1e-14)
