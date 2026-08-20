"""Integration coverage for the BDF2 time scheme wired through the full
``FVMSolver`` → ``PIMPLESolver`` → ``assemble_momentum_equation`` path.

The operator-level order is verified in ``test_temporal_order.py``; this test
guards the *plumbing*: that ``solver.time_scheme="backward"`` reaches the
momentum assembly (via the ``U_old``/``U_old_old`` history ring in
``core.solver.FVMSolver.advance``) and produces a finite solution that differs from
the BDF1 result.
"""

import contextlib
import io
import tempfile

import numpy as np

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


def _run(scheme, n_steps=4):
    mesh = structured_box(6, 6, 1)
    schemes = DiscretizationConfig(convection_scheme="central", time_scheme=scheme)
    linear = LinearSolverConfig(linear_solver="spsolve")
    pimple = PimpleControl(n_correctors=2)
    walls = [BoundaryConfig.wall(n) for n in ("xmin", "xmax", "ymin", "zmin")]
    lid = BoundaryConfig(
        name="ymax",
        velocity_type="fixedValue",
        velocity_value=[1.0, 0, 0],
        pressure_type="zeroGradient",
    )
    cfg = FVMSetup(
        case_name="bdf2",
        time=TimeConfig(time_step_size=0.05, end_time=0.25, output_interval_steps=999),
        schemes=schemes,
        linear=linear,
        pimple=pimple,
        transport=TransportConfig(density=1.0, kinematic_viscosity=0.05),
        boundaries=walls + [lid, BoundaryConfig.empty("zmax")],
        initial_velocity=[0, 0, 0],
        initial_kinematic_pressure=0.0,
    )
    with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
        s = FVMSolver(cfg, case_dir=d, mesh_data=mesh)
        s.auto_write = False
        for _ in range(n_steps):
            s.advance()
        return s.velocity[: mesh["n_cells"]].copy()


def test_bdf2_engages_through_solver():
    u_euler = _run("euler_implicit")
    u_bdf2 = _run("backward")

    assert np.all(np.isfinite(u_euler)) and np.all(np.isfinite(u_bdf2))
    # BDF2 must actually change the result (proves the history ring + ddt scheme
    # reach the momentum assembly) but stay in the same physical ballpark.
    rel = np.linalg.norm(u_bdf2 - u_euler) / (np.linalg.norm(u_euler) + 1e-30)
    assert 1e-4 < rel < 0.5, f"BDF2/Euler relative difference {rel:.3e} unexpected"
