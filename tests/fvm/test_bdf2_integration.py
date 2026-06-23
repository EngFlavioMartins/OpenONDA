"""Integration coverage for the BDF2 time scheme wired through the full
``Solver`` → ``PIMPLESolver`` → ``assemble_momentum_equation`` path.

The operator-level order is verified in ``test_temporal_order.py``; this test
guards the *plumbing*: that ``solver.time_scheme="backward"`` reaches the
momentum assembly (via the ``U_old``/``U_old_old`` history ring in
``core.solver.Solver.evolve``) and produces a finite solution that differs from
the BDF1 result.
"""

import contextlib
import io
import tempfile

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


def _run(scheme, n_steps=4):
    mesh = structured_box(6, 6, 1)
    sp = SolverParams(
        algorithm="PIMPLE", n_correctors=2, linear_solver="spsolve",
        alpha_u=1.0, alpha_p=1.0, convection_scheme="central",
    )
    sp.time_scheme = scheme
    walls = [BoundaryConfig.wall(n) for n in ("xmin", "xmax", "ymin", "zmin")]
    lid = BoundaryConfig(name="ymax", type_U="fixedValue", value_U=[1.0, 0, 0], type_p="zeroGradient")
    cfg = FVMConfig(
        case_name="bdf2",
        time=TimeConfig(delta_t=0.05, end_time=0.25, write_interval=999),
        solver=sp,
        transport=TransportConfig(density=1.0, nu=0.05),
        boundaries=walls + [lid, BoundaryConfig.empty("zmax")],
        initial_U=[0, 0, 0],
        initial_p=0.0,
    )
    with tempfile.TemporaryDirectory() as d:
        with contextlib.redirect_stdout(io.StringIO()):
            s = Solver(cfg, case_dir=d, mesh_data=mesh)
            s.auto_write = False
            for _ in range(n_steps):
                s.evolve()
            return s.U[: mesh["n_elements"]].copy()


def test_bdf2_engages_through_solver():
    u_euler = _run("euler_implicit")
    u_bdf2 = _run("backward")

    assert np.all(np.isfinite(u_euler)) and np.all(np.isfinite(u_bdf2))
    # BDF2 must actually change the result (proves the history ring + ddt scheme
    # reach the momentum assembly) but stay in the same physical ballpark.
    rel = np.linalg.norm(u_bdf2 - u_euler) / (np.linalg.norm(u_euler) + 1e-30)
    assert 1e-4 < rel < 0.5, f"BDF2/Euler relative difference {rel:.3e} unexpected"
