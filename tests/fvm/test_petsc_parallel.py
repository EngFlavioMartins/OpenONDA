"""Collective PETSc verification.

Run this module collectively, for example::

    mpiexec -n 2 python -m pytest -q tests/fvm/test_petsc_parallel.py

It is skipped during the ordinary SciPy-only suite.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

mpi4py = pytest.importorskip("mpi4py", reason="parallel FVM test requires mpi4py")
pytest.importorskip("petsc4py", reason="parallel FVM test requires petsc4py")

from source.solvers.FVM import (  # noqa: E402
    BoundaryConfig,
    ExecutionConfig,
    FVMConfig,
    Solver,
    SolverParams,
    TimeConfig,
    TransportConfig,
)
from source.solvers.FVM.core.parallel import ParallelContext  # noqa: E402
from source.solvers.FVM.solve.linear_interface import (  # noqa: E402
    normalized_residual,
    solve_linear_system,
)

from ._structured_mesh import structured_box  # noqa: E402

pytestmark = pytest.mark.mpi


def test_collective_petsc_solution_matches_scipy():
    context = ParallelContext.create(ExecutionConfig.petsc_replicated())
    n = 31
    A = diags((-np.ones(n - 1), 4.0 * np.ones(n), -np.ones(n - 1)), (-1, 0, 1), format="csr")
    b = np.linspace(1.0, 2.0, n)
    expected = spsolve(A, b)

    actual, info = solve_linear_system(
        A,
        b,
        method="cg",
        equation_type="pressure",
        tol=1e-11,
        maxiter=200,
        backend="petsc",
        parallel_context=context,
        return_info=True,
        failure_policy="raise",
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-11)
    assert info.converged
    assert normalized_residual(A, actual, b) < 1e-10


def test_collective_pimple_step_is_rank_invariant(tmp_path):
    context = ParallelContext.create(ExecutionConfig.petsc_replicated())
    mesh = structured_box(3, 3, 3)
    config = FVMConfig(
        case_name="petsc_pimple",
        execution=ExecutionConfig.petsc_replicated(),
        time=TimeConfig.transient(dt=0.01, duration=0.01, write_interval=100),
        solver=SolverParams.pimple(
            n_correctors=2,
            linear_solver="bicgstab",
            convection_scheme="upwind",
            pressure_tol=1e-10,
        ),
        transport=TransportConfig(density=1.0, nu=0.02),
        boundaries=[
            BoundaryConfig.inlet("xmin", [1.0, 0.0, 0.0]),
            BoundaryConfig.outlet("xmax", 0.0),
            BoundaryConfig.wall("ymin"),
            BoundaryConfig.wall("ymax"),
            BoundaryConfig.wall("zmin"),
            BoundaryConfig.wall("zmax"),
        ],
        initial_U=[1.0, 0.0, 0.0],
    )
    with contextlib.redirect_stdout(io.StringIO()):
        solver = Solver(config, str(tmp_path), mesh_data=mesh)
        solver.auto_write = False
        residuals = solver.solve_pimple(0.01)

    assert residuals["p"] < 1e-8
    assert np.all(np.isfinite(solver.U))
    states = context.comm.allgather(solver.U.copy())
    for state in states[1:]:
        np.testing.assert_allclose(state, states[0], rtol=0.0, atol=1e-12)
