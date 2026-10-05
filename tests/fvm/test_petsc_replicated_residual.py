"""Replicated PETSc convergence uses the native algebraic residual definition."""

import re
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import diags, eye, kron

pytest.importorskip("petsc4py")
from petsc4py import PETSc

from source.solvers.fvm.solve.linear_interface import (
    _solve_petsc,
    normalized_residual_target,
    solve_linear_system,
)


@pytest.mark.parametrize("method", ["amg", "unsupported"])
def test_pressure_solves_leave_no_deferred_petsc_objects(capfd, method):
    """Repeated interface solves must release PCs on success and on errors."""
    PETSc.garbage_cleanup()
    count = 32
    matrix = diags(
        [-np.ones(count - 1), 4 * np.ones(count), -np.ones(count - 1)], [-1, 0, 1]
    ).tocsr()
    rhs = matrix @ np.sin(np.arange(count))
    context = SimpleNamespace(size=PETSc.COMM_WORLD.getSize())
    for _ in range(25):
        if method == "unsupported":
            with pytest.raises(ValueError, match="Unknown PETSc"):
                _solve_petsc(
                    matrix,
                    rhs,
                    method,
                    "kinematic_pressure",
                    1e-7,
                    0.005,
                    1000,
                    None,
                    context,
                    None,
                )
        else:
            _, result = _solve_petsc(
                matrix, rhs, method, "kinematic_pressure", 1e-7, 0.005, 1000, None, context, None
            )
            assert result.converged
    PETSc.garbage_view()
    output = capfd.readouterr().out
    # The collective viewer writes its summary on the communicator root.
    if PETSc.COMM_WORLD.getRank() == 0:
        counts = re.findall(r"Total entries:\s*(\d+)", output)
        assert len(counts) == PETSc.COMM_WORLD.getSize()
        assert all(int(count) == 0 for count in counts), output


@pytest.mark.parametrize("method", ["amg", "cg", "gmres", "bicgstab"])
@pytest.mark.parametrize("amplitude", [1.0e-18, 1.0, 1.0e18])
def test_pressure_true_residual_matches_requested_norm(method, amplitude):
    """GAMG previously stopped after two steps with an unacceptable true residual."""
    side = 16
    one = diags([-np.ones(side - 1), 2 * np.ones(side), -np.ones(side - 1)], [-1, 0, 1])
    matrix = (kron(eye(side), one) + kron(one, eye(side))).tocsr()
    rhs = amplitude * np.ones(side**2)
    context = SimpleNamespace(size=PETSc.COMM_WORLD.getSize(), is_partitioned=False)
    _, target, scale = normalized_residual_target(matrix, rhs, None, 1e-7, 0.005)
    solution, result = solve_linear_system(
        matrix,
        rhs,
        method=method,
        equation_type="kinematic_pressure",
        tol=1e-7,
        rel_tol=0.005,
        maxiter=1000,
        backend="petsc",
        parallel_context=context,
        return_info=True,
    )
    residual = np.linalg.norm(rhs - matrix @ solution) / scale
    assert result.converged
    assert result.final_residual == pytest.approx(residual)
    assert residual <= target * (1 + 1e-10)


def test_warm_mean_flow_still_resolves_the_deviation():
    count = 30
    matrix = diags(
        [-np.ones(count - 1), 4 * np.ones(count), -np.ones(count - 1)], [-1, 0, 1]
    ).tocsr()
    initial = np.full(count, 1e3)
    truth = initial + np.sin(np.arange(count))
    rhs = matrix @ truth
    context = SimpleNamespace(size=PETSc.COMM_WORLD.getSize(), is_partitioned=False)
    _, target, scale = normalized_residual_target(matrix, rhs, initial, 1e-7, 0.005)
    solution, result = solve_linear_system(
        matrix,
        rhs,
        method="amg",
        equation_type="kinematic_pressure",
        tol=1e-7,
        rel_tol=0.005,
        maxiter=1000,
        x0=initial,
        backend="petsc",
        parallel_context=context,
        return_info=True,
    )
    assert result.converged and result.iterations > 0
    assert np.linalg.norm(rhs - matrix @ solution) / scale <= target * (1 + 1e-10)
