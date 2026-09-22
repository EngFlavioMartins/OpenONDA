"""Lagging a preconditioner must not lag the matrix being solved."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

pytest.importorskip("petsc4py")
from petsc4py import PETSc

from source.solvers.fvm.solve.linear_interface import deviation_norm_factor
from source.solvers.fvm.solve.petsc_partitioned import OwnedRowsCSR, PartitionedLinearWorkspace


def test_uniform_state_uses_the_same_roundoff_floor_as_serial():
    if PETSc.COMM_WORLD.getSize() != 1:
        pytest.skip("serial unit test; distributed cases use the MPI regression")
    context = SimpleNamespace(size=1, global_sum=lambda v: v, global_max=lambda v: v)
    matrix = csr_matrix(np.eye(7))
    guess = np.ones(7)
    rhs = guess.copy()
    rhs[0] = np.nextafter(1.0, 2.0)
    workspace = PartitionedLinearWorkspace(context)
    try:
        solution, result = workspace.solve(
            OwnedRowsCSR.from_global(matrix, rhs, 0, 1),
            method="bicgstab",
            tolerance=1e-4,
            max_iterations=100,
            constant_nullspace=False,
            initial_guess=guess,
        )
        expected = np.linalg.norm(rhs - matrix @ solution) / deviation_norm_factor(
            matrix, rhs, guess
        )
        assert result.converged
        assert result.final_residual == pytest.approx(expected)
        np.testing.assert_allclose(solution, rhs, rtol=0, atol=np.finfo(float).eps)
    finally:
        workspace.close()


def test_initial_guess_work_vectors_reuse_rebuild_and_close():
    """Warm guesses retain their PETSc work vectors without retaining an old equation."""
    if PETSc.COMM_WORLD.getSize() != 1:
        pytest.skip("serial unit test; distributed cases use the MPI benchmark")
    context = SimpleNamespace(size=1, global_sum=lambda v: v, global_max=lambda v: v)
    rhs = np.linspace(-2.0, 3.0, 7)
    guess = np.linspace(0.25, 1.0, 7)
    original = np.diag(np.full(7, 4.0)) - np.eye(7, k=1) - np.eye(7, k=-1)
    changed = original.copy()
    changed[2, 3] = changed[3, 2] = -1.25
    workspace = PartitionedLinearWorkspace(context)
    arguments = {
        "method": "bicgstab",
        "tolerance": 1e-11,
        "max_iterations": 100,
        "constant_nullspace": False,
        "initial_guess": guess,
    }
    try:
        solution, result = workspace.solve(
            OwnedRowsCSR.from_global(csr_matrix(original), rhs, 0, 1), **arguments
        )
        expected_initial = np.linalg.norm(rhs - original @ guess) / deviation_norm_factor(
            original, rhs, guess
        )
        assert result.initial_residual == pytest.approx(expected_initial)
        np.testing.assert_allclose(original @ solution, rhs, rtol=0, atol=2e-10)
        work_vectors = (workspace._operator_guess, workspace._reference, workspace._uniform)
        assert all(vector is not None for vector in work_vectors)

        solution, result = workspace.solve(
            OwnedRowsCSR.from_global(csr_matrix(changed), rhs, 0, 1), **arguments
        )
        expected_initial = np.linalg.norm(rhs - changed @ guess) / deviation_norm_factor(
            changed, rhs, guess
        )
        assert result.initial_residual == pytest.approx(expected_initial)
        np.testing.assert_allclose(changed @ solution, rhs, rtol=0, atol=2e-10)
        assert (workspace._operator_guess, workspace._reference, workspace._uniform) == work_vectors

        rebuilt = np.diag(np.full(8, 4.0)) - np.eye(8, k=1) - np.eye(8, k=-1)
        solution, result = workspace.solve(
            OwnedRowsCSR.from_global(csr_matrix(rebuilt), np.arange(8, dtype=float), 0, 1),
            **{**arguments, "initial_guess": np.linspace(0.25, 1.0, 8)},
        )
        np.testing.assert_allclose(
            rebuilt @ solution, np.arange(8, dtype=float), rtol=0, atol=2e-10
        )
        assert result.converged
        assert all(
            vector is not previous
            for vector, previous in zip(
                (workspace._operator_guess, workspace._reference, workspace._uniform),
                work_vectors,
                strict=True,
            )
        )
    finally:
        workspace.close()
    assert workspace._operator_guess is None
    assert workspace._reference is None
    assert workspace._uniform is None


def test_reused_preconditioner_solves_updated_equation():
    if PETSc.COMM_WORLD.getSize() != 1:
        pytest.skip("serial unit test; distributed cases use the MPI benchmark")
    context = SimpleNamespace(size=1, global_sum=lambda v: v, global_max=lambda v: v)
    original = np.diag(np.full(7, 4.0)) - np.eye(7, k=1) - np.eye(7, k=-1)
    rhs = np.linspace(-1.0, 2.0, 7)
    changed = original.copy()
    changed[2, 3] = changed[3, 2] = -1.02
    cumulative_change = original.copy()
    cumulative_change[2, 3] = cumulative_change[3, 2] = -1.31
    large_change = original.copy()
    large_change[2, 3] = large_change[3, 2] = -1.8
    workspace = PartitionedLinearWorkspace(context)
    try:
        for matrix, rebuilt in (
            (original, True),
            (changed, False),
            # Compare with the PC's matrix, not the immediately preceding solve.
            (cumulative_change, True),
            (3 * changed, True),
            (large_change, True),
        ):
            system = OwnedRowsCSR.from_global(csr_matrix(matrix), rhs, 0, 1)
            actual, result = workspace.solve(
                system,
                method="amg",
                tolerance=1e-11,
                max_iterations=100,
                constant_nullspace=False,
                initial_guess=None,
                preconditioner_reuse_tolerance=0.05,
            )
            assert result.preconditioner_rebuilt == rebuilt
            np.testing.assert_allclose(matrix @ actual, rhs, rtol=0, atol=2e-10)
            np.testing.assert_allclose(actual, np.linalg.solve(matrix, rhs), rtol=0, atol=2e-10)
    finally:
        workspace.close()


def test_failed_reused_preconditioner_rebuilds_and_retries_same_system():
    if PETSc.COMM_WORLD.getSize() != 1:
        pytest.skip("serial unit test")
    context = SimpleNamespace(size=1, global_sum=lambda v: v, global_max=lambda v: v)
    matrix = np.diag(np.full(7, 4.0)) - np.eye(7, k=1) - np.eye(7, k=-1)
    rhs = np.arange(7, dtype=float)
    workspace = PartitionedLinearWorkspace(context)
    system = OwnedRowsCSR.from_global(csr_matrix(matrix), rhs, 0, 1)
    arguments = {
        "method": "amg",
        "tolerance": 1e-11,
        "max_iterations": 100,
        "constant_nullspace": False,
        "initial_guess": None,
        "preconditioner_reuse_tolerance": 0.05,
    }
    try:
        workspace.solve(system, **arguments)
        real_ksp = workspace.ksp

        class FailedFirstSolve:
            calls = 0

            def __getattr__(self, name):
                return getattr(real_ksp, name)

            def solve(self, *args):
                self.calls += 1
                return real_ksp.solve(*args)

            def getConvergedReason(self):
                return -3 if self.calls == 1 else real_ksp.getConvergedReason()

        workspace.ksp = FailedFirstSolve()
        solution, result = workspace.solve(system, **arguments)
        assert workspace.ksp.calls == 2
        assert result.converged and result.preconditioner_rebuilt
        np.testing.assert_allclose(matrix @ solution, rhs, rtol=0, atol=2e-10)
    finally:
        workspace.close()
