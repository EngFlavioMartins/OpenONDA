"""Direct sparse reuse is exact, bounded, and independent of solver history."""

import time
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from source.solvers.fvm.solve import linear_interface as linear


def test_direct_factorization_reuses_only_an_identical_matrix(monkeypatch):
    actual_factorize = linear.splu
    factorizations = []

    def count_factorizations(matrix):
        factorizations.append(matrix.copy())
        return actual_factorize(matrix)

    monkeypatch.setattr(linear, "splu", count_factorizations)
    workspace = SimpleNamespace(direct_factorization=None)
    matrix = csr_matrix(np.array([[4.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 3.0]]))
    rhs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    first, info = linear.solve_linear_system(
        matrix, rhs, direct_workspace=workspace, return_info=True
    )
    np.testing.assert_allclose(first, spsolve(matrix, rhs), atol=1e-14, rtol=0)
    assert info.converged
    matrix.data *= 1.0  # The producer may reuse the same mutable CSR object.
    again, info = linear.solve_linear_system(
        matrix, rhs + 1, direct_workspace=workspace, return_info=True
    )
    np.testing.assert_allclose(again, spsolve(matrix, rhs + 1), atol=1e-14, rtol=0)
    assert info.converged
    assert len(factorizations) == 1

    previous = weakref.ref(workspace.direct_factorization)
    original_factorize = linear.splu

    def check_released_before_refactorization(matrix):
        assert previous() is None
        return original_factorize(matrix)

    monkeypatch.setattr(linear, "splu", check_released_before_refactorization)
    matrix.data[0] += 0.5
    changed, _ = linear.solve_linear_system(
        matrix, rhs, direct_workspace=workspace, return_info=True
    )
    np.testing.assert_allclose(changed, spsolve(matrix, rhs), atol=1e-14, rtol=0)
    assert len(factorizations) == 2
    monkeypatch.setattr(linear, "splu", count_factorizations)

    different_pattern = matrix.copy()
    different_pattern[0, 2] = 0.1
    linear.solve_linear_system(different_pattern, rhs, direct_workspace=workspace)
    assert len(factorizations) == 3

    different_dtype = different_pattern.astype(np.float32)
    linear.solve_linear_system(
        different_dtype, np.asarray(rhs, dtype=np.float32), direct_workspace=workspace
    )
    assert len(factorizations) == 4


def test_pressure_anchor_change_and_singular_failure_never_reuse_stale_lu(monkeypatch):
    actual_factorize = linear.splu
    factorizations = []

    def count_factorizations(matrix):
        factorizations.append(matrix.copy())
        return actual_factorize(matrix)

    monkeypatch.setattr(linear, "splu", count_factorizations)
    workspace = SimpleNamespace(direct_factorization=None)
    anchored_first = csr_matrix(np.array([[1.0, 0.0], [-1.0, 2.0]]))
    anchored_second = csr_matrix(np.array([[2.0, -1.0], [0.0, 1.0]]))
    rhs = np.array([0.0, 1.0])
    linear.solve_linear_system(
        anchored_first, rhs, equation_type="kinematic_pressure", direct_workspace=workspace
    )
    second = linear.solve_linear_system(
        anchored_second, rhs, equation_type="kinematic_pressure", direct_workspace=workspace
    )
    np.testing.assert_allclose(second, spsolve(anchored_second, rhs), rtol=0, atol=1e-14)
    assert len(factorizations) == 2
    with pytest.raises(linear.LinearSolveError, match="factorization failed"):
        linear.solve_linear_system(csr_matrix((2, 2)), rhs, direct_workspace=workspace)
    assert workspace.direct_factorization is None
    linear.solve_linear_system(anchored_first, rhs, direct_workspace=workspace)
    assert len(factorizations) == 4


def test_native_ibm_two_predictors_match_uncached_direct_solve(tmp_path, monkeypatch):
    """One accepted IBM step exercises both predictor solves and pressure correction."""
    from tests.fvm.test_freestream_projection_stability import make_solver

    true_solve = linear._solve_direct_with_workspace

    def uncached(A, b, _workspace):
        start = time.perf_counter()
        solution = spsolve(A, b)
        return solution, 0.0, time.perf_counter() - start

    with monkeypatch.context() as patch:
        patch.setattr(linear, "_solve_direct_with_workspace", uncached)
        with make_solver(tmp_path / "fresh", immersed=True) as fresh:
            fresh.advance()
            expected_velocity = fresh.velocity.copy()
            expected_flux = fresh.volumetric_face_flux.copy()
            expected_pressure = fresh.kinematic_pressure.copy()
    assert linear._solve_direct_with_workspace is true_solve
    with make_solver(tmp_path / "cached", immersed=True) as cached:
        cached.advance()
        np.testing.assert_allclose(cached.velocity, expected_velocity, rtol=0, atol=1e-11)
        np.testing.assert_allclose(cached.volumetric_face_flux, expected_flux, rtol=0, atol=1e-12)
        np.testing.assert_allclose(cached.kinematic_pressure, expected_pressure, rtol=0, atol=1e-11)
        assert np.isfinite(cached.ibm.slip_error(cached.velocity))
        # Full-time no-slip and rollback are covered by test_freestream_projection_stability.
        assert all(result.converged for result in cached.algorithm.last_linear_results)
