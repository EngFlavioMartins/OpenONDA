"""Residual-based acceptance contracts for the panel linear solvers."""

from __future__ import annotations

import numpy as np
import pytest

taichi = pytest.importorskip("taichi", reason="VPM requires taichi")

from source.solvers.vpm.boundary_elements.panels.solver.linear_solvers import (  # noqa: E402
    EqualityConstrainedLeastSquaresFactorization,
    PanelScipySolver,
    constrained_least_squares_metrics,
    relative_residual,
)


def _ensure_taichi_cpu() -> None:
    if taichi.lang.impl.get_runtime().prog is None:
        taichi.init(arch=taichi.cpu)


def _solve(influence_matrix: np.ndarray, right_hand_side: np.ndarray) -> tuple[bool, float]:
    _ensure_taichi_cpu()
    n = right_hand_side.shape[0]
    matrix_field = taichi.field(taichi.f64, shape=(n, n))
    matrix_field.from_numpy(influence_matrix)
    right_hand_side_field = taichi.field(taichi.f64, shape=n)
    right_hand_side_field.from_numpy(right_hand_side)
    solution_field = taichi.field(taichi.f64, shape=n)

    solver = PanelScipySolver()
    success = solver.solve(matrix_field, right_hand_side_field, solution_field, n)
    return success, solver.last_residual


def test_well_posed_system_is_accepted():
    success, residual = _solve(2.0 * np.eye(4), np.array([1.0, 2.0, 3.0, 4.0]))
    assert success
    assert residual < 1.0e-12


def test_inconsistent_singular_system_is_rejected():
    # Rank-deficient with a right-hand side outside the column space: a dense
    # direct solve returns a huge finite vector instead of raising, so only a
    # residual test can catch it.
    influence_matrix = np.zeros((4, 4))
    influence_matrix[0, 0] = 1.0
    influence_matrix[1, 1] = 1.0
    right_hand_side = np.array([1.0, 1.0, 1.0, 0.0])

    success, residual = _solve(influence_matrix, right_hand_side)

    assert not success
    assert residual > 1.0e-3


def test_relative_residual_rejects_non_finite_solutions():
    influence_matrix = np.eye(3)
    right_hand_side = np.ones(3)
    solution = np.array([1.0, np.nan, 1.0])
    assert relative_residual(influence_matrix, solution, right_hand_side) == float("inf")


def test_relative_residual_is_scaled_by_the_right_hand_side():
    influence_matrix = np.eye(2)
    # An absolute residual of 1 against a right-hand side of norm 100 is a
    # 1% relative error, and must not be reported as if it were 100%.
    assert relative_residual(
        influence_matrix, np.array([100.0, 1.0]), np.array([100.0, 0.0])
    ) == pytest.approx(1.0 / 100.0)


def test_constrained_metrics_distinguish_optimality_from_equation_compatibility():
    # min ||x - b|| subject to x0 + x1 = 0.  The constrained minimizer
    # x=(1,-1) cannot satisfy x=b=(2,0), but its equation residual gradient
    # lies entirely in range(C.T), so it is a valid constrained optimum.
    influence_matrix = np.eye(2)
    right_hand_side = np.array([2.0, 0.0])
    constraints = np.array([[1.0, 1.0]])
    solution = np.array([1.0, -1.0])

    metrics = constrained_least_squares_metrics(
        influence_matrix, right_hand_side, constraints, solution
    )

    assert metrics["discrete_equation_residual"] == pytest.approx(1.0 / np.sqrt(2.0))
    assert metrics["constraint_residual"] < 1.0e-14
    assert metrics["projected_optimality_residual"] < 1.0e-14


def test_reusable_constrained_factorization_matches_kkt_reference_for_multiple_rhs():
    rng = np.random.default_rng(20260825)
    influence_matrix = rng.normal(size=(8, 8)) + 4.0 * np.eye(8)
    constraints = np.zeros((2, 8))
    constraints[0, :4] = np.array([1.0, 1.5, 0.8, 1.2])
    constraints[1, 4:] = np.array([0.7, 1.1, 1.4, 0.9])
    factorization = EqualityConstrainedLeastSquaresFactorization.factorize(
        influence_matrix, constraints
    )

    normal_matrix = influence_matrix.T @ influence_matrix
    kkt = np.block(
        [
            [normal_matrix, constraints.T],
            [constraints, np.zeros((constraints.shape[0], constraints.shape[0]))],
        ]
    )
    for right_hand_side in (rng.normal(size=8), rng.normal(size=8)):
        solution = factorization.solve(right_hand_side)
        reference = np.linalg.solve(
            kkt,
            np.r_[influence_matrix.T @ right_hand_side, np.zeros(constraints.shape[0])],
        )[:8]
        np.testing.assert_allclose(solution, reference, rtol=1.0e-11, atol=1.0e-12)
        metrics = constrained_least_squares_metrics(
            influence_matrix, right_hand_side, constraints, solution
        )
        assert metrics["constraint_residual"] < 1.0e-12
        assert metrics["projected_optimality_residual"] < 1.0e-12

    assert factorization.memory_bytes > influence_matrix.nbytes
