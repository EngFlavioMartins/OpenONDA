"""Preconditioner rebuild telemetry for static-topology solves."""

import numpy as np
from scipy.sparse import diags

from source.solvers.FVM.solve import linear_interface


def _system():
    n = 40
    matrix = diags((-np.ones(n - 1), 4.0 * np.ones(n), -np.ones(n - 1)), (-1, 0, 1))
    return matrix.tocsr(), np.linspace(1.0, 2.0, n)


def test_ilu_rebuild_and_reuse_are_reported():
    linear_interface._ILU_CACHE.clear()
    matrix, rhs = _system()
    options = {
        "method": "bicgstab",
        "equation_type": "momentum",
        "reuse_ilu": True,
        "ilu_key": "telemetry",
        "tol": 1.0e-10,
        "return_info": True,
    }
    _, first = linear_interface.solve_linear_system(matrix, rhs, **options)
    _, second = linear_interface.solve_linear_system(matrix, rhs, **options)
    assert first.preconditioner_rebuilt is True
    assert second.preconditioner_rebuilt is False


def test_amg_rebuild_and_reuse_are_reported():
    linear_interface._AMG_CACHE.clear()
    matrix, rhs = _system()
    options = {
        "method": "amg",
        "equation_type": "pressure",
        "tol": 1.0e-10,
        "amg_tol": 1.0e-10,
        "return_info": True,
    }
    _, first = linear_interface.solve_linear_system(matrix, rhs, **options)
    _, second = linear_interface.solve_linear_system(matrix, rhs, **options)
    assert first.preconditioner_rebuilt is True
    assert second.preconditioner_rebuilt is False


def test_warm_guess_convergence_uses_deviation_normalization():
    """A mean-dominated warm guess must not satisfy the tolerance for free.

    Regression for the frozen-flow bug: for x-momentum in a unit free stream,
    ``||b||`` is dominated by transport of the mean, so a ``tol * ||b||`` test
    accepted the previous velocity unchanged (zero iterations) while the
    near-wall residual -- the part that grows boundary layers and vortex
    shedding -- stayed entirely unsolved.
    """
    rng = np.random.default_rng(7)
    n = 400
    matrix = diags((-np.ones(n - 1), 4.0 * np.ones(n), -np.ones(n - 1)), (-1, 0, 1)).tocsr()
    exact = 1.0 + 3.0e-5 * rng.standard_normal(n)  # unit mean, small dynamics
    rhs = matrix @ exact
    # Warm guess: mean captured perfectly, ALL small-scale physics missing.
    guess = np.full(n, 1.0)
    tol = 1.0e-4
    residual_vs_b = np.linalg.norm(rhs - matrix @ guess) / np.linalg.norm(rhs)
    assert residual_vs_b < tol, "test setup must sit inside the naive tolerance"

    solution, info = linear_interface.solve_linear_system(
        matrix,
        rhs,
        method="bicgstab",
        equation_type="momentum",
        x0=guess,
        tol=tol,
        return_info=True,
    )
    # The old ||b||-relative test returned the guess bitwise with 0 iterations.
    assert not np.array_equal(solution, guess)
    error = np.linalg.norm(solution - exact) / np.linalg.norm(exact - guess)
    assert error < 0.05, f"small-scale physics still unresolved: {error:.3f}"
