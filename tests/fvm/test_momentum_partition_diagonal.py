"""Mixed boundaries on another rank must determine the shared halo layout."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from source.solvers.fvm.assemble import momentum


@pytest.mark.parametrize("remote_components", [False, True])
def test_rank_without_mixed_faces_shares_matrix_but_matches_diagonal_layout(
    monkeypatch, remote_components
):
    matrix = csr_matrix(np.diag([2.0, 4.0]))
    equations = {name: {"A": matrix, "b": np.ones(2)} for name in ("x", "y", "z")}
    monkeypatch.setattr(momentum, "assemble_momentum_equation", lambda *args, **kwargs: equations)
    relaxed = []

    def solve(matrix, rhs, **kwargs):
        relaxed.append(matrix.diagonal().copy())
        return np.zeros(2), SimpleNamespace(initial_residual=0.0, final_residual=0.0)

    monkeypatch.setattr(momentum, "solve_linear_system", solve)
    exchanged = []
    parallel = SimpleNamespace(
        is_partitioned=True,
        global_max=lambda local: max(local, int(remote_components)),
        exchange_halo=lambda values: exchanged.append(values.copy()),
    )
    _, diagonal = momentum.solve_momentum_predictor(
        np.zeros((2, 3)),
        np.zeros(2),
        np.empty(0),
        1.0,
        0.001,
        {"n_cells": 2, "n_faces": 0, "n_interior_faces": 0},
        {},
        [],
        solver="bicgstab",
        under_relaxation=0.5,
        parallel_context=parallel,
    )
    expected = np.array([4.0, 8.0])
    for value in relaxed:
        np.testing.assert_array_equal(value, expected)
    if remote_components:
        expected = np.repeat(expected[:, None], 3, axis=1)
    np.testing.assert_array_equal(diagonal, expected)
    np.testing.assert_array_equal(exchanged[0], expected)
