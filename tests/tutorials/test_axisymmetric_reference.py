"""The diagnostic's fast Poisson solver must solve the original discrete matrix."""

import numpy as np
from scipy import sparse

from tests._tutorial_helpers import load_tutorial_module

MeridionalPoisson = load_tutorial_module(
    "vpm/vortex_interactions", "assets.axisymmetric_reference"
).MeridionalPoisson


def test_separable_poisson_residual():
    n, m, h = 27, 18, 0.04
    ri = np.arange(1, m + 1) * h
    dx = sparse.diags([-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1]) / h**2
    dr = sparse.diags(
        [(-1 - h / (2 * ri[1:])) / h**2, 2 * np.ones(m) / h**2, (-1 + h / (2 * ri[:-1])) / h**2],
        [-1, 0, 1],
    )
    matrix = sparse.kron(dx, sparse.eye(m)) + sparse.kron(sparse.eye(n), dr)
    rhs = np.random.default_rng(18).normal(size=n * m)
    answer = MeridionalPoisson(n, ri, h).solve(rhs)
    np.testing.assert_allclose(matrix @ answer, rhs, atol=2e-13, rtol=2e-13)
