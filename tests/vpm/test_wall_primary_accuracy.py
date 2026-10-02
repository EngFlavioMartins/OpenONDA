"""Exact finite reflection cancellation, independent of any FMM/mesh."""

import numpy as np
import pytest

from tests.vpm._direct_gaussian_reference import (
    direct_finite_images,
    finite_wall_normal,
    source_only_direct,
)


@pytest.mark.parametrize("upper", [False, True])
def test_finite_normal_is_zero_below_and_only_two_unpaired_images_above(upper):
    rng = np.random.default_rng(1285)
    zmin, zmax, shells = -0.073, 0.12, 4
    x = rng.uniform(-0.1, 0.1, (7, 3))
    x[:, 2] = rng.uniform(zmin, zmax, 7)
    gamma = rng.normal(size=(7, 3))
    core = rng.uniform(0.025, 0.16, 7)
    q = rng.uniform(-0.1, 0.1, (4, 3))
    q[:, 2] = zmax if upper else zmin
    images = [
        (2 * k * (zmax - zmin) + (2 * zmin if odd else 0.0), odd)
        for k in range(-shells, shells + 1)
        for odd in (False, True)
    ]
    direct, _, _ = direct_finite_images(x, gamma, core, q, images)
    expected = finite_wall_normal(
        x, gamma, core, q, zmin=zmin, zmax=zmax, shells=shells, upper=upper
    )
    np.testing.assert_allclose(expected, direct[:, 2], atol=2e-13, rtol=2e-11)
    rounded = q.astype(np.float32).astype(np.float64)
    with pytest.raises(ValueError, match="geometric plane"):
        finite_wall_normal(
            x, gamma, core, rounded, zmin=zmin, zmax=zmax, shells=shells, upper=upper
        )


def test_query_primary_uses_source_core_and_includes_finite_coincidence():
    x, gamma, core = (
        np.zeros((2, 3)),
        np.array([[0.1, 0.2, 0.3], [0.7, -0.4, 0.2]]),
        np.array([0.04, 0.08]),
    )
    u, j = source_only_direct(x, gamma, core, x[:1], chunk=1)
    exact_u, exact_j, _ = direct_finite_images(x, gamma, core, x[:1], [(0.0, False)])
    np.testing.assert_array_equal(u, exact_u)
    np.testing.assert_allclose(j, exact_j, rtol=2e-15)
