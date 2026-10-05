"""M4-prime kernel identities used by particle renewal and GBD."""

import numpy as np

from source.coupler.lattice_transfer import m4_prime


def test_coupler_m4_prime_matches_the_grid_diffusion_kernel():
    from source.solvers.vpm.physics.diffusion.grid import _m4_prime_1d

    rng = np.random.default_rng(116)
    distance = rng.uniform(-2.5, 2.5, size=10_000)
    np.testing.assert_allclose(m4_prime(distance), _m4_prime_1d(distance), rtol=0.0, atol=0.0)


def test_m4_prime_reproduces_quadratic_moments_for_ten_thousand_random_phases():
    rng = np.random.default_rng(20260825)
    phase = rng.uniform(-1.0e4, 1.0e4, size=10_000)
    nodes = np.floor(phase)[:, None] + np.arange(-1, 3)[None, :]
    weights = m4_prime(phase[:, None] - nodes)
    scale = 512.0 * np.finfo(np.float64).eps * (1.0 + np.abs(phase) ** 2)
    assert np.all(np.abs(weights.sum(axis=1) - 1.0) <= scale)
    assert np.all(np.abs((nodes * weights).sum(axis=1) - phase) <= scale)
    assert np.all(np.abs((nodes**2 * weights).sum(axis=1) - phase**2) <= scale)


def test_tensor_m4_prime_reproduces_3d_moments_for_random_phases():
    rng = np.random.default_rng(63)
    phase = rng.uniform(-3.0, 3.0, size=(10_000, 3))
    nodes = np.floor(phase)[:, :, None] + np.arange(-1, 3)[None, None, :]
    axis_weight = m4_prime(phase[:, :, None] - nodes)
    weight = np.einsum("ni,nj,nk->nijk", axis_weight[:, 0], axis_weight[:, 1], axis_weight[:, 2])
    x, y, z = (nodes[:, axis] for axis in range(3))
    np.testing.assert_allclose(weight.sum(axis=(1, 2, 3)), 1.0, rtol=0.0, atol=5.0e-15)
    for axis, coordinate in enumerate((x, y, z)):
        target = np.expand_dims(
            coordinate,
            tuple(other for other in range(1, 4) if other != axis + 1),
        )
        # The insertion above leaves one coordinate axis; broadcasting it over
        # the two remaining stencil dimensions tests the tensor product itself.
        reconstructed = (weight * target).sum(axis=(1, 2, 3))
        np.testing.assert_allclose(reconstructed, phase[:, axis], rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(
        (weight * x[:, :, None, None] ** 2).sum(axis=(1, 2, 3)),
        phase[:, 0] ** 2,
        rtol=0.0,
        atol=3.0e-14,
    )
    np.testing.assert_allclose(
        (weight * x[:, :, None, None] * y[:, None, :, None]).sum(axis=(1, 2, 3)),
        phase[:, 0] * phase[:, 1],
        rtol=0.0,
        atol=3.0e-14,
    )
