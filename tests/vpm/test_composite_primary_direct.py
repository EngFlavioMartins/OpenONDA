"""CPU convention tests independent of the production self-FMM kernels."""

import numpy as np
import pytest

from tests.vpm._direct_gaussian_reference import primary_direct, transposed_rate_f32
from tests.vpm._slip_periodic_gaussian_oracle import gaussian_pairs


def _skew(vector):
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def test_own_particle_has_finite_jacobian_and_zero_velocity_and_stretch():
    x, gamma, sigma = np.zeros((1, 3)), np.array([[0.3, -0.7, 0.2]]), np.array([0.04])
    u, j, _, _ = primary_direct(x, gamma, sigma, np.array([0]))
    np.testing.assert_array_equal(u, 0)
    expected = _skew(gamma[0]) / (3 * np.pi**1.5 * sigma[0] ** 3)
    np.testing.assert_allclose(j[0], expected, rtol=2e-15)
    np.testing.assert_allclose(j[0].T @ gamma[0], 0.0, atol=2e-14)


def test_distinct_coincident_particles_use_pair_mean_not_source_only_cores():
    x = np.zeros((2, 3))
    gamma, sigma = np.array([[0.3, -0.7, 0.2], [-0.4, 0.1, 0.8]]), np.array([0.04, 0.08])
    u, j, _, _ = primary_direct(x, gamma, sigma, np.array([0, 1]))
    for target in range(2):
        expected = sum(
            _skew(gamma[source]) / (3 * np.pi**1.5 * (0.5 * (sigma[source] + sigma[target])) ** 3)
            for source in range(2)
        )
        np.testing.assert_allclose(j[target], expected, rtol=2e-15)
    np.testing.assert_array_equal(u, 0.0)
    _, source_only = gaussian_pairs(x[:1, None] - x[None], gamma[None], sigma[None])
    assert np.linalg.norm(j[0] - source_only.sum(axis=1)[0]) > 1


def test_chunked_primary_includes_selected_own_entries_and_preserves_sources():
    rng = np.random.default_rng(74483)
    x, gamma = rng.normal(size=(17, 3)), rng.normal(size=(17, 3))
    sigma, indices = rng.uniform(0.02, 0.2, 17), np.array([0, 8, 16])
    originals = x.copy(), gamma.copy(), sigma.copy()
    du, dj = gaussian_pairs(
        x[indices, None] - x[None], gamma[None], 0.5 * (sigma[indices, None] + sigma[None])
    )
    u, j, au, aj = primary_direct(x, gamma, sigma, indices, chunk=4)
    np.testing.assert_allclose(u, du.sum(axis=1), rtol=2e-14, atol=1e-14)
    np.testing.assert_allclose(j, dj.sum(axis=1), rtol=2e-14, atol=1e-12)
    np.testing.assert_allclose(au, np.abs(du).sum(axis=1), rtol=2e-14)
    np.testing.assert_allclose(aj, np.abs(dj).sum(axis=1), rtol=2e-14)
    for value, original in zip((x, gamma, sigma), originals, strict=True):
        np.testing.assert_array_equal(value, original)
    with pytest.raises(ValueError, match="bounded"):
        primary_direct(x, gamma, sigma, indices, max_pairs=50)


def test_float32_stretching_is_explicitly_transposed_not_direct():
    gradient = np.arange(9, dtype=np.float32).reshape(1, 3, 3) / 7
    gamma = np.array([[0.2, -0.4, 0.9]], np.float32)
    for fused in (False, True):
        result = transposed_rate_f32(gradient, gamma, fused=fused)
        np.testing.assert_allclose(result, np.einsum("nji,nj->ni", gradient, gamma), rtol=2e-7)
        assert result.dtype == np.float32
        assert np.linalg.norm(result - np.einsum("nij,nj->ni", gradient, gamma)) > 0.1
