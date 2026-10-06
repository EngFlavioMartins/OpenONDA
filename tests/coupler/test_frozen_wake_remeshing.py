"""Independent conservation and field checks for the isolated wake operators."""

import numpy as np
import pytest

from tests.support.cylinder.measure_frozen_wake_remeshing import (
    curl_spectral_bands,
    induced_fields,
    paired_statistics,
    remesh,
    wake_probes,
)


@pytest.mark.parametrize("kernel", ("M4_PRIME", "LAGRANGE6_TEST_ONLY"))
def test_complete_scatter_preserves_signed_circulation_and_quadratic_moments(kernel):
    # Unequal signs, nonuniform fractions, anisotropic support and nonzero crop
    # origin test cancellations that a cardinal or single-particle cloud misses.
    position = np.array([[1.371, -.093, 0.], [1.517, .011, 0.], [1.622, .082, 0.],
                         [1.723, -.053, 0.]], dtype=np.float32)
    strength = np.zeros_like(position)
    strength[:, 2] = [0.17, -0.12, 0.06, -0.025]
    original_position, original_strength = position.copy(), strength.copy()
    output, deposited, report = remesh(position, strength, np.array([-.181, -.123, 0.]),
                                       .04, kernel, 200000, .08)
    assert np.array_equal(position, original_position)
    assert np.array_equal(strength, original_strength)
    assert output.dtype == position.dtype and deposited.dtype == strength.dtype
    assert np.all(output[:, 2] == 0.) and np.all(deposited[:, :2] == 0.)
    assert report["native_moment_guard"]["pruned_node_count"] == 0
    assert report["native_moment_guard"]["correction_fraction"] == 0.
    for first, second in ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)):
        exact = np.sum(position[:, 0].astype(float)**first
                       * position[:, 1].astype(float)**second
                       * strength[:, 2].astype(float))
        actual = np.sum(output[:, 0].astype(float)**first
                        * output[:, 1].astype(float)**second
                        * deposited[:, 2].astype(float))
        assert abs(actual - exact) < 2e-7
    assert max(abs(np.array(report["float64_before_commit_moments"][
        "scaled_xy_moment_residuals_over_strength_l1"]))) < 2e-14
    with pytest.raises(RuntimeError, match="capacity"):
        remesh(position, strength, np.array([-.181, -.123, 0.]), .04, kernel, 2, .08)


def test_cardinal_scatter_preserves_actual_gaussian_field_and_independent_curl():
    position = np.array([[1.5, 0., 0.], [1.75, .25, 0.]], dtype=np.float32)
    strength = np.zeros_like(position)
    strength[:, 2] = [1., -.25]
    output, deposited, _ = remesh(position, strength, np.zeros(3), .125,
                                  "M4_PRIME", 200000, .08)
    points = np.array([[1.5, 0., 0.], [1.61, .04, 0.], [1.8, -.03, 0.], [1.91, .13, 0.]])
    exact = induced_fields(points, position, strength, .04, 1., 1)
    remeshed = induced_fields(points, output, deposited, .04, 1., 2)
    for first, second in zip(exact, remeshed, strict=True):
        np.testing.assert_allclose(first, second, rtol=2e-15, atol=2e-15)
    squared = ((points[:, None, :2] - position[None, :, :2])**2).sum(axis=2)
    gamma = strength[:, 2].astype(np.float64)
    independent_curl = (np.exp(-squared / .04**2) * gamma[None, :]
                        / (np.pi * .04**2)).sum(axis=1)
    np.testing.assert_allclose(exact[2], independent_curl, rtol=1e-13, atol=2e-13)


def test_resolved_wake_band_reports_actual_content_and_signed_gain():
    points, shape = wake_probes(3., .04)
    assert len(points) == 1000 and shape == (25, 40)
    assert np.isclose(points[:, 0].min() - 3., .4)
    x = points[:, 0]
    original = np.sin(2 * np.pi * x / .16) + .3 * x - .2 * points[:, 1] + 2.
    report = curl_spectral_bands(original, .75 * original, shape, .04)
    relevant = report["bands"]["wavelength_0.1_to_0.2m"]
    assert relevant["exact_curl_windowed_power_fraction"] > .9
    assert abs(relevant["rms_gain"] - .75) < 1e-14
    assert abs(relevant["least_squares_gain"] - .75) < 1e-14
    metrics = paired_statistics(np.array([1., -2.]), np.array([.5, -1.]))
    assert metrics["least_squares_amplitude_gain"] == .5
    assert metrics["rms_ratio"] == .5
    assert metrics["relative_l2_error"] == .5
