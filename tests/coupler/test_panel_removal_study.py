"""Independent analytical checks for offline panel-removal diagnostics."""

import numpy as np

from studies.panel_removal.operators import polygon_induction, polygon_sources, volume_induction


def test_polygon_vorticity_converges_to_rankine_vortex_and_is_winding_independent():
    theta = np.arange(512) * (2 * np.pi / 512)
    points = np.column_stack((np.cos(theta), np.sin(theta)))
    targets = np.array([[0.2, 0.1], [-0.1, 0.3], [2.0, 0.0], [0.0, -3.0]])
    radius2 = np.sum(targets**2, axis=1)
    exact = np.column_stack((-targets[:, 1], targets[:, 0])) / (2 * np.maximum(1, radius2[:, None]))
    for indices in (np.arange(512), np.arange(511, -1, -1)):
        sources, weights = polygon_sources(points, np.r_[512, indices], [1.0], order=8)
        actual = polygon_induction(targets, sources, weights)
        np.testing.assert_allclose(actual, exact, atol=7e-6, rtol=0)


def test_finite_span_matches_distant_straight_vortex_segment():
    half_side = 0.01
    points = half_side * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    area = (2 * half_side) ** 2
    sources, weights = polygon_sources(points, np.array([4, 0, 1, 2, 3]), [1 / area], order=8)
    targets = np.array([[2.0, 0.0], [0.0, 3.0]])
    for length in (1.0, 12.0):
        r2 = np.sum(targets**2, axis=1)
        factor = (length / 2) / np.sqrt(r2 + (length / 2) ** 2)
        exact = (
            np.column_stack((-targets[:, 1], targets[:, 0])) * (factor / (2 * np.pi * r2))[:, None]
        )
        np.testing.assert_allclose(
            polygon_induction(targets, sources, weights, length), exact, rtol=2e-5, atol=1e-12
        )


def test_volume_operator_far_field_sign_and_all_vector_components():
    position = np.array([[0.2, -0.1, 0.4], [-0.2, 0.3, -0.4]])
    strength = np.array([[0.3, -0.2, 0.5], [-0.1, 0.7, 0.2]])
    targets = np.array([[2.0, 3.0, 4.0], [-3.0, 1.0, 2.0]])
    delta = targets[:, None] - position[None]
    expected = (
        np.cross(strength[None], delta)
        / (4 * np.pi * np.linalg.norm(delta, axis=-1)[..., None] ** 3)
    ).sum(axis=1)
    np.testing.assert_allclose(
        volume_induction(targets, position, strength, np.array([0.01, 0.02])), expected, rtol=2e-14
    )


def test_volume_gaussian_self_velocity_is_zero_and_core_is_regular():
    position = np.zeros((1, 3))
    strength = np.array([[0.0, 0.0, 1.0]])
    targets = np.array([[0.0, 0.0, 0.0], [1e-3, 0.0, 0.0], [-1e-3, 0.0, 0.0]])
    actual = volume_induction(targets, position, strength, np.array([0.5]))
    np.testing.assert_array_equal(actual[0], 0)
    np.testing.assert_allclose(actual[1], -actual[2], rtol=1e-14)
    # Leading Gaussian-core expansion: u_theta/r = sqrt(2/pi)/(12*pi*sigma^3).
    assert np.isclose(actual[1, 1] / 1e-3, np.sqrt(2 / np.pi) / (12 * np.pi * 0.5**3), rtol=2e-6)
