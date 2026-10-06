"""Independent physical checks for frozen planar field-audit calculations."""

import numpy as np

from tests.support.cylinder.audit_saved_transfer_fields import represented_curl
from tests.support.cylinder.audit_saved_wall_circulation import gaussian_velocity_and_gradient


def test_streamed_gaussian_curl_matches_independent_induction_jacobian():
    sources = np.array([[-0.31, 0.04, 0], [0.18, 0.31, 0], [0.09, -0.07, 0]])
    points = np.array([[0.08, -0.13, 0], [-0.21, 0.04, 0], [0.49, 0.22, 0]])
    strength = np.array([0.014, -0.009, 0.003])
    span = 1.7
    radius = 0.11
    _, gradient = gaussian_velocity_and_gradient(points, sources, strength / span, radius)
    induced_curl = gradient[:, 1, 0] - gradient[:, 0, 1]
    sampled = represented_curl(points, sources, strength, np.full(len(sources), radius), span)
    np.testing.assert_allclose(sampled, induced_curl, rtol=3e-14, atol=1e-16)


def test_streamed_gaussian_curl_respects_source_core_radii_and_span():
    sources = np.array([[0.0, 0.0, 0], [0.1, -0.2, 0]])
    strength = np.array([0.013, -0.018])
    radius = np.array([0.09, 0.14])
    span = 2.0
    points = np.array([[0.0, 0.0, 0], [0.03, -0.06, 0]])
    difference = points[:, None, :2] - sources[None, :, :2]
    expected = np.sum(
        strength[None, :]
        * np.exp(-np.sum(difference**2, axis=2) / radius[None, :] ** 2)
        / (span * np.pi * radius[None, :] ** 2),
        axis=1,
    )
    np.testing.assert_allclose(
        represented_curl(points, sources, strength, radius, span), expected, rtol=1e-15
    )
