"""Partition-of-authority tests for the FVM/VPM overlap blending zone."""

import numpy as np

from source.coupler.core.helpers.continuous_overlap import cosine_eta
from source.coupler.core.helpers.fvm_blending_zone import build_lambda


def test_blending_zone_is_exact_complement_of_particle_authority():
    box = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    distance = np.linspace(0.0, 0.5, 101)
    points = np.column_stack((distance - 1.0, np.zeros_like(distance), np.zeros_like(distance)))
    width = 0.4
    dead_zone = 0.1
    lambda_max = 7.0

    eta = cosine_eta(points, box, width, dead_zone)
    lam = build_lambda(points, tuple(box), width, lambda_max, dead_zone)

    np.testing.assert_allclose(eta + lam / lambda_max, 1.0, atol=2e-15)
    np.testing.assert_allclose(lam[distance <= dead_zone], lambda_max)
    np.testing.assert_allclose(lam[distance >= width], 0.0)


def test_blending_zone_has_no_unowned_dead_zone():
    points = np.array([[-1.01, 0.0, 0.0], [-1.0, 0.0, 0.0], [-0.95, 0.0, 0.0]])
    lam = build_lambda(points, (-1, 1, -1, 1, -1, 1), 0.3, 4.0, dead_zone=0.1)
    np.testing.assert_allclose(lam, 4.0)
