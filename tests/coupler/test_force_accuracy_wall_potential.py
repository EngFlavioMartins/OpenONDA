"""Independent physical checks of the autonomous cylinder potential control."""

import numpy as np

from tests.support.cylinder.wall_potential_control import harmonic_velocity_gradient


def test_unit_freestream_cylinder_dipole_and_far_field_decay():
    points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 0.0, 0.0]])
    velocity, gradient = harmonic_velocity_gradient(points, np.array([1.0, 0.0]))
    np.testing.assert_allclose(velocity, [[-0.25, 0, 0], [0.25, 0, 0], [-0.0625, 0, 0]])
    np.testing.assert_allclose(gradient[0], [[0.5, 0, 0], [0, -0.5, 0], [0, 0, 0]])


def test_harmonic_modes_preserve_curl_divergence_flux_and_circulation():
    theta = np.arange(1024) * 2 * np.pi / 1024
    radial = np.column_stack((np.cos(theta), np.sin(theta), np.zeros(len(theta))))
    tangent = np.column_stack((-np.sin(theta), np.cos(theta), np.zeros(len(theta))))
    coefficient = np.zeros(32)
    coefficient[[0, 3, 12, 21, 30]] = [0.05, -0.02, 0.014, 0.008, -0.005]
    velocity, gradient = harmonic_velocity_gradient(0.75 * radial, coefficient)
    np.testing.assert_allclose(np.trace(gradient, axis1=1, axis2=2), 0, atol=1e-15)
    np.testing.assert_allclose(gradient[:, 1, 0] - gradient[:, 0, 1], 0, atol=1e-15)
    assert abs(np.sum(velocity * radial)) < 1e-13
    assert abs(np.sum(velocity * tangent)) < 1e-13
    for axis in (0, 1):
        offset = np.zeros_like(radial)
        offset[:, axis] = 1e-6
        finite_difference = (
            harmonic_velocity_gradient(0.75 * radial + offset, coefficient)[0]
            - harmonic_velocity_gradient(0.75 * radial - offset, coefficient)[0]
        ) / 2e-6
        np.testing.assert_allclose(finite_difference, gradient[:, :, axis], atol=3e-11, rtol=1e-8)
