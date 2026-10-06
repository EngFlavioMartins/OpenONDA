"""Independent kinematic checks of the frozen circular potential prototype."""

import numpy as np
import pytest

from tests.support.cylinder.audit_circular_wall_potential import (
    fit_circular_wall_potential,
    potential_field,
    validate_potential,
)


def circle_facets(count=104, radius=0.5):
    angles = np.arange(count) * 2 * np.pi / count
    points = radius * np.column_stack((np.cos(angles), np.sin(angles), np.zeros(count)))
    return points, points / radius, np.full(count, 2 * np.pi * radius / count)


def test_uniform_background_recovers_classical_cylinder_dipole():
    wall, normals, area = circle_facets()
    background = np.broadcast_to(np.array([1.0, 0.0, 0.0]), wall.shape)
    coefficients, fit = fit_circular_wall_potential(wall, normals, area, background, modes=16)
    expected_coefficients = np.zeros(32)
    expected_coefficients[0] = 1
    np.testing.assert_allclose(coefficients, expected_coefficients, atol=2e-14)
    _, corrected, _ = potential_field(wall, coefficients)
    np.testing.assert_allclose(
        np.einsum("fi,fi->f", corrected + background, normals), 0, atol=3e-14
    )
    assert fit["condition"] < 2
    points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.8, 0.9, 0.0], [-0.7, 0.6, 0.0]])
    _, velocity, _ = potential_field(points, coefficients)
    x, y = points[:, 0], points[:, 1]
    r2 = x * x + y * y
    analytic = np.column_stack(
        (0.25 * (y * y - x * x) / r2**2, -0.5 * x * y / r2**2, np.zeros(len(points)))
    )
    np.testing.assert_allclose(velocity, analytic, rtol=1e-13, atol=1e-14)
    result = validate_potential(coefficients, points)
    assert result["maximum_divergence"] == 0
    assert result["maximum_curl"] == 0
    assert abs(result["circular_flux"]) < 1e-14
    assert abs(result["circular_circulation"]) < 1e-14
    decay = [row["velocity_rms"] for row in result["radial_decay"]]
    np.testing.assert_allclose(decay, [1, 0.25, 0.0625, 1 / 256], atol=2e-14)


def test_multiple_harmonic_modes_fit_signed_normal_velocity_and_analytic_gradient():
    wall, normals, area = circle_facets()
    coefficients = np.zeros(48)
    coefficients[[0, 3, 12, 21, 46]] = [0.035, -0.021, 0.014, -0.008, 0.003]
    _, negative_velocity, _ = potential_field(wall, -coefficients)
    reconstructed, fit = fit_circular_wall_potential(
        wall, normals, area, negative_velocity, modes=24
    )
    np.testing.assert_allclose(reconstructed, coefficients, atol=2e-14)
    assert fit["remaining_wall_normal_rms"] < 1e-14
    assert abs(fit["correction_wall_flux"]) < 1e-14
    result = validate_potential(reconstructed, wall)
    assert result["maximum_jacobian_finite_difference_error"] < 3e-8
    assert result["maximum_divergence"] == 0
    assert result["maximum_curl"] == 0


def test_rank_deficient_wall_collocation_is_rejected():
    points = np.tile(np.array([0.5, 0.0, 0.0]), (104, 1))
    normals = np.tile(np.array([1.0, 0.0, 0.0]), (104, 1))
    velocity = np.tile(np.array([0.1, 0.0, 0.0]), (104, 1))
    with pytest.raises(ValueError, match="condition bound"):
        fit_circular_wall_potential(points, normals, np.ones(104), velocity, modes=16)
