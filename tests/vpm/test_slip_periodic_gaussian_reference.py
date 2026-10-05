"""Independent infinite-image reference qualification, small CPU clouds only.

No empirical production tail check is used. Tests compare the analytic absolute
remainder against longer sums, check an analytic nonzero spanwise-zero-mode
velocity/Jacobian, and validate physical/source-only core conventions.
"""

import numpy as np
import pytest

from tests.vpm._slip_periodic_gaussian_reference import (
    gaussian_pairs,
    image_remainder_bounds,
    slip_periodic_gaussian,
)


def _cloud():
    return (
        np.array([[0.1, -0.2, 0.12], [-0.12, 0.24, 0.38]]),
        np.array([[0.2, -0.3, 0.7], [-0.2, 0.3, -0.699]]),
        np.array([0.08, 0.17]),
        np.array([[0.3, 0.1, 0.0], [-0.2, 0.25, 0.5], [0.2, 0.3, 0.24]]),
    )


@pytest.mark.parametrize("point", [[0, 0, 0], [1e-5, -2e-5, 3e-5], [0.2, -0.3, 0.4]])
def test_independent_full_jacobian_and_finite_core(point):
    point, strength = np.asarray(point, dtype=float), np.array([0.3, -0.2, 0.7])
    sigma, delta = 0.13, 2e-6
    velocity, gradient = gaussian_pairs(point, strength, sigma)
    finite_difference = np.column_stack(
        [
            (
                gaussian_pairs(point + np.eye(3)[axis] * delta, strength, sigma)[0]
                - gaussian_pairs(point - np.eye(3)[axis] * delta, strength, sigma)[0]
            )
            / (2 * delta)
            for axis in range(3)
        ]
    )
    np.testing.assert_allclose(gradient, finite_difference, rtol=2e-8, atol=2e-8)
    assert abs(np.trace(gradient)) < 2e-14
    if np.all(point == 0):
        np.testing.assert_array_equal(velocity, 0)
        assert np.linalg.norm(gradient) > 0


@pytest.mark.parametrize("shells", [2, 8, 32])
def test_analytic_tail_bounds_enclose_longer_cancelled_mixed_core_sum(shells):
    args = _cloud()
    short = slip_periodic_gaussian(*args, z_min=0, z_max=0.5, shells=shells)
    long = slip_periodic_gaussian(*args, z_min=0, z_max=0.5, shells=2048)
    velocity_difference = np.linalg.norm(short.velocity - long.velocity, axis=1)
    gradient_difference = np.linalg.norm(short.gradient - long.gradient, axis=(1, 2))
    assert np.all(velocity_difference <= short.velocity_tail_bound + long.velocity_tail_bound)
    assert np.all(gradient_difference <= short.gradient_tail_bound + long.gradient_tail_bound)


def test_tail_error_bound_is_absolute_monotone_and_work_bounded():
    args = _cloud()
    previous = None
    for shells in (2, 4, 8, 16, 32):
        bounds = np.stack(image_remainder_bounds(*args, z_min=0, z_max=0.5, shells=shells))
        if previous is not None:
            assert np.all(bounds < previous)
        previous = bounds
    result = slip_periodic_gaussian(
        *args, z_min=0, z_max=0.5, velocity_tolerance=1e-6, gradient_tolerance=1e-6
    )
    assert np.max(result.velocity_tail_bound) <= 1e-6
    assert np.max(result.gradient_tail_bound) <= 1e-6
    assert np.isfinite(result.summation_roundoff_indicator).all()
    with pytest.raises(RuntimeError, match="shell budget"):
        slip_periodic_gaussian(*args, z_min=0, z_max=0.5, velocity_tolerance=1e-15, max_shells=8)


def test_slip_planes_and_periodicity_with_full_vector_sources():
    position, strength, radius, targets = _cloud()
    result = slip_periodic_gaussian(
        position, strength, radius, targets, z_min=0, z_max=0.5, shells=2048
    )
    assert np.all(np.abs(result.velocity[:2, 2]) <= result.velocity_tail_bound[:2] + 1e-13)
    for row, col in ((0, 2), (1, 2), (2, 0), (2, 1)):
        assert np.all(
            np.abs(result.gradient[:2, row, col]) <= result.gradient_tail_bound[:2] + 1e-13
        )
    shifted = slip_periodic_gaussian(
        position, strength, radius, targets + [0, 0, 1], z_min=0, z_max=0.5, shells=2048
    )
    assert np.all(
        np.linalg.norm(result.velocity - shifted.velocity, axis=1)
        <= result.velocity_tail_bound + shifted.velocity_tail_bound
    )
    assert np.all(
        np.linalg.norm(result.gradient - shifted.gradient, axis=(1, 2))
        <= result.gradient_tail_bound + shifted.gradient_tail_bound
    )


def test_nonzero_axial_circulation_zero_mode_matches_analytic_line_vortex():
    # z-averaging the extended Gaussian family gives a 2-D Gaussian vortex of
    # circulation Gamma_z/L, even when the net 3-D source circulation is nonzero.
    length, circulation, sigma, distance = 0.5, 0.7, 0.09, 0.4
    targets = np.column_stack((np.full(64, distance), np.zeros(64), np.arange(64) / 64))
    result = slip_periodic_gaussian(
        [[0, 0, 0.173]],
        [[0, 0, circulation]],
        [sigma],
        targets,
        z_min=0,
        z_max=length,
        shells=4096,
    )
    exponential = np.exp(-((distance / sigma) ** 2))
    factor = circulation / (2 * np.pi * length)
    velocity = np.array([0, factor * (1 - exponential) / distance, 0])
    gradient = np.zeros((3, 3))
    gradient[0, 1] = -factor * (1 - exponential) / distance**2
    gradient[1, 0] = factor * (2 * exponential / sigma**2 - (1 - exponential) / distance**2)
    assert (
        np.linalg.norm(result.velocity.mean(0) - velocity)
        <= result.velocity_tail_bound.mean() + 1e-12
    )
    assert (
        np.linalg.norm(result.gradient.mean(0) - gradient)
        <= result.gradient_tail_bound.mean() + 1e-12
    )
    assert np.linalg.norm(velocity) > 0.5  # A discarded zero mode cannot pass.


def test_physical_mean_core_changes_only_primary_and_includes_self_jacobian():
    position, strength, radius, _ = _cloud()
    target_radius = np.array([0.2, 0.31])
    common = {"z_min": 0, "z_max": 0.5, "shells": 128}
    source_only = slip_periodic_gaussian(position, strength, radius, position, **common)
    mean_core = slip_periodic_gaussian(
        position, strength, radius, position, target_radius=target_radius, **common
    )
    images = slip_periodic_gaussian(position, strength, radius, position, image_only=True, **common)
    displacement = position[:, None, :] - position[None, :, :]
    u_source, j_source = gaussian_pairs(displacement, strength[None, :, :], radius[None, :])
    u_mean, j_mean = gaussian_pairs(
        displacement, strength[None, :, :], (target_radius[:, None] + radius[None, :]) / 2
    )
    np.testing.assert_allclose(source_only.velocity - images.velocity, u_source.sum(1), atol=2e-14)
    np.testing.assert_allclose(source_only.gradient - images.gradient, j_source.sum(1), atol=2e-13)
    np.testing.assert_allclose(mean_core.velocity - images.velocity, u_mean.sum(1), atol=2e-14)
    np.testing.assert_allclose(mean_core.gradient - images.gradient, j_mean.sum(1), atol=2e-13)
    assert np.all(np.linalg.norm(j_source[np.arange(2), np.arange(2)], axis=(1, 2)) > 0)


def test_chunking_and_zero_strength_are_safe():
    args = _cloud()
    left = slip_periodic_gaussian(*args, z_min=0, z_max=0.5, shells=128, chunk_shells=7)
    right = slip_periodic_gaussian(*args, z_min=0, z_max=0.5, shells=128, chunk_shells=128)
    np.testing.assert_allclose(left.velocity, right.velocity, atol=1e-14)
    np.testing.assert_allclose(left.gradient, right.gradient, atol=1e-13)
    zero = slip_periodic_gaussian(
        [[0, 0, 0.1]], [[0, 0, 0]], [0.1], [[10, 20, 30]], z_min=0, z_max=0.5
    )
    np.testing.assert_array_equal(zero.velocity, 0)
    np.testing.assert_array_equal(zero.gradient, 0)
    np.testing.assert_array_equal(zero.velocity_tail_bound, 0)
    np.testing.assert_array_equal(zero.gradient_tail_bound, 0)


@pytest.mark.parametrize("scale", [1e-4, 10.0])
def test_field_and_analytic_error_bound_have_correct_physical_scaling(scale):
    position, strength, radius, targets = _cloud()
    original = slip_periodic_gaussian(
        position, strength, radius, targets, z_min=0, z_max=0.5, shells=32
    )
    scaled = slip_periodic_gaussian(
        scale * position,
        scale**2 * strength,
        scale * radius,
        scale * targets,
        z_min=0,
        z_max=0.5 * scale,
        shells=32,
    )
    np.testing.assert_allclose(scaled.velocity, original.velocity, rtol=3e-13, atol=1e-13)
    np.testing.assert_allclose(scaled.gradient * scale, original.gradient, rtol=3e-13, atol=1e-12)
    np.testing.assert_allclose(scaled.velocity_tail_bound, original.velocity_tail_bound, rtol=1e-14)
    np.testing.assert_allclose(
        scaled.gradient_tail_bound * scale, original.gradient_tail_bound, rtol=1e-14
    )


def test_coincident_boundary_images_have_axial_vector_parity():
    result = slip_periodic_gaussian(
        [[0, 0, 0]],
        [[1, -2, 0]],
        [0.1],
        [[0, 0, 0], [0.3, 0.4, 0.2]],
        z_min=0,
        z_max=0.5,
        shells=32,
    )
    # At z_min these transverse axial-vector sources and their odd images
    # coincide with exactly opposite strengths, including finite self J.
    np.testing.assert_allclose(result.velocity, 0, atol=1e-14)
    np.testing.assert_allclose(result.gradient, 0, atol=1e-13)


def test_finite_but_overflowing_geometry_cannot_return_nan_error_bound():
    with pytest.raises(FloatingPointError, match="tail scale overflow"):
        slip_periodic_gaussian(
            [[0, 0, 0.1]], [[0, 0, 0]], [0.1], [[1e200, 0, 0]], z_min=0, z_max=0.5
        )
