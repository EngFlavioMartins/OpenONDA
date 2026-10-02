"""Independent analytic-field finite-block qualification, not runtime gates."""

import numpy as np
import pytest

from tests.vpm._finite_image_field_mesh_reference import (
    analytic_gaussian_derivatives,
    finite_image_field_mesh,
)
from tests.vpm._finite_image_mesh_reference import direct_finite_images
from tests.vpm._slip_periodic_gaussian_oracle import gaussian_pairs


def test_analytic_kernel_signs_origin_and_full_j_against_independent_pair_oracle():
    displacement = np.array([[0., 0., 0.], [1e-7, -2e-6, 1e-5], [.023, .041, -.018], [1.2, -.1, 4.1]])
    gamma = np.array([.3, -.8, .2])
    grad, hess = analytic_gaussian_derivatives(displacement, .12)
    exact_u, exact_j = gaussian_pairs(displacement, gamma, .12)
    u = np.cross(grad, gamma)
    j = np.stack([np.cross(hess[:, :, axis], gamma) for axis in range(3)], axis=-1)
    np.testing.assert_allclose(u, exact_u, rtol=3e-12, atol=1e-13)
    np.testing.assert_allclose(j, exact_j, rtol=3e-12, atol=1e-11)
    np.testing.assert_array_equal(u[0], 0)
    assert np.linalg.norm(j[0]) > 0


def _cloud():
    x = np.array([[-.047, .018, .073], [.056, -.036, .041], [.011, .029, .096]])
    gamma = np.array([[.3, -.2, .5], [-.7, .4, .2], [.4, -.2, .3]])
    sigma = np.array([.04, .07, .09])
    query = np.array([[.012, -.013, .068], [-.021, .027, .059]])
    return x, gamma, sigma, query


def test_image_only_nonzero_z_mixed_core_fields_and_trace_match_direct():
    x, gamma, sigma, query = _cloud()
    images = [(0., True), (.64, False), (-.64, False), (.64, True)]
    u, j, _ = direct_finite_images(x, gamma, sigma, query, images)
    result = finite_image_field_mesh(x, gamma, sigma, query, images, tau=.25, spacing=.025, order=8)
    for candidate, exact in ((result.velocity, u), (result.gradient, j)):
        assert np.linalg.norm(candidate-exact)/np.linalg.norm(exact) < 1e-4
    np.testing.assert_allclose(np.trace(result.gradient, axis1=1, axis2=2), 0, atol=1e-10)
    assert not result.diagnostics["interpolated_u_J_derivative_consistent"]
    assert not result.diagnostics["exact_coincidence_override"]


@pytest.mark.parametrize("odd", [False, True])
def test_same_stencil_coincident_velocity_cancels_without_override(odd):
    x = np.array([[.013, -.017, .081]])
    gamma, sigma = np.array([[.2, .7, -.4]]), np.array([.04])
    images = [(.162 if odd else 0., odd)]
    u, j, _ = direct_finite_images(x, gamma, sigma, x, images)
    result = finite_image_field_mesh(x, gamma, sigma, x, images, tau=.12, spacing=.04, order=10)
    np.testing.assert_allclose(result.velocity, u, rtol=0, atol=1e-10)
    assert np.linalg.norm(result.gradient-j)/np.linalg.norm(j) < 1e-4


def test_noncommensurate_reflected_coincidence_is_not_hidden_by_single_source_recentering():
    x = np.array([[.013, -.017, .081], [.036, .027, .137]])
    gamma, sigma = np.array([[.2, .7, -.4], [0., 0., 0.]]), np.array([.04, .04])
    query, images = x[:1], [(.162, True)]
    u, j, _ = direct_finite_images(x, gamma, sigma, query, images)
    result = finite_image_field_mesh(x, gamma, sigma, query, images, tau=.2, spacing=.025, order=8)
    # This case has unequal mesh phases: it is convergence, NOT exact W^T K W.
    assert result.diagnostics["z_recentering"] != .081
    np.testing.assert_allclose(result.velocity, u, rtol=0, atol=1e-5)
    assert np.linalg.norm(result.gradient-j)/np.linalg.norm(j) < 1e-4


@pytest.mark.parametrize("tau,spacing,order", [
    (.25, .025, 8),
    pytest.param(.12, .04, 10, marks=pytest.mark.xfail(
        strict=True, reason="Preserved negative qualification: derivative error1.65e-4 exceeds unchanged1e-4 guard")),
])
def test_within_stencil_velocity_derivative_discrepancy_is_measured_not_assumed_zero(tau, spacing, order):
    x, gamma, sigma, query = _cloud()
    images = [(0., True), (.64, False)]
    options = {"tau": tau, "spacing": spacing, "order": order}
    result = finite_image_field_mesh(x, gamma, sigma, query, images, **options)
    epsilon = 1e-6
    finite_difference = np.empty_like(result.gradient)
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = epsilon
        plus = finite_image_field_mesh(x, gamma, sigma, query+step, images, **options)
        minus = finite_image_field_mesh(x, gamma, sigma, query-step, images, **options)
        for key in ("z_recentering", "grid_shape", "fft_shape"):
            assert plus.diagnostics[key] == minus.diagnostics[key] == result.diagnostics[key]
        finite_difference[:, :, axis] = (plus.velocity-minus.velocity)/(2*epsilon)
    _, exact_j, _ = direct_finite_images(x, gamma, sigma, query, images)
    true_error = np.linalg.norm(finite_difference-exact_j)/np.linalg.norm(exact_j)
    discrepancy = np.linalg.norm(finite_difference-result.gradient)/np.linalg.norm(exact_j)
    assert true_error < 1e-4, (tau, spacing, order, true_error, discrepancy)
    assert discrepancy < 1e-4, (tau, spacing, order, true_error, discrepancy)


def test_far_finite_shift_and_resource_failure_preserve_inputs():
    x, gamma, sigma, query = _cloud()
    copies = [value.copy() for value in (x, gamma, sigma, query)]
    images = [(8., False), (-8., False), (8., True), (-8., True)]
    u, j, _ = direct_finite_images(x, gamma, sigma, query, images)
    result = finite_image_field_mesh(x, gamma, sigma, query, images, tau=.25, spacing=.04, order=6)
    np.testing.assert_allclose(result.velocity, u, rtol=1e-4, atol=1e-10)
    np.testing.assert_allclose(result.gradient, j, rtol=1e-4, atol=1e-10)
    with pytest.raises(ValueError, match="grid exceeded"):
        finite_image_field_mesh(x, gamma, sigma, query, images, tau=.25, spacing=.04, max_grid_nodes=10)
    for original, copy in zip((x, gamma, sigma, query), copies, strict=True):
        np.testing.assert_array_equal(original, copy)
