"""Unwired mathematical qualification only; no backend, JIT or GPU."""

import math

import numpy as np
import pytest

from tests.vpm._gaussian_broadening_reference import (
    broadening_tail_bound,
    correction_fields,
    gaussian_fields,
    gaussian_tail,
    singular_defect_bound,
    singular_fields,
)


def test_normalization_and_pair_core_match_repository_host_reference():
    # Importing the existing NumPy reference does not initialize Taichi.
    from source.solvers.vpm.kernels.base import make_vortex_kernel

    kernel = make_vortex_kernel("GAUSSIAN")
    rng = np.random.default_rng(276281)
    points = rng.normal(0., .15, (24, 3))
    points[0] = 0.
    gamma = rng.normal(size=(24, 3))
    target_sigma, source_sigma = rng.uniform(.02, .1, (2, 24))
    expected_u = kernel.velocity_pair(points, gamma, target_sigma, source_sigma)
    expected_j = kernel.gradient_pair(points, gamma, target_sigma, source_sigma)
    actual = [gaussian_fields(r, g, .5*(a+b)) for r, g, a, b in
              zip(points, gamma, target_sigma, source_sigma, strict=True)]
    np.testing.assert_allclose([v[0] for v in actual], expected_u, rtol=8e-14, atol=1e-12)
    np.testing.assert_allclose([v[1] for v in actual], expected_j, rtol=8e-14, atol=1e-12)


@pytest.mark.parametrize("sigma,tau", [(0.04, 0.1), (0.1, 0.1), (0.1, 10.0), (1.0, 1.000000000001)])
def test_exact_split_full_jacobian_and_finite_origin(sigma, tau):
    gamma = np.array([1.7, -0.3, 2.9])
    direction = np.array([2., -3., 7.]) / math.sqrt(62.)
    for r in [0., sigma*1e-9, sigma*.1, sigma, tau, tau*4., tau*10.]:
        physical = gaussian_fields(r*direction, gamma, sigma)
        broad = gaussian_fields(r*direction, gamma, tau)
        local = correction_fields(r*direction, gamma, sigma, tau)
        for expected, smooth, delta in zip(physical, broad, local, strict=True):
            np.testing.assert_allclose(smooth+delta, expected, rtol=8e-14, atol=2e-12)
    assert np.linalg.norm(physical[1]) > 0
    zero = gaussian_fields(np.zeros(3), gamma, sigma)
    np.testing.assert_array_equal(zero[0], 0.)
    assert np.linalg.norm(zero[1]) > 0  # Never apply an electrostatic self subtraction.


@pytest.mark.parametrize("radius", [0., 1e-4, .02, .1, 1.])
def test_correction_jacobian_is_derivative_of_velocity(radius):
    point = np.array([radius, -.7*radius, .2*radius])
    gamma = [2., -3., 1.]
    _, jacobian = correction_fields(point, gamma, .05, .14)
    h = 2e-7
    numerical = np.column_stack([
        (correction_fields(point+h*np.eye(3)[i], gamma, .05, .14)[0]
         - correction_fields(point-h*np.eye(3)[i], gamma, .05, .14)[0])/(2*h)
        for i in range(3)
    ])
    np.testing.assert_allclose(jacobian, numerical, rtol=8e-8, atol=2e-7)
    assert abs(np.trace(jacobian)) < 1e-10


@pytest.mark.parametrize("ratio", [1., 1.01, 2., 20.])
@pytest.mark.parametrize("cutoff_scale", [.05, .5, 1., 2., 5., 8.])
def test_uniform_local_tail_bound_adversarial_orientations(ratio, cutoff_scale):
    sigma, tau = .08/ratio, .08
    cutoff = cutoff_scale*tau
    bound = broadening_tail_bound(cutoff, sigma, tau, math.sqrt(6.))
    for radius in cutoff*np.geomspace(1., 20., 18):
        for gamma in ([0., math.sqrt(6.), 0.], [math.sqrt(6.), 0., 0.], [1., -2., 1.]):
            u, jacobian = correction_fields([radius, 0., 0.], gamma, sigma, tau)
            assert np.linalg.norm(u) <= bound.velocity*(1+2e-13)+1e-290
            assert np.linalg.norm(jacobian) <= bound.gradient*(1+2e-13)+1e-290


def test_zero_net_strength_does_not_make_tail_error_zero():
    strengths = np.array([[0., 0., 3.], [0., 0., -3.]])
    displacements = [[.35, 0., 0.], [.5, 0., 0.]]
    np.testing.assert_array_equal(strengths.sum(axis=0), 0.)
    fields = [correction_fields(r, g, .04, .2) for r, g in zip(displacements, strengths, strict=True)]
    u = sum(value[0] for value in fields)
    jacobian = sum(value[1] for value in fields)
    assert np.linalg.norm(u) > 0.01
    bound = broadening_tail_bound(.35, .04, .2, sum(np.linalg.norm(strengths, axis=1)))
    assert np.linalg.norm(u) <= bound.velocity
    assert np.linalg.norm(jacobian) <= bound.gradient


@pytest.mark.parametrize("cutoff_scale", [.1, 1., 3., 7.])
def test_gaussian_minus_singular_defect_bound(cutoff_scale):
    sigma = .2
    cutoff = cutoff_scale*sigma
    gamma = np.array([2., -1., 3.])
    bound = singular_defect_bound(cutoff, sigma, np.linalg.norm(gamma))
    for radius in cutoff*np.geomspace(1., 2., 10):
        gaussian = gaussian_fields([radius, 0., 0.], gamma, sigma)
        singular = singular_fields([radius, 0., 0.], gamma)
        # Subtraction rounds away an exponentially small true defect. Charge
        # this TEST subtraction's f64 rounding, not the analytic bound.
        for index, limit in enumerate([bound.velocity, bound.gradient]):
            rounding = 8*np.finfo(float).eps*np.linalg.norm(singular[index])
            assert np.linalg.norm(gaussian[index]-singular[index]) <= limit+rounding


def test_images_source_only_and_physical_mean_are_distinct_exact_splits():
    point, gamma = [.13, .02, -.07], [1., -2., 3.]
    source_sigma, target_sigma, tau = .03, .11, .2
    image_sigma = source_sigma
    physical_sigma = (source_sigma+target_sigma)/2.
    assert np.linalg.norm(gaussian_fields(point, gamma, image_sigma)[1]
                          - gaussian_fields(point, gamma, physical_sigma)[1]) > 1.
    for actual in (image_sigma, physical_sigma):
        a, b = gaussian_fields(point, gamma, tau), correction_fields(point, gamma, actual, tau)
        for reconstructed, exact in zip([a[0]+b[0], a[1]+b[1]], gaussian_fields(point, gamma, actual), strict=True):
            np.testing.assert_allclose(reconstructed, exact, rtol=5e-14, atol=1e-12)


def test_odd_image_axial_parity_and_full_jacobian():
    reflection = np.diag([1., 1., -1.])
    point = np.array([.06, -.13, .2])
    source = np.array([-.11, .04, .12])
    gamma = np.array([.8, -.9, 1.4])
    shift = np.array([0., 0., -.08])
    direct = correction_fields(point-(reflection@source+shift), -reflection@gamma, .04, .15)
    query = correction_fields(reflection@(point-shift)-source, gamma, .04, .15)
    np.testing.assert_allclose(direct[0], reflection@query[0], rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(direct[1], reflection@query[1]@reflection, rtol=1e-14, atol=1e-14)


def test_erfc_tail_avoids_catastrophic_far_subtraction():
    assert gaussian_tail(8., 1.) > 0.
    assert 1./(4*math.pi) - (1./(4*math.pi) - gaussian_tail(8., 1.)) == 0.


@pytest.mark.parametrize("args", [(0., .1, .2), (1., .2, .1), (1., 0., .2), (1., .1, math.inf)])
def test_invalid_broadening_certificate_inputs_rejected(args):
    with pytest.raises(ValueError):
        broadening_tail_bound(*args)
