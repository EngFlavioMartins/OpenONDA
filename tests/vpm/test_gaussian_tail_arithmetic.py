"""Exact-rational checks of the new, unwired cancellation-safe arithmetic."""

from decimal import Decimal, localcontext
from fractions import Fraction as F  # noqa: N817 - compact exact-arithmetic notation
import math

import numpy as np
import pytest

from tests.vpm._gaussian_image_tail_moments import moment_tail_bound
from tests.vpm._gaussian_tail_arithmetic import (
    gaussian_defect_upper,
    leading_tail,
    pairwise_sum,
    pi_interval,
    point,
)


def exact_moments(x, g, origin):
    x, g, origin = [[F(float(v)) for v in row] for row in x], [[F(float(v)) for v in row] for row in g], [F(float(v)) for v in origin]
    net = 2*sum((row[2] for row in g), F())
    mx = sum((-4*v[1]*(p[2]-origin[2])-2*v[2]*(p[1]-origin[1]) for p, v in zip(x, g, strict=True)), F())
    my = sum((4*v[0]*(p[2]-origin[2])+2*v[2]*(p[0]-origin[0]) for p, v in zip(x, g, strict=True)), F())
    return net, mx, my


@pytest.mark.parametrize("translated,cancelled", [(False, False), (True, False), (False, True), (True, True)])
def test_moments_and_leading_field_enclose_exact_rational_calculation(translated, cancelled):
    rng = np.random.default_rng(867)
    x = rng.uniform(-.4, .4, (11, 3))
    g = rng.normal(size=x.shape)
    t = rng.uniform(-1., 1., (7, 3))
    zmin, zmax = -.5, .5
    if cancelled:
        g[::2] *= 1e12
        g[1:10:2] = -g[:10:2]
        g[-1] *= 1e-12
    if translated:
        offset = np.array([2.**30, -2.**28, 2.**20])
        x, t = x+offset, t+offset
        zmin, zmax = zmin+offset[2], zmax+offset[2]
    saved = [v.copy() for v in (x, g, t)]
    result = leading_tail(x, g, t, z_min=zmin, z_max=zmax, shells=64)
    net, mx, my = exact_moments(x, g, result.moments["origin"])
    for key, value in zip(("net_z", "moment_x", "moment_y"), (net, mx, my), strict=True):
        interval = result.moments[key]
        assert F(float(interval.lower)) <= value <= F(float(interval.upper))
    period = 2*(F(float(zmax))-F(float(zmin)))
    h3, pi = result.moments["h3"], pi_interval()
    c_lower = F(h3.lower)/(2*F(float(pi.upper))*period**3)
    c_upper = F(h3.upper)/(2*F(float(pi.lower))*period**3)
    origin = result.moments["origin"]
    for i, target in enumerate(t):
        bx = -net*(F(float(target[1]))-F(float(origin[1])))-mx
        by = net*(F(float(target[0]))-F(float(origin[0])))-my
        for axis, value in enumerate((bx, by, F())):
            endpoints = (c_lower*value, c_upper*value)
            assert F(float(result.velocity_interval.lower[i, axis])) <= min(endpoints)
            assert F(float(result.velocity_interval.upper[i, axis])) >= max(endpoints)
        for row, col, value in ((0, 1, -net), (1, 0, net)):
            endpoints = (c_lower*value, c_upper*value)
            assert F(float(result.gradient_interval.lower[i, row, col])) <= min(endpoints)
            assert F(float(result.gradient_interval.upper[i, row, col])) >= max(endpoints)
    for actual, original in zip((x, g, t), saved, strict=True):
        np.testing.assert_array_equal(actual, original)


def test_signed_pairwise_sum_does_not_use_cancelled_result_as_conditioning():
    values = np.array([1e20, 1., -1e20, 3., 1e-20, -2.])
    result = pairwise_sum(point(values))
    exact = sum((F(float(v)) for v in values), F())
    assert F(float(result.lower)) <= exact <= F(float(result.upper))
    assert result.upper-result.lower > 0


def test_pi_interval_has_independent_decimal_pi_inside():
    result = pi_interval()
    with localcontext() as context:
        context.prec = 70
        pi = Decimal("3.1415926535897932384626433832795028841971693993751058209749445923078")
        assert Decimal(float(result.lower)) < pi < Decimal(float(result.upper))


def test_returned_constant_storage_cannot_poison_later_calls():
    result = pi_interval()
    result.lower[...] = -100.
    assert pi_interval().lower > 3.


def test_nominal_affine_field_derivative_and_prior_helper_agreement():
    rng = np.random.default_rng(72)
    x, g, t = rng.uniform(-.4, .4, (15, 3)), rng.normal(size=(15, 3)), rng.normal(size=(4, 3))
    result = leading_tail(x, g, t, z_min=-.5, z_max=.5, shells=64)
    old = moment_tail_bound(x, g, np.ones(15)*.04, t, z_min=-.5, z_max=.5, shells=64)
    assert np.all(np.linalg.norm(result.velocity-old.leading_velocity, axis=1) < result.velocity_error_bound+1e-16)
    assert np.all(np.linalg.norm(result.gradient-old.leading_gradient, axis=(1, 2)) < result.gradient_error_bound+1e-16)
    for axis in range(3):
        moved = t.copy()
        moved[:, axis] += 1e-3
        other = leading_tail(x, g, moved, z_min=-.5, z_max=.5, shells=64)
        np.testing.assert_allclose((other.velocity-result.velocity)/1e-3, result.gradient[:, :, axis], rtol=1e-10, atol=1e-17)


@pytest.mark.parametrize("gap,sigma", [(2., 1.), (4., 2.), (18., .04), (120., .04)])
def test_rational_gaussian_encloses_high_precision_analytic_defect(gap, sigma):
    period, strength = 1.92, 15.59
    result = gaussian_defect_upper(gap_lower=gap, sigma_upper=sigma, period_lower=period,
                                   absolute_strength_upper=strength)
    with localcontext() as context:
        context.prec = 90
        d, s, a, strength = map(Decimal, (gap, sigma, period, strength))
        pi = Decimal("3.141592653589793238462643383279502884197169399375105820974944592307816406286208998")
        exponential = (-(d/s)**2).exp()
        common = strength*s*s*exponential/(a*d)
        cu = (s/d**3+2/(s*d))/(4*pi*pi.sqrt())
        cj = Decimal(5).sqrt()*(s/d**4+2/(s*d*d))/(4*pi*pi.sqrt())+1/(pi*pi.sqrt()*s**3)
        assert Decimal(result["velocity_upper"]) >= common*cu
        assert Decimal(result["gradient_frobenius_upper"]) >= common*cj
        assert Decimal(result["exponential_upper"]) >= exponential
        assert result["velocity_upper"] > 0 and result["gradient_frobenius_upper"] > 0


def test_gaussian_domain_and_nonfinite_inputs_fail_closed():
    for gap in (.1, -1., math.inf):
        with pytest.raises(ValueError):
            gaussian_defect_upper(gap_lower=gap, sigma_upper=1., period_lower=2., absolute_strength_upper=1.)
    with pytest.raises(ValueError):
        gaussian_defect_upper(gap_lower=3., sigma_upper=1., period_lower=2., absolute_strength_upper=1., maximum_order=True)


@pytest.mark.parametrize("count", [0, 7])
def test_empty_and_zero_strength_fields_are_bounded(count):
    result = leading_tail(np.zeros((count, 3)), np.zeros((count, 3)), np.zeros((2, 3)),
                          z_min=-.5, z_max=.5, shells=4)
    np.testing.assert_array_equal(result.velocity, 0.)
    np.testing.assert_array_equal(result.gradient, 0.)
    assert np.all(result.velocity_error_bound >= 0)


def test_work_caps_and_source_domain_rejected():
    x, g, t = np.zeros((3, 3)), np.ones((3, 3)), np.zeros((2, 3))
    with pytest.raises(ValueError):
        leading_tail(x, g, t, z_min=-.5, z_max=.5, shells=4, max_sources=2)
    x[0, 2] = 1.
    with pytest.raises(ValueError):
        leading_tail(x, g, t, z_min=-.5, z_max=.5, shells=4)
