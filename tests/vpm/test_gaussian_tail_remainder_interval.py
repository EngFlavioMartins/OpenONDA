"""Small exact-moment and independent many-shell remainder checks."""

from fractions import Fraction

import numpy as np
import pytest

from tests.vpm._gaussian_image_tail_moments import moment_tail_bound
from tests.vpm._gaussian_tail_arithmetic import leading_tail
from tests.vpm._gaussian_tail_remainder_interval import tail_remainder
from tests.vpm.test_gaussian_image_tail_moments import cloud, explicit_tail


@pytest.mark.parametrize("kind", ["random", "cancelled", "axial", "translated", "near_admission"])
def test_independent_many_shell_tail_is_enclosed_after_leading_completion(kind):
    x, g, sigma, t, zmin, zmax, k = cloud(kind)
    saved = [value.copy() for value in (x, g, sigma, t)]
    result = tail_remainder(x, g, sigma, t, z_min=zmin, z_max=zmax, shells=k)
    leading = leading_tail(x, g, t, z_min=zmin, z_max=zmax, shells=k)
    u, j = explicit_tail(x, g, sigma, t, zmin, zmax, k+1, 512)
    end_remainder = tail_remainder(x, g, sigma, t, z_min=zmin, z_max=zmax, shells=512)
    end_leading = leading_tail(x, g, t, z_min=zmin, z_max=zmax, shells=512)
    # The finite explicit sum differs from the infinite sum by the remaining
    # M=512 tail. Bound it independently instead of treating it as exact zero.
    end_u = np.linalg.norm(end_leading.velocity, axis=1)+end_leading.velocity_error_bound+end_remainder.velocity
    end_j = np.linalg.norm(end_leading.gradient, axis=(1, 2))+end_leading.gradient_error_bound+end_remainder.gradient
    assert np.all(np.linalg.norm(u-leading.velocity, axis=1)
                  <= result.velocity+leading.velocity_error_bound+end_u+2e-14)
    assert np.all(np.linalg.norm(j-leading.gradient, axis=(1, 2))
                  <= result.gradient+leading.gradient_error_bound+end_j+2e-14)
    old = moment_tail_bound(x, g, sigma, t, z_min=zmin, z_max=zmax, shells=k)
    assert np.all(result.singular_velocity >= old.singular_velocity_remainder)
    assert np.all(result.singular_gradient >= old.singular_gradient_remainder)
    for actual, original in zip((x, g, sigma, t), saved, strict=True):
        np.testing.assert_array_equal(actual, original)


def test_origin_centred_weighted_moments_enclose_exact_rational_values():
    x = np.array([[1024.1, -128.2, 8.1], [1024.3, -127.8, 8.8], [1024.7, -128., 8.4]])
    g = np.array([[1e12, .01, -1e12], [-1e12, 1e-8, 1e12], [.1, -.2, .3]])
    result = tail_remainder(x, g, np.ones(3)*.04, x, z_min=8., z_max=9., shells=16)
    exact_x = [[Fraction(float(v)) for v in row] for row in x]
    exact_g = [[Fraction(float(v)) for v in row] for row in g]
    origin = [Fraction(v) for v in result.diagnostics["origin"]]
    w = [sum((abs(v) for v in row), Fraction()) for row in exact_g]
    for family in result.diagnostics["families"]:
        local = [[p[i]-origin[i] for i in range(3)] for p in exact_x]
        if family["odd"]:
            for p in local:
                p[2] = -p[2]
        s0 = sum(w, Fraction())
        s1 = [sum((weight*p[i] for weight, p in zip(w, local, strict=True)), Fraction()) for i in range(3)]
        s2 = sum((weight*sum((v*v for v in p), Fraction()) for weight, p in zip(w, local, strict=True)), Fraction())
        for name, exact in (("source_strength", s0), ("second_moment", s2)):
            assert Fraction(family[name]["lower"]) <= exact <= Fraction(family[name]["upper"])
        for i, exact in enumerate(s1):
            assert Fraction(family["first_moment"]["lower"][i]) <= exact <= Fraction(family["first_moment"]["upper"][i])


def test_singular_remainder_decays_with_more_complete_shells():
    x, g, sigma, t, zmin, zmax, _ = cloud("random")
    small = tail_remainder(x, g, sigma, t, z_min=zmin, z_max=zmax, shells=8)
    large = tail_remainder(x, g, sigma, t, z_min=zmin, z_max=zmax, shells=16)
    assert np.all(large.velocity < small.velocity/10)
    assert np.all(large.gradient < small.gradient/10)


@pytest.mark.parametrize("count", [0, 5])
def test_zero_sources_and_zero_strength_have_no_tail(count):
    result = tail_remainder(np.zeros((count, 3)), np.zeros((count, 3)), np.ones(count),
                             np.zeros((2, 3)), z_min=-.5, z_max=.5, shells=4)
    np.testing.assert_array_equal(result.velocity, 0.)
    np.testing.assert_array_equal(result.gradient, 0.)


@pytest.mark.parametrize("option,value", [("shells", True), ("shells", 0), ("max_sources", 2),
                                          ("max_targets", 1)])
def test_invalid_work_limits_fail_closed(option, value):
    options = {"z_min": -.5, "z_max": .5, "shells": 4, option: value}
    with pytest.raises(ValueError):
        tail_remainder(np.zeros((3, 3)), np.ones((3, 3)), np.ones(3)*.04, np.zeros((2, 3)), **options)


def test_insufficient_separation_and_nonfinite_fields_rejected():
    x, g, sigma = np.zeros((1, 3)), np.ones((1, 3)), np.ones(1)*.04
    with pytest.raises(ValueError, match="L1 extent"):
        tail_remainder(x, g, sigma, np.array([[2., 2., 2.]]), z_min=-.5, z_max=.5, shells=1)
    g[0, 0] = np.inf
    with pytest.raises(ValueError):
        tail_remainder(x, g, sigma, x, z_min=-.5, z_max=.5, shells=4)
