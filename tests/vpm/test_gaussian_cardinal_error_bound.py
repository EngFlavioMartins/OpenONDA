"""Exact constants, scaling and honest conservatism of the unwired bound."""

from fractions import Fraction as F  # noqa: N817
import math

import pytest

from tests.vpm._gaussian_cardinal_error_bound import slab_cardinal_interpolation_bound


def test_central_stencil_constants_and_lebesgue_maximum_exactly():
    coefficients = [F(0)]*10
    for i in range(10):
        polynomial = [F(1)]
        denominator = 1
        for j in range(10):
            if i == j:
                continue
            root = F(2*j-9, 2)
            other = [F(0)]*(len(polynomial)+1)
            for n, coefficient in enumerate(polynomial):
                other[n] -= root*coefficient
                other[n+1] += coefficient
            polynomial = other
            denominator *= i-j
        sign = 1 if polynomial[0]/denominator > 0 else -1
        for n, coefficient in enumerate(polynomial):
            coefficients[n] += sign*coefficient/denominator
    assert coefficients == [F(25609, 16384), 0, -F(22349, 9216), 0, F(3259, 4608), 0,
                            -F(37, 576), 0, F(1, 576), 0]
    # Negative terms can be dropped for an upper bound of dLambda/d(y²).
    assert coefficients[2]+2*coefficients[4]/4+4*coefficients[8]/64 < 0
    omega = math.prod(F(2*j+1, 2)**2 for j in range(5))
    assert omega/F(math.factorial(10)) == F(63, 262144)


def test_native_separation_helps_but_does_not_certify_requested_accuracy():
    value = slab_cardinal_interpolation_bound(15.58986064, tau=.12, spacing=.035, width=.96, shells=128)
    assert value["image_count"] == 513
    assert value["near_hull_images"] == 2
    assert value["separated_total_velocity"] == pytest.approx(.05744778007020467)
    assert value["separated_total_gradient"] == pytest.approx(2.3587653998771936)
    assert value["uniform_all_images_gradient"] > 200*value["separated_total_gradient"]
    assert value["near_images_velocity"] > 1e-4
    assert value["near_images_gradient"] > 1e-4
    assert not value["runtime_admissible"]


def test_absolute_strength_and_unit_covariance():
    base = slab_cardinal_interpolation_bound(2., tau=.12, spacing=.035, width=.96, shells=32)
    for scale in (1e-3, 1e3):
        changed = slab_cardinal_interpolation_bound(2*scale**2, tau=.12*scale, spacing=.035*scale,
                                                    width=.96*scale, shells=32)
        assert changed["separated_total_velocity"] == pytest.approx(base["separated_total_velocity"], rel=2e-13)
        assert scale*changed["separated_total_gradient"] == pytest.approx(base["separated_total_gradient"], rel=2e-13)


@pytest.mark.parametrize("key,value", [("tau", 0), ("spacing", math.inf), ("width", -1),
                                        ("shells", True), ("shells", 0), ("shells", 16385)])
def test_invalid_or_unbounded_work_rejected(key, value):
    kwargs = {"tau": .12, "spacing": .035, "width": .96, "shells": 128}
    kwargs[key] = value
    with pytest.raises(ValueError):
        slab_cardinal_interpolation_bound(2., **kwargs)
