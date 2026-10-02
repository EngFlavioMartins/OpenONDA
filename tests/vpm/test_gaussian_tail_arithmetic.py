"""Independent exact-rational checks of production interval arithmetic."""

from decimal import Decimal, localcontext
from fractions import Fraction as F  # noqa: N817 - compact exact-arithmetic notation
import math

import numpy as np
import pytest

from source.solvers.vpm.physics.induction.gaussian_tail._interval import (
    gaussian_defect_upper,
    pairwise_sum,
    pi_interval,
    point,
)


def test_signed_pairwise_sum_does_not_use_cancelled_result_as_conditioning():
    values = np.array([1e20, 1.0, -1e20, 3.0, 1e-20, -2.0])
    result = pairwise_sum(point(values))
    exact = sum((F(float(v)) for v in values), F())
    assert F(float(result.lower)) <= exact <= F(float(result.upper))
    assert result.upper - result.lower > 0


def test_pi_interval_has_independent_decimal_pi_inside():
    result = pi_interval()
    with localcontext() as context:
        context.prec = 70
        pi = Decimal("3.1415926535897932384626433832795028841971693993751058209749445923078")
        assert Decimal(float(result.lower)) < pi < Decimal(float(result.upper))


def test_returned_constant_storage_cannot_poison_later_calls():
    result = pi_interval()
    result.lower[...] = -100.0
    assert pi_interval().lower > 3.0


@pytest.mark.parametrize("gap,sigma", [(2.0, 1.0), (4.0, 2.0), (18.0, 0.04), (120.0, 0.04)])
def test_rational_gaussian_encloses_high_precision_analytic_defect(gap, sigma):
    period, strength = 1.92, 15.59
    result = gaussian_defect_upper(
        gap_lower=gap, sigma_upper=sigma, period_lower=period, absolute_strength_upper=strength
    )
    with localcontext() as context:
        context.prec = 90
        d, s, a, strength = map(Decimal, (gap, sigma, period, strength))
        pi = Decimal(
            "3.141592653589793238462643383279502884197169399375105820974944592307816406286208998"
        )
        exponential = (-((d / s) ** 2)).exp()
        common = strength * s * s * exponential / (a * d)
        cu = (s / d**3 + 2 / (s * d)) / (4 * pi * pi.sqrt())
        cj = Decimal(5).sqrt() * (s / d**4 + 2 / (s * d * d)) / (4 * pi * pi.sqrt()) + 1 / (
            pi * pi.sqrt() * s**3
        )
        assert Decimal(result["velocity_upper"]) >= common * cu
        assert Decimal(result["gradient_frobenius_upper"]) >= common * cj
        assert Decimal(result["exponential_upper"]) >= exponential
        assert result["velocity_upper"] > 0 and result["gradient_frobenius_upper"] > 0


def test_gaussian_domain_and_nonfinite_inputs_fail_closed():
    for gap in (0.1, -1.0, math.inf):
        with pytest.raises(ValueError):
            gaussian_defect_upper(
                gap_lower=gap, sigma_upper=1.0, period_lower=2.0, absolute_strength_upper=1.0
            )
    with pytest.raises(ValueError):
        gaussian_defect_upper(
            gap_lower=3.0,
            sigma_upper=1.0,
            period_lower=2.0,
            absolute_strength_upper=1.0,
            maximum_order=True,
        )
