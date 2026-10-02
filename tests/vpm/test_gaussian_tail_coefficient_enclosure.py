"""Independent exact-rational and high-precision coefficient checks."""

from decimal import Decimal, localcontext
from fractions import Fraction
import math

import pytest
from scipy.special import zeta

from tests.vpm._gaussian_tail_coefficient_enclosure import coefficient_enclosure


@pytest.mark.parametrize("shells,prefix", [(1, 1), (2, 7), (16, 32), (128, 64),
                                          (1 << 30, 3)])
def test_finite_prefix_and_integral_endpoints_enclose_exact_rationals(shells, prefix):
    result = coefficient_enclosure(shells, prefix_terms=prefix)
    exact = sum((Fraction(1, k**3) for k in range(shells+1, shells+prefix+1)), Fraction())
    assert Fraction(result.prefix_lower) <= exact <= Fraction(result.prefix_upper)
    lower = exact+Fraction(1, 2*(result.last_prefix_shell+1)**2)
    upper = exact+Fraction(1, 2*result.last_prefix_shell**2)
    assert Fraction(result.lower) <= lower <= upper <= Fraction(result.upper)
    assert Fraction(result.midpoint)-Fraction(result.radius) <= Fraction(result.lower)
    assert Fraction(result.midpoint)+Fraction(result.radius) >= Fraction(result.upper)


@pytest.mark.parametrize("shells", [1, 16, 32, 64, 128, 256])
def test_interval_contains_independent_decimal_tail_and_scipy_value(shells):
    result = coefficient_enclosure(shells, prefix_terms=1024)
    # A separate longer positive sum plus its narrower integral bracket,
    # calculated at 70 decimal digits. It must fit strictly in the enclosure.
    last = shells+4096
    with localcontext() as context:
        context.prec = 70
        prefix = sum(Decimal(1)/(Decimal(k)**3) for k in range(shells+1, last+1))
        lower = prefix+Decimal(1)/(2*Decimal(last+1)**2)
        upper = prefix+Decimal(1)/(2*Decimal(last)**2)
        assert Decimal(result.lower) < lower < upper < Decimal(result.upper)
    scipy_value = float(zeta(3, shells+1))
    assert result.lower < scipy_value < result.upper
    assert abs(scipy_value-result.midpoint) <= result.radius


def test_longer_prefix_reduces_coefficient_uncertainty_without_fixed_zeta():
    short = coefficient_enclosure(64, prefix_terms=32)
    long = coefficient_enclosure(64, prefix_terms=1024)
    assert long.radius < short.radius/1000
    assert long.lower > short.lower
    assert long.upper < short.upper


@pytest.mark.parametrize("shells,prefix", [(True, 1), (0, 1), (1.5, 1), (1 << 31, 1),
                                          (1, True), (1, 0), (1, 16_385), (math.inf, 1)])
def test_invalid_or_unbounded_work_rejected(shells, prefix):
    with pytest.raises(ValueError):
        coefficient_enclosure(shells, prefix_terms=prefix)
