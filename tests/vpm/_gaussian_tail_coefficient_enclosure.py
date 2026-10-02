"""UNWIRED binary64 enclosure of the paired-image leading coefficient.

For H_K=sum_{k=K+1}^infinity k^-3, sum a bounded positive prefix through M
and bracket the rest by 1/[2(M+1)^2] <= tail <= 1/(2M^2).
Every division and addition in that finite-prefix calculation is rounded
outward with nextafter. Integer denominators are constructed exactly, then
their binary64 conversion is bracketed too. This gives an interval for H_K
under IEEE754 binary64 round-to-nearest arithmetic with gradual underflow.
The bounded integer domain below prevents overflow/subnormal operations.
No scipy zeta value is used to construct the interval.

Using midpoint h and radius delta in the analytic leading field h*B(t),
the coefficient-only error is <=delta*||B(t)||. For the Jacobian it is
<=delta*||DB||_F. A separate enclosure of B/DB and of multiplying by h is
still required for a fully certified field. In particular this helper does
NOT certify the previous moment helper's fsum/product/norm calculations,
Gaussian exp/log calls, or any finite-image interpolation/FFT/GPU arithmetic.
"""

from dataclasses import dataclass
import math
from numbers import Integral
import sys


@dataclass(frozen=True)
class CoefficientEnclosure:
    shells: int
    prefix_terms: int
    last_prefix_shell: int
    lower: float
    upper: float
    midpoint: float
    radius: float
    prefix_lower: float
    prefix_upper: float
    integral_lower: float
    integral_upper: float


def _integer(value, name, maximum):
    if isinstance(value, bool) or not isinstance(value, Integral) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1,{maximum}]")
    return int(value)


def _reciprocal_integer(denominator):
    """Enclose the exact positive rational 1/denominator, including conversion."""
    rounded = float(denominator)
    lower_denominator = math.nextafter(rounded, -math.inf)
    upper_denominator = math.nextafter(rounded, math.inf)
    return (math.nextafter(1/upper_denominator, -math.inf),
            math.nextafter(1/lower_denominator, math.inf))


def coefficient_enclosure(shells, *, prefix_terms=1024):
    """Bounded-work enclosure; radius includes rounding of its own midpoint."""
    shells = _integer(shells, "shells", 1 << 30)
    prefix_terms = _integer(prefix_terms, "prefix_terms", 16_384)
    if (sys.float_info.radix != 2 or sys.float_info.mant_dig != 53
            or sys.float_info.rounds != 1):
        raise RuntimeError("IEEE754 binary64 round-to-nearest arithmetic required")
    lower, upper = 0., 0.
    last = shells+prefix_terms
    for k in range(shells+1, last+1):
        term_lower, term_upper = _reciprocal_integer(k*k*k)
        lower = math.nextafter(lower+term_lower, -math.inf)
        upper = math.nextafter(upper+term_upper, math.inf)
    integral_lower = _reciprocal_integer(2*(last+1)*(last+1))[0]
    integral_upper = _reciprocal_integer(2*last*last)[1]
    total_lower = math.nextafter(lower+integral_lower, -math.inf)
    total_upper = math.nextafter(upper+integral_upper, math.inf)
    midpoint = total_lower+.5*(total_upper-total_lower)
    # Do not assume a rounded midpoint lies at the exact interval centre.
    radius = math.nextafter(max(midpoint-total_lower, total_upper-midpoint), math.inf)
    if not 0 < total_lower <= midpoint <= total_upper < math.inf or radius <= 0:
        raise FloatingPointError("invalid coefficient enclosure")
    return CoefficientEnclosure(shells, prefix_terms, last, total_lower, total_upper,
                                midpoint, radius, lower, upper, integral_lower, integral_upper)
