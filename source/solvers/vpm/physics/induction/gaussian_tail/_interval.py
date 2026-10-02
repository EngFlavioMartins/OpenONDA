"""Outward binary64 arithmetic for Gaussian image-tail certificates.

Basic operations and signed pairwise sums are rounded outward. Machin's
identity encloses pi with exact rational arithmetic; a finite positive prefix
and integral remainder enclose the inverse-cubic image coefficient. Gaussian
defects use a rational exponential majorant, not libm accuracy assumptions.

These routines do not modify rounding or denormal controls. Admission
requires round-to-nearest and non-flushing binary32/binary64 arithmetic.
They certify mathematical tail calculations, not mesh, FFT or GPU fields.
"""

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import math
from numbers import Integral
import sys

import numpy as np


@dataclass(frozen=True)
class Interval:
    lower: np.ndarray
    upper: np.ndarray


def _checked(lower, upper):
    lower, upper = np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower > upper):
        raise FloatingPointError("interval arithmetic overflow or invalid enclosure")
    return Interval(lower, upper)


def point(value):
    value = np.asarray(value, dtype=np.float64)
    return _checked(value, value)


def _outward(lower, upper):
    return _checked(np.nextafter(lower, -np.inf), np.nextafter(upper, np.inf))


def add(a, b):
    return _outward(a.lower+b.lower, a.upper+b.upper)


def neg(a):
    return _checked(-a.upper, -a.lower)


def sub(a, b):
    return add(a, neg(b))


def mul(a, b):
    products = [a.lower*b.lower, a.lower*b.upper, a.upper*b.lower, a.upper*b.upper]
    return _outward(np.minimum.reduce(products), np.maximum.reduce(products))


def div_positive(a, b):
    if np.any(b.lower <= 0):
        raise ValueError("strictly positive denominator required")
    reciprocal = _outward(1/b.upper, 1/b.lower)
    return mul(a, reciprocal)


def pairwise_sum(a):
    """Directed signed reduction over axis0; work and storage are O(N)."""
    lower, upper = np.array(a.lower, copy=True), np.array(a.upper, copy=True)
    if not len(lower):
        return point(np.zeros(lower.shape[1:]))
    while len(lower) > 1:
        pairs = len(lower)//2
        next_lower = np.nextafter(lower[:2*pairs:2]+lower[1:2*pairs:2], -np.inf)
        next_upper = np.nextafter(upper[:2*pairs:2]+upper[1:2*pairs:2], np.inf)
        if len(lower) % 2:
            next_lower = np.concatenate((next_lower, lower[-1:]), axis=0)
            next_upper = np.concatenate((next_upper, upper[-1:]), axis=0)
        lower, upper = next_lower, next_upper
    return _checked(lower[0], upper[0])


def _midpoint(a):
    value = .5*a.lower+.5*a.upper
    if not np.isfinite(value).all() or np.any(value < a.lower) or np.any(value > a.upper):
        raise FloatingPointError("invalid midpoint")
    return value


def _fraction_interval(value):
    rounded = float(value)
    lower = rounded if Fraction(rounded) <= value else math.nextafter(rounded, -math.inf)
    upper = rounded if Fraction(rounded) >= value else math.nextafter(rounded, math.inf)
    if not Fraction(lower) <= value <= Fraction(upper):
        raise FloatingPointError("rational conversion not enclosed")
    return _checked(lower, upper)


@lru_cache(maxsize=1)
def _pi_rational_bounds():
    def arctan_inverse(denominator):
        terms = 24
        total = sum((Fraction((-1)**k, (2*k+1)*denominator**(2*k+1))
                     for k in range(terms)), Fraction())
        # An even number of alternating terms ends below the exact arctan.
        return total, total+Fraction(1, (2*terms+1)*denominator**(2*terms+1))
    alo, ahi = arctan_inverse(5)
    blo, bhi = arctan_inverse(239)
    lower, upper = 16*alo-4*bhi, 16*ahi-4*blo
    return lower, upper


def pi_interval():
    # Only immutable rational numbers are shared; returned arrays are private.
    lower, upper = _pi_rational_bounds()
    return _checked(_fraction_interval(lower).lower, _fraction_interval(upper).upper)


def _platform():
    """Fail closed without changing the caller's rounding/denormal controls."""
    if sys.float_info.radix != 2 or sys.float_info.mant_dig != 53:
        raise RuntimeError("IEEE binary64 arithmetic required")
    one = np.float64(1.)
    half_ulp, above_half = np.float64(2.**-53), np.float64(3.*2.**-54)
    successor = np.asarray([0x3FF0000000000001], dtype=np.uint64).view(np.float64)[0]
    half_normal64 = np.asarray([0x0008000000000000], dtype=np.uint64).view(np.float64)[0]
    half_normal32 = np.asarray([0x00400000], dtype=np.uint32).view(np.float32)[0]
    # Startup sys.float_info.rounds is insufficient after a runtime changes
    # FENV. Both nearest tie direction and above-half rounding are checked.
    if one+half_ulp != one or one+above_half != successor:
        raise RuntimeError("round-to-nearest arithmetic required")
    # Test FTZ (subnormal output) and DAZ (subnormal input) independently,
    # including f32 inputs that will be promoted to the certificate's f64.
    if (np.float64(2.**-1022)*np.float64(.5) == 0.
            or half_normal64*np.float64(2.) != np.float64(2.**-1022)
            or np.float32(2.**-126)*np.float32(.5) == 0.
            or half_normal32*np.float32(2.) != np.float32(2.**-126)):
        raise RuntimeError("gradual underflow without FTZ/DAZ required")


def _cap(value, name, maximum):
    if isinstance(value, bool) or not isinstance(value, Integral) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1,{maximum}]")
    return int(value)


def gaussian_defect_upper(*, gap_lower, sigma_upper, period_lower, absolute_strength_upper,
                          maximum_order=96):
    """One source-family +/- infinite-tail defect using only rational bounds.

    Inputs must themselves be valid one-sided bounds. A safe source strength
    is sum_j ||Gamma_j||_1, rather than assuming a rounded L2 norm is an upper
    bound. The caller separately sums the original and reflected families.
    """
    _platform()
    maximum_order = _cap(maximum_order, "maximum_order", 128)
    values = (gap_lower, sigma_upper, period_lower, absolute_strength_upper)
    if not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError("positive finite scalar one-sided inputs required")
    gap, sigma, period, strength = (point(v) for v in values)
    ratio = div_positive(gap, sigma)
    q = mul(ratio, ratio)
    if q.lower < 1.5:
        raise ValueError("uniform-core Gaussian density monotonicity not admitted")
    order = min(maximum_order, math.floor(float(q.lower)))
    exponential_upper = point(1.)
    for k in range(1, order+1):
        exponential_upper = mul(exponential_upper, div_positive(point(float(k)), point(float(q.lower))))
    exp_bound = point(min(1., float(exponential_upper.upper)))
    gap2, sigma2 = mul(gap, gap), mul(sigma, sigma)
    gap3, sigma3 = mul(gap2, gap), mul(sigma2, sigma)
    gap4 = mul(gap2, gap2)
    # pi**1.5>5 and sqrt(5)<3 give deliberately loose rational constants.
    cu = div_positive(add(div_positive(sigma, gap3), div_positive(point(2.), mul(sigma, gap))), point(20.))
    cj = add(div_positive(mul(point(3.), add(div_positive(sigma, gap4),
                      div_positive(point(2.), mul(sigma, gap2)))), point(20.)),
             div_positive(point(1.), mul(point(5.), sigma3)))
    common = div_positive(mul(strength, mul(sigma2, exp_bound)), mul(period, gap))
    return {"velocity_upper": float(mul(common, cu).upper),
            "gradient_frobenius_upper": float(mul(common, cj).upper),
            "exponential_upper": float(exp_bound.upper), "order": order,
            "q_lower": float(q.lower), "scope": "one source family, complete +/- omitted shells"}


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
    _platform()
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
