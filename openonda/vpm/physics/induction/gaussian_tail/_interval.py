"""Outward binary64 arithmetic for Gaussian image-tail error_bounds.

Basic operations and signed pairwise sums are rounded outward. Machin's
identity encloses pi with exact rational arithmetic. Gaussian defects use a
rational exponential majorant, not libm accuracy assumptions.

These routines do not modify rounding or denormal controls. Checks
requires round-to-nearest and non-flushing binary32/binary64 arithmetic.
They validate mathematical tail calculations, not mesh, FFT or GPU fields.
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
    """Closed interval [lower, upper] of float64 arrays."""

    lower: np.ndarray
    upper: np.ndarray


def _checked(lower: float | np.ndarray, upper: float | np.ndarray) -> "Interval | None":
    lower, upper = np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower > upper):
        raise FloatingPointError("interval arithmetic overflow or invalid enclosure")
    return Interval(lower, upper)


def point(value: float | np.ndarray | tuple[float, float, float]) -> "Interval":
    """Degenerate interval of an exact value."""
    value = np.asarray(value, dtype=np.float64)
    return _checked(value, value)


def _outward(lower: float | np.ndarray, upper: float | np.ndarray) -> "Interval | None":
    return _checked(np.nextafter(lower, -np.inf), np.nextafter(upper, np.inf))


def add(a: "Interval", b: "Interval") -> "Interval":
    """Outward-rounded interval sum."""
    return _outward(a.lower + b.lower, a.upper + b.upper)


def neg(a: "Interval") -> "Interval":
    """Interval negation; exact."""
    return _checked(-a.upper, -a.lower)


def sub(a: "Interval", b: "Interval") -> "Interval":
    """Outward-rounded interval difference."""
    return add(a, neg(b))


def mul(a: "Interval", b: "Interval") -> "Interval | None":
    """Outward-rounded interval product."""
    products = [a.lower * b.lower, a.lower * b.upper, a.upper * b.lower, a.upper * b.upper]
    return _outward(np.minimum.reduce(products), np.maximum.reduce(products))


def div_positive(a: "Interval", b: "Interval") -> "Interval":
    """Outward-rounded quotient by a strictly positive interval."""
    if np.any(b.lower <= 0):
        raise ValueError("strictly positive denominator required")
    reciprocal = _outward(1 / b.upper, 1 / b.lower)
    return mul(a, reciprocal)


def pairwise_sum(a: "Interval") -> "Interval":
    """Directed signed reduction over axis0; work and storage are O(N)."""
    lower, upper = np.array(a.lower, copy=True), np.array(a.upper, copy=True)
    if not len(lower):
        return point(np.zeros(lower.shape[1:]))
    while len(lower) > 1:
        pairs = len(lower) // 2
        next_lower = np.nextafter(lower[: 2 * pairs : 2] + lower[1 : 2 * pairs : 2], -np.inf)
        next_upper = np.nextafter(upper[: 2 * pairs : 2] + upper[1 : 2 * pairs : 2], np.inf)
        if len(lower) % 2:
            next_lower = np.concatenate((next_lower, lower[-1:]), axis=0)
            next_upper = np.concatenate((next_upper, upper[-1:]), axis=0)
        lower, upper = next_lower, next_upper
    return _checked(lower[0], upper[0])


def _fraction_interval(value: Fraction) -> "Interval":
    rounded = float(value)
    lower = rounded if Fraction(rounded) <= value else math.nextafter(rounded, -math.inf)
    upper = rounded if Fraction(rounded) >= value else math.nextafter(rounded, math.inf)
    if not Fraction(lower) <= value <= Fraction(upper):
        raise FloatingPointError("rational conversion not enclosed")
    return _checked(lower, upper)


@lru_cache(maxsize=1)
def _pi_rational_bounds() -> tuple[Fraction, Fraction]:
    def arctan_inverse(denominator: int) -> tuple[Fraction, Fraction]:
        terms = 24
        total = sum(
            (Fraction((-1) ** k, (2 * k + 1) * denominator ** (2 * k + 1)) for k in range(terms)),
            Fraction(),
        )
        # An even number of alternating terms ends below the exact arctan.
        return total, total + Fraction(1, (2 * terms + 1) * denominator ** (2 * terms + 1))

    alo, ahi = arctan_inverse(5)
    blo, bhi = arctan_inverse(239)
    lower, upper = 16 * alo - 4 * bhi, 16 * ahi - 4 * blo
    return lower, upper


def pi_interval() -> "Interval":
    # Only immutable rational numbers are shared; returned arrays are private.
    """Float64 interval enclosing pi, from Machin's formula in rationals."""
    lower, upper = _pi_rational_bounds()
    return _checked(_fraction_interval(lower).lower, _fraction_interval(upper).upper)


def _platform() -> None:
    """Fail closed without changing the caller's rounding/denormal controls."""
    if sys.float_info.radix != 2 or sys.float_info.mant_dig != 53:
        raise RuntimeError("IEEE binary64 arithmetic required")
    one = np.float64(1.0)
    half_ulp, above_half = np.float64(2.0**-53), np.float64(3.0 * 2.0**-54)
    successor = np.asarray([0x3FF0000000000001], dtype=np.uint64).view(np.float64)[0]
    half_normal64 = np.asarray([0x0008000000000000], dtype=np.uint64).view(np.float64)[0]
    half_normal32 = np.asarray([0x00400000], dtype=np.uint32).view(np.float32)[0]
    # Startup sys.float_info.rounds is insufficient after a runtime changes
    # FENV. Both nearest tie direction and above-half rounding are checked.
    if one + half_ulp != one or one + above_half != successor:
        raise RuntimeError("round-to-nearest arithmetic required")
    # Test FTZ (subnormal output) and DAZ (subnormal input) independently,
    # including f32 inputs that will be promoted to the tail-bound binary64 arithmetic.
    if (
        np.float64(2.0**-1022) * np.float64(0.5) == 0.0
        or half_normal64 * np.float64(2.0) != np.float64(2.0**-1022)
        or np.float32(2.0**-126) * np.float32(0.5) == 0.0
        or half_normal32 * np.float32(2.0) != np.float32(2.0**-126)
    ):
        raise RuntimeError("gradual underflow without FTZ/DAZ required")


def _cap(value: int, name: str, maximum: int) -> int | None:
    if isinstance(value, bool) or not isinstance(value, Integral) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1,{maximum}]")
    return int(value)


def gaussian_defect_upper(
    *,
    gap_lower: float,
    sigma_upper: float,
    period_lower: float,
    absolute_strength_upper: float,
    maximum_order: int = 96,
) -> dict[str, float] | None:
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
        raise ValueError("uniform-core Gaussian density monotonicity not checked")
    order = min(maximum_order, math.floor(float(q.lower)))
    exponential_upper = point(1.0)
    for k in range(1, order + 1):
        exponential_upper = mul(
            exponential_upper, div_positive(point(float(k)), point(float(q.lower)))
        )
    exp_bound = point(min(1.0, float(exponential_upper.upper)))
    gap2, sigma2 = mul(gap, gap), mul(sigma, sigma)
    gap3, sigma3 = mul(gap2, gap), mul(sigma2, sigma)
    gap4 = mul(gap2, gap2)
    # pi**1.5>5 and sqrt(5)<3 give deliberately loose rational constants.
    cu = div_positive(
        add(div_positive(sigma, gap3), div_positive(point(2.0), mul(sigma, gap))), point(20.0)
    )
    cj = add(
        div_positive(
            mul(
                point(3.0),
                add(div_positive(sigma, gap4), div_positive(point(2.0), mul(sigma, gap2))),
            ),
            point(20.0),
        ),
        div_positive(point(1.0), mul(point(5.0), sigma3)),
    )
    common = div_positive(mul(strength, mul(sigma2, exp_bound)), mul(period, gap))
    return {
        "velocity_upper": float(mul(common, cu).upper),
        "gradient_frobenius_upper": float(mul(common, cj).upper),
        "exponential_upper": float(exp_bound.upper),
        "order": order,
        "q_lower": float(q.lower),
        "scope": "one source family, complete +/- omitted shells",
    }
