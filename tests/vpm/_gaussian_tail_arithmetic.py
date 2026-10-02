"""UNWIRED bounded leading-tail arithmetic and rational Gaussian defect.

The enclosure concerns exact real operations on the supplied binary64 input
values, under IEEE754 binary64 round-to-nearest with gradual underflow. Every
basic array operation is bracketed with nextafter; signed sums use a directed
pairwise reduction, not a relative error estimate on a cancelled result.
Machin's identity and alternating rational arctangent series enclose pi.
The H3 interval is supplied by the separately qualified coefficient helper.

For the original and axial-vector reflected source families, with origin_z
equal to zmin, the exact identities are
 G=(0,0,2 sum Gz), Mx=-4 sum Gy(z-zmin)-2 sum Gz(y-oy),
 My=4 sum Gx(z-zmin)+2 sum Gz(x-ox), Mz=0.
They avoid constructing rounded reflected source positions. The returned
nominal fields use one affine midpoint formula, so nominal J=dU/dx in real
arithmetic; the returned bounds also cover rounding that stored formula.

Gaussian-minus-singular tails use exp(-q)<=m!/q**m, following the positive
exponential series, and only positive outward-rounded rational operations.
No exp/log/erfc/sqrt library accuracy is assumed. This is NOT a certificate
for the singular Taylor remainder, finite image mesh, FFT, or GPU arithmetic.
"""

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import math
from numbers import Integral
import sys

import numpy as np

from tests.vpm._gaussian_tail_coefficient_enclosure import coefficient_enclosure


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
    if (sys.float_info.radix != 2 or sys.float_info.mant_dig != 53
            or sys.float_info.rounds != 1 or np.float64(2.0**-1022)*.5 == 0):
        raise RuntimeError("binary64 round-to-nearest with gradual underflow required")


def _cap(value, name, maximum):
    if isinstance(value, bool) or not isinstance(value, Integral) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1,{maximum}]")
    return int(value)


@dataclass(frozen=True)
class LeadingTail:
    velocity: np.ndarray
    gradient: np.ndarray
    velocity_error_bound: np.ndarray
    gradient_error_bound: np.ndarray
    velocity_interval: Interval
    gradient_interval: Interval
    moments: dict


def _stored_error(interval, stored):
    # L1 dominates Euclidean/Frobenius norms and avoids sqrt assumptions.
    error = np.maximum(sub(point(stored), interval).upper,
                       sub(interval, point(stored)).upper)
    return pairwise_sum(point(np.moveaxis(error, -1, 0))).upper


def leading_tail(position, strength, targets, *, z_min, z_max, shells,
                 prefix_terms=1024, max_sources=400_000, max_targets=400_000):
    """Enclose the leading infinite paired tail, not its Taylor remainder."""
    _platform()
    max_sources = _cap(max_sources, "max_sources", 1_000_000)
    max_targets = _cap(max_targets, "max_targets", 1_000_000)
    x, gamma, target = (np.asarray(v) for v in (position, strength, targets))
    if (x.ndim != 2 or x.shape[1:] != (3,) or gamma.shape != x.shape
            or target.ndim != 2 or target.shape[1:] != (3,)
            or len(x) > max_sources or len(target) > max_targets):
        raise ValueError("bounded source/strength/target shapes required")
    x, gamma, target = (np.array(v, dtype=np.float64, copy=True) for v in (x, gamma, target))
    if (not all(np.isfinite(v).all() for v in (x, gamma, target))
            or not math.isfinite(z_min) or not math.isfinite(z_max) or z_max <= z_min
            or np.any(x[:, 2] < z_min) or np.any(x[:, 2] > z_max)):
        raise ValueError("bounded finite fields inside ordered physical slab required")
    h3 = coefficient_enclosure(shells, prefix_terms=prefix_terms)
    period = mul(point(2.), sub(point(z_max), point(z_min)))
    if period.lower <= 0:
        raise ValueError("positive enclosed period required")
    if not len(x):
        origin = np.array([0., 0., z_min])
    else:
        origin = .5*x.min(axis=0)+.5*x.max(axis=0)
        origin[2] = z_min
    dx, dy, dz = (sub(point(x[:, i]), point(origin[i])) for i in range(3))
    gx, gy, gz = (point(gamma[:, i]) for i in range(3))
    net_z = mul(point(2.), pairwise_sum(gz))
    moment_x = pairwise_sum(neg(add(mul(point(4.), mul(gy, dz)), mul(point(2.), mul(gz, dy)))))
    moment_y = pairwise_sum(add(mul(point(4.), mul(gx, dz)), mul(point(2.), mul(gz, dx))))
    coefficient = div_positive(_checked(h3.lower, h3.upper),
                               mul(point(2.), mul(pi_interval(), mul(period, mul(period, period)))))
    tx, ty = (sub(point(target[:, i]), point(origin[i])) for i in range(2))
    ux = mul(coefficient, sub(neg(mul(net_z, ty)), moment_x))
    uy = mul(coefficient, sub(mul(net_z, tx), moment_y))
    cross = mul(coefficient, net_z)
    ulo = np.column_stack((ux.lower, uy.lower, np.zeros(len(target))))
    uhi = np.column_stack((ux.upper, uy.upper, np.zeros(len(target))))
    jlo, jhi = np.zeros((len(target), 3, 3)), np.zeros((len(target), 3, 3))
    jlo[:, 0, 1], jhi[:, 0, 1] = -cross.upper, -cross.lower
    jlo[:, 1, 0], jhi[:, 1, 0] = cross.lower, cross.upper
    coefficient_mid, net_mid = float(_midpoint(coefficient)), float(_midpoint(net_z))
    mx_mid, my_mid = float(_midpoint(moment_x)), float(_midpoint(moment_y))
    u = np.zeros((len(target), 3))
    u[:, 0] = coefficient_mid*(-net_mid*(target[:, 1]-origin[1])-mx_mid)
    u[:, 1] = coefficient_mid*(net_mid*(target[:, 0]-origin[0])-my_mid)
    j = np.zeros((len(target), 3, 3))
    j[:, 0, 1], j[:, 1, 0] = -coefficient_mid*net_mid, coefficient_mid*net_mid
    if not np.isfinite(u).all() or not np.isfinite(j).all():
        raise FloatingPointError("nominal leading field overflow")
    ui, ji = _checked(ulo, uhi), _checked(jlo, jhi)
    uerr = _stored_error(ui, u)
    jerr = _stored_error(_checked(jlo.reshape(-1, 9), jhi.reshape(-1, 9)), j.reshape(-1, 9))
    moments = {"origin": origin, "net_z": net_z, "moment_x": moment_x, "moment_y": moment_y,
               "coefficient": coefficient, "h3": h3,
               "source_l1_strength_upper": float(pairwise_sum(point(np.abs(gamma).reshape(-1))).upper)}
    return LeadingTail(u, j, uerr, jerr, ui, ji, moments)


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
