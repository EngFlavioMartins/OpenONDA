"""Float64 evaluation of the slab truncation bounds, with one safety factor.

Every bound of the slab image field is a finite expression in sums, products
and quotients of positive float64 numbers. A float64 operation on positive
operands has relative error at most u = 2**-53, so an expression of n such
operations differs from its exact value by less than 1.2 n u. The longest
chain here is the exponential series, 128 terms of two operations each and 128
additions, fewer than 400 operations: 5e-14. Sums over many positive terms
(the strengths of the sources, the omitted images) go through
``positive_sum``, whose relative error is below 4100 u = 4.6e-13 for any
number of terms. ``UPPER_SAFETY`` exceeds both by a factor above 200, and every
returned upper bound is multiplied by it, so it is not smaller than the exact
bound it evaluates.

A distance lower bound is a difference, and a difference of nearly equal
numbers loses relative accuracy. Lower bounds of distances therefore subtract
``COORDINATE_SLACK`` times the magnitude of the coordinates they are computed
from. At most 32 operations of relative error u on operands below that
magnitude enter such a distance, which is the 32 in ``COORDINATE_SLACK``.

Flush to zero. Taichi sets flush-to-zero and denormals-are-zero for the host
thread, so a subnormal result is replaced by zero. An operation flushes only
when its exact result is below 2**-1022 = 2.2e-308, and the flush changes it by
less than that. The factors that multiply it afterwards are at most 2**300
for lengths between 2**-30 m and 2**30 m (1 nm to 1e6 km) and strength sums
below 2**100: the strength itself, the cube of the inverse of the smallest
distance, and a few constants. A flushed value therefore changes the result by
less than 2**-722 = 1.2e-217, and ``UNDERFLOW_ALLOWANCE = 2**-700`` is added to
every returned bound. The comparison ``bound <= tolerance * scale`` then keeps
its meaning for every tolerance times scale of at least 1e-210; the smallest
bound the slab tests and tutorials produce is 3e-17. A bound that is
exactly zero because a source strength is zero is reported as zero.

These functions bound the mathematical truncation only, not the error of the
mesh, the FFT or the GPU.
"""

import math

import numpy as np

UNIT_ROUNDOFF = 2.0**-53
UPPER_SAFETY = 1.0 + 1e-10
COORDINATE_SLACK = 32 * UNIT_ROUNDOFF
UNDERFLOW_ALLOWANCE = 2.0**-700
_SUM_BLOCK = 4096
_EXPONENT_LIMIT = 64.0
_SERIES_TERMS = 128


def upper_bound(value: float | np.ndarray) -> float | np.ndarray:
    """A float64 evaluation of a positive bound, raised to a valid upper bound."""
    return value * UPPER_SAFETY + UNDERFLOW_ALLOWANCE


def positive_sum(values: np.ndarray) -> float:
    """Sum of nonnegative numbers with relative error below 4100 u for any length.

    Blocks of 4096 are summed by numpy, in whatever order it chooses, and the
    block sums are added exactly by ``math.fsum``.
    """
    flat = np.asarray(values, dtype=np.float64).ravel()
    blocks = [flat[start : start + _SUM_BLOCK].sum() for start in range(0, len(flat), _SUM_BLOCK)]
    return math.fsum(blocks)


def exponential_upper(squared_lower: np.ndarray) -> np.ndarray:
    """exp(-q) <= 1/sum_{k<=128} q^k/k!, with q clipped to 64 from above.

    ``squared_lower`` is a lower bound of q. The series has only positive
    terms; no libm exponential is used.
    """
    q = np.minimum(squared_lower, _EXPONENT_LIMIT)
    total = term = np.ones_like(q)
    for index in range(1, _SERIES_TERMS + 1):
        term = term * q / float(index)
        total = total + term
    return 1.0 / total
