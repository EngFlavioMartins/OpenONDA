"""Outward bounds on a non-Gaussian kernel's deviation from Biot-Savart.

A kernel's definition carries an envelope of its core defect
T(x) = 1/(4 pi) - q and of the gradient defect 3T + x^3 zeta, in the partner
coordinate x = r/s, valid and decreasing for x >= minimum_ratio. One source
of strength |Gamma| at distance r then perturbs the velocity by at most
|Gamma| E_T/r^2 and the full Jacobian, in the Frobenius norm, by at most
sqrt(2) |Gamma| (E_T + E_J)/r^3. Because E(r/s)/r^k grows with s once
x >= minimum_ratio, the largest partner width bounds every source.

Omitted image rows lie at g, g + P, g + 2P, ... on each side. When x^4 E is
non-increasing, r^6 E_T(r/s)/r^2 and r^7 E_J(r/s)/r^3 do not increase, so
sum_j f(g + jP) <= f(g) + (1/P) int_g^inf f <= f(g) (1 + g/((m - 1) P)),
with m = 6 for the velocity and m = 7 for the Jacobian.

The exponential is bounded by a rational series, pi^1.5 > 5 and
sqrt(2) < 99/70; no libm accuracy is assumed.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from ....kernels.definition import AlgebraicDefectEnvelope, GaussianDefectEnvelope
from ._interval import add, div_positive, mul, pi_interval, point

if TYPE_CHECKING:
    from ._interval import Interval

_SQRT_TWO_UPPER = 99.0 / 70.0


class KernelCoreSeparationError(ValueError):
    """A defect envelope was asked for inside the kernel's minimum partner ratio."""


def _exponential_upper(squared_lower: np.ndarray) -> "Interval":
    """exp(-q) <= 1/sum_{k<=128} q^k/k!, with q clipped to 64 from below."""
    q = point(np.minimum(squared_lower, 64.0))
    total = term = point(np.ones_like(squared_lower))
    for index in range(1, 129):
        term = div_positive(mul(term, q), point(float(index)))
        total = add(total, term)
    return div_positive(point(1.0), point(total.lower))


def defect_upper(
    envelope: "AlgebraicDefectEnvelope | GaussianDefectEnvelope", ratio_lower: np.ndarray
) -> tuple["Interval", "Interval"] | None:
    """Upper bounds of E_T and E_J at partner ratios at least ``ratio_lower``.

    ``ratio_lower`` must be valid lower bounds of r/s_max, all at least the
    envelope's ``minimum_ratio``; the envelopes decrease beyond it.
    """
    x = np.asarray(ratio_lower, dtype=np.float64)
    if np.any(x < envelope.minimum_ratio):
        raise KernelCoreSeparationError(
            "kernel defect envelope used inside its minimum partner ratio"
        )
    x = point(x)
    if isinstance(envelope, AlgebraicDefectEnvelope):
        power = point(1.0)
        for _ in range(envelope.power):
            power = mul(power, x)
        denominator = mul(mul(point(4.0), pi_interval()), power)
        return (
            div_positive(point(envelope.velocity), denominator),
            div_positive(point(envelope.gradient), denominator),
        )
    if not isinstance(envelope, GaussianDefectEnvelope):
        raise TypeError("unsupported kernel defect envelope")
    weight = div_positive(_exponential_upper(mul(x, x).lower), point(5.0))

    def polynomial(terms: Sequence[tuple[int, float]]) -> "Interval":
        total = point(np.zeros_like(x.lower))
        for power, coefficient in terms:
            value = point(coefficient)
            for _ in range(abs(power)):
                value = mul(value, x) if power > 0 else div_positive(value, x)
            total = add(total, value)
        return mul(weight, total)

    return polynomial(envelope.velocity), polynomial(envelope.gradient)


def source_defect_upper(
    envelope: "AlgebraicDefectEnvelope | GaussianDefectEnvelope",
    distance_lower: float | np.ndarray,
    core_upper: float,
    strength_upper: float,
) -> tuple["Interval", "Interval"] | None:
    """Velocity and Jacobian bounds of sources at ``distance_lower`` or beyond."""
    distance, core = point(distance_lower), point(core_upper)
    velocity, gradient = defect_upper(envelope, div_positive(distance, core).lower)
    strength = point(strength_upper)
    r2 = mul(distance, distance)
    return (
        div_positive(mul(strength, velocity), r2),
        div_positive(
            mul(mul(point(_SQRT_TWO_UPPER), strength), add(velocity, gradient)),
            mul(r2, distance),
        ),
    )


def row_defect_upper(
    envelope: "AlgebraicDefectEnvelope | GaussianDefectEnvelope",
    *,
    gap_lower: float,
    core_upper: float,
    period_lower: float,
    strength_upper: float,
) -> tuple[float, float]:
    """One source family, both sides: omitted rows from ``gap_lower`` outward."""
    velocity, gradient = source_defect_upper(envelope, gap_lower, core_upper, strength_upper)
    rows = div_positive(point(gap_lower), point(period_lower))
    velocity = mul(velocity, add(point(1.0), div_positive(rows, point(5.0))))
    gradient = mul(gradient, add(point(1.0), div_positive(rows, point(6.0))))
    return (
        float(mul(point(2.0), velocity).upper),
        float(mul(point(2.0), gradient).upper),
    )
