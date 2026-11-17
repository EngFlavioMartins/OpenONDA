"""Bounds on a non-Gaussian kernel's deviation from Biot-Savart.

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

The exponential is bounded by a positive series, pi^1.5 > 5 and
sqrt(2) < 99/70; no libm accuracy is assumed. The helpers return float64
evaluations of the bounds; ``row_defect_upper`` applies the safety factor of
``float_bounds``, the callers of the others apply it to what they sum.
"""

import math

import numpy as np

from ....kernels.definition import AlgebraicDefectEnvelope, GaussianDefectEnvelope
from .float_bounds import exponential_upper, upper_bound

_SQRT_TWO_UPPER = 99.0 / 70.0


class KernelCoreSeparationError(ValueError):
    """A defect envelope was asked for inside the kernel's minimum partner ratio."""


def defect_upper(
    envelope: "AlgebraicDefectEnvelope | GaussianDefectEnvelope", ratio_lower: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Upper bounds of E_T and E_J at partner ratios at least ``ratio_lower``.

    ``ratio_lower`` must be lower bounds of r/s_max, all at least the
    envelope's ``minimum_ratio``; the envelopes decrease beyond it.
    """
    x = np.asarray(ratio_lower, dtype=np.float64)
    if np.any(x < envelope.minimum_ratio):
        raise KernelCoreSeparationError(
            "kernel defect envelope used inside its minimum partner ratio"
        )
    if isinstance(envelope, AlgebraicDefectEnvelope):
        denominator = 4.0 * math.pi * x**envelope.power
        return envelope.velocity / denominator, envelope.gradient / denominator
    if not isinstance(envelope, GaussianDefectEnvelope):
        raise TypeError("unsupported kernel defect envelope")
    weight = exponential_upper(x * x) / 5.0

    def polynomial(terms: tuple[tuple[int, float], ...]) -> np.ndarray:
        total = np.zeros_like(x)
        for power, coefficient in terms:
            total = total + coefficient * x**power
        return weight * total

    return polynomial(envelope.velocity), polynomial(envelope.gradient)


def source_defect_upper(
    envelope: "AlgebraicDefectEnvelope | GaussianDefectEnvelope",
    distance_lower: float | np.ndarray,
    core_upper: float,
    strength_upper: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Velocity and Jacobian bounds of sources at ``distance_lower`` or beyond."""
    distance = np.asarray(distance_lower, dtype=np.float64)
    velocity, gradient = defect_upper(envelope, distance / core_upper)
    r2 = distance * distance
    return (
        strength_upper * velocity / r2,
        _SQRT_TWO_UPPER * strength_upper * (velocity + gradient) / (r2 * distance),
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
    rows = gap_lower / period_lower
    velocity = velocity * (1.0 + rows / 5.0)
    gradient = gradient * (1.0 + rows / 6.0)
    return float(upper_bound(2.0 * velocity)), float(upper_bound(2.0 * gradient))
