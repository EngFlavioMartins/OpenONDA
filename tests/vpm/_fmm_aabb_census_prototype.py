"""Unwired strict source-decision bounds for a transformed target AABB.

No opening angle, core gate, cancellation gate, or target-local radius changes.
This helper may prove ALL/NONE where the enclosing-sphere test says MIXED.
It must never force a mixed packet to open a source for already-accepted points.
"""

from dataclasses import dataclass

import numpy as np

_U32 = 2**-24
_TINY32 = np.finfo(np.float32).tiny


@dataclass(frozen=True)
class SourceMetadata:
    centre: np.ndarray
    com: np.ndarray
    half_size: float
    mean_core: float
    min_core: float
    max_core: float
    strength: np.ndarray
    theta_sq: float = 0.01
    core_cutoff: float = 7.0


def transform_query(position, shift, odd):
    query = np.asarray(position, dtype=np.float32).copy()
    query[..., 2] = (
        np.float32(shift) - query[..., 2]
        if odd else query[..., 2] - np.float32(shift)
    )
    return query


def _norm32(vector):
    return np.sqrt(np.sum(np.asarray(vector, np.float32) ** 2, dtype=np.float32), dtype=np.float32)


def _accept_from_distance(source, distance_sq, distance):
    half_size = np.float32(source.half_size)
    diameter = np.float32(2) * half_size
    strength = np.asarray(source.strength, np.float32)
    net_sq = np.sum(strength * strength, dtype=np.float32)
    mean, low, high = map(np.float32, (source.mean_core, source.min_core, source.max_core))
    common = high - low <= np.float32(1e-5) * max(mean, np.float32(1e-12))
    extent = half_size + _norm32(np.asarray(source.com, np.float32) - np.asarray(source.centre, np.float32))
    outside = distance - extent > np.float32(source.core_cutoff) * high
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        angle = diameter * diameter / distance_sq < np.float32(source.theta_sq)
    return bool(
        distance > max(np.float32(1e-8), mean)
        and angle
        and (common or outside)
        and net_sq > np.float32(1e-24)
    )


def legacy_point_accept(source, position, shift=0.0, odd=False):
    query = transform_query(position, shift, odd)
    displacement = query - np.asarray(source.com, np.float32)
    square = np.sum(displacement * displacement, dtype=np.float32)
    return _accept_from_distance(source, square, np.sqrt(square, dtype=np.float32))


def aabb_distance_bounds(low, high, com, shift=0.0, odd=False):
    """Outward bounds on legacy f32 dot/sqrt distances, or None if unproved.

    Float32 addition/subtraction is monotone. Transforming the AABB endpoints
    with the SAME inverse-query arithmetic therefore encloses every transformed
    target, including cancellation at very large coordinate offsets. Subtract
    the source COM in float32 too. No real-arithmetic source-image transform is
    substituted for that sequence.

    Interval displacement endpoints enclose every actual f32 displacement.
    Dot products of the minimum/maximum component magnitudes then bound the
    exact squared norm. The outward 16u margin dominates either three products
    plus two sums or a fused dot's rounding error (gamma_5), including the new
    bound calculation. One smallest-normal additive term covers subnormal/FTZ
    square loss. A further 8u outward margin covers the square root calculation.
    Nonfinite intervals decline classification. This proof presumes standard
    finite IEEE f32 arithmetic; device qualification must compare actual MACs.
    """
    low, high, com = (np.asarray(x, np.float32) for x in (low, high, com))
    if any(x.shape != (3,) for x in (low, high, com)) or np.any(low > high):
        raise ValueError("require ordered three-dimensional AABB endpoints")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        first = transform_query(low, shift, odd) - com
        last = transform_query(high, shift, odd) - com
        lower, upper = np.minimum(first, last), np.maximum(first, last)
        near = np.maximum(np.maximum(lower, -upper), np.float32(0))
        far = np.maximum(np.abs(lower), np.abs(upper))
        low_sq = np.sum(near * near, dtype=np.float32)
        high_sq = np.sum(far * far, dtype=np.float32)
        low_sq = np.maximum(np.float32(0), low_sq * np.float32(1 - 16 * _U32) - _TINY32)
        high_sq = high_sq * np.float32(1 + 16 * _U32) + _TINY32
        low_distance = np.sqrt(low_sq, dtype=np.float32) * np.float32(1 - 8 * _U32)
        high_distance = np.sqrt(high_sq, dtype=np.float32) * np.float32(1 + 8 * _U32)
    result = (low_sq, high_sq, low_distance, high_distance)
    return result if np.all(np.isfinite(result)) else None


def classify_aabb(source, low, high, shift=0.0, odd=False):
    bounds = aabb_distance_bounds(low, high, source.com, shift, odd)
    if bounds is None:
        return "mixed"
    low_sq, high_sq, low_distance, high_distance = bounds
    # Legacy admission is monotone in squared distance and distance, with
    # source-only metadata held fixed. Use its identical endpoint predicates.
    if _accept_from_distance(source, low_sq, low_distance):
        return "all"
    if not _accept_from_distance(source, high_sq, high_distance):
        return "none"
    return "mixed"
