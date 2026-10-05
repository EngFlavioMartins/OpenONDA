"""Mathematical envelope for an omitted finite Gaussian core correction.

For actual source cores sigma <= tau and omitted distances r >= r_c,
  |du| <= |Gamma| T_tau(r_c)/r_c**2,
  ||dJ||_F <= sqrt(2)|Gamma| (2*T_tau(r_c)/r_c**3 + Z_tau(r_c)).
Here T=(erfc(r/tau)+2*(r/tau)*exp(-(r/tau)**2)/sqrt(pi))/(4*pi).
The bounds follow from the positive integral for narrow-minus-broad radial
factors. They include the full Jacobian, not only its symmetric part.

Use erfc(q) <= exp(-q*q)/(sqrt(pi)*q), pi**1.5 > 5, sqrt(2)<3/2,
and a positive rational lower sum for exp(q*q). All arithmetic is enclosed.
The source L1 norm dominates its Euclidean norm, without a sqrt assumption.

This bounds mathematical correction truncation ONLY. The caller must provide
a true lower bound on omitted physical distances (accounting for its cutoff
classification), and separately qualify finite-grid, GPU and roundoff errors.
It is not permission to change a physical source core or discard an image.
"""

from dataclasses import dataclass
from itertools import islice
import math
from numbers import Integral

import numpy as np

from ..gaussian_tail._interval import (
    Interval,
    _platform,
    add,
    div_positive,
    mul,
    pairwise_sum,
    point,
    sub,
)
from ..gaussian_tail.error_bounds import PreparedTailSource


@dataclass(frozen=True)
class CorrectionTailBound:
    velocity_upper: float
    gradient_upper: float
    absolute_strength_upper: float
    image_count: int


def local_correction_bound(strength, *, tau, omitted_distance_lower, image_count):
    """Uniform finite-image u/J tail bound for all target points.

    This intentionally charges every source in every finite image. Geometry
    may justify a tighter bound, but cancellation of strengths never does.
    The caller supplies an checked IEEE host arithmetic scope.
    """
    _platform()
    gamma = np.asarray(strength)
    if (
        gamma.ndim != 2
        or gamma.shape[1:] != (3,)
        or gamma.dtype.kind != "f"
        or gamma.dtype.itemsize not in (4, 8)
    ):
        raise ValueError("finite binary32/binary64 strengths (N,3) required")
    gamma = np.asarray(gamma, dtype=np.float64)
    if not np.isfinite(gamma).all():
        raise ValueError("finite real strengths (N,3) required")
    if (
        isinstance(image_count, bool)
        or not isinstance(image_count, Integral)
        or not 0 <= image_count <= 1_000_000
    ):
        raise ValueError("bounded nonnegative integer image count required")
    if not all(
        isinstance(v, (float, np.float32, np.float64)) and math.isfinite(v) and v > 0
        for v in (tau, omitted_distance_lower)
    ):
        raise ValueError("positive finite broadening and omitted-distance lower bound required")
    if not image_count or not np.any(gamma):
        return CorrectionTailBound(0.0, 0.0, 0.0, int(image_count))
    strength_sum = pairwise_sum(point(np.abs(gamma).reshape(-1)))
    radius, broadening = point(omitted_distance_lower), point(tau)
    rho = div_positive(radius, broadening)
    squared = mul(rho, rho)
    # Clip downward, not upward: exp(-actual q) <= exp(-min(q,64)). This
    # avoids overflowing a positive series for extremely separated sources.
    q = point(min(float(squared.lower), 64.0))
    if q.lower <= 0:
        raise ValueError("positive representable separation ratio required")
    total, term = point(1.0), point(1.0)
    for index in range(1, 129):
        term = div_positive(mul(term, q), point(float(index)))
        total = add(total, term)
    exponential = div_positive(point(1.0), point(float(total.lower)))
    r2 = mul(radius, radius)
    r3, r4 = mul(r2, radius), mul(r2, r2)
    tau3 = mul(broadening, mul(broadening, broadening))
    u_factor = div_positive(
        add(div_positive(broadening, r3), div_positive(point(2.0), mul(broadening, radius))),
        point(20.0),
    )
    j_factor = add(
        div_positive(
            mul(
                point(3.0),
                add(div_positive(broadening, r4), div_positive(point(2.0), mul(broadening, r2))),
            ),
            point(20.0),
        ),
        div_positive(point(3.0), mul(point(10.0), tau3)),
    )
    common = mul(mul(strength_sum, point(float(image_count))), exponential)
    return CorrectionTailBound(
        float(mul(common, u_factor).upper),
        float(mul(common, j_factor).upper),
        float(strength_sum.upper),
        int(image_count),
    )


@dataclass(frozen=True)
class FiniteCorrectionTailBound:
    """Outward finite-image sum; immutable per-image evidence, no cancellation."""

    velocity_upper: float
    gradient_upper: float
    absolute_strength_upper: float
    image_count: int
    cutoff_limited_images: int
    descriptors: tuple
    distance_lower_bounds: tuple
    velocity_upper_by_image: tuple
    gradient_upper_by_image: tuple
    source_sha256: str


def _frozen_interval(value, shape):
    lower, upper = np.asarray(value.lower, np.float64), np.asarray(value.upper, np.float64)
    if (
        lower.shape != shape
        or upper.shape != shape
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or np.any(lower > upper)
    ):
        raise ValueError("invalid immutable validated source interval")
    return Interval(lower, upper)


def _vector_correction_envelopes(distance, tau, strength_upper):
    """Same rational proof as the scalar bound, vectorized over ALL images.

    A single128-iteration positive exponential series operates on the image
    vector. No libm exponential, rounded norm, or cancellation is used.
    ``distance`` and ``strength_upper`` are already validated one-sided inputs.
    """
    radius, broadening = point(distance), point(tau)
    rho = div_positive(radius, broadening)
    squared = mul(rho, rho)
    q = point(np.minimum(squared.lower, 64.0))
    if np.any(q.lower <= 0):
        raise ValueError("positive representable separation ratio required")
    total = term = point(np.ones_like(distance))
    for index in range(1, 129):
        term = div_positive(mul(term, q), point(float(index)))
        total = add(total, term)
    exponential = div_positive(point(1.0), point(total.lower))
    r2 = mul(radius, radius)
    r3, r4 = mul(r2, radius), mul(r2, r2)
    tau3 = mul(broadening, mul(broadening, broadening))
    u_factor = div_positive(
        add(div_positive(broadening, r3), div_positive(point(2.0), mul(broadening, radius))),
        point(20.0),
    )
    j_factor = add(
        div_positive(
            mul(
                point(3.0),
                add(div_positive(broadening, r4), div_positive(point(2.0), mul(broadening, r2))),
            ),
            point(20.0),
        ),
        div_positive(point(3.0), mul(point(10.0), tau3)),
    )
    common = mul(point(strength_upper), exponential)
    return mul(common, u_factor), mul(common, j_factor)


def finite_image_correction_bound(
    snapshot, query_lower, query_upper, *, images, tau, omitted_distance_lower, max_images=4096
):
    """Enclose omitted source-only core correction for an explicit finite set.

    ``snapshot`` must be produced by ``prepare_tail_source``. Its owned exact
    source bytes, max actual core, outward L1 strength and both source-family
    AABBs are reused; no second particle reduction or mutable array is retained.
    The caller MUST validate the snapshot's source epoch before using this
    result with device fields. A hash alone is not that validation.

    Each ``(k, odd)`` denotes translation2*k*(zmax-zmin), plus reflection about
    zmin when odd. Source and query boxes are expressed relative to the same
    frozen origin; this avoids loss from reconstructing large world offsets.
    The outward interval displacement contains every source-target pair.
    Its maximum axis separation is a LOWER bound on all Euclidean distances.
    We take its maximum with the supplied TRUE omitted-distance lower bound.

    Every supplied image is charged independently, including duplicates and
    primary(0,False) if supplied. Empty sets are allowed. This does not select
    images, check_context a primary particle operator, infer a cutoff-classification
    error bound, or validate FFT/interpolation/device arithmetic errors.
    """
    _platform()
    if type(snapshot) is not PreparedTailSource:
        raise TypeError("prepared immutable source snapshot required")
    if (
        isinstance(max_images, bool)
        or not isinstance(max_images, Integral)
        or not 1 <= max_images <= 1_000_000
    ):
        raise ValueError("bounded positive image cap required")
    if not all(
        isinstance(v, (float, np.float32, np.float64)) and math.isfinite(v) and v > 0
        for v in (tau, omitted_distance_lower)
    ):
        raise ValueError(
            "positive finite binary32/binary64 broadening and omitted-distance bound required"
        )
    if not math.isfinite(snapshot.core_max) or snapshot.core_max < 0 or snapshot.core_max > tau:
        raise ValueError("broadening must cover every actual source-only core")
    bounds = [np.asarray(value) for value in (query_lower, query_upper)]
    if any(
        value.shape != (3,)
        or value.dtype.kind != "f"
        or value.dtype.itemsize not in (4, 8)
        or not np.isfinite(value).all()
        for value in bounds
    ):
        raise ValueError("finite binary32/binary64 query AABB required")
    lower, upper = (np.array(value, dtype=np.float64, copy=True) for value in bounds)
    if np.any(lower > upper):
        raise ValueError("ordered query AABB required")
    descriptors = tuple(islice(iter(images), int(max_images) + 1))
    if len(descriptors) > max_images:
        raise ValueError("finite image list exceeds cap")
    saved = []
    for descriptor in descriptors:
        if not isinstance(descriptor, (tuple, list)) or len(descriptor) != 2:
            raise ValueError("integer index and boolean reflection descriptors required")
        k, odd = descriptor
        if (
            isinstance(k, bool)
            or not isinstance(k, Integral)
            or abs(k) > 2**30
            or type(odd) not in (bool, np.bool_)
        ):
            raise ValueError("bounded integer index and boolean reflection required")
        saved.append((int(k), bool(odd)))
    descriptors = tuple(saved)
    strength = _frozen_interval(snapshot.absolute_strength, ())
    strength_upper = float(strength.upper)
    if strength_upper < 0:
        raise ValueError("nonnegative validated strength required")
    if not descriptors:
        return FiniteCorrectionTailBound(
            0.0, 0.0, strength_upper, 0, 0, (), (), (), (), snapshot.source_sha256
        )
    period = _frozen_interval(snapshot.period, ())
    if period.lower <= 0:
        raise ValueError("positive enclosed slab period required")
    origin = np.asarray(snapshot.origin, np.float64)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("finite immutable source origin required")
    target = sub(Interval(lower, upper), point(origin))
    families = tuple(_frozen_interval(item.source_box, (3,)) for item in snapshot.families)
    if len(families) != 2:
        raise ValueError("both validated source families required")
    odd = np.asarray([value[1] for value in descriptors], dtype=np.int32)
    index = np.asarray([value[0] for value in descriptors], dtype=np.float64)
    source_lower = np.stack([value.lower for value in families])[odd].copy()
    source_upper = np.stack([value.upper for value in families])[odd].copy()
    shift = mul(point(index), period)
    z = add(Interval(source_lower[:, 2], source_upper[:, 2]), shift)
    source_lower[:, 2], source_upper[:, 2] = z.lower, z.upper
    displacement = sub(Interval(source_lower, source_upper), target)
    distance_lower = np.maximum(0.0, np.maximum(displacement.lower, -displacement.upper)).max(
        axis=1
    )
    distance = np.maximum(float(omitted_distance_lower), distance_lower)
    cutoff_limited = int(np.count_nonzero(distance_lower <= omitted_distance_lower))
    if snapshot.identically_zero or not snapshot.source_count:
        zero = (0.0,) * len(descriptors)
        return FiniteCorrectionTailBound(
            0.0,
            0.0,
            strength_upper,
            len(descriptors),
            cutoff_limited,
            descriptors,
            tuple(distance.tolist()),
            zero,
            zero,
            snapshot.source_sha256,
        )
    u, j = _vector_correction_envelopes(distance, float(tau), strength_upper)
    return FiniteCorrectionTailBound(
        float(pairwise_sum(u).upper),
        float(pairwise_sum(j).upper),
        strength_upper,
        len(descriptors),
        cutoff_limited,
        descriptors,
        tuple(distance.tolist()),
        tuple(u.upper.tolist()),
        tuple(j.upper.tolist()),
        snapshot.source_sha256,
    )
