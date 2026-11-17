"""Mathematical envelope for an omitted finite core correction.

For Gaussian source cores sigma <= tau and omitted distances r >= r_c,
  |du| <= |Gamma| T_tau(r_c)/r_c**2,
  ||dJ||_F <= sqrt(2)|Gamma| (2*T_tau(r_c)/r_c**3 + Z_tau(r_c)).
Here T=(erfc(r/tau)+2*(r/tau)*exp(-(r/tau)**2)/sqrt(pi))/(4*pi).
The bounds follow from the positive integral for narrow-minus-broad radial
factors. They include the full Jacobian, not only its symmetric part.

As sigma -> 0 the same bounds hold for Biot-Savart minus the broad Gaussian.
Another kernel K differs from the broad Gaussian by that plus its own core
defect, Biot-Savart minus K, which its definition's envelope bounds in the
partner coordinate; the source cores are then the partner widths.

Use erfc(q) <= exp(-q*q)/(sqrt(pi)*q), pi**1.5 > 5, sqrt(2)<3/2,
and a positive lower sum for exp(q*q). The bounds are float64 evaluations
raised by the safety factor of ``float_bounds``. The source L1 norm dominates
its Euclidean norm, without a sqrt assumption.

This bounds mathematical correction truncation ONLY. The caller must provide
a true lower bound on omitted physical distances (accounting for its cutoff
classification, see ``omitted_distance_lower``), and separately qualify
finite-grid, GPU and roundoff errors. It is not permission to change a
physical source core or discard an image.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import islice
import math
from numbers import Integral
import struct
from typing import TYPE_CHECKING

import numpy as np

from ..gaussian_tail.error_bounds import PreparedTailSource
from ..gaussian_tail.float_bounds import (
    COORDINATE_SLACK,
    exponential_upper,
    positive_sum,
    upper_bound,
)
from ..gaussian_tail.kernel_defect import source_defect_upper

if TYPE_CHECKING:
    from ....kernels.definition import AlgebraicDefectEnvelope, GaussianDefectEnvelope


@dataclass(frozen=True)
class FiniteCorrectionTailBound:
    """Upper bounds of the finite-image sum; immutable per-image evidence, no cancellation."""

    velocity_upper: float
    gradient_upper: float
    absolute_strength_upper: float
    descriptors: tuple
    source_sha256: str


def _vector_correction_envelopes(
    distance: np.ndarray, tau: float, strength_upper: float
) -> tuple[np.ndarray, np.ndarray]:
    """Same bound as the scalar one, vectorized over ALL images.

    A single 128-term positive exponential series operates on the image
    vector. No libm exponential, rounded norm, or cancellation is used.
    ``distance`` and ``strength_upper`` are already validated one-sided inputs.
    """
    rho = distance / tau
    squared = rho * rho
    if np.any(np.minimum(squared, 64.0) <= 0):
        raise ValueError("positive representable separation ratio required")
    exponential = exponential_upper(squared)
    r2 = distance * distance
    r3, r4 = r2 * distance, r2 * r2
    tau3 = tau * (tau * tau)
    u_factor = (tau / r3 + 2.0 / (tau * distance)) / 20.0
    j_factor = 3.0 * (tau / r4 + 2.0 / (tau * r2)) / 20.0 + 3.0 / (10.0 * tau3)
    common = strength_upper * exponential
    return common * u_factor, common * j_factor


def omitted_distance_lower(
    snapshot: "PreparedTailSource",
    query_lower: np.ndarray,
    query_upper: np.ndarray,
    *,
    cutoff: float,
    images: Sequence[tuple[int, bool]],
) -> float:
    """Lower bound on the distance of every pair the correction omits.

    The correction (``correction.py``) omits the pairs its cell classifier
    puts beyond ``cutoff``. The classifier works in binary64. Let e = 2**-52
    and let G >= 1 bound the absolute original coordinates, origins, image
    shifts and the cutoff. A pair strictly closer than cutoff - D is never
    omitted, with D = 1024 e G plus the error of the represented image shift:

    * An image shift is at most four rounded operations, each below e/2 of a
      quantity at most G, so its error is below 2 e G.
    * Host box subtraction and the unpadded device far-grid return discard a
      pair only at distance >= cutoff - 32 e G. The upper grid endpoint
      n = floor(fl(fl(max - origin)/cutoff)) + 1 gives n cutoff at least the
      true source extent minus 6 e G.
    * The cell quotients are multiplied by the cutoff back, so the source key
      errs by less than 8 e G and the padded query end by less than 64 e G.
      The computed lower quotient is below the computed source quotient, which
      is below the computed upper one, and a monotone floor keeps the source key.
    * A rounded squared norm that passes the exclusion test gives
      ||computed d|| >= cutoff (1 - 8 e), so the distance exceeds cutoff - 32 e G.

    These constants are below 1024 even with the grid endpoint products and
    the padding. D = 1032 e G covers the shift error and the rounding of
    ``cutoff - D``, which is at most e G / 2 because G >= cutoff. D is below
    1e-9 cutoff whenever G < 4.3e3 cutoff. The slab tests and the tutorial
    settings (domains of 35 m by 10 m, cutoffs of 0.3 m to 1.8 m) have G/cutoff
    of at most 100, where D is below 2.4e-11 cutoff. Beyond D >= cutoff/8, or
    G > 2**250, the cutoff is unresolved at the coordinate scale and the call
    raises. This is NOT a bound on the radial
    evaluation, FFT or interpolation arithmetic. The correction call must use
    these images and the same cutoff, source values matching ``snapshot`` and
    targets inside the query box; a different classifier voids the bound.
    """
    zmin, zmax = struct.unpack("!dd", snapshot.source_data.slab_bytes)
    origin = np.asarray(snapshot.origin, np.float64)
    families = [
        (
            np.asarray(item.source_box.lower, np.float64),
            np.asarray(item.source_box.upper, np.float64),
        )
        for item in snapshot.families
    ]
    corners = np.concatenate(
        (
            np.asarray(query_lower, np.float64),
            np.asarray(query_upper, np.float64),
            origin,
            *(box + origin for family in families for box in family),
        )
    )
    largest_index = max((abs(k) for k, _ in images), default=0)
    scale = max(
        1.0,
        cutoff,
        abs(zmin),
        abs(zmax),
        snapshot.period,
        float(np.max(np.abs(corners))),
        largest_index * snapshot.period + 2.0 * abs(zmin),
    )
    if not math.isfinite(scale) or scale > 2.0**250:
        raise ValueError("coordinate scale outside proven binary64 classification range")
    loss = 1032.0 * np.finfo(np.float64).eps * scale
    if loss >= cutoff / 8.0:
        raise ValueError("cutoff is unresolved at this coordinate/image-shift scale")
    radius = cutoff - loss
    box_lower, box_upper = families[0]
    extent = upper_bound((box_upper - box_lower) / cutoff)
    if not np.isfinite(extent).all() or np.any(extent > np.iinfo(np.int32).max - 2):
        raise ValueError("source grid extent outside proven index range")
    shape_upper = np.floor(np.maximum(0.0, extent)).astype(np.int64) + 1
    if math.prod(shape_upper.tolist()) > np.iinfo(np.int32).max:
        raise ValueError("source cell-count outside proven int32 range")
    return float(radius)


def finite_image_correction_bound(
    snapshot: "PreparedTailSource",
    query_lower: np.ndarray,
    query_upper: np.ndarray,
    *,
    images: Sequence[tuple[int, bool]],
    tau: float,
    omitted_distance_lower: float,
    max_images: int = 4096,
    kernel_defect: "AlgebraicDefectEnvelope | GaussianDefectEnvelope | None" = None,
) -> "FiniteCorrectionTailBound":
    """Bound the omitted source-only core correction for an explicit finite set.

    ``snapshot`` must be produced by ``prepare_tail_source``. Its owned exact
    source bytes, max actual core, L1 strength and both source-family
    AABBs are reused; no second particle reduction or mutable array is retained.
    The caller MUST validate the snapshot's source epoch before using this
    result with device fields. A hash alone is not that validation.

    Each ``(k, odd)`` denotes translation 2*k*(zmax-zmin), plus reflection about
    zmin when odd. Source and query boxes are expressed relative to the same
    frozen origin; this avoids loss from reconstructing large world offsets.
    The displacement box contains every source-target pair, to the coordinate
    slack of ``float_bounds``. Its maximum axis separation is a LOWER bound on
    all Euclidean distances. We take its maximum with the supplied TRUE
    omitted-distance lower bound.

    Every supplied image is charged independently, including duplicates and
    primary(0,False) if supplied. Empty sets are allowed. This does not select
    images, validate a primary particle operator, infer a cutoff-classification
    error bound, or validate FFT/interpolation/device arithmetic errors.
    """
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
            or abs(int(k)) > 2**30
            or type(odd) not in (bool, np.bool_)
        ):
            raise ValueError("bounded integer index and boolean reflection required")
        saved.append((int(k), bool(odd)))
    descriptors = tuple(saved)
    strength_upper = snapshot.absolute_strength
    if strength_upper < 0:
        raise ValueError("nonnegative validated strength required")
    if not descriptors:
        return FiniteCorrectionTailBound(0.0, 0.0, strength_upper, (), snapshot.source_sha256)
    period = snapshot.period
    if period <= 0:
        raise ValueError("positive enclosed slab period required")
    origin = np.asarray(snapshot.origin, np.float64)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("finite immutable source origin required")
    target_lower, target_upper = lower - origin, upper - origin
    if len(snapshot.families) != 2:
        raise ValueError("both validated source families required")
    boxes = [
        (
            np.asarray(item.source_box.lower, np.float64),
            np.asarray(item.source_box.upper, np.float64),
        )
        for item in snapshot.families
    ]
    if any(
        low.shape != (3,)
        or high.shape != (3,)
        or not np.isfinite(low).all()
        or not np.isfinite(high).all()
        or np.any(low > high)
        for low, high in boxes
    ):
        raise ValueError("invalid immutable validated source interval")
    odd = np.asarray([value[1] for value in descriptors], dtype=np.int32)
    index = np.asarray([value[0] for value in descriptors], dtype=np.float64)
    source_lower = np.stack([low for low, _ in boxes])[odd].copy()
    source_upper = np.stack([high for _, high in boxes])[odd].copy()
    shift = index * period
    source_lower[:, 2] += shift
    source_upper[:, 2] += shift
    separation = np.maximum(
        0.0, np.maximum(source_lower - target_upper, target_lower - source_upper)
    ).max(axis=1)
    coordinates = max(
        float(np.max(np.abs(target_lower))),
        float(np.max(np.abs(target_upper))),
        *(float(np.max(np.abs(box))) for pair in boxes for box in pair),
    )
    distance_lower = np.maximum(0.0, separation - COORDINATE_SLACK * (np.abs(shift) + coordinates))
    distance = np.maximum(float(omitted_distance_lower), distance_lower)
    if snapshot.identically_zero or not snapshot.source_count:
        return FiniteCorrectionTailBound(
            0.0, 0.0, strength_upper, descriptors, snapshot.source_sha256
        )
    u, j = _vector_correction_envelopes(distance, float(tau), strength_upper)
    if kernel_defect is not None:
        kernel_u, kernel_j = source_defect_upper(
            kernel_defect, distance, snapshot.core_max, strength_upper
        )
        u, j = u + kernel_u, j + kernel_j
    return FiniteCorrectionTailBound(
        float(upper_bound(positive_sum(u))),
        float(upper_bound(positive_sum(j))),
        strength_upper,
        descriptors,
        snapshot.source_sha256,
    )
