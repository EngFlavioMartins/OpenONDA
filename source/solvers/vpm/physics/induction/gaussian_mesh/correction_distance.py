"""Distance validation for the existing binary64 correction-cell classifier.

This is NOT a bound on radial evaluation, FFT or interpolation arithmetic.
It validates a lower physical distance for pairs omitted by correction.py,
provided that its source snapshot, query AABB, image list and cutoff match the
actual call. The source epoch still requires exact-value validation by the
session; a source hash is source information, not that validation.

The distance check is intentionally conservative. Let e=2**-52 and let G>=1 bound
the absolute original coordinates, origins, supplied world shifts and cutoff.
Require G<=2**250 and an outward loss D=1024*e*G+shift_error < cutoff/8.
The floor/index operations are binary64 RN, with correctly rounded floor,
and the kernel has no fast-math/approximate-division compilation option.
Fused multiply-add only tightens the bounds used below. Underflow in tiny
differences is absorbed by e*G; cutoff squared cannot be subnormal under D.

Why every omission is covered (errors below are in physical length units):

* Forward or inverse reflection/translation has at most two rounded sums;
  each displacement component differs by <8eG from its exact represented-
  shift value. The exact slab shift differs by at most shift_error.
* Host AABB subtraction and the unpadded device far-grid return can discard
  a pair only at distance >= cutoff-32eG. The upper grid endpoint follows
  from n=floor(fl(fl(max-origin)/cutoff))+1, so n*cutoff is at least the
  true source extent minus6eG. No exact affine source lattice is assumed.
* For a pair strictly inside cutoff-D, its exact source coordinate is
  strictly between inverse_query +/- cutoff with D-shift_error slack.
  Multiplying each computed cell quotient by cutoff, source-key error is
  <8eG and padded query-endpoint error is <64eG (at most five operations,
  all numerator intermediates <8G). Existing nonnegative64e*local_scale
  padding is helpful, not needed for the1024eG slack. Hence computed lower
  quotient < computed source quotient < computed upper quotient; monotonic
  floor includes the source key. The source grid is checked int32-sized,
  so the device's early bounds also keep all int64 conversions representable.
* If the rounded squared norm passes the exclusion test, five nonnegative
  product/sum operations and one cutoff square imply ||computed_d|| >=
  cutoff*(1-8e). Combining displacement errors gives distance >=cutoff-32eG.

These conservative32/64 constants are dominated by1024 even when including
rounding of grid endpoint products and padding. Thus no physical pair with
r < downward(cutoff-D) can be omitted through any of the four paths. This
does not assert an exact r=cutoff decision, nor an exact finite image field.
"""

from dataclasses import dataclass
from itertools import islice
import math
from numbers import Integral
import struct

import numpy as np

from ..gaussian_tail._interval import Interval, _platform, add, div_positive, mul, point, sub
from ..gaussian_tail.error_bounds import PreparedTailSource


@dataclass(frozen=True)
class CorrectionClassificationBound:
    omitted_distance_lower: float
    cutoff: float
    coordinate_scale_upper: float
    arithmetic_loss_upper: float
    world_shift_error_upper: float
    descriptors: tuple
    world_images: tuple
    query_lower: tuple
    query_upper: tuple
    source_cell_shape_upper: tuple
    source_sha256: str


def correction_classification_bound(
    snapshot, query_lower, query_upper, *, cutoff, images, max_images=4096
):
    """Validate the fixed correction.py classifier and return immutable evidence.

    Images are integer slab descriptors(k,odd). ``world_images`` uses exactly
    coordinates.finite_images' binary64 expression. The correction call MUST
    use that list, source values matching snapshot, and targets within the
    checked AABB. Caller mutation of internal correction-solver device fields,
    different compilation flags, or a different classifier voids validation.
    No GPU state is inspected or changed by this pure-host function.
    """
    _platform()
    if type(snapshot) is not PreparedTailSource:
        raise TypeError("prepared immutable source snapshot required")
    if (
        not isinstance(cutoff, (float, np.float32, np.float64))
        or not math.isfinite(cutoff)
        or cutoff <= 0
    ):
        raise ValueError("positive finite binary32/binary64 cutoff required")
    cutoff = float(cutoff)
    if (
        isinstance(max_images, bool)
        or not isinstance(max_images, Integral)
        or not 1 <= max_images <= 1_000_000
    ):
        raise ValueError("bounded positive image cap required")
    bounds = [np.asarray(value) for value in (query_lower, query_upper)]
    if any(
        value.shape != (3,)
        or value.dtype.kind != "f"
        or value.dtype.itemsize not in (4, 8)
        or not np.isfinite(value).all()
        for value in bounds
    ):
        raise ValueError("finite binary32/binary64 query AABB required")
    lower, upper = (np.array(value, np.float64, copy=True) for value in bounds)
    if np.any(lower > upper):
        raise ValueError("ordered query AABB required")
    descriptors = tuple(islice(iter(images), int(max_images) + 1))
    if len(descriptors) > max_images:
        raise ValueError("finite image-count cap exceeded")
    saved = []
    for descriptor in descriptors:
        if not isinstance(descriptor, (tuple, list)) or len(descriptor) != 2:
            raise ValueError("integer index and boolean reflection required")
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
    zmin, zmax = struct.unpack("!dd", snapshot.source_data.slab_bytes)
    if not math.isfinite(zmin) or not math.isfinite(zmax) or zmax <= zmin:
        raise ValueError("finite ordered immutable slab required")
    period = Interval(np.asarray(snapshot.period.lower), np.asarray(snapshot.period.upper))
    if (
        period.lower.shape != ()
        or period.upper.shape != ()
        or not np.isfinite(period.lower)
        or not np.isfinite(period.upper)
        or period.lower <= 0
        or period.upper < period.lower
    ):
        raise ValueError("positive finite enclosed source period required")
    family = snapshot.families[0].source_box
    origin = np.asarray(snapshot.origin, np.float64)
    source_box = Interval(
        np.asarray(family.lower, np.float64), np.asarray(family.upper, np.float64)
    )
    if (
        origin.shape != (3,)
        or source_box.lower.shape != (3,)
        or source_box.upper.shape != (3,)
        or not np.isfinite(origin).all()
        or not np.isfinite(source_box.lower).all()
        or not np.isfinite(source_box.upper).all()
        or np.any(source_box.lower > source_box.upper)
    ):
        raise ValueError("finite ordered immutable source enclosure required")
    source_box = add(source_box, point(origin))
    world, shift_error = [], 0.0
    scale = max(
        1.0,
        cutoff,
        abs(zmin),
        abs(zmax),
        float(period.upper),
        float(
            np.max(
                np.abs(np.concatenate((lower, upper, origin, source_box.lower, source_box.upper)))
            )
        ),
    )
    for k, odd in descriptors:
        exact_shift = mul(point(float(k)), period)
        if odd:
            exact_shift = add(exact_shift, mul(point(2.0), point(zmin)))
        # Keep this expression identical to finite_images, not a midpoint of
        # an interval or an algebraically rearranged physical image shift.
        represented = 2 * k * (zmax - zmin) + (2 * zmin if odd else 0.0)
        if not math.isfinite(represented):
            raise ValueError("finite world image shifts required")
        defect = sub(point(represented), exact_shift)
        shift_error = max(shift_error, abs(float(defect.lower)), abs(float(defect.upper)))
        scale = max(
            scale, abs(represented), abs(float(exact_shift.lower)), abs(float(exact_shift.upper))
        )
        world.append((represented, odd))
    if not math.isfinite(scale) or scale > 2.0**250:
        raise ValueError("coordinate scale outside proven binary64 classification range")
    arithmetic_loss = float(mul(point(1024.0 * np.finfo(np.float64).eps), point(scale)).upper)
    loss = add(point(arithmetic_loss), point(shift_error))
    if float(loss.upper) >= cutoff / 8.0:
        raise ValueError("cutoff is unresolved at this coordinate/image-shift scale")
    radius = float(sub(point(cutoff), loss).lower)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("positive omitted-distance lower bound is not representable")
    extent = div_positive(sub(point(source_box.upper), point(source_box.lower)), point(cutoff))
    if not np.isfinite(extent.upper).all() or np.any(extent.upper > np.iinfo(np.int32).max - 2):
        raise ValueError("source grid extent outside proven index range")
    shape_upper = tuple((np.floor(np.maximum(0.0, extent.upper)).astype(np.int64) + 1).tolist())
    if math.prod(shape_upper) > np.iinfo(np.int32).max:
        raise ValueError("source cell-count outside proven int32 range")
    return CorrectionClassificationBound(
        radius,
        cutoff,
        scale,
        arithmetic_loss,
        shift_error,
        descriptors,
        tuple(world),
        tuple(lower.tolist()),
        tuple(upper.tolist()),
        shape_upper,
        snapshot.source_sha256,
    )


def correction_omission_radius(
    snapshot, query_lower, query_upper, *, cutoff, images, max_images=4096
):
    """Convenience scalar lower bound; see correction_classification_bound."""
    return correction_classification_bound(
        snapshot, query_lower, query_upper, cutoff=cutoff, images=images, max_images=max_images
    ).omitted_distance_lower
