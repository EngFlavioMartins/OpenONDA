"""Immutable source snapshots and conservative bounds on the completed image tail.

Preparation captures exact source dtype, shape and bytes, then computes the
absolute strength and both source-family boxes once. The field operator keeps
the image shells |k| <= K and replaces every omitted image row by its
closed-form integral with the first Euler-Maclaurin end correction. A query
bounds what that completion leaves over a continuous target AABB in constant
work.

This module does not select a solver stopping settings or publish induction
fields. Its bounds are float64 evaluations with the safety factor of
``float_bounds``. Callers must validate an explicit source epoch before
reusing a snapshot; which sources produced it is never inferred from object identity.
Finite-image interpolation, FFT and GPU accuracy remain separate obligations.
"""

from dataclasses import dataclass, field
import hashlib
import math
from numbers import Integral
import struct
from typing import TYPE_CHECKING, NotRequired, TypedDict

import numpy as np

from .float_bounds import COORDINATE_SLACK, positive_sum, upper_bound
from .kernel_defect import row_defect_upper

if TYPE_CHECKING:
    from ....kernels.definition import AlgebraicDefectEnvelope, GaussianDefectEnvelope


class CoreDefectBound(TypedDict):
    """Velocity and Jacobian bounds of one source family's core defect."""

    velocity_upper: float
    gradient_frobenius_upper: float
    exponential_upper: NotRequired[float]
    order: NotRequired[int]
    q_lower: NotRequired[float]
    scope: NotRequired[str]


@dataclass(frozen=True)
class SourceBox:
    """Immutable lower and upper corner of an axis-aligned box."""

    lower: tuple
    upper: tuple


def gaussian_defect_upper(
    *, gap_lower: float, sigma_upper: float, period_lower: float, absolute_strength_upper: float
) -> CoreDefectBound:
    """One source-family +/- infinite-tail defect of Gaussian cores.

    Inputs must themselves be valid one-sided bounds. A safe source strength
    is sum_j ||Gamma_j||_1, rather than assuming a rounded L2 norm is an upper
    bound. The caller separately sums the original and reflected families.
    """
    values = (gap_lower, sigma_upper, period_lower, absolute_strength_upper)
    if not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError("positive finite scalar one-sided inputs required")
    gap, sigma, period, strength = values
    q = (gap / sigma) ** 2
    if q < 1.5:
        raise ValueError("uniform-core Gaussian density monotonicity not checked")
    # exp(-q) <= order!/q^order for every order; the largest useful one is taken.
    order = min(96, math.floor(q))
    exponential = 1.0
    for k in range(1, order + 1):
        exponential *= k / q
    exponential_bound = min(1.0, exponential)
    gap2, sigma2 = gap * gap, sigma * sigma
    gap3, sigma3 = gap2 * gap, sigma2 * sigma
    gap4 = gap2 * gap2
    # pi**1.5>5 and sqrt(5)<3 give loose rational constants.
    cu = (sigma / gap3 + 2.0 / (sigma * gap)) / 20.0
    cj = 3.0 * (sigma / gap4 + 2.0 / (sigma * gap2)) / 20.0 + 1.0 / (5.0 * sigma3)
    common = strength * (sigma2 * exponential_bound) / (period * gap)
    return {
        "velocity_upper": float(upper_bound(common * cu)),
        "gradient_frobenius_upper": float(upper_bound(common * cj)),
        "exponential_upper": float(upper_bound(exponential_bound)),
        "order": order,
        "q_lower": q,
        "scope": "one source family, complete +/- omitted shells",
    }


@dataclass(frozen=True)
class SourceFamily:
    """Bounding box of one image family of the sources, about z_min."""

    source_box: SourceBox


@dataclass(frozen=True)
class SourceArrayData:
    """Original source metadata and immutable owned bytes, in logical C order."""

    dtype: str
    shape: tuple
    data_bytes: bytes = field(repr=False)


@dataclass(frozen=True)
class GaussianSourceData:
    """Owned bytes of the source arrays and slab planes of a snapshot."""

    arrays: tuple[SourceArrayData, SourceArrayData, SourceArrayData]
    slab_bytes: bytes


class SourceValueMismatchError(ValueError):
    """A prepared mathematical snapshot no longer matches supplied sources."""


def _capture_source_data(
    position: np.ndarray,
    strength: np.ndarray,
    radius: np.ndarray,
    z_min: float,
    z_max: float,
    max_sources: int,
) -> "GaussianSourceData":
    x, gamma, sigma = (np.asarray(v) for v in (position, strength, radius))
    if (
        x.ndim != 2
        or x.shape[1:] != (3,)
        or gamma.shape != x.shape
        or sigma.shape != (len(x),)
        or len(x) > max_sources
    ):
        raise ValueError("bounded source/strength/core shapes required")
    if not all(v.dtype.kind == "f" and v.dtype.itemsize in (4, 8) for v in (x, gamma, sigma)):
        raise ValueError("binary32 or binary64 source arrays required")
    if not math.isfinite(z_min) or not math.isfinite(z_max) or z_max <= z_min:
        raise ValueError("finite ordered physical slab required")
    x_data, gamma_data, sigma_data = (
        SourceArrayData(v.dtype.str, tuple(v.shape), np.ascontiguousarray(v).tobytes(order="C"))
        for v in (x, gamma, sigma)
    )
    return GaussianSourceData((x_data, gamma_data, sigma_data), struct.pack("!dd", z_min, z_max))


def _source_data_hash(source_data: "GaussianSourceData") -> str:
    digest = hashlib.sha256(b"gaussian-tail-source-v1")
    digest.update(source_data.slab_bytes)
    for item in source_data.arrays:
        digest.update(item.dtype.encode("ascii") + b"\0")
        digest.update(struct.pack("!Q", len(item.shape)))
        for extent in item.shape:
            digest.update(struct.pack("!Q", extent))
        digest.update(item.data_bytes)
    return digest.hexdigest()


def validate_source_values(
    snapshot: "PreparedTailSource",
    position: np.ndarray,
    strength: np.ndarray,
    radius: np.ndarray,
    *,
    z_min: float,
    z_max: float,
) -> None:
    """Require exact original dtype, shape, signed zeros and values.

    This does not refresh or rebuild a mismatched snapshot. A caller must
    prepare a new snapshot after a source or slab change. SHA256 is source information
    only; validation compares the retained immutable bytes directly.
    """
    if type(snapshot) is not PreparedTailSource:
        raise TypeError("prepared immutable source snapshot required")
    actual = _capture_source_data(position, strength, radius, z_min, z_max, snapshot.source_count)
    if actual != snapshot.source_data:
        raise SourceValueMismatchError("Gaussian tail source values or slab configuration changed")


@dataclass(frozen=True)
class PreparedTailSource:
    """Immutable source snapshot with its strength, boxes and largest core.

    ``period`` is 2 (z_max - z_min) and ``absolute_strength`` the sum of the
    absolute strength components, both float64 evaluations whose rounding the
    bounds of ``float_bounds`` absorb. The boxes are relative to ``origin``.
    """

    source_count: int
    source_sha256: str
    source_data: GaussianSourceData = field(repr=False)
    origin: tuple
    period: float
    absolute_strength: float
    families: tuple[SourceFamily, SourceFamily]
    core_max: float
    identically_zero: bool


def prepare_tail_source(
    position: np.ndarray,
    strength: np.ndarray,
    radius: np.ndarray,
    *,
    z_min: float,
    z_max: float,
    max_sources: int = 400_000,
) -> "PreparedTailSource":
    """Capture owned exact source values, their absolute strength and family boxes.

    Only binary32/binary64 source fields are checked. The retained immutable
    bytes are intentional: later validation compares values exactly, not hashes.
    No caller arrays, views or device fields are retained.
    """
    if (
        isinstance(max_sources, bool)
        or not isinstance(max_sources, Integral)
        or not 1 <= max_sources <= 1_000_000
    ):
        raise ValueError("bounded positive source cap required")
    source_data = _capture_source_data(position, strength, radius, z_min, z_max, max_sources)
    x, gamma, sigma = (
        np.frombuffer(item.data_bytes, dtype=np.dtype(item.dtype))
        .reshape(item.shape)
        .astype(np.float64, copy=True)
        for item in source_data.arrays
    )
    if (
        not all(np.isfinite(v).all() for v in (x, gamma, sigma))
        or np.any(sigma <= 0)
        or np.any(x[:, 2] < z_min)
        or np.any(x[:, 2] > z_max)
    ):
        raise ValueError("finite sources inside physical slab and positive cores required")
    digest = _source_data_hash(source_data)
    period = 2.0 * (z_max - z_min)
    if not period > 0:
        raise ValueError("positive enclosed period required")
    if not len(x):
        zero = SourceBox((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        family = SourceFamily(zero)
        return PreparedTailSource(
            0, digest, source_data, (0.0, 0.0, z_min), period, 0.0, (family, family), 0.0, True
        )
    origin = 0.5 * x.min(axis=0) + 0.5 * x.max(axis=0)
    origin[2] = z_min
    local = x - origin
    box_lower, box_upper = local.min(axis=0), local.max(axis=0)
    # Odd images reflect about z_min, the origin of the local z axis.
    reflected_lower, reflected_upper = box_lower.copy(), box_upper.copy()
    reflected_lower[2], reflected_upper[2] = -box_upper[2], -box_lower[2]
    return PreparedTailSource(
        len(x),
        digest,
        source_data,
        tuple(origin.tolist()),
        period,
        positive_sum(np.abs(gamma)),
        (
            SourceFamily(SourceBox(tuple(box_lower.tolist()), tuple(box_upper.tolist()))),
            SourceFamily(
                SourceBox(tuple(reflected_lower.tolist()), tuple(reflected_upper.tolist()))
            ),
        ),
        float(sigma.max()),
        not np.any(gamma),
    )


@dataclass(frozen=True)
class QueryTailBound:
    """Bounds on what the completed image tail leaves over a query box."""

    velocity_upper: float
    gradient_upper: float
    diagnostics: dict


class ImageTailSeparationError(ValueError):
    """The retained images do not separate the omitted rows from the query box."""


def query_tail_bound(
    snapshot: "PreparedTailSource",
    query_lower: np.ndarray,
    query_upper: np.ndarray,
    *,
    shells: int,
    minimum_separation: float = 0.0,
    kernel_defect: "AlgebraicDefectEnvelope | GaussianDefectEnvelope | None" = None,
) -> "QueryTailBound":
    """Enclose what the completed image tail leaves over a whole query box.

    Each source family repeats with period P in z. The field keeps the rows
    |k| <= K and adds, on each side, the omitted rows t_j = Z + (j + 1/2) P
    as (1/P) int_Z^inf f dt + (P/24) f'(Z) of the singular Biot-Savart
    kernel f, where Z is the distance from the query to the midpoint boundary
    of the omitted rows. The midpoint Euler-Maclaurin remainder is at most
    (P^3/384) int_Z^inf |d4f/dt4| dt, because |B4(x) - B4(1/2)| <= 1/16.

    The fourth z derivatives are fifth and sixth derivatives of 1/(4 pi R).
    Banach's theorem for symmetric multilinear forms and |P_n| <= 1 bound
    them by 5!/(4 pi R^6) for the velocity and, for the trace-free gradient,
    by sqrt(2) 6!/(4 pi R^7) in the Frobenius norm. With S the summed L1
    vortex strength, each side of each family therefore leaves at most

      |du| <= S P^3/(64 pi Z^5),   ||dJ||_F <= 5 sqrt(2) S P^3/(64 pi Z^6),

    plus the core defect of the omitted images: the Gaussian one, or the
    ``kernel_defect`` envelope of another kernel, whose cores are then the
    snapshot's partner widths. The completion is smooth on the auxiliary mesh
    only when Z exceeds ``minimum_separation``.
    """
    if type(snapshot) is not PreparedTailSource:
        raise TypeError("prepared immutable source snapshot required")
    lower, upper = (
        np.asarray(query_lower, dtype=np.float64),
        np.asarray(query_upper, dtype=np.float64),
    )
    if (
        lower.shape != (3,)
        or upper.shape != (3,)
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or np.any(lower > upper)
    ):
        raise ValueError("finite ordered query AABB required")
    if isinstance(shells, bool) or not isinstance(shells, Integral) or not 1 <= shells <= 1 << 30:
        raise ValueError("shells must be a positive integer")
    if not math.isfinite(minimum_separation) or minimum_separation < 0:
        raise ValueError("finite nonnegative minimum separation required")
    diagnostics = {
        "shells": int(shells),
        "source_count": snapshot.source_count,
        "source_sha256": snapshot.source_sha256,
        "query_lower": lower.tolist(),
        "query_upper": upper.tolist(),
        "families": [],
        "runtime_admissible": False,
        "scope": "omitted image rows after their closed-form midpoint completion",
        "finite_image_error_bounded": False,
    }
    if snapshot.identically_zero:
        return QueryTailBound(0.0, 0.0, diagnostics)
    origin = np.asarray(snapshot.origin)
    target_z_lower, target_z_upper = float(lower[2] - origin[2]), float(upper[2] - origin[2])
    period, strength = snapshot.period, snapshot.absolute_strength
    retained = period * (shells + 0.5)
    # Both sides of one family; 99/70 exceeds sqrt(2).
    velocity_coefficient = 2.0 * (strength * (period * period * period)) / (64.0 * math.pi)
    gradient_coefficient = (495.0 / 70.0) * velocity_coefficient
    ru = rj = gu = gj = 0.0
    for family in snapshot.families:
        source_z_lower, source_z_upper = family.source_box.lower[2], family.source_box.upper[2]
        reach = max(source_z_upper - target_z_lower, target_z_upper - source_z_lower)
        z_scale = max(
            abs(source_z_lower), abs(source_z_upper), abs(target_z_lower), abs(target_z_upper)
        )
        lost = COORDINATE_SLACK * (retained + z_scale)
        boundary = retained - reach - lost
        if boundary <= 0 or boundary <= minimum_separation:
            raise ImageTailSeparationError(
                "retained shells must separate the omitted image rows from the query box"
            )
        # The nearest omitted image lies one period beyond this Gaussian gap.
        gap = retained - reach - 0.5 * period - lost
        ratio = max(gap, 0.0) / snapshot.core_max
        if kernel_defect is None:
            if gap <= 0 or ratio * ratio < 1.5:
                raise ImageTailSeparationError("omitted images must lie beyond the Gaussian cores")
        elif gap <= 0 or ratio < kernel_defect.minimum_ratio:
            raise ImageTailSeparationError("omitted images must lie beyond the kernel cores")
        fifth = (boundary * boundary) * (boundary * boundary) * boundary
        ur = velocity_coefficient / fifth
        jr = gradient_coefficient / (fifth * boundary)
        core: CoreDefectBound
        if kernel_defect is None:
            core = gaussian_defect_upper(
                gap_lower=gap,
                sigma_upper=snapshot.core_max,
                period_lower=period,
                absolute_strength_upper=strength,
            )
        else:
            velocity, gradient = row_defect_upper(
                kernel_defect,
                gap_lower=gap,
                core_upper=snapshot.core_max,
                period_lower=period,
                strength_upper=strength,
            )
            core = {"velocity_upper": velocity, "gradient_frobenius_upper": gradient}
        ru, rj = ru + ur, rj + jr
        gu, gj = gu + core["velocity_upper"], gj + core["gradient_frobenius_upper"]
        diagnostics["families"].append(
            {"completion_distance": boundary, "minimum_gap": gap, "core_defect": core}
        )
    total_u = float(upper_bound(ru + gu))
    total_j = float(upper_bound(rj + gj))
    return QueryTailBound(total_u, total_j, diagnostics)
