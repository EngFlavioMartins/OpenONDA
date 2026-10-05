"""Immutable source snapshots and conservative Gaussian image-tail bounds.

Preparation captures exact source dtype, shape and bytes, then computes
signed and absolute moments once. A query bounds the whole omitted infinite
tail over a continuous target AABB in constant work. Extent and weighted
squared distance are bounded separately before multiplication.

This module does not select a solver stopping settings or publish induction
fields. It does not modify rounding or denormal controls. Callers must supply
the checked IEEE environment and validate an explicit source epoch before
reusing a snapshot; source source_data is never inferred from object source_data.
Finite-image interpolation, FFT and GPU accuracy remain separate obligations.
"""

from dataclasses import dataclass, field
import hashlib
import math
from numbers import Integral
import struct

import numpy as np

from ._interval import (
    Interval,
    _platform,
    add,
    coefficient_enclosure,
    div_positive,
    gaussian_defect_upper,
    mul,
    neg,
    pairwise_sum,
    pi_interval,
    point,
    sub,
)


@dataclass(frozen=True)
class FrozenBounds:
    lower: float | tuple
    upper: float | tuple


def _freeze(value):
    if value.lower.ndim == 0:
        return FrozenBounds(float(value.lower), float(value.upper))
    if value.lower.shape != (3,):
        raise ValueError("only scalar or vec3 immutable moments supported")
    return FrozenBounds(tuple(value.lower.tolist()), tuple(value.upper.tolist()))


def _thaw(value):
    return Interval(np.asarray(value.lower), np.asarray(value.upper))


def _axis_sum(value):
    return pairwise_sum(Interval(np.moveaxis(value.lower, -1, 0), np.moveaxis(value.upper, -1, 0)))


def _squared(value):
    product = mul(value, value)
    return Interval(np.maximum(product.lower, 0.0), product.upper)


def _l1_upper(value):
    return float(pairwise_sum(point(np.maximum(np.abs(value.lower), np.abs(value.upper)))).upper)


@dataclass(frozen=True)
class FamilyMoments:
    first_moment: FrozenBounds
    source_box: FrozenBounds


@dataclass(frozen=True)
class SourceArrayData:
    """Original source metadata and immutable owned bytes, in logical C order."""

    dtype: str
    shape: tuple
    data_bytes: bytes = field(repr=False)


@dataclass(frozen=True)
class GaussianSourceData:
    arrays: tuple[SourceArrayData, SourceArrayData, SourceArrayData]
    slab_bytes: bytes

    @property
    def retained_bytes(self):
        return sum(len(item.data_bytes) for item in self.arrays) + len(self.slab_bytes)


class SourceValueMismatchError(ValueError):
    """A prepared mathematical snapshot no longer matches supplied sources."""


def _capture_source_data(position, strength, radius, z_min, z_max, max_sources):
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
    items = tuple(
        SourceArrayData(v.dtype.str, tuple(v.shape), np.ascontiguousarray(v).tobytes(order="C"))
        for v in (x, gamma, sigma)
    )
    return GaussianSourceData(items, struct.pack("!dd", z_min, z_max))


def _source_data_hash(source_data):
    digest = hashlib.sha256(b"gaussian-tail-source-v1")
    digest.update(source_data.slab_bytes)
    for item in source_data.arrays:
        digest.update(item.dtype.encode("ascii") + b"\0")
        digest.update(struct.pack("!Q", len(item.shape)))
        for extent in item.shape:
            digest.update(struct.pack("!Q", extent))
        digest.update(item.data_bytes)
    return digest.hexdigest()


def validate_source_values(snapshot, position, strength, radius, *, z_min, z_max):
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
    source_count: int
    source_sha256: str
    source_data: GaussianSourceData = field(repr=False)
    origin: tuple
    period: FrozenBounds
    net_z: FrozenBounds
    moment_x: FrozenBounds
    moment_y: FrozenBounds
    absolute_strength: FrozenBounds
    second_moment: FrozenBounds
    families: tuple[FamilyMoments, FamilyMoments]
    core_max: float
    identically_zero: bool


def prepare_tail_source(position, strength, radius, *, z_min, z_max, max_sources=400_000):
    """Capture owned exact source values and their mathematical tail moments.

    Only binary32/binary64 source fields are checked. The retained immutable
    bytes are intentional: later validation compares values exactly, not hashes.
    No caller arrays, views or device fields are retained.
    """
    _platform()
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
    period = mul(point(2.0), sub(point(z_max), point(z_min)))
    if period.lower <= 0:
        raise ValueError("positive enclosed period required")
    if not len(x):
        zero, vector = _freeze(point(0.0)), _freeze(point(np.zeros(3)))
        family = FamilyMoments(vector, vector)
        return PreparedTailSource(
            0,
            digest,
            source_data,
            (0.0, 0.0, z_min),
            _freeze(period),
            zero,
            zero,
            zero,
            zero,
            zero,
            (family, family),
            0.0,
            True,
        )
    origin = 0.5 * x.min(axis=0) + 0.5 * x.max(axis=0)
    origin[2] = z_min
    local = sub(point(x), point(origin))
    dx, dy, dz = (Interval(local.lower[:, i], local.upper[:, i]) for i in range(3))
    gx, gy, gz = (point(gamma[:, i]) for i in range(3))
    net_z = mul(point(2.0), pairwise_sum(gz))
    moment_x = pairwise_sum(neg(add(mul(point(4.0), mul(gy, dz)), mul(point(2.0), mul(gz, dy)))))
    moment_y = pairwise_sum(add(mul(point(4.0), mul(gx, dz)), mul(point(2.0), mul(gz, dx))))
    weight = _axis_sum(point(np.abs(gamma)))
    weight = Interval(np.maximum(weight.lower, 0.0), weight.upper)
    s0 = pairwise_sum(weight)
    s1 = pairwise_sum(mul(Interval(weight.lower[:, None], weight.upper[:, None]), local))
    s2 = pairwise_sum(mul(weight, _axis_sum(_squared(local))))
    box = Interval(local.lower.min(axis=0), local.upper.max(axis=0))
    reflected_first = Interval(s1.lower.copy(), s1.upper.copy())
    reflected_first.lower[2], reflected_first.upper[2] = -s1.upper[2], -s1.lower[2]
    reflected_box = Interval(box.lower.copy(), box.upper.copy())
    reflected_box.lower[2], reflected_box.upper[2] = -box.upper[2], -box.lower[2]
    families = (
        FamilyMoments(_freeze(s1), _freeze(box)),
        FamilyMoments(_freeze(reflected_first), _freeze(reflected_box)),
    )
    return PreparedTailSource(
        len(x),
        digest,
        source_data,
        tuple(origin.tolist()),
        _freeze(period),
        _freeze(net_z),
        _freeze(moment_x),
        _freeze(moment_y),
        _freeze(s0),
        _freeze(s2),
        families,
        float(sigma.max()),
        not np.any(gamma),
    )


@dataclass(frozen=True)
class QueryTailBound:
    velocity_upper: float
    gradient_upper: float
    leading_velocity_upper: float
    leading_gradient_upper: float
    singular_velocity_upper: float
    singular_gradient_upper: float
    gaussian_velocity_upper: float
    gaussian_gradient_upper: float
    diagnostics: dict


def query_tail_bound(snapshot, query_lower, query_upper, *, shells, prefix_terms=1024):
    """Enclose the unchanged finite-image sum's omitted tail over a whole box."""
    _platform()
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
    h3 = coefficient_enclosure(shells, prefix_terms=prefix_terms)
    diagnostics = {
        "shells": int(shells),
        "source_count": snapshot.source_count,
        "source_sha256": snapshot.source_sha256,
        "query_lower": lower.tolist(),
        "query_upper": upper.tolist(),
        "families": [],
        "runtime_admissible": False,
        "scope": "whole omitted infinite tail; no completion added",
        "finite_image_error_bounded": False,
    }
    if snapshot.identically_zero:
        return QueryTailBound(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, diagnostics)
    target = sub(Interval(lower, upper), point(snapshot.origin))
    period = _thaw(snapshot.period)
    s0, s2 = _thaw(snapshot.absolute_strength), _thaw(snapshot.second_moment)
    net_z, mx, my = (
        _thaw(value) for value in (snapshot.net_z, snapshot.moment_x, snapshot.moment_y)
    )
    coefficient = div_positive(
        Interval(np.asarray(h3.lower), np.asarray(h3.upper)),
        mul(point(2.0), mul(pi_interval(), mul(period, mul(period, period)))),
    )
    tx, ty = (Interval(target.lower[i], target.upper[i]) for i in range(2))
    ux = mul(coefficient, sub(neg(mul(net_z, ty)), mx))
    uy = mul(coefficient, sub(mul(net_z, tx), my))
    leading_u = _l1_upper(
        Interval(np.array([ux.lower, uy.lower, 0.0]), np.array([ux.upper, uy.upper, 0.0]))
    )
    cross = mul(coefficient, net_z)
    jlo, jhi = np.zeros(9), np.zeros(9)
    jlo[1], jhi[1], jlo[3], jhi[3] = -cross.upper, -cross.lower, cross.lower, cross.upper
    leading_j = _l1_upper(Interval(jlo, jhi))
    target_squared = pairwise_sum(_squared(target))
    cu = div_positive(point(51.0), mul(point(48.0), mul(pi_interval(), period)))
    cj = div_positive(point(51.0), mul(point(16.0), mul(pi_interval(), period)))
    ru = rj = gu = gj = 0.0
    for family in snapshot.families:
        s1, box = _thaw(family.first_moment), _thaw(family.source_box)
        squared_moment = add(
            sub(mul(s0, target_squared), mul(point(2.0), pairwise_sum(mul(target, s1)))), s2
        )
        if squared_moment.upper < 0:
            raise FloatingPointError("squared-distance moment lost enclosure")
        moment_upper = float(squared_moment.upper)
        extent_upper = _l1_upper(sub(target, box))
        gap = sub(mul(period, point(float(shells))), point(extent_upper))
        if gap.lower <= 0:
            raise ValueError("a*K must exceed enclosed query/source L1 extent")
        gap_squared = mul(point(float(gap.lower)), point(float(gap.lower)))
        gap_fourth = mul(gap_squared, gap_squared)
        # Independent extrema, not a corner maximum of their product.
        ur = div_positive(mul(cu, mul(point(extent_upper), point(moment_upper))), gap_fourth)
        jr = div_positive(mul(cj, point(moment_upper)), gap_fourth)
        gaussian = gaussian_defect_upper(
            gap_lower=float(gap.lower),
            sigma_upper=snapshot.core_max,
            period_lower=float(period.lower),
            absolute_strength_upper=float(s0.upper),
        )
        ru = float(add(point(ru), point(float(ur.upper))).upper)
        rj = float(add(point(rj), point(float(jr.upper))).upper)
        gu = float(add(point(gu), point(gaussian["velocity_upper"])).upper)
        gj = float(add(point(gj), point(gaussian["gradient_frobenius_upper"])).upper)
        diagnostics["families"].append(
            {
                "maximum_l1_extent": extent_upper,
                "maximum_weighted_squared_distance": moment_upper,
                "minimum_gap": float(gap.lower),
                "gaussian": gaussian,
            }
        )
    total_u = float(add(point(leading_u), add(point(ru), point(gu))).upper)
    total_j = float(add(point(leading_j), add(point(rj), point(gj))).upper)
    return QueryTailBound(total_u, total_j, leading_u, leading_j, ru, rj, gu, gj, diagnostics)
