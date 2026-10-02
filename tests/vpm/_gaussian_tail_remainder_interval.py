"""UNWIRED outward-rounded singular and Gaussian paired-image tail bounds.

Uses L1 source strength and AABB L1 source-target extent, deliberately looser
than the earlier ordinary-f64 Euclidean bound. sqrt(2520)<51. The source
weighted squared distance is evaluated with origin-centred moments:
 sum_j w_j |t-x_j|² = S0|t|² - 2 t dot S1 + S2.
Every signed moment and operation is enclosed by the existing directed basic
operation helper; no cancellation-sensitive relative-error gate is used.
Reflection is exact in relative coordinates: z-zmin -> -(z-zmin), avoiding
an intermediate rounded world-coordinate reflection. The Gaussian defect
uses each family's global minimum enclosed gap and a rational exponential
bound. These bounds concern mathematical infinite-image truncation only,
not finite-image mesh/FFT/GPU errors or production stopping behaviour.
"""

from dataclasses import dataclass
import math
from numbers import Integral

import numpy as np

from tests.vpm._gaussian_tail_arithmetic import (
    Interval,
    add,
    div_positive,
    gaussian_defect_upper,
    mul,
    pairwise_sum,
    pi_interval,
    point,
    sub,
)


def _axis_sum(value):
    return pairwise_sum(Interval(np.moveaxis(value.lower, -1, 0),
                                 np.moveaxis(value.upper, -1, 0)))


def _squared(value):
    product = mul(value, value)
    return Interval(np.maximum(product.lower, 0.), product.upper)


def _record(value):
    return {"lower": value.lower.tolist(), "upper": value.upper.tolist()}


@dataclass(frozen=True)
class TailRemainder:
    singular_velocity: np.ndarray
    singular_gradient: np.ndarray
    gaussian_velocity: np.ndarray
    gaussian_gradient: np.ndarray
    velocity: np.ndarray
    gradient: np.ndarray
    diagnostics: dict


def tail_remainder(position, strength, radius, targets, *, z_min, z_max, shells,
                   max_sources=400_000, max_targets=400_000):
    if (isinstance(shells, bool) or not isinstance(shells, Integral) or not 1 <= shells <= 1 << 30
            or any(isinstance(v, bool) or not isinstance(v, Integral) or not 1 <= v <= 1_000_000
                   for v in (max_sources, max_targets))):
        raise ValueError("positive bounded integer shell/work limits required")
    x, gamma, sigma, target = (np.asarray(v) for v in (position, strength, radius, targets))
    if (x.ndim != 2 or x.shape[1:] != (3,) or gamma.shape != x.shape or sigma.shape != (len(x),)
            or target.ndim != 2 or target.shape[1:] != (3,) or len(x) > max_sources or len(target) > max_targets):
        raise ValueError("bounded source/strength/core/target shapes required")
    x, gamma, sigma, target = (np.array(v, dtype=np.float64, copy=True) for v in (x, gamma, sigma, target))
    if (not all(np.isfinite(v).all() for v in (x, gamma, sigma, target)) or np.any(sigma <= 0)
            or not math.isfinite(z_min) or not math.isfinite(z_max) or z_max <= z_min
            or np.any(x[:, 2] < z_min) or np.any(x[:, 2] > z_max)):
        raise ValueError("finite sources inside the physical slab and positive cores required")
    ru, rj, gu, gj = (np.zeros(len(target)) for _ in range(4))
    diagnostics = {"shells": int(shells), "families": [], "runtime_admissible": False,
                   "norms": "source strengths L1; AABB target-source L1 extent; squared distance Euclidean",
                   "assumptions": "IEEE754 binary64 basic operations, round-to-nearest and gradual underflow"}
    if not len(x) or not len(target) or not np.any(gamma):
        return TailRemainder(ru, rj, gu, gj, ru.copy(), rj.copy(), diagnostics)
    origin = .5*x.min(axis=0)+.5*x.max(axis=0)
    origin[2] = z_min
    source_local, target_local = sub(point(x), point(origin)), sub(point(target), point(origin))
    weight = _axis_sum(point(np.abs(gamma)))
    weight = Interval(np.maximum(weight.lower, 0.), weight.upper)
    wcol = Interval(weight.lower[:, None], weight.upper[:, None])
    s0 = pairwise_sum(weight)
    period = mul(point(2.), sub(point(z_max), point(z_min)))
    period_shell = mul(period, point(float(shells)))
    cu = div_positive(point(51.), mul(point(48.), mul(pi_interval(), period)))
    cj = div_positive(point(51.), mul(point(16.), mul(pi_interval(), period)))
    target_squared = _axis_sum(_squared(target_local))
    for odd in (False, True):
        source = Interval(source_local.lower.copy(), source_local.upper.copy())
        if odd:
            source.lower[:, 2], source.upper[:, 2] = -source_local.upper[:, 2], -source_local.lower[:, 2]
        s1 = pairwise_sum(mul(wcol, source))
        s2 = pairwise_sum(mul(weight, _axis_sum(_squared(source))))
        squared_moment = add(sub(mul(s0, target_squared),
                                 mul(point(2.), _axis_sum(mul(target_local, s1)))), s2)
        if np.any(squared_moment.upper < 0):
            raise FloatingPointError("squared-distance moment lost enclosure")
        squared_moment = Interval(np.maximum(squared_moment.lower, 0.), squared_moment.upper)
        source_box = Interval(source.lower.min(axis=0), source.upper.max(axis=0))
        displacement = sub(target_local, source_box)
        axis_maximum = np.maximum(np.abs(displacement.lower), np.abs(displacement.upper))
        extent = _axis_sum(point(axis_maximum)).upper
        gap = sub(period_shell, point(extent))
        if np.any(gap.lower <= 0):
            raise ValueError("a*K must exceed each enclosed source-target L1 extent")
        gap_point = point(gap.lower)
        gap_squared = mul(gap_point, gap_point)
        gap_fourth = mul(gap_squared, gap_squared)
        u = div_positive(mul(cu, mul(point(extent), squared_moment)), gap_fourth).upper
        j = div_positive(mul(cj, squared_moment), gap_fourth).upper
        gaussian = gaussian_defect_upper(gap_lower=float(gap.lower.min()), sigma_upper=float(sigma.max()),
                                         period_lower=float(period.lower), absolute_strength_upper=float(s0.upper))
        ru, rj = add(point(ru), point(u)).upper, add(point(rj), point(j)).upper
        gu = add(point(gu), point(gaussian["velocity_upper"])).upper
        gj = add(point(gj), point(gaussian["gradient_frobenius_upper"])).upper
        diagnostics["families"].append({"odd": odd, "source_strength": _record(s0),
                                        "first_moment": _record(s1), "second_moment": _record(s2),
                                        "maximum_l1_extent": float(extent.max()),
                                        "minimum_gap": float(gap.lower.min()), "gaussian": gaussian})
    diagnostics["origin"] = origin.tolist()
    return TailRemainder(ru, rj, gu, gj, add(point(ru), point(gu)).upper,
                         add(point(rj), point(gj)).upper, diagnostics)
