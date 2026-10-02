"""UNWIRED moment-aware remainder for Gaussian +/- slip-slab image shells.

This is a mathematical qualification helper, not a production stopping rule.
It leaves the existing two-consecutive-block gate untouched.

Let F_G(d)=C G cross d/|d|^3, C=1/(4*pi), a=2*(zmax-zmin), e=ez,
A=I-3ee^T. For one source family, d=t-x, R=ak and D>=|d|<aK:

 F_G(d+Re)+F_G(d-Re) = 2C/R^3 [G cross] A d + R_u,
 J_G(d+Re)+J_G(d-Re) = 2C/R^3 [G cross] A   + R_J.

The norm of D^4(1/r) is sqrt(2520)/r^5. To verify the constant, rotate r
onto z: the nonzero tensor entries are zzzz=24, permutations of xxzz/yyzz
=-12 (12 entries), xxxx/yyyy=9, and permutations of xxyy=3 (6 entries).
Their squared sum is2520. Taylor's two cubic velocity remainders sum to
1/3 of this derivative bound, and the two quadratic J remainders to1.

Since sum_{k>K}(ak-D)^-5 <= 1/[4a(aK-D)^4], the infinite remainder bounds
are C*sqrt(2520)/(12a) * sum|G|D_j^3/(aK-Dmax)^4 for velocity, and
C*sqrt(2520)/(4a) * sum|G|D_j^2/(aK-Dmax)^4 for J Frobenius. We compute
sum|G|D_j^2 exactly in real arithmetic from weighted moments and bound
sum|G|D_j^3 <= Dmax*sum|G|D_j^2. The leading term is summed analytically
using zeta(3,K+1), retaining net strength AND the first spatial moment.
Only the remainder uses absolute strengths, so cancellation cannot hide it.

The Gaussian-minus-singular remainder is charged separately, using
erfc(r/sigma)<=exp(-(r/sigma)^2)/(sqrt(pi)*r/sigma) and an integral bound
on the positive exponential tail. Source-only radii are required. It is
uniform over sigma<=sigma_max when aK-Dmax>=sqrt(3/2)*sigma_max.

All inequalities above concern real arithmetic. Padded ordinary-f64
evaluation below is NOT an interval, FFT, FMM or summation certificate.
"""

from dataclasses import dataclass
import math
from numbers import Integral

import numpy as np
from scipy.special import zeta


@dataclass(frozen=True)
class MomentTailBound:
    leading_velocity: np.ndarray
    leading_gradient: np.ndarray
    singular_velocity_remainder: np.ndarray
    singular_gradient_remainder: np.ndarray
    gaussian_velocity_defect: np.ndarray
    gaussian_gradient_defect: np.ndarray
    velocity_bound: np.ndarray
    gradient_bound: np.ndarray
    diagnostics: dict


def _cap(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _sum_columns(values):
    return np.array([math.fsum(values[:, axis].tolist()) for axis in range(values.shape[1])])


def _gaussian_defect(gap, sigma, period, absolute_strength):
    if np.any(gap < math.sqrt(1.5)*sigma):
        raise ValueError("Gaussian far-tail density monotonicity not admitted")
    # Sum exp(-((ak-D)/sigma)^2) <= sigma² exp(-(gap/sigma)^2)/(2a*gap).
    logarithm = -(gap/sigma)**2 + 2*math.log(sigma)-np.log(2*period*gap)
    cu = (sigma/gap**3+2/(sigma*gap))/(4*math.pi**1.5)
    cj = (math.sqrt(5)*(sigma/gap**4+2/(sigma*gap**2))/(4*math.pi**1.5)
          +1/(math.pi**1.5*sigma**3))
    results = []
    for coefficient in (cu, cj):
        log_bound = math.log(2*absolute_strength)+np.log(coefficient)+logarithm
        # A positive analytic upper floor, not a reported zero from underflow.
        results.append(np.exp(np.maximum(log_bound, math.log(1e-300))))
    return tuple(results)


def moment_tail_bound(position, strength, radius, targets, *, z_min, z_max, shells,
                      max_sources=400_000, max_targets=400_000):
    """O(N+M) absolute infinite-image remainder after the complete +/-K sum."""
    shells = _cap(shells, "shells")
    max_sources, max_targets = _cap(max_sources, "max_sources"), _cap(max_targets, "max_targets")
    x, gamma, sigma, target = (np.asarray(value, dtype=np.float64)
                               for value in (position, strength, radius, targets))
    if (x.ndim != 2 or x.shape[1:] != (3,) or gamma.shape != x.shape
            or sigma.shape != (len(x),) or target.ndim != 2 or target.shape[1:] != (3,)
            or len(x) > max_sources or len(target) > max_targets):
        raise ValueError("bounded source/strength/core/target shapes required")
    if (not all(np.isfinite(value).all() for value in (x, gamma, sigma, target))
            or np.any(sigma <= 0) or not math.isfinite(z_min) or not math.isfinite(z_max)
            or z_max <= z_min or np.any(x[:, 2] < z_min) or np.any(x[:, 2] > z_max)):
        raise ValueError("finite source fields inside ordered physical slab required")
    period = 2*(z_max-z_min)
    if not math.isfinite(period) or not math.isfinite(period*shells):
        raise ValueError("finite image-period scale required")
    u, j = np.zeros((len(target), 3)), np.zeros((len(target), 3, 3))
    ru, rj, gu, gj = (np.zeros(len(target)) for _ in range(4))
    diagnostics = {"shells": shells, "source_count": len(x), "target_count": len(target),
                   "period": period, "runtime_admissible": False,
                   "arithmetic": "padded f64 evaluation of real-arithmetic inequalities, not interval certification",
                   "leading_zeta3": float(zeta(3, shells+1)), "families": []}
    if not len(x) or not len(target) or not np.any(gamma):
        return MomentTailBound(u, j, ru, rj, gu, gj, ru.copy(), rj.copy(), diagnostics)
    magnitude = np.linalg.norm(gamma, axis=1)
    absolute_strength = math.fsum(magnitude.tolist())
    if not math.isfinite(absolute_strength):
        raise FloatingPointError("absolute source strength overflow")
    eps = np.finfo(float).eps
    padded_strength = np.nextafter(absolute_strength*(1+128*eps), np.inf)
    sigma_max = float(sigma.max())
    original_bounds = x.min(axis=0), x.max(axis=0)
    origin = .5*original_bounds[0]+.5*original_bounds[1]
    origin[2] = z_min
    total_strength = np.array([0., 0., 2*math.fsum(gamma[:, 2].tolist())])
    first_moments = []
    a_diagonal = np.array([1., 1., -2.])
    c = 1/(4*math.pi)
    coefficient_u, coefficient_j = c*math.sqrt(2520)/(12*period), c*math.sqrt(2520)/(4*period)
    for odd in (False, True):
        source, vectors = x.copy(), gamma.copy()
        if odd:
            source[:, 2] = 2*z_min-source[:, 2]
            vectors[:, :2] *= -1
        if not np.isfinite(source).all():
            raise FloatingPointError("reflected source overflow")
        local_source, local_target = source-origin, target-origin
        first_moments.append(_sum_columns(np.cross(vectors, local_source*a_diagonal)))
        lower, upper = local_source.min(axis=0), local_source.max(axis=0)
        coordinate_scale = max(float(np.abs(source).max()), float(np.abs(target).max()), 1.)
        padding = 128*eps*coordinate_scale
        axis_distance = np.maximum(np.abs(local_target-lower), np.abs(local_target-upper))+padding
        extent = np.nextafter(np.linalg.norm(axis_distance, axis=1), np.inf)
        gap = period*shells-extent
        if np.any(gap <= 0):
            raise ValueError("paired moment tail requires a*K greater than every source-target extent")
        centre = _sum_columns(magnitude[:, None]*local_source)/absolute_strength
        variance = math.fsum((magnitude*np.sum((local_source-centre)**2, axis=1)).tolist())/absolute_strength
        rms = np.sqrt(np.sum((local_target-centre)**2, axis=1)+variance)+padding
        second_absolute_moment = padded_strength*np.nextafter(rms, np.inf)**2
        ru += coefficient_u*extent*second_absolute_moment/gap**4
        rj += coefficient_j*second_absolute_moment/gap**4
        defect_u, defect_j = _gaussian_defect(gap, sigma_max, period, padded_strength)
        gu += defect_u
        gj += defect_j
        diagnostics["families"].append({"odd": odd, "maximum_extent": float(extent.max()),
                                        "minimum_gap": float(gap.min()), "absolute_strength": absolute_strength})
    moment = np.array([math.fsum(value[axis] for value in first_moments) for axis in range(3)])
    leading_scale = 2*c*float(zeta(3, shells+1))/period**3
    u = leading_scale*(np.cross(total_strength, (target-origin)*a_diagonal)-moment)
    skew = np.array([[0., -total_strength[2], total_strength[1]],
                     [total_strength[2], 0., -total_strength[0]],
                     [-total_strength[1], total_strength[0], 0.]])
    j[:] = leading_scale*skew*a_diagonal[None, :]
    velocity_bound = np.linalg.norm(u, axis=1)+ru+gu
    gradient_bound = np.linalg.norm(j, axis=(1, 2))+rj+gj
    if not all(np.isfinite(value).all() for value in (u, j, ru, rj, gu, gj, velocity_bound, gradient_bound)):
        raise FloatingPointError("moment tail arithmetic overflow")
    diagnostics.update(moment_origin=origin.tolist(), two_family_net_strength=total_strength.tolist(),
                       two_family_cross_first_moment=moment.tolist(), source_core_max=sigma_max)
    return MomentTailBound(u, j, ru, rj, gu, gj, velocity_bound, gradient_bound, diagnostics)
