"""UNWIRED f64 Gaussian slip-slab oracle with an analytic image-tail bound.

This is a small-cloud qualification oracle, not a particle-mesh implementation.
It does not call any production induction or image-tail code. Sources in a slab
of width L have period a=2L and two source families: (x,y,z,G) and
(x,y,2*z_min-z,(-Gx,-Gy,Gz)). This is the axial-vector reflection convention.
The zero spanwise mode is retained, including nonzero net axial circulation.

Mathematical context: Saffar Shamshirgar and Tornberg, "The Spectral Ewald
method for singly periodic domains", https://arxiv.org/abs/1611.09538. This
module instead evaluates real-space pairs and derives its own remainder below;
it does not apply the paper's charge-neutrality requirement to vortex velocity.

For K>=1, sum BOTH +k and -k for each family through K. Let d be the
target-minus-untranslated-source vector, D=|d|, C=1/(4*pi), and a*K>D.
The Gaussian vector kernel is odd. Therefore its paired velocity is an integral
of its Jacobian along [a*k*ez-d, a*k*ez+d], of length 2D. With
zeta_sigma(r)=exp(-(r/sigma)^2)/(pi^1.5*sigma^3), the exact radial Jacobian is

  [G cross] { (q/r^3)(I-3*n*n.T) + zeta_sigma*n*n.T }, 0<=q<=C.

Hence ||J||op <= |G|(2*C/r^3+zeta_sigma), and
||J||F <= |G|(sqrt(5)*C/r^3+zeta_sigma). The sqrt(5) follows directly from
||[G cross](I-3*n*n.T)||F^2=2|G|^2+3|G cross n|^2.
On the segment r>=a*k-D. Decreasing positive envelopes give the rigorous
real-arithmetic bounds

  S3 = sum_{k>K}(a*k-D)^-3 <= 1/(2*a*(a*K-D)^2),
  E  = sum_{k>K}zeta_sigma(a*k-D)
     <= erfc((a*K-D)/sigma)/(2*pi*a*sigma^2),
  velocity tail <= sum_families 2*D*|G|*(2*C*S3+E),
  Jacobian tail <= sum_families 2*|G|*(sqrt(5)*C*S3+E).

These are absolute Euclidean/Frobenius bounds, not a difference-of-blocks
heuristic, and survive arbitrarily cancelled strengths. Floating-point kernel
evaluation and summation are NOT enclosed by this analytic truncation bound.
The result reports a separate conservative summation-scale indicator; it is
explicitly not a certified bound on all floating-point evaluation errors.
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy.special import erfc, gammainc


@dataclass(frozen=True)
class PeriodicGaussianResult:
    velocity: np.ndarray
    gradient: np.ndarray
    velocity_tail_bound: np.ndarray
    gradient_tail_bound: np.ndarray
    shells: int
    summation_roundoff_indicator: np.ndarray


def _checked_inputs(position, strength, radius, targets, z_min, z_max):
    x, g, s, t = (
        np.asarray(value, dtype=np.float64) for value in (position, strength, radius, targets)
    )
    if x.ndim != 2 or x.shape[1:] != (3,) or g.shape != x.shape or s.shape != (len(x),):
        raise ValueError("source position/strength/radius shapes must be (N,3)/(N,3)/(N,)")
    if t.ndim != 2 or t.shape[1:] != (3,):
        raise ValueError("target shape must be (M,3)")
    if not all(np.isfinite(value).all() for value in (x, g, s, t)) or np.any(s <= 0):
        raise ValueError("finite inputs and positive Gaussian radii are required")
    if not math.isfinite(z_min) or not math.isfinite(z_max) or z_max <= z_min:
        raise ValueError("finite ordered slab planes are required")
    if np.any(x[:, 2] < z_min) or np.any(x[:, 2] > z_max):
        raise ValueError("physical source is outside the slab")
    reflected_x, reflected_g = x.copy(), g.copy()
    reflected_x[:, 2] = 2 * z_min - x[:, 2]
    reflected_g[:, :2] *= -1
    return (
        np.concatenate((x, reflected_x)),
        np.concatenate((g, reflected_g)),
        np.tile(s, 2),
        t,
    )


def gaussian_pairs(displacement, strength, radius):
    """Independent normalized Gaussian velocity and full row-major Jacobian.

    The incomplete gamma expression avoids erf cancellation in q. A convergent
    18-term integrated-density series evaluates the small-core radial factors;
    at rho<=1/2 its first omitted dimensionless terms are below 2e-28.
    Coincident velocity is zero and its finite nonzero self-Jacobian is included.
    """
    r = np.asarray(displacement, dtype=np.float64)
    g = np.broadcast_to(np.asarray(strength, dtype=np.float64), r.shape)
    sigma = np.broadcast_to(np.asarray(radius, dtype=np.float64), r.shape[:-1])
    distance = np.linalg.norm(r, axis=-1)
    rho2 = (distance / sigma) ** 2
    small = rho2 <= 0.25
    first = np.zeros_like(distance)
    second = np.zeros_like(distance)
    large = ~small
    q = gammainc(1.5, rho2[large]) / (4 * np.pi)
    first[large] = q / distance[large] ** 3
    second[large] = 3 * q / distance[large] ** 5 - np.exp(-rho2[large]) / (
        np.pi**1.5 * sigma[large] ** 3 * distance[large] ** 2
    )
    a, b = np.zeros(np.count_nonzero(small)), np.zeros(np.count_nonzero(small))
    for n in reversed(range(18)):
        a = -rho2[small] * a + 1 / (math.factorial(n) * (2 * n + 3))
        b = -rho2[small] * b + 2 / (math.factorial(n) * (2 * n + 5))
    first[small] = a / (np.pi**1.5 * sigma[small] ** 3)
    second[small] = b / (np.pi**1.5 * sigma[small] ** 5)
    skew = np.zeros((*r.shape[:-1], 3, 3))
    skew[..., 0, 1], skew[..., 0, 2] = -g[..., 2], g[..., 1]
    skew[..., 1, 0], skew[..., 1, 2] = g[..., 2], -g[..., 0]
    skew[..., 2, 0], skew[..., 2, 1] = -g[..., 1], g[..., 0]
    velocity = np.cross(g, r) * first[..., None]
    gradient = (
        skew * first[..., None, None]
        + np.cross(r, g)[..., :, None] * r[..., None, :] * second[..., None, None]
    )
    if not np.isfinite(velocity).all() or not np.isfinite(gradient).all():
        raise FloatingPointError("oracle scale overflow: rescale this qualification problem")
    return velocity, gradient


def image_remainder_bounds(position, strength, radius, targets, *, z_min, z_max, shells):
    """Bound the omitted complete +/- shells; no numerical image evaluation."""
    if not isinstance(shells, int) or shells < 1:
        raise ValueError("shells must be a positive integer")
    x, g, sigma, t = _checked_inputs(position, strength, radius, targets, z_min, z_max)
    return _family_bounds(x, g, sigma, t, 2 * (z_max - z_min), shells)


def _family_bounds(x, g, sigma, targets, period, shells):
    with np.errstate(over="ignore", invalid="ignore"):
        extent = np.linalg.norm(targets[:, None, :] - x[None, :, :], axis=-1)
        magnitude = np.linalg.norm(g, axis=-1)[None, :]
    if not np.isfinite(extent).all() or not np.isfinite(magnitude).all():
        raise FloatingPointError("oracle tail scale overflow: rescale this qualification problem")
    gap = period * shells - extent
    valid = gap > 0
    safe_gap = np.where(valid, gap, 1)
    s3 = 1 / (2 * period * safe_gap**2)
    exponential = erfc(safe_gap / sigma[None, :]) / (2 * np.pi * period * sigma[None, :] ** 2)
    velocity = 2 * extent * magnitude * (s3 / (2 * np.pi) + exponential)
    gradient = 2 * magnitude * (math.sqrt(5) * s3 / (4 * np.pi) + exponential)
    velocity = np.where(valid | (magnitude == 0), velocity, np.inf)
    gradient = np.where(valid | (magnitude == 0), gradient, np.inf)
    if np.isnan(velocity).any() or np.isnan(gradient).any():
        raise FloatingPointError("oracle tail arithmetic overflow: rescale this qualification problem")
    return velocity.sum(axis=1), gradient.sum(axis=1)


def slip_periodic_gaussian(
    position,
    strength,
    radius,
    targets,
    *,
    z_min,
    z_max,
    velocity_tolerance=1e-7,
    gradient_tolerance=1e-7,
    shells=None,
    max_shells=131072,
    chunk_shells=128,
    target_radius=None,
    image_only=False,
):
    """Compute the controlled infinite-image field of a small Gaussian cloud.

    Default primary and all images use source-only radii, suitable for generic
    targets. With target_radius, only the physical primary uses the production
    arithmetic mean of target/source radii. Images always use source radii.
    image_only omits the physical primary, not the k=0 odd reflection.
    Fixed shells is useful for validating the certificate; otherwise doubling
    chooses a count whose analytic truncation bounds meet both tolerances.
    The explicit budget prevents accidental use as a native-particle solver.
    """
    if not all(math.isfinite(v) and v > 0 for v in (velocity_tolerance, gradient_tolerance)):
        raise ValueError("positive finite absolute tolerances are required")
    if (
        not isinstance(max_shells, int)
        or max_shells < 1
        or not isinstance(chunk_shells, int)
        or chunk_shells < 1
    ):
        raise ValueError("positive integer work budgets are required")
    x, g, sigma, t = _checked_inputs(position, strength, radius, targets, z_min, z_max)
    original_count = len(x) // 2
    period = 2 * (z_max - z_min)
    count = 1 if shells is None else shells
    if not isinstance(count, int) or count < 1 or count > max_shells:
        raise ValueError("shell count is outside the qualification budget")
    vb, gb = _family_bounds(x, g, sigma, t, period, count)
    if shells is None:
        while np.any(vb > velocity_tolerance) or np.any(gb > gradient_tolerance):
            if count >= max_shells:
                raise RuntimeError("analytic tail certificate exceeds the explicit shell budget")
            count = min(2 * count, max_shells)
            vb, gb = _family_bounds(x, g, sigma, t, period, count)
    primary_sigma = np.broadcast_to(sigma[None, :], (len(t), len(x))).copy()
    if target_radius is not None:
        target_sigma = np.asarray(target_radius, dtype=np.float64)
        if (
            target_sigma.shape != (len(t),)
            or not np.isfinite(target_sigma).all()
            or np.any(target_sigma <= 0)
        ):
            raise ValueError("target_radius must contain one positive finite radius per target")
        primary_sigma[:, :original_count] = (
            target_sigma[:, None] + sigma[None, :original_count]
        ) / 2
    v, j = gaussian_pairs(t[:, None, :] - x[None, :, :], g[None, :, :], primary_sigma)
    if image_only:
        v[:, :original_count] = 0
        j[:, :original_count] = 0
    velocity, gradient = v.sum(axis=1), j.sum(axis=1)
    absolute_sum = np.stack(
        (np.linalg.norm(v, axis=-1).sum(axis=1), np.linalg.norm(j, axis=(-2, -1)).sum(axis=1)),
        axis=1,
    )
    vc, jc = np.zeros_like(velocity), np.zeros_like(gradient)
    for first in range(1, count + 1, chunk_shells):
        k = np.arange(first, min(count + 1, first + chunk_shells))
        shift = np.zeros((2, len(k), 3))
        shift[0, :, 2], shift[1, :, 2] = period * k, -period * k
        displacement = (
            t[:, None, None, None, :] - x[None, :, None, None, :] - shift[None, None, :, :, :]
        )
        v, j = gaussian_pairs(displacement, g[None, :, None, None, :], sigma[None, :, None, None])
        absolute_sum[:, 0] += np.linalg.norm(v, axis=-1).sum(axis=(1, 2, 3))
        absolute_sum[:, 1] += np.linalg.norm(j, axis=(-2, -1)).sum(axis=(1, 2, 3))
        # Pair +/- first; compensate inter-chunk summation separately.
        dv, dj = v.sum(axis=2).sum(axis=(1, 2)) - vc, j.sum(axis=2).sum(axis=(1, 2)) - jc
        nv, nj = velocity + dv, gradient + dj
        vc, jc = (nv - velocity) - dv, (nj - gradient) - dj
        velocity, gradient = nv, nj
    terms = max(1, len(x) * (2 * count + 1))
    eps_terms = terms * np.finfo(np.float64).eps
    indicator = absolute_sum * (eps_terms / (1 - eps_terms))
    return PeriodicGaussianResult(velocity, gradient, vb, gb, count, indicator)
