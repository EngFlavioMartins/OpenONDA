"""UNWIRED continuous-Fourier reference for a smooth Gaussian slip operator.

This qualifies mathematics, NOT an FFT implementation or its padding error.
Fourier convention: fhat(k)=integral f(x)exp(-ik.x)dx; inverse /(2*pi)^d.
The z Fourier series has period P=2L and normalization 1/P. For mu=|kz|>0,
the free-xy Green function of (-Delta_xy+mu^2) is K0(mu*r)/(2*pi).
For mu=0 use log(R/r)/(2*pi): R is a harmless gauge before truncation.

Truncating these Green functions radially at R gives
  mu>0: [1-mu*R*K1(mu*R)*J0(s*R)+s*R*K0(mu*R)*J1(s*R)]/(s^2+mu^2),
  mu=0: [1-J0(s*R)]/s^2, with s=0 limit R^2/4.
These follow by the Bessel Wronskian integration identity. A constant cannot
be subtracted from the mu>0 Green function: its z derivative is not gauge.
Gaussian broadening multiplies this transform by exp(-tau^2*(s^2+mu^2)/4).

Background: Vico, Greengard, Ferrando, https://arxiv.org/abs/1604.03155;
Saffar Shamshirgar, Tornberg, https://arxiv.org/abs/1611.09538. Gaussian sources
are not compact: truncating the unsmoothed Green kernel then smoothing is NOT
exact, even when every unsmoothed source-target displacement lies below R.
The separate cutoff bound below controls that difference for rho<R.

For H_tau(w)=exp(-|w|^2/tau^2)/(pi*tau^2), let b=R-rho>0 and
  M_n=integral_{|w|>=b}|w|^n H_tau(w)dw
     =tau^n Gamma(1+n/2,b^2/tau^2).
At mu>0, |G_mu(y)|<=K0(mu*R)/(2*pi) for |y|>=R. At mu=0,
|log(R/|y|)|/(2*pi)<=|w|/(2*pi*R), since y=x-w and |x|=rho<R.
Differentiate H, not the discontinuously truncated G, to bound the omitted
convolution: ||grad H||<=2|w|H/tau^2 and
||Hess H||F<=(4|w|^2/tau^4+2*sqrt(2)/tau^2)H.
Add z derivatives via |kz| and multiply exp(-tau^2*mu^2/4).

The omitted *untruncated* z modes are bounded separately by absolute Fourier
integrals. With alpha=tau^2/4, c=2*pi*sqrt(alpha)/P, total absolute strength S,
and modes |m|<=M retained, the following conservative bounds suffice:
  |u_tail| <= S exp(-(c*M)^2)/(4*pi^2*alpha),
  |J_tail|F <= S erfc(c*M)/(4*sqrt(pi)*P*alpha*c).
The derivation integrates the monotone single-mode envelopes over m>M.

All bounds concern real-arithmetic truncation. scipy quadrature reports a
numerical error estimate, not a rigorous interval certificate. The optional
xy-periodization bound controls spatial wrap of the continuous smoothed kernel;
it does not control a finite FFT's omitted frequencies or spread/gather errors.
No complete FFT or machine-roundoff guarantee is made.
"""

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
from scipy.integrate import quad
from scipy.special import erfc, gamma, gammaincc, j0, j1, jv, k0, k1

from tests.vpm._slip_periodic_gaussian_oracle import _checked_inputs


def truncated_green_transform(s, mu, cutoff):
    """Radial 2-D transform, including the essential nonzero zero-mode limit.

    Tiny arguments in the modified-Helmholtz numerator use an independent
    radial quadrature to avoid subtracting two nearly equal unit terms. This
    is reference-only handling, not a proposed per-mode production strategy.
    """
    s, mu, cutoff = abs(float(s)), abs(float(mu)), float(cutoff)
    if not all(math.isfinite(v) for v in (s, mu, cutoff)) or cutoff <= 0:
        raise ValueError("finite wavenumbers and positive cutoff required")
    z = s * cutoff
    if mu == 0:
        if z < 0.05:
            value = 0.0
            for n in reversed(range(12)):
                value = -(z * z / 4) * value + 1 / math.factorial(n + 1) ** 2
            return cutoff**2 * value / 4
        return (1 - j0(z)) / s**2
    a = mu * cutoff
    if max(a, z) < 0.05:
        return quad(lambda r: r * j0(s * r) * k0(mu * r), 0, cutoff, epsabs=2e-13)[0]
    return (1 - a * k1(a) * j0(z) + z * k0(a) * j1(z)) / (s * s + mu * mu)


def compact_padding_is_alias_free(period_xy, pair_extent_xy, cutoff, gaussian_support):
    """Conservative sufficient no-wrap test for TWO COMPACT factors only.

    If G has support radius R and a deliberately truncated Gaussian has support
    radius B, every nonzero xy lattice translation is excluded when each
    P_i-D_i>R+B, where D_i bounds target-minus-source displacements. This can be
    wasteful for anisotropic boxes. A real Gaussian has infinite support, so a
    finite B additionally requires a proved omitted-Gaussian/alias budget.
    Passing this function alone does NOT certify a free-space FFT solution.
    """
    period = np.asarray(period_xy, dtype=float)
    extent = np.asarray(pair_extent_xy, dtype=float)
    if (
        period.shape != (2,)
        or extent.shape != (2,)
        or not np.isfinite(period).all()
        or not np.isfinite(extent).all()
    ):
        raise ValueError("finite two-dimensional periods and displacement bounds required")
    if np.any(period <= 0) or np.any(extent < 0) or cutoff <= 0 or gaussian_support < 0:
        raise ValueError("invalid compact support geometry")
    return bool(np.all(period - extent > cutoff + gaussian_support))


def smooth_cutoff_bounds(rho, mu, cutoff, tau):
    """Potential-gradient/Hessian error from cutting G before Gaussian blur."""
    if (
        not all(math.isfinite(v) for v in (rho, mu, cutoff, tau))
        or not 0 <= rho < cutoff
        or mu < 0
        or tau <= 0
    ):
        raise ValueError("require 0<=rho<cutoff, mu>=0, and tau>0")
    q = ((cutoff - rho) / tau) ** 2
    moment = [tau**n * gamma(1 + n / 2) * gammaincc(1 + n / 2, q) for n in range(4)]
    if mu == 0:
        b0 = moment[1] / (2 * np.pi * cutoff)
        b1 = 2 * moment[2] / (2 * np.pi * cutoff * tau**2)
        b2 = (4 * moment[3] / tau**4 + 2 * np.sqrt(2) * moment[1] / tau**2) / (2 * np.pi * cutoff)
    else:
        amplitude = k0(mu * cutoff) / (2 * np.pi)
        b0 = amplitude * moment[0]
        b1 = amplitude * 2 * moment[1] / tau**2
        b2 = amplitude * (4 * moment[2] / tau**4 + 2 * np.sqrt(2) * moment[0] / tau**2)
    smooth_z = math.exp(-(tau**2) * mu**2 / 4)
    return smooth_z * (b1 + mu * b0), smooth_z * (b2 + 2 * mu * b1 + mu**2 * b0)


def smooth_xy_periodization_bounds(period_xy, pair_extent_xy, mu, cutoff, tau):
    """Bound continuous smoothed-kernel aliases from every nonzero xy period.

    Let p=min(Px,Py), Q=min(Px-Dx,Py-Dy), b=Q-R, B=p-b. Require b>=2*tau.
    In square lattice ring max(|nx|,|ny|)=k there are 8k translations, and every
    Gaussian argument in the truncated convolution has length >=p*k-B.
    With this gap, k*(p*k-B)^j*exp(-(p*k-B)^2/tau^2) decreases for j=0,1,2.
    Sum <= first term + integral from1 toinfinity; those integrals are upper
    incomplete gamma functions. Multiply by ||G_R||L1, then charge derivative
    Gaussian moments and z differentiation as in smooth_cutoff_bounds.

    This gives a finite bound for the actual infinite Gaussian, unlike assuming
    compactness from padding alone. It is deliberately conservative; anisotropic
    Vico kernel precomputation/cropping can admit smaller runtime grids but needs
    its own derivation. The discrete FFT's high-frequency tail is separate.
    """
    period = np.asarray(period_xy, dtype=float)
    extent = np.asarray(pair_extent_xy, dtype=float)
    if (
        period.shape != (2,)
        or extent.shape != (2,)
        or not np.isfinite(period).all()
        or not np.isfinite(extent).all()
    ):
        raise ValueError("finite two-dimensional periods/extents required")
    if (
        np.any(period <= 0)
        or np.any(extent < 0)
        or not all(math.isfinite(v) for v in (mu, cutoff, tau))
        or mu < 0
        or cutoff <= 0
        or tau <= 0
    ):
        raise ValueError("invalid periodization parameters")
    p = float(np.min(period))
    gap = float(np.min(period - extent)) - cutoff
    if gap < 2 * tau:
        raise ValueError("padding requires min(P_i-D_i)-R >= 2*tau for this bound")
    offset = p - gap
    q = (gap / tau) ** 2
    integral = [
        0.5 * tau ** (n + 1) * gamma((n + 1) / 2) * gammaincc((n + 1) / 2, q) for n in range(4)
    ]
    sums = [
        8
        / (np.pi * tau**2)
        * (gap**j * math.exp(-q) + (integral[j + 1] + offset * integral[j]) / p**2)
        for j in range(3)
    ]
    norm = truncated_green_transform(0, mu, cutoff)
    b0, b1 = norm * sums[0], norm * 2 * sums[1] / tau**2
    b2 = norm * (4 * sums[2] / tau**4 + 2 * np.sqrt(2) * sums[0] / tau**2)
    smooth_z = math.exp(-(tau**2) * mu**2 / 4)
    return smooth_z * (b1 + mu * b0), smooth_z * (b2 + 2 * mu * b1 + mu**2 * b0)


def z_mode_tail_bounds(period, tau, modes, absolute_strength):
    if period <= 0 or tau <= 0 or not isinstance(modes, int) or modes < 0 or absolute_strength < 0:
        raise ValueError("invalid Fourier-mode tail parameters")
    alpha = tau**2 / 4
    c = 2 * np.pi * math.sqrt(alpha) / period
    return (
        absolute_strength * math.exp(-((c * modes) ** 2)) / (4 * np.pi**2 * alpha),
        absolute_strength * erfc(c * modes) / (4 * math.sqrt(np.pi) * period * alpha * c),
    )


@lru_cache(maxsize=512)
def _radial_mode(rho, mu, cutoff, tau):
    """Continuous inverse radial Fourier transform and its first derivatives."""

    def spectrum(s):
        return (
            truncated_green_transform(s, mu, cutoff)
            * math.exp(-(tau**2) * (s * s + mu * mu) / 4)
            / (2 * np.pi)
        )

    functions = (
        lambda s: s * j0(s * rho) * spectrum(s),
        lambda s: -(s**2) * j1(s * rho) * spectrum(s),
        lambda s: -0.5 * s**3 * (j0(s * rho) - jv(2, s * rho)) * spectrum(s),
    )
    results = [
        quad(function, 0, np.inf, epsabs=2e-11, epsrel=2e-11, limit=200) for function in functions
    ]
    return np.asarray([r[0] for r in results]), np.asarray([r[1] for r in results])


@dataclass(frozen=True)
class SmoothSpectralResult:
    velocity: np.ndarray
    gradient: np.ndarray
    cutoff_velocity_bound: np.ndarray
    cutoff_gradient_bound: np.ndarray
    mode_velocity_bound: float
    mode_gradient_bound: float
    quadrature_velocity_estimate: np.ndarray
    quadrature_gradient_estimate: np.ndarray


def slip_smooth_spectral_reference(
    position, strength, targets, *, z_min, z_max, tau, cutoff, modes
):
    """Small-cloud continuous-Fourier smooth field, not a particle-mesh solve."""
    if not isinstance(modes, int) or modes < 0 or not math.isfinite(tau) or tau <= 0:
        raise ValueError("positive tau and nonnegative integer modes required")
    x, g, _, t = _checked_inputs(
        position, strength, np.full(len(position), tau), targets, z_min, z_max
    )
    period = 2 * (z_max - z_min)
    velocity, gradient = np.zeros((len(t), 3)), np.zeros((len(t), 3, 3))
    cv, cj, qv, qj = (np.zeros(len(t)) for _ in range(4))
    for target_index, target in enumerate(t):
        for source, strength_vector in zip(x, g, strict=True):
            delta = target - source
            rho = float(np.linalg.norm(delta[:2]))
            if rho >= cutoff:
                raise ValueError("all source-target xy distances must be strictly below cutoff")
            direction = delta[:2] / rho if rho else np.zeros(2)
            magnitude = float(np.linalg.norm(strength_vector))
            for mode in range(modes + 1):
                mu = 2 * np.pi * mode / period
                (value, first, second), error = _radial_mode(rho, mu, cutoff, tau)
                weight = 1 if mode == 0 else 2
                cosine, sine = math.cos(mu * delta[2]), math.sin(mu * delta[2])
                grad = np.r_[cosine * first * direction, -mu * sine * value] * weight
                hessian = np.zeros((3, 3))
                if rho:
                    hessian[:2, :2] = cosine * (
                        second * np.outer(direction, direction)
                        + first / rho * (np.eye(2) - np.outer(direction, direction))
                    )
                    transverse_error = error[2] + error[1] / rho
                else:
                    hessian[:2, :2] = cosine * second * np.eye(2)
                    transverse_error = math.sqrt(2) * error[2]
                hessian[:2, 2] = hessian[2, :2] = -mu * sine * first * direction
                hessian[2, 2] = -(mu**2) * cosine * value
                hessian *= weight
                velocity[target_index] += np.cross(grad, strength_vector) / period
                gradient[target_index] += np.cross(hessian.T, strength_vector).T / period
                vb, jb = smooth_cutoff_bounds(rho, mu, cutoff, tau)
                cv[target_index] += weight * magnitude * vb / period
                cj[target_index] += weight * magnitude * jb / period
                qv[target_index] += weight * magnitude * (error[1] + mu * error[0]) / period
                qj[target_index] += (
                    weight
                    * magnitude
                    * (transverse_error + 2 * mu * error[1] + mu**2 * error[0])
                    / period
                )
    mv, mj = z_mode_tail_bounds(period, tau, modes, float(np.linalg.norm(g, axis=1).sum()))
    return SmoothSpectralResult(velocity, gradient, cv, cj, mv, mj, qv, qj)
