"""Unwired Gaussian broadening identity and absolute correction-tail bounds.

No production backend, gridding, FFT, or physical core change is implemented.
This module depends only on NumPy and the standard library. Floating point
evaluations of the analytic bounds are NOT interval-arithmetic error_bounds.

Repository convention (kernels/gaussian.py) is
    Z_s(r) = exp(-(r/s)**2)/(pi**1.5*s**3),
    q(r/s) = [erf(r/s)-2(r/s)exp(-(r/s)**2)/sqrt(pi)]/(4*pi).
Put A_s=q/r**3, B_s=3q/r**5-Z_s/r**2, C_G v=Gamma cross v.
Then u_s=A_s C_G r and J_s=C_G(A_s I-B_s r r^T), including r=0.
The full Jacobian is du_i/dx_j, NOT its symmetric or trace-free part.

For 0<sigma<=tau, changing variables in q'=rho**2 exp(-rho**2)/pi**1.5
gives the cancellation-free identities
    dA=pi**-1.5 integral_[1/tau,1/sigma] s**2 exp(-r*r*s*s) ds,
    dB=2*pi**-1.5 integral_[1/tau,1/sigma] s**4 exp(-r*r*s*s) ds.
Thus K_sigma=K_tau+(K_sigma-K_tau) EXACTLY, also for its full Jacobian.
These integrals define the independent numerical reference below. Gaussian
convolution gives tau**2=sigma**2+eta**2 and Fourier multiplier
exp(-sigma**2 |k|**2/4); eta is an algorithmic broadening, not a new core.

PROOF OF UNIFORM TAIL ENVELOPES. Write T_s=1/(4*pi)-q(r/s)>=0.
dA>=0, dB>=0, dA<=T_tau/r**3. The radial tensor dA I-dB rr^T
has tangential eigenvalue a=dA and radial eigenvalue b=dZ-2a,
where dZ=Z_sigma-Z_tau=3a-r*r*dB. Consequently b<=a and
b>=-Z_tau-2a. Its operator norm is <=2T_tau/r**3+Z_tau.
||C_G||_2=|Gamma| and ||C_G||_F=sqrt(2)|Gamma| give
    |du| <= |Gamma| T_tau/r**2,
    ||dJ||_F <= sqrt(2)|Gamma| (2T_tau/r**3+Z_tau).
All terms on the right decrease with r>0, so substitute any r_c>0 to
bound ALL omitted pairs r>=r_c. Sum |Gamma_j| times per-pair envelopes;
net circulation cancellation never licenses omission. These bounds include
1/(4*pi), unlike bounds for a normalized enclosed-circulation fraction.

The Gaussian-minus-SINGULAR defect has eigenvalues a=-T_sigma/r**3 and
b=Z_sigma+2T_sigma/r**3. The identical envelope with tau=sigma follows.
No singular self field should be formed: at r=0 the finite Gaussian J is
C_G/(3*pi**1.5*sigma**3); the correction must restore this finite self J.

Repository conventions that a future split MUST preserve:
* physical particle pairs use sigma_ij=(sigma_i+sigma_j)/2, not an RMS;
* source queries and slab images use sigma_j only, even at particle targets;
* the physical self-J term is included (velocity self term is zero);
* reflected axial strengths are (-Gamma_x,-Gamma_y,+Gamma_z).
An FFT common tau can represent the broad field, but each local correction
must use the appropriate actual pair sigma. Variable cores cannot silently
be replaced by an average. Periodicity, mesh/quadrature, Fourier truncation,
roundoff and image-shell errors are separate, unproved error budgets here.

The erf series used to verify normalization is DLMF 7.6.1:
https://dlmf.nist.gov/7.6.E1 . Everything else above is a direct derivation.
"""

from dataclasses import dataclass
import math

import numpy as np

_PI15 = math.pi**-1.5
_FOUR_PI = 4.0 * math.pi
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(40)


@dataclass(frozen=True)
class FieldBound:
    """Absolute velocity Euclidean and full-Jacobian Frobenius envelopes."""

    velocity: float
    gradient: float


def _positive_finite(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _strength_norm(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("absolute_strength must be nonnegative and finite")
    return value


def gaussian_tail(distance, sigma):
    """T_sigma without subtracting two nearly equal O(1) quantities."""
    sigma = _positive_finite(sigma, "sigma")
    distance = float(distance)
    if not math.isfinite(distance) or distance < 0:
        raise ValueError("distance must be nonnegative and finite")
    rho = distance / sigma
    return (math.erfc(rho) + 2.0 / math.sqrt(math.pi) * rho * math.exp(-rho * rho)) / _FOUR_PI


def gaussian_density(distance, sigma):
    sigma = _positive_finite(sigma, "sigma")
    return _PI15 / sigma**3 * math.exp(-((distance / sigma) ** 2))


def singular_defect_bound(cutoff, sigma, absolute_strength=1.0):
    """Bound Gaussian-minus-singular u/J for every distance >= cutoff."""
    cutoff = _positive_finite(cutoff, "cutoff")
    absolute_strength = _strength_norm(absolute_strength)
    tail = gaussian_tail(cutoff, sigma)
    density = gaussian_density(cutoff, sigma)
    return FieldBound(
        absolute_strength * tail / cutoff**2,
        math.sqrt(2.0) * absolute_strength * (2.0 * tail / cutoff**3 + density),
    )


def broadening_tail_bound(cutoff, sigma, tau, absolute_strength=1.0):
    """Uniform omitted LOCAL-correction bound; excludes all mesh/FFT errors."""
    sigma = _positive_finite(sigma, "sigma")
    tau = _positive_finite(tau, "tau")
    cutoff = _positive_finite(cutoff, "cutoff")
    absolute_strength = _strength_norm(absolute_strength)
    if tau < sigma:
        raise ValueError("tau must be >= the actual pair sigma")
    if tau == sigma:
        return FieldBound(0.0, 0.0)
    return singular_defect_bound(cutoff, tau, absolute_strength)


def correction_factors(distance, sigma, tau):
    """Independent positive-integral dA,dB reference, not production arithmetic.

    Doubling subintervals avoid a narrow endpoint layer when tau/sigma is
    large; fixed Gaussian quadrature is merely a high-accuracy reference,
    not a rigorous quadrature enclosure. The analytic bounds do not use it.
    """
    sigma = _positive_finite(sigma, "sigma")
    tau = _positive_finite(tau, "tau")
    distance = float(distance)
    if not math.isfinite(distance) or distance < 0 or tau < sigma:
        raise ValueError("require distance >= 0 and tau >= sigma")
    if sigma == tau:
        return 0.0, 0.0
    if distance == 0.0:
        # expm1 avoids cancellation for almost identical broadening radii.
        ratio_log = math.log(tau / sigma)
        return (
            _PI15 / (3.0 * sigma**3) * -math.expm1(-3.0 * ratio_log),
            2.0 * _PI15 / (5.0 * sigma**5) * -math.expm1(-5.0 * ratio_log),
        )
    lower, end = 1.0 / tau, 1.0 / sigma
    a = b = 0.0
    while lower < end:
        upper = min(2.0 * lower, end)
        nodes = lower + (upper - lower) * (_NODES + 1.0) / 2.0
        weights = _WEIGHTS * (upper - lower) / 2.0
        exponent = np.exp(-((distance * nodes) ** 2))
        a += float(weights @ (nodes**2 * exponent))
        b += float(weights @ (nodes**4 * exponent))
        lower = upper
    return _PI15 * a, 2.0 * _PI15 * b


def gaussian_factors(distance, sigma):
    """Finite Gaussian A,B using series at the origin and erf elsewhere."""
    sigma = _positive_finite(sigma, "sigma")
    distance = float(distance)
    if not math.isfinite(distance) or distance < 0:
        raise ValueError("distance must be nonnegative and finite")
    rho = distance / sigma
    if rho < 1.0:
        coefficients = np.array(
            [(-1.0) ** n / (math.factorial(n) * (2 * n + 3)) for n in range(24)]
        )
        derivative = np.array([-2.0 * n * coefficients[n] for n in range(1, 24)])
        return (
            _PI15 / sigma**3 * np.polynomial.polynomial.polyval(rho * rho, coefficients),
            _PI15 / sigma**5 * np.polynomial.polynomial.polyval(rho * rho, derivative),
        )
    q = 1.0 / _FOUR_PI - gaussian_tail(distance, sigma)
    return q / distance**3, 3.0 * q / distance**5 - gaussian_density(distance, sigma) / distance**2


def _fields(displacement, strength, factors):
    displacement = np.asarray(displacement, dtype=float)
    strength = np.asarray(strength, dtype=float)
    if displacement.shape != (3,) or strength.shape != (3,):
        raise ValueError("expect one 3-vector displacement and strength")
    if not np.all(np.isfinite(displacement)) or not np.all(np.isfinite(strength)):
        raise ValueError("vectors must be finite")
    x, y, z = strength
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    a, b = factors
    return (
        a * (cross @ displacement),
        cross @ (a * np.eye(3) - b * np.outer(displacement, displacement)),
    )


def gaussian_fields(displacement, strength, sigma):
    return _fields(displacement, strength, gaussian_factors(np.linalg.norm(displacement), sigma))


def correction_fields(displacement, strength, sigma, tau):
    return _fields(
        displacement,
        strength,
        correction_factors(np.linalg.norm(displacement), sigma, tau),
    )


def singular_fields(displacement, strength):
    radius = _positive_finite(np.linalg.norm(displacement), "singular distance")
    return _fields(
        displacement, strength, (1.0 / (_FOUR_PI * radius**3), 3.0 / (_FOUR_PI * radius**5))
    )
