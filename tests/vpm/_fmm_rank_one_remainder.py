"""UNWIRED rank-one source Taylor and regularization error envelopes.

These are analytic bounds, evaluated as ordinary f64 reference arithmetic,
NOT interval-certified device admission. Native census must continue reporting
runtime_admissible_interactions=0: finite arithmetic, stored moment errors,
target-local truncation and image-series remainder remain separate obligations.

DERIVATION. Fix a target displacement R=d*e and let z(phi)=e+i*t(phi), where
t traverses unit vectors perpendicular to e. In a neighbourhood of R,
    1/|X| = average_phi 1/(z(phi).X).
The integral follows directly from integral 1/(a+i*b*cos(phi)); its geometric
series is the Laplace representation of the Legendre polynomials, NIST DLMF
18.10.5, https://dlmf.nist.gov/18.10.E5 . Hold z fixed while differentiating.
For a REAL source offset h, |z.h|<=|h| and ||z||_2=sqrt(2). Thus the nth
source term of u=Gamma cross grad(1/r)/(4*pi) has norm at most
    sqrt(2)*(n+1)*|Gamma|*|h|**n/(4*pi*d**(n+2));
its full Jacobian has Frobenius norm at most
    2*(n+1)*(n+2)*|Gamma|*|h|**n/(4*pi*d**(n+3)).
There is no exponential-in-n Frobenius-tensor factor: every source derivative
is contracted with the SAME h. For a>=max|h| and mu_n=sum|Gamma|*|h|**n,
mu_(n+k)<=mu_n*a**k. Sum the resulting positive geometric derivatives below.
The bound depends on absolute moments, never the signed/net circulation.
It therefore remains valid if every stored signed moment through p vanishes.
For a target packet use a conservative MINIMUM d to the SOURCE expansion
centre. Rounded centres, inverse-image transform errors and target extent
must be included before evaluating this scalar formula. Require d>a.

CORE DEFECT. Put C=1/(4*pi), T=C-q(r/sigma)>=0 and Z=zeta(r/sigma)/sigma**3.
The regularized-minus-singular radial tensor has eigenvalues -T/r**3 twice
and Z+2*T/r**3 once. Hence |du|<=|Gamma|*T/r**2 and
||dJ||_F<=sqrt(2)*|Gamma|*(2*T/r**3+Z). T decreases in r and increases in
sigma for both supported positive kernels. Gaussian T uses the Mills upper
bound erfc(rho)<=exp(-rho**2)/(sqrt(pi)*rho), clipped by C. Winckelmans uses
an algebraically exact, cancellation-free factorization of 1-q/C.
For variable cores 0<sigma<=sigma_max, density is maximized at
min(sigma_max,r*sqrt(2/3)) for Gaussian and min(sigma_max,2*r/sqrt(3)) for
Winckelmans. Thus the core envelope covers every actual source-only image
pair at distances>=r; common cores do not license dropping this charge.
"""

import math

from tests.vpm._fmm_source_remainder_prototype import FieldBound

_C = 1.0 / (4 * math.pi)


def rank_one_remainder(order, derivative, distance, radius, absolute_moment):
    """Absolute velocity (m=1) or full-Jacobian (m=2) tail after source p."""
    if not isinstance(order, int) or order < 0 or derivative not in (1, 2):
        raise ValueError("require nonnegative integer order and derivative 1 or 2")
    if not all(math.isfinite(x) for x in (distance, radius, absolute_moment)):
        return math.inf
    if distance <= radius or radius < 0 or absolute_moment < 0:
        return math.inf
    if absolute_moment == 0:
        return 0.0
    n, q = order + 1, radius / distance
    inverse = 1 / (1 - q)
    if derivative == 1:
        series = math.sqrt(2) * ((n + 1) * inverse + q * inverse**2)
    else:
        series = 2 * (
            (n + 1) * (n + 2) * inverse
            + (2 * n + 3) * q * inverse**2
            + q * (1 + q) * inverse**3
        )
    logarithm = (
        math.log(absolute_moment) + math.log(series) - math.log(4 * math.pi)
        - (n + derivative + 1) * math.log(distance)
    )
    try:
        return math.nextafter(math.exp(logarithm), math.inf)
    except OverflowError:
        return math.inf


def core_tail_bound(kernel, distance, maximum_core, absolute_strength):
    """Uniform image-kernel singular-defect envelope, including variable cores."""
    if kernel not in ("GAUSSIAN", "WINCKELMANS"):
        raise ValueError("only Gaussian and Winckelmans are certified analytically")
    values = (distance, maximum_core, absolute_strength)
    if (not all(math.isfinite(x) for x in values)
            or distance <= 0 or maximum_core <= 0 or absolute_strength < 0):
        return FieldBound(math.inf, math.inf)
    if absolute_strength == 0:
        return FieldBound(0.0, 0.0)
    try:
        tail, density = _core_terms(kernel, distance, maximum_core)
        velocity = absolute_strength * tail / distance**2
        gradient = math.sqrt(2) * absolute_strength * (2 * tail / distance**3 + density)
    except (OverflowError, ZeroDivisionError):
        return FieldBound(math.inf, math.inf)
    if not all(math.isfinite(x) and x >= 0 for x in (velocity, gradient)):
        return FieldBound(math.inf, math.inf)
    return FieldBound(velocity, gradient)


def _core_terms(kernel, distance, maximum_core):
    rho = distance / maximum_core
    if not math.isfinite(rho) or rho <= 0:
        return math.inf, math.inf
    if kernel == "GAUSSIAN":
        tail = _C * min(1.0, math.exp(-rho * rho) * (2 * rho + 1 / rho) / math.sqrt(math.pi))
        sigma = min(maximum_core, distance * math.sqrt(2 / 3))
        density = math.exp(-(distance / sigma)**2) / (math.pi**1.5 * sigma**3)
    else:
        if rho >= 1:
            t = 1 / rho**2
            root = math.sqrt(1 + t)
            tail = _C * t**2 * (root**2 + root + 2 - 0.5 / (root + 1)) / ((root + 1) * root**5)
        else:
            tail = _C * (1 - rho**3 * (rho**2 + 2.5) / (1 + rho**2)**2.5)
        sigma = min(maximum_core, 2 * distance / math.sqrt(3))
        density = 7.5 * _C * sigma**4 / (distance**2 + sigma**2)**3.5
    return tail, density
