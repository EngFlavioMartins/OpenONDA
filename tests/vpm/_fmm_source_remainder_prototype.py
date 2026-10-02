"""Unwired absolute-error certificates for a prospective source expansion.

This is a mathematical feasibility prototype, NOT production admission. In
particular, it does not invent a tolerance from the legacy opening angle or
from the slab image-tail stopping threshold. A caller must supply an absolute
velocity/Jacobian budget and independently certified core, local, and rounding
errors. Keeping old ALL-node decisions precedes this extra admission test.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class FieldBound:
    velocity: float
    gradient: float

    def __add__(self, other):
        return FieldBound(self.velocity + other.velocity, self.gradient + other.gradient)


def _validate_bound(value):
    if any(math.isnan(x) or x < 0 for x in (value.velocity, value.gradient)):
        raise ValueError("error bounds must be nonnegative and not NaN")


def source_derivative_remainder(order, derivative, distance, radius, absolute_moment):
    """Bound the singular Biot--Savart source-truncation error.

    ``distance`` is the MINIMUM target distance to the source expansion centre,
    including rounded-centre/enclosure corrections. ``radius`` encloses every
    source offset from that same centre. ``absolute_moment`` is
    sum_j |Gamma_j| * |source_j - centre|**(order+1), not a signed/net moment.

    For G=1/r, ||D^k G||_F = sqrt(k! (2k-1)!!) / r**(k+1). With n=order+1,
    the leading omitted source contraction is bounded by mu_n*c_n/d**(n+m+1).
    All later absolute moments obey mu_(n+j) <= mu_n*radius**j. The coefficient
    ratio c_(n+1)/c_n decreases for derivative m>=1; its first ratio therefore
    yields a conservative geometric majorant. A ratio >=1 means this bound
    cannot certify convergence (not that the physical series diverges).

    derivative=1 bounds velocity's Euclidean norm; derivative=2 bounds the
    Jacobian's Frobenius norm. Curl/contraction with Gamma contributes no
    extra factor because |Gamma cross a| <= |Gamma| |a|, also columnwise.
    All returned errors include 1/(4*pi). Infinite errors decline admission.
    """
    if not isinstance(order, int) or order < 0 or derivative not in (1, 2):
        raise ValueError("require nonnegative integer order and derivative 1 or 2")
    if not all(math.isfinite(x) for x in (distance, radius, absolute_moment)):
        return math.inf
    if distance <= 0 or radius < 0 or absolute_moment < 0:
        return math.inf
    if radius >= distance:
        return math.inf
    if absolute_moment == 0:
        return 0.0
    n = order + 1
    ratio = math.sqrt((n + derivative + 1) * (2 * (n + derivative) + 1)) / (n + 1)
    tail_ratio = ratio * radius / distance
    if tail_ratio >= 1:
        return math.inf
    k = n + derivative
    # (2k-1)!! = (2k)! / (2**k k!), so k! cancels exactly.
    log_coefficient = 0.5 * (math.lgamma(2 * k + 1) - k * math.log(2)) - math.lgamma(n + 1)
    logarithm = (
        math.log(absolute_moment) + log_coefficient
        - (n + derivative + 1) * math.log(distance)
        - math.log(4 * math.pi) - math.log1p(-tail_ratio)
    )
    try:
        return math.exp(logarithm)
    except OverflowError:
        return math.inf


def certify_extra_source_admission(
    *, legacy_all, order, distance, radius, absolute_moment,
    budget, core_error, local_error, rounding_error,
):
    """Return decision and summed certificate; never replace old ALL nodes.

    Missing regularization/core, target-local, or arithmetic certificates
    deliberately decline the new path. An explicit zero is valid only when
    the corresponding operation is exact (e.g. singular test or point target).
    Budgets are absolute and remain meaningful for zero-net-strength clouds.
    This function does not claim that a selected budget matches old accuracy.
    """
    if legacy_all:
        return "legacy", None
    if any(x is None for x in (budget, core_error, local_error, rounding_error)):
        return "refine", None
    for value in (budget, core_error, local_error, rounding_error):
        _validate_bound(value)
    if not all(math.isfinite(x) for x in (budget.velocity, budget.gradient)):
        raise ValueError("admission requires finite explicit budgets")
    source_error = FieldBound(
        source_derivative_remainder(order, 1, distance, radius, absolute_moment),
        source_derivative_remainder(order, 2, distance, radius, absolute_moment),
    )
    total = source_error + core_error + local_error + rounding_error
    accepted = total.velocity <= budget.velocity and total.gradient <= budget.gradient
    return ("higher_order" if accepted else "refine"), total


def legacy_envelope_budget(theta, absolute_strength, maximum_descendant_distance):
    """A partition-comparable, deliberately tightened monopole error envelope.

    This is an ALTERNATIVE qualification contract, not a claim about measured
    pointwise old error. The existing diameter/distance MAC has theta=.1. A
    concentric spherical node at its boundary therefore has q0=theta/2; its
    worst-case monopole remainder defines reference coefficients B0,u/B0,J.
    Old node COM can be displaced from its AABB centre, so its true source
    extent need not satisfy q<=q0. Using this concentric reference is tighter
    than the full geometry-only legacy worst-case envelope, not a claim that
    the old source enclosure was concentric.

    The denominator must bound the distance from EVERY target in this packet
    to EVERY possible accepted descendant COM. For centre separation R,
    enclosed target radius b and source radius a, R+a+b suffices. Positive
    absolute-strength COM weights keep descendants inside the source ball.
    Thus B0*sum|Gamma|/(R+a+b)**k is no greater than the SUM of the reference
    budgets B0*sum_child|Gamma|/distance_child**k over any descendant partition.
    This prevents obtaining a looser global budget simply by coarsening.
    """
    if not (math.isfinite(theta) and 0 < theta < 1):
        raise ValueError("require finite opening angle in (0,1)")
    if not (math.isfinite(absolute_strength) and absolute_strength >= 0):
        raise ValueError("require finite nonnegative absolute strength")
    if not (math.isfinite(maximum_descendant_distance) and maximum_descendant_distance > 0):
        raise ValueError("require finite positive maximum descendant distance")
    reference_radius = theta / 2
    return FieldBound(*(
        source_derivative_remainder(0, derivative, 1, reference_radius, reference_radius)
        * absolute_strength / maximum_descendant_distance ** (derivative + 1)
        for derivative in (1, 2)
    ))


def regularization_error_bound(absolute_strength, nearest_source_distance, q_error, b_error):
    """Charge certified radial-tail coefficient suprema, source-only cores.

    q_error bounds |q-1/(4pi)|; b_error bounds |3q-rho^3*zeta-3/(4pi)| for
    EVERY actual source/target pair. Obtaining these suprema and verifying
    the entire source ball is beyond all core tails is a separate obligation.
    Common cores alone do not qualify a singular multipole near their cores.
    """
    values = (absolute_strength, nearest_source_distance, q_error, b_error)
    if not all(math.isfinite(x) for x in values) or any(x < 0 for x in values):
        return FieldBound(math.inf, math.inf)
    if nearest_source_distance == 0:
        return FieldBound(math.inf, math.inf)
    velocity = absolute_strength * q_error / nearest_source_distance**2
    gradient = absolute_strength * (math.sqrt(2) * q_error + b_error) / nearest_source_distance**3
    return FieldBound(velocity, gradient)


def singular_source_taylor(position, strength, target, centre, order):
    """Small float64 oracle only; Cartesian potential Taylor with exact derivatives."""
    import numpy as np

    position, strength, target, centre = (
        np.asarray(x, dtype=np.float64) for x in (position, strength, target, centre)
    )
    displacement = target - centre
    distance = float(np.linalg.norm(displacement))
    direction = displacement / distance
    # Each term stores x**powers * r**(-denominator). Differentiate this
    # representation algebraically, then evaluate with normalized coordinates
    # to avoid overflow of cancelling powers at small physical units.
    cache = {(0, 0, 0): {(0, 0, 0, 1): 1.0}}

    def polynomial(alpha):
        if alpha not in cache:
            axis = next(i for i, count in enumerate(alpha) if count)
            lower = list(alpha)
            lower[axis] -= 1
            result = {}
            for term, coefficient in polynomial(tuple(lower)).items():
                powers, denominator = list(term[:3]), term[3]
                if powers[axis]:
                    reduced = powers.copy()
                    reduced[axis] -= 1
                    key = (*reduced, denominator)
                    result[key] = result.get(key, 0.0) + coefficient * powers[axis]
                powers[axis] += 1
                key = (*powers, denominator + 2)
                result[key] = result.get(key, 0.0) - denominator * coefficient
            cache[alpha] = result
        return cache[alpha]

    def derivative(alpha):
        angular = sum(
            coefficient * np.prod(direction ** np.asarray(term[:3]))
            for term, coefficient in polynomial(tuple(alpha)).items()
        )
        return angular / distance ** (sum(alpha) + 1) / (4 * math.pi)

    first = np.zeros((3, 3))  # derivative axis, vector-potential component
    second = np.zeros((3, 3, 3))
    offsets = centre - position
    for a in range(order + 1):
        for b in range(order - a + 1):
            for c in range(order - a - b + 1):
                alpha = np.array([a, b, c])
                moment = (
                    strength * np.prod(offsets ** alpha, axis=1)[:, None]
                ).sum(axis=0) / (math.factorial(a) * math.factorial(b) * math.factorial(c))
                for i in range(3):
                    beta = alpha.copy()
                    beta[i] += 1
                    first[i] += moment * derivative(beta)
                    for j in range(3):
                        gamma = beta.copy()
                        gamma[j] += 1
                        second[i, j] += moment * derivative(gamma)
    velocity = np.array([first[1, 2] - first[2, 1], first[2, 0] - first[0, 2], first[0, 1] - first[1, 0]])
    gradient = np.array([
        second[1, :, 2] - second[2, :, 1],
        second[2, :, 0] - second[0, :, 2],
        second[0, :, 1] - second[1, :, 0],
    ])
    return velocity, gradient
