"""
Factory for the Winckelmans-Leonard algebraic regularization kernel set.
"""

from collections.abc import Callable
import math

import numpy as np
import taichi as ti

from openonda.core._numba import cacheable_njit as njit

from ..config.constants import ONE_OVER_FOUR_PI
from .definition import AlgebraicDefectEnvelope, KernelDefinition, SlabImageSplit
from .gaussian import GAUSSIAN_Q_SERIES_COEFFICIENTS
from .gaussian import radial_factors_host as gaussian_radial_factors_host

# Second moment m2 = 3/2, as for the Gaussian; this makes the kernel second order.
SECOND_MOMENT = 1.5


def create_winckelmans_kernels(dtype: object = ti.f32) -> dict[str, Callable]:
    """Create Winckelmans kernel functions with specified precision.

    Args:
        dtype: Taichi data type (ti.f32 or ti.f64)

    Returns:
        Dictionary with q, density, energy and diffusion functions, plus
        finite ``radial_factors_`` for velocity and its Jacobian.
    """

    # Half-integer powers of (ρ²+1) are written as muls + one sqrt instead of the
    # generic ``pow`` (≈ exp+log): (ρ²+1) > 0 always, so x**2.5 = x²·√x etc.  This
    # is algebraically exact (and slightly *more* accurate than pow) and drops two
    # transcendentals to one sqrt on the velocity/gradient hot path.
    @ti.func
    def zeta_(density: ti.template()) -> ti.template():
        base = density * density + 1.0
        return ti.cast(7.5 / (base * base * base * ti.sqrt(base)) * ONE_OVER_FOUR_PI, dtype)

    @ti.func
    def q_(density: ti.template()) -> ti.template():
        d2 = density * density
        base = d2 + 1.0
        return ti.cast(
            density * d2 * (d2 + 2.5) / (base * base * ti.sqrt(base)) * ONE_OVER_FOUR_PI,
            dtype,
        )

    @ti.func
    def radial_factors_(density: ti.template(), sigma: ti.template(), with_gradient: ti.template()):
        """Return exact algebraic blob factors ``q/r³`` and its radial derivative.

        The outputs have units m⁻³ and m⁻⁵; both are finite at ``density=0``.
        """
        d2 = density * density
        base = d2 + 1.0
        root = ti.sqrt(base)
        first = ONE_OVER_FOUR_PI * (d2 + 2.5) / (sigma**3 * base * base * root)
        second = ti.cast(0.0, dtype)
        if ti.static(with_gradient):
            second = ONE_OVER_FOUR_PI * (3.0 * d2 + 10.5) / (sigma**5 * base**3 * root)
        return ti.Vector([first, second])

    @ti.func
    def g_(density: ti.template()) -> ti.template():
        base = density * density + 1.0
        return ti.cast((base + 0.5) / (base * ti.sqrt(base)) * ONE_OVER_FOUR_PI, dtype)

    @ti.func
    def diffusivity_constant_():
        """Core-spreading rate d(sigma^2)/dt = C kinematic_viscosity, with C = 4.

        A self-similar blob w = (vortex_strength/sigma^3) zeta(r/sigma) has <r^2> = m2 sigma^2,
        and the heat equation gives d<r^2>/dt = 6 kinematic_viscosity, so C = 6/m2.  This kernel has
        m2 = 3/2 exactly -- the same as the Gaussian, which is the property that makes
        it a second-order match -- hence C = 4, identical to the Gaussian.

        Was 256/45 = 5.689 (hand-calibrated, no derivation).  Note that core spreading
        is an exact self-similar solution of the heat equation only for the Gaussian;
        for an algebraic kernel it is a model, so a calibrated value is legitimate --
        but it must be labelled as one.  See ``openonda/vpm/REFERENCES.md``.
        """
        return ti.cast(4.0, dtype)

    @ti.func
    def energy_equivalence_constant_():
        return ti.cast(ti.sqrt(6.0), dtype)

    @ti.func
    def volume_correction_constant_():
        """Volume correction constant for radius evolution: dσ/dt = σ * C * div(u).

        For 3D spherical particles, C = 1/3 (Alvarez 2022).
        """
        return ti.cast(1.0 / 3.0, dtype)

    @ti.func
    def angular_impulse_correction_constant_():
        """Angular impulse correction constant for second moment correction.

        For Winckelmans kernel: C = 1.5 (Winckelmans 1993)
        Used in: A = (1/3) Σ x × (x × Γ) - (2/9) C σ² Γ_total
        """
        return ti.cast(SECOND_MOMENT, dtype)

    return {
        "q_": q_,
        "zeta_": zeta_,
        "radial_factors_": radial_factors_,
        "g_": g_,
        "diffusivity_constant_": diffusivity_constant_,
        "energy_equivalence_constant_": energy_equivalence_constant_,
        "volume_correction_constant_": volume_correction_constant_,
        "angular_impulse_correction_constant_": angular_impulse_correction_constant_,
    }


def q_host(rho: np.ndarray | float) -> np.ndarray:
    """Enclosed circulation factor rho^3 (rho^2 + 5/2)/(4 pi (rho^2 + 1)^(5/2))."""
    rho = np.asarray(rho, dtype=np.float64)
    d2 = rho * rho
    base = d2 + 1.0
    return rho * d2 * (d2 + 2.5) / (base * base * np.sqrt(base)) / (4.0 * math.pi)


def zeta_host(rho: np.ndarray | float) -> np.ndarray:
    """Normalized density 15/(8 pi (rho^2 + 1)^(7/2)) in float64."""
    rho = np.asarray(rho, dtype=np.float64)
    base = rho * rho + 1.0
    return 7.5 / (base * base * base * np.sqrt(base)) / (4.0 * math.pi)


@njit(cache=True)
def winckelmans_residual(radius: float, width: float) -> tuple[float, float]:
    """Winckelmans minus Gaussian factors at equal width.

    Below rho = 2 the two factors differ by more than 7 %, so their difference
    is direct. Beyond it both are written through the core defects
    T = 1/(4 pi) - q, which never cancel: with u = rho/sqrt(rho^2 + 1),
    4 pi rho^4 T_W = u^4 (1 + 2u + 3u^2 + 3u^3/2)/(1 + u)^2.
    """
    rho = radius / width
    d2 = rho * rho
    base = d2 + 1.0
    root = math.sqrt(base)
    inverse_four_pi = 1.0 / (4.0 * math.pi)
    if rho < 2.0:
        a_g, b_g = gaussian_radial_factors_host(radius, width)
        a_w = inverse_four_pi * (d2 + 2.5) / (width**3 * base * base * root)
        b_w = inverse_four_pi * (3.0 * d2 + 10.5) / (width**5 * base**3 * root)
        return a_w - a_g, b_w - b_g
    u = rho / root
    defect_w = inverse_four_pi * u**4 * (1.0 + 2.0 * u + 3.0 * u * u + 1.5 * u**3)
    defect_w /= (1.0 + u) ** 2 * d2 * d2
    gradient_w = 3.0 * defect_w + inverse_four_pi * 7.5 * d2 * rho / (base**3 * root)
    e = math.exp(-d2)
    defect_g = (math.erfc(rho) + 2.0 / math.sqrt(math.pi) * rho * e) * inverse_four_pi
    gradient_g = 3.0 * defect_g + d2 * rho * e * math.pi**-1.5
    r3 = radius * radius * radius
    return (defect_g - defect_w) / r3, (gradient_g - gradient_w) / (r3 * radius * radius)


WINCKELMANS_RESIDUAL_CUDA = (
    r"""
__device__ inline void residual(Real r, Real s, Real &a, Real &b) {
    const Real inv4pi = (Real)0.0795774715459476678844;
    const Real pi15 = (Real)0.179587122125166561689;
    const Real two_sqrtpi = (Real)1.12837916709551257390;
    const Real series[18] = {SERIES};
    Real inverse = 1 / s, rho = r * inverse, d2 = rho * rho, base = d2 + 1;
    Real root = rr_sqrt(base), e = rr_exp(-d2);
    Real s3 = inverse * inverse * inverse, s5 = s3 * inverse * inverse;
    if (rho < (Real)2) {
        Real ag, bg;
        if (rho < (Real)1) {
            Real first = 0, second = 0;
            for (int n = 17; n >= 0; n--) {
                first = first * d2 + series[n];
                if (n > 0) second = second * d2 - 2 * n * series[n];
            }
            ag = pi15 * first * s3;
            bg = pi15 * second * s5;
        } else {
            Real q = ((1 - rr_erfc(rho)) - two_sqrtpi * rho * e) * inv4pi;
            Real r3 = r * r * r;
            ag = q / r3;
            bg = (3 * q - d2 * rho * pi15 * e) / (r3 * r * r);
        }
        a = inv4pi * (d2 + (Real)2.5) * s3 / (base * base * root) - ag;
        b = inv4pi * (3 * d2 + (Real)10.5) * s5 / (base * base * base * root) - bg;
        return;
    }
    Real u = rho / root;
    Real tw = inv4pi * u * u * u * u * (1 + 2 * u + 3 * u * u + (Real)1.5 * u * u * u)
              / ((1 + u) * (1 + u) * d2 * d2);
    Real jw = 3 * tw + inv4pi * (Real)7.5 * d2 * rho / (base * base * base * root);
    Real tg = (rr_erfc(rho) + two_sqrtpi * rho * e) * inv4pi;
    Real jg = 3 * tg + d2 * rho * e * pi15;
    Real r3 = r * r * r;
    a = (tg - tw) / r3;
    b = (jg - jw) / (r3 * r * r);
}
"""
).replace("SERIES", ",".join(format(value, ".18e") for value in GAUSSIAN_Q_SERIES_COEFFICIENTS))

# With t = rho^2/(rho^2 + 1) and u = sqrt(t): 4 pi rho^4 T = u^4 p(u)/(1 + u)^2,
# p(u) = 1 + 2u + 3u^2 + 3u^3/2, and 4 pi rho^4 rho^3 zeta = 15 u^7/2. Both rise
# monotonically to 15/8 and 15/2, so 4 pi rho^4 |3T + rho^3 zeta| <= 105/8.
WINCKELMANS_DEFECT = AlgebraicDefectEnvelope(
    velocity=15.0 / 8.0, gradient=105.0 / 8.0, power=4, minimum_ratio=1.0
)

DEFINITION = KernelDefinition(
    name="winckelmans",
    q=q_host,
    zeta=zeta_host,
    device_factory=create_winckelmans_kernels,
    second_moment=SECOND_MOMENT,
    slab_split=SlabImageSplit(
        partner_ratio=1.0,
        residual=winckelmans_residual,
        residual_cuda=WINCKELMANS_RESIDUAL_CUDA,
        defect_envelope=WINCKELMANS_DEFECT,
    ),
    treecode_kernel_id=1,
)
