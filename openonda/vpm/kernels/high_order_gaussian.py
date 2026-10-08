"""
High-Order Gaussian kernel module for VPM solver.
==================================================
Second-order corrected Gaussian regularization kernel for Biot-Savart
velocity induction.  The vorticity density function is

    ζ(ρ) = (1/π^{3/2}) × (2.5 − ρ²) × exp(−ρ²)

which is a standard Gaussian multiplied by the polynomial correction
(2.5 − ρ²) that cancels its second moment: the fourth-order Gaussian
kernel of Beale & Majda (1985).

The moment-cancelling correction changes near-core induction and can take
negative vorticity-density values. Select and qualify it for the intended
flow; higher algebraic order alone is not a production-accuracy guarantee.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from collections.abc import Callable
import math

import numpy as np
import taichi as ti

from openonda.core._numba import cacheable_njit as njit

from .definition import GaussianDefectEnvelope, KernelDefinition, SlabImageSplit
from .gaussian import create_gaussian_kernels
from .gaussian import q_host as gaussian_q_host

# The (2.5 - rho^2) correction cancels the second moment of the Gaussian.
SECOND_MOMENT = 0.0
# Weight of rho^3 exp(-rho^2) by which q exceeds the Gaussian q.
CORRECTION_SCALE = 0.5 * math.pi**-1.5


def create_high_order_gaussian_kernels(dtype: object = ti.f32) -> dict[str, Callable]:
    """Create High-Order Gaussian kernel functions with specified precision.

    Returns a dict of Taichi-compiled kernel functions compatible with the
    VPM physics sequence (same interface as all other kernel factories).
    """

    gaussian = create_gaussian_kernels(dtype)
    gaussian_q, gaussian_g, gaussian_zeta = gaussian["q_"], gaussian["g_"], gaussian["zeta_"]
    gaussian_radial_factors = gaussian["radial_factors_"]
    coefficient = CORRECTION_SCALE

    @ti.func
    def zeta_(density: ti.template()) -> ti.template():
        """Normalized corrected density: (2.5-rho²) exp(-rho²)/pi^(3/2)."""
        return ti.cast((2.5 - density * density) * gaussian_zeta(density), dtype)

    @ti.func
    def q_(density: ti.template()) -> ti.template():
        """Integrated corrected density, expressed without small-radius cancellation."""
        rho_sq = density * density
        return ti.cast(
            gaussian_q(density) + coefficient * density * rho_sq * ti.exp(-rho_sq), dtype
        )

    @ti.func
    def radial_factors_(density: ti.template(), sigma: ti.template(), with_gradient: ti.template()):
        """Return corrected Gaussian radial factors in m⁻³ and m⁻⁵."""
        factors = gaussian_radial_factors(density, sigma, with_gradient)
        correction = coefficient * ti.exp(-density * density)
        first = factors[0] + correction / sigma**3
        second = ti.cast(0.0, dtype)
        if ti.static(with_gradient):
            second = factors[1] + 2.0 * correction / sigma**5
        return ti.Vector([first, second])

    @ti.func
    def g_(density: ti.template()) -> ti.template():
        """Potential satisfying -g'(rho)=q(rho)/rho² and g(infinity)=0."""
        return ti.cast(gaussian_g(density) + 0.5 * coefficient * ti.exp(-density * density), dtype)

    @ti.func
    def diffusivity_constant_():
        return ti.cast(4.0, dtype)

    @ti.func
    def energy_equivalence_constant_():
        return ti.cast(ti.sqrt(6.0), dtype)

    @ti.func
    def volume_correction_constant_():
        return ti.cast(1.0 / 3.0, dtype)

    @ti.func
    def angular_impulse_correction_constant_():
        # The 2nd-order polynomial correction cancels the standard Gaussian
        # angular-impulse correction term; constant = 0.
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
    """Gaussian q plus the correction, without small-radius cancellation."""
    rho = np.asarray(rho, dtype=np.float64)
    return gaussian_q_host(rho) + CORRECTION_SCALE * rho**3 * np.exp(-rho * rho)


def zeta_host(rho: np.ndarray | float) -> np.ndarray:
    """Corrected density (2.5 - rho^2) exp(-rho^2)/pi^1.5 in float64."""
    rho = np.asarray(rho, dtype=np.float64)
    return (2.5 - rho * rho) * np.exp(-rho * rho) / math.pi**1.5


@njit(cache=True)
def corrected_gaussian_residual(radius: float, width: float) -> tuple[float, float]:
    """Corrected-Gaussian minus Gaussian factors at equal width, exactly.

    q - q_G = c rho^3 exp(-rho^2) gives a = c exp(-rho^2)/s^3 and
    b = 2 c exp(-rho^2)/s^5, with c = 1/(2 pi^1.5).
    """
    e = math.exp(-((radius / width) ** 2))
    return CORRECTION_SCALE * e / width**3, 2.0 * CORRECTION_SCALE * e / width**5


CORRECTED_GAUSSIAN_RESIDUAL_CUDA = r"""
__device__ inline void residual(Real r, Real s, Real &a, Real &b) {
    const Real scale = (Real)0.0897935610625832808445;
    Real inverse = 1 / s, rho = r * inverse;
    Real e = rr_exp(-rho * rho) * scale * inverse * inverse * inverse;
    a = e;
    b = 2 * e * inverse * inverse;
}
"""

# With x = r/s, T = T_G - c x^3 exp(-x^2) and erfc(x) <= exp(-x^2)/(sqrt(pi) x):
#   |T| <= pi^-1.5 exp(-x^2) (1/(4x) + x/2 + x^3/2),
# and for x^2 >= 2.5, |3T + x^3 zeta| <= 3|T| + pi^-1.5 exp(-x^2) x^5.
CORRECTED_GAUSSIAN_DEFECT = GaussianDefectEnvelope(
    velocity=((-1, 0.25), (1, 0.5), (3, 0.5)),
    gradient=((-1, 0.75), (1, 1.5), (3, 1.5), (5, 1.0)),
    minimum_ratio=3.0,
)

DEFINITION = KernelDefinition(
    name="HIGH_ORDER_GAUSSIAN",
    q=q_host,
    zeta=zeta_host,
    device_factory=create_high_order_gaussian_kernels,
    second_moment=SECOND_MOMENT,
    slab_split=SlabImageSplit(
        partner_ratio=1.0,
        residual=corrected_gaussian_residual,
        residual_cuda=CORRECTED_GAUSSIAN_RESIDUAL_CUDA,
        defect_envelope=CORRECTED_GAUSSIAN_DEFECT,
    ),
)
