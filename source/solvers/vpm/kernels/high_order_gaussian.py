"""
High-Order Gaussian kernel module for VPM solver.
==================================================
Second-order corrected Gaussian regularization kernel for Biot-Savart
velocity induction.  The vorticity density function is

    ζ(ρ) = (1/π^{3/2}) × (2.5 − ρ²) × exp(−ρ²)

which is a standard Gaussian multiplied by the 2nd-order polynomial
correction (2.5 − ρ²), also known in the literature as a "corrected"
or "algebraically-enhanced" Gaussian kernel (Winckelmans & Leonard 1993,
Section 3.2).

The moment-cancelling correction changes near-core induction and can take
negative vorticity-density values. Select and qualify it for the intended
flow; higher algebraic order alone is not a production-accuracy guarantee.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import math

import taichi as ti

from .gaussian import create_gaussian_kernels


def create_high_order_gaussian_kernels(dtype=ti.f32):
    """Create High-Order Gaussian kernel functions with specified precision.

    Returns a dict of Taichi-compiled kernel functions compatible with the
    VPM physics pipeline (same interface as all other kernel factories).
    """

    gaussian = create_gaussian_kernels(dtype)
    gaussian_q, gaussian_g, gaussian_zeta = gaussian["q_"], gaussian["g_"], gaussian["zeta_"]
    gaussian_radial_factors = gaussian["radial_factors_"]
    coefficient = 0.5 * math.pi**-1.5

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
        # angular-impulse correction term; constant = 0 (Winckelmans 1993).
        return ti.cast(0.0, dtype)

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
