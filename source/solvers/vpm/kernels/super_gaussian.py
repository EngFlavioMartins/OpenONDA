"""
Factory for the super-Gaussian (high-order) regularization kernel set.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import math

import taichi as ti

from .high_order_gaussian import create_high_order_gaussian_kernels


def create_super_gaussian_kernels(dtype=ti.f32):
    """Create Super-Gaussian kernel functions with specified precision.

    Args:
        dtype: Taichi data type (ti.f32 or ti.f64)

    Returns:
        Dictionary with the selected q, density, energy and diffusion functions,
        plus finite ``radial_factors_`` for velocity and its Jacobian.
    """

    corrected = create_high_order_gaussian_kernels(dtype)
    corrected_q, corrected_g, corrected_zeta = corrected["q_"], corrected["g_"], corrected["zeta_"]
    corrected_radial_factors = corrected["radial_factors_"]
    inverse_sqrt_two = 1.0 / math.sqrt(2.0)

    @ti.func
    def zeta_(density: ti.template()) -> ti.template():
        """Corrected Gaussian density rescaled to exp(-rho²/2)."""
        return ti.cast(corrected_zeta(density * inverse_sqrt_two) * 0.5 * inverse_sqrt_two, dtype)

    @ti.func
    def q_(density: ti.template()) -> ti.template():
        """Enclosed circulation is invariant under the core-coordinate rescaling."""
        return ti.cast(corrected_q(density * inverse_sqrt_two), dtype)

    @ti.func
    def radial_factors_(density: ti.template(), sigma: ti.template(), with_gradient: ti.template()):
        """Return corrected Gaussian factors after the sqrt(2) core rescaling."""
        factors = corrected_radial_factors(density * inverse_sqrt_two, sigma, with_gradient)
        first = factors[0] * 0.5 * inverse_sqrt_two
        second = ti.cast(0.0, dtype)
        if ti.static(with_gradient):
            second = factors[1] * 0.25 * inverse_sqrt_two
        return ti.Vector([first, second])

    @ti.func
    def g_(density: ti.template()) -> ti.template():
        """Rescale the potential with its required inverse-length factor."""
        return ti.cast(corrected_g(density * inverse_sqrt_two) * inverse_sqrt_two, dtype)

    @ti.func
    def diffusivity_constant_():
        return ti.cast(2.0, dtype)

    @ti.func
    def energy_equivalence_constant_():
        return ti.cast(ti.sqrt(15.0 / 2.0), dtype)

    @ti.func
    def volume_correction_constant_():
        """Volume correction constant for radius evolution: dσ/dt = σ * C * div(u).

        For 3D spherical particles, C = 1/3 (Alvarez 2022).
        """
        return ti.cast(1.0 / 3.0, dtype)

    @ti.func
    def angular_impulse_correction_constant_():
        """Second moment m2 = ∫|q|² ζ(|q|) d³q of the regularization kernel.

        The (2.5 - ρ²/2) polynomial makes this kernel second-order accurate, so
        its second moment vanishes identically -- as it does for
        HIGH_ORDER_GAUSSIAN.  It was 1.875 here, described as "intermediate
        between Gaussian and Winckelmans", which is not how the moment works:
        a moment-cancelling polynomial gives 0, not an interpolated value.
        """
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
