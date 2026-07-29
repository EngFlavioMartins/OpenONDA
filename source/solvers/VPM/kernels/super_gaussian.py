"""
Factory for the super-Gaussian (high-order) regularization kernel set.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import taichi as ti

from ..config.constants import (
    ONE_OVER_FOUR_PI,
    ONE_OVER_SQRT2,
    SQRT_2_OVER_PI,
    ONE_OVER_TWO_PI_POW_1p05,
)


def create_super_gaussian_kernels(dtype=ti.f32):
    """Create Super-Gaussian kernel functions with specified precision.

    Args:
        dtype: Taichi data type (ti.f32 or ti.f64)

    Returns:
        Dictionary with keys: 'q_', 'zeta_', 'g_', 'diffusivity_constant_'
    """

    @ti.func
    def err_func(x):
        # Abramowitz & Stegun constants
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.327591100

        sign = 1.0
        if x < 0:
            sign = -1.0
            x = -x
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * ti.exp(-x * x))
        return sign * y

    @ti.func
    def zeta_(density: ti.template()) -> ti.template():  # type: ignore
        exp_term = SQRT_2_OVER_PI * ti.exp(-density * density / 2.0)
        poly_term = 2.5 - density * density / 2.0
        result = exp_term * poly_term
        return ti.cast(result * ONE_OVER_FOUR_PI, dtype)

    @ti.func
    def q_(density: ti.template()) -> ti.template():  # type: ignore
        result = 0.0
        if density < 1e-4:
            rho_sq = density * density
            result = SQRT_2_OVER_PI * (
                (5.0 / 6.0) * density * rho_sq - (7.0 / 20.0) * density * rho_sq * rho_sq
            )
        else:
            err_term = err_func(density * ONE_OVER_SQRT2)
            exp_term = SQRT_2_OVER_PI * density * ti.exp(-density * density / 2.0)
            poly_term = 1.0 - density * density / 2.0
            result = err_term - poly_term * exp_term
        return ti.cast(result * ONE_OVER_FOUR_PI, dtype)

    @ti.func
    def g_(density: ti.template()) -> ti.template():  # type: ignore
        result = 0.0
        if density < 1e-4:
            result = 1.5 * SQRT_2_OVER_PI - (5.0 / 12.0) * SQRT_2_OVER_PI * density**2
        else:
            safe_density = ti.max(density, 1e-12)
            erf_term = err_func(density * ONE_OVER_SQRT2) / safe_density
            decay_term = ONE_OVER_TWO_PI_POW_1p05 * ti.exp(-density * density / 2.0)
            result = erf_term + decay_term
        return ti.cast(result * ONE_OVER_FOUR_PI, dtype)

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
        "g_": g_,
        "diffusivity_constant_": diffusivity_constant_,
        "energy_equivalence_constant_": energy_equivalence_constant_,
        "volume_correction_constant_": volume_correction_constant_,
        "angular_impulse_correction_constant_": angular_impulse_correction_constant_,
    }
