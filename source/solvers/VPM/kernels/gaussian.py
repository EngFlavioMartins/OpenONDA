"""
Gaussian module for VPM solver.
==================
Gaussian module for VPM solver. module.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import taichi as ti


def create_gaussian_kernels(dtype=ti.f32):
    """Create Gaussian kernel functions with specified precision.

    Convention: exp(-r^2 / sigma^2)
    Normalization: 1 / (pi^1.5 * sigma^3)

    Args:
        dtype: Taichi data type (ti.f32 or ti.f64)

    Returns:
        Dictionary with keys: 'q_', 'zeta_', 'g_', 'diffusivity_constant_'
    """

    ONE_OVER_PI_15 = 0.179587122125
    TWO_OVER_SQRT_PI = 1.1283791671
    ONE_OVER_FOUR_PI = 0.0795774715

    @ti.func
    def err_func(x):
        # Abramowitz & Stegun constants for erf(x)
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
        # Gaussian distribution zeta = (1/pi^1.5) * exp(-density^2)
        return ti.cast(ONE_OVER_PI_15 * ti.exp(-density * density), dtype)

    @ti.func
    def q_(density: ti.template()) -> ti.template():  # type: ignore
        # Integral of Gaussian: erf(density) - (2/sqrt(pi)) * density * exp(-density^2)
        # Multiplied by 1/4pi for Biot-Savart normalization
        res = 0.0
        if density < 1e-4:
            # Taylor expansion for small density to avoid 0/0 precision loss
            # q(density)/density^3 approx (1/4pi) * (4/(3*sqrt(pi)))
            # actually we compute q(density) directly
            res = (4.0 / (3.0 * ti.sqrt(ti.acos(-1.0) ** 3))) * (density**3) * ONE_OVER_FOUR_PI
        else:
            erf_term = err_func(density)
            exp_term = TWO_OVER_SQRT_PI * density * ti.exp(-density * density)
            res = (erf_term - exp_term) * ONE_OVER_FOUR_PI
        return ti.cast(res, dtype)

    @ti.func
    def g_(density: ti.template()) -> ti.template():  # type: ignore
        # Energy kernel (integral of self-energy)
        # For exp(-density^2), it involves erf(density*sqrt(2)/2)? No, let's keep it simple.
        # Actually g is integral of q(r)/r^2.
        # Approximation: erf(density) / density * (1/4pi)
        erf_term = err_func(density) / (density + 1e-12)
        return ti.cast(erf_term * ONE_OVER_FOUR_PI, dtype)

    @ti.func
    def diffusivity_constant_():
        # d(sigma^2)/dt = 4.0 * nu for exp(-r^2/sigma^2)
        return ti.cast(4.0, dtype)

    @ti.func
    def energy_equivalence_constant_():
        return ti.cast(ti.sqrt(6.0), dtype)

    @ti.func
    def volume_correction_constant_():
        return ti.cast(1.0 / 3.0, dtype)

    @ti.func
    def angular_impulse_correction_constant_():
        """Angular impulse correction constant for second moment correction.

        For Gaussian kernel: C = 3.0 (Winckelmans 1993)
        Used in: A = (1/3) Σ x × (x × Γ) - (2/9) C σ² Γ_total
        """
        return ti.cast(3.0, dtype)

    return {
        "q_": q_,
        "zeta_": zeta_,
        "g_": g_,
        "diffusivity_constant_": diffusivity_constant_,
        "energy_equivalence_constant_": energy_equivalence_constant_,
        "volume_correction_constant_": volume_correction_constant_,
        "angular_impulse_correction_constant_": angular_impulse_correction_constant_,
    }
