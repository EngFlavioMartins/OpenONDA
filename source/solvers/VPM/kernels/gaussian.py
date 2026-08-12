"""
Factory for the Gaussian vortex-blob regularization kernel set.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import taichi as ti

from ..config.constants import FOUR_OVER_THREE_SQRT_PI, GAUSSIAN_Q_SERIES_CROSSOVER


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
        # Enclosed-circulation fraction of the Gaussian blob, with the 1/4pi of
        # the Biot-Savart law folded in:
        #     q(r) = [erf(r) - (2/sqrt(pi)) r exp(-r^2)] / (4 pi)
        #
        # The bracket subtracts two quantities that both tend to (2/sqrt(pi)) r
        # while their difference is only O(r^3), so the closed form loses all
        # significance for small r (f32 relative error ~1.5 eps / r^2: 5.6e-2 at
        # r = 1e-2, 7e+4 at r = 1e-4, where it even changes sign).  Below the
        # crossover use the series
        #     q(r) = C r^3 [1 - (3/5) r^2 + (3/14) r^4] / (4 pi) + O(r^9),
        # with C = 4/(3 sqrt(pi)), from
        #     erf(r) - (2/sqrt(pi)) r exp(-r^2)
        #         = (2/sqrt(pi)) [ (2/3) r^3 - (2/5) r^5 + (1/7) r^7 - ... ].
        # Coefficient and crossover are shared with the treecode's copy of this
        # kernel through config.constants so the two cannot drift apart.
        res = ti.cast(0.0, dtype)
        if density < GAUSSIAN_Q_SERIES_CROSSOVER:
            d2 = density * density
            res = (
                FOUR_OVER_THREE_SQRT_PI
                * density
                * d2
                * (1.0 - 0.6 * d2 + (3.0 / 14.0) * d2 * d2)
                * ONE_OVER_FOUR_PI
            )
        else:
            erf_term = err_func(density)
            exp_term = TWO_OVER_SQRT_PI * density * ti.exp(-density * density)
            res = (erf_term - exp_term) * ONE_OVER_FOUR_PI
        return ti.cast(res, dtype)

    @ti.func
    def g_(density: ti.template()) -> ti.template():  # type: ignore
        # Energy kernel g = ∫_density^∞ q(s)/s² ds
        #                 = erf(density)/(4*pi*density).
        # The explicit origin limit is the finite vortex-blob self energy.
        res = ti.cast(0.0, dtype)
        if density < 1e-4:
            res = ONE_OVER_PI_15 * (0.5 - density * density / 6.0)
        else:
            safe_density = ti.max(density, 1e-12)
            res = err_func(density) / safe_density * ONE_OVER_FOUR_PI
        return ti.cast(res, dtype)

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
        """Second moment m2 = ∫|q|² ζ(|q|) d³q of the regularization kernel.

        Used in: A = (1/3) Σ x × (x × Γ) - (2/9) m2 σ² Γ, which follows from
        ∫ x × (x × ω) dV = d × (d × Γ) - (2/3) m2 σ² Γ for a blob at d.
        """
        return ti.cast(1.5, dtype)

    return {
        "q_": q_,
        "zeta_": zeta_,
        "g_": g_,
        "diffusivity_constant_": diffusivity_constant_,
        "energy_equivalence_constant_": energy_equivalence_constant_,
        "volume_correction_constant_": volume_correction_constant_,
        "angular_impulse_correction_constant_": angular_impulse_correction_constant_,
    }
