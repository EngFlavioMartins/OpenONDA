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

This is the recommended production kernel: it improves far-field accuracy
over the plain Gaussian kernel while maintaining the infinite-support
Gaussian smoothness (no compact-support discontinuities).

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import taichi as ti

from ..config.constants import ONE_OVER_FOUR_PI


def create_high_order_gaussian_kernels(dtype=ti.f32):
    """Create High-Order Gaussian kernel functions with specified precision.

    Returns a dict of Taichi-compiled kernel functions compatible with the
    VPM physics pipeline (same interface as all other kernel factories).
    """

    ONE_OVER_PI_15 = 0.179587122125
    TWO_OVER_SQRT_PI = 1.1283791671
    # The closed form subtracts two O(rho) terms.  The f32 error crossover is
    # much larger than 1e-4; keep a convergent odd-power series through rho^17
    # and use it for both f32 and f64.  The omitted next term is below 1e-7 of
    # the retained bracket at rho=0.5.
    SERIES_CROSSOVER = 0.5

    @ti.func
    def err_func(x):
        # Abramowitz & Stegun approximation for erf(x) — max error < 1.5e-7
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
        # ζ(ρ) = (1/π^{3/2}) × (2.5 − ρ²) × exp(−ρ²)
        rho_sq = density * density
        return ti.cast(ONE_OVER_PI_15 * (2.5 - rho_sq) * ti.exp(-rho_sq), dtype)

    @ti.func
    def q_(density: ti.template()) -> ti.template():  # type: ignore
        # q(ρ) = [erf(ρ) + (2/√π) ρ (ρ²−1) exp(−ρ²)] / (4π)
        # Small-ρ Taylor fallback to avoid catastrophic cancellation.  For
        # k>=1, the coefficient of rho^(2k+1) in the bracket is
        # (-1)^(k-1)(2k+3)/((k-1)!(2k+1)).
        rho_sq = density * density
        res = 0.0
        if density < SERIES_CROSSOVER:
            polynomial = 5.0 / 3.0 + rho_sq * (
                -7.0 / 5.0
                + rho_sq
                * (
                    9.0 / 14.0
                    + rho_sq
                    * (
                        -11.0 / 54.0
                        + rho_sq
                        * (
                            13.0 / 264.0
                            + rho_sq
                            * (
                                -1.0 / 104.0
                                + rho_sq * (17.0 / 10800.0 + rho_sq * (-19.0 / 85680.0))
                            )
                        )
                    )
                )
            )
            series = TWO_OVER_SQRT_PI * density * rho_sq * polynomial
            res = series * ONE_OVER_FOUR_PI
        else:
            erf_term = err_func(density)
            corr_term = TWO_OVER_SQRT_PI * density * (rho_sq - 1.0) * ti.exp(-rho_sq)
            res = (erf_term + corr_term) * ONE_OVER_FOUR_PI
        return ti.cast(res, dtype)

    @ti.func
    def g_(density: ti.template()) -> ti.template():  # type: ignore
        # Integrating -g'(ρ) = q(ρ)/ρ² gives
        # g(ρ) = [erf(ρ)/ρ + exp(-ρ²)/sqrt(pi)]/(4π).
        # This extra Gaussian term is required by the corrected q kernel.
        res = 0.0
        if density < SERIES_CROSSOVER:
            d2 = density * density
            res = TWO_OVER_SQRT_PI * (
                1.5
                + d2
                * (
                    -5.0 / 6.0
                    + d2
                    * (
                        7.0 / 20.0
                        + d2
                        * (
                            -3.0 / 28.0
                            + d2
                            * (
                                11.0 / 432.0
                                + d2
                                * (
                                    -13.0 / 2640.0
                                    + d2
                                    * (
                                        1.0 / 1248.0
                                        + d2 * (-17.0 / 151200.0 + d2 * (19.0 / 1370880.0))
                                    )
                                )
                            )
                        )
                    )
                )
            )
        else:
            safe_density = ti.max(density, 1e-12)
            res = err_func(density) / safe_density + 0.5 * TWO_OVER_SQRT_PI * ti.exp(
                -density * density
            )
        return ti.cast(res * ONE_OVER_FOUR_PI, dtype)

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
        "g_": g_,
        "diffusivity_constant_": diffusivity_constant_,
        "energy_equivalence_constant_": energy_equivalence_constant_,
        "volume_correction_constant_": volume_correction_constant_,
        "angular_impulse_correction_constant_": angular_impulse_correction_constant_,
    }
