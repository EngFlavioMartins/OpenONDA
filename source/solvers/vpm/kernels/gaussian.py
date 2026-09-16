"""
Factory for the Gaussian vortex-blob regularization kernel set.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import math

import taichi as ti

from ..config.constants import GAUSSIAN_Q_SERIES_CROSSOVER

# Integrating q'(rho) = rho**2 exp(-rho**2) / pi**1.5 gives these
# coefficients. See DLMF 7.6.1, https://dlmf.nist.gov/7.6.E1.
GAUSSIAN_Q_SERIES_COEFFICIENTS = tuple(
    (-1.0) ** n / (math.factorial(n) * (2 * n + 3)) for n in range(18)
)
_ERF_SERIES_COEFFICIENTS = tuple((-1.0) ** n / (math.factorial(n) * (2 * n + 1)) for n in range(18))

# Cephes erfc rational approximation, used only for 1 <= rho < 6.
# Coefficients from SciPy 1.14.1 special/cephes/ndtr.h (P and Q).
# Copyright 1984, 1987, 1988, 1992 Stephen L. Moshier; SciPy developers.
# Attribution and redistribution terms: THIRD_PARTY_NOTICES.md.
_ERFC_NUMERATOR = (
    2.46196981473530512524e-10,
    5.64189564831068821977e-1,
    7.46321056442269912687,
    4.86371970985681366614e1,
    1.96520832956077098242e2,
    5.26445194995477358631e2,
    9.34528527171957607540e2,
    1.02755188689515710272e3,
    5.57535335369399327526e2,
)
_ERFC_DENOMINATOR = (
    1.0,
    1.32281951154744992508e1,
    8.67072140885989742329e1,
    3.54937778887819891062e2,
    9.75708501743205489753e2,
    1.82390916687909736289e3,
    2.24633760818710981792e3,
    1.65666309194161350182e3,
    5.57535340817727675546e2,
)


def create_gaussian_kernels(dtype=ti.f32):
    """Create Gaussian kernel functions with specified precision.

    Convention: exp(-r^2 / sigma^2)
    Normalization: 1 / (pi^1.5 * sigma^3)

    Args:
        dtype: Taichi data type (ti.f32 or ti.f64)

    Returns:
        Dictionary containing q, density, energy and diffusion functions,
        plus ``radial_factors_`` for finite velocity/Jacobian coefficients.
    """

    ONE_OVER_PI_15 = math.pi**-1.5
    TWO_OVER_SQRT_PI = 2.0 / math.sqrt(math.pi)
    ONE_OVER_FOUR_PI = 1.0 / (4.0 * math.pi)
    # At rho <= 1 the first omitted terms are below the selected precision:
    # q: < 1.8e-10 (f32), < 7.3e-19 (f64), before rounding.
    series_terms = 11 if dtype == ti.f32 else 18

    @ti.func
    def err_func(x):
        radius = ti.abs(x)
        result = ti.cast(1.0, dtype)
        if radius < 1.0:
            polynomial = ti.cast(_ERF_SERIES_COEFFICIENTS[series_terms - 1], dtype)
            for n in ti.static(range(series_terms - 2, -1, -1)):
                polynomial = polynomial * radius * radius + _ERF_SERIES_COEFFICIENTS[n]
            result = TWO_OVER_SQRT_PI * radius * polynomial
        elif radius < 6.0:
            numerator = ti.cast(_ERFC_NUMERATOR[0], dtype)
            denominator = ti.cast(_ERFC_DENOMINATOR[0], dtype)
            for n in ti.static(range(1, len(_ERFC_NUMERATOR))):
                numerator = numerator * radius + _ERFC_NUMERATOR[n]
                denominator = denominator * radius + _ERFC_DENOMINATOR[n]
            result = 1.0 - ti.exp(-radius * radius) * numerator / denominator
        # erfc(6) < 2.2e-17: erf rounds to one in both supported precisions.
        if x < 0:
            result = -result
        return result

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
        # Avoid the cancellation of two O(r) terms to obtain an O(r^3)
        # difference. The integrated Gaussian series and accurate erf agree
        # to the selected precision at the crossover. LBVH uses this factory.
        res = ti.cast(0.0, dtype)
        if density < GAUSSIAN_Q_SERIES_CROSSOVER:
            d2 = density * density
            polynomial = ti.cast(GAUSSIAN_Q_SERIES_COEFFICIENTS[series_terms - 1], dtype)
            for n in ti.static(range(series_terms - 2, -1, -1)):
                polynomial = polynomial * d2 + GAUSSIAN_Q_SERIES_COEFFICIENTS[n]
            res = ONE_OVER_PI_15 * density * d2 * polynomial
        else:
            erf_term = err_func(density)
            exp_term = TWO_OVER_SQRT_PI * density * ti.exp(-density * density)
            res = (erf_term - exp_term) * ONE_OVER_FOUR_PI
        return ti.cast(res, dtype)

    @ti.func
    def radial_factors_(density: ti.template(), sigma: ti.template(), with_gradient: ti.template()):
        """Return Gaussian ``q/r³`` [m⁻³] and ``3q/r⁵-ζ/(σ³r²)`` [m⁻⁵].

        ``density=r/sigma`` is dimensionless. The integrated-density series
        evaluates both finite core limits without an f32 inverse fifth power.
        The second entry is zero for velocity-only calls.
        """
        first = ti.cast(0.0, dtype)
        second = ti.cast(0.0, dtype)
        if density < 1.0:
            d2 = density * density
            polynomial = ti.cast(GAUSSIAN_Q_SERIES_COEFFICIENTS[series_terms - 1], dtype)
            for n in ti.static(range(series_terms - 2, -1, -1)):
                polynomial = polynomial * d2 + GAUSSIAN_Q_SERIES_COEFFICIENTS[n]
            first = ONE_OVER_PI_15 * polynomial / (sigma * sigma * sigma)
            if ti.static(with_gradient):
                derivative = ti.cast(
                    -2.0 * (series_terms - 1) * GAUSSIAN_Q_SERIES_COEFFICIENTS[series_terms - 1],
                    dtype,
                )
                for n in ti.static(range(series_terms - 2, 0, -1)):
                    derivative = derivative * d2 - 2.0 * n * GAUSSIAN_Q_SERIES_COEFFICIENTS[n]
                second = ONE_OVER_PI_15 * derivative / (sigma**5)
        else:
            q_value = q_(density)
            first = q_value / (sigma**3 * density**3)
            if ti.static(with_gradient):
                second = (3.0 * q_value / density**5 - zeta_(density) / density**2) / sigma**5
        return ti.Vector([first, second])

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
        # d(sigma^2)/dt = 4.0 * kinematic_viscosity for exp(-r^2/sigma^2)
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
        "radial_factors_": radial_factors_,
        "g_": g_,
        "diffusivity_constant_": diffusivity_constant_,
        "energy_equivalence_constant_": energy_equivalence_constant_,
        "volume_correction_constant_": volume_correction_constant_,
        "angular_impulse_correction_constant_": angular_impulse_correction_constant_,
    }
