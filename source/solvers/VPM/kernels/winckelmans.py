"""
Factory for the Winckelmans-Leonard algebraic regularization kernel set.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import taichi as ti

from ..config.constants import ONE_OVER_FOUR_PI


def create_winckelmans_kernels(dtype=ti.f32):
    """Create Winckelmans kernel functions with specified precision.

    Args:
        dtype: Taichi data type (ti.f32 or ti.f64)

    Returns:
        Dictionary with keys: 'q_', 'zeta_', 'g_',  'diffusivity_constant_', 'energy_equivalence_constant_'
    """

    # Half-integer powers of (ρ²+1) are written as muls + one sqrt instead of the
    # generic ``pow`` (≈ exp+log): (ρ²+1) > 0 always, so x**2.5 = x²·√x etc.  This
    # is algebraically exact (and slightly *more* accurate than pow) and drops two
    # transcendentals to one sqrt on the velocity/gradient hot path.
    @ti.func
    def zeta_(density: ti.template()) -> ti.template():  # type: ignore
        base = density * density + 1.0
        return ti.cast(7.5 / (base * base * base * ti.sqrt(base)) * ONE_OVER_FOUR_PI, dtype)

    @ti.func
    def q_(density: ti.template()) -> ti.template():  # type: ignore
        d2 = density * density
        base = d2 + 1.0
        return ti.cast(
            density * d2 * (d2 + 2.5) / (base * base * ti.sqrt(base)) * ONE_OVER_FOUR_PI,
            dtype,
        )

    @ti.func
    def g_(density: ti.template()) -> ti.template():  # type: ignore
        base = density * density + 1.0
        return ti.cast((base + 0.5) / (base * ti.sqrt(base)) * ONE_OVER_FOUR_PI, dtype)

    @ti.func
    def diffusivity_constant_():
        # Winckelmans--Leonard high-order algebraic core-spreading constant.
        # The reference VPM formulation uses d(sigma²)/dt = (256/45) nu.
        return ti.cast(256.0 / 45.0, dtype)

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
