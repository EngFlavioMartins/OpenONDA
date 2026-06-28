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

    @ti.func
    def zeta_(density: ti.template()) -> ti.template():  # type: ignore
        return ti.cast(7.5 / ((density * density + 1.0) ** 3.5) * ONE_OVER_FOUR_PI, dtype)

    @ti.func
    def q_(density: ti.template()) -> ti.template():  # type: ignore
        return ti.cast(
            density**3 * (density**2 + 2.5) / (density * density + 1.0) ** 2.5 * ONE_OVER_FOUR_PI,
            dtype,
        )

    @ti.func
    def g_(density: ti.template()) -> ti.template():  # type: ignore
        return ti.cast(
            (density * density + 1.5) / ((density * density + 1) ** 1.5) * ONE_OVER_FOUR_PI, dtype
        )

    @ti.func
    def diffusivity_constant_():
        return ti.cast(5.0, dtype)

    @ti.func
    def energy_equivalence_constant_():
        return ti.cast(ti.sqrt(6.0), dtype)

    @ti.func
    def volume_correction_constant_():
        """Volume correction constant for rVPM radius evolution: dσ/dt = σ * C * div(u).

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
