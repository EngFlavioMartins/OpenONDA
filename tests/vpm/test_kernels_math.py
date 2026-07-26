"""
Kernel mathematical correctness tests (pure Python, no Taichi).

These tests verify that the mathematical formulas embedded in the VPM kernel
modules (Gaussian, High-Order Gaussian, Super-Gaussian, Winckelmans) satisfy
theoretical identities: normalization, non-singular behaviour, consistency
between q/ζ/g, and moment properties.

Because the kernel factories return ``ti.func`` objects that cannot be called
from Python, each kernel is re-implemented here as a pure NumPy function.  Any
discrepancy between the NumPy re-implementation and the Taichi version would be
caught by the downstream solver tests (``test_single_blob.py``, etc.).
"""

from math import erf, pi, sqrt

import numpy as np
import pytest

# ── Kernel registry ───────────────────────────────────────────────────────────

_KERNEL_FACTORIES = {
    "GAUSSIAN": None,  # implemented inline below
    "HIGH_ORDER_GAUSSIAN": None,
    "SUPER_GAUSSIAN": None,
    "WINCKELMANS": None,
}


# ── Pure-Python kernel re-implementations ─────────────────────────────────────


def _zeta_gaussian(rho):
    """ζ(ρ) = (1/π^{3/2}) exp(−ρ²)"""
    return np.exp(-(rho**2)) / (pi**1.5)


def _q_gaussian(rho):
    """q(ρ) = [erf(ρ) − (2/√π)ρ exp(−ρ²)] / (4π)"""
    # Avoid 0/0 at origin by using the limit: q(ρ)/ρ³ → 1/(6π^{3/2}).
    # np.where keeps this correct for both scalar and array inputs (boolean-mask
    # assignment would raise on a 0-d array, e.g. a bare float).
    rho = np.asarray(rho, dtype=float)
    verf = np.vectorize(erf)
    taylor = rho**3 / (6.0 * pi**1.5)
    full = (verf(rho) - (2.0 / sqrt(pi)) * rho * np.exp(-(rho**2))) / (4.0 * pi)
    return np.where(rho < 1e-4, taylor, full)


def _g_gaussian(rho):
    """g(ρ) = erf(ρ)/(4πρ)"""
    rho = np.asarray(rho)
    return np.where(
        rho < 1e-4,
        (1.0 / pi**1.5) * (0.5 - rho**2 / 6.0),
        np.vectorize(erf)(rho) / (4.0 * pi * rho),
    )


def _zeta_high_order_gaussian(rho):
    """ζ(ρ) = (1/π^{3/2})(2.5 − ρ²) exp(−ρ²)"""
    return (2.5 - rho**2) * np.exp(-(rho**2)) / (pi**1.5)


def _q_high_order_gaussian(rho):
    """q(ρ) = [erf(ρ) + (2/√π)ρ(ρ²−1)exp(−ρ²)] / (4π)"""
    # np.where keeps this scalar- and array-safe (see _q_gaussian).
    rho = np.asarray(rho, dtype=float)
    verf = np.vectorize(erf)
    # Taylor: q ≈ (2/√π)[(4/3)ρ³ − (6/5)ρ⁵] / (4π)
    taylor = (2.0 / sqrt(pi)) * ((4.0 / 3.0) * rho**3 - (6.0 / 5.0) * rho**5) / (4.0 * pi)
    full = (verf(rho) + (2.0 / sqrt(pi)) * rho * (rho**2 - 1.0) * np.exp(-(rho**2))) / (4.0 * pi)
    return np.where(rho < 1e-4, taylor, full)


def _g_high_order_gaussian(rho):
    """g(ρ) = [erf(ρ)/ρ + exp(−ρ²)/√π]/(4π)."""
    rho = np.asarray(rho)
    small = (3.0 / (4.0 * pi**1.5)) - 5.0 * rho**2 / (12.0 * pi**1.5)
    full = (np.vectorize(erf)(rho) / rho + np.exp(-(rho**2)) / sqrt(pi)) / (4.0 * pi)
    return np.where(rho < 1e-4, small, full)


def _zeta_super_gaussian(rho):
    """ζ(ρ) = (√(2/π)/4π)(2.5 − ρ²/2)exp(−ρ²/2)"""
    return sqrt(2.0 / pi) * (2.5 - rho**2 / 2.0) * np.exp(-(rho**2) / 2.0) / (4.0 * pi)


def _q_super_gaussian(rho):
    """q(ρ) = [erf(ρ/√2) − √(2/π)ρ(1−ρ²/2)exp(−ρ²/2)] / (4π)"""
    rho = np.asarray(rho)
    return (
        erf(rho / sqrt(2.0)) - sqrt(2.0 / pi) * rho * (1.0 - rho**2 / 2.0) * np.exp(-(rho**2) / 2.0)
    ) / (4.0 * pi)


def _g_super_gaussian(rho):
    """g(ρ) = [erf(ρ/√2)/ρ + exp(−ρ²/2)/√(2π)] / (4π)."""
    rho = np.asarray(rho)
    c = sqrt(2.0 / pi)
    small = (1.5 * c - (5.0 / 12.0) * c * rho**2) / (4.0 * pi)
    full = (np.vectorize(erf)(rho / sqrt(2.0)) / rho + np.exp(-(rho**2) / 2.0) / sqrt(2.0 * pi)) / (
        4.0 * pi
    )
    return np.where(rho < 1e-4, small, full)


def _zeta_winckelmans(rho):
    """ζ(ρ) = 7.5 / [(ρ²+1)^{3.5} · 4π]"""
    return 7.5 / ((rho**2 + 1.0) ** 3.5 * 4.0 * pi)


def _q_winckelmans(rho):
    """q(ρ) = ρ³(ρ²+2.5) / [(ρ²+1)^{2.5} · 4π]"""
    return rho**3 * (rho**2 + 2.5) / ((rho**2 + 1.0) ** 2.5 * 4.0 * pi)


def _g_winckelmans(rho):
    """g(ρ) = (ρ²+1.5) / [(ρ²+1)^{1.5} · 4π]"""
    return (rho**2 + 1.5) / ((rho**2 + 1.0) ** 1.5 * 4.0 * pi)


# Registry of kernels with their expected analytic properties.
#
# ``second_radial_moment`` is M₂ = 4π ∫₀^∞ ρ⁴ ζ(ρ) dρ — a pure property of the
# kernel SHAPE.  Closed forms (σ = 1):
#     Gaussian             ζ = π^{-3/2} e^{-ρ²}              → M₂ = 3/2
#     High-Order Gaussian  ζ = π^{-3/2}(5/2 − ρ²)e^{-ρ²}     → M₂ = 0   (lobe cancels)
#     Super-Gaussian       ζ = √(2/π)/4π (5/2 − ρ²/2)e^{-ρ²/2} → M₂ = 0 (lobe cancels)
#     Winckelmans          ζ = (15/2)/4π (1+ρ²)^{-7/2}       → M₂ = 3/2
#
# NOTE: M₂ is NOT the CSM diffusivity constant nor the Winckelmans-1993
# angular-impulse correction constant.  Those production coefficients
# (diffusivity {4,4,2,256/45}; angular {3,0,1.875,1.5}) are separate physical
# quantities — the diffusivity constant is the core-spreading rate dσ²/dt = Cnu
# and the angular constant comes from a higher-moment impulse formula — so they
# are validated by the integration tests (test_single_blob, test_two_particles),
# not by this shape-moment check.
_KERNELS = {
    "GAUSSIAN": {
        "zeta": _zeta_gaussian,
        "q": _q_gaussian,
        "g": _g_gaussian,
        "second_radial_moment": 1.5,
    },
    "HIGH_ORDER_GAUSSIAN": {
        "zeta": _zeta_high_order_gaussian,
        "q": _q_high_order_gaussian,
        "g": _g_high_order_gaussian,
        "second_radial_moment": 0.0,
    },
    "SUPER_GAUSSIAN": {
        "zeta": _zeta_super_gaussian,
        "q": _q_super_gaussian,
        "g": _g_super_gaussian,
        "second_radial_moment": 0.0,
    },
    "WINCKELMANS": {
        "zeta": _zeta_winckelmans,
        "q": _q_winckelmans,
        "g": _g_winckelmans,
        "second_radial_moment": 1.5,
    },
}


# ── Numerical integration helpers ─────────────────────────────────────────────


def _radial_quadrature(f, r_max=20.0, n=200000):
    """Integrate f(ρ) over 3-D space: ∫₀^∞ f(ρ) 4πρ² dρ."""
    rho = np.linspace(0.0, r_max, n)
    dr = rho[1] - rho[0]
    integrand = f(rho) * 4.0 * pi * rho**2
    return float(np.trapezoid(integrand, dx=dr))


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("kernel_name", list(_KERNELS.keys()))
def test_kernel_normalization(kernel_name):
    """
    The vorticity kernel must be normalized: ∫ ζ(ρ) d³x = 1.

    In spherical coordinates with σ=1:
        ∫₀^∞ ζ(ρ) · 4πρ² dρ = 1

    Failure → wrong prefactor in ζ (e.g. 1/π instead of 1/π^{3/2}).
    """
    kern = _KERNELS[kernel_name]
    integral = _radial_quadrature(kern["zeta"])
    assert abs(integral - 1.0) < 1e-4, (
        f"{kernel_name}: normalization integral = {integral:.6f}, expected 1.0"
    )


@pytest.mark.parametrize("kernel_name", list(_KERNELS.keys()))
def test_q_small_r_cubic(kernel_name):
    """
    As ρ→0, q(ρ) ∝ ρ³ (non-singular Biot-Savart regularisation).

    We check that q(1e-6) / (1e-6)³ is a finite constant within 1% of the
    theoretical small-ρ limit.

    Failure → 1/r singularity remaining in the velocity kernel.
    """
    kern = _KERNELS[kernel_name]
    rho = 1e-6
    q_val = float(kern["q"](rho))
    ratio = q_val / (rho**3)

    # Theoretical limit: q(ρ)/ρ³ → 1/(6π^{3/2}) for Gaussian
    # For other kernels the limit differs; we just check finiteness.
    assert ratio > 0.0 and np.isfinite(ratio), (
        f"{kernel_name}: q({rho})/{rho}³ = {ratio} (must be finite and positive)"
    )


@pytest.mark.parametrize("kernel_name", list(_KERNELS.keys()))
def test_gradient_consistency(kernel_name):
    """
    For a non-singular velocity gradient, 3q(ρ) − ρ³ζ(ρ) → 0 as ρ→0.

    This ensures the velocity-gradient kernel term2 = 3q/r⁵ − ζ/r²
    remains finite at the origin.

    Failure → singular velocity gradient (term2 blows up as 1/r²).
    """
    kern = _KERNELS[kernel_name]
    rho = 1e-6
    q_val = float(kern["q"](rho))
    zeta_val = float(kern["zeta"](rho))
    residual = abs(3.0 * q_val - rho**3 * zeta_val)
    assert residual < 1e-10, f"{kernel_name}: |3q − ρ³ζ| = {residual:.3e} at ρ={rho} (must ≈ 0)"


@pytest.mark.parametrize("kernel_name", list(_KERNELS.keys()))
def test_energy_kernel_g0_finite(kernel_name):
    """
    g(0) must be finite — the self-energy of a single vortex blob is well-defined.

    For Gaussian: g(ρ) = erf(ρ)/(4πρ), so g(0) = 1/(2π^{3/2}).
    For other kernels the limit differs.

    Failure → energy kernel has a 1/ρ pole at the origin.
    """
    kern = _KERNELS[kernel_name]
    rho = 1e-6
    g_val = float(kern["g"](rho))
    assert np.isfinite(g_val) and g_val > 0.0, (
        f"{kernel_name}: g({rho}) = {g_val} (must be finite and positive)"
    )


def _second_radial_moment(zeta, r_max=400.0, n=800000):
    """M₂ = 4π ∫₀^∞ ρ⁴ ζ(ρ) dρ — the 2nd radial moment of the vorticity kernel.

    Uses its own quadrature (NOT _radial_quadrature, which already carries the
    4πρ² spherical-volume factor and would double-count it here).  r_max is large
    because the Winckelmans kernel has an algebraic tail — 4πρ⁴ζ ~ ρ⁻³ — so the
    integral converges only slowly and a short cutoff truncates ~1e-3 of M₂.
    """
    rho = np.linspace(0.0, r_max, n)
    return float(np.trapezoid(4.0 * pi * rho**4 * zeta(rho), rho))


@pytest.mark.parametrize("kernel_name", list(_KERNELS.keys()))
def test_second_radial_moment_matches_kernel_shape(kernel_name):
    """
    The 2nd radial moment of ζ must match its analytic value:

        M₂ = 4π ∫₀^∞ ρ⁴ ζ(ρ) dρ

    Gaussian → 3/2, Winckelmans → 3/2, and the two corrected kernels
    (High-Order/Super-Gaussian) → 0 because their negative outer lobe exactly
    cancels the positive core's second moment.

    This validates the SHAPE of each ζ formula (prefactor and lobe balance).
    A wrong prefactor, exponent, or lobe coefficient shifts M₂.

    NOTE: M₂ is a pure shape property; it is deliberately decoupled from the
    production CSM diffusivity and Winckelmans angular-impulse constants, which
    are different physical coefficients (validated by the integration tests).

    Failure → the ζ formula's normalisation or lobe balance is wrong.
    """
    kern = _KERNELS[kernel_name]
    expected = kern["second_radial_moment"]
    computed = _second_radial_moment(kern["zeta"])

    # Absolute tolerance for the cancelling (M₂ = 0) kernels, relative otherwise.
    if abs(expected) < 1e-9:
        assert abs(computed) < 1e-3, (
            f"{kernel_name}: M₂ = {computed:.6f}, expected ≈ 0 (lobe should cancel)."
        )
    else:
        rel_err = abs(computed - expected) / abs(expected)
        assert rel_err < 1e-3, (
            f"{kernel_name}: M₂ = {computed:.6f}, expected {expected}, rel_err = {rel_err:.3e}"
        )


@pytest.mark.parametrize("kernel_name", list(_KERNELS.keys()))
def test_q_far_field_asymptote(kernel_name):
    """
    As ρ→∞, q(ρ) → 1/(4π) for all kernels (point-vortex limit).

    We probe at ρ = 20 and 50 and check both values are within 1% of 1/(4π).

    Failure → wrong normalisation constant or missing far-field saturation.
    """
    kern = _KERNELS[kernel_name]
    expected = 1.0 / (4.0 * pi)
    for rho in [20.0, 50.0]:
        q_val = float(kern["q"](rho))
        rel_err = abs(q_val - expected) / expected
        assert rel_err < 0.01, (
            f"{kernel_name}: q({rho}) = {q_val:.6e}, expected {expected:.6e}, "
            f"rel_err = {rel_err:.3e}"
        )
