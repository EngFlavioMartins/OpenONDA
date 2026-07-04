#!/usr/bin/env python3
"""
Theoretical Models for Flat Plate VLM-VPM Validation
======================================================
Consolidates analytical reference functions for flat-plate validation cases.

Modules
-------
wagner_cl(tau, CL_ss)
    Quasi-steady lift buildup via Jones (1940) Wagner approximation.

impulsive_cl(tau, alpha_deg, AR, tau_ramp, chord, U_inf)
    Full impulsive-start CL (circulatory + added mass) for sin²-ramp motion.

theodorsen_C(k)
    Theodorsen lift-deficiency function C(k) = H₁²/[H₁² + i H₀²].

theodorsen_H(k)
    Full (H_total, H_circ, H_nc) complex transfer functions per unit amplitude.

prandtl_a3D(AR)
    Finite-AR lift slope: a_3D = 2π / (1 + 2/AR).

sinusoidal_fit(t, signal, omega)
    Least-squares fit of A0 + A1 sin(ωt) + A2 cos(ωt).

References
----------
Jones, R.T. (1940). NACA TN-682.
Theodorsen, T. (1935). NACA TR-496.
Wagner, H. (1925). ZfAM 5:17–35.
Prandtl, L. (1918). Tragflügeltheorie.

Author:  Flavio A. C. Martins, OpenONDA Team
Date: March 2026
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import hankel2

# --- Jones (1940) Wagner approximation constants -----------------------------
_A1, _b1 = 0.165, 0.0455  # slow exponential
_A2, _b2 = 0.335, 0.300  # fast exponential
# phi(tau=0) = 1 - 0.165 - 0.335 = 0.5  (correct starting value)


# =====================================================================
# Finite-AR lift slope
# =====================================================================


def prandtl_a3D(AR: float) -> float:
    """
    Prandtl finite-AR lift slope: a_3D = 2π / (1 + 2/AR).

    Parameters
    ----------
    AR : full-span aspect ratio

    Returns
    -------
    float: lift-curve slope [rad⁻¹]
    """
    return 2.0 * np.pi / (1.0 + 2.0 / AR)


# =====================================================================
# Wagner / impulsive-start theory
# =====================================================================


def _phi(tau: np.ndarray) -> np.ndarray:
    """Jones (1940) Wagner lift-deficiency function φ(τ)."""
    return 1.0 - _A1 * np.exp(-_b1 * tau) - _A2 * np.exp(-_b2 * tau)


def wagner_cl(tau: np.ndarray, CL_ss: float) -> np.ndarray:
    """
    Quasi-steady Wagner buildup for an impulsively started plate
    (instantaneous step change in velocity, no ramp).

    Parameters
    ----------
    tau   : non-dimensional time [chord-lengths]
    CL_ss : steady-state lift coefficient CL_ss = a_3D * sin(α)

    Returns
    -------
    np.ndarray: CL(τ) history
    """
    return CL_ss * _phi(np.asarray(tau, dtype=float))


def _int_phi(tau_arr: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """
    Integral of φ(τ-σ) from σ=0 to σ=Σ (analytic form for Jones approx).

    Integral = Σ − (A1/b1)[exp(−b1(τ−Σ)) − exp(−b1 τ)]
                 − (A2/b2)[exp(−b2(τ−Σ)) − exp(−b2 τ)]
    """
    result = np.array(Sigma, dtype=float, copy=True) * np.ones_like(tau_arr)
    result -= (_A1 / _b1) * (np.exp(-_b1 * (tau_arr - Sigma)) - np.exp(-_b1 * tau_arr))
    result -= (_A2 / _b2) * (np.exp(-_b2 * (tau_arr - Sigma)) - np.exp(-_b2 * tau_arr))
    return result


def impulsive_cl(
    tau: np.ndarray,
    alpha_deg: float,
    AR: float,
    tau_ramp: float,
    chord: float = 1.0,
    U_inf: float = 10.0,
) -> pd.DataFrame:
    """
    Full impulsive-start CL history for a sin²-ramp motion.

    The wing accelerates from rest via U(t) = 0.5·U∞·[1 − cos(πt/t_ramp)]
    over t ∈ [0, t_ramp], then cruises at constant velocity.

    Parameters
    ----------
    tau       : array of chord-lengths traveled (1D)
    alpha_deg : angle of attack [degrees]
    AR        : full-span aspect ratio
    tau_ramp  : ramp length [chord-lengths]
    chord     : chord [m]
    U_inf     : final cruise speed [m/s]

    Returns
    -------
    pd.DataFrame with columns:
        chords, CL_circulatory, CL_added_mass, CL_total
    """
    tau = np.asarray(tau, dtype=float)
    alpha = np.radians(alpha_deg)
    a_3D = prandtl_a3D(AR)
    CL_ss = a_3D * np.sin(alpha)
    t_ramp = 2.0 * tau_ramp * chord / U_inf  # sin² ramp duration [s]

    # -- Circulatory component: Duhamel superposition ------------------
    cl_circ = np.zeros_like(tau)
    mask_ramp = tau <= tau_ramp
    if mask_ramp.any():
        # During ramp: upper limit of Duhamel integral = tau
        cl_circ[mask_ramp] = (CL_ss / tau_ramp) * _int_phi(tau[mask_ramp], tau[mask_ramp])
    mask_post = ~mask_ramp
    if mask_post.any():
        # Post-ramp: upper limit fixed at tau_ramp
        cl_circ[mask_post] = (CL_ss / tau_ramp) * _int_phi(tau[mask_post], tau_ramp)

    # -- Added-mass component: ρ·(π/4)·c²·dU/dt -----------------------
    # For a thin flat plate: CL_am = (π·c)/(2·U²) · (dU/dt) · sin(α)·cos(α)
    # sin²-ramp: dU/dt = (π/2)·U_inf/t_ramp · sin(π·t/t_ramp)
    # Mapping to τ: t = τ·c/U_inf  → dU/dt = (π·U_inf)/(2·t_ramp) · sin(π·τ/tau_ramp)
    cl_am = np.zeros_like(tau)
    if mask_ramp.any():
        dU_dt_peak = np.pi * U_inf / (2.0 * t_ramp) * np.sin(np.pi * tau[mask_ramp] / tau_ramp)
        cl_am[mask_ramp] = (
            (np.pi * chord / (2.0 * U_inf**2)) * np.sin(alpha) * np.cos(alpha) * dU_dt_peak
        )

    cl_total = cl_circ + cl_am

    return pd.DataFrame(
        {
            "chords": tau,
            "CL_circulatory": cl_circ,
            "CL_added_mass": cl_am,
            "CL_total": cl_total,
        }
    )


# =====================================================================
# Theodorsen theory
# =====================================================================


def theodorsen_C(k: float) -> complex:
    """
    Theodorsen lift-deficiency function C(k).

    C(k) = H₁²(k) / [H₁²(k) + i · H₀²(k)]

    Parameters
    ----------
    k : reduced frequency k = ω·c / (2·U)

    Returns
    -------
    complex: C(k)
    """
    if k < 1e-8:
        return complex(1.0, 0.0)
    H1 = hankel2(1, k)
    H0 = hankel2(0, k)
    return H1 / (H1 + 1j * H0)


def theodorsen_H(k: float):
    """
    Complex lift transfer functions per unit pitch amplitude.

    Parameters
    ----------
    k : reduced frequency k = ω·c / (2·U)

    Returns
    -------
    (H_total, H_circ, H_nc) : complex transfer functions
        H_circ  = 2π · C(k) · (1 + ik)        — circulatory
        H_nc    = π · (ik − k²/2)               — non-circulatory (added-mass)
        H_total = H_circ + H_nc                 — total
    """
    C = theodorsen_C(k)
    H_circ = 2.0 * np.pi * C * (1.0 + 1j * k)
    H_nc = np.pi * (1j * k - 0.5 * k**2)
    H_total = H_circ + H_nc
    return H_total, H_circ, H_nc


def theodorsen_cl_timeseries(
    t: np.ndarray,
    alpha_amp_deg: float,
    k: float,
    U_inf: float = 10.0,
    chord: float = 1.0,
    AR: float = None,
) -> pd.DataFrame:
    """
    2D Theodorsen CL time series for sinusoidal pitch α(t) = α̂·sin(ωt).

    Parameters
    ----------
    t             : time array [s]
    alpha_amp_deg : pitch amplitude [degrees]
    k             : reduced frequency k = ω·c/(2·U)
    U_inf         : freestream speed [m/s]
    chord         : chord [m]
    AR            : if given, scales circulatory term by a_3D/(2π) for finite span

    Returns
    -------
    pd.DataFrame with columns: t, alpha_deg, CL_kj, CL_total
    """
    omega = 2.0 * k * U_inf / chord  # rad/s
    alpha_amp = np.radians(alpha_amp_deg)

    H_total, H_circ, H_nc = theodorsen_H(k)

    scale = 1.0
    if AR is not None:
        scale = prandtl_a3D(AR) / (2.0 * np.pi)

    CL_kj = alpha_amp * np.real(scale * H_circ * np.exp(1j * omega * t))
    CL_total = alpha_amp * np.real(
        scale * H_circ * np.exp(1j * omega * t) + H_nc * np.exp(1j * omega * t)
    )

    return pd.DataFrame(
        {
            "t": t,
            "alpha_deg": np.degrees(alpha_amp * np.sin(omega * t)),
            "CL_kj": CL_kj,
            "CL_total": CL_total,
        }
    )


# =====================================================================
# Signal fitting
# =====================================================================


def sinusoidal_fit(t: np.ndarray, signal: np.ndarray, omega: float):
    """
    Least-squares fit: y(t) ≈ A0 + A1·sin(ωt) + A2·cos(ωt).

    Parameters
    ----------
    t      : time array [s]
    signal : signal to fit
    omega  : angular frequency [rad/s]

    Returns
    -------
    (offset, amplitude, phase) : float, float, float
        amplitude = sqrt(A1² + A2²)
        phase     = arctan2(A2, A1)  [rad]
    """
    A = np.column_stack([np.ones_like(t), np.sin(omega * t), np.cos(omega * t)])
    coeffs, _, _, _ = np.linalg.lstsq(A, signal, rcond=None)
    amplitude = np.sqrt(coeffs[1] ** 2 + coeffs[2] ** 2)
    phase = np.arctan2(coeffs[2], coeffs[1])
    return float(coeffs[0]), float(amplitude), float(phase)


# =====================================================================
# Spanwise lift distribution — lifting-line theory
# =====================================================================


def liftingline_circulation(
    y: np.ndarray,
    b: float,
    c: float,
    alpha_rad: float,
    U_inf: float = 1.0,
    a0: float = 2.0 * np.pi,
    n_terms: int = 20,
) -> pd.DataFrame:
    """Prandtl/Glauert lifting-line spanwise circulation for a rectangular wing.

    Solves the monoplane equation for odd Fourier harmonics A_n at collocation
    points, then evaluates Γ(y), L'(y) = ρ U∞ Γ(y), and
    cl(y) = L'(y) / (q∞ c(y)).  The caller must supply ρ if absolute L'(y)
    is needed; this function returns the shape (ρ=1 by default).

    Parameters
    ----------
    y        : spanwise positions to evaluate [m], −b/2 … b/2
    b        : full span [m]
    c        : chord [m] (uniform rectangular wing)
    alpha_rad: angle of attack [rad]
    U_inf    : freestream speed [m/s]  (used for L' = ρ U∞ Γ; ρ=1 here)
    a0       : section lift-curve slope [1/rad] (default 2π for thin plate)
    n_terms  : number of odd Fourier harmonics (accuracy vs. cost)

    Returns
    -------
    pd.DataFrame with columns: y, y_over_b, Gamma, L_prime, cl
    """
    y = np.asarray(y, dtype=float)
    AR = b / c
    mu = a0 * c / (4.0 * b)  # Prandtl's mu parameter

    # collocation angles θ_k (exclude 0 and π to avoid singularities)
    k_vals = np.arange(1, n_terms + 1)
    theta_k = np.linspace(np.pi / (2 * n_terms + 2), np.pi - np.pi / (2 * n_terms + 2), n_terms)

    # Build linear system: Σ A_n sin(nθ) [sinθ + μ n] = μ α sinθ
    # (Anderson, "Fundamentals of Aerodynamics", Eq. 5.59)
    # Only odd harmonics contribute (symmetric loading for symmetric wing+α)
    n_odd = 2 * k_vals - 1  # 1, 3, 5, …
    sin_theta = np.sin(theta_k)
    M = np.zeros((n_terms, n_terms))
    for col_j, n in enumerate(n_odd):
        M[:, col_j] = np.sin(n * theta_k) * (sin_theta + mu * n)
    rhs = mu * alpha_rad * sin_theta

    A_n = np.linalg.solve(M, rhs)

    # Evaluate at requested y positions
    theta_y = np.arccos(np.clip(-2.0 * y / b, -1.0, 1.0))  # θ = arccos(−2y/b)
    Gamma_y = np.zeros_like(y)
    for col_j, n in enumerate(n_odd):
        Gamma_y += A_n[col_j] * np.sin(n * theta_y)
    Gamma_y *= 2.0 * b * U_inf

    q_inf = 0.5 * U_inf**2  # ρ=1
    L_prime_y = Gamma_y * U_inf  # ρ=1 → L'=ρ U∞ Γ
    cl_y = L_prime_y / (q_inf * c)  # = 2Γ/(U∞·c)

    return pd.DataFrame(
        {
            "y": y,
            "y_over_b": 2.0 * y / b,
            "Gamma": Gamma_y,
            "L_prime": L_prime_y,
            "cl": cl_y,
        }
    )


def elliptic_cl(y: np.ndarray, b: float, CL_total: float, AR: float) -> np.ndarray:
    """Sectional lift coefficient for an elliptic spanwise distribution.

    cl(y) = cl_root * sqrt(1 − (2y/b)²),  where cl_root integrates to CL_total.

    Parameters
    ----------
    y        : spanwise positions [m]
    b        : full span [m]
    CL_total : integrated lift coefficient
    AR       : aspect ratio (b²/S, used only to set the root value via CL_total)

    Returns
    -------
    np.ndarray: cl(y)
    """
    y = np.asarray(y, dtype=float)
    cl_root = 4.0 * CL_total / np.pi
    return cl_root * np.sqrt(np.clip(1.0 - (2.0 * y / b) ** 2, 0.0, None))


def spanwise_reference(
    kind: str,
    y: np.ndarray,
    b: float,
    c: float,
    alpha_rad: float,
    U_inf: float = 1.0,
    CL_total: float | None = None,
    AR: float | None = None,
    **kw,
) -> pd.DataFrame:
    """Dispatch spanwise cl(y) reference distribution.

    Parameters
    ----------
    kind      : 'liftingline' | 'elliptic' | 'parabolic'
    y, b, c, alpha_rad, U_inf : geometry / flow parameters
    CL_total  : required for 'elliptic' and 'parabolic' normalisation
    AR        : required for 'elliptic'

    Returns
    -------
    pd.DataFrame with at least columns: y, y_over_b, cl
    """
    y = np.asarray(y, dtype=float)
    if kind == "liftingline":
        return liftingline_circulation(y, b, c, alpha_rad, U_inf, **kw)
    elif kind == "elliptic":
        if CL_total is None or AR is None:
            raise ValueError("elliptic reference requires CL_total and AR")
        cl_y = elliptic_cl(y, b, CL_total, AR)
        return pd.DataFrame({"y": y, "y_over_b": 2.0 * y / b, "cl": cl_y})
    elif kind == "parabolic":
        if CL_total is None:
            raise ValueError("parabolic reference requires CL_total")
        # parabolic: cl(y) = cl_0 * (1 − (2y/b)²)  → ∫ cl dy = cl_0 * 2b/3
        cl_0 = 1.5 * CL_total  # normalise so ∫ cl dy = CL_total * c (via b*c = S)
        cl_y = cl_0 * (1.0 - (2.0 * y / b) ** 2)
        return pd.DataFrame({"y": y, "y_over_b": 2.0 * y / b, "cl": cl_y})
    else:
        raise ValueError(
            f"Unknown spanwise reference kind: '{kind}'. "
            "Choose 'liftingline', 'elliptic', or 'parabolic'."
        )


# =====================================================================
# Convenience export
# =====================================================================

__all__ = [
    "prandtl_a3D",
    "wagner_cl",
    "impulsive_cl",
    "theodorsen_C",
    "theodorsen_H",
    "theodorsen_cl_timeseries",
    "sinusoidal_fit",
    "liftingline_circulation",
    "elliptic_cl",
    "spanwise_reference",
]
