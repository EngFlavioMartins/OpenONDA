#!/usr/bin/env python3
"""
Theoretical Models for Flat Plate VLM-VPM Validation
======================================================
Consolidates analytical reference functions for flat-plate validation cases.

Modules
-------
wagner_lift_coefficient(nondimensional_time, steady_lift_coefficient)
    Quasi-steady lift buildup via Jones (1940) Wagner approximation.

impulsive_lift_coefficient(nondimensional_time, angle_of_attack_degrees, aspect_ratio, nondimensional_ramp_time, chord, freestream_speed)
    Full impulsive-start CL (circulatory + added mass) for sin²-ramp motion.

theodorsen_function(k)
    Theodorsen lift-deficiency function C(k) = H₁²/[H₁² + i H₀²].

theodorsen_lift_transfer_functions(k)
    Full (H_total, H_circ, H_nc) complex transfer functions per unit amplitude.

prandtl_finite_span_lift_curve_slope(aspect_ratio)
    Finite-aspect_ratio lift slope: finite_span_lift_curve_slope = 2π / (1 + 2/aspect_ratio).

sinusoidal_fit(t, signal, angular_frequency)
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
# phi(nondimensional_time=0) = 1 - 0.165 - 0.335 = 0.5  (correct starting value)


# =====================================================================
# Finite-aspect_ratio lift slope
# =====================================================================


def prandtl_finite_span_lift_curve_slope(aspect_ratio: float) -> float:
    """
    Prandtl finite-aspect_ratio lift slope: finite_span_lift_curve_slope = 2π / (1 + 2/aspect_ratio).

    Parameters
    ----------
    aspect_ratio : full-span aspect ratio

    Returns
    -------
    float: lift-curve slope [rad⁻¹]
    """
    return 2.0 * np.pi / (1.0 + 2.0 / aspect_ratio)


# =====================================================================
# Wagner / impulsive-start theory
# =====================================================================


def _phi(nondimensional_time: np.ndarray) -> np.ndarray:
    """Jones (1940) Wagner lift-deficiency function φ(τ)."""
    return 1.0 - _A1 * np.exp(-_b1 * nondimensional_time) - _A2 * np.exp(-_b2 * nondimensional_time)


def wagner_lift_coefficient(
    nondimensional_time: np.ndarray, steady_lift_coefficient: float
) -> np.ndarray:
    """
    Quasi-steady Wagner buildup for an impulsively started plate
    (instantaneous step change in velocity, no ramp).

    Parameters
    ----------
    nondimensional_time   : non-dimensional time [chord-lengths]
    steady_lift_coefficient : steady-state lift coefficient steady_lift_coefficient = finite_span_lift_curve_slope * sin(α)

    Returns
    -------
    np.ndarray: CL(τ) history
    """
    return steady_lift_coefficient * _phi(np.asarray(nondimensional_time, dtype=float))


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


def impulsive_lift_coefficient(
    nondimensional_time: np.ndarray,
    angle_of_attack_degrees: float,
    aspect_ratio: float,
    nondimensional_ramp_time: float,
    chord: float = 1.0,
    freestream_speed: float = 10.0,
) -> pd.DataFrame:
    """
    Full impulsive-start CL history for a sin²-ramp motion.

    The wing accelerates from rest via U(t) = 0.5·U∞·[1 − cos(πt/t_ramp)]
    over t ∈ [0, t_ramp], then cruises at constant velocity.

    Parameters
    ----------
    nondimensional_time       : array of chord-lengths traveled (1D)
    angle_of_attack_degrees : angle of attack [degrees]
    aspect_ratio        : full-span aspect ratio
    nondimensional_ramp_time  : ramp length [chord-lengths]
    chord     : chord [m]
    freestream_speed     : final cruise speed [m/s]

    Returns
    -------
    pd.DataFrame with columns:
        chords, circulatory_lift_coefficient, added_mass_lift_coefficient, total_lift_coefficient
    """
    nondimensional_time = np.asarray(nondimensional_time, dtype=float)
    angle_of_attack_radians = np.radians(angle_of_attack_degrees)
    finite_span_lift_curve_slope = prandtl_finite_span_lift_curve_slope(aspect_ratio)
    steady_lift_coefficient = finite_span_lift_curve_slope * np.sin(angle_of_attack_radians)
    ramp_time = 2.0 * nondimensional_ramp_time * chord / freestream_speed

    # -- Circulatory component: Duhamel superposition ------------------
    circulatory_lift_coefficient = np.zeros_like(nondimensional_time)
    mask_ramp = nondimensional_time <= nondimensional_ramp_time
    if mask_ramp.any():
        # During ramp: upper limit of Duhamel integral = nondimensional_time
        circulatory_lift_coefficient[mask_ramp] = (
            steady_lift_coefficient / nondimensional_ramp_time
        ) * _int_phi(nondimensional_time[mask_ramp], nondimensional_time[mask_ramp])
    mask_post = ~mask_ramp
    if mask_post.any():
        # Post-ramp: upper limit fixed at nondimensional_ramp_time
        circulatory_lift_coefficient[mask_post] = (
            steady_lift_coefficient / nondimensional_ramp_time
        ) * _int_phi(nondimensional_time[mask_post], nondimensional_ramp_time)

    # -- Added-mass component: ρ·(π/4)·c²·dU/dt -----------------------
    # For a thin flat plate: CL_am = (π·c)/(2·U²) · (dU/dt) · sin(α)·cos(α)
    # sin²-ramp: dU/dt = (π/2)·freestream_speed/t_ramp · sin(π·t/t_ramp)
    # Mapping to τ: t = τ·c/freestream_speed  → dU/dt = (π·freestream_speed)/(2·t_ramp) · sin(π·τ/nondimensional_ramp_time)
    added_mass_lift_coefficient = np.zeros_like(nondimensional_time)
    if mask_ramp.any():
        peak_acceleration = (
            np.pi
            * freestream_speed
            / (2.0 * ramp_time)
            * np.sin(np.pi * nondimensional_time[mask_ramp] / nondimensional_ramp_time)
        )
        added_mass_lift_coefficient[mask_ramp] = (
            (np.pi * chord / (2.0 * freestream_speed**2))
            * np.sin(angle_of_attack_radians)
            * np.cos(angle_of_attack_radians)
            * peak_acceleration
        )

    total_lift_coefficient = circulatory_lift_coefficient + added_mass_lift_coefficient

    return pd.DataFrame(
        {
            "nondimensional_distance_travelled": nondimensional_time,
            "circulatory_lift_coefficient": circulatory_lift_coefficient,
            "added_mass_lift_coefficient": added_mass_lift_coefficient,
            "total_lift_coefficient": total_lift_coefficient,
        }
    )


# =====================================================================
# Theodorsen theory
# =====================================================================


def theodorsen_function(reduced_frequency: float) -> complex:
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
    if reduced_frequency < 1e-8:
        return complex(1.0, 0.0)
    H1 = hankel2(1, reduced_frequency)
    H0 = hankel2(0, reduced_frequency)
    return H1 / (H1 + 1j * H0)


def theodorsen_lift_transfer_functions(reduced_frequency: float):
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
    C = theodorsen_function(reduced_frequency)
    H_circ = 2.0 * np.pi * C * (1.0 + 1j * reduced_frequency)
    H_nc = np.pi * (1j * reduced_frequency - 0.5 * reduced_frequency**2)
    H_total = H_circ + H_nc
    return H_total, H_circ, H_nc


def theodorsen_lift_coefficient_time_series(
    time: np.ndarray,
    angle_of_attack_amplitude_degrees: float,
    reduced_frequency: float,
    freestream_speed: float = 10.0,
    chord: float = 1.0,
    aspect_ratio: float | None = None,
) -> pd.DataFrame:
    """
    2D Theodorsen CL time series for sinusoidal pitch α(t) = α̂·sin(ωt).

    Parameters
    ----------
    t             : time array [s]
    angle_of_attack_amplitude_degrees : pitch amplitude [degrees]
    k             : reduced frequency k = ω·c/(2·U)
    freestream_speed         : freestream speed [m/s]
    chord         : chord [m]
    aspect_ratio            : if given, scales circulatory term by finite_span_lift_curve_slope/(2π) for finite span

    Returns
    -------
    pd.DataFrame with columns: t, angle_of_attack_degrees, kutta_joukowski_lift_coefficient, total_lift_coefficient
    """
    angular_frequency = 2.0 * reduced_frequency * freestream_speed / chord
    angle_of_attack_amplitude = np.radians(angle_of_attack_amplitude_degrees)

    H_total, H_circ, H_nc = theodorsen_lift_transfer_functions(reduced_frequency)

    scale = 1.0
    if aspect_ratio is not None:
        scale = prandtl_finite_span_lift_curve_slope(aspect_ratio) / (2.0 * np.pi)

    kutta_joukowski_lift_coefficient = angle_of_attack_amplitude * np.real(
        scale * H_circ * np.exp(1j * angular_frequency * time)
    )
    total_lift_coefficient = angle_of_attack_amplitude * np.real(
        scale * H_circ * np.exp(1j * angular_frequency * time)
        + H_nc * np.exp(1j * angular_frequency * time)
    )

    return pd.DataFrame(
        {
            "time": time,
            "angle_of_attack_degrees": np.degrees(
                angle_of_attack_amplitude * np.sin(angular_frequency * time)
            ),
            "kutta_joukowski_lift_coefficient": kutta_joukowski_lift_coefficient,
            "total_lift_coefficient": total_lift_coefficient,
        }
    )


# =====================================================================
# Signal fitting
# =====================================================================


def sinusoidal_fit(time: np.ndarray, signal: np.ndarray, angular_frequency: float):
    """
    Least-squares fit: y(t) ≈ A0 + A1·sin(ωt) + A2·cos(ωt).

    Parameters
    ----------
    t      : time array [s]
    signal : signal to fit
    angular_frequency  : angular frequency [rad/s]

    Returns
    -------
    (offset, amplitude, phase) : float, float, float
        amplitude = sqrt(A1² + A2²)
        phase     = arctan2(A2, A1)  [rad]
    """
    A = np.column_stack(
        [
            np.ones_like(time),
            np.sin(angular_frequency * time),
            np.cos(angular_frequency * time),
        ]
    )
    coeffs, _, _, _ = np.linalg.lstsq(A, signal, rcond=None)
    amplitude = np.sqrt(coeffs[1] ** 2 + coeffs[2] ** 2)
    phase = np.arctan2(coeffs[2], coeffs[1])
    return float(coeffs[0]), float(amplitude), float(phase)


# =====================================================================
# Spanwise lift distribution — lifting-line theory
# =====================================================================


def lifting_line_circulation(
    span_position: np.ndarray,
    reference_span: float,
    reference_chord: float,
    angle_of_attack_radians: float,
    freestream_speed: float = 1.0,
    two_dimensional_lift_curve_slope: float = 2.0 * np.pi,
    n_fourier_terms: int = 20,
) -> pd.DataFrame:
    """Prandtl/Glauert lifting-line spanwise circulation for a rectangular wing.

    Solves the monoplane equation for odd Fourier harmonics fourier_coefficient at collocation
    points, then evaluates Γ(y), L'(y) = ρ U∞ Γ(y), and
    cl(y) = L'(y) / (q∞ c(y)).  The caller must supply ρ if absolute L'(y)
    is needed; this function returns the shape (ρ=1 by default).

    Parameters
    ----------
    y        : spanwise position to evaluate [m], −b/2 … b/2
    b        : full span [m]
    c        : chord [m] (uniform rectangular wing)
    angle_of_attack_radians: angle of attack [rad]
    freestream_speed    : freestream speed [m/s]  (used for L' = ρ U∞ Γ; ρ=1 here)
    two_dimensional_lift_curve_slope       : section lift-curve slope [1/rad] (default 2π for thin plate)
    n_terms  : number of odd Fourier harmonics (accuracy vs. cost)

    Returns
    -------
    pd.DataFrame with columns: y, y_over_b, circulation, lift_per_span, cl
    """
    span_position = np.asarray(span_position, dtype=float)
    aspect_ratio = reference_span / reference_chord
    mu = two_dimensional_lift_curve_slope * reference_chord / (4.0 * reference_span)

    # collocation angles θ_k (exclude 0 and π to avoid singularities)
    k_vals = np.arange(1, n_fourier_terms + 1)
    theta_k = np.linspace(
        np.pi / (2 * n_fourier_terms + 2),
        np.pi - np.pi / (2 * n_fourier_terms + 2),
        n_fourier_terms,
    )

    # Build linear system: Σ fourier_coefficient sin(nθ) [sinθ + μ n] = μ α sinθ
    # (Anderson, "Fundamentals of Aerodynamics", Eq. 5.59)
    # Only odd harmonics contribute (symmetric loading for symmetric wing+α)
    n_odd = 2 * k_vals - 1  # 1, 3, 5, …
    sin_theta = np.sin(theta_k)
    M = np.zeros((n_fourier_terms, n_fourier_terms))
    for col_j, n in enumerate(n_odd):
        M[:, col_j] = np.sin(n * theta_k) * (sin_theta + mu * n)
    rhs = mu * angle_of_attack_radians * sin_theta

    fourier_coefficient = np.linalg.solve(M, rhs)

    # Evaluate at requested y position
    theta_y = np.arccos(np.clip(-2.0 * span_position / reference_span, -1.0, 1.0))
    circulation_y = np.zeros_like(span_position)
    for col_j, n in enumerate(n_odd):
        circulation_y += fourier_coefficient[col_j] * np.sin(n * theta_y)
    circulation_y *= 2.0 * reference_span * freestream_speed

    q_inf = 0.5 * freestream_speed**2  # ρ=1
    L_prime_y = circulation_y * freestream_speed  # ρ=1 → L'=ρ U∞ Γ
    section_lift_coefficient = L_prime_y / (q_inf * reference_chord)

    return pd.DataFrame(
        {
            "span_coordinate": span_position,
            "span_coordinate_normalized": 2.0 * span_position / reference_span,
            "circulation": circulation_y,
            "lift_per_span": L_prime_y,
            "section_lift_coefficient": section_lift_coefficient,
        }
    )


def elliptic_section_lift_coefficient(
    span_position: np.ndarray,
    reference_span: float,
    total_lift_coefficient: float,
    aspect_ratio: float,
) -> np.ndarray:
    """Sectional lift coefficient for an elliptic spanwise distribution.

    cl(y) = root_section_lift_coefficient * sqrt(1 − (2y/b)²),  where root_section_lift_coefficient integrates to total_lift_coefficient.

    Parameters
    ----------
    y        : spanwise position [m]
    b        : full span [m]
    total_lift_coefficient : integrated lift coefficient
    aspect_ratio       : aspect ratio (b²/S, used only to set the root value via total_lift_coefficient)

    Returns
    -------
    np.ndarray: cl(y)
    """
    span_position = np.asarray(span_position, dtype=float)
    root_section_lift_coefficient = 4.0 * total_lift_coefficient / np.pi
    return root_section_lift_coefficient * np.sqrt(
        np.clip(1.0 - (2.0 * span_position / reference_span) ** 2, 0.0, None)
    )


def spanwise_reference(
    distribution_model: str,
    span_position: np.ndarray,
    reference_span: float,
    reference_chord: float,
    angle_of_attack_radians: float,
    freestream_speed: float = 1.0,
    total_lift_coefficient: float | None = None,
    aspect_ratio: float | None = None,
    **model_options,
) -> pd.DataFrame:
    """Dispatch spanwise cl(y) reference distribution.

    Parameters
    ----------
    kind      : 'liftingline' | 'elliptic' | 'parabolic'
    y, b, c, angle_of_attack_radians, freestream_speed : geometry / flow parameters
    total_lift_coefficient  : required for 'elliptic' and 'parabolic' normalisation
    aspect_ratio        : required for 'elliptic'

    Returns
    -------
    pd.DataFrame with at least columns: y, y_over_b, cl
    """
    span_position = np.asarray(span_position, dtype=float)
    if distribution_model == "lifting_line":
        return lifting_line_circulation(
            span_position,
            reference_span,
            reference_chord,
            angle_of_attack_radians,
            freestream_speed,
            **model_options,
        )
    elif distribution_model == "elliptic":
        if total_lift_coefficient is None or aspect_ratio is None:
            raise ValueError("elliptic reference requires total_lift_coefficient and aspect_ratio")
        section_lift_coefficient = elliptic_section_lift_coefficient(
            span_position,
            reference_span,
            total_lift_coefficient,
            aspect_ratio,
        )
    elif distribution_model == "parabolic":
        if total_lift_coefficient is None:
            raise ValueError("parabolic reference requires total_lift_coefficient")
        # parabolic: cl(y) = centre_section_lift_coefficient * (1 − (2y/b)²)  → ∫ cl dy = centre_section_lift_coefficient * 2b/3
        centre_section_lift_coefficient = (
            1.5 * total_lift_coefficient
        )  # normalise so ∫ cl dy = total_lift_coefficient * c (via b*c = S)
        section_lift_coefficient = centre_section_lift_coefficient * (
            1.0 - (2.0 * span_position / reference_span) ** 2
        )
    else:
        raise ValueError(
            f"Unknown spanwise reference model: '{distribution_model}'. "
            "Choose 'lifting_line', 'elliptic', or 'parabolic'."
        )
    return pd.DataFrame(
        {
            "span_coordinate": span_position,
            "span_coordinate_normalized": 2.0 * span_position / reference_span,
            "section_lift_coefficient": section_lift_coefficient,
        }
    )


# =====================================================================
# Convenience export
# =====================================================================

__all__ = [
    "prandtl_finite_span_lift_curve_slope",
    "wagner_lift_coefficient",
    "impulsive_lift_coefficient",
    "theodorsen_function",
    "theodorsen_lift_transfer_functions",
    "theodorsen_lift_coefficient_time_series",
    "sinusoidal_fit",
    "lifting_line_circulation",
    "elliptic_section_lift_coefficient",
    "spanwise_reference",
]
