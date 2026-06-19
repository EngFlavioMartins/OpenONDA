"""
Rotor reference theory for VLM+VPM validation.

Provides a BEM solver matched to the flat-plate VLM polar (Cl = 2π sinα)
so the comparison is apples-to-apples, plus actuator-disk momentum theory
for far-wake velocity-deficit validation.

Functions
---------
bem_solve(r, chord, twist_rad, B, R, U_inf, omega, Cl_slope, n_iter, tol)
    Classic iterative BEM with Glauert-corrected Prandtl tip-loss.
    Returns DataFrame with spanwise aerodynamic quantities and Ct, Cp.

actuator_disk_deficit(a, r_over_R)
    Far-wake axial velocity from actuator-disk momentum theory: u/U = 1 − 2a.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: June 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def bem_solve(
    r: np.ndarray,
    chord: np.ndarray,
    twist_rad: np.ndarray,
    B: int,
    R: float,
    U_inf: float,
    omega: float,
    Cl_slope: float = 2.0 * np.pi,
    Cd_const: float = 0.0,
    n_iter: int = 200,
    tol: float = 1e-7,
) -> pd.DataFrame:
    """Iterative BEM with Prandtl tip-loss, matched to a flat-plate lift polar.

    Parameters
    ----------
    r          : radial stations [m], shape (N,), must satisfy hub_r < r < R
    chord      : local chord at each station [m], shape (N,)
    twist_rad  : local geometric pitch angle [rad] (positive → LE into wind)
    B          : number of blades
    R          : rotor tip radius [m]
    U_inf      : axial freestream velocity [m/s]
    omega      : rotor angular velocity [rad/s]
    Cl_slope   : section lift-curve slope [1/rad] (default 2π for thin plate)
    Cd_const   : section drag coefficient (constant; default 0 for flat-plate VLM)
    n_iter     : max BEM iterations per station
    tol        : convergence tolerance on induction factor a

    Returns
    -------
    pd.DataFrame with columns:
        r, r_over_R, chord, twist_deg, phi_deg,
        alpha_deg, Cl, Cd, a, a_prime,
        dT_dr [N/m], dQ_dr [N·m/m],
        Gamma [m²/s]      (bound circulation Γ = ½ c Cl V_rel)
    And scalar attributes stored as df.attrs:
        Ct, Cp, T [N], Q [N·m], P [W]
    """
    r = np.asarray(r, dtype=float)
    chord = np.asarray(chord, dtype=float)
    twist_rad = np.asarray(twist_rad, dtype=float)
    N = len(r)

    a_arr = np.zeros(N)
    ap_arr = np.zeros(N)
    phi_arr = np.zeros(N)
    alpha_arr = np.zeros(N)
    Cl_arr = np.zeros(N)
    Cd_arr = np.full(N, Cd_const)
    dT_arr = np.zeros(N)
    dQ_arr = np.zeros(N)
    Gamma_arr = np.zeros(N)

    for k in range(N):
        rk = r[k]
        ck = chord[k]
        twist_k = twist_rad[k]
        mu = rk / R  # r/R

        a = 1.0 / 3.0  # initial guess (Betz optimum)
        ap = 0.0

        for _ in range(n_iter):
            # Relative flow angle
            Vax = U_inf * (1.0 - a)
            Vtan = omega * rk * (1.0 + ap)
            if abs(Vax) < 1e-12 and abs(Vtan) < 1e-12:
                break
            phi = np.arctan2(Vax, Vtan)

            # Prandtl tip-loss factor (Glauert formulation)
            f_arg = (B / 2.0) * (1.0 - mu) / (mu * abs(np.sin(phi)) + 1e-14)
            f_arg = np.clip(f_arg, 0.0, 50.0)
            F = (2.0 / np.pi) * np.arccos(np.exp(-f_arg))
            F = max(F, 1e-6)

            # Section aerodynamics (flat-plate polar)
            alpha = phi - twist_k
            Cl = Cl_slope * np.sin(alpha)
            Cd = Cd_const

            # Solidity at this station
            sigma = B * ck / (2.0 * np.pi * rk)

            # Thrust and torque coefficients (Glauert BEM)
            Cn = Cl * np.cos(phi) + Cd * np.sin(phi)
            Ct_sec = Cl * np.sin(phi) - Cd * np.cos(phi)

            denom_a = 4.0 * F * np.sin(phi) ** 2
            a_new = 1.0 / (denom_a / (sigma * Cn) + 1.0)

            denom_ap = 4.0 * F * np.sin(phi) * np.cos(phi)
            ap_new = 1.0 / (denom_ap / (sigma * Ct_sec) - 1.0)
            ap_new = max(ap_new, 0.0)

            if abs(a_new - a) < tol and abs(ap_new - ap) < tol:
                a = a_new
                ap = ap_new
                break
            a = a_new
            ap = ap_new

        # Converged — store results
        Vax = U_inf * (1.0 - a)
        Vtan = omega * rk * (1.0 + ap)
        V_rel = np.sqrt(Vax**2 + Vtan**2)
        phi = np.arctan2(Vax, Vtan)
        alpha = phi - twist_rad[k]
        Cl = Cl_slope * np.sin(alpha)
        Gamma = 0.5 * chord[k] * Cl * V_rel

        rho = 1.0  # caller multiplies by rho if needed; BEM force is per unit ρ here
        # Use standard BEM rotor-plane element forces
        # dT = ½ ρ V_rel² c Cn B dr  →  stored as per-unit-ρ
        Cn = Cl * np.cos(phi) + Cd_const * np.sin(phi)
        Ct_sec = Cl * np.sin(phi) - Cd_const * np.cos(phi)
        dT = 0.5 * V_rel**2 * chord[k] * Cn * B
        dQ = 0.5 * V_rel**2 * chord[k] * Ct_sec * B * rk

        a_arr[k] = a
        ap_arr[k] = ap
        phi_arr[k] = phi
        alpha_arr[k] = alpha
        Cl_arr[k] = Cl
        dT_arr[k] = dT
        dQ_arr[k] = dQ
        Gamma_arr[k] = Gamma

    # Integrate thrust and torque
    T = float(np.trapezoid(dT_arr, r))
    Q = float(np.trapezoid(dQ_arr, r))
    P = Q * omega
    A = np.pi * R**2
    q_inf = 0.5 * U_inf**2  # per unit ρ
    Ct = T / (q_inf * A)
    Cp = P / (q_inf * A * U_inf)

    df = pd.DataFrame(
        {
            "r": r,
            "r_over_R": r / R,
            "chord": chord,
            "twist_deg": np.degrees(twist_rad),
            "phi_deg": np.degrees(phi_arr),
            "alpha_deg": np.degrees(alpha_arr),
            "Cl": Cl_arr,
            "Cd": Cd_arr,
            "a": a_arr,
            "a_prime": ap_arr,
            "dT_dr": dT_arr,
            "dQ_dr": dQ_arr,
            "Gamma": Gamma_arr,
        }
    )
    df.attrs["Ct"] = Ct
    df.attrs["Cp"] = Cp
    df.attrs["T"] = T
    df.attrs["Q"] = Q
    df.attrs["P"] = P

    return df


def actuator_disk_deficit(
    a: float,
    r_over_R: np.ndarray,
    R: float = 1.0,
) -> np.ndarray:
    """Far-wake axial velocity deficit from actuator-disk momentum theory.

    In the far wake (x → ∞), u/U∞ = 1 − 2a uniformly within the wake disc.
    Outside the disc (|r/R| > 1) the deficit vanishes.

    Parameters
    ----------
    a       : axial induction factor (scalar, from BEM Ct via 1−√(1−Ct) / 2)
    r_over_R: radial positions normalised by rotor radius
    R       : rotor radius [m] (used only if r_over_R is already physical r)

    Returns
    -------
    u_over_Uinf : np.ndarray, same shape as r_over_R
    """
    r_over_R = np.asarray(r_over_R, dtype=float)
    u_over_Uinf = np.ones_like(r_over_R)
    inside = np.abs(r_over_R) <= 1.0
    u_over_Uinf[inside] = 1.0 - 2.0 * a
    return u_over_Uinf


def induction_from_ct(Ct: float) -> float:
    """Axial induction factor from Ct via momentum theory: 1 − √(1−Ct) / 2."""
    if Ct >= 1.0:
        return 0.5  # turbulent wake state (momentum theory invalid)
    return 0.5 * (1.0 - np.sqrt(max(0.0, 1.0 - Ct)))


__all__ = [
    "bem_solve",
    "actuator_disk_deficit",
    "induction_from_ct",
]
