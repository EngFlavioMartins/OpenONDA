#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Theoretical Model for Impulsively Started Flat Plate
====================================================

Computes the theoretical lift coefficient for an impulsively started flat plate
with a linear velocity ramp (0 → U_inf over tau_ramp chord-lengths of travel),
followed by constant-velocity cruise.

The total CL is composed of two components:

1. CIRCULATORY LIFT (Duhamel / Jones 1940 finite-AR Wagner)
   Uses the Jones approximation of the Wagner indicial function:
       phi(tau) = 1 - A1*exp(-b1*tau) - A2*exp(-b2*tau)
   where (A1, b1, A2, b2) = (0.165, 0.0455, 0.335, 0.300)  [Jones 1940 / Theodorsen]
   The 3D finite-AR lift slope a_3D = 2*pi / (1 + 2/AR) replaces a_2D = 2*pi.
   Duhamel superposition for a linear-ramp inflow history yields an
   analytically integrable form (see inline derivation below).

2. ADDED-MASS LIFT (non-circulatory, proportional to dU/dt)
   For a thin flat plate of chord c:
       L_am = rho * pi * (c/2)^2 * dU/dt * sin(alpha) * cos(alpha)
       C_L,am = (pi*c) / (2*U_inf^2) * (dU/dt) * sin(alpha) * cos(alpha)
   During the ramp: dU/dt = U_inf / t_ramp  → constant
   After the ramp:  dU/dt = 0

This script is fully parametrised by (alpha_deg, AR, tau_ramp, tau_max, U_inf,
chord) and saves tagged output files `theoretical_total_aoa{nn}.csv`.

References
----------
Jones, R.T. (1940). The unsteady lift of a finite wing.  NACA TN-682.
Theodorsen, T. (1935). General theory of aerodynamic instability.  NACA TR-496.
Wagner, H. (1925). Über die Entstehung des dynamischen Auftriebes von
    Tragflügeln.  ZfAM 5:17–35.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# -- Jones (1940) Wagner approximation constants ------------------
_A1, _b1 = 0.165, 0.0455  # slow exponential
_A2, _b2 = 0.335, 0.300  # fast exponential
# phi(0) = 1 - 0.165 - 0.335 = 0.5  (non-circulatory origin of added mass)


def _phi(tau: np.ndarray) -> np.ndarray:
    """Jones (1940) approximation to the Wagner lift-deficiency function."""
    return 1.0 - _A1 * np.exp(-_b1 * tau) - _A2 * np.exp(-_b2 * tau)


def _duhamel_ramp_analytical(tau: np.ndarray, tau_ramp: float, CL_ss: float) -> np.ndarray:
    """
    Circulatory CL for a linear velocity ramp via Duhamel superposition,
    evaluated analytically using the Jones Wagner function.

    The normalised velocity history is:
        U*(sigma) = sigma / tau_ramp   for 0 <= sigma <= tau_ramp
        U*(sigma) = 1                  for sigma > tau_ramp

    Duhamel integral (starts-from-rest form):
        CL(tau) = CL_ss * integral_0^{min(tau, tau_ramp)} phi(tau-sigma)/tau_ramp  d_sigma

    Analytical result for phi = 1 - A*exp(-b*tau):
        integral_0^Sigma phi(tau-sigma) d_sigma
          = Sigma - (A/b) * [exp(-b*(tau-Sigma)) - exp(-b*tau)]
    """

    def _int_phi(tau_arr, Sigma):
        """Integral of phi from 0 to Sigma, evaluated at array of tau values."""
        result = Sigma * np.ones_like(tau_arr)
        result -= (_A1 / _b1) * (np.exp(-_b1 * (tau_arr - Sigma)) - np.exp(-_b1 * tau_arr))
        result -= (_A2 / _b2) * (np.exp(-_b2 * (tau_arr - Sigma)) - np.exp(-_b2 * tau_arr))
        return result

    cl = np.zeros_like(tau)
    # During ramp: upper limit of Duhamel integral = tau
    mask_ramp = tau <= tau_ramp
    if mask_ramp.any():
        cl[mask_ramp] = (CL_ss / tau_ramp) * _int_phi(tau[mask_ramp], tau[mask_ramp])
    # After ramp: upper limit is fixed at tau_ramp
    mask_post = ~mask_ramp
    if mask_post.any():
        cl[mask_post] = (CL_ss / tau_ramp) * _int_phi(tau[mask_post], tau_ramp)
    return cl


def compute_theoretical_cl(
    alpha_deg: float = 5.0,
    AR: float = 8.0,
    tau_ramp: float = 0.6,
    tau_max: float = 5.5,
    n_points: int = 200,
    U_inf: float = 10.0,
    chord: float = 1.0,
    save: bool = True,
) -> pd.DataFrame:
    """
    Compute theoretical CL history for an impulsively started flat plate.

    Parameters
    ----------
    alpha_deg : angle of attack [degrees]
    AR        : full-span aspect ratio
    tau_ramp  : chord-lengths of travel during the linear velocity ramp
    tau_max   : total chord-lengths of travel to compute
    n_points  : number of evaluation points
    U_inf     : final cruise speed [m/s]  (needed only for dimensional t_ramp)
    chord     : chord length [m]
    save      : if True, write result to assets/test1_ref_data/

    Returns
    -------
    pd.DataFrame with columns: chords, cl_circulatory, cl_added_mass, cl_total
    """
    alpha_rad = np.radians(alpha_deg)
    t_ramp = 2.0 * tau_ramp * chord / U_inf  # ramp duration [s]

    # -- 3D steady-state lift slope (Prandtl/Helmbold finite-AR) --
    a_3D = (2.0 * np.pi) / (1.0 + 2.0 / AR)
    CL_ss = a_3D * np.sin(alpha_rad)

    # -- Evaluation grid (avoid tau=0 to prevent log(0) issues) ---
    tau = np.linspace(1e-4, tau_max, n_points)

    # -- Component 1: circulatory (Duhamel–Wagner) -----------------
    cl_circ = _duhamel_ramp_analytical(tau, tau_ramp, CL_ss)

    # -- Component 2: added mass (non-circulatory) -----------------
    # Constant during ramp, zero after
    cl_am_ramp = (np.pi * chord / (2.0 * U_inf * t_ramp)) * np.sin(alpha_rad) * np.cos(alpha_rad)
    cl_am = np.where(tau <= tau_ramp, cl_am_ramp, 0.0)

    cl_total = cl_circ + cl_am

    df = pd.DataFrame(
        {
            "chords": tau,
            "cl_circulatory": cl_circ,
            "cl_added_mass": cl_am,
            "cl_total": cl_total,
        }
    )

    if save:
        tag = int(round(alpha_deg))
        ref_dir = Path(__file__).parent / "test1_ref_data"
        ref_dir.mkdir(parents=True, exist_ok=True)
        out_csv = ref_dir / f"theoretical_total_aoa{tag:02d}.csv"
        df.to_csv(out_csv, index=False)
        print(
            f"  Saved {out_csv}  (alpha={alpha_deg}°, AR={AR}, CL_ss={CL_ss:.4f}, "
            f"CL_am={cl_am_ramp:.4f})"
        )

    return df


def main():
    """Generate theory curves for all validation AoA values and save to CSV.

    Curve sets:
      • AR=4  AoA=5°, 15°  — Beckwith & Babinsky reference geometry
      • AR=5  AoA=10°      — matches tutorials/VPM/flatPlateVLM  (this tutorial)
    """
    print("Generating theoretical CL curves...")
    # Test1 reference curves (Beckwith & Babinsky geometry: AR=4)
    for aoa in [5.0, 15.0]:
        compute_theoretical_cl(alpha_deg=aoa, AR=4.0)
    # Tutorial reference curve (flatPlateVLM: AR=5, AoA=10°)
    compute_theoretical_cl(alpha_deg=10.0, AR=5.0)
    print("Done.")


if __name__ == "__main__":
    main()
