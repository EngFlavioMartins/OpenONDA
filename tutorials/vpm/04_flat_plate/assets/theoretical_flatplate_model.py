#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Theoretical Model for Impulsively Started Flat Plate
====================================================

Computes the theoretical lift coefficient for an impulsively started flat plate
with a linear velocity ramp (0 → freestream_speed over nondimensional_ramp_time chord-lengths of travel),
followed by constant-velocity cruise.

The total CL is composed of two components:

1. CIRCULATORY LIFT (Duhamel / Jones 1940 finite-aspect_ratio Wagner)
   Uses the Jones approximation of the Wagner indicial function:
       phi(nondimensional_time) = 1 - A1*exp(-b1*nondimensional_time) - A2*exp(-b2*nondimensional_time)
   where (A1, b1, A2, b2) = (0.165, 0.0455, 0.335, 0.300)  [Jones 1940 / Theodorsen]
   The 3D finite-aspect_ratio lift slope finite_span_lift_curve_slope = 2*pi / (1 + 2/aspect_ratio) replaces a_2D = 2*pi.
   Duhamel superposition for a linear-ramp inflow history yields an
   analytically integrable form (see inline derivation below).

2. ADDED-MASS LIFT (non-circulatory, proportional to dU/dt)
   For a thin flat plate of chord c:
       L_am = rho * pi * (c/2)^2 * dU/dt * sin(alpha) * cos(alpha)
       C_L,am = (pi*c) / (2*freestream_speed^2) * (dU/dt) * sin(alpha) * cos(alpha)
   During the ramp: dU/dt = freestream_speed / t_ramp  → constant
   After the ramp:  dU/dt = 0

This script is fully parametrised by (angle_of_attack_degrees, aspect_ratio, nondimensional_ramp_time, tau_max, freestream_speed,
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


def _phi(nondimensional_time: np.ndarray) -> np.ndarray:
    """Jones (1940) approximation to the Wagner lift-deficiency function."""
    return 1.0 - _A1 * np.exp(-_b1 * nondimensional_time) - _A2 * np.exp(-_b2 * nondimensional_time)


def _duhamel_ramp_analytical(
    nondimensional_time: np.ndarray, nondimensional_ramp_time: float, steady_lift_coefficient: float
) -> np.ndarray:
    """
    Circulatory CL for a linear velocity ramp via Duhamel superposition,
    evaluated analytically using the Jones Wagner function.

    The normalised velocity history is:
        U*(sigma) = sigma / nondimensional_ramp_time   for 0 <= sigma <= nondimensional_ramp_time
        U*(sigma) = 1                  for sigma > nondimensional_ramp_time

    Duhamel integral (starts-from-rest form):
        CL(nondimensional_time) = steady_lift_coefficient * integral_0^{min(nondimensional_time, nondimensional_ramp_time)} phi(nondimensional_time-sigma)/nondimensional_ramp_time  d_sigma

    Analytical result for phi = 1 - A*exp(-b*nondimensional_time):
        integral_0^Sigma phi(nondimensional_time-sigma) d_sigma
          = Sigma - (A/b) * [exp(-b*(nondimensional_time-Sigma)) - exp(-b*nondimensional_time)]
    """

    def _int_phi(tau_arr, Sigma):
        """Integral of phi from 0 to Sigma, evaluated at array of nondimensional_time values."""
        result = Sigma * np.ones_like(tau_arr)
        result -= (_A1 / _b1) * (np.exp(-_b1 * (tau_arr - Sigma)) - np.exp(-_b1 * tau_arr))
        result -= (_A2 / _b2) * (np.exp(-_b2 * (tau_arr - Sigma)) - np.exp(-_b2 * tau_arr))
        return result

    cl = np.zeros_like(nondimensional_time)
    # During ramp: upper limit of Duhamel integral = nondimensional_time
    mask_ramp = nondimensional_time <= nondimensional_ramp_time
    if mask_ramp.any():
        cl[mask_ramp] = (steady_lift_coefficient / nondimensional_ramp_time) * _int_phi(
            nondimensional_time[mask_ramp], nondimensional_time[mask_ramp]
        )
    # After ramp: upper limit is fixed at nondimensional_ramp_time
    mask_post = ~mask_ramp
    if mask_post.any():
        cl[mask_post] = (steady_lift_coefficient / nondimensional_ramp_time) * _int_phi(
            nondimensional_time[mask_post], nondimensional_ramp_time
        )
    return cl


def compute_theoretical_lift_coefficient(
    angle_of_attack_degrees: float = 5.0,
    aspect_ratio: float = 8.0,
    nondimensional_ramp_time: float = 0.6,
    max_nondimensional_distance_travelled: float = 5.5,
    n_evaluation_points: int = 200,
    freestream_speed: float = 10.0,
    chord: float = 1.0,
    write_output: bool = True,
) -> pd.DataFrame:
    """
    Compute theoretical CL history for an impulsively started flat plate.

    Parameters
    ----------
    angle_of_attack_degrees : angle of attack [degrees]
    aspect_ratio        : full-span aspect ratio
    nondimensional_ramp_time  : chord-lengths of travel during the linear velocity ramp
    max_nondimensional_distance_travelled: total chord-lengths to compute
    n_evaluation_points: number of evaluation points
    freestream_speed     : final cruise speed [m/s]  (needed only for dimensional t_ramp)
    chord     : chord length [m]
    write_output: if True, write result to assets/test1_ref_data/

    Returns
    -------
    pd.DataFrame with columns: chords, cl_circulatory, cl_added_mass, total_lift_coefficient
    """
    angle_of_attack_radians = np.radians(angle_of_attack_degrees)
    ramp_time = 2.0 * nondimensional_ramp_time * chord / freestream_speed

    # -- 3D steady-state lift slope (Prandtl/Helmbold finite-aspect_ratio) --
    finite_span_lift_curve_slope = (2.0 * np.pi) / (1.0 + 2.0 / aspect_ratio)
    steady_lift_coefficient = finite_span_lift_curve_slope * np.sin(angle_of_attack_radians)

    # -- Evaluation grid (avoid nondimensional_time=0 to prevent log(0) issues) ---
    nondimensional_time = np.linspace(
        1e-4,
        max_nondimensional_distance_travelled,
        n_evaluation_points,
    )

    # -- Component 1: circulatory (Duhamel–Wagner) -----------------
    circulatory_lift_coefficient = _duhamel_ramp_analytical(
        nondimensional_time, nondimensional_ramp_time, steady_lift_coefficient
    )

    # -- Component 2: added mass (non-circulatory) -----------------
    # Constant during ramp, zero after
    ramp_added_mass_lift_coefficient = (
        (np.pi * chord / (2.0 * freestream_speed * ramp_time))
        * np.sin(angle_of_attack_radians)
        * np.cos(angle_of_attack_radians)
    )
    added_mass_lift_coefficient = np.where(
        nondimensional_time <= nondimensional_ramp_time,
        ramp_added_mass_lift_coefficient,
        0.0,
    )

    total_lift_coefficient = circulatory_lift_coefficient + added_mass_lift_coefficient

    df = pd.DataFrame(
        {
            "nondimensional_distance_travelled": nondimensional_time,
            "circulatory_lift_coefficient": circulatory_lift_coefficient,
            "added_mass_lift_coefficient": added_mass_lift_coefficient,
            "total_lift_coefficient": total_lift_coefficient,
        }
    )

    if write_output:
        tag = int(round(angle_of_attack_degrees))
        ref_dir = Path(__file__).parent / "test1_ref_data"
        ref_dir.mkdir(parents=True, exist_ok=True)
        out_csv = ref_dir / f"theoretical_total_aoa{tag:02d}.csv"
        df.to_csv(out_csv, index=False)
        print(
            f"  Saved {out_csv}  (alpha={angle_of_attack_degrees}°, aspect_ratio={aspect_ratio}, steady_lift_coefficient={steady_lift_coefficient:.4f}, "
            f"added_mass_lift_coefficient={ramp_added_mass_lift_coefficient:.4f})"
        )

    return df


def main():
    """Generate theory curves for all validation AoA values and save to CSV.

    Curve sets:
      • aspect_ratio=4  AoA=5°, 15°  — Beckwith & Babinsky reference geometry
      • aspect_ratio=5  AoA=10°      — matches tutorials/vpm/flat_plateVLM  (this tutorial)
    """
    print("Generating theoretical CL curves...")
    # Test1 reference curves (Beckwith & Babinsky geometry: aspect_ratio=4)
    for aoa in [5.0, 15.0]:
        compute_theoretical_lift_coefficient(
            angle_of_attack_degrees=aoa,
            aspect_ratio=4.0,
        )
    # Tutorial reference curve (flat_plateVLM: aspect_ratio=5, AoA=10°)
    compute_theoretical_lift_coefficient(
        angle_of_attack_degrees=10.0,
        aspect_ratio=5.0,
    )
    print("Done.")


if __name__ == "__main__":
    main()
