"""Finite-distance right-vortex-cylinder reference for the authored rotor.

The elementary-cylinder influence functions follow Eqs. (30)--(45) of
Li et al., *Wind Energy Science* 10, 2515--2535 (2025), which summarize the
right-cylinder equations of Branlard & Gaunaa (2015).  This module evaluates
the axial and longitudinal-sheet tangential components needed by the
diagnostic; it does not claim the omitted radial component.  This is an
analytical axisymmetric reference, not a reconstruction of particle fields or
a tuned fit to the VPM output.

The reference assumes a planar, non-yawed, non-tilted rotor, an inviscid
non-expanding right-cylinder wake, infinite blade count, piecewise-constant
bound circulation, and constant far-wake convection.  The authored finite
blade, LES, expanding/mixing wake is therefore compared as a model diagnostic;
failure does not identify a solver defect by itself.  The BEM annulus
induction used to close the cylinder is the attached-flow reference already
used by the tutorial.  High-thrust annuli (``C_T >= 1``) are rejected instead
of silently applying a different correction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import special

# Predeclared diagnostic gates.  They are intentionally broader than numerical
# round-off because the reference is infinite-blade/inviscid/non-expanding,
# while the case is finite-blade, viscous/LES and time-dependent.  They must be
# reported unchanged, not adjusted after seeing a production result.
AXIAL_RELATIVE_TOLERANCE = 0.25
TANGENTIAL_RELATIVE_TOLERANCE = 0.35
AXIAL_REFERENCE_FLOOR = 0.05
TANGENTIAL_REFERENCE_FLOOR = 0.02


@dataclass(frozen=True)
class VortexCylinderSystem:
    """Piecewise-constant right-cylinder strengths at trailing radii."""

    trailing_radii: np.ndarray
    tangential_sheet_strength: np.ndarray
    longitudinal_sheet_strength: np.ndarray
    bound_circulation: np.ndarray
    annulus_induction: np.ndarray


def _complete_elliptic_pi(n: float, m: float) -> float:
    """Return complete ``Pi(n, m)`` through Carlson symmetric integrals."""
    # scipy.special has no portable ellippi symbol across the supported SciPy
    # versions.  Carlson's identity is Pi(n,m)=RF(0,1-m,1)+n RJ(...)/3.
    m = float(np.clip(m, 0.0, 1.0 - 1.0e-13))
    n = float(np.clip(n, 0.0, 1.0 - 1.0e-13))
    return float(
        special.elliprf(0.0, 1.0 - m, 1.0) + n * special.elliprj(0.0, 1.0 - m, 1.0, 1.0 - n) / 3.0
    )


def right_cylinder_influence(radius: float, cylinder_radius: float, downstream: float):
    """Return unit-sheet ``(u_axial, u_tangential_longitudinal)`` influence.

    The result is per unit ``gamma_t`` and ``gamma_l`` respectively.  Points
    on the sheet are not used by the annular comparison; their limiting value
    is set to the arithmetic mean of the two sides.
    """
    r = float(radius)
    R = float(cylinder_radius)
    y = float(downstream)
    if r <= 0.0 or R <= 0.0 or y <= 0.0:
        raise ValueError("right-cylinder influence requires r, R and y > 0")
    difference = abs(R - r)
    if difference <= 1.0e-12 * max(R, r):
        inside_axial = 0.5
        outside_longitudinal = 0.5
    elif R > r:
        inside_axial = 1.0
        outside_longitudinal = 0.0
    else:
        inside_axial = 0.0
        outside_longitudinal = 1.0
    denominator = (R + r) ** 2 + y**2
    m = float(np.clip(4.0 * r * R / denominator, 0.0, 1.0 - 1.0e-13))
    n = float(np.clip(4.0 * r * R / (R + r) ** 2, 0.0, 1.0 - 1.0e-13))
    root_m = np.sqrt(m)
    elliptic_k = float(special.ellipk(m))
    elliptic_pi = _complete_elliptic_pi(n, m)
    common = y * root_m / (2.0 * np.pi * np.sqrt(r * R))
    ratio = (R - r) / (R + r)
    axial = 0.5 * (inside_axial + common * (elliptic_k + ratio * elliptic_pi))
    tangential = (
        0.5 * (R / r) * (outside_longitudinal + common * (elliptic_k - ratio * elliptic_pi))
    )
    return axial, tangential


def build_system(
    bem,
    *,
    number_of_blades: int,
    freestream_speed: float,
    angular_velocity: float,
    hub_radius: float,
    rotor_radius: float,
) -> VortexCylinderSystem:
    """Build the untuned cylinder strengths from the matched BEM table."""
    radius = np.asarray(bem["radial_position"], dtype=float)
    circulation = number_of_blades * np.asarray(bem["circulation"], dtype=float)
    tangential_induction = np.asarray(bem["tangential_induction_factor"], dtype=float)
    if radius.ndim != 1 or len(radius) < 2 or np.any(np.diff(radius) <= 0.0):
        raise ValueError("BEM radial stations must be strictly increasing")
    if np.any((radius <= hub_radius) | (radius >= rotor_radius)):
        raise ValueError("BEM radial stations must lie strictly inside the rotor")
    if freestream_speed <= 0.0 or angular_velocity <= 0.0:
        raise ValueError("freestream speed and angular velocity must be positive")

    # Li et al. (2025), Eqs. (40)/(41), use the planar Kutta--Joukowski
    # coefficient C_T = k_s(1+a') for this closure.  This deliberately omits
    # the optional wake-rotation subtraction C_T,rot from Li et al. (2022),
    # Eq. (14); the longitudinal sheets still carry the matched circulation
    # and therefore retain the corresponding tangential induction.  Keeping
    # the finite-blade BEM table here makes the comparison matched, while the
    # cylinder itself remains the infinite-blade analytical reference.
    circulation_parameter = angular_velocity * circulation / (np.pi * freestream_speed**2)
    ct_effective = circulation_parameter * (1.0 + tangential_induction)
    if np.any(~np.isfinite(ct_effective)) or np.any(ct_effective >= 1.0):
        raise ValueError("finite-distance reference requires all effective annulus C_T < 1")
    annulus_induction = 0.5 * (1.0 - np.sqrt(np.maximum(1.0 - ct_effective, 0.0)))

    trailing_radii = np.r_[
        float(hub_radius),
        0.5 * (radius[:-1] + radius[1:]),
        float(rotor_radius),
    ]
    annulus_with_ghosts = np.r_[0.0, annulus_induction, 0.0]
    # The published kernel takes positive ``gamma_t`` as positive axial
    # velocity, while this module reports the standard positive wind-turbine
    # induction ``a = -u_a / U``.  Convert the positive BEM induction jump to
    # the corresponding signed sheet orientation here.
    tangential_sheet_strength = -2.0 * freestream_speed * np.diff(annulus_with_ghosts)
    # Retain the Γ_j−Γ_{j+1} orientation for the longitudinal sheets.  The
    # root ghost then gives the negative inside-rotor swirl required by
    # ``a' = -u_t / (Ω r)`` for positive circulation and Ω.
    circulation_with_ghosts = np.r_[0.0, circulation, 0.0]
    trailed_circulation = circulation_with_ghosts[:-1] - circulation_with_ghosts[1:]
    longitudinal_sheet_strength = trailed_circulation / (2.0 * np.pi * trailing_radii)
    return VortexCylinderSystem(
        trailing_radii=trailing_radii,
        tangential_sheet_strength=tangential_sheet_strength,
        longitudinal_sheet_strength=longitudinal_sheet_strength,
        bound_circulation=circulation,
        annulus_induction=annulus_induction,
    )


def induced_velocity(
    radius: float,
    downstream: float,
    system: VortexCylinderSystem,
    *,
    freestream_speed: float,
    angular_velocity: float,
):
    """Return analytical induced velocities and induction factors at ``(r,y)``."""
    axial = 0.0
    tangential = 0.0
    for cylinder_radius, gamma_t, gamma_l in zip(
        system.trailing_radii,
        system.tangential_sheet_strength,
        system.longitudinal_sheet_strength,
        strict=True,
    ):
        axial_influence, tangential_influence = right_cylinder_influence(
            radius, cylinder_radius, downstream
        )
        axial += axial_influence * gamma_t
        tangential += tangential_influence * gamma_l
    return {
        "axial_velocity": axial,
        "tangential_velocity": tangential,
        "axial_induction": -axial / freestream_speed,
        "tangential_induction": -tangential / (angular_velocity * radius),
    }
