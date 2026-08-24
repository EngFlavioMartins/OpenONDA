"""Deterministic, divergence-free initial wake perturbation for the cylinder.

The controlled-seed experiment (``OPENONDA_SEED_AMPLITUDE``) needs exactly the
same analytic perturbation applied to the initial FVM velocity field of both
the coupled hybrid and the fully meshed reference, so that any remaining onset
offset can be attributed to the coupling rather than to different forcing.

The perturbation is written from a 2-D streamfunction in the (x, y) plane,

    psi = eps * U_inf * D * exp(-((x - x_s)^2 + y^2) / a^2) * g(z),

with

    u'_x = dpsi/dy,   u'_y = -dpsi/dx,   u'_z = 0,

which is divergence-free by construction and carries zero z-vorticity source at
y = 0.  ``g(z)`` is a smooth spanwise taper that is one through the central
portion of the cylinder and vanishes toward the domain edges, so the seed
excites the primary Karman mode without reaching the boundary layers.

The caller normalises the sampled field so that ``max|u'| == eps * U_inf``.
"""

from __future__ import annotations

import numpy as np


def streamfunction_perturbation(
    points: np.ndarray,
    *,
    epsilon: float,
    freestream_speed: float = 1.0,
    diameter: float = 1.0,
    streamfunction_centre_x: float = 0.75,
    gaussian_half_width: float = 0.4,
    z_taper_begin: float = 3.0,
    z_taper_end: float = 4.75,
) -> np.ndarray:
    """Return the raw (not normalised) perturbation velocity at ``points``.

    Args:
        points: Cell-centre coordinates ``(N, 3)``.
        epsilon: Dimensionless seed amplitude (multiplies ``freestream_speed * diameter``).
        freestream_speed: Freestream speed.
        diameter: Cylinder diameter (the streamfunction length scale).
        streamfunction_centre_x: Streamfunction centre ``x/D`` (downstream of the cylinder).
        gaussian_half_width: Gaussian half-width relative to the cylinder diameter.
        z_taper_begin: ``|z|`` at which the spanwise taper starts (D units).
        z_taper_end: ``|z|`` at which the spanwise taper reaches zero (D units).

    Returns:
        Array ``(N, 3)`` with ``u'_z = 0``.  Not normalised: the peak value is
        approximately ``epsilon * freestream_speed`` only after :func:`normalise_seed`.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    taper_weight = np.ones_like(z)
    z_abs = np.abs(z)
    taper = (z_abs > z_taper_begin) & (z_abs < z_taper_end)
    taper_weight[taper] = 0.5 * (
        1.0 + np.cos(np.pi * (z_abs[taper] - z_taper_begin) / (z_taper_end - z_taper_begin))
    )
    taper_weight[z_abs >= z_taper_end] = 0.0

    radial_distance_squared = (x - streamfunction_centre_x) ** 2 + y**2
    envelope = np.exp(-radial_distance_squared / gaussian_half_width**2) * taper_weight

    perturbation_velocity = np.empty_like(points)
    perturbation_velocity[:, 0] = -2.0 * y / gaussian_half_width**2 * envelope
    perturbation_velocity[:, 1] = (
        2.0 * (x - streamfunction_centre_x) / gaussian_half_width**2 * envelope
    )
    perturbation_velocity[:, 2] = 0.0
    return epsilon * freestream_speed * diameter * perturbation_velocity


def normalise_seed(
    perturbation: np.ndarray, *, epsilon: float, freestream_speed: float = 1.0
) -> np.ndarray:
    """Scale a raw perturbation to ``epsilon * freestream_speed`` exactly."""
    perturb = np.asarray(perturbation, dtype=np.float64)
    peak = float(np.linalg.norm(perturb, axis=1).max())
    if not np.isfinite(peak) or peak <= 0.0:
        raise RuntimeError("seed perturbation field is identically zero or non-finite")
    return perturb * (epsilon * freestream_speed / peak)


def build_seed_velocity(
    cell_centre: np.ndarray,
    *,
    base_velocity,
    epsilon: float,
    freestream_speed: float = 1.0,
    diameter: float = 1.0,
) -> np.ndarray:
    """Return the seeded velocity with the requested perturbation magnitude.

    ``base_velocity`` is broadcast to ``(N, 3)``.  When ``epsilon <= 0`` the
    unperturbed base field is returned unchanged (no seed applied).
    """
    n = cell_centre.shape[0]
    base = np.broadcast_to(np.asarray(base_velocity, dtype=np.float64), (n, 3)).copy()
    if epsilon is None or float(epsilon) <= 0.0:
        return base
    raw = streamfunction_perturbation(
        cell_centre,
        epsilon=float(epsilon),
        freestream_speed=freestream_speed,
        diameter=diameter,
    )
    seed = normalise_seed(raw, epsilon=float(epsilon), freestream_speed=freestream_speed)
    return base + seed
