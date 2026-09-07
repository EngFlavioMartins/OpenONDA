"""Azimuthally averaged Gaussian vorticity about the x axis.

This measures the represented field, independent of particle group labels.
Averaging is a diagnostic, not an imposed symmetry of the simulation.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import i1e


def azimuthal_circulation(position, vortex_strength, core_radius):
    """Signed integral of mean omega_theta over the entire meridional half-plane.

    Exact Gaussian integral, independent of material labels. The axis factor
    accounts for cancellation across r=0; summing |alpha| cannot measure this.
    """
    p = np.asarray(position, dtype=float)
    a = np.asarray(vortex_strength, dtype=float)
    sigma = np.asarray(core_radius, dtype=float)
    if p.shape != a.shape or p.shape != (len(sigma), 3):
        raise ValueError("particle arrays must agree")
    if not all(np.isfinite(v).all() for v in (p, a, sigma)) or np.any(sigma <= 0):
        raise ValueError("sources must be finite and core radii positive")
    radius_squared = p[:, 1] ** 2 + p[:, 2] ** 2
    factor = np.divide(
        -np.expm1(-radius_squared / sigma**2),
        radius_squared,
        out=1 / sigma**2,
        where=radius_squared > 0,
    )
    return float(np.sum((p[:, 1] * a[:, 2] - p[:, 2] * a[:, 1]) * factor) / (2 * np.pi))


def azimuthal_vorticity(position, vortex_strength, core_radius, axial_radial_targets):
    """Return mean omega_theta at (x, r) targets using exact angular integration.

    Integrating a Gaussian around its source orbit yields I1(2*r*rj/sigma²).
    The exponentially scaled Bessel function avoids overflow near thin cores.
    No interpolation, particle labels, or assumption of uniform cores is used.
    For grids of targets, sources beyond eight maximum core widths are omitted;
    their Gaussian factor is below exp(-64), avoiding work on negligible tails.
    """
    position = np.asarray(position, dtype=np.float64)
    strength = np.asarray(vortex_strength, dtype=np.float64)
    sigma = np.asarray(core_radius, dtype=np.float64)
    targets = np.asarray(axial_radial_targets, dtype=np.float64).reshape(-1, 2)
    if position.shape != strength.shape or position.shape != (len(sigma), 3):
        raise ValueError("particle position, strength and core arrays must agree")
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError("Gaussian core radii must be finite and positive")
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(strength)):
        raise ValueError("particle sources must be finite")
    if not np.all(np.isfinite(targets)) or np.any(targets[:, 1] < 0):
        raise ValueError("targets must be finite with nonnegative radial coordinate")
    radius = np.hypot(position[:, 1], position[:, 2])
    tangent_strength = np.divide(
        position[:, 1] * strength[:, 2] - position[:, 2] * strength[:, 1],
        radius,
        out=np.zeros(len(radius)),
        where=radius > 0,
    )
    variance = sigma**2
    amplitude = tangent_strength / (np.pi**1.5 * sigma**3)
    tree = (
        cKDTree(np.column_stack((position[:, 0], radius)))
        if len(targets) > 64 and len(sigma)
        else None
    )
    cutoff = 8 * float(sigma.max()) if len(sigma) else 0.0
    result = np.empty(len(targets))
    for i, (x, r) in enumerate(targets):
        selected = tree.query_ball_point([x, r], cutoff) if tree is not None else slice(None)
        result[i] = np.sum(
            amplitude[selected]
            * np.exp(
                -((x - position[selected, 0]) ** 2 + (r - radius[selected]) ** 2)
                / variance[selected]
            )
            * i1e(2 * r * radius[selected] / variance[selected])
        )
    return result
