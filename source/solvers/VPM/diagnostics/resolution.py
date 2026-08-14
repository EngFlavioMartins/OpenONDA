"""
Discretization-health diagnostics for the VPM particle field.
=============================================================

Conservation and stability are different properties.  A structure-preserving
scheme conserves circulation, impulse and energy *by construction*, whether or
not the particle field still resolves the flow — so invariant drift alone
cannot tell you that a run is trustworthy.  These metrics measure the other
half: whether the particle cloud is still a faithful discretization of a
divergence-free vorticity field.

Three quantities, all cheap and all with an unambiguous "good" direction:

``overlap_ratio``
    Mean nearest-neighbour spacing over the core radius, h/sigma.  Particle
    quadrature converges only while blobs overlap; the error term of the
    vortex-method estimate is (h/sigma)^m, so h/sigma > 1 means the velocity
    field is no longer a consistent approximation.  Lagrangian distortion grows
    this ratio in the stretching directions of the flow map.

``vorticity_divergence_error``
    ||div w|| / ||grad w||, the |w|-weighted mean over particles.  The exact
    vorticity field is solenoidal; the discrete one, w_h = sum Gamma_j zeta_s,
    is not, and vortex stretching amplifies precisely its divergent part.  This
    is the standard signature of the classical 3-D vortex-method instability
    (Cottet & Koumoutsakos 2000, Sec. 5.3; Pedrizzetti 1992).

``strength_misalignment_deg``
    Angle between Gamma_p and w(x_p).  In the continuum they are parallel; the
    DIRECT and TRANSPOSED stretching forms differ by exactly Gamma x w, so this
    angle simultaneously measures how far the particle field is from being a
    vorticity field and how much the choice of stretching form can matter.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: July 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import numpy as np
from scipy.spatial import cKDTree

__all__ = ["discretization_health"]

# Gaussian zeta has decayed to ~1e-7 by 4 sigma; beyond that a neighbour adds
# nothing to div(w) at f32 precision.  Keeps the divergence sum O(N k) rather
# than O(N^2).
_SUPPORT_SIGMAS = 4.0

# Neighbours averaged into the nominal local spacing h_nn.  Six is the
# coordination number of the hexagonal close packing the distributor emits, so
# for an undistorted lattice h_nn is exactly the lattice spacing.
_NEIGHBOURS = 6
_MAX_PROBES = 512


def _vorticity_gradient_metrics(
    positions: np.ndarray,
    circulations: np.ndarray,
    radii: np.ndarray,
    tree: cKDTree,
    probes: np.ndarray,
) -> tuple[float, np.ndarray]:
    """|w|-weighted mean of |div w| / ||grad w||_F over particles.

    For w_h(x) = sum_j Gamma_j zeta(|x-x_j|/s_j)/s_j^3 with a Gaussian zeta,

        d(w_b)/d(x_a) = sum_j Gamma_jb * (-2 (x-x_j)_a / s_j^2) * zeta_j / s_j^3

    so the divergence is the trace and the Frobenius norm bounds it.  The ratio
    is scale-free: 0 for a perfectly solenoidal discrete field, O(1) once the
    field has lost its vortex-line structure.
    """
    inv_pi15 = np.pi**-1.5
    radius = _SUPPORT_SIGMAS * float(radii.max())
    divergence = np.zeros(len(probes))
    frobenius = np.zeros(len(probes))
    vorticity = np.zeros((len(probes), 3))
    for sample_index, i in enumerate(probes):
        idx = tree.query_ball_point(positions[i], radius)
        j = np.asarray(idx, dtype=np.intp)
        d = positions[i] - positions[j]
        s = radii[j]
        z = inv_pi15 * np.exp(-np.einsum("ka,ka->k", d, d) / s**2) / s**3
        vorticity[sample_index] = z @ circulations[j]
        # jac[a, b] = d(w_b)/d(x_a)
        jac = np.einsum("k,ka,kb->ab", -2.0 * z / s**2, d, circulations[j])
        divergence[sample_index] = np.trace(jac)
        frobenius[sample_index] = np.sqrt(np.einsum("ab,ab->", jac, jac))

    ratio = np.abs(divergence) / np.maximum(frobenius, np.finfo(float).tiny)
    magnitude = np.linalg.norm(vorticity, axis=1)
    total = magnitude.sum()
    if not np.isfinite(total) or total <= 0.0:
        return float("nan"), vorticity
    return float(ratio @ (magnitude / total)), vorticity


def discretization_health(
    positions: np.ndarray,
    circulations: np.ndarray,
    radii: np.ndarray,
    vorticities: np.ndarray | None = None,
) -> dict[str, float]:
    """Return the resolution/consistency metrics for one particle field.

    Args:
        positions: (N, 3) particle positions.
        circulations: (N, 3) particle circulations Gamma_p [m^3/s].
        radii: (N,) particle core radii sigma_p [m].
        vorticities: optional (N, 3) w(x_p) already evaluated by the solver.
            When omitted, the same compact Gaussian-neighbour sum used for the
            divergence diagnostic supplies vorticity at the probe particles.

    Returns:
        dict of scalar metrics; every value is NaN when it cannot be formed.
    """
    nan = float("nan")
    empty = {
        "core_radius_mean": nan,
        "overlap_ratio": nan,
        "overlap_ratio_max": nan,
        "vorticity_divergence_error": nan,
        "strength_misalignment_deg": nan,
    }
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    n = len(positions)
    if n < _NEIGHBOURS + 1:
        return empty

    circulations = np.ascontiguousarray(circulations, dtype=np.float64)
    radii = np.ascontiguousarray(radii, dtype=np.float64)
    if not np.isfinite(positions).all() or not np.isfinite(radii).all() or radii.max() <= 0.0:
        return empty

    # compact_nodes=False: the default (balanced + compact) build recurses and
    # writes out of bounds for regular-lattice clouds, corrupting the heap
    # (observed only after Taichi shifts the heap layout).  Queries are exact
    # regardless of the compaction flag; only the tree structure differs.
    tree = cKDTree(positions, compact_nodes=False)
    if n <= _MAX_PROBES:
        probes = np.arange(n, dtype=np.intp)
    else:
        probes = np.linspace(0, n - 1, _MAX_PROBES, dtype=np.intp)
    distances, _ = tree.query(positions[probes], k=_NEIGHBOURS + 1, workers=-1)
    spacing = distances[:, 1:].mean(axis=1)
    overlap = spacing / np.maximum(radii[probes], np.finfo(float).tiny)

    metrics = dict(empty)
    metrics["core_radius_mean"] = float(radii.mean())
    metrics["overlap_ratio"] = float(overlap.mean())
    metrics["overlap_ratio_max"] = float(overlap.max())
    divergence_error, reconstructed_vorticity = _vorticity_gradient_metrics(
        positions, circulations, radii, tree, probes
    )
    metrics["vorticity_divergence_error"] = divergence_error

    if vorticities is None:
        sampled_vorticity = reconstructed_vorticity
    else:
        vorticities = np.ascontiguousarray(vorticities, dtype=np.float64)
        sampled_vorticity = vorticities[probes]
    sampled_circulation = circulations[probes]
    weight = np.linalg.norm(sampled_circulation, axis=1)
    norm = weight * np.linalg.norm(sampled_vorticity, axis=1)
    usable = norm > np.finfo(float).tiny
    if usable.any() and weight[usable].sum() > 0.0:
        cosine = (
            np.einsum(
                "ka,ka->k",
                sampled_circulation[usable],
                sampled_vorticity[usable],
            )
            / norm[usable]
        )
        angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        share = weight[usable] / weight[usable].sum()
        metrics["strength_misalignment_deg"] = float(angle @ share)

    return metrics
