"""
Discretization-health diagnostics for the VPM particle field.
=============================================================

Conservation and stability are different properties.  A structure-preserving
scheme conserves vortex strength, impulse and energy *by construction*, whether or
not the particle field still resolves the flow — so invariant drift alone
cannot tell you that a run is trustworthy.  These metrics measure the other
half: whether the particle cloud is still a faithful discretization of a
divergence-free vorticity field.

Three quantities, all cheap and all with an unambiguous "good" direction:

``mean_overlap_ratio``
    Mean nearest-neighbour spacing over the core radius, h/sigma.  Particle
    quadrature converges only while blobs overlap; the error term of the
    vortex-method estimate is (h/sigma)^m, so h/sigma > 1 means the velocity
    field is no longer a consistent approximation.  Lagrangian distortion grows
    this ratio in the stretching directions of the flow map.

``vorticity_divergence_error``
    ||div w|| / ||grad w||, the |w|-weighted mean over particles.  The exact
    vorticity field is solenoidal; the discrete one, w_h = sum alpha_j zeta_s,
    is not, and vortex stretching amplifies precisely its divergent part.  This
    is the standard signature of the classical 3-D vortex-method instability
    (Cottet & Koumoutsakos 2000, Sec. 5.3; Pedrizzetti 1992).

``vortex_strength_misalignment_degrees``
    Angle between alpha_p and w(x_p). In the continuum they are parallel; the
    DIRECT and TRANSPOSED stretching forms differ by exactly alpha x w, so this
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
    position: np.ndarray,
    vortex_strength: np.ndarray,
    core_radius: np.ndarray,
    tree: cKDTree,
    probes: np.ndarray,
) -> tuple[float, np.ndarray]:
    """|w|-weighted mean of |div w| / ||grad w||_F over particles.

    For w_h(x) = sum_j alpha_j zeta(|x-x_j|/s_j)/s_j^3 with a Gaussian zeta,

        d(w_b)/d(x_a) = sum_j alpha_jb * (-2 (x-x_j)_a / s_j^2) * zeta_j / s_j^3

    so the divergence is the trace and the Frobenius norm bounds it.  The ratio
    is scale-free: 0 for a perfectly solenoidal discrete field, O(1) once the
    field has lost its vortex-line structure.
    """
    inv_pi15 = np.pi**-1.5
    support_radius = _SUPPORT_SIGMAS * float(core_radius.max())
    divergence = np.zeros(len(probes))
    frobenius = np.zeros(len(probes))
    vorticity = np.zeros((len(probes), 3))
    for sample_index, i in enumerate(probes):
        idx = tree.query_ball_point(position[i], support_radius)
        j = np.asarray(idx, dtype=np.intp)
        d = position[i] - position[j]
        s = core_radius[j]
        z = inv_pi15 * np.exp(-np.einsum("ka,ka->k", d, d) / s**2) / s**3
        vorticity[sample_index] = z @ vortex_strength[j]
        # jac[a, b] = d(w_b)/d(x_a)
        jac = np.einsum("k,ka,kb->ab", -2.0 * z / s**2, d, vortex_strength[j])
        divergence[sample_index] = np.trace(jac)
        frobenius[sample_index] = np.sqrt(np.einsum("ab,ab->", jac, jac))

    ratio = np.abs(divergence) / np.maximum(frobenius, np.finfo(float).tiny)
    magnitude = np.linalg.norm(vorticity, axis=1)
    total = magnitude.sum()
    if not np.isfinite(total) or total <= 0.0:
        return float("nan"), vorticity
    return float(ratio @ (magnitude / total)), vorticity


def discretization_health(
    position: np.ndarray,
    vortex_strength: np.ndarray,
    core_radius: np.ndarray,
    vorticity: np.ndarray | None = None,
) -> dict[str, float]:
    """Return the resolution/consistency metrics for one particle field.

    Args:
        position: (N, 3) particle position.
        vortex_strength: (N, 3) particle vortex strength [m³/s].
        core_radius: (N,) particle core radius [m].
        vorticity: optional (N, 3) w(x_p) already evaluated by the solver.
            When omitted, the same compact Gaussian-neighbour sum used for the
            divergence diagnostic supplies vorticity at the probe particles.

    Returns:
        dict of scalar metrics; every value is NaN when it cannot be formed.
    """
    nan = float("nan")
    empty = {
        "mean_core_radius": nan,
        "mean_overlap_ratio": nan,
        "max_overlap_ratio": nan,
        "vorticity_divergence_error": nan,
        "vortex_strength_misalignment_degrees": nan,
    }
    position = np.ascontiguousarray(position, dtype=np.float64)
    n = len(position)
    if n < _NEIGHBOURS + 1:
        return empty

    vortex_strength = np.ascontiguousarray(vortex_strength, dtype=np.float64)
    core_radius = np.ascontiguousarray(core_radius, dtype=np.float64)
    if (
        not np.isfinite(position).all()
        or not np.isfinite(core_radius).all()
        or core_radius.max() <= 0.0
    ):
        return empty

    # compact_nodes=False: the default (balanced + compact) build recurses and
    # writes out of bounds for regular-lattice clouds, corrupting the heap
    # (observed only after Taichi shifts the heap layout).  Queries are exact
    # regardless of the compaction flag; only the tree structure differs.
    tree = cKDTree(position, compact_nodes=False)
    if n <= _MAX_PROBES:
        probes = np.arange(n, dtype=np.intp)
    else:
        probes = np.linspace(0, n - 1, _MAX_PROBES, dtype=np.intp)
    distances, _ = tree.query(position[probes], k=_NEIGHBOURS + 1, workers=-1)
    spacing = distances[:, 1:].mean(axis=1)
    overlap = spacing / np.maximum(core_radius[probes], np.finfo(float).tiny)

    metrics = dict(empty)
    metrics["mean_core_radius"] = float(core_radius.mean())
    metrics["mean_overlap_ratio"] = float(overlap.mean())
    metrics["max_overlap_ratio"] = float(overlap.max())
    divergence_error, reconstructed_vorticity = _vorticity_gradient_metrics(
        position, vortex_strength, core_radius, tree, probes
    )
    metrics["vorticity_divergence_error"] = divergence_error

    if vorticity is None:
        sampled_vorticity = reconstructed_vorticity
    else:
        vorticity = np.ascontiguousarray(vorticity, dtype=np.float64)
        sampled_vorticity = vorticity[probes]
    sampled_vortex_strength = vortex_strength[probes]
    weight = np.linalg.norm(sampled_vortex_strength, axis=1)
    norm = weight * np.linalg.norm(sampled_vorticity, axis=1)
    usable = norm > np.finfo(float).tiny
    if usable.any() and weight[usable].sum() > 0.0:
        cosine = (
            np.einsum(
                "ka,ka->k",
                sampled_vortex_strength[usable],
                sampled_vorticity[usable],
            )
            / norm[usable]
        )
        angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        share = weight[usable] / weight[usable].sum()
        metrics["vortex_strength_misalignment_degrees"] = float(angle @ share)

    return metrics
