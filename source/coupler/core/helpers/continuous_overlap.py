"""Conservative overlap transport from FVM cells to VPM particles.

The hand-off remeshes onto a regular lattice, blends FVM and VPM circulation
with a smooth partition of unity, and recovers selected integral invariants
after pruning. The implementation is independent of the FVM and VPM runtimes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging

import numpy as np

from source.coupler.core.helpers.fvm_velocity_trace import CachedVelocityTrace
from source.coupler.diagnostics.injection_correction import recover_invariants
from source.coupler.remesh import remesh_to_grid

logger = logging.getLogger("coupler")

# Particle core radius relative to the regular overlap lattice spacing.
RADIUS_RATIO = 1.0

# M4-prime kernel support half-width in lattice cells.
_M4P_SUPPORT = 2.0


def _invariants(pos: np.ndarray, circ: np.ndarray) -> dict[str, np.ndarray]:
    """Total circulation, linear impulse and (raw) angular impulse
    (Winckelmans 1993).  Kernel-correction-free: the prune redistribution
    uses the same definition on both sides, so the σ² term cancels."""
    if len(pos) == 0:
        z = np.zeros(3)
        return {"circulation": z, "linear_impulse": z, "angular_impulse": z}
    r_x_g = np.cross(pos, circ)
    return {
        "circulation": np.sum(circ, axis=0),
        "linear_impulse": 0.5 * np.sum(r_x_g, axis=0),
        "angular_impulse": (1.0 / 3.0) * np.sum(np.cross(pos, r_x_g), axis=0),
    }


def required_buffer_length(u_max: float, dt: float, h: float, safety: float = 1.5) -> float:
    """Minimum downstream buffer length for a dt-robust hand-off:
    ``L_buf ≥ safety · u_max · dt + 2h`` (M4′ stencil must stay interior)."""
    return float(safety * abs(u_max) * abs(dt) + _M4P_SUPPORT * h)


def max_stable_dt(u_max: float, l_buf: float, h: float, safety: float = 1.5) -> float:
    """Largest ``dt`` for which the given buffer keeps the hand-off exact
    (inverse of :func:`required_buffer_length`)."""
    u = abs(u_max)
    if u < 1e-30:
        return float("inf")
    return float(max(l_buf - _M4P_SUPPORT * h, 0.0) / (safety * u))


def circulation_from_velocity_trace(
    positions: np.ndarray,
    h: float,
    velocity_at,
) -> np.ndarray:
    """Integrate ``n × u`` over each cubic particle control volume."""
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    circulation = np.zeros_like(pos)
    offset = np.zeros(3, dtype=np.float64)
    for axis in range(3):
        offset.fill(0.0)
        offset[axis] = 0.5 * h
        delta_u = np.asarray(velocity_at(pos + offset), dtype=np.float64) - np.asarray(
            velocity_at(pos - offset), dtype=np.float64
        )
        normal = np.zeros(3, dtype=np.float64)
        normal[axis] = 1.0
        circulation += h**2 * np.cross(normal, delta_u)
    return circulation


def _outflow_band_mask(
    grid_pos: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    h: float,
    u_inf=None,
) -> np.ndarray:
    """Boolean mask of the downstream-most h-layer of the box.

    Direction-agnostic: the outflow face is the box face whose outward normal
    is most aligned with the freestream ``u_inf``.  With ``u_inf=None`` (or a
    zero vector) it defaults to the +x face.
    Used only by the flux-ratio diagnostic — the hand-off itself treats every
    face identically.
    """
    axis, sign = 0, +1.0
    if u_inf is not None:
        u = np.asarray(u_inf, dtype=np.float64).reshape(-1)
        if u.size == 3 and np.any(u != 0.0):
            axis = int(np.argmax(np.abs(u)))
            sign = float(np.sign(u[axis]))
    if sign >= 0:
        return grid_pos[:, axis] >= hi[axis] - h
    return grid_pos[:, axis] <= lo[axis] + h


def cosine_eta(
    grid_pos: np.ndarray,
    box: np.ndarray,
    ramp_width: float,
    dead_zone: float,
) -> np.ndarray:
    """C¹ partition-of-unity FVM-authority weight η(x) ∈ [0, 1].

    Built from the minimum signed distance to any box face:
    η = 1 for ``dist ≥ ramp_width`` (FVM core), η = 0 for ``dist ≤ dead_zone``
    and outside the box, cosine ramp (zero slope at both ends) in between.
    """
    pos = np.asarray(grid_pos, dtype=np.float64)
    b = np.asarray(box, dtype=np.float64)
    dist = np.minimum.reduce(
        [
            pos[:, 0] - b[0],
            b[1] - pos[:, 0],
            pos[:, 1] - b[2],
            b[3] - pos[:, 1],
            pos[:, 2] - b[4],
            b[5] - pos[:, 2],
        ]
    )
    eta = np.zeros(len(pos), dtype=np.float64)
    if ramp_width <= 0.0:
        eta[dist > 0.0] = 1.0
        return eta

    eta[dist >= ramp_width] = 1.0
    lo = max(dead_zone, 0.0)
    width = max(ramp_width - lo, 1e-30)
    ramp = (dist > lo) & (dist < ramp_width)
    if ramp.any():
        eta[ramp] = 0.5 * (1.0 - np.cos(np.pi * (dist[ramp] - lo) / width))
    return eta


def beale_strength_correction(
    circ_grid: np.ndarray,
    target_circ: np.ndarray,
    eta: np.ndarray,
    shape: tuple[int, int, int],
    h: float,
    *,
    sigma: float,
) -> tuple[np.ndarray, float, float]:
    """Apply one Gaussian mollification correction in the FVM authority zone."""
    h3 = float(h) ** 3

    g = np.asarray(circ_grid, dtype=np.float64).reshape(*shape, 3).copy()
    t_omega = np.asarray(target_circ, dtype=np.float64).reshape(*shape, 3) / h3
    eta_g = np.asarray(eta, dtype=np.float64).reshape(*shape)[..., None]

    denom = float(np.linalg.norm(t_omega * eta_g)) + 1e-30
    residual = (t_omega - _gaussian_mollified_circulation(g, shape, h, sigma=sigma) / h3) * eta_g
    res_pre = float(np.linalg.norm(residual)) / denom
    g += residual * h3
    residual = (t_omega - _gaussian_mollified_circulation(g, shape, h, sigma=sigma) / h3) * eta_g
    res_post = float(np.linalg.norm(residual)) / denom

    return g.reshape(-1, 3), res_pre, res_post


def _gaussian_mollified_circulation(
    circ_grid: np.ndarray,
    shape: tuple[int, int, int],
    h: float,
    *,
    sigma: float,
) -> np.ndarray:
    """Return circulation represented by Gaussian particles on their lattice."""
    from scipy.ndimage import gaussian_filter

    # ζ_σ ∝ e^{-r²/σ²} = e^{-r²/(2 s²)} with s = σ/√2; gaussian_filter takes
    # s in grid cells and normalizes its discrete kernel to Σw = 1, which for
    # s ≳ 1 cell equals the sampled ζ_σ·h³ weights to machine precision.
    s_cells = float(sigma) / (np.sqrt(2.0) * float(h))
    grid = np.asarray(circ_grid, dtype=np.float64).reshape(*shape, 3)
    return np.stack(
        [
            gaussian_filter(grid[..., component], s_cells, mode="constant", truncate=5.0)
            for component in range(3)
        ],
        axis=-1,
    )


# =========================================================
# Result container
# =========================================================
@dataclass
class HandoffResult:
    """New particle field + per-step diagnostics from one hand-off."""

    pos: np.ndarray
    circ: np.ndarray
    vol: np.ndarray
    rad: np.ndarray

    n_remesh_in: int = 0  # particles fed into the lattice
    n_remesh_out: int = 0  # particles produced from the lattice
    n_free: int = 0  # untouched free-exterior particles
    n_excluded: int = 0  # input particles removed from a physical solid
    n_pruned: int = 0
    pruned_circulation_l1: float = 0.0
    pruned_circulation_fraction: float = 0.0
    n_population_pruned: int = 0
    population_pruned_circulation_fraction: float = 0.0
    population_pruned_velocity_bound: float = 0.0
    cfl: float = 0.0  # U_max·dt / L_buf  (should stay < ~0.7)
    conservation_drift: dict[str, float] = field(default_factory=dict)
    # Invariant diagnostics around the conservative prune correction.  These
    # are deliberately separate from ``conservation_drift`` (the historical
    # post-correction summary): validation needs to know whether a small final
    # error came from a small raw mismatch or a large correction that happened
    # to close it.
    conservation_raw_mismatch: dict[str, float] = field(default_factory=dict)
    conservation_applied_correction: dict[str, float] = field(default_factory=dict)
    conservation_corrected_mismatch: dict[str, float] = field(default_factory=dict)
    # Σ|Γσ|_VPM / Σ|Γ|_FVM over the outflow band (L1,
    # well-conditioned; 1 = the mollified particle field carries the FVM's exit
    # vorticity content). See the computation in continuous_handoff for why
    # raw, deconvolved particle strengths cannot be compared to the FVM trace.
    flux_ratio: float = 0.0

    # Strength-correction diagnostics (η-weighted relative residual
    # ‖ω_target − ω_σ‖/‖ω_target‖ before and after the Picard iterations)
    strength_corr_residual_pre: float = 0.0
    strength_corr_residual_post: float = 0.0

    # Body-mask audit.  These are L1 sums of |Gamma| removed from actual input
    # particles, VPM remesh support, and FVM-target remesh support.
    excluded_input_circulation_l1: float = 0.0
    excluded_remesh_circulation_l1: float = 0.0
    excluded_target_circulation_l1: float = 0.0

    @property
    def n_total(self) -> int:
        return len(self.pos)


# =========================================================
# Pure-NumPy numerical core
# =========================================================
def continuous_handoff(
    pos: np.ndarray,
    circ: np.ndarray,
    box: np.ndarray | list[float],
    h: float,
    *,
    circulation_at_node,
    u_inf=None,
    inside_mesh_at_node=None,
    excluded_at_node=None,
    ramp_width: float | None = None,
    dead_zone: float = 0.0,
    buffer_length: float = 0.0,
    threshold_abs: float = 0.0,
    radius_ratio: float = RADIUS_RATIO,
    u_max: float = 0.0,
    dt: float = 0.0,
    lattice_anchor=None,
    max_output_particles: int | None = None,
) -> HandoffResult:
    """One continuous, conservative, dt-robust FVM→VPM hand-off.

    Parameters
    ----------
    pos, circ : ndarray (N, 3)
        Current particle positions and circulations.
    box : (6,) ``[x0, x1, y0, y1, z0, z1]``
        FVM domain bounds (the interface Γ is the box surface).
    h : float
        Lattice spacing (= VPM particle spacing).
    circulation_at_node : callable
        FVM circulation reconstructed from the weighted velocity trace.
    u_inf : ndarray (3,) or None
        Freestream velocity vector used to orient the outflow-band diagnostic.
    max_output_particles : int or None
        Post-handoff population cap. Transported exterior particles are
        retained before weak reconstructed overlap particles are discarded.
    inside_mesh_at_node : callable ``grid_pos -> bool (M,)`` or None
        Mask of nodes that have FVM data (inside the mesh, outside the body).
        Nodes outside are forced η = 0 (pure Lagrangian).
    excluded_at_node : callable ``positions -> bool (M,)`` or None
        Exact physical-solid mask.  Unlike ``inside_mesh_at_node=False``, which
        merely selects the Lagrangian representation, excluded positions are
        removed from both representations and can never become particles.
    ramp_width : float
        Width of the η cosine ramp.  Defaults to ``2·dead_zone`` or ``4h``.
    dead_zone : float
        Thickness of the η = 0 band at each face.
    buffer_length : float
        Downstream/outward extension of the remesh lattice beyond the box
        (size with :func:`required_buffer_length`).
    threshold_abs : float
        |Γ| below which a lattice node is pruned (then redistributed).
    u_max, dt : float
        Used only for the CFL diagnostic.
    Returns
    -------
    HandoffResult
    """
    box = np.asarray(box, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
    circ = np.asarray(circ, dtype=np.float64).reshape(-1, 3)
    n = len(pos)

    if ramp_width is None:
        ramp_width = max(2.0 * dead_zone, 4.0 * h)

    lo = np.array([box[0], box[2], box[4]], dtype=np.float64)
    hi = np.array([box[1], box[3], box[5]], dtype=np.float64)

    # ── Active region = box ⊕ buffer_length on every face.  Particles here are
    #    remeshed; beyond it they are purely Lagrangian "free exterior".
    # The lattice extends a full M4′ guard band (2h) beyond the active region
    # so that every remeshed particle's 4-node stencil stays inside the
    # lattice.  If the active region reached the lattice edge, exiting
    # particles would have truncated stencils, partition-of-unity would break,
    # and the conservative prune would amplify the surviving boundary nodes.
    guard = _M4P_SUPPORT * h
    lo_active = lo - buffer_length
    hi_active = hi + buffer_length
    lo_lat = lo_active - guard
    hi_lat = hi_active + guard

    # Keep the remesh lattice aligned with the FVM lattice.
    if lattice_anchor is not None:
        anchor = np.asarray(lattice_anchor, dtype=np.float64).reshape(3)
        shift = np.floor((lo_lat - anchor) / float(h))
        lo_lat = anchor + shift * float(h)

    excluded_input = np.zeros(n, dtype=bool)
    if excluded_at_node is not None and n > 0:
        excluded_input = np.asarray(excluded_at_node(pos), dtype=bool).reshape(-1)
        if excluded_input.shape != (n,):
            raise ValueError(f"excluded_at_node returned {excluded_input.shape}, expected ({n},)")
    valid_input = ~excluded_input
    excluded_input_l1 = float(np.linalg.norm(circ[excluded_input], axis=1).sum())
    if n > 0:
        in_region = valid_input & np.all((pos >= lo_active) & (pos <= hi_active), axis=1)
    else:
        in_region = np.zeros(0, dtype=bool)
    free_mask = valid_input & ~in_region

    shape = (
        int(np.ceil((hi_lat[0] - lo_lat[0]) / h)) + 1,
        int(np.ceil((hi_lat[1] - lo_lat[1]) / h)) + 1,
        int(np.ceil((hi_lat[2] - lo_lat[2]) / h)) + 1,
    )

    # ── P2M: conservative M4′ scatter of the in-region wake onto the lattice ──
    grid_pos, grid_circ = remesh_to_grid(
        pos[in_region],
        circ[in_region],
        lo_lat,
        h,
        shape,
    )

    excluded_grid = np.zeros(len(grid_pos), dtype=bool)
    if excluded_at_node is not None:
        excluded_grid = np.asarray(excluded_at_node(grid_pos), dtype=bool).reshape(-1)
        if excluded_grid.shape != (len(grid_pos),):
            raise ValueError(
                f"excluded_at_node returned {excluded_grid.shape}, expected ({len(grid_pos)},)"
            )
    excluded_remesh_l1 = float(np.linalg.norm(grid_circ[excluded_grid], axis=1).sum())
    grid_circ[excluded_grid] = 0.0

    # ── η partition of unity (FVM authority) ─────────────────────────────────
    eta = cosine_eta(grid_pos, box, ramp_width, dead_zone)

    # ── Build the FVM target circulation on the lattice ──────────────────────
    target = np.asarray(circulation_at_node(grid_pos), dtype=np.float64).reshape(-1, 3)

    excluded_target_l1 = 0.0
    if excluded_grid.any():
        excluded_target_l1 = float(np.linalg.norm(target[excluded_grid], axis=1).sum())
        target[excluded_grid] = 0.0

    # ── Blend toward the FVM target where η > 0 (under-relaxed by α) ──────────
    ok = None
    if inside_mesh_at_node is not None:
        ok = np.asarray(inside_mesh_at_node(grid_pos), dtype=bool)
        eta = eta * ok
        target[~ok] = 0.0
    grid_blended = grid_circ + eta[:, None] * (target - grid_circ)

    # ── Beale/Picard strength correction (η-localized deconvolution) ─────────
    # BODY GUARD: the target ends in a step at the body wall (no cells inside).
    # Deconvolving across that discontinuity produces Gibbs-like ringing with
    # up-to-(M+1)× amplification in a ±2σ shell — including nodes just inside
    # the wall (which are `ok` because they sit within 1.5h of exterior cell
    # centres), injecting circulation INSIDE the body.  Restrict the correction
    # to nodes whose entire kernel support sees valid FVM data: erode the ok
    # mask by the kernel support (≈ 2σ/h cells) before weighting.
    corr_weight = eta
    if ok is not None:
        from scipy.ndimage import binary_erosion

        support_cells = max(int(np.ceil(2.0 * radius_ratio)), 1)
        ok_eroded = binary_erosion(
            ok.reshape(shape), iterations=support_cells, border_value=0
        ).ravel()
        corr_weight = eta * ok_eroded
    grid_blended, corr_pre, corr_post = beale_strength_correction(
        grid_blended,
        target,
        corr_weight,
        shape,
        h,
        sigma=radius_ratio * h,
    )

    # Deconvolution and remeshing have non-compact physical kernels near a
    # wall.  Re-assert the exact solid mask after every correction so no
    # particle centre can be regenerated inside the body.
    grid_blended[excluded_grid] = 0.0

    # ── Conservative prune: drop weak nodes, redistribute their moments ──────
    target_inv = _invariants(grid_pos, grid_blended)
    mag = np.linalg.norm(grid_blended, axis=1)
    candidates = (mag > 0.0) & ~excluded_grid
    keep = (mag >= threshold_abs) & ~excluded_grid
    pruned = candidates & ~keep
    active_l1 = float(mag[candidates].sum())
    pruned_l1 = float(mag[pruned].sum())
    new_pos = grid_pos[keep]
    new_circ = grid_blended[keep]

    pre_correction = _invariants(new_pos, new_circ)
    post_correction = pre_correction

    def _invariant_norms(left: dict[str, np.ndarray], right: dict[str, np.ndarray]):
        return {
            name: float(np.linalg.norm(left[name] - right[name]))
            for name in ("circulation", "linear_impulse", "angular_impulse")
        }

    raw_mismatch = _invariant_norms(target_inv, pre_correction)
    applied_correction = _invariant_norms(pre_correction, post_correction)
    corrected_mismatch = _invariant_norms(target_inv, post_correction)
    drift: dict[str, float] = {}
    if len(new_pos) > 0:
        new_vol_tmp = np.full(len(new_pos), h**3)
        # Angular impulse is intentionally excluded because the remeshing
        # operator does not conserve that second moment.
        new_circ = recover_invariants(
            new_pos,
            new_circ,
            target_inv,
            volumes=new_vol_tmp,
        )
        post_correction = _invariants(new_pos, new_circ)
        applied_correction = _invariant_norms(pre_correction, post_correction)
        corrected_mismatch = _invariant_norms(target_inv, post_correction)
        ref = float(np.linalg.norm(target_inv["circulation"])) + 1e-30
        drift = {
            **corrected_mismatch,
            "circulation_rel": float(corrected_mismatch["circulation"] / ref),
        }

    new_vol = np.full(len(new_pos), h**3)
    new_rad = np.full(len(new_pos), h * radius_ratio)

    # ── Re-attach the free-exterior particles unchanged ──────────────────────
    if free_mask.any():
        out_pos = np.vstack([new_pos, pos[free_mask]])
        out_circ = np.vstack([new_circ, circ[free_mask]])
        out_vol = np.concatenate([new_vol, np.full(int(free_mask.sum()), h**3)])
        out_rad = np.concatenate([new_rad, np.full(int(free_mask.sum()), h * radius_ratio)])
    else:
        out_pos, out_circ, out_vol, out_rad = new_pos, new_circ, new_vol, new_rad

    population_pruned = 0
    population_pruned_fraction = 0.0
    population_pruned_velocity_bound = 0.0
    if max_output_particles is not None and len(out_pos) > max_output_particles:
        target_count = int(max_output_particles)
        combined_target = _invariants(out_pos, out_circ)
        combined_mag = np.linalg.norm(out_circ, axis=1)

        n_remesh = len(new_pos)
        n_free = len(out_pos) - n_remesh
        if n_free < target_count:
            remesh_budget = target_count - n_free
            remesh_mag = combined_mag[:n_remesh]
            remesh_keep = np.argpartition(remesh_mag, -remesh_budget)[-remesh_budget:]
            keep_indices = np.concatenate(
                [remesh_keep, np.arange(n_remesh, len(out_pos), dtype=np.int64)]
            )
        else:
            free_mag = combined_mag[n_remesh:]
            free_keep = np.argpartition(free_mag, -target_count)[-target_count:] + n_remesh
            keep_indices = free_keep

        keep_mask = np.zeros(len(out_pos), dtype=bool)
        keep_mask[keep_indices] = True
        discarded_l1 = float(combined_mag[~keep_mask].sum())
        delta = np.maximum(np.maximum(lo - out_pos, out_pos - hi), 0.0)
        distance_sq = np.einsum("ij,ij->i", delta, delta)
        population_pruned_velocity_bound = float(
            np.sum(
                combined_mag[~keep_mask]
                / (
                    4.0
                    * np.pi
                    * np.maximum(distance_sq[~keep_mask] + out_rad[~keep_mask] ** 2, 1.0e-30)
                )
            )
        )
        population_pruned_fraction = discarded_l1 / (float(combined_mag.sum()) + 1e-30)
        population_pruned = int((~keep_mask).sum())

        out_pos = out_pos[keep_indices]
        out_circ = out_circ[keep_indices]
        out_vol = out_vol[keep_indices]
        out_rad = out_rad[keep_indices]
        out_circ = recover_invariants(
            out_pos,
            out_circ,
            combined_target,
            volumes=out_vol,
        )

    # ── Diagnostics ──────────────────────────────────────────────────────────
    # CFL = fraction of the active buffer a freestream particle crosses per
    # step; < ~0.7 guarantees hand-over to the free population happens while
    # the stencil is still interior.
    cfl = float(abs(u_max) * abs(dt) / (buffer_length + 1e-30))

    # Outflow-band vorticity-content ratio (downstream-most h-layer of the box).
    # Direction-agnostic: the outflow face is the one whose outward normal is
    # most aligned with the freestream (derived from ``u_inf``); defaults to
    # +x only when no direction is supplied.
    #
    # This is an L1 ratio Σ|Γ| / Σ|Γ|, NOT |ΣΓ| / |ΣΓ|.  The vector sum of ω
    # over a wake cross-section is ~0 (vortex lines close), so the ratio of
    # vector sums is a quotient of two near-cancelling quantities: it swung
    # over 0.02–29 on the cube case purely from cancellation noise, with no
    # corresponding change in the fields, and that reading was once
    # mis-attributed to the donor interior source. Raw ``grid_blended`` strengths
    # are also invalid here: the Beale correction deliberately deconvolves them,
    # whereas the FVM trace is a physical (mollified) circulation. Comparing
    # those two made the cube diagnostic climb to 2.5 while directly sampled
    # VPM/FVM vorticity remained near 1.0. Mollify the particle strengths first,
    # then sum magnitudes; 1.0 means physical-field agreement.
    flux_ratio = 0.0
    if len(grid_pos) > 0:
        band = _outflow_band_mask(grid_pos, lo, hi, h, u_inf)
        if band.any():
            mollified = _gaussian_mollified_circulation(
                grid_blended,
                shape,
                h,
                sigma=radius_ratio * h,
            ).reshape(-1, 3)
            g_vpm = float(np.linalg.norm(mollified[band], axis=1).sum())
            g_fvm = float(np.linalg.norm(target[band], axis=1).sum())
            flux_ratio = float(g_vpm / (g_fvm + 1e-30))

    return HandoffResult(
        pos=out_pos,
        circ=out_circ,
        vol=out_vol,
        rad=out_rad,
        n_remesh_in=int(in_region.sum()),
        n_remesh_out=int(len(new_pos)),
        n_free=int(free_mask.sum()),
        n_excluded=int(excluded_input.sum()),
        n_pruned=int(pruned.sum()),
        pruned_circulation_l1=pruned_l1,
        pruned_circulation_fraction=pruned_l1 / (active_l1 + 1e-30),
        n_population_pruned=population_pruned,
        population_pruned_circulation_fraction=population_pruned_fraction,
        population_pruned_velocity_bound=population_pruned_velocity_bound,
        cfl=cfl,
        conservation_drift=drift,
        conservation_raw_mismatch=raw_mismatch,
        conservation_applied_correction=applied_correction,
        conservation_corrected_mismatch=corrected_mismatch,
        flux_ratio=flux_ratio,
        strength_corr_residual_pre=corr_pre,
        strength_corr_residual_post=corr_post,
        excluded_input_circulation_l1=excluded_input_l1,
        excluded_remesh_circulation_l1=excluded_remesh_l1,
        excluded_target_circulation_l1=excluded_target_l1,
    )


# =========================================================
# Solver-facing wrapper (reads native FVM fields, writes the VPM field)
# =========================================================
class ContinuousOverlapInjector:
    """Thin FVM/VPM wrapper around :func:`continuous_handoff`."""

    def __init__(self, coupler):
        cfg = coupler.config
        self.config = cfg
        self.h = float(cfg.h)
        self.nu = float(cfg.nu)
        self.threshold_abs = float(cfg.prune_vorticity_min) * self.h**3

        self.ramp_width = float(cfg.buffer_thickness)
        self.dead_zone = float(cfg.dead_zone_h) * self.h
        self.radius_ratio = float(cfg.overlap_radius_ratio)
        self.u_inf = float(np.linalg.norm(cfg.u_inf))
        self.dt = float(coupler.dt_vpm)

        kernel = getattr(getattr(coupler, "vpm", None), "config", None)
        kernel = getattr(kernel, "particles_kernel", None)
        if kernel is not None and kernel != "GAUSSIAN":
            raise ValueError("The FVM–VPM handoff requires the GAUSSIAN particle kernel")

        self._box: np.ndarray | None = None
        self._cell_tree = None
        self._cell_centers: np.ndarray | None = None
        self._body_bounds: np.ndarray | None = None
        self._solid_bodies: tuple = ()
        self._lattice_anchor: np.ndarray | None = None
        self._velocity_trace: CachedVelocityTrace | None = None
        self.step = 0

    # ── setup ────────────────────────────────────────────────────────────────
    def setup(self, fvm):
        self._box = np.asarray(self.config.fvm_box, dtype=np.float64)
        from scipy.spatial import cKDTree

        self._cell_centers = np.asarray(fvm.get_cell_center_coordinates(), dtype=np.float64)
        # Non-master MPI ranks receive empty arrays from the gather; skip tree.
        # inject() is only called on master, so the None tree is never queried.
        if self._cell_centers.shape[0] > 0:
            self._cell_tree = cKDTree(self._cell_centers)
            self._velocity_trace = CachedVelocityTrace(
                self._cell_centers,
                self._cell_tree,
                neighbours=4,
            )

        # For an axis-aligned body the same exact bounds that carve the FVM
        # mesh also define the VPM exclusion region.  A nearest-cell query
        # cannot serve this purpose: it deliberately marks a tolerance shell
        # inside the hole as data-less Lagrangian space, allowing old solid
        # particles to survive forever.  The bounds come from the injected
        # solver's wall-patch geometry (single owner), and are used only when
        # every wall face lies on one of the six bounding planes (i.e. the
        # body really is an axis-aligned box).
        wall = self.config.wall_patch_name
        if wall:
            wf = np.asarray(
                fvm.get_boundary_face_center_coordinates(wall), dtype=np.float64
            ).reshape(-1, 3)
            if wf.shape[0] > 0:
                bounds = np.array(
                    [
                        wf[:, 0].min(),
                        wf[:, 0].max(),
                        wf[:, 1].min(),
                        wf[:, 1].max(),
                        wf[:, 2].min(),
                        wf[:, 2].max(),
                    ]
                )
                on_planes = np.zeros(len(wf), dtype=bool)
                for ax in range(3):
                    on_planes |= np.isclose(wf[:, ax], bounds[2 * ax], atol=1e-9)
                    on_planes |= np.isclose(wf[:, ax], bounds[2 * ax + 1], atol=1e-9)
                if on_planes.all():
                    self._body_bounds = bounds
                    self._lattice_anchor = bounds[[0, 2, 4]] - 0.5 * self.h
                else:
                    logger.warning(
                        "[Handoff] wall patch %r is not an axis-aligned box; no "
                        "exact particle exclusion mask is available",
                        wall,
                    )

        # Native immersed boundaries carry their exact interior geometry on
        # the same objects that define the FVM marker forcing.  Reuse that
        # single source of truth for particle exclusion instead of asking a
        # tutorial to duplicate masks or exposing filesystem paths.
        ibm = getattr(fvm, "ibm", None)
        bodies = tuple(getattr(ibm, "bodies", ()))
        self._solid_bodies = tuple(
            body for body in bodies if bool(getattr(body, "has_solid_geometry", False))
        )
        if self._solid_bodies:
            self._body_bounds = None
            if self._cell_centers is not None and len(self._cell_centers):
                self._lattice_anchor = self._cell_centers[0].copy()
            logger.info(
                "[Handoff] exact immersed-solid exclusion enabled for: %s",
                ", ".join(str(body.name) for body in self._solid_bodies),
            )

        l_buf = self.buffer_length
        logger.info(
            "[Handoff] ready: box x∈[%.2f,%.2f]  h=%.3f  σ=%.2fh  ramp=%.3f  "
            "dead_zone=%.3f  L_buf=%.3f  "
            "(CFL max_dt=%.3e s)  prune: |ω|<%.3g 1/s  (|Γ|<%.3g m³/s)",
            self._box[0],
            self._box[1],
            self.h,
            self.radius_ratio,
            self.ramp_width,
            self.dead_zone,
            l_buf,
            max_stable_dt(self.u_inf, l_buf, self.h),
            self.config.prune_vorticity_min,
            self.threshold_abs,
        )
        logger.info("[Handoff] weighted trace (k=4), aligned remesh, circulation cap")

    @property
    def buffer_length(self) -> float:
        return required_buffer_length(self.u_inf, self.dt, self.h)

    def _points_in_solid(self, points, *, include_boundary: bool) -> np.ndarray:
        """Union of native IBM solids and the legacy fitted-box solid."""
        query = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        inside = np.zeros(len(query), dtype=bool)
        for body in self._solid_bodies:
            inside |= np.asarray(
                body.contains(query, include_boundary=include_boundary), dtype=bool
            ).reshape(-1)
        if self._body_bounds is not None:
            bounds = self._body_bounds
            lo_body = bounds[[0, 2, 4]]
            hi_body = bounds[[1, 3, 5]]
            if include_boundary:
                inside |= np.all((query >= lo_body) & (query <= hi_body), axis=1)
            else:
                inside |= np.all((query > lo_body) & (query < hi_body), axis=1)
        return inside

    # ── inject ────────────────────────────────────────────────────────────────
    def inject(
        self,
        vpm,
        velocity,
        velocity_gradient,
    ):
        """Execute one continuous overlap hand-off and write the VPM field.

        The source is the weighted, gradient-corrected FVM velocity trace.
        """
        self.step += 1
        assert self._box is not None and self._cell_tree is not None

        n = vpm.particles.number_of_particles
        if n > 0:
            pos = np.asarray(vpm.particles_positions, dtype=np.float64).reshape(-1, 3)
            circ = np.asarray(vpm.particles_circulation, dtype=np.float64).reshape(-1, 3)
        else:
            pos = np.zeros((0, 3))
            circ = np.zeros((0, 3))

        tree = self._cell_tree
        h = self.h
        cell_pos = self._cell_centers
        assert cell_pos is not None

        velocity_values = np.asarray(velocity, dtype=np.float64).reshape(-1, 3)
        gradient_values = np.asarray(velocity_gradient, dtype=np.float64).reshape(-1, 3, 3)
        if len(velocity_values) != len(cell_pos) or len(gradient_values) != len(cell_pos):
            raise ValueError("FVM velocity, gradient, and cell-centre counts must match")

        def inside_mesh_at_node(grid_pos):
            d, _ = tree.query(grid_pos)
            return d < 1.5 * h

        def excluded_at_node(grid_pos):
            # The open interior is solid.  Nodes exactly on the surface may
            # carry a boundary vortex sheet and are therefore retained.
            return self._points_in_solid(grid_pos, include_boundary=False)

        def velocity_at(points):
            points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
            assert self._velocity_trace is not None
            sampled = self._velocity_trace.sample(points, velocity_values, gradient_values)
            on_or_inside = self._points_in_solid(points, include_boundary=True)
            sampled[on_or_inside] = 0.0
            return sampled

        def circulation_at_node(grid_pos):
            return circulation_from_velocity_trace(grid_pos, h, velocity_at)

        res = continuous_handoff(
            pos,
            circ,
            self._box,
            h,
            circulation_at_node=circulation_at_node,
            u_inf=self.config.U_inf,
            inside_mesh_at_node=inside_mesh_at_node,
            excluded_at_node=excluded_at_node,
            ramp_width=self.ramp_width,
            dead_zone=self.dead_zone,
            buffer_length=self.buffer_length,
            threshold_abs=self.threshold_abs,
            radius_ratio=self.radius_ratio,
            u_max=self.u_inf,
            dt=self.dt,
            lattice_anchor=self._lattice_anchor,
            max_output_particles=self.config.handoff_max_particles,
        )

        vpm_dt = getattr(vpm.particles, "_np_float_dtype", np.float32)
        k = res.n_total
        vpm.replace_vortex_particles(
            position=res.pos.astype(vpm_dt),
            velocity=np.zeros((k, 3), dtype=vpm_dt),
            circulation=res.circ.astype(vpm_dt),
            radius=res.rad.astype(vpm_dt),
            volume=res.vol.astype(vpm_dt),
            viscosity=np.full(k, self.nu, dtype=vpm_dt),
            viscosity_turbulent=np.zeros(k, dtype=vpm_dt),
            zone_id=np.zeros(k, dtype=np.int32),
        )

        logger.info(
            "[Handoff step=%d] in=%d → out=%d  free=%d  solid_removed=%d  "
            "pruned=%d (%.3f%% Σ|Γ|)  cap_pruned=%d (%.3f%% Σ|Γ|, "
            "δu_bound=%.3e m/s)  "
            "CFL=%.2f  |ΔΓ|/|Γ|=%.2e  flux_ratio=%.3f",
            self.step,
            res.n_remesh_in,
            res.n_remesh_out,
            res.n_free,
            res.n_excluded,
            res.n_pruned,
            100.0 * res.pruned_circulation_fraction,
            res.n_population_pruned,
            100.0 * res.population_pruned_circulation_fraction,
            res.population_pruned_velocity_bound,
            res.cfl,
            res.conservation_drift.get("circulation_rel", 0.0),
            res.flux_ratio,
        )
        if (
            res.excluded_input_circulation_l1 > 0.0
            or res.excluded_remesh_circulation_l1 > 0.0
            or res.excluded_target_circulation_l1 > 0.0
        ):
            logger.info(
                "     [Body mask] removed Σ|Γ|: input=%.3e  remesh=%.3e  target=%.3e",
                res.excluded_input_circulation_l1,
                res.excluded_remesh_circulation_l1,
                res.excluded_target_circulation_l1,
            )
        logger.info(
            "     [Beale] mollification residual: %.1f%% → %.1f%%",
            100.0 * res.strength_corr_residual_pre,
            100.0 * res.strength_corr_residual_post,
        )
        if res.cfl > 0.7:
            logger.warning(
                "[Handoff] CFL=%.2f > 0.7 — buffer too short for this dt; "
                "reduce dt (max_dt≈%.3e s).",
                res.cfl,
                max_stable_dt(self.u_inf, self.buffer_length, self.h),
            )

        return res
