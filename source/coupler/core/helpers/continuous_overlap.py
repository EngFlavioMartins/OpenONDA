"""Conservative overlap transport from FVM cells to VPM particles.

The hand-off remeshes onto a regular lattice, blends FVM and VPM circulation
with a smooth partition of unity, and recovers selected integral invariants
after pruning. The implementation is independent of the FVM and VPM runtimes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging

import numpy as np

from source.coupler.diagnostics.injection_correction import recover_invariants
from source.coupler.remesh import remesh_to_grid

logger = logging.getLogger("coupler")

# Particle core radius relative to the regular overlap lattice spacing.
RADIUS_RATIO = 1.5

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
    zero vector) it defaults to the +x face, preserving the legacy behaviour.
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
    iterations: int,
    relax: float = 1.0,
) -> tuple[np.ndarray, float, float]:
    """Iterate lattice strengths against Gaussian-mollified target vorticity.

    The partition weight localizes the correction to FVM-controlled nodes.
    ``iterations=0`` leaves the blended circulation unchanged.

    Parameters
    ----------
    circ_grid   : (M, 3) blended node circulations Γ [m³/s] (flat, C-order)
    target_circ : (M, 3) FVM target circulations ω_FVM·h³ on the same lattice
    eta         : (M,)  correction weight in [0, 1]
    shape       : lattice dimensions (Nx, Ny, Nz)
    h           : lattice spacing [m]
    sigma       : particle core radius σ in the e^{-r²/σ²} convention [m]
    iterations  : number of Picard iterations M (2–3 recovers resolved scales)
    relax       : under-relaxation λ ∈ (0, 1]

    Returns
    -------
    (corrected (M, 3) circulations, η-weighted relative residual before,
     same residual after)
    """
    from scipy.ndimage import gaussian_filter

    # ζ_σ ∝ e^{-r²/σ²} = e^{-r²/(2 s²)} with s = σ/√2; gaussian_filter takes
    # s in grid cells and normalizes its discrete kernel to Σw = 1, which for
    # s ≳ 1 cell equals the sampled ζ_σ·h³ weights to machine precision.
    s_cells = float(sigma) / (np.sqrt(2.0) * float(h))
    h3 = float(h) ** 3

    g = np.asarray(circ_grid, dtype=np.float64).reshape(*shape, 3).copy()
    t_omega = np.asarray(target_circ, dtype=np.float64).reshape(*shape, 3) / h3
    eta_g = np.asarray(eta, dtype=np.float64).reshape(*shape)[..., None]

    def omega_sigma(gc: np.ndarray) -> np.ndarray:
        return (
            np.stack(
                [
                    gaussian_filter(gc[..., c], s_cells, mode="constant", truncate=5.0)
                    for c in range(3)
                ],
                axis=-1,
            )
            / h3
        )

    denom = float(np.linalg.norm(t_omega * eta_g)) + 1e-30
    res_pre = res_post = 0.0
    for m in range(iterations + 1):
        r = (t_omega - omega_sigma(g)) * eta_g
        rn = float(np.linalg.norm(r)) / denom
        if m == 0:
            res_pre = rn
        res_post = rn
        if m == iterations:
            break
        g += relax * r * h3

    return g.reshape(-1, 3), res_pre, res_post


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
    flux_ratio: float = 0.0  # |Γ_VPM_exit| / |Γ_FVM_exit| at the outflow band

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
        return self.n_remesh_out + self.n_free


# =========================================================
# Pure-NumPy numerical core
# =========================================================
def continuous_handoff(
    pos: np.ndarray,
    circ: np.ndarray,
    box: np.ndarray | list[float],
    h: float,
    *,
    omega_at_node=None,
    fvm_cell_pos=None,
    fvm_cell_circ=None,
    u_inf=None,
    inside_mesh_at_node=None,
    excluded_at_node=None,
    ramp_width: float | None = None,
    dead_zone: float = 0.0,
    buffer_length: float = 0.0,
    threshold_abs: float = 0.0,
    radius_ratio: float = RADIUS_RATIO,
    blend_relaxation: float = 1.0,
    strength_correction_iterations: int = 0,
    strength_correction_relax: float = 1.0,
    u_max: float = 0.0,
    dt: float = 0.0,
    conserve: bool = True,
    eta_fn=None,
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
    fvm_cell_pos, fvm_cell_circ : ndarray (M, 3) or None
        Conservative FVM→lattice source: cell centres and their circulation
        ``Γ_cell = ω_cell · V_cell``, scattered onto the lattice with the same
        M4′ kernel as the wake.  The injected target circulation equals
        ``Σω_cell·V_cell`` exactly, independent of the FVM mesh refinement.
    u_inf : ndarray (3,) or None
        Freestream velocity vector used to orient the outflow-band diagnostic.
    omega_at_node : callable ``grid_pos -> ω (M, 3)`` or None
        Test/no-body path (``target = ω·h³``); ignored when ``fvm_cell_pos``
        is given.  When both are None the hand-off is pure transport.
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
    blend_relaxation : float
        Under-relaxation α ∈ (0, 1] of the η-blend toward the FVM target.
        α = 1 is the hard overwrite ``(1−η)Γ + η·target``.
    strength_correction_iterations : int
        Beale/Picard iterations after the blend (0 = direct assignment).
        Body-guarded: see :func:`beale_strength_correction` and the inline
        comment at the call site.
    strength_correction_relax : float
        Under-relaxation λ ∈ (0, 1] of the Beale iteration.
    u_max, dt : float
        Used only for the CFL diagnostic.
    conserve : bool
        If True, redistribute pruned moments so the invariants are exact.

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

    # When the FVM cell centres themselves form an h-lattice (the native cube
    # case), preserve that phase exactly.  The previous origin was determined
    # by the dt-dependent buffer length; for dt=0.075 and h=0.05 this shifted
    # the remesh lattice by h/4.  M4' is interpolating only at integer offsets,
    # so the shift spread every wall-adjacent FVM circulation over the solid
    # before the body mask could act.
    if fvm_cell_pos is not None and len(fvm_cell_pos) > 0:
        source_pos = np.asarray(fvm_cell_pos, dtype=np.float64).reshape(-1, 3)
        anchor = source_pos[0]
        q = (source_pos - anchor) / float(h)
        lattice_residual = np.max(np.abs(q - np.rint(q)), axis=0)
        aligned_axes = lattice_residual < 1e-7
        if np.any(aligned_axes):
            shift = np.floor((lo_lat - anchor) / float(h))
            lo_aligned = anchor + shift * float(h)
            lo_lat[aligned_axes] = lo_aligned[aligned_axes]

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
    grid_pos, grid_circ = remesh_to_grid(pos[in_region], circ[in_region], lo_lat, h, shape)

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
    if eta_fn is not None:
        eta = np.asarray(eta_fn(grid_pos), dtype=np.float64).reshape(-1)
    else:
        eta = cosine_eta(grid_pos, box, ramp_width, dead_zone)

    # ── Build the FVM target circulation on the lattice ──────────────────────
    target = None
    if fvm_cell_pos is not None and fvm_cell_circ is not None and len(fvm_cell_pos) > 0:
        _, target = remesh_to_grid(
            np.asarray(fvm_cell_pos, dtype=np.float64).reshape(-1, 3),
            np.asarray(fvm_cell_circ, dtype=np.float64).reshape(-1, 3),
            lo_lat,
            h,
            shape,
        )
    elif omega_at_node is not None:
        target = np.asarray(omega_at_node(grid_pos), dtype=np.float64).reshape(-1, 3) * (h**3)

    excluded_target_l1 = 0.0
    if target is not None and excluded_grid.any():
        excluded_target_l1 = float(np.linalg.norm(target[excluded_grid], axis=1).sum())
        target[excluded_grid] = 0.0

    # ── Blend toward the FVM target where η > 0 (under-relaxed by α) ──────────
    ok = None
    if target is not None:
        if inside_mesh_at_node is not None:
            ok = np.asarray(inside_mesh_at_node(grid_pos), dtype=bool)
            eta = eta * ok  # no FVM data → pure Lagrangian
            target[~ok] = 0.0
        grid_blended = grid_circ + (blend_relaxation * eta)[:, None] * (target - grid_circ)
    else:
        # Pure transport: no FVM source — leave the remeshed wake untouched.
        grid_blended = grid_circ

    # ── Beale/Picard strength correction (η-localized deconvolution) ─────────
    # BODY GUARD: the target ends in a step at the body wall (no cells inside).
    # Deconvolving across that discontinuity produces Gibbs-like ringing with
    # up-to-(M+1)× amplification in a ±2σ shell — including nodes just inside
    # the wall (which are `ok` because they sit within 1.5h of exterior cell
    # centres), injecting circulation INSIDE the body.  Restrict the correction
    # to nodes whose entire kernel support sees valid FVM data: erode the ok
    # mask by the kernel support (≈ 2σ/h cells) before weighting.
    corr_pre = corr_post = 0.0
    if strength_correction_iterations > 0 and target is not None:
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
            iterations=strength_correction_iterations,
            relax=strength_correction_relax,
        )

    # Deconvolution and remeshing have non-compact physical kernels near a
    # wall.  Re-assert the exact solid mask after every correction so no
    # particle centre can be regenerated inside the body.
    grid_blended[excluded_grid] = 0.0

    # ── Conservative prune: drop weak nodes, redistribute their moments ──────
    target_inv = _invariants(grid_pos, grid_blended)
    mag = np.linalg.norm(grid_blended, axis=1)
    keep = (mag >= threshold_abs) & ~excluded_grid
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
    if conserve and len(new_pos) > 0:
        new_vol_tmp = np.full(len(new_pos), h**3)
        # Angular impulse is intentionally excluded because the remeshing
        # operator does not conserve that second moment.
        new_circ = recover_invariants(
            new_pos,
            new_circ,
            target_inv,
            volumes=new_vol_tmp,
            weighting="volume",
            conserve_circulation=True,
            conserve_linear_impulse=True,
            conserve_angular_impulse=False,
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

    # ── Diagnostics ──────────────────────────────────────────────────────────
    # CFL = fraction of the active buffer a freestream particle crosses per
    # step; < ~0.7 guarantees hand-over to the free population happens while
    # the stencil is still interior.
    cfl = float(abs(u_max) * abs(dt) / (buffer_length + 1e-30))

    # Outflow-band vorticity-flux ratio (downstream-most h-layer of the box).
    # Direction-agnostic: the outflow face is the one whose outward normal is
    # most aligned with the freestream (derived from ``u_inf``); defaults to
    # +x only when no direction is supplied.
    flux_ratio = 0.0
    if target is not None and len(grid_pos) > 0:
        band = _outflow_band_mask(grid_pos, lo, hi, h, u_inf)
        if band.any():
            g_vpm = np.linalg.norm(np.sum(grid_blended[band], axis=0))
            g_fvm = np.linalg.norm(np.sum(target[band], axis=0))
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
# Solver-facing wrapper (reads OpenFOAM, writes the VPM field)
# =========================================================
class ContinuousOverlapInjector:
    """Thin FVM/VPM wrapper around :func:`continuous_handoff`."""

    def __init__(self, coupler):
        cfg = coupler.config
        self.config = cfg
        self.h = float(cfg.h)
        self.nu = float(cfg.nu)
        self.threshold_abs = float(cfg.prune_vorticity_min) * self.h**3

        # η geometry
        self.ramp_width = float(cfg.buffer_thickness)
        self.dead_zone = float(cfg.dead_zone_h) * self.h
        if self.ramp_width <= self.dead_zone:
            logger.warning(
                "[Handoff] η ramp has ZERO width (buffer_thickness=%.3f ≤ "
                "dead_zone=%.3f): the FVM-authority weight is a step function, "
                "not the designed C¹ ramp.  Set buffer_thickness > dead_zone_h·h "
                "(e.g. buffer_thickness=%.3f for a %d-cell ramp).",
                self.ramp_width,
                self.dead_zone,
                self.dead_zone + 3.0 * self.h,
                3,
            )

        # Particle core radius σ = radius_ratio·h (sets Beale deconvolution bandwidth)
        self.radius_ratio = float(cfg.overlap_radius_ratio)
        if self.radius_ratio < 1.0:
            logger.warning(
                "[Handoff] overlap_radius_ratio=%.2f < 1.0 breaks particle "
                "overlap — the Biot-Savart field will ripple between particles.",
                self.radius_ratio,
            )

        # dt-robust downstream buffer, sized from U_inf and dt (CFL criterion)
        self.u_inf = float(np.linalg.norm(cfg.u_inf))
        self.dt = float(coupler.dt_vpm)

        self.blend_relaxation = float(cfg.blend_relaxation)
        self.strength_corr_iters = int(cfg.strength_correction_iterations)
        self.strength_corr_relax = float(cfg.strength_correction_relax)
        # The Beale deconvolution assumes the GAUSSIAN mollifier; read the
        # actual kernel from the injected VPM (the coupling config no longer
        # owns it).  Skipped on non-master (coupler.vpm is None there).
        kernel = getattr(getattr(coupler, "vpm", None), "config", None)
        kernel = getattr(kernel, "particles_kernel", None)
        if self.strength_corr_iters > 0 and kernel is not None and kernel != "GAUSSIAN":
            logger.warning(
                "[Handoff] strength_correction assumes the GAUSSIAN particle "
                "kernel (mollifier e^{-r²/σ²}); the injected VPM uses %s — the "
                "deconvolution operator will not match the induced field.",
                kernel,
            )

        self._box: np.ndarray | None = None
        self._cell_tree = None
        self._cell_centers: np.ndarray | None = None
        self._cell_volumes: np.ndarray | None = None
        self._body_bounds: np.ndarray | None = None
        self.step = 0

    # ── setup ────────────────────────────────────────────────────────────────
    def setup(self, fvm):
        self._box = np.asarray(self.config.fvm_box, dtype=np.float64)
        from scipy.spatial import cKDTree

        self._cell_centers = np.asarray(fvm.get_cell_center_coordinates(), dtype=np.float64)
        self._cell_volumes = np.asarray(fvm.get_cell_volumes(), dtype=np.float64)
        # Non-master MPI ranks receive empty arrays from the gather; skip tree.
        # inject() is only called on master, so the None tree is never queried.
        if self._cell_centers.shape[0] > 0:
            self._cell_tree = cKDTree(self._cell_centers)

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
                else:
                    logger.warning(
                        "[Handoff] wall patch %r is not an axis-aligned box; no "
                        "exact particle exclusion mask is available",
                        wall,
                    )

        l_buf = self.buffer_length
        logger.info(
            "[Handoff] ready: box x∈[%.2f,%.2f]  h=%.3f  σ=%.2fh  ramp=%.3f  "
            "dead_zone=%.3f  L_buf=%.3f  α=%.2f  beale_iters=%d  "
            "(CFL max_dt=%.3e s)  prune: |ω|<%.3g 1/s  (|Γ|<%.3g m³/s)",
            self._box[0],
            self._box[1],
            self.h,
            self.radius_ratio,
            self.ramp_width,
            self.dead_zone,
            l_buf,
            self.blend_relaxation,
            self.strength_corr_iters,
            max_stable_dt(self.u_inf, l_buf, self.h),
            self.config.prune_vorticity_min,
            self.threshold_abs,
        )

    @property
    def buffer_length(self) -> float:
        return required_buffer_length(self.u_inf, self.dt, self.h)

    # ── inject ────────────────────────────────────────────────────────────────
    def inject(self, fvm, vpm, eta_fn=None, omega=None):
        """Execute one continuous overlap hand-off and write the VPM field.

        Parameters
        ----------
        fvm : fvm_solver
        vpm : VPM_Solver
        eta_fn : callable, optional
        omega : ndarray (N_cells, 3), optional
            Pre-fetched global vorticity field.  If provided, skips the
            collective ``fvm.get_vorticity_field()`` call.  Pass when this
            method is called inside a rank-0-only section so other ranks can
            pre-fetch omega collectively before the gate.
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

        if omega is None:
            omega = np.asarray(fvm.get_vorticity_field(), dtype=np.float64).reshape(-1, 3)
        else:
            omega = np.asarray(omega, dtype=np.float64).reshape(-1, 3)
        tree = self._cell_tree
        h = self.h

        # Conservative FVM→lattice source Γ_cell = ω_cell·V_cell, restricted to
        # cells that can actually deposit on the lattice.
        cell_pos = self._cell_centers
        cell_circ = omega * self._cell_volumes[:, None]
        box = self._box
        lo = np.array([box[0], box[2], box[4]])
        hi = np.array([box[1], box[3], box[5]])
        margin = self.buffer_length + (_M4P_SUPPORT + 2.0) * h
        in_bbox = np.all((cell_pos >= lo - margin) & (cell_pos <= hi + margin), axis=1)

        def inside_mesh_at_node(grid_pos):
            d, _ = tree.query(grid_pos)
            return d < 1.5 * h

        def excluded_at_node(grid_pos):
            if self._body_bounds is None:
                return np.zeros(len(grid_pos), dtype=bool)
            b = self._body_bounds
            lo_body = b[[0, 2, 4]]
            hi_body = b[[1, 3, 5]]
            # The open interior is solid.  Nodes exactly on the surface may
            # carry a boundary vortex sheet and are therefore retained.
            return np.all((grid_pos > lo_body) & (grid_pos < hi_body), axis=1)

        res = continuous_handoff(
            pos,
            circ,
            self._box,
            h,
            fvm_cell_pos=cell_pos[in_bbox],
            fvm_cell_circ=cell_circ[in_bbox],
            u_inf=self.config.U_inf,
            inside_mesh_at_node=inside_mesh_at_node,
            excluded_at_node=excluded_at_node,
            ramp_width=self.ramp_width,
            dead_zone=self.dead_zone,
            buffer_length=self.buffer_length,
            threshold_abs=self.threshold_abs,
            radius_ratio=self.radius_ratio,
            blend_relaxation=self.blend_relaxation,
            strength_correction_iterations=self.strength_corr_iters,
            strength_correction_relax=self.strength_corr_relax,
            u_max=self.u_inf,
            dt=self.dt,
            eta_fn=eta_fn,
        )

        # Write the rebuilt cloud back to the VPM solver.
        # Match the VPM's configured float precision so the hand-off is exact
        # (no f32←f64 / f64←f32 precision-loss warnings) and a precision='f64'
        # coupled run stays end-to-end double-precision.  ``particles._np_float_dtype``
        # is the ground-truth dtype of the Taichi fields we write into.
        vpm_dt = getattr(vpm.particles, "_np_float_dtype", np.float32)
        k = res.n_total
        if hasattr(vpm, "replace_vortex_particles"):
            vpm.replace_vortex_particles(
                position=res.pos.astype(vpm_dt),
                velocity=np.zeros((k, 3), dtype=vpm_dt),
                circulation=res.circ.astype(vpm_dt),
                radius=res.rad.astype(vpm_dt),
                volume=res.vol.astype(vpm_dt),
                viscosity=np.full(k, self.nu, dtype=vpm_dt),
                # ν_t starts at zero: the VPM Smagorinsky model computes the
                # real eddy viscosity before diffusion (solver.py:743).  Stamping
                # ν_t = ν here previously made ν_eff = 2ν whenever LES was off.
                viscosity_turbulent=np.zeros(k, dtype=vpm_dt),
                zone_id=np.zeros(k, dtype=np.int32),
            )
        else:
            vpm.remove_particles(remove_all=True)
            if k > 0:
                vpm.add_vortex_particles(
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
            "CFL=%.2f  |ΔΓ|/|Γ|=%.2e  flux_ratio=%.3f",
            self.step,
            res.n_remesh_in,
            res.n_remesh_out,
            res.n_free,
            res.n_excluded,
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
        if self.strength_corr_iters > 0:
            logger.info(
                "     [Beale] mollification residual: %.1f%% → %.1f%%  (%d iters, λ=%.2f)",
                100.0 * res.strength_corr_residual_pre,
                100.0 * res.strength_corr_residual_post,
                self.strength_corr_iters,
                self.strength_corr_relax,
            )
        if res.cfl > 0.7:
            logger.warning(
                "[Handoff] CFL=%.2f > 0.7 — buffer too short for this dt; "
                "reduce dt (max_dt≈%.3e s).",
                res.cfl,
                max_stable_dt(self.u_inf, self.buffer_length, self.h),
            )

        return res
