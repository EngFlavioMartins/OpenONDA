"""
Continuous Overlap Transport with Conservative Hand-off.
========================================================

The FVM→VPM particle hand-off, built on three guarantees:

  1. **Conservation (no trapping).**  Vorticity is a conserved quantity
     (Helmholtz / Kelvin): no operation here ever discards circulation.
     Pruning of weak particles is followed by a moment-conserving
     redistribution (``recover_invariants``) so total circulation and
     linear impulse are invariant under the prune.

  2. **Time-step insensitivity.**  The overlap region *plus a downstream
     buffer* is re-meshed onto a regular h-lattice every step, so particle
     density is independent of the local convective velocity.  The buffer
     is sized from the CFL-like criterion ``L_buf ≥ 1.5·U_max·dt + 2h`` so
     a particle advecting at freestream speed always lands on a node that
     still exists on the next step.

  3. **Continuous interface.**  FVM authority is a single C¹ partition-of-
     unity weight η(x): η = 1 deep inside the FVM core, ramping smoothly to
     0 over ``ramp_width``, held at 0 in a dead-zone at every face and
     outside the box.  The blended source

         Γ_node = (1 − η)·Γ_remeshed  +  η·(ω_FVM · h³)

     is C¹ across Γ, so the induced Biot–Savart velocity is continuous.
     Exiting particles (η = 0 band) carry pure Lagrangian circulation.

  After the blend, a Beale/Picard strength correction
  (:func:`beale_strength_correction`) deconvolves the Gaussian-core
  mollification on resolved scales, body-guarded so it never acts across
  the wall discontinuity.

References
----------
- Cottet & Koumoutsakos (2000), *Vortex Methods*, §5 (hybrid overlap),
  §7 (M4′ remeshing & moment conservation).
- Winckelmans (1993), PhD thesis — integral invariants of vortex methods.
- Beale (1988) — iterated vortex-strength assignment.
- Daeninck & Winckelmans (2003); Stock, Gharakhani & Stone (2010);
  Billuart et al. (2023) — Eulerian/Lagrangian overlap coupling.

The numerical core (:func:`continuous_handoff`) has no Taichi / OpenFOAM
dependency and is unit-tested in ``tests/coupler/test_continuous_overlap.py``
and ``tests/coupler/test_strength_correction.py``.
:class:`ContinuousOverlapInjector` is the thin solver-facing wrapper that
reads OpenFOAM and writes the VPM particle field.

Author:  OpenONDA Team
Date:    June 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging

import numpy as np

from source.coupler.diagnostics.injection_correction import recover_invariants
from source.coupler.remesh import remesh_to_grid

logger = logging.getLogger("coupler")

# Particle core radius = RADIUS_RATIO × h  (overlap-region particles are on a
# regular h-lattice, so their nominal volume is h³).
RADIUS_RATIO = 1.5

# M4′ kernel support half-width (in cells).  A particle must stay within this
# many cells of the lattice interior for the scatter to be exact, which sets
# the guard band and the CFL buffer margin.
_M4P_SUPPORT = 2.0

# =========================================================
# FIX B: velocity-based target circulation (Billuart 2023 §3.3)
# Conservative via the discrete Stokes theorem on the data-extent sub-grid
# (NOT the full lattice — see _velocity_based_target docstring for the
# zero-circulation defect that motivated the sub-grid + axis-wise hole fill).
# =========================================================
def _fill_holes_axiswise(u_sub: np.ndarray, mask_sub: np.ndarray) -> np.ndarray:
    """Fill ``mask=False`` nodes in a 3-D sub-grid by axis-wise edge padding.

    Forward/backward fill of the nearest ``mask=True`` value along each axis
    (cascading x → y → z).  The interior (``mask=True``) is left untouched; only
    the bounding-box holes created by the non-rectangular M4′-support data
    region (Euclidean 2h support erodes the corners of the axis-aligned data
    extent) are filled.  This gives a Neumann condition at the data boundary so
    the lattice curl sees a smooth field instead of a zero discontinuity that
    ``np.gradient`` would amplify into a spurious vorticity sheet.
    """
    u = u_sub.copy()
    m = mask_sub.astype(bool)
    for ax in range(3):
        u = np.swapaxes(u, 0, ax)
        m = np.swapaxes(m, 0, ax)
        for i in range(1, u.shape[0]):
            empty = ~m[i]
            if empty.any():
                u[i, empty] = u[i - 1, empty]
                m[i, empty] = m[i - 1, empty]
        for i in range(u.shape[0] - 2, -1, -1):
            empty = ~m[i]
            if empty.any():
                u[i, empty] = u[i + 1, empty]
                m[i, empty] = m[i + 1, empty]
        u = np.swapaxes(u, 0, ax)
    return u

def _curl_on_regular_grid(u_grid: np.ndarray, shape: tuple[int, int, int], h: float) -> np.ndarray:
    """Compute ω = ∇×u on a regular h-lattice via 2nd-order central differences.

    Parameters
    ----------
    u_grid : (Nx*Ny*Nz, 3)  velocity field on the lattice (flat, C-order)
    shape  : (Nx, Ny, Nz)
    h      : lattice spacing

    Returns
    -------
    omega_grid : (Nx*Ny*Nz, 3)  vorticity on the lattice (flat)
    """
    u = u_grid.reshape(*shape, 3)  # (Nx, Ny, Nz, 3)
    ux, uy, uz = u[..., 0], u[..., 1], u[..., 2]

    # Central differences with one-sided at boundaries (Neumann edge)
    def dcd(f, axis):
        return np.gradient(f, h, axis=axis, edge_order=1)

    omega = np.stack(
        [
            dcd(uz, 1) - dcd(uy, 2),  # ω_x = ∂u_z/∂y - ∂u_y/∂z
            dcd(ux, 2) - dcd(uz, 0),  # ω_y = ∂u_x/∂z - ∂u_z/∂x
            dcd(uy, 0) - dcd(ux, 1),  # ω_z = ∂u_y/∂x - ∂u_x/∂y
        ],
        axis=-1,
    )
    return omega.reshape(-1, 3)

def _velocity_based_target(
    fvm_cell_pos: np.ndarray,
    fvm_cell_vel: np.ndarray,
    fvm_cell_vol: np.ndarray,
    u_inf: np.ndarray,
    lo_lat: np.ndarray,
    h: float,
    shape: tuple[int, int, int],
) -> np.ndarray:
    """Conservative velocity-based target circulation on the VPM lattice.

    Interpolates the FVM velocity onto the lattice (scatter ``u·V`` and ``V``
    separately, divide to get the volume-weighted velocity ``u_lat``), then
    computes ``ω = ∇×u`` via central differences on the **data-extent
    sub-grid** — the bounding box of lattice nodes that actually received FVM
    data (Billuart et al., JCP 2023, §3.3).

    Why the sub-grid, not the full lattice: the lattice extends a full
    ``buffer_length + 2h`` beyond ``fvm_box`` and **no FVM cells live there**,
    so ``u_lat = 0`` in the buffer.  Taking the curl on the full lattice makes
    the discrete Stokes sum telescope to the *lattice* boundary, where
    ``n̂×u_lat = 0`` ⇒ ``Σ_node Γ ≡ 0`` for every flow — the hand-off could not
    transfer the shed wake.  Restricting the curl to the data-extent sub-grid
    closes discrete Stokes over the *data* boundary (≈ the box faces):

        Σ_node Γ = ∮_∂data (n̂ × u_lat) dS ≈ ∫_box ω_FVM dV

    to interpolation accuracy (the data boundary sits ~2h beyond the box via
    the M4′ support, so for a vortex compact inside the box the two integrals
    coincide; for vorticity that reaches the faces the velocity path sums over
    the slightly enlarged data extent — the same smear the vorticity path's M4′
    scatter applies).  The sub-grid edges use one-sided differences
    (``edge_order=1``), i.e. a Neumann condition at the data boundary, so no
    spurious vorticity sheet is manufactured at the coupling faces — the
    cubeFlow interface-noise pathology.

    The configured freestream ``u_inf`` is subtracted before the scatter so the
    interpolated field is the perturbation velocity.  ``∇×U∞ ≡ 0`` (exactly, in
    discrete too), so the curl is unchanged by the subtraction; it makes the
    intent explicit and keeps the field fed to the curl a small perturbation
    rather than ``U∞`` plus a zero-cFO cutoff.

    Returns
    -------
    target : (Nx*Ny*Nz, 3)  target circulation Γ = ω · h³ on the lattice
    """
    pos = np.asarray(fvm_cell_pos, dtype=np.float64).reshape(-1, 3)
    vel = np.asarray(fvm_cell_vel, dtype=np.float64).reshape(-1, 3)
    # Subtract the freestream: the curl of a constant is zero, so this does not
    # change ω, but it makes u_lat the perturbation field (no U∞ step at the
    # data boundary) and is robust to any future nonlinear use of u_lat.
    u_rel = vel - np.asarray(u_inf, dtype=np.float64).reshape(1, 3)
    vol = np.asarray(fvm_cell_vol, dtype=np.float64).ravel()

    # Scatter u·V  → volume-weighted velocity on lattice
    _, u_weighted = remesh_to_grid(pos, u_rel * vol[:, None], lo_lat, h, shape)
    # Scatter V    → volume weight on lattice
    vol_vec = np.stack([vol, vol, vol], axis=1)
    _, vol_weight = remesh_to_grid(pos, vol_vec, lo_lat, h, shape)

    # Interpolated velocity = Σ(u·V·W) / Σ(V·W)  (volume-weighted average)
    vw = vol_weight[:, 0].copy()  # all 3 components of vol_weight are equal
    mask = vw > 1e-30
    u_lat = np.zeros_like(u_weighted)
    u_lat[mask] = u_weighted[mask] / vw[mask][:, None]

    # ω = ∇×u on the data-extent sub-grid only (see docstring: closing Stokes
    # over the data boundary instead of the zero-padded lattice boundary).
    omega_lat = np.zeros_like(u_lat)
    if mask.any():
        mask3 = mask.reshape(shape)
        any_x = np.asarray(mask3.any(axis=(1, 2)))
        any_y = np.asarray(mask3.any(axis=(0, 2)))
        any_z = np.asarray(mask3.any(axis=(0, 1)))
        i_lo = np.array([int(np.argmax(any_x)), int(np.argmax(any_y)), int(np.argmax(any_z))])
        i_hi = np.array(
            [
                int(len(any_x) - np.argmax(any_x[::-1]) - 1),
                int(len(any_y) - np.argmax(any_y[::-1]) - 1),
                int(len(any_z) - np.argmax(any_z[::-1]) - 1),
            ]
        )
        sub_shape = (
            int(i_hi[0] - i_lo[0] + 1),
            int(i_hi[1] - i_lo[1] + 1),
            int(i_hi[2] - i_lo[2] + 1),
        )
        u3 = u_lat.reshape(shape + (3,))
        m_sub = mask3[i_lo[0] : i_hi[0] + 1, i_lo[1] : i_hi[1] + 1, i_lo[2] : i_hi[2] + 1]
        u_sub = u3[i_lo[0] : i_hi[0] + 1, i_lo[1] : i_hi[1] + 1, i_lo[2] : i_hi[2] + 1, :]
        # The data region is non-rectangular (Euclidean M4′ support erodes the
        # corners of the axis-aligned bounding box), so the sub-grid has
        # mask=False holes where u_lat = 0.  Fill them by axis-wise edge padding
        # (Neumann) before the curl, else np.gradient amplifies the zero
        # discontinuity into a spurious vorticity sheet at every hole.
        if (~m_sub).any():
            u_sub = _fill_holes_axiswise(u_sub, m_sub)
        omega_sub = _curl_on_regular_grid(u_sub.reshape(-1, 3), sub_shape, h)
        omega3 = omega_lat.reshape(shape + (3,))
        omega3[i_lo[0] : i_hi[0] + 1, i_lo[1] : i_hi[1] + 1, i_lo[2] : i_hi[2] + 1, :] = (
            omega_sub.reshape(sub_shape + (3,))
        )

    # Target circulation = ω · h³
    return omega_lat * (h**3)

# =========================================================
# Integral invariants
# =========================================================
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

# =========================================================
# Geometry / CFL helpers
# =========================================================
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

# =========================================================
# Beale/Picard iterated strength assignment (regularized deconvolution)
# =========================================================
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
    """Iterate lattice strengths so the *mollified* particle vorticity matches
    the FVM target at the nodes (Beale 1988).

    Direct assignment ``Γ_node = ω_FVM·h³`` matches the *bare* circulation, but
    the velocity the VPM induces comes from the mollified field

        ω_σ(x) = Σ_p Γ_p ζ_σ(x − x_p),      ζ_σ(r) = π^{-3/2}σ^{-3} e^{-r²/σ²},

    which attenuates each Fourier mode by exp(−k²σ²/4) — a systematic low-pass
    bias (≈20 % on 0.5 D wake structures for σ = 1.5h).  The Picard iteration

        Γ ← Γ + λ·η·(ω_target − ω_σ[Γ])·h³

    is a regularized deconvolution: after M iterations the residual at
    wavenumber k is (1 − e^{-k²σ²/4})^{M+1} — resolved scales converge
    geometrically, sub-kernel scales are deliberately left untouched
    (amplification bounded by M+1).

    The η weight localizes the correction; the free Lagrangian wake (η = 0)
    keeps its standard remeshed strengths.  Because particles sit on the
    regular h-lattice, ω_σ is a separable Gaussian convolution — O(N)/iter.

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
    cfl: float = 0.0  # U_max·dt / L_buf  (should stay < ~0.7)
    conservation_drift: dict[str, float] = field(default_factory=dict)
    flux_ratio: float = 0.0  # |Γ_VPM_exit| / |Γ_FVM_exit| at the outflow band

    # Strength-correction diagnostics (η-weighted relative residual
    # ‖ω_target − ω_σ‖/‖ω_target‖ before and after the Picard iterations)
    strength_corr_residual_pre: float = 0.0
    strength_corr_residual_post: float = 0.0

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
    fvm_cell_vel=None,
    fvm_cell_vol=None,
    u_inf=None,
    inside_mesh_at_node=None,
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
        Conservative FVM→lattice source (``handoff_target_mode="vorticity"``):
        cell centres and their circulation ``Γ_cell = ω_cell · V_cell``,
        scattered onto the lattice with the same M4′ kernel as the wake.  The
        injected target circulation equals ``Σ ω_cell·V_cell`` exactly,
        independent of the FVM mesh refinement.
    fvm_cell_vel, fvm_cell_vol : ndarray (M, 3), (M,) or None
        Velocity-based source (``handoff_target_mode="velocity"``, FIX B):
        cell-centre velocities and cell volumes.  ``u·V`` and ``V`` are
        scattered separately, divided to the interpolated velocity, then
        ``ω = ∇×u`` is taken on the data-extent sub-grid (discrete-Stokes
        closed; see :func:`_velocity_based_target`).  Requires ``u_inf``.
    u_inf : ndarray (3,) or None
        Freestream velocity vector subtracted before the velocity-based scatter
        (``∇×U∞ ≡ 0``; see :func:`_velocity_based_target`).  Ignored by the
        vorticity path.
    omega_at_node : callable ``grid_pos -> ω (M, 3)`` or None
        Test/no-body path (``target = ω·h³``); ignored when ``fvm_cell_pos``
        is given.  When both are None the hand-off is pure transport.
    inside_mesh_at_node : callable ``grid_pos -> bool (M,)`` or None
        Mask of nodes that have FVM data (inside the mesh, outside the body).
        Nodes outside are forced η = 0 (pure Lagrangian).
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

    if n > 0:
        in_region = np.all((pos >= lo_active) & (pos <= hi_active), axis=1)
    else:
        in_region = np.zeros(0, dtype=bool)
    free_mask = ~in_region

    shape = (
        int(np.ceil((hi_lat[0] - lo_lat[0]) / h)) + 1,
        int(np.ceil((hi_lat[1] - lo_lat[1]) / h)) + 1,
        int(np.ceil((hi_lat[2] - lo_lat[2]) / h)) + 1,
    )

    # ── P2M: conservative M4′ scatter of the in-region wake onto the lattice ──
    grid_pos, grid_circ = remesh_to_grid(pos[in_region], circ[in_region], lo_lat, h, shape)

    # ── η partition of unity (FVM authority) ─────────────────────────────────
    if eta_fn is not None:
        eta = np.asarray(eta_fn(grid_pos), dtype=np.float64).reshape(-1)
    else:
        eta = cosine_eta(grid_pos, box, ramp_width, dead_zone)

    # ── Build the FVM target circulation on the lattice ──────────────────────
    target = None
    if fvm_cell_vel is not None and fvm_cell_vol is not None and len(fvm_cell_vel) > 0:
        # FIX B (Billuart §3.3): interpolate u_FVM, compute ω=∇×u on the
        # data-extent sub-grid (discrete-Stokes-closed over the data boundary,
        # not the full lattice whose zero-padded buffer forces ΣΓ ≡ 0).
        target = _velocity_based_target(
            fvm_cell_pos if fvm_cell_pos is not None else np.zeros((0, 3)),
            fvm_cell_vel,
            fvm_cell_vol,
            np.asarray(u_inf, dtype=np.float64).reshape(3) if u_inf is not None else np.zeros(3),
            lo_lat,
            h,
            shape,
        )
    elif fvm_cell_pos is not None and fvm_cell_circ is not None and len(fvm_cell_pos) > 0:
        _, target = remesh_to_grid(
            np.asarray(fvm_cell_pos, dtype=np.float64).reshape(-1, 3),
            np.asarray(fvm_cell_circ, dtype=np.float64).reshape(-1, 3),
            lo_lat,
            h,
            shape,
        )
    elif omega_at_node is not None:
        target = np.asarray(omega_at_node(grid_pos), dtype=np.float64).reshape(-1, 3) * (h**3)

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

    # ── Conservative prune: drop weak nodes, redistribute their moments ──────
    target_inv = _invariants(grid_pos, grid_blended)
    mag = np.linalg.norm(grid_blended, axis=1)
    keep = mag >= threshold_abs
    new_pos = grid_pos[keep]
    new_circ = grid_blended[keep]

    drift: dict[str, float] = {}
    if conserve and len(new_pos) > 0:
        new_vol_tmp = np.full(len(new_pos), h**3)
        try:
            # Conserve the 0th moment (circulation) and 1st moment (linear
            # impulse) — both exactly preserved by the M4′ scatter and restored
            # linearly after the prune.  The 2nd moment (angular impulse) is
            # deliberately NOT forced: it is not conserved by remeshing
            # (Cottet & Koumoutsakos §7), and enforcing it on a small
            # boundary-straddling cluster makes the Lagrange system
            # rank-deficient and injects spurious ±circulation.
            new_circ = recover_invariants(
                new_pos,
                new_circ,
                target_inv,
                volumes=new_vol_tmp,
                conserve_circulation=True,
                conserve_linear_impulse=True,
                conserve_angular_impulse=False,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("[Handoff] recover_invariants skipped: %s", exc)
        post = _invariants(new_pos, new_circ)
        ref = float(np.linalg.norm(target_inv["circulation"])) + 1e-30
        drift = {
            "circulation": float(np.linalg.norm(target_inv["circulation"] - post["circulation"])),
            "linear_impulse": float(
                np.linalg.norm(target_inv["linear_impulse"] - post["linear_impulse"])
            ),
            "angular_impulse": float(
                np.linalg.norm(target_inv["angular_impulse"] - post["angular_impulse"])
            ),
            "circulation_rel": float(
                np.linalg.norm(target_inv["circulation"] - post["circulation"]) / ref
            ),
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

    # Outflow-band vorticity-flux ratio (downstream-most h-layer of the box)
    flux_ratio = 0.0
    if target is not None and len(grid_pos) > 0:
        band = grid_pos[:, 0] >= hi[0] - h
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
        cfl=cfl,
        conservation_drift=drift,
        flux_ratio=flux_ratio,
        strength_corr_residual_pre=corr_pre,
        strength_corr_residual_post=corr_post,
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
        self.radius_ratio = float(getattr(cfg, "overlap_radius_ratio", RADIUS_RATIO))
        if self.radius_ratio < 1.0:
            logger.warning(
                "[Handoff] overlap_radius_ratio=%.2f < 1.0 breaks particle "
                "overlap — the Biot-Savart field will ripple between particles.",
                self.radius_ratio,
            )

        # dt-robust downstream buffer, sized from U_inf and dt (CFL criterion)
        self.u_inf = float(np.linalg.norm(cfg.u_inf))
        self.dt = float(cfg.dt)

        self.blend_relaxation = float(cfg.blend_relaxation)
        self.strength_corr_iters = int(cfg.strength_correction_iterations)
        self.strength_corr_relax = float(cfg.strength_correction_relax)
        if self.strength_corr_iters > 0 and cfg.particles_kernel != "GAUSSIAN":
            logger.warning(
                "[Handoff] strength_correction assumes the GAUSSIAN particle "
                "kernel (mollifier e^{-r²/σ²}); particles_kernel=%s — the "
                "deconvolution operator will not match the induced field.",
                cfg.particles_kernel,
            )

        self._box: np.ndarray | None = None
        self._cell_tree = None
        self._cell_centers: np.ndarray | None = None
        self._cell_volumes: np.ndarray | None = None
        self._velocity_buffer: np.ndarray | None = None
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

        # FIX B: when handoff_target_mode="velocity", compute the target from
        # ∇×u_FVM on the data-extent sub-grid (Billuart §3.3) instead of
        # scattering ω·V.  The freestream is subtracted before the scatter.
        handoff_mode = getattr(self.config, "handoff_target_mode", "vorticity")
        kw_target = {}
        if handoff_mode == "velocity":
            if self._velocity_buffer is None:
                self._velocity_buffer = np.ascontiguousarray(
                    fvm.get_velocity_field(), dtype=np.float64
                ).reshape(-1, 3)
            else:
                fvm.get_velocity_field_into(self._velocity_buffer)
            kw_target = {
                "fvm_cell_vel": self._velocity_buffer[in_bbox],
                "fvm_cell_vol": self._cell_volumes[in_bbox],
                "u_inf": self.config.U_inf,
            }
        else:
            kw_target = {"fvm_cell_circ": cell_circ[in_bbox]}

        def inside_mesh_at_node(grid_pos):
            d, _ = tree.query(grid_pos)
            return d < 1.5 * h

        res = continuous_handoff(
            pos,
            circ,
            self._box,
            h,
            fvm_cell_pos=cell_pos[in_bbox],
            inside_mesh_at_node=inside_mesh_at_node,
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
            **kw_target,
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
            "[Handoff step=%d] in=%d → out=%d  free=%d  CFL=%.2f  |ΔΓ|/|Γ|=%.2e  flux_ratio=%.3f",
            self.step,
            res.n_remesh_in,
            res.n_remesh_out,
            res.n_free,
            res.cfl,
            res.conservation_drift.get("circulation_rel", 0.0),
            res.flux_ratio,
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
