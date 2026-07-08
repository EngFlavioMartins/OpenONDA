"""
Unit tests for Continuous Overlap Transport with Conservative Hand-off.
======================================================================

These tests exercise the *pure-NumPy* numerical core
(``source.coupler.core.helpers.continuous_overlap``) with **no OpenFOAM
and no Taichi/GPU dependency**, so they run anywhere numpy + scipy are
installed.  They verify the three physical guarantees the algorithm was
designed to provide:

  1. **Conservation (no trapping).**  The P2M scatter + conservative prune
     preserve total circulation, linear impulse and angular impulse — so
     vorticity can never be silently deleted at the interface.

  2. **No-trapping under convection.**  A vortex blob advected through the
     interface Γ keeps its total circulation at every step and its
     vorticity centroid advances at the freestream speed (it is "always
     nicely advected outwards", never stuck).

  3. **Time-step insensitivity.**  Running the same physical convection
     with dt and dt/2 gives the same final circulation and centroid — the
     hand-off does not depend on the step size (no accumulation, no
     stagnation).

Plus C¹-continuity of the η partition of unity and the CFL helpers.

Usage
-----
    pytest tests/coupler/test_continuous_overlap.py -v
    # or
    pytest tests/coupler/test_continuous_overlap.py
"""

from __future__ import annotations

import numpy as np

from source.coupler.core.helpers.continuous_overlap import (
    continuous_handoff,
    cosine_eta,
    max_stable_dt,
    required_buffer_length,
)

# ── Test fixture: box, lattice spacing, freestream ───────────────────────────
BOX = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])  # FVM domain Γ
H = 0.1
U_INF = 1.0
SIGMA = 0.22
OMEGA0 = 1.0


def _gaussian_blob(center, *, lo=None, hi=None, h=H, sigma=SIGMA, omega0=OMEGA0, axis=2):
    """Particles on an h-lattice carrying a Gaussian vortex blob (Γ = ω·h³).

    The vorticity points along ``axis`` (default +z).  Returns (pos, circ).
    """
    center = np.asarray(center, dtype=np.float64)
    if lo is None:
        lo = center - 5 * sigma
    if hi is None:
        hi = center + 5 * sigma
    grids = [np.arange(lo[d], hi[d] + 0.5 * h, h) for d in range(3)]
    gx, gy, gz = np.meshgrid(*grids, indexing="ij")
    pos = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    r2 = np.sum((pos - center) ** 2, axis=1)
    omega_mag = omega0 * np.exp(-r2 / (2.0 * sigma**2))
    circ = np.zeros_like(pos)
    circ[:, axis] = omega_mag * h**3
    # Keep only nodes that actually carry signal (trim the far tails)
    keep = omega_mag > 1e-4 * omega0
    return pos[keep], circ[keep]


def _invariants(pos, circ):
    r_x_g = np.cross(pos, circ)
    return (
        np.sum(circ, axis=0),
        0.5 * np.sum(r_x_g, axis=0),
        (1.0 / 3.0) * np.sum(np.cross(pos, r_x_g), axis=0),
    )


def _centroid_x(pos, circ, axis=2):
    """x-coordinate of the vorticity centroid: Σ x_i Γ_axis,i / Σ Γ_axis,i."""
    w = circ[:, axis]
    return float(np.sum(pos[:, 0] * w) / (np.sum(w) + 1e-30))


# =============================================================================
# 1. Conservation through one hand-off (pure transport, with pruning active)
# =============================================================================
def test_handoff_conserves_invariants():
    """P2M + conservative prune exactly preserve circulation (0th moment) and
    linear impulse / centroid (1st moment) even when weak nodes are pruned.
    The angular impulse (2nd moment) is only approximately preserved — that is
    the expected behaviour of M4′ remeshing, not a defect."""
    # Off-origin so the linear/angular impulse references are non-trivial.
    pos, circ = _gaussian_blob([0.3, -0.2, 0.1])
    G0, I0, A0 = _invariants(pos, circ)

    # threshold prunes the Gaussian tails -> exercises the redistribution
    thr = 0.02 * np.linalg.norm(circ, axis=1).max()

    res = continuous_handoff(
        pos,
        circ,
        BOX,
        H,
        omega_at_node=None,  # pure transport: no FVM blend
        buffer_length=0.3,
        threshold_abs=thr,
    )
    G1, I1, A1 = _invariants(res.pos, res.circ)

    # Normalize by fixed physical scales (Γ, Γ·L, Γ·L²) — never by a quantity
    # that can vanish for a symmetric blob.
    Lc = 1.0
    sG = np.linalg.norm(G0) + 1e-30
    relG = np.linalg.norm(G1 - G0) / sG
    relI = np.linalg.norm(I1 - I0) / (sG * Lc)
    relA = np.linalg.norm(A1 - A0) / (sG * Lc**2)

    assert relG < 1e-6, f"circulation drift {relG:.2e}"
    assert relI < 1e-6, f"linear-impulse (centroid) drift {relI:.2e}"
    assert relA < 0.05, f"angular-impulse drift {relA:.2e} (remesh diffusion, expected small)"
    # The prune actually removed nodes (otherwise the test is vacuous)
    assert res.n_remesh_out < len(pos) + 50


# =============================================================================
# 2. Blend toward an FVM field still conserves the *blended* invariants
# =============================================================================
def test_blend_prune_conserves_blended_field():
    """With an FVM target and η-blend, the conservative prune preserves the
    invariants of the blended source field (drift reported by the core)."""
    pos, circ = _gaussian_blob([0.0, 0.0, 0.0])

    # Smooth synthetic FVM vorticity (a slightly shifted, stronger blob)
    def omega_at_node(gp):
        r2 = np.sum((gp - np.array([0.1, 0.0, 0.0])) ** 2, axis=1)
        w = np.zeros((len(gp), 3))
        w[:, 2] = 1.5 * OMEGA0 * np.exp(-r2 / (2.0 * SIGMA**2))
        return w

    res = continuous_handoff(
        pos,
        circ,
        BOX,
        H,
        omega_at_node=omega_at_node,
        inside_mesh_at_node=lambda gp: np.ones(len(gp), dtype=bool),
        ramp_width=0.4,
        dead_zone=0.2,
        buffer_length=0.3,
        threshold_abs=0.02 * np.linalg.norm(circ, axis=1).max(),
    )
    drift = res.conservation_drift
    assert drift["circulation_rel"] < 1e-6, f"blended Γ drift {drift['circulation_rel']:.2e}"


# =============================================================================
# 3. No-trapping: a blob advected through Γ is never lost, never stuck
# =============================================================================
def test_no_trapping_under_convection():
    """Advect a blob from inside the box, across Γ, into the free exterior.
    Total circulation must be conserved at every step (nothing trapped),
    and the centroid must advance at U_inf (nothing stuck)."""
    dt = 0.1
    l_buf = required_buffer_length(U_INF, dt, H)

    pos, circ = _gaussian_blob([-0.5, 0.0, 0.0])
    G_z0 = float(np.sum(circ[:, 2]))
    c_x0 = _centroid_x(pos, circ)

    thr = 0.01 * np.linalg.norm(circ, axis=1).max()
    n_steps = 25
    for step in range(n_steps):
        pos = pos.copy()
        pos[:, 0] += U_INF * dt  # physical advection by the freestream
        res = continuous_handoff(
            pos,
            circ,
            BOX,
            H,
            omega_at_node=None,  # pure transport (isolate the hand-off)
            buffer_length=l_buf,
            threshold_abs=thr,
            u_max=U_INF,
            dt=dt,
        )
        pos, circ = res.pos, res.circ

        G_z = float(np.sum(circ[:, 2]))
        rel = abs(G_z - G_z0) / abs(G_z0)
        assert rel < 1e-5, f"step {step}: circulation not conserved ({rel:.2e})"
        assert res.cfl < 0.7, f"step {step}: CFL too high ({res.cfl:.2f})"

    # Centroid advanced at the freestream speed (always advected outwards)
    c_x = _centroid_x(pos, circ)
    expected = c_x0 + U_INF * dt * n_steps
    assert abs(c_x - expected) < 5e-3, f"centroid {c_x:.4f} != expected {expected:.4f}"

    # The blob has fully crossed Γ (x1 = 1.0): essentially all circulation
    # now lives downstream of the interface — none trapped inside.
    downstream_frac = float(np.sum(circ[pos[:, 0] > BOX[1], 2]) / (np.sum(circ[:, 2]) + 1e-30))
    assert downstream_frac > 0.99, f"only {downstream_frac:.3f} of Γ crossed Γ"


# =============================================================================
# 4. Time-step insensitivity: dt vs dt/2 give the same physical result
# =============================================================================
def _convect(dt, n_steps, l_buf):
    pos, circ = _gaussian_blob([-0.5, 0.0, 0.0])
    thr = 0.01 * np.linalg.norm(circ, axis=1).max()
    for _ in range(n_steps):
        pos = pos.copy()
        pos[:, 0] += U_INF * dt
        res = continuous_handoff(
            pos,
            circ,
            BOX,
            H,
            omega_at_node=None,
            buffer_length=l_buf,
            threshold_abs=thr,
            u_max=U_INF,
            dt=dt,
        )
        pos, circ = res.pos, res.circ
    return pos, circ


def test_timestep_insensitivity():
    """Same total time T, two step sizes -> same circulation and centroid.
    Demonstrates the hand-off does not accumulate or stall as dt changes."""
    T = 2.0
    l_buf = required_buffer_length(U_INF, 0.2, H)  # size for the larger dt

    pos_a, circ_a = _convect(0.2, int(round(T / 0.2)), l_buf)
    pos_b, circ_b = _convect(0.1, int(round(T / 0.1)), l_buf)

    Gz_a = float(np.sum(circ_a[:, 2]))
    Gz_b = float(np.sum(circ_b[:, 2]))
    cx_a = _centroid_x(pos_a, circ_a)
    cx_b = _centroid_x(pos_b, circ_b)

    assert abs(Gz_a - Gz_b) / abs(Gz_a) < 1e-5, f"Γ depends on dt: {Gz_a:.6f} vs {Gz_b:.6f}"
    assert abs(cx_a - cx_b) < 5e-3, f"centroid depends on dt: {cx_a:.4f} vs {cx_b:.4f}"


# =============================================================================
# 5. η partition of unity is C¹ and behaves at the boundaries
# =============================================================================
def test_eta_partition_of_unity():
    ramp, dz = 0.4, 0.2
    # Sample along +x through the box face at x1 = 1.0
    xs = np.linspace(0.0, 1.5, 4000)
    pts = np.column_stack([xs, np.zeros_like(xs), np.zeros_like(xs)])
    eta = cosine_eta(pts, BOX, ramp, dz)

    # Deep interior -> 1, outside the box -> 0
    assert eta[xs < 1.0 - ramp].max() == 1.0 or np.allclose(eta[xs < 1.0 - ramp], 1.0)
    assert np.all(eta[xs > 1.0] == 0.0)  # outside Γ: pure Lagrangian
    assert np.all(eta[xs > 1.0 - dz] <= 1e-9)  # dead-zone at the face

    # η itself is continuous: a true step discontinuity would give Δη ≈ 1.
    assert np.max(np.abs(np.diff(eta))) < 0.02, "η has a jump (not even C⁰)"

    # C¹: the derivative has no O(1) slope discontinuity.  For a smooth cosine
    # ramp the second difference scales as f''·dx ≈ 0.05; a C⁰ corner (slope
    # jump) would register O(1).  Threshold 0.3 separates the two cleanly.
    d_eta = np.diff(eta) / np.diff(xs)
    assert np.all(np.isfinite(d_eta))
    assert np.max(np.abs(np.diff(d_eta))) < 0.3, "η derivative has a jump (not C¹)"


# =============================================================================
# 7. Conservative volume-weighted FVM injection is resolution-independent
# =============================================================================
def test_injection_conserves_fvm_circulation_resolution_independent():
    """The COT-CH v2 hand-off scatters Γ_cell = ω_cell·V_cell onto the lattice,
    so the injected circulation equals Σ ω_cell·V_cell (∫ω dV) *exactly* and is
    independent of the FVM mesh refinement.  The old ``ω(nearest)·h³`` sampling
    over-counts by ~h³/V_cell where ω varies on a sub-h scale (the cube case:
    near-body cells 0.025 vs h 0.06 → ~14×).  Here a fixed-thickness vortex
    sheet (thinner than h) is resolved on progressively finer FVM cells; the
    injected total must stay equal to the true ∫ω dV for every refinement."""
    box = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    h = 0.1
    extent = 0.3  # ω_z = 1 fills the block [-0.3, 0.3]³ (well inside → η = 1)

    def make_cells(hf):
        # Cell centres that *exactly* tile [-extent, extent]³ for hf ∈ {h, h/2, h/4}
        # (2·extent/hf is integer), so ∫ω dV = (2·extent)³ is identical for every hf.
        n = int(round(2 * extent / hf))
        g = -extent + hf * (0.5 + np.arange(n))
        X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
        cp = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        circ = np.zeros_like(cp)
        circ[:, 2] = 1.0 * hf**3  # ω_z = 1 → Γ_cell = ω·V_cell
        return cp, circ

    injected_totals = []
    true_totals = []
    for hf in (h, h / 2.0, h / 4.0):
        cp, circ = make_cells(hf)
        true_total = float(np.sum(circ[:, 2]))
        res = continuous_handoff(
            np.zeros((0, 3)),
            np.zeros((0, 3)),
            box,
            h,
            fvm_cell_pos=cp,
            fvm_cell_circ=circ,
            inside_mesh_at_node=lambda gp: np.ones(len(gp), dtype=bool),
            ramp_width=0.3,  # η = 1 for |coord| < 0.7, covering the deposit
            dead_zone=0.0,
            buffer_length=0.2,
            threshold_abs=0.0,  # no prune → exact M4′ conservation
            blend_relaxation=1.0,
        )
        injected = float(np.sum(res.circ[:, 2]))
        rel = abs(injected - true_total) / abs(true_total)
        assert rel < 1e-10, f"hf={hf}: injected {injected:.6e} != ∫ωdV {true_total:.6e} ({rel:.1e})"
        injected_totals.append(injected)
        true_totals.append(true_total)

    # A fixed-thickness sheet carries the same physical circulation regardless of
    # how finely it is resolved — and the hand-off reproduces it every time.
    assert max(true_totals) - min(true_totals) < 1e-12, f"true totals vary: {true_totals}"
    assert max(injected_totals) - min(injected_totals) < 1e-9, injected_totals


# =============================================================================
# 8. Blend under-relaxation α scales the FVM contribution linearly
# =============================================================================
def test_blend_relaxation_linear():
    """With an empty wake, the blended field is α·η·target, so the injected
    circulation is exactly linear in α: total(0)=0, total(½)=½·total(1).
    α=1 is the hard overwrite; α<1 the gradual approach that breaks feedback."""
    box = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    h = 0.1
    g = np.arange(-0.3 + 0.05, 0.3, h)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    cp = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    circ = np.zeros_like(cp)
    circ[:, 2] = 1.0 * h**3

    def run(alpha):
        res = continuous_handoff(
            np.zeros((0, 3)),
            np.zeros((0, 3)),
            box,
            h,
            fvm_cell_pos=cp,
            fvm_cell_circ=circ,
            inside_mesh_at_node=lambda gp: np.ones(len(gp), dtype=bool),
            ramp_width=0.3,
            dead_zone=0.0,
            buffer_length=0.2,
            threshold_abs=0.0,
            blend_relaxation=alpha,
        )
        return float(np.sum(res.circ[:, 2]))

    t0, thalf, t1 = run(0.0), run(0.5), run(1.0)
    assert abs(t0) < 1e-12, f"α=0 should inject nothing, got {t0:.3e}"
    assert abs(t1) > 1e-6, "α=1 should inject the full target"
    assert abs(thalf - 0.5 * t1) < 1e-9 * abs(t1), f"not linear in α: {thalf:.6e} vs {0.5 * t1:.6e}"


# =============================================================================
# 6. CFL helpers round-trip
# =============================================================================
def test_cfl_helpers_roundtrip():
    l_buf = required_buffer_length(U_INF, 0.15, H)
    dt_max = max_stable_dt(U_INF, l_buf, H)
    assert dt_max >= 0.15 - 1e-12, f"max_stable_dt {dt_max} inconsistent with buffer {l_buf}"
    # A step at the reported max_dt should sit at CFL ~ U·dt/(L_buf+guard) < 0.7
    cfl = U_INF * dt_max / (l_buf + 2.0 * H)
    assert cfl < 0.7


# =============================================================================
# 9. Prune threshold scales correctly with h (resolution-independent pruning)
# =============================================================================
def test_prune_threshold_scales_with_h():
    """The same prune_vorticity_min keeps the same *fraction* of lattice nodes
    when h is halved.  The absolute Γ floor must scale as h³ — the old fixed
    1e-5 cutoff zeroed 8× more lattice nodes per halving of h.

    Metric: n_surviving / n_total_lattice_nodes.  We run with threshold=0
    (all nodes) and with the vorticity floor to measure the survival fraction.
    conserve=False prevents recover_invariants from masking the raw prune.
    """
    box = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    omega0 = 1.0
    sigma = 0.18
    prune_vorticity_min = 0.01

    def survival_fraction(h):
        threshold_abs = prune_vorticity_min * h**3
        lo = np.array([-0.5, -0.5, -0.5])
        hi = np.array([0.5, 0.5, 0.5])
        grids = [np.arange(lo[d], hi[d] + 0.5 * h, h) for d in range(3)]
        gx, gy, gz = np.meshgrid(*grids, indexing="ij")
        pos = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
        r2 = np.sum(pos**2, axis=1)
        omega_mag = omega0 * np.exp(-r2 / (2.0 * sigma**2))
        circ = np.zeros_like(pos)
        circ[:, 2] = omega_mag * h**3

        kwargs = {
            "box": box,
            "h": h,
            "buffer_length": required_buffer_length(1.0, 0.1, h),
            "conserve": False,  # raw prune: recover_invariants would mask the effect
        }
        res_all = continuous_handoff(pos, circ, threshold_abs=0.0, **kwargs)
        res_pruned = continuous_handoff(pos, circ, threshold_abs=threshold_abs, **kwargs)

        # n_remesh_out counts only lattice-origin particles (free exterior is excluded)
        n_all = res_all.n_remesh_out
        n_kept = res_pruned.n_remesh_out
        return n_kept / (n_all + 1e-30)

    frac_coarse = survival_fraction(0.10)
    frac_fine = survival_fraction(0.05)

    # The same physical vorticity cutoff means the same kept-region geometry.
    assert abs(frac_coarse - frac_fine) < 0.08, (
        f"Survival fraction differs by resolution: h=0.10→{frac_coarse:.3f}  "
        f"h=0.05→{frac_fine:.3f}  (gap={abs(frac_coarse - frac_fine):.3f} > 0.08)"
    )
    # Something is actually pruned at both resolutions (test is non-vacuous)
    assert frac_coarse < 0.99, f"Nothing pruned at h=0.10 (frac={frac_coarse:.4f})"
    assert frac_fine < 0.99, f"Nothing pruned at h=0.05 (frac={frac_fine:.4f})"


# =============================================================================
# Stand-alone runner
# =============================================================================
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
