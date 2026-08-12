"""
Conservation test for the GBD/DVH conservative prune.

The grid-diffusion regen keeps only nodes above a magnitude threshold and used
to **drop** the rest, silently deleting their circulation (a non-physical decay,
worst in 'absolute' mode where it clips the far wake every regen).  The
conservative prune redistributes the dropped nodes' moments onto the survivors
with a weighted minimum-norm correction so the regeneration preserves:

  * the 0th moment  Σ Γ            (total circulation, Kelvin)        — exactly,
  * the 1st moment  Σ x × Γ        (linear impulse)                   — exactly,
  * angular impulse Σ x×(x×Γ)/3 (transverse second moment)        — exactly.

These are pure-NumPy tests of ``_redistribute_pruned_moments`` (no Taichi/GPU).
"""

from __future__ import annotations

import numpy as np
import pytest

from source.solvers.VPM.physics.diffusion import _GridDiffusionMixin

redistribute = _GridDiffusionMixin._redistribute_pruned_moments
cap_survivors = _GridDiffusionMixin._cap_surviving_nodes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_grid(seed: int = 0):
    """A 3-D circulation grid with a strong core and a weak (prune-able) halo."""
    rng = np.random.default_rng(seed)
    nx = ny = nz = 24
    h = 0.05
    grid_min = np.array([-0.3, -0.4, -0.2])
    grid = np.zeros((nx, ny, nz, 3), dtype=np.float32)

    # Strong, asymmetric core (survives) — placed off-centre so the 1st moment
    # is non-trivial and the survivor set is genuinely 3-D (well-conditioned M).
    for cx, cy, cz, amp in [(8, 10, 12, 1.0), (15, 13, 11, 0.7), (11, 16, 14, 0.5)]:
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    r2 = (i - cx) ** 2 + (j - cy) ** 2 + (k - cz) ** 2
                    w = np.exp(-r2 / 6.0)
                    grid[i, j, k] += (amp * w * np.array([0.3, -0.2, 1.0])).astype(np.float32)

    # A faint, broad halo (the part that gets pruned) — carries real circulation.
    halo = 0.01 * rng.standard_normal((nx, ny, nz, 3)).astype(np.float32)
    grid += halo
    return grid, grid_min, float(h)


def _moments(grid, grid_min, h, mask=None):
    nx, ny, nz = grid.shape[:3]
    if mask is None:
        ii, jj, kk = np.where(np.linalg.norm(grid, axis=-1) > 0)
    else:
        ii, jj, kk = np.where(mask)
    G = grid[ii, jj, kk].astype(np.float64)
    X = np.stack([grid_min[0] + ii * h, grid_min[1] + jj * h, grid_min[2] + kk * h], axis=1)
    return (
        G.sum(axis=0),
        np.cross(X, G).sum(axis=0),
        np.cross(X, np.cross(X, G)).sum(axis=0) / 3.0,
        (ii, jj, kk, X, G),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_redistribution_conserves_circulation_and_impulse():
    grid, grid_min, h = _make_grid()
    circ_mag = np.linalg.norm(grid, axis=-1)

    G_full, L_full, A_full, _ = _moments(grid, grid_min, h)

    # Prune the faint halo (keep ~strong core); ensure a real cut happens.
    thr = 0.05 * float(circ_mag.max())
    ix, iy, iz = np.where(circ_mag >= thr)
    n_total = int(np.count_nonzero(circ_mag > 0))
    assert 4 < len(ix) < n_total, "test needs a non-trivial prune with enough survivors"

    Xk = np.stack([grid_min[0] + ix * h, grid_min[1] + iy * h, grid_min[2] + iz * h], axis=1)

    # Raw (non-conservative) survivors LOSE circulation/impulse.
    G_raw = grid[ix, iy, iz].astype(np.float64).sum(axis=0)
    assert np.linalg.norm(G_full - G_raw) > 1e-3, "prune should drop measurable circulation"

    # Conservative redistribution restores both moments.
    corrected = redistribute(grid, circ_mag, ix, iy, iz, grid_min, h).astype(np.float64)
    G_post = corrected.sum(axis=0)
    L_post = np.cross(Xk, corrected).sum(axis=0)
    A_post = np.cross(Xk, np.cross(Xk, corrected)).sum(axis=0) / 3.0

    ref = np.linalg.norm(G_full) + 1e-30
    assert np.linalg.norm(G_full - G_post) / ref < 1e-5, (
        f"0th moment not conserved: {G_full} vs {G_post}"
    )
    refL = np.linalg.norm(L_full) + 1e-30
    assert np.linalg.norm(L_full - L_post) / refL < 1e-5, (
        f"1st moment not conserved: {L_full} vs {L_post}"
    )
    refA = np.linalg.norm(A_full) + 1e-30
    assert np.linalg.norm(A_full - A_post) / refA < 1e-5, (
        f"angular impulse not conserved: {A_full} vs {A_post}"
    )


def test_no_prune_is_exact_noop():
    """When the threshold keeps every non-empty node, correction = identity."""
    grid, grid_min, h = _make_grid(seed=1)
    circ_mag = np.linalg.norm(grid, axis=-1)
    ix, iy, iz = np.where(circ_mag > 0.0)  # keep everything
    raw = grid[ix, iy, iz].astype(np.float64)
    corrected = redistribute(grid, circ_mag, ix, iy, iz, grid_min, h).astype(np.float64)
    assert np.allclose(corrected, raw, atol=1e-7), "no-prune case must be a no-op"


def test_degenerate_survivors_fall_back_gracefully():
    """Too few survivors → return raw survivors (no crash, 0th-moment only path)."""
    grid, grid_min, h = _make_grid(seed=2)
    circ_mag = np.linalg.norm(grid, axis=-1)
    # Keep only the 2 strongest nodes (len < 4 → graceful fallback).
    flat = circ_mag.ravel()
    keep_lin = np.argsort(-flat)[:2]
    ix, iy, iz = np.unravel_index(keep_lin, circ_mag.shape)
    corrected = redistribute(grid, circ_mag, ix, iy, iz, grid_min, h)
    assert corrected.shape == (2, 3)
    assert np.allclose(corrected, grid[ix, iy, iz], atol=1e-7)


def test_survivor_cap_is_exact_even_with_tied_cutoff_values():
    """The count cap must not be exceeded when many nodes share |Γ|."""
    circ_mag = np.ones((4, 4, 4), dtype=np.float32)
    ix, iy, iz = np.where(circ_mag >= 1.0)

    ix_keep, iy_keep, iz_keep, threshold, old_count = cap_survivors(circ_mag, ix, iy, iz, cap=10)

    assert old_count == 64
    assert len(ix_keep) == len(iy_keep) == len(iz_keep) == 10
    assert threshold == pytest.approx(1.0)


def test_circulation_conserved_across_threshold_sweep():
    """0th moment conserved for a range of prune aggressiveness."""
    grid, grid_min, h = _make_grid(seed=3)
    circ_mag = np.linalg.norm(grid, axis=-1)
    G_full, _, _, _ = _moments(grid, grid_min, h)
    ref = np.linalg.norm(G_full) + 1e-30
    for frac in (0.02, 0.1, 0.3, 0.5):
        thr = frac * float(circ_mag.max())
        ix, iy, iz = np.where(circ_mag >= thr)
        if len(ix) < 4:
            continue
        corrected = redistribute(grid, circ_mag, ix, iy, iz, grid_min, h).astype(np.float64)
        err = np.linalg.norm(G_full - corrected.sum(axis=0)) / ref
        assert err < 1e-5, f"frac={frac}: 0th moment error {err:.2e}"


def test_redistribution_conserves_each_vortex_group():
    """Opposite-signed vortices retain their individual diffusive moments."""
    shape = (28, 20, 12)
    h = 0.05
    grid_min = np.array([-0.7, -0.5, -0.3])
    ii, jj, kk = np.indices(shape)
    left = np.exp(-((ii - 8) ** 2 + (jj - 10) ** 2 + (kk - 6) ** 2) / 18.0)
    right = np.exp(-((ii - 20) ** 2 + (jj - 10) ** 2 + (kk - 6) ** 2) / 18.0)
    labels = np.where(ii < 14, 0, 1).astype(np.int32)
    grid = np.zeros((*shape, 3), dtype=np.float32)
    grid[..., 2] = np.where(labels == 0, left, -right)
    circ_mag = np.linalg.norm(grid, axis=-1)

    threshold = 0.08 * float(circ_mag.max())
    ix, iy, iz = np.where(circ_mag >= threshold)
    corrected = redistribute(
        grid,
        circ_mag,
        ix,
        iy,
        iz,
        grid_min,
        h,
        labels=labels,
    ).astype(np.float64)
    positions = np.stack(
        [grid_min[0] + ix * h, grid_min[1] + iy * h, grid_min[2] + iz * h],
        axis=1,
    )
    survivor_labels = labels[ix, iy, iz]

    for label in (0, 1):
        full_mask = labels == label
        circulation, linear, angular, _ = _moments(grid, grid_min, h, mask=full_mask)
        selected = survivor_labels == label
        group_circulation = corrected[selected].sum(axis=0)
        group_linear = np.cross(positions[selected], corrected[selected]).sum(axis=0)
        group_angular = (
            np.cross(positions[selected], np.cross(positions[selected], corrected[selected])).sum(
                axis=0
            )
            / 3.0
        )
        assert np.allclose(group_circulation, circulation, rtol=1e-5, atol=1e-7)
        assert np.allclose(group_linear, linear, rtol=1e-5, atol=1e-7)
        assert np.allclose(group_angular, angular, rtol=1e-5, atol=1e-7)
