"""
Tests for the Beale/Picard iterated strength assignment
(``beale_strength_correction`` in continuous_overlap.py).

The correction is a regularized deconvolution of the Gaussian-core
mollification: after M iterations the residual at wavenumber k is
(1 − e^{−k²σ²/4})^{M+1}, so the mollified particle vorticity converges to
the FVM target on resolved scales while sub-kernel scales stay untouched.

Test field: a solenoidal swirl ω = (−y, x, 0)·g(r), g = A·e^{−r²/R²},
which is divergence-free by construction and localized (so the lattice
boundary plays no role).  With R = 4h and σ = 1.5h, the analytic peak
attenuation of direct assignment is (R²/(R²+σ²))^{3/2} ≈ 0.82 — an ~18 %
bias the iteration must remove.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from source.coupler.core.helpers.continuous_overlap import (
    RADIUS_RATIO,
    beale_strength_correction,
    continuous_handoff,
)

H = 0.05
SIGMA = RADIUS_RATIO * H  # 1.5h, the hand-off lattice core radius
N = 40  # lattice nodes per axis
SHAPE = (N, N, N)


def _lattice():
    """Centered cubic lattice (N³ nodes, spacing H)."""
    ax = (np.arange(N) - (N - 1) / 2.0) * H
    gx, gy, gz = np.meshgrid(ax, ax, ax, indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])


def _swirl_omega(pos: np.ndarray, R: float = 4.0 * H) -> np.ndarray:
    """Divergence-free swirl ω = (−y, x, 0)·e^{−r²/R²} (analytically ∇·ω = 0)."""
    r2 = np.sum(pos**2, axis=1)
    g = np.exp(-r2 / R**2)
    return np.column_stack([-pos[:, 1] * g, pos[:, 0] * g, np.zeros(len(pos))])


def _mollified(circ_flat: np.ndarray) -> np.ndarray:
    """Gaussian-core mollified vorticity at the nodes (same operator as the
    correction): ω_σ = Σ Γ_p ζ_σ(x − x_p) with ζ_σ ∝ e^{−r²/σ²}."""
    s_cells = SIGMA / (np.sqrt(2.0) * H)
    g = circ_flat.reshape(*SHAPE, 3)
    out = (
        np.stack(
            [gaussian_filter(g[..., c], s_cells, mode="constant", truncate=5.0) for c in range(3)],
            axis=-1,
        )
        / H**3
    )
    return out.reshape(-1, 3)


@pytest.fixture
def field():
    pos = _lattice()
    omega = _swirl_omega(pos)
    target = omega * H**3  # direct-assignment circulations
    eta = np.ones(len(pos))
    return pos, omega, target, eta


def _residual(circ_flat, omega):
    """Relative L² mismatch between the mollified field and the target."""
    return np.linalg.norm(_mollified(circ_flat) - omega) / np.linalg.norm(omega)


# ---------------------------------------------------------------------------
def test_direct_assignment_has_significant_mollification_bias(field):
    """Sanity: the bias the correction targets actually exists (~15–25 %)."""
    _, omega, target, _ = field
    res0 = _residual(target, omega)
    assert 0.10 < res0 < 0.35, f"expected O(20%) bias for R=4h, got {res0:.3f}"


def test_residual_decreases_monotonically(field):
    """Each Picard iteration must reduce the η-weighted residual."""
    _, omega, target, eta = field
    residuals = []
    for m in [0, 1, 2, 3, 5]:
        corrected, _, res_post = beale_strength_correction(
            target.copy(), target, eta, SHAPE, H, sigma=SIGMA, iterations=m
        )
        if m == 0:
            # iterations=0 must return the input unchanged
            np.testing.assert_allclose(corrected, target, rtol=0, atol=0)
        residuals.append(_residual(corrected, omega))
    assert all(b < a for a, b in zip(residuals, residuals[1:], strict=False)), residuals


def test_three_iterations_remove_most_of_the_bias(field):
    """M=3 must cut the resolved-scale bias by at least 5× (theory: ~(0.2)⁴)."""
    _, omega, target, eta = field
    corrected, res_pre, res_post = beale_strength_correction(
        target.copy(), target, eta, SHAPE, H, sigma=SIGMA, iterations=3
    )
    assert res_post < 0.2 * res_pre, f"pre={res_pre:.4f} post={res_post:.4f}"
    # and the independently-computed mollified field agrees
    assert _residual(corrected, omega) < 0.2 * _residual(target, omega)


def test_total_circulation_preserved(field):
    """The Gaussian kernel sums to 1, so deconvolution must not change the
    total circulation of a localized field (edge losses are negligible)."""
    _, _, target, eta = field
    corrected, _, _ = beale_strength_correction(
        target.copy(), target, eta, SHAPE, H, sigma=SIGMA, iterations=3
    )
    scale = np.abs(target).sum()
    drift = np.linalg.norm(corrected.sum(axis=0) - target.sum(axis=0))
    assert drift < 1e-9 * scale


def test_eta_zero_region_untouched(field):
    """Where η = 0 (free Lagrangian wake) the strengths must not change."""
    pos, _, target, _ = field
    eta = (pos[:, 0] > 0).astype(float)  # correct only the +x half
    corrected, _, _ = beale_strength_correction(
        target.copy(), target, eta, SHAPE, H, sigma=SIGMA, iterations=3
    )
    frozen = eta == 0.0
    np.testing.assert_allclose(corrected[frozen], target[frozen], rtol=0, atol=0)


def test_handoff_wires_the_correction():
    """End-to-end through continuous_handoff via the legacy ω-sample path:
    iterations=3 must report a reduced residual; iterations=0 reports none."""
    box = [-0.8, 0.8, -0.8, 0.8, -0.8, 0.8]

    def omega_at_node(grid_pos):
        return _swirl_omega(np.asarray(grid_pos))

    common = {
        "pos": np.zeros((0, 3)),
        "circ": np.zeros((0, 3)),
        "box": box,
        "h": H,
        "omega_at_node": omega_at_node,
        "ramp_width": 4 * H,
        "dead_zone": 0.0,
        "buffer_length": 2 * H,
    }
    res_off = continuous_handoff(**common, strength_correction_iterations=0)
    assert res_off.strength_corr_residual_pre == 0.0
    assert res_off.strength_corr_residual_post == 0.0

    res_on = continuous_handoff(**common, strength_correction_iterations=3)
    assert res_on.strength_corr_residual_pre > 0.0
    assert res_on.strength_corr_residual_post < 0.5 * res_on.strength_corr_residual_pre


# ---------------------------------------------------------------------------
# Body-hole regression (cubeFlow 2026-06-10): the FVM target ends in a step at
# the body wall (no cells inside).  Deconvolving across that discontinuity
# rings, injecting circulation INSIDE the body — particles then advect through
# the body footprint.  The hand-off must erode the FVM-data mask by the kernel
# support before weighting the correction.
# ---------------------------------------------------------------------------
BODY_HALF = 0.25  # cube body half-side [m] (5 cells)


def _body_mask(pos: np.ndarray) -> np.ndarray:
    return np.max(np.abs(pos), axis=1) < BODY_HALF


def _holed_omega(pos: np.ndarray) -> np.ndarray:
    """Swirl field with a hard zero inside the body (the wall step)."""
    omega = _swirl_omega(pos, R=8.0 * H)
    omega[_body_mask(pos)] = 0.0
    return omega


def test_unguarded_correction_rings_into_the_body():
    """Documents the bug: with the raw η weight, the iteration injects
    spurious circulation into the near-wall body shell."""
    pos = _lattice()
    target = _holed_omega(pos) * H**3
    eta = np.ones(len(pos))  # NO body guard — the buggy configuration
    corrected, _, _ = beale_strength_correction(
        target.copy(), target, eta, SHAPE, H, sigma=SIGMA, iterations=3
    )
    in_body = _body_mask(pos)
    injected = np.abs(corrected[in_body]).sum()
    assert injected > 1e-3 * np.abs(target).sum(), "expected ringing into the body"


def test_handoff_guard_keeps_the_body_clean():
    """End-to-end: with inside_mesh_at_node defining the body hole, the
    eroded-mask guard must leave the body interior at zero strength while
    still correcting the resolved exterior field."""
    box = [-0.9, 0.9, -0.9, 0.9, -0.9, 0.9]

    def inside_mesh_at_node(grid_pos):
        # FVM data exists outside the body, plus a 1.5h tolerance shell
        # inside it (mimics the cKDTree nearest-cell check in the injector).
        return np.max(np.abs(np.asarray(grid_pos)), axis=1) > BODY_HALF - 1.5 * H

    common = {
        "pos": np.zeros((0, 3)),
        "circ": np.zeros((0, 3)),
        "box": box,
        "h": H,
        "omega_at_node": _holed_omega,
        "inside_mesh_at_node": inside_mesh_at_node,
        "ramp_width": 4 * H,
        "dead_zone": 0.0,
        "buffer_length": 2 * H,
        "threshold_abs": 0.0,
        "conserve": False,  # keep node strengths raw for the assertion
    }
    res = continuous_handoff(**common, strength_correction_iterations=3)
    in_body = _body_mask(res.pos)
    body_gamma = np.linalg.norm(res.circ[in_body], axis=1).sum() if in_body.any() else 0.0
    total_gamma = np.linalg.norm(res.circ, axis=1).sum()
    assert body_gamma < 1e-12 * total_gamma, (
        f"correction injected |Γ|={body_gamma:.3e} inside the body"
    )
    # the exterior resolved field must still benefit from the correction
    assert res.strength_corr_residual_post < res.strength_corr_residual_pre
