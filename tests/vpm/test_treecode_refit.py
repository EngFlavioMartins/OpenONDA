"""
Treecode topology-reuse (refit) tests.

`refit()` reuses the LBVH topology from the previous `build()` and updates only
the position-dependent multipoles.  For a small per-stage displacement (< h)
the refitted tree must produce velocity/gradients that match a full rebuild
at the same displaced position to well within the Barnes–Hut tolerance.
"""

import math

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.acceleration.treecode_gpu import TaichiTreecode

_ERF = np.vectorize(math.erf)


def _exact_velocities(pos, circ, rad):
    """Direct O(N²) regularized Biot–Savart, matching the treecode Gaussian q."""
    N = len(pos)
    two_over_sqrt_pi = 1.1283791671
    one_over_four_pi = 0.07957747154594767
    v = np.zeros((N, 3))
    for i in range(N):
        r = pos[i] - pos  # (N,3): x_i - x_j
        rmag = np.linalg.norm(r, axis=1)
        sigma = 0.5 * (rad[i] + rad)
        rho = np.divide(rmag, sigma, out=np.zeros_like(rmag), where=sigma > 0)
        q = (_ERF(rho) - two_over_sqrt_pi * rho * np.exp(-rho * rho)) * one_over_four_pi
        cross = np.cross(r, circ)  # r × Γ_j
        mask = rmag > 1e-10
        contrib = np.zeros((N, 3))
        contrib[mask] = -(q[mask] / rmag[mask] ** 3)[:, None] * cross[mask]
        v[i] = contrib.sum(axis=0)
    return v


@pytest.fixture(scope="module", autouse=True)
def _ti_init():
    ti.init(arch=ti.cpu, default_fp=ti.f32, random_seed=0)
    yield


def _make_tree(n, theta=0.5):
    tree = TaichiTreecode(max_n_particles=n + 16, kernel_type="GAUSSIAN")
    tree.theta = theta
    tree.theta_sq = theta * theta
    return tree


def _velocities(tree, pos_np, circ_np, rad_np, build=True, refit=False):
    N = len(pos_np)
    pos = ti.Vector.field(3, ti.f32, shape=N)
    circ = ti.Vector.field(3, ti.f32, shape=N)
    rad = ti.field(ti.f32, shape=N)
    pos.from_numpy(pos_np.astype(np.float32))
    circ.from_numpy(circ_np.astype(np.float32))
    rad.from_numpy(rad_np.astype(np.float32))
    if refit:
        tree.refit(pos, N)
    else:
        tree.build(pos, circ, rad, N)
    tree.compute_velocities_gpu(np.zeros(3, np.float32))
    return tree.velocity.to_numpy()[:N].copy()


def test_refit_accuracy_matches_full_build():
    """Refit's error vs the exact field must be no worse than a full rebuild's.

    Two different tree topologies each carry the Barnes–Hut approximation error,
    so refit and full-build velocity differ by O(tree error) — that is NOT a
    refit defect.  The correct invariant is that refit does not *degrade*
    accuracy: its error against the exact O(N²) sum matches a full rebuild's.
    """
    rng = np.random.default_rng(1)
    N = 1500
    h = 0.06
    pos0 = rng.uniform(-1, 1, (N, 3))
    circ = rng.normal(0, 0.1, (N, 3))
    rad = np.full(N, 2 * h)

    tree = _make_tree(N, theta=0.3)
    _velocities(tree, pos0, circ, rad, build=True)  # build at x_n

    # Displace by < h (typical RK sub-stage move).
    pos1 = pos0 + rng.uniform(-0.3 * h, 0.3 * h, (N, 3))
    v_exact = _exact_velocities(pos1, circ, rad)
    v_refit = _velocities(tree, pos1, circ, rad, refit=True)

    tree_full = _make_tree(N, theta=0.3)
    v_full = _velocities(tree_full, pos1, circ, rad, build=True)

    denom = np.linalg.norm(v_exact) + 1e-30
    err_refit = np.linalg.norm(v_refit - v_exact) / denom
    err_full = np.linalg.norm(v_full - v_exact) / denom
    # Refit accuracy must track the full build (allow a small margin for the
    # slightly looser node bounds of the reused topology).
    assert err_refit < err_full + 0.01, (
        f"refit degraded accuracy: err_refit={err_refit:.2e} vs err_full={err_full:.2e}"
    )


def test_refit_zero_displacement_matches_build_exactly():
    """With no displacement, refit must reproduce the full build to ~machine eps."""
    rng = np.random.default_rng(7)
    N = 800
    pos = rng.uniform(-1, 1, (N, 3))
    circ = rng.normal(0, 0.1, (N, 3))
    rad = np.full(N, 0.12)

    tree = _make_tree(N, theta=0.4)
    v_build = _velocities(tree, pos, circ, rad, build=True)
    v_refit = _velocities(tree, pos, circ, rad, refit=True)  # same position

    rel = np.linalg.norm(v_refit - v_build) / (np.linalg.norm(v_build) + 1e-30)
    assert rel < 1e-5, f"refit at zero displacement differs by {rel:.2e}"


def test_circulation_refit_matches_full_rebuild_exactly():
    """Fixed-position RK stretching may reuse topology after vortex_strength change."""
    rng = np.random.default_rng(17)
    N = 900
    pos = rng.uniform(-1, 1, (N, 3)).astype(np.float32)
    circ0 = rng.normal(0, 0.1, (N, 3)).astype(np.float32)
    circ1 = (circ0 + rng.normal(0, 0.02, (N, 3))).astype(np.float32)
    rad = np.full(N, 0.12, dtype=np.float32)

    pos_field = ti.Vector.field(3, ti.f32, shape=N)
    circ0_field = ti.Vector.field(3, ti.f32, shape=N)
    circ1_field = ti.Vector.field(3, ti.f32, shape=N)
    rad_field = ti.field(ti.f32, shape=N)
    pos_field.from_numpy(pos)
    circ0_field.from_numpy(circ0)
    circ1_field.from_numpy(circ1)
    rad_field.from_numpy(rad)

    refitted = _make_tree(N, theta=0.3)
    refitted.build(pos_field, circ0_field, rad_field, N)
    sorted_before = refitted.sorted_indices.to_numpy()[:N].copy()
    refitted.refit_vortex_strength(circ1_field, N)
    refitted.compute_velocity_gradients_gpu()
    gradient_refit = refitted.velocity_gradient.to_numpy()[:N].copy()

    rebuilt = _make_tree(N, theta=0.3)
    rebuilt.build(pos_field, circ1_field, rad_field, N)
    rebuilt.compute_velocity_gradients_gpu()
    gradient_rebuild = rebuilt.velocity_gradient.to_numpy()[:N].copy()

    np.testing.assert_array_equal(refitted.sorted_indices.to_numpy()[:N], sorted_before)
    np.testing.assert_allclose(gradient_refit, gradient_rebuild, rtol=2e-6, atol=2e-6)


def test_refit_preserves_topology_fields():
    rng = np.random.default_rng(2)
    N = 512
    pos0 = rng.uniform(-1, 1, (N, 3))
    circ = rng.normal(0, 0.1, (N, 3))
    rad = np.full(N, 0.1)

    tree = _make_tree(N)
    _velocities(tree, pos0, circ, rad, build=True)
    sorted_before = tree.sorted_indices.to_numpy()[:N].copy()

    pos1 = pos0 + rng.uniform(-0.01, 0.01, (N, 3))
    _velocities(tree, pos1, circ, rad, refit=True)
    sorted_after = tree.sorted_indices.to_numpy()[:N].copy()

    # Refit must NOT touch the sorted order / topology.
    assert np.array_equal(sorted_before, sorted_after)


def test_refit_rejects_particle_count_change():
    rng = np.random.default_rng(3)
    N = 256
    pos = rng.uniform(-1, 1, (N, 3))
    circ = rng.normal(0, 0.1, (N, 3))
    rad = np.full(N, 0.1)
    tree = _make_tree(N + 8)
    _velocities(tree, pos, circ, rad, build=True)

    posN = ti.Vector.field(3, ti.f32, shape=N + 4)
    with pytest.raises(RuntimeError):
        tree.refit(posN, N + 4)
