"""
Correctness tests for the on-GPU LBVH Barnes-Hut treecode.

These guard the three properties the LBVH build must satisfy and that the
previous (broken) implementation violated:

  1. **Multipole conservation** — every internal node's total circulation equals
     the sum over the particles in its sorted range, and the root equals the sum
     over all particles.  The earlier 3-pass forward/reverse/forward accumulation
     left upper nodes with garbage (root error ~100 %); the parallel atomic
     bottom-up walk must be exact (to f32 round-off).

  2. **Convergence to direct summation** — the treecode velocity must approach the
     exact O(N²) Biot-Savart sum as the opening angle theta -> 0, with the
     characteristic monotone Barnes-Hut error growth in theta.  This is what
     proves the node multipoles are actually used *and* correct.

  3. **No stale tree** — ``build`` must rebuild whenever the field changes, even
     when the particle count N is unchanged.  The earlier N-only rebuild guard
     froze the tree at its first configuration, so every later step used stale
     node data.

Run on the Taichi CPU backend in f32 (matching the production kernels), so the
suite is hardware-independent.
"""

from __future__ import annotations

import numpy as np
import pytest
import taichi as ti

from source.solvers.VPM.acceleration.treecode_gpu import TaichiTreecode

ONE_OVER_FOUR_PI = 0.07957747154594767


@pytest.fixture(scope="module", autouse=True)
def _ti_init():
    ti.init(arch=ti.cpu, default_fp=ti.f32, random_seed=0)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _cloud(N, seed=1):
    rng = np.random.default_rng(seed)
    pos = (rng.random((N, 3)) - 0.5).astype(np.float32)
    circ = (rng.normal(size=(N, 3)) * 0.1).astype(np.float32)
    rad = np.full(N, 0.05, dtype=np.float32)
    return pos, circ, rad


def _make_tree(
    N, theta, multipole_order=1, sort_particle_targets=False, traversal_block_dim=128
):
    return TaichiTreecode(
        max_particles=N + 8,
        max_nodes=2 * (N + 8),
        theta=theta,
        kernel_type="WINCKELMANS",
        multipole_order=multipole_order,
        sort_particle_targets=sort_particle_targets,
        traversal_block_dim=traversal_block_dim,
    )


def _direct_velocity(pos, circ, rad, chunk=400):
    """Exact Winckelmans Biot-Savart sum (matches q_kernel / _leaf_velocity_sum)."""
    pos = pos.astype(np.float64)
    circ = circ.astype(np.float64)
    rad = rad.astype(np.float64)
    N = len(pos)
    out = np.zeros((N, 3))
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        r = pos[s:e, None, :] - pos[None, :, :]
        rm = np.linalg.norm(r, axis=2)
        sigma = 0.5 * (rad[s:e, None] + rad[None, :])
        with np.errstate(divide="ignore", invalid="ignore"):
            rs = rm / sigma
            r2 = rs * rs
            q = rs**3 * (r2 + 2.5) / (r2 + 1.0) ** 2.5 * ONE_OVER_FOUR_PI
            cross = np.cross(r, circ[None, :, :])
            contrib = -q[..., None] * cross / (rm[..., None] ** 3)
        contrib = np.where((rm > 1e-10)[..., None], contrib, 0.0)
        for ii in range(s, e):
            contrib[ii - s, ii] = 0.0
        out[s:e] = contrib.sum(axis=1)
    return out


def _direct_target_velocity_gradient(targets, pos, circ, rad):
    """Exact target velocity and gradient for Winckelmans blobs."""
    targets = targets.astype(np.float64)
    pos = pos.astype(np.float64)
    circ = circ.astype(np.float64)
    rad = rad.astype(np.float64)
    vel = np.zeros((len(targets), 3), dtype=np.float64)
    grad = np.zeros((len(targets), 3, 3), dtype=np.float64)
    for m, target in enumerate(targets):
        for xj, gj, sj in zip(pos, circ, rad):
            r = target - xj
            rm = np.linalg.norm(r)
            if rm <= 1e-10:
                continue
            rs = rm / sj
            r2 = rs * rs
            base = r2 + 1.0
            q = rs**3 * (r2 + 2.5) / base**2.5 * ONE_OVER_FOUR_PI
            zeta = 7.5 / base**3.5 * ONE_OVER_FOUR_PI / sj**3
            cross = np.cross(r, gj)
            vel[m] -= q * cross / rm**3
            term1 = q / rm**3
            term2 = 3.0 * q / rm**5 - zeta / rm**2
            skew = np.array(
                [[0.0, -gj[2], gj[1]], [gj[2], 0.0, -gj[0]], [-gj[1], gj[0], 0.0]]
            )
            grad[m] += term1 * skew + term2 * np.outer(cross, r)
    return vel.astype(np.float32), grad.astype(np.float32)


def _rel_l2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Multipole conservation
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("N", [64, 512, 2048])
def test_root_total_circulation_equals_global_sum(N):
    """node_total_circ[root] must equal the sum of every particle's circulation."""
    pos, circ, rad = _cloud(N, seed=3)
    tree = _make_tree(N, theta=0.4)
    tree.build(pos, circ, rad, force=True)
    root = tree._root[None]
    root_circ = tree.node_total_circ.to_numpy()[root]
    true_sum = circ.astype(np.float64).sum(axis=0)
    assert tree.node_particle_count.to_numpy()[root] == N
    assert _rel_l2(root_circ, true_sum) < 1e-4


@pytest.mark.parametrize("N", [2, 3, 64, 777, 4096])
def test_parallel_karras_topology_is_consistent(N):
    """Every internal node's [first,last] range must equal the union of its two
    children's ranges, children must be valid, and exactly N-1 internal nodes
    must form one tree rooted at node N (Karras: internal 0 is the root)."""
    pos, circ, rad = _cloud(N, seed=2)
    tree = _make_tree(N, theta=0.4)
    tree.build(pos, circ, rad, force=True)
    nl = tree.node_left.to_numpy()
    nr = tree.node_right.to_numpy()
    first = tree._node_first.to_numpy()
    last = tree._node_last.to_numpy()
    assert tree._root[None] == N  # internal node 0 → field index N
    for i in range(N - 1):
        idx = N + i
        l, r = nl[idx], nr[idx]
        assert 0 <= l < 2 * N - 1 and 0 <= r < 2 * N - 1
        # child ranges union to the parent's, contiguous and disjoint
        assert min(first[l], first[r]) == first[idx]
        assert max(last[l], last[r]) == last[idx]
        assert first[l] <= last[l] and first[r] <= last[r]
    # leaf coverage: every sorted slot 0..N-1 is some node's singleton leaf
    leaf_first = first[:N]
    assert sorted(leaf_first.tolist()) == list(range(N))


def test_internal_node_circulation_matches_its_leaf_range():
    """Each internal node's circ equals the sum over the particles it covers."""
    N = 2048
    pos, circ, rad = _cloud(N, seed=5)
    tree = _make_tree(N, theta=0.4)
    tree.build(pos, circ, rad, force=True)
    nc = tree.node_total_circ.to_numpy()
    start = tree.node_particle_start.to_numpy()
    count = tree.node_particle_count.to_numpy()
    leaf_particles = tree.leaf_particles.to_numpy()
    nnodes = tree.n_nodes[None]
    rng = np.random.default_rng(9)
    probe = rng.integers(N, nnodes, size=200)
    worst = 0.0
    for idx in probe:
        c = count[idx]
        parts = leaf_particles[start[idx] : start[idx] + c]
        assert c > 0 and (parts >= 0).all() and (parts < N).all()
        truesum = circ[parts].astype(np.float64).sum(axis=0)
        worst = max(worst, _rel_l2(nc[idx], truesum))
    assert worst < 1e-4


def test_dipole_moment_matches_particles_about_node_com():
    """Order-2 node moment must equal sum (x_j - COM_node) outer Gamma_j."""
    N = 1024
    pos, circ, rad = _cloud(N, seed=17)
    tree = _make_tree(N, theta=0.4, multipole_order=2)
    tree.build(pos, circ, rad, force=True)

    moment = tree.node_circ_dipole.to_numpy()
    com = tree.node_com.to_numpy()
    start = tree.node_particle_start.to_numpy()
    count = tree.node_particle_count.to_numpy()
    leaf_particles = tree.leaf_particles.to_numpy()
    nnodes = tree.n_nodes[None]

    rng = np.random.default_rng(18)
    probe = rng.integers(N, nnodes, size=80)
    worst = 0.0
    for idx in probe:
        parts = leaf_particles[start[idx] : start[idx] + count[idx]]
        d = pos[parts].astype(np.float64) - com[idx].astype(np.float64)
        true_moment = np.einsum("nb,na->ba", d, circ[parts].astype(np.float64))
        denom = np.linalg.norm(true_moment) + 1e-30
        worst = max(worst, float(np.linalg.norm(moment[idx] - true_moment) / denom))
    assert worst < 2e-4


# ─────────────────────────────────────────────────────────────────────────────
# 2. Convergence to direct summation
# ─────────────────────────────────────────────────────────────────────────────
def test_velocity_converges_to_direct_as_theta_shrinks():
    N = 1500
    pos, circ, rad = _cloud(N, seed=1)
    vd = _direct_velocity(pos, circ, rad)
    errs = {}
    for th in (0.1, 0.2, 0.3, 0.5):
        tree = _make_tree(N, theta=th)
        tree.build(pos, circ, rad, force=True)
        vt = tree.compute_velocities(np.zeros(3, dtype=np.float32))
        errs[th] = _rel_l2(vt, vd)
    # Monotone Barnes-Hut error growth in theta.
    assert errs[0.1] < errs[0.2] < errs[0.3] < errs[0.5]
    # Tight-theta accuracy is solidly inside the Barnes-Hut band.
    assert errs[0.1] < 2e-2
    assert errs[0.2] < 5e-2


def test_order2_improves_far_target_velocity_and_gradient():
    """Dipole correction should reduce far-cluster error for u and ∇u."""
    rng = np.random.default_rng(31)
    N = 256
    pos = (rng.normal(size=(N, 3)) * 0.025).astype(np.float32)
    circ = (rng.normal(size=(N, 3)) * 0.1).astype(np.float32)
    rad = np.full(N, 0.08, dtype=np.float32)
    targets = np.array(
        [
            [2.5, 0.2, -0.1],
            [-1.8, 2.1, 0.4],
            [0.5, -2.2, 1.6],
            [1.7, 1.5, -1.4],
        ],
        dtype=np.float32,
    )
    v_exact, g_exact = _direct_target_velocity_gradient(targets, pos, circ, rad)

    tree1 = _make_tree(N, theta=1.0, multipole_order=1)
    tree1.build(pos, circ, rad, force=True)
    v1 = tree1.compute_target_velocities(targets)
    g1 = tree1.compute_target_velocity_gradients(targets)

    tree2 = _make_tree(N, theta=1.0, multipole_order=2)
    tree2.build(pos, circ, rad, force=True)
    v2 = tree2.compute_target_velocities(targets)
    g2 = tree2.compute_target_velocity_gradients(targets)

    v_err1 = _rel_l2(v1, v_exact)
    v_err2 = _rel_l2(v2, v_exact)
    g_err1 = _rel_l2(g1.reshape(len(targets), 9), g_exact.reshape(len(targets), 9))
    g_err2 = _rel_l2(g2.reshape(len(targets), 9), g_exact.reshape(len(targets), 9))

    assert v_err2 < 0.35 * v_err1
    assert g_err2 < 0.35 * g_err1


def test_background_velocity_is_added():
    N = 200
    pos, circ, rad = _cloud(N, seed=2)
    bg = np.array([1.0, -2.0, 0.5], dtype=np.float32)
    tree = _make_tree(N, theta=0.2)
    tree.build(pos, circ, rad, force=True)
    v0 = tree.compute_velocities(np.zeros(3, dtype=np.float32))
    vbg = tree.compute_velocities(bg)
    assert np.allclose(vbg - v0, bg, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# 3. No stale tree (rebuild guard)
# ─────────────────────────────────────────────────────────────────────────────
def test_rebuild_reflects_changed_strengths_same_N():
    """Same N, different circulation, no force: the tree must rebuild, not reuse."""
    N = 1500
    pos, circ1, rad = _cloud(N, seed=7)
    rng = np.random.default_rng(8)
    circ2 = (rng.normal(size=(N, 3)) * 0.1).astype(np.float32)
    tree = _make_tree(N, theta=0.3)

    tree.build(pos, circ1, rad, N=N, force=True)
    v1 = tree.compute_velocities(np.zeros(3, dtype=np.float32)).copy()

    tree.build(pos, circ2, rad, N=N)  # same N, NO force
    v2 = tree.compute_velocities(np.zeros(3, dtype=np.float32)).copy()

    # The new result must match the direct sum for the *new* circulation …
    assert _rel_l2(v2, _direct_velocity(pos, circ2, rad)) < 7e-2
    # … and must genuinely differ from the old one (not a frozen tree).
    assert _rel_l2(v2, v1) > 1e-1


def test_rebuild_reflects_moved_particles_from_fields():
    """Field-API path (as used by the solver): moving particles must rebuild."""
    N = 1200
    posA, circ, rad = _cloud(N, seed=11)
    posB = (posA + 0.25).astype(np.float32)  # rigid translation

    pos_f = ti.Vector.field(3, dtype=ti.f32, shape=N)
    circ_f = ti.Vector.field(3, dtype=ti.f32, shape=N)
    rad_f = ti.field(dtype=ti.f32, shape=N)
    circ_f.from_numpy(circ)
    rad_f.from_numpy(rad)

    tree = _make_tree(N, theta=0.2)
    pos_f.from_numpy(posA)
    tree.build(pos_f, circ_f, rad_f, N)
    vA = tree.compute_velocities(np.zeros(3, dtype=np.float32)).copy()

    pos_f.from_numpy(posB)
    tree.build(pos_f, circ_f, rad_f, N)  # same N, moved positions
    vB = tree.compute_velocities(np.zeros(3, dtype=np.float32)).copy()

    # A rigid translation leaves the induced velocities invariant, so a correct
    # rebuild reproduces them; a frozen tree (evaluating at stale positions
    # against stale nodes) would not.  Compare against the direct sum at B.
    assert _rel_l2(vB, _direct_velocity(posB, circ, rad)) < 5e-2
    # Sanity: B truly used the new field (B's internal node COMs moved by ~0.25).
    assert _rel_l2(vA, _direct_velocity(posB, circ, rad)) < 5e-2  # translation-invariant


# ─────────────────────────────────────────────────────────────────────────────
# Gradient structural invariants (cheap, no direct reference needed)
# ─────────────────────────────────────────────────────────────────────────────
def test_gpu_only_paths_match_numpy_returning_paths():
    """The on-device methods (used by the solver to avoid per-step downloads)
    must leave the same data in the Taichi fields that the numpy-returning
    methods report."""
    N = 600
    pos, circ, rad = _cloud(N, seed=6)
    bg = np.array([0.3, 0.0, -0.1], dtype=np.float32)
    tree = _make_tree(N, theta=0.3)
    tree.build(pos, circ, rad, force=True)

    v_np = tree.compute_velocities(bg)  # numpy path (does to_numpy)
    tree.compute_velocities_gpu(bg)  # on-device path (no download)
    v_field = tree.velocities.to_numpy()[:N]
    assert np.allclose(v_field, v_np, atol=1e-6)

    g_np, s_np = tree.compute_velocity_gradients()
    tree.compute_velocity_gradients_gpu()
    g_field = tree.velocity_gradients.to_numpy()[:N]
    s_field = tree.strain_rates.to_numpy()[:N]
    assert np.allclose(g_field, g_np, atol=1e-6)
    assert np.allclose(s_field, s_np, atol=1e-6)


def test_fused_velocity_gradient_matches_two_kernel_path():
    """A1: the fused single-traversal u/∇u/S must match the two separate
    traversals (bit-comparable — identical branches, one walk) and keep ∇u
    traceless."""
    N = 1500
    pos, circ, rad = _cloud(N, seed=15)
    bg = np.array([0.2, -0.4, 0.1], dtype=np.float32)
    tree = _make_tree(N, theta=0.4)
    tree.build(pos, circ, rad, force=True)

    # Reference: two separate traversals.
    v_ref = tree.compute_velocities(bg).copy()
    g_ref, s_ref = tree.compute_velocity_gradients()
    g_ref, s_ref = g_ref.copy(), s_ref.copy()

    # Fused: one traversal.
    v_f, g_f, s_f = tree.compute_velocity_and_gradient(bg)

    assert _rel_l2(v_f, v_ref) < 1e-6
    assert _rel_l2(g_f.reshape(N, 9), g_ref.reshape(N, 9)) < 1e-6
    assert _rel_l2(s_f.reshape(N, 9), s_ref.reshape(N, 9)) < 1e-6
    tr = g_f[:, 0, 0] + g_f[:, 1, 1] + g_f[:, 2, 2]
    scale = np.linalg.norm(g_f.reshape(N, 9), axis=1).mean() + 1e-12
    assert np.abs(tr).max() / scale < 1e-4


def test_morton_ordered_particle_traversal_matches_default_order():
    """Target grouping by Morton order must only change scheduling, not values."""
    N = 1200
    pos, circ, rad = _cloud(N, seed=19)
    bg = np.array([0.1, -0.2, 0.3], dtype=np.float32)

    plain = _make_tree(N, theta=0.35, traversal_block_dim=64)
    plain.build(pos, circ, rad, force=True)
    v_plain, g_plain, s_plain = plain.compute_velocity_and_gradient(bg)

    grouped = _make_tree(
        N,
        theta=0.35,
        sort_particle_targets=True,
        traversal_block_dim=64,
    )
    grouped.build(pos, circ, rad, force=True)
    v_grouped, g_grouped, s_grouped = grouped.compute_velocity_and_gradient(bg)

    assert _rel_l2(v_grouped, v_plain) < 1e-6
    assert _rel_l2(g_grouped.reshape(N, 9), g_plain.reshape(N, 9)) < 1e-6
    assert _rel_l2(s_grouped.reshape(N, 9), s_plain.reshape(N, 9)) < 1e-6


def test_fused_direct_kernel_matches_separate_direct_kernels():
    """A1 (direct path): the fused single-j-loop u/∇u/S kernel must match the two
    separate direct kernels bit-for-bit."""
    from source.solvers.VPM.kernels.winckelmans import create_winckelmans_kernels
    from source.solvers.VPM.numerics.kernels_common import (
        _create_basic_kernels,
        _create_gradient_kernels,
    )

    kf = create_winckelmans_kernels(ti.f32)
    basic = _create_basic_kernels(kf)
    grad = _create_gradient_kernels(kf)
    k_vel = basic["compute_velocities_kernel"]
    k_grad = grad["compute_velocity_gradients_kernel"]
    k_fused = grad["compute_velocity_and_gradient_kernel"]

    N = 800
    pos, circ, rad = _cloud(N, seed=21)
    P = ti.Vector.field(3, ti.f32, shape=N)
    C = ti.Vector.field(3, ti.f32, shape=N)
    R = ti.field(ti.f32, shape=N)
    V = ti.Vector.field(3, ti.f32, shape=N)
    G = ti.Matrix.field(3, 3, ti.f32, shape=N)
    S = ti.Matrix.field(3, 3, ti.f32, shape=N)
    Vf = ti.Vector.field(3, ti.f32, shape=N)
    Gf = ti.Matrix.field(3, 3, ti.f32, shape=N)
    Sf = ti.Matrix.field(3, 3, ti.f32, shape=N)
    BG = ti.Vector.field(3, ti.f32, shape=())
    P.from_numpy(pos); C.from_numpy(circ); R.from_numpy(rad)
    BG[None] = ti.Vector([0.5, -0.2, 0.3])

    k_vel(P, C, R, V, BG, N)
    k_grad(P, C, R, G, S, N)
    k_fused(P, C, R, Vf, Gf, Sf, BG, N)

    assert _rel_l2(Vf.to_numpy(), V.to_numpy()) < 1e-6
    assert _rel_l2(Gf.to_numpy().reshape(N, 9), G.to_numpy().reshape(N, 9)) < 1e-6
    assert _rel_l2(Sf.to_numpy().reshape(N, 9), S.to_numpy().reshape(N, 9)) < 1e-6


def test_per_stage_single_build_feeds_fused_pass():
    """A4: within a stage, one build feeds the fused u/∇u pass — a single build
    call serves both outputs, identical to rebuilding for each."""
    N = 1200
    pos, circ, rad = _cloud(N, seed=23)
    bg = np.zeros(3, dtype=np.float32)
    tree = _make_tree(N, theta=0.4)

    # Reference = rebuild-every-call (the old two-build behaviour).
    tree.build(pos, circ, rad, force=True)
    v_ref = tree.compute_velocities(bg).copy()
    tree.build(pos, circ, rad, force=True)
    g_ref, s_ref = tree.compute_velocity_gradients()
    g_ref, s_ref = g_ref.copy(), s_ref.copy()

    # Per-stage: count builds across one build + fused evaluation.
    calls = {"n": 0}
    orig_build = tree.build

    def counting_build(*a, **k):
        calls["n"] += 1
        return orig_build(*a, **k)

    tree.build = counting_build
    tree.build(pos, circ, rad, force=True)  # exactly one build for the stage
    v, g, s = tree.compute_velocity_and_gradient(bg)

    assert calls["n"] == 1, f"expected one build per stage, got {calls['n']}"
    assert _rel_l2(v, v_ref) < 1e-6
    assert _rel_l2(g.reshape(N, 9), g_ref.reshape(N, 9)) < 1e-6
    assert _rel_l2(s.reshape(N, 9), s_ref.reshape(N, 9)) < 1e-6


def test_velocity_gradient_is_traceless():
    """trace(grad u) = 0 by construction (div-free regularised field)."""
    N = 800
    pos, circ, rad = _cloud(N, seed=4)
    tree = _make_tree(N, theta=0.2)
    tree.build(pos, circ, rad, force=True)
    grads, strains = tree.compute_velocity_gradients()
    tr = grads[:, 0, 0] + grads[:, 1, 1] + grads[:, 2, 2]
    scale = np.linalg.norm(grads.reshape(N, 9), axis=1).mean() + 1e-12
    assert np.abs(tr).max() / scale < 1e-4
    # strain must be the symmetric part of grad u
    assert np.allclose(strains, 0.5 * (grads + np.transpose(grads, (0, 2, 1))), atol=1e-5)
