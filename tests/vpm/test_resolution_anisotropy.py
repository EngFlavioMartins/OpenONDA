"""Stage 1D: is the Lagrangian deformation gradient a viable anisotropic
resolution measure, and can it be tracked cheaply from the velocity
gradient already computed for stretching/LES?

Two independent CPU oracles, no production code:

1. Material-neighbour least-squares fit of F from a *fixed* (reference-time)
   neighbour graph -- ``estimate_deformation_gradient``. This is a geometric
   ground truth: it never recomputes neighbours after deformation, which is
   exactly the selection bias that falsified the k-NN filter-width estimator
   (see test_local_spacing.py).

2. Direct time integration of Fdot = L(t) F  (equivalently Bdot = LB + BL^T
   for B = F F^T), from an externally supplied velocity-gradient history
   L(t) = grad(u) -- ``integrate_deformation_gradient`` /
   ``integrate_cauchy_green``. This is the candidate production mechanism:
   OpenONDA already computes and stores ``particles.velocity_gradient``
   every step for stretching/LES, so no new spatial query is needed.

Synthetic tests compare both oracles against closed-form affine solutions.
"""

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.spatial import cKDTree

pytestmark = pytest.mark.unit


# ---- geometric (material-neighbour) oracle --------------------------------


def build_material_neighbors(reference_positions: np.ndarray, k: int) -> np.ndarray:
    """Fixed neighbour IDs computed once at the reference configuration.
    Never recompute after deformation -- see module docstring."""
    tree = cKDTree(reference_positions)
    _, idx = tree.query(reference_positions, k=k + 1)
    return idx[:, 1:]  # drop self


def estimate_deformation_gradient(
    X: np.ndarray, x: np.ndarray, neighbor_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-particle weighted (uniform-weight) least-squares fit of the local
    affine map F such that x_j - x_i ~= F (X_j - X_i) over fixed material
    neighbours j of i.

    Returns (F, condition_numbers) with F shape (N,3,3).
    """
    N = len(X)
    F = np.empty((N, 3, 3))
    cond = np.empty(N)
    for i in range(N):
        nbrs = neighbor_ids[i]
        Xi = X[nbrs] - X[i]  # (k,3) reference offsets
        xi = x[nbrs] - x[i]  # (k,3) deformed offsets
        num = xi.T @ Xi  # sum_j r_ij outer Xi_ij
        den = Xi.T @ Xi  # sum_j Xi_ij outer Xi_ij
        cond[i] = np.linalg.cond(den)
        # pinv degrades gracefully (finite least-norm result) for a rank-
        # deficient den (e.g. exactly coplanar neighbours); inv() would
        # raise. The large/inf condition number is the actual reject signal.
        F[i] = num @ np.linalg.pinv(den)
    return F, cond


def principal_stretches(F: np.ndarray) -> np.ndarray:
    """Singular values of F, descending: s1 >= s2 >= s3."""
    return np.linalg.svd(F, compute_uv=False)


# ---- synthetic clouds -------------------------------------------------------


def jittered_lattice(n_per_axis: int, spacing: float, jitter_frac: float, seed: int) -> np.ndarray:
    coords = np.arange(n_per_axis) * spacing
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")
    pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
    pts = pts - pts.mean(axis=0)
    rng = np.random.default_rng(seed)
    return pts + rng.uniform(-jitter_frac * spacing, jitter_frac * spacing, size=pts.shape)


def interior_mask(n_per_axis: int, margin: int) -> np.ndarray:
    idx = np.arange(n_per_axis)
    ii, jj, kk = np.meshgrid(idx, idx, idx, indexing="ij")
    keep = (
        (ii >= margin)
        & (ii < n_per_axis - margin)
        & (jj >= margin)
        & (jj < n_per_axis - margin)
        & (kk >= margin)
        & (kk < n_per_axis - margin)
    )
    return keep.ravel()


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    a = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(a)
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q


def rotation_about_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def skew(w: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


N_PER_AXIS = 12
SPACING = 1.0
MARGIN = 3
K_NEIGHBORS = 18


@pytest.fixture(scope="module")
def reference_cloud():
    X = jittered_lattice(N_PER_AXIS, SPACING, jitter_frac=0.0, seed=0)
    mask = interior_mask(N_PER_AXIS, MARGIN)
    nbrs = build_material_neighbors(X, K_NEIGHBORS)
    return X, mask, nbrs


def _apply_and_recover(X, mask, nbrs, F_applied):
    x = X @ F_applied.T
    F_hat, cond = estimate_deformation_gradient(X, x, nbrs)
    return F_hat[mask], cond[mask]


def test_identity_recovers_identity(reference_cloud):
    X, mask, nbrs = reference_cloud
    F_hat, cond = _apply_and_recover(X, mask, nbrs, np.eye(3))
    np.testing.assert_allclose(F_hat, np.broadcast_to(np.eye(3), F_hat.shape), atol=1e-9)
    assert np.all(cond < 1e6)


def test_rigid_rotation_all_singular_values_unity():
    rng = np.random.default_rng(1)
    X = jittered_lattice(N_PER_AXIS, SPACING, jitter_frac=0.0, seed=0)
    mask = interior_mask(N_PER_AXIS, MARGIN)
    nbrs = build_material_neighbors(X, K_NEIGHBORS)
    R = random_rotation(rng)
    F_hat, _ = _apply_and_recover(X, mask, nbrs, R)
    s = principal_stretches(F_hat)
    np.testing.assert_allclose(s, np.ones_like(s), atol=1e-8)


@pytest.mark.parametrize("lam", [1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
def test_volume_preserving_uniaxial_stretch_exact_recovery(reference_cloud, lam):
    X, mask, nbrs = reference_cloud
    F_applied = np.diag([lam, lam**-0.5, lam**-0.5])
    F_hat, _ = _apply_and_recover(X, mask, nbrs, F_applied)
    s = principal_stretches(F_hat)
    expected = np.sort([lam, lam**-0.5, lam**-0.5])[::-1]
    np.testing.assert_allclose(s, np.broadcast_to(expected, s.shape), atol=1e-7, rtol=1e-6)
    # This is the case that killed r_k: s_max must actually equal lambda,
    # not collapse toward the compressed spacing lambda^-1/2.
    assert np.all(s[:, 0] == pytest.approx(lam, rel=1e-6))


@pytest.mark.parametrize("lam", [2.0, 8.0, 32.0])
def test_pancake_deformation_exact_recovery(reference_cloud, lam):
    """Complementary geometry: two directions elongated, one compressed.
    det F = lam^-1 * lam^1/2 * lam^1/2 = 1."""
    X, mask, nbrs = reference_cloud
    F_applied = np.diag([lam**-1.0, lam**0.5, lam**0.5])
    F_hat, _ = _apply_and_recover(X, mask, nbrs, F_applied)
    s = principal_stretches(F_hat)
    expected = np.sort([lam**-1.0, lam**0.5, lam**0.5])[::-1]
    np.testing.assert_allclose(s, np.broadcast_to(expected, s.shape), atol=1e-6, rtol=1e-6)


def test_simple_shear_matches_direct_numpy_svd(reference_cloud):
    X, mask, nbrs = reference_cloud
    gamma = 0.6
    F_applied = np.array([[1.0, gamma, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    F_hat, _ = _apply_and_recover(X, mask, nbrs, F_applied)
    s_direct = np.linalg.svd(F_applied, compute_uv=False)
    s_direct = np.sort(s_direct)[::-1]
    s_hat = principal_stretches(F_hat)
    np.testing.assert_allclose(s_hat, np.broadcast_to(s_direct, s_hat.shape), atol=1e-7, rtol=1e-6)


def test_rotated_stretching_singular_values_invariant(reference_cloud):
    X, mask, nbrs = reference_cloud
    lam = 6.0
    F0 = np.diag([lam, lam**-0.5, lam**-0.5])
    s_unrotated = np.sort(np.linalg.svd(F0, compute_uv=False))[::-1]

    rng = np.random.default_rng(7)
    for _ in range(4):
        R = random_rotation(rng)
        F_applied = R @ F0 @ R.T
        F_hat, _ = _apply_and_recover(X, mask, nbrs, F_applied)
        s_hat = principal_stretches(F_hat)
        np.testing.assert_allclose(
            s_hat, np.broadcast_to(s_unrotated, s_hat.shape), atol=1e-6, rtol=1e-6
        )


def test_noisy_lattice_still_recovers_exact_F():
    """The LS-fit oracle solves a linear regression, not a shell-counting
    statistic like r_k -- for a genuinely affine map it must recover F
    exactly regardless of reference-cloud irregularity."""
    X = jittered_lattice(N_PER_AXIS, SPACING, jitter_frac=0.15, seed=99)
    mask = interior_mask(N_PER_AXIS, MARGIN)
    nbrs = build_material_neighbors(X, K_NEIGHBORS)
    for F_applied in (
        np.diag([3.0, 3.0**-0.5, 3.0**-0.5]),
        np.array([[1.0, 0.4, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    ):
        F_hat, cond = _apply_and_recover(X, mask, nbrs, F_applied)
        assert np.all(cond < 1e6)
        np.testing.assert_allclose(
            F_hat, np.broadcast_to(F_applied, F_hat.shape), atol=1e-6, rtol=1e-6
        )


def test_permutation_invariance(reference_cloud):
    X, mask, nbrs = reference_cloud
    lam = 5.0
    F_applied = np.diag([lam, lam**-0.5, lam**-0.5])
    x = X @ F_applied.T

    rng = np.random.default_rng(3)
    perm = rng.permutation(len(X))
    inv_perm = np.argsort(perm)
    X_p, x_p = X[perm], x[perm]
    # The point of this test is that recomputing on X_p, x_p and reading off
    # entry perm^{-1}(i) matches the unpermuted computation for particle i.
    nbrs_p_native = build_material_neighbors(X_p, K_NEIGHBORS)
    F_hat_orig, _ = estimate_deformation_gradient(X, x, nbrs)
    F_hat_perm, _ = estimate_deformation_gradient(X_p, x_p, nbrs_p_native)

    np.testing.assert_allclose(F_hat_perm[inv_perm][mask], F_hat_orig[mask], atol=1e-8)


def test_degenerate_coplanar_neighborhood_flagged_by_condition_number():
    # All neighbours in the z=0 plane: the reference moment matrix is
    # singular in z, must be flagged by a large condition number.
    rng = np.random.default_rng(5)
    center = np.zeros(3)
    planar_nbrs = rng.uniform(-1, 1, size=(20, 3))
    planar_nbrs[:, 2] = 0.0
    Xi = planar_nbrs - center  # reference offsets to particle 0's neighbours
    den = Xi.T @ Xi
    cond = np.linalg.cond(den)
    assert cond > 1e10


# ---- gradient-integrated deformation (Fdot = L F, Bdot = LB + BL^T) -------


def integrate_F(L_func, F0: np.ndarray, T: float, n_steps: int) -> np.ndarray:
    """RK4 integration of Fdot(t) = L(t) F(t)."""
    time_step_size = T / n_steps
    F = F0.copy()
    t = 0.0
    for _ in range(n_steps):
        k1 = L_func(t) @ F
        k2 = L_func(t + 0.5 * time_step_size) @ (F + 0.5 * time_step_size * k1)
        k3 = L_func(t + 0.5 * time_step_size) @ (F + 0.5 * time_step_size * k2)
        k4 = L_func(t + time_step_size) @ (F + time_step_size * k3)
        F = F + (time_step_size / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += time_step_size
    return F


def integrate_B(L_func, B0: np.ndarray, T: float, n_steps: int) -> np.ndarray:
    """RK4 integration of Bdot(t) = L(t) B(t) + B(t) L(t)^T."""

    def rhs(t, B):
        L = L_func(t)
        return L @ B + B @ L.T

    time_step_size = T / n_steps
    B = B0.copy()
    t = 0.0
    for _ in range(n_steps):
        k1 = rhs(t, B)
        k2 = rhs(t + 0.5 * time_step_size, B + 0.5 * time_step_size * k1)
        k3 = rhs(t + 0.5 * time_step_size, B + 0.5 * time_step_size * k2)
        k4 = rhs(t + time_step_size, B + time_step_size * k3)
        B = B + (time_step_size / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += time_step_size
    return B


def convergence_order(time_step_sizes: np.ndarray, errors: np.ndarray) -> float:
    slope, _ = np.polyfit(np.log(time_step_sizes), np.log(np.maximum(errors, 1e-300)), 1)
    return slope


def test_constant_incompressible_extension_matches_matrix_exponential_and_converges():
    a = 0.35
    L = np.diag([a, -a / 2.0, -a / 2.0])
    T = 1.0
    F_exact = expm(L * T)

    step_counts = np.array([5, 10, 20, 40, 80])
    time_step_sizes = T / step_counts
    errors = np.array(
        [np.linalg.norm(integrate_F(lambda t: L, np.eye(3), T, n) - F_exact) for n in step_counts]
    )
    assert errors[-1] < 1e-6
    order = convergence_order(
        time_step_sizes[:-1], errors[:-1]
    )  # last point near float noise floor
    assert order > 3.5, f"expected ~4th order RK4 convergence, measured {order:.2f}"

    B_num = integrate_B(lambda t: L, np.eye(3), T, 40)
    B_exact = F_exact @ F_exact.T
    np.testing.assert_allclose(B_num, B_exact, atol=1e-8)
    assert np.linalg.det(B_num) == pytest.approx(1.0, abs=1e-7)  # incompressible: det B == 1


def test_rigid_rotation_no_false_anisotropy_over_many_periods():
    """Integrating B directly from B0=I under pure rotation is a degenerate
    non-test: Bdot = Omega*I + I*Omega^T = Omega + Omega^T = 0 identically
    (Omega skew), so B=I is an exact fixed point and no numerical work ever
    happens. The real question is whether F(t), which *does* evolve
    non-trivially (F(t) = R(t) = expm(Omega t)), accumulates orthogonality
    drift under repeated RK4 stepping -- checked here via B = F F^T."""
    omega = 1.3
    L = skew(np.array([0.0, 0.0, omega]))
    period = 2.0 * np.pi / omega
    n_periods = 10
    T = n_periods * period
    n_steps = 4000  # ~400 steps/period

    F = np.eye(3)
    time_step_size = T / n_steps
    checkpoints = []
    for step in range(n_steps):
        F = integrate_F(lambda tt: L, F, time_step_size, 1)
        if step % (n_steps // (4 * n_periods)) == 0:
            B = F @ F.T
            eigvals = np.linalg.eigvalsh(B)
            anisotropy = np.sqrt(eigvals.max() / max(eigvals.min(), 1e-300))
            checkpoints.append(anisotropy)
    checkpoints = np.array(checkpoints)
    assert np.all(checkpoints < 1.001), (
        f"pure rotation spuriously created anisotropy up to {checkpoints.max():.6f} "
        f"after {n_periods} periods at dt={time_step_size:.4g}"
    )


def test_simple_shear_constant_L_matches_exact():
    gamma_dot = 0.7
    T = 2.0
    L = np.array([[0.0, gamma_dot, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    F_exact_closed_form = np.eye(3) + L * T  # L^2 = 0 (nilpotent)
    np.testing.assert_allclose(expm(L * T), F_exact_closed_form, atol=1e-12)

    F_num = integrate_F(lambda t: L, np.eye(3), T, 40)
    np.testing.assert_allclose(F_num, F_exact_closed_form, atol=1e-9)


def test_rotating_principal_strain_matches_analytic_F():
    """Non-constant, non-commuting L(t): principal stretch axes rotate at
    rate omega while stretching incompressibly along them. Exact solution
    F(t) = R(t) D(t) is constructed directly and L(t) = Fdot F^-1 derived
    analytically (see conversation record), giving a genuine closed-form
    check for time-dependent L."""
    omega = 0.8
    a = 0.25
    T = 2.0
    Lambda = np.diag([a, -a / 2.0, -a / 2.0])
    Sz = skew(np.array([0.0, 0.0, 1.0]))

    def R(t):
        return rotation_about_z(omega * t)

    def D(t):
        return np.diag([np.exp(a * t), np.exp(-a * t / 2.0), np.exp(-a * t / 2.0)])

    def F_exact(t):
        return R(t) @ D(t)

    def L(t):
        Rt = R(t)
        return omega * Sz + Rt @ Lambda @ Rt.T

    step_counts = np.array([10, 20, 40, 80, 160])
    time_step_sizes = T / step_counts
    errors = np.array(
        [np.linalg.norm(integrate_F(L, np.eye(3), T, n) - F_exact(T)) for n in step_counts]
    )
    assert errors[-1] < 1e-6
    order = convergence_order(time_step_sizes[:-1], errors[:-1])
    assert order > 3.0, (
        f"expected ~4th order convergence for time-dependent L, measured {order:.2f}"
    )
