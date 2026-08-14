"""Exact-reference (scipy.spatial.cKDTree) characterization of the k-th
nearest-neighbour distance r_k as a candidate LES filter-width sensor h_i.

h_i is *not* validated as a general "local resolution" measure here -- only
as a candidate for the specific defect it is proposed to detect: loss of
Lagrangian resolution under volume-preserving anisotropic stretching (see
`test_anisotropic_shear_*` below, which is the deciding experiment).
"""

import numpy as np
import pytest
from scipy.spatial import cKDTree

pytestmark = pytest.mark.unit

K_VALUES = (8, 16, 32)


def kth_neighbor_distance(positions: np.ndarray, k: int) -> np.ndarray:
    """Exact r_k: distance from each point to its k-th nearest neighbour
    (self excluded). Reference oracle -- not the production estimator."""
    tree = cKDTree(positions)
    dists, _ = tree.query(positions, k=k + 1)
    return dists[:, -1]


# ---- deterministic synthetic clouds ---------------------------------------


def uniform_lattice(n_per_axis: int, spacing: float = 1.0) -> np.ndarray:
    coords = np.arange(n_per_axis) * spacing
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)


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


def jittered_lattice(n_per_axis: int, spacing: float, jitter_frac: float, seed: int) -> np.ndarray:
    pts = uniform_lattice(n_per_axis, spacing)
    rng = np.random.default_rng(seed)
    return pts + rng.uniform(-jitter_frac * spacing, jitter_frac * spacing, size=pts.shape)


def two_density_cloud(n_per_axis: int, spacing_dense: float, spacing_sparse: float):
    dense = uniform_lattice(n_per_axis, spacing_dense)
    sparse = uniform_lattice(n_per_axis, spacing_sparse)
    gap = 5.0 * spacing_sparse
    sparse[:, 0] += dense[:, 0].max() + gap
    labels = np.concatenate([np.zeros(len(dense), dtype=bool), np.ones(len(sparse), dtype=bool)])
    return np.concatenate([dense, sparse], axis=0), labels


def sheared_lattice(n_per_axis: int, spacing: float, lam: float) -> np.ndarray:
    pts = uniform_lattice(n_per_axis, spacing)
    pts = pts - pts.mean(axis=0)
    F = np.diag([lam, lam**-0.5, lam**-0.5])
    return pts @ F.T


def sheet_cloud(n_per_axis: int, spacing: float) -> np.ndarray:
    coords = np.arange(n_per_axis) * spacing
    xx, yy = np.meshgrid(coords, coords, indexing="ij")
    zz = np.zeros_like(xx)
    return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)


def filament_cloud(n_points: int, spacing: float) -> np.ndarray:
    pts = np.zeros((n_points, 3))
    pts[:, 0] = np.arange(n_points) * spacing
    return pts


def morton_boundary_cloud(n_per_side: int = 40, gap: float = 1e-3, seed: int = 5) -> np.ndarray:
    """Two dense clusters straddling x = 0.5 in the unit cube -- the top
    Morton-curve split under the standard [0,1)^3 bit-interleaving. Exact
    kNN must treat them as immediate neighbours despite the coordinate jump.
    """
    rng = np.random.default_rng(seed)
    left = rng.uniform([0.5 - 0.02, 0.0, 0.0], [0.5 - gap, 1.0, 1.0], size=(n_per_side, 3))
    right = rng.uniform([0.5 + gap, 0.0, 0.0], [0.5 + 0.02, 1.0, 1.0], size=(n_per_side, 3))
    return np.concatenate([left, right], axis=0)


# ---- mechanical correctness ------------------------------------------------


def test_permutation_invariance():
    rng = np.random.default_rng(1)
    positions = rng.uniform(-1, 1, size=(300, 3))
    perm = rng.permutation(len(positions))
    for k in K_VALUES:
        r = kth_neighbor_distance(positions, k)
        r_perm = kth_neighbor_distance(positions[perm], k)
        np.testing.assert_allclose(r_perm, r[perm], atol=1e-12)


def test_uniform_lattice_exact_shell_calibration():
    """Interior points of a simple-cubic lattice have exactly known r_k:
    r_8 = r_16 = a*sqrt(2) (18-point second shell covers ranks 7-18),
    r_32 = 2a (32-point fourth shell boundary, exact tie).
    This fixes what C_h must be for VOLUME/ADAPTIVE to agree on an
    undisturbed lattice: C_h(k=8)=C_h(k=16)=1/sqrt(2), C_h(k=32)=1/2.
    """
    n, a, margin = 12, 1.0, 3
    pts = uniform_lattice(n, a)
    mask = interior_mask(n, margin)
    expected = {8: a * np.sqrt(2), 16: a * np.sqrt(2), 32: 2.0 * a}
    for k, exp in expected.items():
        r = kth_neighbor_distance(pts, k)[mask]
        np.testing.assert_allclose(r, exp, atol=1e-9)


def test_jittered_lattice_stays_close_to_unperturbed():
    n, a, margin = 12, 1.0, 3
    mask = interior_mask(n, margin)
    base = uniform_lattice(n, a)
    jittered = jittered_lattice(n, a, jitter_frac=0.1, seed=42)
    for k in K_VALUES:
        r0 = kth_neighbor_distance(base, k)[mask]
        r1 = kth_neighbor_distance(jittered, k)[mask]
        rel = np.abs(r1 - r0) / r0
        assert np.median(rel) < 0.15
        assert np.max(rel) < 0.5


def test_two_density_cloud_identifies_coarser_half():
    positions, is_sparse = two_density_cloud(10, spacing_dense=0.5, spacing_sparse=1.5)
    for k in K_VALUES:
        r = kth_neighbor_distance(positions, k)
        mean_dense = r[~is_sparse].mean()
        mean_sparse = r[is_sparse].mean()
        assert mean_sparse > 2.0 * mean_dense


def test_rarefaction_hole_elevates_local_spacing():
    """A hole only elevates r_k once the hole radius exceeds the k-th
    neighbour's intrinsic bulk search radius -- a point sitting right on the
    hole boundary still finds ~half its cubic-lattice shell (~9 of 18 points
    at the sqrt(2) shell) on the outward-facing side alone. k=8 is too small
    relative to this hole to be perturbed at all (verified: ratio == 1.0
    exactly); k=16 and k=32 straddle the hole boundary and do respond.
    Both facts -- the k=8 blindness and the k=16/32 sensitivity -- are the
    finding, not test noise.
    """
    n, a, margin = 28, 1.0, 5
    pts = uniform_lattice(n, a)
    mask = interior_mask(n, margin)
    center = pts.mean(axis=0)

    hole_radius = 4.0
    d_all = np.linalg.norm(pts - center, axis=1)
    keep = d_all > hole_radius
    thinned = pts[keep]
    thinned_mask = mask[keep]
    d_thinned = np.linalg.norm(thinned - center, axis=1)

    near_hole = thinned_mask & (d_thinned < hole_radius + 1.5)
    far_bulk = thinned_mask & (d_thinned > hole_radius + 6.0)
    assert near_hole.sum() > 10 and far_bulk.sum() > 10

    r8 = kth_neighbor_distance(thinned, 8)
    assert r8[near_hole].mean() == pytest.approx(r8[far_bulk].mean(), rel=1e-9)

    for k in (16, 32):
        r = kth_neighbor_distance(thinned, k)
        assert r[near_hole].mean() > 1.08 * r[far_bulk].mean()


def test_sheet_and_filament_are_finite_and_dimension_sensitive():
    sheet = sheet_cloud(20, 1.0)
    filament = filament_cloud(80, 1.0)
    bulk = uniform_lattice(12, 1.0)
    bulk_mask = interior_mask(12, 3)

    for k in K_VALUES:
        r_sheet = kth_neighbor_distance(sheet, k)
        r_fil = kth_neighbor_distance(filament, k)
        r_bulk = kth_neighbor_distance(bulk, k)[bulk_mask]
        assert np.all(np.isfinite(r_sheet)) and np.all(r_sheet > 0)
        assert np.all(np.isfinite(r_fil)) and np.all(r_fil > 0)
        # Lower-dimensional clouds need a larger radius to reach the same k
        # (fewer neighbours per unit volume at fixed spacing).
        assert r_fil.max() > r_bulk.mean()
        assert r_sheet[len(r_sheet) // 2] > 0.9 * r_bulk.mean()


def test_filament_exact_calibration():
    n, a, margin = 60, 1.0, 16
    pts = filament_cloud(n, a)
    interior = np.arange(margin, n - margin)
    expected = {8: 4.0 * a, 16: 8.0 * a, 32: 16.0 * a}
    for k, exp in expected.items():
        r = kth_neighbor_distance(pts, k)[interior]
        np.testing.assert_allclose(r, exp, atol=1e-9)


def test_morton_boundary_cross_cluster_neighbors_are_found():
    """Exact reference must not miss neighbours across a Morton-relevant
    coordinate discontinuity -- this is the ground truth Phase 2's
    Morton-window GPU estimator will be diffed against."""
    positions = morton_boundary_cloud()
    n_side = len(positions) // 2
    k = 8
    r_full = kth_neighbor_distance(positions, k)

    left, right = positions[:n_side], positions[n_side:]
    r_left_only = kth_neighbor_distance(left, k)
    r_right_only = kth_neighbor_distance(right, k)
    r_same_side_only = np.concatenate([r_left_only, r_right_only])

    # Cross-boundary neighbours can only shrink the true k-th distance.
    assert np.all(r_full <= r_same_side_only + 1e-12)
    assert np.any(r_full < r_same_side_only - 1e-9)


# ---- the deciding experiment: volume-preserving anisotropic shear --------


@pytest.mark.parametrize("k", K_VALUES)
def test_anisotropic_shear_kill_switch(k):
    """F = diag(lam, lam^-1/2, lam^-1/2), det F = 1: a Lagrangian element
    that stretches along x and compresses in y,z while conserving volume.

    Claim under test: r_k detects the anisotropic resolution loss (i.e.
    grows with lam, tracking the elongated spacing lam*a).

    Analytic prediction (verified independently of this code, see the
    conversation record): for fixed k, once lam is large enough that the
    k nearest neighbours all lie in-plane, r_k ~ lam^-1/2 -- it *shrinks*
    with lam and never reflects the lam*a elongation. V^(1/3) stays exactly
    a for all lam (volume-preserving), so max(V^(1/3), C_h r_k) reduces to
    the unchanged V^(1/3) floor: the adaptive term contributes nothing in
    exactly the regime it was proposed to fix.
    """
    n, a, margin = 12, 1.0, 3
    mask = interior_mask(n, margin)
    lambdas = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    # C_h calibrated on the undeformed (lambda=1) lattice so that
    # C_h * r_k == V^(1/3) == a exactly there (test_uniform_lattice_exact_
    # shell_calibration establishes these are exactly sqrt(2), sqrt(2), 2).
    c_h = {8: 1.0 / np.sqrt(2.0), 16: 1.0 / np.sqrt(2.0), 32: 0.5}[k]

    r_k_of_lambda = []
    for lam in lambdas:
        pts = sheared_lattice(n, a, lam)
        r = kth_neighbor_distance(pts, k)[mask]
        r_k_of_lambda.append(np.median(r))
    r_k_of_lambda = np.array(r_k_of_lambda)
    delta_adaptive = c_h * r_k_of_lambda

    elongated_spacing = lambdas * a  # the direction whose resolution is truly lost
    volume_cbrt = np.full_like(lambdas, a)  # exactly constant: det F = 1

    # Calibration sanity: at lambda=1, C_h*r_k must reproduce V^(1/3) exactly.
    assert delta_adaptive[0] == pytest.approx(volume_cbrt[0], rel=1e-9)

    # The scalar kNN estimator must NOT track the elongated spacing. It need
    # not be monotone at every step (a small uptick near lambda=2 is expected
    # before the large-lambda in-plane asymptote dominates), but it must
    # never rise meaningfully above the isotropic value, must be monotone
    # decreasing in the asymptotic regime, and must fall well below the
    # constant volume floor as lambda grows -- i.e. it can never win the
    # max() in Delta = max(V^(1/3), C_h*r_k) once lambda is large, the
    # opposite of "detecting" the defect it was proposed to catch.
    assert np.all(delta_adaptive < 1.15 * volume_cbrt), (
        f"C_h*r_k rose above the isotropic value: {delta_adaptive}"
    )
    assert np.all(np.diff(delta_adaptive[lambdas >= 4.0]) < 0), (
        f"C_h*r_k not monotone decreasing in the asymptotic regime: {delta_adaptive}"
    )
    assert delta_adaptive[-1] < 0.5 * volume_cbrt[-1], (
        "C_h*r_k should have collapsed well below the constant volume floor "
        f"by lambda={lambdas[-1]}, got {delta_adaptive[-1]:.4f} vs "
        f"V^(1/3)={volume_cbrt[-1]:.4f}"
    )
    # It is also, unsurprisingly, unrelated to the quantity that actually
    # grows: the elongated-direction spacing.
    correlation = np.corrcoef(r_k_of_lambda, elongated_spacing)[0, 1]
    assert correlation < 0, (
        f"expected r_k to be anti-correlated with the true elongated "
        f"spacing, got correlation={correlation:.3f}"
    )
