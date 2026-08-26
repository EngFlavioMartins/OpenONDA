"""Conservation contracts for GBD threshold pruning and regeneration."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.spatial

from source.solvers.vpm.physics.diffusion import _GridDiffusionMixin
from source.solvers.vpm.physics.diffusion import grid as grid_module


def _make_grid(seed: int = 0) -> tuple[np.ndarray, np.ndarray, float]:
    """Return an asymmetric strong core surrounded by a weak removable halo."""
    rng = np.random.default_rng(seed)
    shape = (24, 24, 24)
    grid_min = np.array([-0.3, -0.4, -0.2])
    particle_spacing = 0.05
    ii, jj, kk = np.indices(shape)
    grid = np.zeros((*shape, 3), dtype=np.float32)
    direction = np.array([0.3, -0.2, 1.0], dtype=np.float32)
    for cx, cy, cz, amplitude in (
        (8, 10, 12, 1.0),
        (15, 13, 11, 0.7),
        (11, 16, 14, 0.5),
    ):
        radius_squared = (ii - cx) ** 2 + (jj - cy) ** 2 + (kk - cz) ** 2
        grid += (amplitude * np.exp(-radius_squared / 6.0))[..., None] * direction
    grid += 0.01 * rng.standard_normal((*shape, 3)).astype(np.float32)
    return grid, grid_min, particle_spacing


def _moments(
    grid: np.ndarray,
    grid_min: np.ndarray,
    particle_spacing: float,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mask is None:
        mask = np.linalg.norm(grid, axis=-1) > 0.0
    ii, jj, kk = np.where(mask)
    vortex_strength = grid[ii, jj, kk].astype(np.float64)
    position = np.stack(
        [
            grid_min[0] + ii * particle_spacing,
            grid_min[1] + jj * particle_spacing,
            grid_min[2] + kk * particle_spacing,
        ],
        axis=1,
    )
    return (
        vortex_strength.sum(axis=0),
        np.cross(position, vortex_strength).sum(axis=0),
        np.cross(position, np.cross(position, vortex_strength)).sum(axis=0) / 3.0,
    )


def _retained_moments(
    corrected: np.ndarray,
    ix: np.ndarray,
    iy: np.ndarray,
    iz: np.ndarray,
    grid_min: np.ndarray,
    particle_spacing: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position = np.stack(
        [
            grid_min[0] + ix * particle_spacing,
            grid_min[1] + iy * particle_spacing,
            grid_min[2] + iz * particle_spacing,
        ],
        axis=1,
    )
    strength = corrected.astype(np.float64)
    return (
        strength.sum(axis=0),
        np.cross(position, strength).sum(axis=0),
        np.cross(position, np.cross(position, strength)).sum(axis=0) / 3.0,
    )


def _assert_moments_close(
    expected: tuple[np.ndarray, np.ndarray, np.ndarray],
    actual: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    for expected_value, actual_value in zip(expected, actual, strict=True):
        scale = np.linalg.norm(expected_value) + 1.0e-30
        assert np.linalg.norm(actual_value - expected_value) / scale < 1.0e-5


def test_prune_recovery_preserves_vortex_strength_and_impulse_moments(monkeypatch):
    monkeypatch.setattr(grid_module, "_GBD_MOMENT_CHUNK_SIZE", 37)
    grid, grid_min, particle_spacing = _make_grid()
    magnitude = np.linalg.norm(grid, axis=-1)
    threshold = 0.05 * float(magnitude.max())
    ix, iy, iz = np.where(magnitude >= threshold)
    assert 4 < len(ix) < int(np.count_nonzero(magnitude > 0.0))

    raw = grid[ix, iy, iz].astype(np.float64)
    expected = _moments(grid, grid_min, particle_spacing)
    assert np.linalg.norm(expected[0] - raw.sum(axis=0)) > 1.0e-3
    diagnostics: dict[str, bool | int | float] = {}
    corrected = _GridDiffusionMixin._redistribute_pruned_moments(
        grid,
        magnitude,
        ix,
        iy,
        iz,
        grid_min,
        particle_spacing,
        diagnostics=diagnostics,
    )

    assert corrected.dtype == grid.dtype
    assert np.isfinite(corrected).all()
    _assert_moments_close(
        expected,
        _retained_moments(corrected, ix, iy, iz, grid_min, particle_spacing),
    )
    assert 0.0 < diagnostics["correction_fraction"] < 0.5
    for name in (
        "normalized_vortex_strength_residual",
        "normalized_linear_impulse_residual",
        "normalized_angular_impulse_residual",
    ):
        assert 0.0 <= float(diagnostics[name]) < 1.0e-5


def test_no_prune_recovery_is_an_exact_noop():
    grid, grid_min, particle_spacing = _make_grid(seed=1)
    magnitude = np.linalg.norm(grid, axis=-1)
    ix, iy, iz = np.where(magnitude > 0.0)

    corrected = _GridDiffusionMixin._redistribute_pruned_moments(
        grid,
        magnitude,
        ix,
        iy,
        iz,
        grid_min,
        particle_spacing,
    )

    np.testing.assert_array_equal(corrected, grid[ix, iy, iz])


def test_too_few_survivors_fail_instead_of_silently_discarding_moments():
    grid, grid_min, particle_spacing = _make_grid(seed=2)
    magnitude = np.linalg.norm(grid, axis=-1)
    strongest = np.argsort(-magnitude.ravel(), kind="stable")[:2]
    ix, iy, iz = np.unravel_index(strongest, magnitude.shape)

    with pytest.raises(RuntimeError, match="at least four retained nodes"):
        _GridDiffusionMixin._redistribute_pruned_moments(
            grid,
            magnitude,
            ix,
            iy,
            iz,
            grid_min,
            particle_spacing,
        )


def test_one_threshold_survivor_is_minimally_augmented_to_full_rank():
    shape = (5, 5, 5)
    grid_min = np.full(3, -0.2)
    particle_spacing = 0.1
    grid = np.zeros((*shape, 3), dtype=np.float32)
    direction = np.array([0.8, -0.6, 0.5], dtype=np.float32)
    grid[2, 2, 2] = direction
    for index in (
        (1, 2, 2),
        (3, 2, 2),
        (2, 1, 2),
        (2, 3, 2),
        (2, 2, 1),
        (2, 2, 3),
    ):
        grid[index] = 0.004 * direction
    for ii in (1, 3):
        for jj in (1, 3):
            for kk in (1, 3):
                grid[ii, jj, kk] = 0.001 * direction
    magnitude = np.linalg.norm(grid, axis=-1)
    ix, iy, iz = np.where(magnitude >= 0.01 * float(magnitude.max()))
    assert len(ix) == 1

    augmented_ix, augmented_iy, augmented_iz, added, preserve_groups = (
        _GridDiffusionMixin._augment_moment_recovery_support(
            grid,
            magnitude,
            ix,
            iy,
            iz,
            grid_min,
            particle_spacing,
            cap=4,
            labels=np.zeros(shape, dtype=np.int32),
        )
    )

    assert added == 3
    assert len(augmented_ix) == 4
    assert not preserve_groups
    np.testing.assert_array_equal(
        np.stack((augmented_ix, augmented_iy, augmented_iz), axis=1)[0],
        [2, 2, 2],
    )
    diagnostics: dict[str, bool | int | float] = {}
    corrected = _GridDiffusionMixin._redistribute_pruned_moments(
        grid,
        magnitude,
        augmented_ix,
        augmented_iy,
        augmented_iz,
        grid_min,
        particle_spacing,
        diagnostics=diagnostics,
    )
    for expected, actual in zip(
        _moments(grid, grid_min, particle_spacing),
        _retained_moments(
            corrected,
            augmented_ix,
            augmented_iy,
            augmented_iz,
            grid_min,
            particle_spacing,
        ),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=1.0e-7)
    assert diagnostics["normalized_vortex_strength_residual"] < 1.0e-5
    assert diagnostics["normalized_linear_impulse_residual"] < 1.0e-5
    assert diagnostics["normalized_angular_impulse_residual"] < 1.0e-5

    with pytest.raises(RuntimeError, match="within the regeneration cap"):
        _GridDiffusionMixin._augment_moment_recovery_support(
            grid,
            magnitude,
            ix,
            iy,
            iz,
            grid_min,
            particle_spacing,
            cap=3,
            labels=np.zeros(shape, dtype=np.int32),
        )


def test_no_moment_basis_is_required_when_every_nonzero_node_fits_under_cap():
    shape = (4, 4, 4)
    grid = np.zeros((*shape, 3), dtype=np.float32)
    grid[1, 1, 1] = (1.0, -0.5, 0.25)
    grid[2, 1, 1] = (0.1, -0.05, 0.025)
    grid[1, 2, 1] = (0.05, -0.025, 0.0125)
    magnitude = np.linalg.norm(grid, axis=-1)
    ix, iy, iz = np.where(magnitude >= 0.5 * float(magnitude.max()))
    assert len(ix) == 1

    augmented_ix, augmented_iy, augmented_iz, introduced, preserve_groups = (
        _GridDiffusionMixin._augment_moment_recovery_support(
            grid,
            magnitude,
            ix,
            iy,
            iz,
            np.zeros(3),
            particle_spacing=0.1,
            cap=3,
        )
    )

    assert len(augmented_ix) == 3
    assert introduced == 2
    assert not preserve_groups
    assert set(
        zip(
            augmented_ix.tolist(),
            augmented_iy.tolist(),
            augmented_iz.tolist(),
            strict=True,
        )
    ) == {(1, 1, 1), (2, 1, 1), (1, 2, 1)}


def test_cap_full_rank_deficient_support_swaps_weakest_without_exceeding_cap():
    shape = (6, 6, 6)
    grid_min = np.zeros(3)
    particle_spacing = 0.1
    grid = np.zeros((*shape, 3), dtype=np.float32)
    direction = np.array([0.8, -0.6, 0.5], dtype=np.float32)
    retained_nodes = ((1, 2, 2), (2, 2, 2), (3, 2, 2), (4, 2, 2))
    for index, amplitude in zip(
        retained_nodes,
        (1.0, 0.9, 0.8, 0.7),
        strict=True,
    ):
        grid[index] = amplitude * direction
    for index, amplitude in zip(
        ((2, 3, 2), (2, 2, 3), (2, 3, 3)),
        (0.6, 0.5, 0.4),
        strict=True,
    ):
        grid[index] = amplitude * direction
    magnitude = np.linalg.norm(grid, axis=-1)
    ix, iy, iz = np.where(magnitude >= 0.7 * float(np.linalg.norm(direction)))
    assert len(ix) == 4

    augmented_ix, augmented_iy, augmented_iz, introduced, preserve_groups = (
        _GridDiffusionMixin._augment_moment_recovery_support(
            grid,
            magnitude,
            ix,
            iy,
            iz,
            grid_min,
            particle_spacing,
            cap=4,
        )
    )

    final_nodes = set(
        zip(
            augmented_ix.tolist(),
            augmented_iy.tolist(),
            augmented_iz.tolist(),
            strict=True,
        )
    )
    assert len(final_nodes) == 4
    assert introduced == 1
    assert not preserve_groups
    assert retained_nodes[0] in final_nodes
    assert retained_nodes[-1] not in final_nodes
    assert (2, 3, 2) in final_nodes
    final_position = np.stack((augmented_ix, augmented_iy, augmented_iz), axis=1) * particle_spacing
    rank, condition = grid_module._gbd_moment_support_quality(
        final_position,
        grid[augmented_ix, augmented_iy, augmented_iz],
        particle_spacing,
    )
    assert rank == 9
    assert condition <= grid_module._GBD_MOMENT_CONDITION_LIMIT


def test_cap_full_support_exchange_stays_within_each_vortex_group():
    shape = (10, 6, 6)
    grid = np.zeros((*shape, 3), dtype=np.float32)
    direction = np.array([0.7, -0.4, 0.9], dtype=np.float32)
    left_nodes = ((1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2))
    right_nodes = ((5, 2, 2), (6, 2, 2), (7, 2, 2), (8, 2, 2))
    for index, amplitude in zip(
        left_nodes,
        (1.3, 1.2, 1.1, 1.0),
        strict=True,
    ):
        grid[index] = amplitude * direction
    for index, amplitude in zip(
        right_nodes,
        (0.9, 0.8, 0.7, 0.6),
        strict=True,
    ):
        grid[index] = amplitude * direction
    right_candidate = (6, 3, 2)
    grid[right_candidate] = 0.5 * direction
    magnitude = np.linalg.norm(grid, axis=-1)
    ix, iy, iz = np.where(magnitude >= 0.55 * float(np.linalg.norm(direction)))
    labels = np.where(np.indices(shape)[0] < 5, 0, 1).astype(np.int32)
    assert np.count_nonzero(labels[ix, iy, iz] == 0) == 4
    assert np.count_nonzero(labels[ix, iy, iz] == 1) == 4

    augmented_ix, augmented_iy, augmented_iz, introduced, preserve_groups = (
        _GridDiffusionMixin._augment_moment_recovery_support(
            grid,
            magnitude,
            ix,
            iy,
            iz,
            np.zeros(3),
            particle_spacing=0.1,
            cap=8,
            labels=labels,
        )
    )

    final_nodes = set(
        zip(
            augmented_ix.tolist(),
            augmented_iy.tolist(),
            augmented_iz.tolist(),
            strict=True,
        )
    )
    assert len(final_nodes) == 8
    assert introduced == 1
    assert preserve_groups
    assert set(left_nodes).issubset(final_nodes)
    assert right_candidate in final_nodes
    assert right_nodes[-1] not in final_nodes
    assert np.count_nonzero(labels[augmented_ix, augmented_iy, augmented_iz] == 0) == 4
    assert np.count_nonzero(labels[augmented_ix, augmented_iy, augmented_iz] == 1) == 4


def test_collinear_survivors_fail_with_an_explicit_rank_error():
    shape = (12, 1, 1)
    grid = np.zeros((*shape, 3), dtype=np.float32)
    coordinate = np.arange(shape[0], dtype=np.float32)
    grid[:, 0, 0, 1] = 0.2 + 0.03 * coordinate
    grid[:, 0, 0, 2] = 0.1 + 0.02 * coordinate**2
    magnitude = np.linalg.norm(grid, axis=-1)
    ix = np.arange(3, 9, dtype=np.int64)
    iy = np.zeros_like(ix)
    iz = np.zeros_like(ix)

    with pytest.raises(RuntimeError, match="rank-deficient or ill-conditioned"):
        _GridDiffusionMixin._redistribute_pruned_moments(
            grid,
            magnitude,
            ix,
            iy,
            iz,
            np.zeros(3),
            0.1,
        )


def test_nondegenerate_coplanar_survivors_recover_all_moments():
    shape = (18, 16, 1)
    grid_min = np.array([-0.45, -0.4, 0.2])
    particle_spacing = 0.05
    ii, jj, _kk = np.indices(shape)
    first = np.exp(-((ii - 6) ** 2 + (jj - 7) ** 2) / 12.0)
    second = np.exp(-((ii - 12) ** 2 + (jj - 10) ** 2) / 9.0)
    grid = np.zeros((*shape, 3), dtype=np.float32)
    grid[..., 0] = 0.25 * second
    grid[..., 1] = -0.15 * first
    grid[..., 2] = first + 0.6 * second
    magnitude = np.linalg.norm(grid, axis=-1)
    ix, iy, iz = np.where(magnitude >= 0.04 * float(magnitude.max()))

    corrected = _GridDiffusionMixin._redistribute_pruned_moments(
        grid,
        magnitude,
        ix,
        iy,
        iz,
        grid_min,
        particle_spacing,
    )

    assert np.isfinite(corrected).all()
    _assert_moments_close(
        _moments(grid, grid_min, particle_spacing),
        _retained_moments(corrected, ix, iy, iz, grid_min, particle_spacing),
    )


def test_nearly_coincident_geometry_fails_as_ill_conditioned():
    grid, _grid_min, particle_spacing = _make_grid(seed=6)
    magnitude = np.linalg.norm(grid, axis=-1)
    ix, iy, iz = np.where(magnitude >= 0.05 * float(magnitude.max()))
    translated_grid_min = np.array([1.0e10, -2.0e10, 3.0e10])

    with pytest.raises(RuntimeError, match="rank-deficient or ill-conditioned"):
        _GridDiffusionMixin._redistribute_pruned_moments(
            grid,
            magnitude,
            ix,
            iy,
            iz,
            translated_grid_min,
            particle_spacing,
        )


def test_excessive_recovery_fraction_fails_before_returning_a_cloud():
    grid, grid_min, particle_spacing = _make_grid(seed=5)
    magnitude = np.linalg.norm(grid, axis=-1)
    ix, iy, iz = np.where(magnitude >= 0.01 * float(magnitude.max()))
    ix, iy, iz, _, _ = _GridDiffusionMixin._cap_surviving_nodes(
        magnitude,
        ix,
        iy,
        iz,
        30,
    )

    with pytest.raises(RuntimeError, match="excessive strength correction"):
        _GridDiffusionMixin._redistribute_pruned_moments(
            grid,
            magnitude,
            ix,
            iy,
            iz,
            grid_min,
            particle_spacing,
        )


def test_post_cast_residual_gate_rejects_a_failed_storage_precision_closure(monkeypatch):
    monkeypatch.setattr(grid_module, "_GBD_MOMENT_RESIDUAL_LIMIT", 1.0e-12)
    grid, grid_min, particle_spacing = _make_grid(seed=0)
    magnitude = np.linalg.norm(grid, axis=-1)
    ix, iy, iz = np.where(magnitude >= 0.05 * float(magnitude.max()))

    with pytest.raises(RuntimeError, match="post-cast closure"):
        _GridDiffusionMixin._redistribute_pruned_moments(
            grid,
            magnitude,
            ix,
            iy,
            iz,
            grid_min,
            particle_spacing,
        )


def test_nearest_node_query_contains_only_discarded_nodes(monkeypatch):
    grid, grid_min, particle_spacing = _make_grid(seed=7)
    magnitude = np.linalg.norm(grid, axis=-1)
    ix, iy, iz = np.where(magnitude >= 0.05 * float(magnitude.max()))
    expected_discarded = int(np.count_nonzero(magnitude > 0.0) - len(ix))
    queried_counts: list[int] = []
    real_tree = scipy.spatial.cKDTree

    class RecordingTree:
        def __init__(self, *args, **kwargs):
            self._tree = real_tree(*args, **kwargs)

        def query(self, values, *args, **kwargs):
            queried_counts.append(len(values))
            return self._tree.query(values, *args, **kwargs)

    monkeypatch.setattr(scipy.spatial, "cKDTree", RecordingTree)

    _GridDiffusionMixin._redistribute_pruned_moments(
        grid,
        magnitude,
        ix,
        iy,
        iz,
        grid_min,
        particle_spacing,
    )

    assert queried_counts == [expected_discarded]


def test_threshold_then_population_cap_still_preserves_all_moments():
    grid, grid_min, particle_spacing = _make_grid(seed=3)
    magnitude = np.linalg.norm(grid, axis=-1)
    threshold = 0.02 * float(magnitude.max())
    ix, iy, iz = np.where(magnitude >= threshold)
    cap = 320
    ix, iy, iz, _, candidate_count = _GridDiffusionMixin._cap_surviving_nodes(
        magnitude,
        ix,
        iy,
        iz,
        cap,
    )
    assert candidate_count > cap
    assert len(ix) == cap

    corrected = _GridDiffusionMixin._redistribute_pruned_moments(
        grid,
        magnitude,
        ix,
        iy,
        iz,
        grid_min,
        particle_spacing,
    )

    assert len(corrected) == cap
    assert np.isfinite(corrected).all()
    _assert_moments_close(
        _moments(grid, grid_min, particle_spacing),
        _retained_moments(corrected, ix, iy, iz, grid_min, particle_spacing),
    )


def test_prune_recovery_preserves_each_vortex_group_when_both_survive():
    shape = (28, 20, 12)
    particle_spacing = 0.05
    grid_min = np.array([-0.7, -0.5, -0.3])
    ii, jj, kk = np.indices(shape)
    labels = np.where(ii < 14, 0, 1).astype(np.int32)
    left = np.exp(-((ii - 8) ** 2 + (jj - 10) ** 2 + (kk - 6) ** 2) / 18.0)
    right = np.exp(-((ii - 20) ** 2 + (jj - 10) ** 2 + (kk - 6) ** 2) / 18.0)
    grid = np.zeros((*shape, 3), dtype=np.float32)
    grid[..., 2] = np.where(labels == 0, left, -right)
    magnitude = np.linalg.norm(grid, axis=-1)
    ix, iy, iz = np.where(magnitude >= 0.08 * float(magnitude.max()))

    corrected = _GridDiffusionMixin._redistribute_pruned_moments(
        grid,
        magnitude,
        ix,
        iy,
        iz,
        grid_min,
        particle_spacing,
        labels=labels,
    )
    survivor_labels = labels[ix, iy, iz]
    for label in (0, 1):
        selected = survivor_labels == label
        _assert_moments_close(
            _moments(grid, grid_min, particle_spacing, mask=labels == label),
            _retained_moments(
                corrected[selected],
                ix[selected],
                iy[selected],
                iz[selected],
                grid_min,
                particle_spacing,
            ),
        )


def test_sparse_group_falls_back_to_global_moment_recovery():
    grid, grid_min, particle_spacing = _make_grid(seed=4)
    magnitude = np.linalg.norm(grid, axis=-1)
    threshold = 0.05 * float(magnitude.max())
    ix, iy, iz = np.where(magnitude >= threshold)
    labels = np.zeros(magnitude.shape, dtype=np.int32)
    labels[ix[0], iy[0], iz[0]] = 1
    pruned = np.argwhere((magnitude > 0.0) & (magnitude < threshold))
    labels[tuple(pruned[0])] = 1

    corrected = _GridDiffusionMixin._redistribute_pruned_moments(
        grid,
        magnitude,
        ix,
        iy,
        iz,
        grid_min,
        particle_spacing,
        labels=labels,
    )

    _assert_moments_close(
        _moments(grid, grid_min, particle_spacing),
        _retained_moments(corrected, ix, iy, iz, grid_min, particle_spacing),
    )


def test_production_gbd_writes_recovery_before_building_particle_arrays(monkeypatch):
    events: list[str] = []
    grid = np.zeros((5, 5, 5, 3), dtype=np.float32)
    for index in ((1, 1, 1), (1, 1, 2), (1, 2, 1), (2, 1, 1), (2, 2, 2)):
        grid[index] = [1.0, 0.2, -0.1]
    grid[3, 3, 3] = [0.1, 0.0, 0.0]
    sentinel = np.array([0.125, -0.25, 0.5], dtype=np.float32)

    class Particles:
        n_particles_total = 1
        capacity = 100
        position = np.zeros((1, 3), dtype=np.float32)
        vortex_strength = np.ones((1, 3), dtype=np.float32)

        @staticmethod
        def zone_id_cpu():
            return np.zeros(1, dtype=np.int32)

        @staticmethod
        def group_id_cpu():
            return np.zeros(1, dtype=np.int32)

        @staticmethod
        def position_cpu():
            return np.zeros((1, 3), dtype=np.float32)

        @staticmethod
        def vortex_strength_cpu():
            return np.ones((1, 3), dtype=np.float32)

        @staticmethod
        def eddy_viscosity_cpu():
            return np.zeros(1, dtype=np.float32)

    class Harness(_GridDiffusionMixin):
        def __init__(self):
            self._fixed_grid_min = None
            self._max_grid_dims = None
            self._grid_a = object()
            self._grid_b = object()
            self._ping = True
            self.core_radius_ratio = 1.0

        @staticmethod
        def _compute_grid_bounds(*_args, **_kwargs):
            return np.zeros(3), grid.shape[:3]

        @staticmethod
        def _ensure_grid_capacity(nx, ny, nz):
            return nx, ny, nz

        @staticmethod
        def _zero_grid_kernel(*_args, **_kwargs):
            return None

        @staticmethod
        def _m4_scatter_gpu_kernel(*_args, **_kwargs):
            return None

        @staticmethod
        def _prepare_body_mask_current_grid(*_args, **_kwargs):
            return None

        @staticmethod
        def _apply_body_mask_current_grid(*_args, **_kwargs):
            return None

        @staticmethod
        def _advance_gbd_laplacian(**_kwargs):
            return 1, 0.0

        @staticmethod
        def _scatter_zone_ids(*_args, **_kwargs):
            return np.zeros(grid.shape[:3], dtype=np.int32)

        @staticmethod
        def _scatter_scalar_weighted(*_args, **_kwargs):
            return np.zeros(grid.shape[:3], dtype=np.float32)

        @staticmethod
        def _download_active_vec_grid(*_args, **_kwargs):
            return grid.copy()

        @staticmethod
        def _scatter_id_field(*_args, **_kwargs):
            return np.zeros(grid.shape[:3], dtype=np.int32)

        @staticmethod
        def _redistribute_pruned_moments(
            grid_np,
            _vortex_strength_magnitude,
            ix,
            iy,
            iz,
            _grid_min_np,
            _particle_spacing,
            labels=None,
            diagnostics=None,
        ):
            del labels
            events.append("recover")
            if diagnostics is not None:
                diagnostics.update(
                    correction_fraction=0.25,
                    normalized_vortex_strength_residual=1.0e-8,
                    normalized_linear_impulse_residual=2.0e-8,
                    normalized_angular_impulse_residual=3.0e-8,
                )
            return np.broadcast_to(sentinel, (len(ix), 3)).copy().astype(grid_np.dtype)

        @staticmethod
        def _build_diffusion_particle_arrays(
            ix,
            iy,
            iz,
            grid_np,
            *_args,
            **_kwargs,
        ):
            events.append("build")
            actual = grid_np[ix, iy, iz]
            np.testing.assert_array_equal(
                actual,
                np.broadcast_to(sentinel, actual.shape),
            )
            return {"vortex_strength": actual.copy()}

    monkeypatch.setattr(grid_module.ti, "sync", lambda: None)
    harness = Harness()
    result = harness._gbd_diffusion_impl(
        Particles(),
        time_step_size=0.01,
        particle_spacing=0.1,
        kinematic_viscosity=1.0e-3,
        regen_threshold=0.5,
        regen_threshold_mode="absolute",
    )

    assert events == ["recover", "build"]
    assert result is not None
    np.testing.assert_array_equal(
        result["vortex_strength"],
        np.broadcast_to(sentinel, (5, 3)),
    )
    diagnostic = harness.last_gbd_moment_recovery
    assert diagnostic == {
        "applied": True,
        "nonzero_node_count": 6,
        "retained_node_count": 5,
        "pruned_node_count": 1,
        "support_augmented_node_count": 0,
        "correction_fraction": 0.25,
        "normalized_vortex_strength_residual": 1.0e-8,
        "normalized_linear_impulse_residual": 2.0e-8,
        "normalized_angular_impulse_residual": 3.0e-8,
    }
    diagnostic["applied"] = False
    assert harness.last_gbd_moment_recovery["applied"] is True

    empty_particles = Particles()
    empty_particles.n_particles_total = 0
    assert (
        harness._gbd_diffusion_impl(
            empty_particles,
            time_step_size=0.01,
            particle_spacing=0.1,
            kinematic_viscosity=1.0e-3,
        )
        is None
    )
    assert harness.last_gbd_moment_recovery == harness._empty_gbd_moment_recovery()
