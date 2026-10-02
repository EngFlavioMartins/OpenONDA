"""Pure NumPy qualification; no FMM or Taichi runtime is imported."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest
import scipy.spatial


@pytest.fixture
def algebra(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    path = (
        root
        / "tests/support/cylinder/verify_gbd_moment_gram.py"
    )
    spec = spec_from_file_location("gbd_gram_algebra_asset", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    reference = module.load_reference(root / "source/solvers/vpm/physics/diffusion/grid.py")
    original_tree = scipy.spatial.cKDTree

    class SingleThreadTree:
        def __init__(self, *args, **kwargs):
            self.tree = original_tree(*args, **kwargs)

        def query(self, *args, **kwargs):
            kwargs["workers"] = 1
            return self.tree.query(*args, **kwargs)

    monkeypatch.setattr(scipy.spatial, "cKDTree", SingleThreadTree)
    return module, reference


@pytest.mark.parametrize(
    "geometry", ["random", "plane", "line", "point", "translated", "small", "large", "cancelled"]
)
@pytest.mark.parametrize("chunk", [37, 65536])
def test_closed_gram_matches_actual_nine_constraint_tensor(algebra, geometry, chunk):
    module, reference = algebra
    rng = np.random.default_rng(739)
    position = rng.normal(size=(413, 3))
    strength = rng.normal(size=(413, 3)) * np.exp(rng.uniform(-12, 12, size=(413, 1)))
    spacing = 0.04
    if geometry == "plane":
        position[:, 2] = 0
    elif geometry == "line":
        position[:, 1:] = 0
    elif geometry == "point":
        position[:] = (1, 2, 3)
    elif geometry == "translated":
        position += (1e6, -2e6, 3e6)
    elif geometry in ("small", "large"):
        scale = 1e-6 if geometry == "small" else 1e6
        position *= scale
        spacing *= scale
    elif geometry == "cancelled":
        strength[1::2] = -strength[:-1:2]
        strength[-1] = 0
    reference["_GBD_MOMENT_CHUNK_SIZE"] = chunk
    old = reference["_gbd_moment_gram"](position, strength, spacing)
    new = module.closed_moment_gram(position, strength, spacing, chunk_size=chunk)
    np.testing.assert_allclose(new[0], old[0], rtol=2e-12, atol=2e-14)
    np.testing.assert_array_equal(new[2], old[2])
    assert new[1] == old[1]
    old_rank, old_condition = reference["_gbd_moment_gram_quality"](old[0])
    new_rank, new_condition = reference["_gbd_moment_gram_quality"](new[0])
    assert new_rank == old_rank
    assert (old_condition <= reference["_GBD_MOMENT_CONDITION_LIMIT"]) == (
        new_condition <= reference["_GBD_MOMENT_CONDITION_LIMIT"]
    )


@pytest.mark.parametrize("strength_value", [0.0, np.nan, np.inf])
def test_invalid_weight_rejection_is_unchanged(algebra, strength_value):
    module, reference = algebra
    for function in (reference["_gbd_moment_gram"], module.closed_moment_gram):
        with pytest.raises(RuntimeError, match="no finite retained strength"):
            function(np.ones((8, 3)), np.full((8, 3), strength_value), 0.04)


@pytest.mark.parametrize(
    "kind", ["random", "cancelled", "directional", "grouped", "small", "translated"]
)
def test_complete_recovery_and_postcast_closure_are_equivalent(algebra, kind):
    module, reference = algebra
    rng = np.random.default_rng(733)
    shape = (13, 12, 11)
    i, j, k = np.indices(shape)
    profile = np.exp(-((i - 5) ** 2 + (j - 6) ** 2 + (k - 5) ** 2) / 7)
    grid = profile[..., None] * np.array([0.3, -0.2, 1.0])
    if kind != "directional":
        grid += 1e-3 * rng.normal(size=grid.shape)
    if kind == "cancelled":
        grid *= np.where(i[..., None] < 6, 1.0, -1.0)
    grid = grid.astype(np.float32)
    spacing = 0.05 * (1e-5 if kind == "small" else 1.0)
    origin = np.array([-0.3, -0.4, -0.2]) * (1e-5 if kind == "small" else 1.0)
    if kind == "translated":
        origin += (3.0, -2.0, 1.0)
    magnitude = np.linalg.norm(grid, axis=-1)
    selected = np.where(magnitude >= 0.025 * magnitude.max())
    labels = np.asarray(i >= 6, dtype=np.int32) if kind == "grouped" else None

    def run():
        augmented = reference["_augment_moment_recovery_support"](
            grid,
            magnitude,
            *selected,
            origin,
            spacing,
            grid.size // 3,
            labels=labels,
            strict_labels=labels is not None,
        )
        diagnostics = {}
        result = reference["_redistribute_pruned_moments"](
            grid,
            magnitude,
            *augmented[:3],
            origin,
            spacing,
            labels=labels,
            strict_labels=labels is not None,
            diagnostics=diagnostics,
        )
        return augmented, result, diagnostics

    old = run()
    reference["_gbd_moment_gram"] = module.closed_moment_gram
    new = run()
    for old_indices, new_indices in zip(old[0][:3], new[0][:3], strict=True):
        np.testing.assert_array_equal(old_indices, new_indices)
    assert old[0][3:] == new[0][3:]
    # Storage rounding is the acceptance boundary. Permit four f32 ulps of the
    # actual strongest value, not a looser physical moment-closure threshold.
    np.testing.assert_allclose(new[1], old[1], rtol=4 * np.finfo(np.float32).eps, atol=1e-12)
    for key in (
        "normalized_vortex_strength_residual",
        "normalized_linear_impulse_residual",
        "normalized_angular_impulse_residual",
    ):
        assert new[2][key] <= reference["_GBD_MOMENT_RESIDUAL_LIMIT"]
        assert new[2][key] <= old[2][key] + 4 * np.finfo(np.float32).eps


def test_rank_deficient_recovery_rejection_is_unchanged(algebra):
    module, reference = algebra
    grid = np.zeros((9, 3, 3, 3), dtype=np.float32)
    rng = np.random.default_rng(137)
    grid[:, 1, 1] = rng.normal(size=(9, 3))
    grid[4, 1, 1] *= 0.001
    magnitude = np.linalg.norm(grid, axis=-1)
    selected = np.where(magnitude > 0.01)
    for function in (reference["_gbd_moment_gram"], module.closed_moment_gram):
        reference["_gbd_moment_gram"] = function
        with pytest.raises(RuntimeError, match="rank-deficient|ill-conditioned"):
            reference["_redistribute_pruned_moments"](grid, magnitude, *selected, np.zeros(3), 0.04)
