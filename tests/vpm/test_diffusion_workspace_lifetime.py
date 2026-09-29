"""Diffusion storage may grow without dropping support or retaining old grids."""

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.physics.diffusion.grid import _GridDiffusionMixin


@ti.data_oriented
class Grid(_GridDiffusionMixin):
    pass


@pytest.fixture
def grid():
    ti.reset()
    ti.init(arch=ti.cpu, cpu_max_num_threads=1, offline_cache=False)
    physics = Grid()
    physics._init_grid_diffusion()
    yield physics
    ti.reset()


def test_repeated_growth_releases_old_grids_without_clipping(grid):
    sentinel = ti.field(dtype=ti.f32, shape=1)
    sentinel[0] = 7.0
    old_tree = None
    for _ in range(8):
        nx = 5 if grid._grid_shape is None else grid._grid_shape[0] + 1
        assert grid._ensure_grid_capacity(nx, 5, 5) == (nx, 5, 5)
        if old_tree is not None:
            assert old_tree.destroyed
        assert grid._grid_shape[0] >= nx
        grid._grid_a.fill(3.0)
        grid._zero_grid_kernel(grid._grid_a, nx, 5, 5)
        np.testing.assert_array_equal(grid._grid_a.to_numpy()[:nx, :5, :5], 0)
        assert sentinel[0] == 7.0
        old_tree = grid._grid_tree


def test_domain_allocation_has_no_arbitrary_byte_budget(grid, monkeypatch):
    shapes = []
    monkeypatch.setattr(grid, "_allocate_grid", shapes.append)
    grid.require_fixed_grid_allocation()
    grid.configure_max_grid_extent([0, 400, 0, 300, 0, 300], 1.0, padding=3)
    assert shapes == [(407, 307, 307)]
    assert np.prod(shapes[0]) * 36 > 1024**3


def test_grid_bounds_keep_all_source_support(grid):
    positions = np.array([[0.0, 0.0, 0.0], [2500.0, 0.0, 0.0]])
    origin, shape = grid._compute_grid_bounds(positions, 1.0, 3.0)
    assert shape[0] > 2500
    assert origin[0] <= -3
    assert origin[0] + shape[0] - 1 >= 2503
    with pytest.raises(ValueError, match="finite particle"):
        grid._compute_grid_bounds(np.array([[np.nan, 0, 0]]), 1.0, 3.0)


def test_fixed_domain_rejects_excess_support_instead_of_clipping(grid):
    grid.configure_max_grid_extent([-1, 1] * 3, 1, padding=3)
    shape = grid._grid_shape
    tree = grid._grid_tree
    with pytest.raises(ValueError, match="exceeds configured domain"):
        grid._ensure_grid_capacity(shape[0] + 1, 5, 5)
    assert grid._grid_tree is tree and not tree.destroyed
