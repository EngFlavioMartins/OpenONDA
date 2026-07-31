import numpy as np

from source.coupler.remesh import m4p, remesh_to_grid


def test_m4p_interpolates_and_partitions_unity():
    np.testing.assert_allclose(m4p(np.array([0.0, 1.0, 2.0])), [1.0, 0.0, 0.0])
    for offset in np.linspace(0.0, 1.0, 9):
        nodes = np.arange(-2, 3)
        weights = m4p(offset - nodes)
        assert np.isclose(weights.sum(), 1.0)
        assert np.isclose(np.dot(offset - nodes, weights), 0.0)


def test_remesh_conserves_circulation_and_linear_impulse():
    rng = np.random.default_rng(8)
    h = 0.1
    origin = np.array([-1.0, -1.0, -1.0])
    positions = rng.uniform(-0.5, 0.5, (100, 3))
    circulation = rng.normal(size=(100, 3)) * 1.0e-3
    grid_positions, grid_circulation = remesh_to_grid(
        positions, circulation, origin, h, (21, 21, 21)
    )

    np.testing.assert_allclose(grid_circulation.sum(0), circulation.sum(0), atol=1.0e-14)
    before = 0.5 * np.cross(positions, circulation).sum(0)
    after = 0.5 * np.cross(grid_positions, grid_circulation).sum(0)
    np.testing.assert_allclose(after, before, atol=1.0e-14)


def test_aligned_particles_are_deposited_directly():
    origin = np.array([-0.2, -0.2, -0.2])
    positions = np.array([[0.0, 0.1, 0.2], [0.1, 0.0, -0.1]])
    circulation = np.array([[1.0, 0.0, 0.2], [-0.1, 0.4, 0.0]])
    _, grid = remesh_to_grid(positions, circulation, origin, 0.1, (8, 8, 8))
    assert np.count_nonzero(np.linalg.norm(grid, axis=1)) == 2
    np.testing.assert_allclose(grid.sum(0), circulation.sum(0))


def test_empty_remesh():
    positions, circulation = remesh_to_grid(
        np.empty((0, 3)), np.empty((0, 3)), np.zeros(3), 0.1, (4, 4, 4)
    )
    assert positions.shape == (64, 3)
    assert not circulation.any()
