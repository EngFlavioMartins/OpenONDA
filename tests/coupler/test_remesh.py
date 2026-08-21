import numpy as np

from source.coupler.remesh import m4p, remesh_to_grid


def test_m4p_interpolates_and_partitions_unity():
    np.testing.assert_allclose(m4p(np.array([0.0, 1.0, 2.0])), [1.0, 0.0, 0.0])
    for offset in np.linspace(0.0, 1.0, 9):
        nodes = np.arange(-2, 3)
        weights = m4p(offset - nodes)
        assert np.isclose(weights.sum(), 1.0)
        assert np.isclose(np.dot(offset - nodes, weights), 0.0)


def test_remesh_conserves_strength_and_linear_impulse():
    rng = np.random.default_rng(8)
    h = 0.1
    origin = np.array([-1.0, -1.0, -1.0])
    positions = rng.uniform(-0.5, 0.5, (100, 3))
    vortex_strength = rng.normal(size=(100, 3)) * 1.0e-3
    grid_positions, grid_strength = remesh_to_grid(
        positions, vortex_strength, origin, h, (21, 21, 21)
    )

    np.testing.assert_allclose(grid_strength.sum(0), vortex_strength.sum(0), atol=1.0e-14)
    before = 0.5 * np.cross(positions, vortex_strength).sum(0)
    after = 0.5 * np.cross(grid_positions, grid_strength).sum(0)
    np.testing.assert_allclose(after, before, atol=1.0e-14)


def test_aligned_particles_are_deposited_directly():
    origin = np.array([-0.2, -0.2, -0.2])
    positions = np.array([[0.0, 0.1, 0.2], [0.1, 0.0, -0.1]])
    vortex_strength = np.array([[1.0, 0.0, 0.2], [-0.1, 0.4, 0.0]])
    _, grid = remesh_to_grid(positions, vortex_strength, origin, 0.1, (8, 8, 8))
    assert np.count_nonzero(np.linalg.norm(grid, axis=1)) == 2
    np.testing.assert_allclose(grid.sum(0), vortex_strength.sum(0))


def test_empty_remesh():
    positions, vortex_strength = remesh_to_grid(
        np.empty((0, 3)), np.empty((0, 3)), np.zeros(3), 0.1, (4, 4, 4)
    )
    assert positions.shape == (64, 3)
    assert not vortex_strength.any()
