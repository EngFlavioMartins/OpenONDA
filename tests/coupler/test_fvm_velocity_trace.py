from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from source.coupler.interpolation import FVMVelocityInterpolator


def _affine_velocity(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gradient = np.array(
        [[0.2, -0.1, 0.3], [0.4, 0.1, -0.2], [-0.3, 0.2, 0.15]],
        dtype=np.float64,
    )
    velocity = np.array([0.8, -0.2, 0.1]) + points @ gradient
    return velocity, np.broadcast_to(gradient, (len(points), 3, 3)).copy()


def test_weighted_trace_is_affine_exact_and_reuses_stencil():
    axis = np.linspace(-1.0, 1.0, 7)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    centres = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    velocity, gradient = _affine_velocity(centres)
    targets = np.array([[-0.73, 0.18, 0.44], [0.02, -0.31, 0.11], [0.81, 0.62, -0.58]])

    sampler = FVMVelocityInterpolator(centres, cKDTree(centres), neighbours=8)
    first = sampler.sample(targets, velocity, gradient)
    second = sampler.sample(targets.copy(), velocity, gradient)
    expected, _ = _affine_velocity(targets)

    np.testing.assert_allclose(first, expected, atol=2.0e-15)
    np.testing.assert_allclose(second, expected, atol=2.0e-15)
    assert len(sampler._cache) == 1


def test_weighted_trace_smooths_cellwise_velocity_noise():
    centres = np.column_stack((np.arange(8.0), np.zeros(8), np.zeros(8)))
    velocity = np.column_stack((centres[:, 0] + 0.2 * (-1.0) ** centres[:, 0], np.zeros((8, 2))))
    gradient = np.zeros((8, 3, 3))
    gradient[:, 0, 0] = 1.0
    targets = np.column_stack((np.linspace(2.45, 2.55, 101), np.zeros((101, 2))))

    nearest = np.empty((len(targets), 3))
    tree = cKDTree(centres)
    _, index = tree.query(targets)
    nearest[:] = velocity[index] + np.einsum(
        "ni,nij->nj", targets - centres[index], gradient[index]
    )

    sampler = FVMVelocityInterpolator(centres, tree, neighbours=4)
    weighted = sampler.sample(targets, velocity, gradient)

    assert np.max(np.abs(np.diff(weighted[:, 0]))) < 0.1 * np.max(np.abs(np.diff(nearest[:, 0])))
