"""The span probe must exercise 3D dynamics without violating slip symmetry."""

import numpy as np

from openonda.cylinder_campaign import cylinder_initial_velocity


def test_perturbation_is_solenoidal_compact_and_respects_slip_planes():
    span = 0.96
    points = np.array([[0.55, 0.1, 0.0], [0.75, -0.1, 0.2], [0.9, 0.15, -0.3]])
    eps = 1e-5
    divergence = np.zeros(len(points))
    for axis in range(3):
        displacement = np.eye(3)[axis] * eps
        upper = cylinder_initial_velocity(points + displacement, span)
        lower = cylinder_initial_velocity(points - displacement, span)
        divergence += (upper[:, axis] - lower[:, axis]) / (2 * eps)
    np.testing.assert_allclose(divergence, 0, atol=2e-11)
    assert np.max(np.abs(cylinder_initial_velocity(points, span)[:, 2])) > 1e-4
    # A nonzero centreline transverse velocity excites the shedding mode even
    # on a perfectly reflection-symmetric mesh; it is independent of z.
    centreline = cylinder_initial_velocity(np.array([[0.9, 0, -0.24], [0.9, 0, 0.24]]), span)
    assert abs(centreline[0, 1]) > 1e-4
    assert centreline[0, 1] == centreline[1, 1]
    wall = np.array([[0.55, 0.1, -0.48], [0.75, -0.1, 0.48]])
    np.testing.assert_allclose(cylinder_initial_velocity(wall, span)[:, 2], 0, atol=1e-18)
    upper = cylinder_initial_velocity(wall + [0, 0, eps], span)
    lower = cylinder_initial_velocity(wall - [0, 0, eps], span)
    np.testing.assert_allclose((upper[:, :2] - lower[:, :2]) / (2 * eps), 0, atol=1e-11)
    np.testing.assert_array_equal(
        cylinder_initial_velocity(np.array([[1.48, 0, 0], [-1.48, 0.2, 0.1]]), span),
        [[1.0, 0, 0], [1.0, 0, 0]],
    )
