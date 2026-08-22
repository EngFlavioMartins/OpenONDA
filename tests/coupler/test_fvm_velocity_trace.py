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


def test_constant_and_solid_body_rotation_are_exact_on_a_graded_mesh():
    axes = (
        np.linspace(-1.0, 1.0, 8) ** 3,
        np.sign(np.linspace(-1.0, 1.0, 9)) * np.abs(np.linspace(-1.0, 1.0, 9)) ** 1.5,
        np.linspace(-1.0, 1.0, 7),
    )
    centres = np.column_stack(
        [component.ravel() for component in np.meshgrid(*axes, indexing="ij")]
    )
    rng = np.random.default_rng(13)
    targets = rng.uniform(-0.85, 0.85, (200, 3))
    sampler = FVMVelocityInterpolator(centres, cKDTree(centres), neighbours=4)

    constant = np.tile([0.7, -0.1, 0.2], (len(centres), 1))
    zero_gradient = np.zeros((len(centres), 3, 3))
    np.testing.assert_allclose(
        sampler.sample(targets, constant, zero_gradient),
        np.tile([0.7, -0.1, 0.2], (len(targets), 1)),
        atol=3.0e-16,
    )

    angular_velocity = np.array([0.3, -0.4, 0.8])
    rotation_gradient = 0.5 * np.array(
        [
            [0.0, angular_velocity[2], -angular_velocity[1]],
            [-angular_velocity[2], 0.0, angular_velocity[0]],
            [angular_velocity[1], -angular_velocity[0], 0.0],
        ]
    )
    velocity = 0.5 * np.cross(angular_velocity, centres)
    gradient = np.broadcast_to(rotation_gradient, (len(centres), 3, 3)).copy()
    expected = 0.5 * np.cross(angular_velocity, targets)
    np.testing.assert_allclose(sampler.sample(targets, velocity, gradient), expected, atol=8e-16)


def _smooth_velocity(points):
    x, y, z = np.asarray(points).T
    velocity = np.column_stack(
        (
            np.sin(x) + 0.2 * y * z,
            np.cos(y) - 0.3 * x * z,
            np.exp(0.2 * z) + 0.1 * x * y,
        )
    )
    gradient = np.empty((len(points), 3, 3))
    gradient[:, 0, :] = np.column_stack((np.cos(x), -0.3 * z, 0.1 * y))
    gradient[:, 1, :] = np.column_stack((0.2 * z, -np.sin(y), 0.1 * x))
    gradient[:, 2, :] = np.column_stack((0.2 * y, -0.3 * x, 0.2 * np.exp(0.2 * z)))
    return velocity, gradient


def _graded_interpolation_error(n):
    uniform = np.linspace(-1.0, 1.0, n)
    graded = np.sign(uniform) * np.abs(uniform) ** 1.5
    centres = np.column_stack(
        [component.ravel() for component in np.meshgrid(graded, uniform, graded, indexing="ij")]
    )
    velocity, gradient = _smooth_velocity(centres)
    target_axis = np.linspace(-0.78, 0.78, 11)
    targets = np.column_stack(
        [component.ravel() for component in np.meshgrid(target_axis, target_axis, target_axis)]
    )
    exact, _ = _smooth_velocity(targets)
    sampled = FVMVelocityInterpolator(centres, cKDTree(centres), neighbours=4).sample(
        targets, velocity, gradient
    )
    return float(np.linalg.norm(sampled - exact) / np.linalg.norm(exact))


def test_quadratic_smooth_trace_is_second_order_on_graded_meshes():
    errors = np.array([_graded_interpolation_error(n) for n in (7, 13, 25)])
    orders = np.log2(errors[:-1] / errors[1:])
    assert np.all(orders > 1.7), (errors, orders)


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
