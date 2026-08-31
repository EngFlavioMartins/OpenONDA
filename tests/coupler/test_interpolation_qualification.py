"""Manufactured-field qualifications for the FVM-to-VPM interpolator."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from source.coupler.interpolation import FVMVelocityInterpolator


def _quadratic_velocity(position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = np.asarray(position).T
    velocity = np.column_stack(
        (
            x**2 + 0.3 * y * z,
            y**2 - 0.2 * x * z,
            z**2 + 0.1 * x * y,
        )
    )
    gradient = np.empty((len(position), 3, 3))
    gradient[:, 0, :] = np.column_stack((2.0 * x, -0.2 * z, 0.1 * y))
    gradient[:, 1, :] = np.column_stack((0.3 * z, 2.0 * y, 0.1 * x))
    gradient[:, 2, :] = np.column_stack((0.3 * y, -0.2 * x, 2.0 * z))
    return velocity, gradient


def _graded_interpolation_error(node_count: int) -> float:
    uniform = np.linspace(-1.0, 1.0, node_count)
    graded = np.sign(uniform) * np.abs(uniform) ** 1.5
    mesh = np.meshgrid(graded, uniform, graded, indexing="ij")
    cell_centre = np.column_stack([component.ravel() for component in mesh])
    velocity, gradient = _quadratic_velocity(cell_centre)
    target_axis = np.linspace(-0.78, 0.78, 11)
    target_mesh = np.meshgrid(target_axis, target_axis, target_axis, indexing="ij")
    target = np.column_stack([component.ravel() for component in target_mesh])
    expected, _ = _quadratic_velocity(target)
    sampled = FVMVelocityInterpolator(
        cell_centre,
        cKDTree(cell_centre),
        neighbour_count=4,
    ).sample(target, velocity, gradient)
    return float(np.linalg.norm(sampled - expected) / np.linalg.norm(expected))


def test_interpolation_is_affine_exact_and_second_order_on_graded_meshes(record_property):
    axis = np.linspace(-1.0, 1.0, 7)
    mesh = np.meshgrid(axis, axis, axis, indexing="ij")
    cell_centre = np.column_stack([component.ravel() for component in mesh])
    gradient_matrix = np.array([[0.2, -0.1, 0.3], [0.4, 0.1, -0.2], [-0.3, 0.2, 0.15]])
    offset = np.array([0.8, -0.2, 0.1])
    velocity = offset + cell_centre @ gradient_matrix
    gradient = np.broadcast_to(gradient_matrix, (len(cell_centre), 3, 3)).copy()
    target = np.array([[-0.73, 0.18, 0.44], [0.02, -0.31, 0.11], [0.81, 0.62, -0.58]])
    interpolator = FVMVelocityInterpolator(
        cell_centre,
        cKDTree(cell_centre),
        neighbour_count=4,
    )
    affine_error = float(
        np.max(
            np.abs(
                interpolator.sample(target, velocity, gradient)
                - (offset + target @ gradient_matrix)
            )
        )
    )

    node_counts = (7, 13, 25)
    errors = np.array([_graded_interpolation_error(n) for n in node_counts])
    orders = np.log2(errors[:-1] / errors[1:])
    record_property("affine_max_abs_error", affine_error)
    for node_count, error in zip(node_counts, errors, strict=True):
        record_property(f"relative_l2_error_n{node_count}", float(error))
    for coarse_count, order in zip(node_counts[:-1], orders, strict=True):
        record_property(f"observed_order_from_n{coarse_count}", float(order))

    assert affine_error < 2.0e-15
    assert np.all(orders > 1.8), (errors, orders)
