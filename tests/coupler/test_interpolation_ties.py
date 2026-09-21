"""Equal-distance donor shells must not introduce a preferred direction."""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from source.coupler.interpolation import FVMVelocityInterpolator


@pytest.mark.parametrize("seed", range(10))
def test_curved_velocity_is_independent_of_tied_donor_order(seed):
    positions = np.array(np.meshgrid([-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0])).reshape(3, -1).T
    velocity = positions**3
    gradient = np.zeros((len(positions), 3, 3))
    gradient[:, range(3), range(3)] = 3 * positions**2
    order = np.random.default_rng(seed).permutation(len(positions))
    trace = FVMVelocityInterpolator(positions[order], cKDTree(positions[order]), neighbour_count=4)
    sampled = trace.sample(np.zeros((1, 3)), velocity[order], gradient[order])
    # Reflection symmetry requires zero at the centre. A four-of-eight tie
    # previously produced order-dependent components between -2 and +2.
    np.testing.assert_allclose(sampled, 0, atol=1e-14, rtol=0)


def test_ties_larger_than_one_extra_donor_are_complete():
    angle = np.arange(32) * (2 * np.pi / 32)
    positions = np.column_stack((np.cos(angle), np.sin(angle), np.zeros(32)))
    field = positions.copy()
    trace = FVMVelocityInterpolator(positions, cKDTree(positions), neighbour_count=4)
    np.testing.assert_allclose(trace.sample_cell_field(np.zeros((1, 3)), field), 0, atol=1e-14)


def test_untied_stencil_keeps_original_inverse_distance_taylor_value():
    rng = np.random.default_rng(991)
    positions = rng.normal(size=(31, 3))
    points = rng.normal(size=(7, 3))
    velocity = rng.normal(size=(31, 3))
    gradient = rng.normal(size=(31, 3, 3))
    tree = cKDTree(positions)
    distance, indices = tree.query(points, k=4)
    weight = 1 / distance**2
    weight /= weight.sum(axis=1, keepdims=True)
    expected = np.einsum(
        "nk,nkj->nj",
        weight,
        velocity[indices]
        + np.einsum("nki,nkij->nkj", points[:, None] - positions[indices], gradient[indices]),
    )
    trace = FVMVelocityInterpolator(positions, tree, neighbour_count=4)
    np.testing.assert_allclose(trace.sample(points, velocity, gradient), expected, atol=1e-14)
    np.testing.assert_array_equal(trace.sample(positions, velocity, gradient), velocity)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("chunk_size", [1, 17, 100_000])
def test_fused_reconstruction_matches_vectorized_taylor_sum(dtype, chunk_size):
    rng = np.random.default_rng(427)
    axis = np.linspace(-1.0, 1.0, 5)
    positions = np.array(np.meshgrid(axis, axis, axis, indexing="ij")).reshape(3, -1).T
    # Include exact donors, four/eight-way ties and generic off-grid queries.
    points = np.concatenate((positions[::11], positions[::13] + 0.25, rng.normal(size=(19, 3))))
    velocity = rng.normal(size=(len(positions), 3)).astype(dtype)
    gradient = rng.normal(size=(len(positions), 3, 3)).astype(dtype).transpose(0, 2, 1)
    trace = FVMVelocityInterpolator(positions, cKDTree(positions))
    indices, weights = trace._stencil(points)
    expected = np.einsum(
        "nk,nkj->nj",
        weights,
        velocity.astype(float)[indices]
        + np.einsum(
            "nki,nkij->nkj",
            points[:, None] - positions[indices],
            gradient.astype(float)[indices],
            optimize=True,
        ),
        optimize=True,
    )
    np.testing.assert_allclose(
        trace.sample(points, velocity, gradient, chunk_size=chunk_size),
        expected,
        rtol=2e-15,
        atol=2e-15,
    )
    assert trace.sample(np.empty((0, 3)), velocity, gradient).shape == (0, 3)


@pytest.mark.parametrize("velocity_count,gradient_count", [(3, 4), (4, 3)])
def test_reconstruction_rejects_mismatched_donor_fields(velocity_count, gradient_count):
    positions = np.arange(12, dtype=float).reshape(4, 3)
    trace = FVMVelocityInterpolator(positions, cKDTree(positions))
    with pytest.raises(ValueError, match="one row per FVM donor cell"):
        trace.sample(
            positions,
            np.zeros((velocity_count, 3)),
            np.zeros((gradient_count, 3, 3)),
        )
