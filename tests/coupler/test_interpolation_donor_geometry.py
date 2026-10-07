"""Unresolved donor directions must not change interpolation weights."""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from source.coupler.interpolation import FVMVelocityInterpolator


@pytest.mark.parametrize("angle", [0.0, 0.63])
def test_curved_taylor_field_has_no_artificial_variation_normal_to_donor_plane(angle):
    normal = np.array([np.sin(angle), 0, np.cos(angle)])
    tangent = np.array([np.cos(angle), 0, -np.sin(angle)])
    a, b = np.meshgrid(np.linspace(-1, 1, 9), np.linspace(-1, 1, 9))
    donors = a.ravel()[:, None] * tangent + b.ravel()[:, None] * np.array([0, 1, 0])
    velocity = np.column_stack((a.ravel() ** 2, np.sin(b.ravel()), a.ravel() * b.ravel()))
    gradient = np.zeros((len(donors), 3, 3))
    gradient[:, :, 0] = 2 * a.ravel()[:, None] * tangent
    gradient[:, :, 1] = np.cos(b.ravel())[:, None] * np.array([0, 1, 0])
    gradient[:, :, 2] = b.ravel()[:, None] * tangent + a.ravel()[:, None] * np.array([0, 1, 0])
    targets = np.array([0.17, -0.11]) @ np.array([tangent, [0, 1, 0]])
    query = targets + np.array([-0.4, -0.2, 0, 0.2, 0.4])[:, None] * normal
    trace = FVMVelocityInterpolator(donors, cKDTree(donors), neighbour_count=12)
    sampled = trace.prepare(query).sample(velocity, gradient)
    np.testing.assert_allclose(
        sampled, np.broadcast_to(sampled[2], sampled.shape), atol=2e-14, rtol=0
    )
    # Supplied normal derivatives still propagate through the complete 3D
    # Taylor reconstruction; geometric weights must not erase this response.
    gradient += normal[None, :, None] * np.array([0.3, -0.2, 0.4])
    tilted = trace.prepare(query).sample(velocity, gradient)
    expected = sampled + np.array([-0.4, -0.2, 0, 0.2, 0.4])[:, None] * np.array([0.3, -0.2, 0.4])
    np.testing.assert_allclose(tilted, expected, atol=2e-14, rtol=0)
