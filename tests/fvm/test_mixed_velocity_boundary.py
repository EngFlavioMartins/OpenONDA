"""Manufactured identities for the Billuart-style mixed velocity boundary."""

import numpy as np
import pytest

from source.coupler.boundary import tangential_normal_velocity_gradient
from source.solvers.fvm.fields.mixed_velocity_boundary import (
    reconstruct_normal_velocity_tangential_gradient,
)


def test_mixed_boundary_exactly_reconstructs_divergence_free_linear_fields():
    rng = np.random.default_rng(20260826)
    n_faces = 32
    normals = rng.normal(size=(n_faces, 3))
    normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]
    distance = rng.uniform(0.01, 0.2, size=n_faces)
    face_centre = rng.normal(size=(n_faces, 3))
    owner_centre = face_centre - distance[:, np.newaxis] * normals

    jacobian = rng.normal(size=(n_faces, 3, 3))
    trace = np.trace(jacobian, axis1=1, axis2=2)
    jacobian[:, 2, 2] -= trace
    offset = rng.normal(size=(n_faces, 3))
    owner_velocity = np.einsum("fij,fj->fi", jacobian, owner_centre) + offset
    face_velocity = np.einsum("fij,fj->fi", jacobian, face_centre) + offset
    normal_velocity = np.einsum("fi,fi->f", face_velocity, normals)
    tangential_gradient = tangential_normal_velocity_gradient(jacobian, normals)

    reconstructed = reconstruct_normal_velocity_tangential_gradient(
        owner_velocity,
        normals,
        distance,
        normal_velocity,
        tangential_gradient,
    )

    np.testing.assert_allclose(reconstructed, face_velocity, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(
        np.einsum("fi,fi->f", reconstructed, normals),
        normal_velocity,
        rtol=2e-14,
        atol=2e-14,
    )


def test_tangential_gradient_has_no_normal_component():
    normals = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    jacobian = np.arange(27, dtype=np.float64).reshape(3, 3, 3)

    tangential_gradient = tangential_normal_velocity_gradient(jacobian, normals)

    np.testing.assert_allclose(np.einsum("fi,fi->f", tangential_gradient, normals), 0.0)


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("normal_sign", [-1.0, 1.0])
def test_vortical_trace_remains_exact_when_flow_reverses_on_each_box_face(axis, normal_sign):
    normal = np.zeros((1, 3))
    normal[0, axis] = normal_sign
    face_centre = np.array([[0.37, -0.26, 0.19]])
    distance = np.array([0.08])
    owner_centre = face_centre - distance[:, None] * normal
    # Divergence-free linear velocity, with nonzero vorticity in all directions.
    jacobian = np.array([[[0.2, -0.7, 0.4], [0.8, -0.1, -0.3], [-0.5, 0.6, -0.1]]])
    tangential_gradient = tangential_normal_velocity_gradient(jacobian, normal)

    for flow_sign in (-1.0, 1.0):
        offset = 2.0 * flow_sign * normal
        owner_velocity = np.einsum("fij,fj->fi", jacobian, owner_centre) + offset
        face_velocity = np.einsum("fij,fj->fi", jacobian, face_centre) + offset
        normal_velocity = np.einsum("fi,fi->f", face_velocity, normal)
        assert np.sign(normal_velocity[0]) == flow_sign
        reconstructed = reconstruct_normal_velocity_tangential_gradient(
            owner_velocity, normal, distance, normal_velocity, tangential_gradient
        )
        np.testing.assert_allclose(reconstructed, face_velocity, rtol=2e-14, atol=2e-14)
