"""Manufactured identities for the Billuart-style mixed velocity boundary."""

import numpy as np

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
