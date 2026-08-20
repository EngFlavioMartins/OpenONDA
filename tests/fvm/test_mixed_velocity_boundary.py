"""Certification for the normal-value/tangential-gradient velocity BC."""

import numpy as np

from source.solvers.FVM.assemble.diffusion import assemble_diffusion_term
from source.solvers.FVM.assemble.momentum import solve_momentum_predictor
from source.solvers.FVM.fields.mixed_velocity_boundary import (
    reconstruct_normal_velocity_tangential_gradient,
)


def _one_cell_cube():
    sf = np.array(
        [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]],
        dtype=float,
    )
    face_centres = 0.5 * sf
    mesh = {
        "n_cells": 1,
        "n_interior_faces": 0,
        "n_faces": 6,
        "owners": np.zeros(6, dtype=np.int32),
        "neighbours": np.zeros(0, dtype=np.int32),
        "boundary_neighbours": np.full(6, -1, dtype=np.int32),
        "boundary": [],
    }
    geo = {
        "face_sf": sf,
        "face_cf_vector": face_centres,
        "face_centroids": face_centres,
        "cell_centroids": np.zeros((1, 3)),
        "face_weights": np.full(6, 0.5),
        "wall_dist": np.full(6, 0.5),
        "cell_volumes": np.ones(1),
        "gradient_scheme": "gauss",
    }
    return mesh, geo


def test_reconstruction_satisfies_both_directional_constraints():
    rng = np.random.default_rng(41)
    n_faces = 40
    normals = rng.normal(size=(n_faces, 3))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    owner = rng.normal(size=(n_faces, 3))
    distance = rng.uniform(0.05, 0.7, size=n_faces)
    normal_velocity = rng.normal(size=n_faces)
    gradient = rng.normal(size=(n_faces, 3))
    gradient -= np.einsum("ij,ij->i", gradient, normals)[:, None] * normals

    face = reconstruct_normal_velocity_tangential_gradient(
        owner, normals, distance, normal_velocity, gradient
    )

    np.testing.assert_allclose(np.einsum("ij,ij->i", face, normals), normal_velocity, atol=2.0e-14)
    sn_grad = (face - owner) / distance[:, None]
    tangent_sn_grad = sn_grad - np.einsum("ij,ij->i", sn_grad, normals)[:, None] * normals
    np.testing.assert_allclose(tangent_sn_grad, gradient, atol=2.0e-14)


def test_linear_manufactured_field_has_exact_mixed_diffusive_flux():
    mesh, geo = _one_cell_cube()
    jacobian = np.array([[0.2, -0.3, 0.5], [0.4, -0.1, 0.7], [-0.6, 0.2, -0.1]])
    offset = np.array([0.8, -0.4, 0.2])
    face_velocity = geo["face_centroids"] @ jacobian.T + offset
    normals = geo["face_sf"]
    d_u_dn = normals @ jacobian.T
    normal_velocity = np.einsum("ij,ij->i", face_velocity, normals)
    tangential_gradient = d_u_dn - np.einsum("ij,ij->i", d_u_dn, normals)[:, None] * normals
    patch = {
        "name": "cut",
        "start_face": 0,
        "n_faces": 6,
        "velocity_type": "normalValueTangentialGradient",
        "normal_velocity_field": normal_velocity,
        "tangential_gradient_field": tangential_gradient,
    }
    mesh["boundary"] = [patch]
    velocity = np.vstack((offset, face_velocity))
    kinematic_viscosity = 0.17

    for component in range(3):
        flux = assemble_diffusion_term(
            velocity[:, component],
            np.zeros((7, 3)),
            kinematic_viscosity,
            mesh,
            geo,
            [patch],
            vector_field=velocity,
            component=component,
        )
        expected = -kinematic_viscosity * d_u_dn[:, component]
        np.testing.assert_allclose(flux["flux_tf"], expected, atol=2.0e-14)


def test_mixed_boundary_activates_component_momentum_diagonals():
    mesh, geo = _one_cell_cube()
    mixed = {
        "name": "xcut",
        "start_face": 0,
        "n_faces": 2,
        "velocity_type": "normalValueTangentialGradient",
        "normal_velocity_field": np.zeros(2),
        "tangential_gradient_field": np.zeros((2, 3)),
    }
    floating = {
        "name": "other",
        "start_face": 2,
        "n_faces": 4,
        "velocity_type": "zeroGradient",
    }
    boundaries = [mixed, floating]
    mesh["boundary"] = boundaries
    velocity = np.zeros((7, 3))
    pressure = np.zeros(7)
    face_flux = np.zeros(6)

    _, momentum_diagonal = solve_momentum_predictor(
        velocity,
        pressure,
        face_flux,
        1.0,
        0.1,
        mesh,
        geo,
        boundaries,
        convection_scheme="upwind",
        solver="spsolve",
        under_relaxation=1.0,
        time_step_size=0.2,
        velocity_old=velocity,
    )

    assert momentum_diagonal.shape == (1, 3)
    assert momentum_diagonal[0, 0] > momentum_diagonal[0, 1]
    np.testing.assert_allclose(momentum_diagonal[0, 1], momentum_diagonal[0, 2])
