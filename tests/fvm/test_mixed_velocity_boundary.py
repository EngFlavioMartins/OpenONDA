"""Certification for the normal-value/tangential-gradient velocity BC."""

import numpy as np

from source.solvers.fvm.assemble.diffusion import assemble_diffusion_term
from source.solvers.fvm.assemble.momentum import solve_momentum_predictor
from source.solvers.fvm.fields.mixed_velocity_boundary import (
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
        "boundary_neighbour_cell": np.full(6, -1, dtype=np.int32),
        "boundary": [],
    }
    geo = {
        "face_area_vector": sf,
        "cell_connection_vector": face_centres,
        "face_centre": face_centres,
        "cell_centre": np.zeros((1, 3)),
        "face_interpolation_weight": np.full(6, 0.5),
        "wall_distance": np.full(6, 0.5),
        "cell_volume": np.ones(1),
        "gradient_scheme": "gauss",
    }
    return mesh, geo


def test_reconstruction_satisfies_both_directional_constraints():
    rng = np.random.default_rng(41)
    n_faces = 40
    normal = rng.normal(size=(n_faces, 3))
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    owner = rng.normal(size=(n_faces, 3))
    distance = rng.uniform(0.05, 0.7, size=n_faces)
    normal_velocity = rng.normal(size=n_faces)
    gradient = rng.normal(size=(n_faces, 3))
    gradient -= np.einsum("ij,ij->i", gradient, normal)[:, None] * normal

    face = reconstruct_normal_velocity_tangential_gradient(
        owner, normal, distance, normal_velocity, gradient
    )

    np.testing.assert_allclose(np.einsum("ij,ij->i", face, normal), normal_velocity, atol=2.0e-14)
    sn_grad = (face - owner) / distance[:, None]
    tangent_sn_grad = sn_grad - np.einsum("ij,ij->i", sn_grad, normal)[:, None] * normal
    np.testing.assert_allclose(tangent_sn_grad, gradient, atol=2.0e-14)


def test_linear_manufactured_field_has_exact_mixed_diffusive_flux():
    mesh, geo = _one_cell_cube()
    jacobian = np.array([[0.2, -0.3, 0.5], [0.4, -0.1, 0.7], [-0.6, 0.2, -0.1]])
    offset = np.array([0.8, -0.4, 0.2])
    face_velocity = geo["face_centre"] @ jacobian.T + offset
    normal = geo["face_area_vector"]
    normal_velocity_gradient = normal @ jacobian.T
    normal_velocity = np.einsum("ij,ij->i", face_velocity, normal)
    tangential_gradient = (
        normal_velocity_gradient
        - np.einsum("ij,ij->i", normal_velocity_gradient, normal)[:, None] * normal
    )
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
        expected = -kinematic_viscosity * normal_velocity_gradient[:, component]
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
    volumetric_face_flux = np.zeros(6)

    _, momentum_diagonal = solve_momentum_predictor(
        velocity,
        pressure,
        volumetric_face_flux,
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
