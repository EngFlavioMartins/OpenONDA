"""Affine consistency of native mixed traces on genuinely skew boundary faces."""

import numpy as np
import pytest

from source.coupler.boundary import tangential_normal_velocity_gradient
from source.solvers.fvm.assemble.convection import assemble_convection_term_boundary
from source.solvers.fvm.assemble.diffusion import assemble_diffusion_term
from source.solvers.fvm.fields.mixed_velocity_boundary import (
    reconstruct_normal_velocity_tangential_gradient,
    update_normal_velocity_tangential_gradient_boundary,
)
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from tests.support.fvm_mesh import structured_box


@pytest.mark.parametrize(
    "jacobian",
    [
        np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        np.diag([1.0, -1.0, 0.0]),
        np.array([[0.2, -0.7, 1.3], [0.6, -0.1, -0.4], [-0.8, 1.2, -0.1]]),
    ],
    ids=["shear", "rotation", "strain", "three_dimensional"],
)
@pytest.mark.parametrize("include_total_flux", [False, True])
def test_skew_mixed_faces_and_momentum_fluxes_preserve_affine_flow(
    jacobian, include_total_flux
):
    mesh = structured_box(4, 3, 3)
    mesh["vertex_position"] = mesh["vertex_position"] @ np.array(
        [[1.0, 0.3, -0.2], [0.1, 1.2, 0.4], [0.2, -0.1, 0.8]]
    ).T
    geometry = compute_mesh_geometry(mesh)
    n_cells, n_interior = mesh["n_cells"], mesh["n_interior_faces"]
    faces = np.arange(n_interior, mesh["n_faces"])
    ghosts = n_cells + faces - n_interior
    owner_velocity = geometry["cell_centre"] @ jacobian.T + [-0.3, 0.2, -0.1]
    exact_face = geometry["face_centre"] @ jacobian.T + [-0.3, 0.2, -0.1]
    velocity = np.concatenate((owner_velocity, np.zeros((len(faces), 3))))
    normal = geometry["face_area_vector"] / geometry["face_area"][:, None]
    un = np.einsum("ij,ij->i", normal, exact_face)
    gt = tangential_normal_velocity_gradient(
        np.broadcast_to(jacobian, (mesh["n_faces"], 3, 3)), normal
    )
    phi = un * geometry["face_area"]
    assert np.any(phi[faces] < 0) and np.any(phi[faces] > 0)
    displacement = geometry["cell_connection_vector"][faces]
    dn = np.einsum("ij,ij->i", displacement, normal[faces])
    rt = displacement - dn[:, None] * normal[faces]
    assert np.max(np.linalg.norm(rt, axis=1)) > 0.01

    for patch in mesh["boundary"]:
        patch_faces = np.arange(patch["start_face"], patch["start_face"] + patch["n_faces"])
        patch.update(
            velocity_type="normalValueTangentialGradient",
            normal_velocity_field=un[patch_faces].copy(),
            tangential_gradient_field=gt[patch_faces].copy(),
        )
        update_normal_velocity_tangential_gradient_boundary(velocity, patch, mesh, geometry)
        # The derivative remains physical boundary data, not a container for
        # a geometric correction to the face value.
        np.testing.assert_array_equal(patch["tangential_gradient_field"], gt[patch_faces])

    np.testing.assert_allclose(velocity[ghosts], exact_face[faces], rtol=0, atol=3e-13)
    np.testing.assert_allclose(
        np.einsum("ij,ij->i", velocity[ghosts], normal[faces]), un[faces], rtol=0, atol=3e-14
    )
    viscosity = 0.037
    expected_diffusion = -viscosity * geometry["face_area"][faces, None] * (
        normal[faces] @ jacobian.T
    )
    for component in range(3):
        gradient = np.broadcast_to(jacobian[component], (len(velocity), 3))
        diffusion = assemble_diffusion_term(
            velocity[:, component],
            gradient,
            viscosity,
            mesh,
            geometry,
            mesh["boundary"],
            vector_field=velocity,
            component=component,
            include_total_flux=include_total_flux,
        )
        actual = (
            diffusion["flux_cf"][faces] * velocity[mesh["owners"][faces], component]
            + diffusion["flux_vf"][faces]
        )
        np.testing.assert_allclose(actual, expected_diffusion[:, component], rtol=0, atol=3e-14)
        if include_total_flux:
            np.testing.assert_allclose(diffusion["flux_tf"][faces], actual, rtol=0, atol=3e-14)
        else:
            assert "flux_tf" not in diffusion
        for patch in mesh["boundary"]:
            terms = assemble_convection_term_boundary(
                velocity[:, component], phi, patch, mesh, geometry, component=component
            )
            patch_faces = terms["face_indices"]
            np.testing.assert_allclose(
                terms["flux_tf"],
                phi[patch_faces] * exact_face[patch_faces, component],
                rtol=0,
                atol=3e-14,
            )


def test_skew_increment_is_tangential_and_does_not_modify_input_trace():
    normal = np.array([[0.6, 0.8, 0.0]])
    owner = np.array([[0.1, -0.3, 0.7]])
    gt = np.array([[-0.8, 0.6, 0.2]])
    increment = np.array([[1.1, -0.4, 0.8]])
    original_gt, original_increment = gt.copy(), increment.copy()
    actual = reconstruct_normal_velocity_tangential_gradient(
        owner, normal, np.array([0.03]), np.array([-0.25]), gt,
        skew_velocity_increment=increment,
    )
    assert np.dot(actual[0], normal[0]) == pytest.approx(-0.25, abs=3e-16)
    np.testing.assert_array_equal(gt, original_gt)
    np.testing.assert_array_equal(increment, original_increment)


def test_orthogonal_mixed_boundary_skips_gradient_reconstruction(monkeypatch):
    def unexpected_gradient(*args, **kwargs):
        raise AssertionError("Orthogonal mixed faces must not reconstruct a gradient")

    monkeypatch.setattr(
        "source.solvers.fvm.fields.mixed_velocity_boundary.boundary_owner_gradient",
        unexpected_gradient,
    )
    mesh = structured_box(2, 2, 2)
    geometry = compute_mesh_geometry(mesh)
    patch = mesh["boundary"][0]
    indices = np.arange(patch["start_face"], patch["start_face"] + patch["n_faces"])
    patch.update(
        velocity_type="normalValueTangentialGradient",
        normal_velocity_field=np.full(len(indices), -0.7),
        tangential_gradient_field=np.tile([0.0, 0.3, -0.1], (len(indices), 1)),
    )
    velocity = np.tile(
        [0.2, -0.4, 0.1], (mesh["n_cells"] + mesh["n_faces"] - mesh["n_interior_faces"], 1)
    )
    update_normal_velocity_tangential_gradient_boundary(velocity, patch, mesh, geometry)
    for component in range(3):
        assemble_diffusion_term(
            velocity[:, component], np.zeros_like(velocity), 0.03, mesh, geometry,
            [patch], vector_field=velocity, component=component,
        )


@pytest.mark.parametrize("increment", [np.zeros((2, 3)), np.full((1, 3), np.nan)])
def test_skew_increment_rejects_invalid_data(increment):
    with pytest.raises(ValueError, match="skew_velocity_increment"):
        reconstruct_normal_velocity_tangential_gradient(
            np.zeros((1, 3)), np.array([[1.0, 0.0, 0.0]]), np.array([0.1]),
            np.array([0.0]), np.zeros((1, 3)), skew_velocity_increment=increment,
        )
