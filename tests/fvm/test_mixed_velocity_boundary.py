"""Manufactured identities for the Billuart-style mixed velocity boundary."""

import numpy as np
import pytest

from openonda.fvm.fields.mixed_velocity_boundary import (
    reconstruct_mixed_velocity_boundary,
)


def tangential_velocity_normal_gradient(jacobian, normals):
    """Tangential part of du/dn = J n, with J[i, j] = du_i/dx_j."""
    gradient = np.einsum("fij,fj->fi", jacobian, normals)
    return gradient - np.einsum("fi,fi->f", gradient, normals)[:, np.newaxis] * normals


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
    normal_gradient = tangential_velocity_normal_gradient(jacobian, normals)

    reconstructed = reconstruct_mixed_velocity_boundary(
        owner_velocity,
        normals,
        distance,
        normal_velocity,
        normal_gradient,
    )

    np.testing.assert_allclose(reconstructed, face_velocity, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(
        np.einsum("fi,fi->f", reconstructed, normals),
        normal_velocity,
        rtol=2e-14,
        atol=2e-14,
    )


def test_tangential_velocity_normal_gradient_has_no_normal_component():
    normals = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    jacobian = np.arange(27, dtype=np.float64).reshape(3, 3, 3)

    normal_gradient = tangential_velocity_normal_gradient(jacobian, normals)

    np.testing.assert_allclose(np.einsum("fi,fi->f", normal_gradient, normals), 0.0)


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("normal_sign", [-1.0, 1.0])
def test_vortical_boundary_data_remain_exact_when_flow_reverses_on_each_box_face(axis, normal_sign):
    normal = np.zeros((1, 3))
    normal[0, axis] = normal_sign
    face_centre = np.array([[0.37, -0.26, 0.19]])
    distance = np.array([0.08])
    owner_centre = face_centre - distance[:, None] * normal
    # Divergence-free linear velocity, with nonzero vorticity in all directions.
    jacobian = np.array([[[0.2, -0.7, 0.4], [0.8, -0.1, -0.3], [-0.5, 0.6, -0.1]]])
    normal_gradient = tangential_velocity_normal_gradient(jacobian, normal)

    for flow_sign in (-1.0, 1.0):
        offset = 2.0 * flow_sign * normal
        owner_velocity = np.einsum("fij,fj->fi", jacobian, owner_centre) + offset
        face_velocity = np.einsum("fij,fj->fi", jacobian, face_centre) + offset
        normal_velocity = np.einsum("fi,fi->f", face_velocity, normal)
        assert np.sign(normal_velocity[0]) == flow_sign
        reconstructed = reconstruct_mixed_velocity_boundary(
            owner_velocity, normal, distance, normal_velocity, normal_gradient
        )
        np.testing.assert_allclose(reconstructed, face_velocity, rtol=2e-14, atol=2e-14)


# Direction-aware coupling condition ------------------------------------------------


def test_inflow_weight_is_dirichlet_on_inflow_and_mixed_beyond_the_transition():
    from openonda.fvm.fields.mixed_velocity_boundary import coupled_inflow_weight

    normal_velocity = np.array([-2.0, -1.0e-9, 0.0, 0.025, 0.05, 0.075, 0.1, 0.3])
    weight = coupled_inflow_weight(normal_velocity, 0.1)
    np.testing.assert_array_equal(weight[:3], 1.0)
    np.testing.assert_array_equal(weight[-2:], 0.0)
    assert np.all(np.diff(weight) <= 0.0)
    np.testing.assert_allclose(weight[4], 0.5)
    with pytest.raises(ValueError):
        coupled_inflow_weight(normal_velocity, 0.0)


def _direction_aware_faces(seed=20261030):
    rng = np.random.default_rng(seed)
    n_faces = 24
    normals = rng.normal(size=(n_faces, 3))
    normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]
    distance = rng.uniform(0.01, 0.1, size=n_faces)
    owner = rng.normal(size=(n_faces, 3))
    external = rng.normal(size=(n_faces, 3))
    normal_velocity = np.einsum("ij,ij->i", external, normals)
    gradient = tangential_velocity_normal_gradient(rng.normal(size=(n_faces, 3, 3)), normals)
    return normals, distance, owner, external, normal_velocity, gradient


def test_direction_aware_faces_take_the_external_velocity_on_inflow_and_the_mixed_value_on_outflow():
    from openonda.fvm.fields.mixed_velocity_boundary import coupled_inflow_weight

    normals, distance, owner, external, normal_velocity, gradient = _direction_aware_faces()
    weight = coupled_inflow_weight(normal_velocity, 0.2)
    mixed = reconstruct_mixed_velocity_boundary(owner, normals, distance, normal_velocity, gradient)
    blended = reconstruct_mixed_velocity_boundary(
        owner,
        normals,
        distance,
        normal_velocity,
        gradient,
        inflow_velocity=external,
        inflow_weight=weight,
    )
    inflow = weight == 1.0
    outflow = weight == 0.0
    assert inflow.any() and outflow.any()
    np.testing.assert_allclose(blended[inflow], external[inflow], rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(blended[outflow], mixed[outflow], rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(
        np.einsum("ij,ij->i", blended, normals), normal_velocity, rtol=1e-13, atol=1e-13
    )


def test_inflow_faces_do_not_feed_the_owner_velocity_back_into_the_cell():
    # The NACA 4412 instability: a tangential Neumann closure on an inflow face
    # returns the owner's own (extrapolated) velocity into the owner. With the
    # direction-aware condition an inflow face value is independent of the owner.
    from openonda.fvm.fields.mixed_velocity_boundary import coupled_inflow_weight

    normals, distance, owner, external, normal_velocity, gradient = _direction_aware_faces(7)
    weight = coupled_inflow_weight(normal_velocity, 0.2)
    perturbed = owner + np.random.default_rng(3).normal(size=owner.shape)
    face = [
        reconstruct_mixed_velocity_boundary(
            value,
            normals,
            distance,
            normal_velocity,
            gradient,
            inflow_velocity=external,
            inflow_weight=weight,
        )
        for value in (owner, perturbed)
    ]
    inflow = weight == 1.0
    np.testing.assert_array_equal(face[0][inflow], face[1][inflow])
    assert not np.allclose(face[0][~inflow], face[1][~inflow])


def _direction_aware_box(external_velocity):
    from openonda.fvm.fields.mixed_velocity_boundary import (
        coupled_inflow_weight,
        update_mixed_velocity_boundary,
    )
    from openonda.fvm.mesh.geometry import compute_mesh_geometry
    from tests.support.fvm_mesh import structured_box

    mesh = structured_box(3, 3, 3)
    geometry = compute_mesh_geometry(mesh, compute_lsq=False)
    n_cells = mesh["n_cells"]
    n_total = n_cells + mesh["n_faces"] - mesh["n_interior_faces"]
    rng = np.random.default_rng(11)
    velocity = np.zeros((n_total, 3))
    velocity[:n_cells] = external_velocity + 0.3 * rng.normal(size=(n_cells, 3))
    for patch in mesh["boundary"]:
        start, count = patch["start_face"], patch["n_faces"]
        sf = geometry["face_area_vector"][start : start + count]
        normal = sf / np.linalg.norm(sf, axis=1)[:, None]
        external = np.tile(external_velocity, (count, 1))
        patch["velocity_type"] = "mixed_velocity"
        patch["normal_velocity_field"] = normal @ external_velocity
        patch["tangential_velocity_normal_gradient_field"] = rng.normal(size=(count, 3))
        patch["tangential_velocity_normal_gradient_field"] -= (
            np.einsum("ij,ij->i", patch["tangential_velocity_normal_gradient_field"], normal)[
                :, None
            ]
            * normal
        )
        patch["inflow_velocity_field"] = external
        patch["inflow_weight_field"] = coupled_inflow_weight(patch["normal_velocity_field"], 0.5)
        update_mixed_velocity_boundary(velocity, patch, mesh, geometry)
    return mesh, geometry, velocity


@pytest.mark.parametrize("component", range(3))
def test_direction_aware_diffusion_flux_matches_the_reconstructed_face_velocity(component):
    from openonda.fvm.assemble import diffusion

    # An oblique stream: every box face is inflow or outflow, two lie in the
    # transition band of the smooth weight.
    mesh, geometry, velocity = _direction_aware_box(np.array([0.9, -0.25, 0.4]))
    viscosity = 1.0e-3
    flux = diffusion.assemble_diffusion_term(
        velocity[:, component],
        np.zeros((len(velocity), 3)),
        viscosity,
        mesh,
        geometry,
        mesh["boundary"],
        vector_field=velocity,
        component=component,
    )
    for patch in mesh["boundary"]:
        faces = np.arange(patch["start_face"], patch["start_face"] + patch["n_faces"])
        owners = mesh["owners"][faces]
        ghosts = mesh["n_cells"] + faces - mesh["n_interior_faces"]
        area = np.linalg.norm(geometry["face_area_vector"][faces], axis=1)
        distance = np.einsum(
            "ij,ij->i",
            geometry["cell_connection_vector"][faces],
            geometry["face_area_vector"][faces] / area[:, None],
        )
        coefficient = viscosity * area / distance
        weight = patch["inflow_weight_field"]
        # Dirichlet share: nu A (u_b - u_P) / d; mixed share of the tangential
        # part: the prescribed gradient flux nu A g_t.
        normal = geometry["face_area_vector"][faces] / area[:, None]
        jump = velocity[ghosts] - velocity[owners]
        normal_jump = np.einsum("ij,ij->i", jump, normal)[:, None] * normal
        dirichlet_jump = patch["inflow_velocity_field"] - velocity[owners]
        dirichlet_jump -= np.einsum("ij,ij->i", dirichlet_jump, normal)[:, None] * normal
        g = patch["tangential_velocity_normal_gradient_field"]
        expected = -(
            coefficient[:, None] * normal_jump
            + weight[:, None] * coefficient[:, None] * dirichlet_jump
            + (1.0 - weight)[:, None] * viscosity * area[:, None] * g
        )[:, component]
        np.testing.assert_allclose(flux["flux_tf"][faces], expected, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("component", range(3))
def test_direction_aware_convection_never_gives_inflow_faces_an_owner_coefficient(component):
    from openonda.fvm.assemble import convection

    external = np.array([0.9, -0.25, 0.4])
    mesh, geometry, velocity = _direction_aware_box(external)
    face_flux = np.einsum("ij,j->i", geometry["face_area_vector"], external)
    for patch in mesh["boundary"]:
        result = convection.assemble_convection_term_boundary(
            velocity[:, component],
            face_flux,
            patch,
            mesh,
            geometry,
            scheme="linear_upwind",
            component=component,
        )
        faces = result["face_indices"]
        inflow = patch["inflow_weight_field"] == 1.0
        np.testing.assert_array_equal(result["flux_cf"][inflow], 0.0)
        assert np.all(result["flux_cf"] * np.sign(face_flux[faces]) >= 0.0)
        # The total flux is the face flux times the reconstructed face value.
        ghosts = mesh["n_cells"] + faces - mesh["n_interior_faces"]
        np.testing.assert_allclose(
            result["flux_tf"],
            face_flux[faces] * velocity[ghosts, component],
            rtol=1e-12,
            atol=1e-14,
        )
