"""Independent consistency and conservation checks for moment acquisition."""

import numpy as np
import pytest

from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.mesh.rectilinear import box_mesh_3d
from studies.coupler_accuracy.native_linear_volume_induction_3d import (
    LinearNativeVolumeSources,
    curl_affine,
)
from studies.coupler_accuracy.native_moment_reconstruction_3d import (
    CellLeastSquares,
    reconstruct_linear_face_velocity,
    weak_curl_moments,
)
from studies.coupler_accuracy.native_volume_induction_3d import NativeVolumeSources


def geometry():
    mesh = box_mesh_3d(np.array([-.7, -.2, .3, 1.1]), np.array([-.4, 0., .5, 1.2]), np.array([-.6, -.1, .2, .8]))
    mesh["vertex_position"] += np.random.default_rng(718).normal(0, .007, mesh["vertex_position"].shape)
    geo = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    native = NativeVolumeSources.from_mesh(mesh)
    linear = LinearNativeVolumeSources.from_native(native, geo["cell_centre"])
    return mesh, geo, native, linear


def test_neighbour_reconstruction_recovers_affine_cell_averages_and_constants():
    mesh, _, _, linear = geometry()
    fit = CellLeastSquares.from_mesh(mesh, linear.centroid)
    matrices = np.array([[[.3, -.2, .7], [.5, -.8, .1], [-.4, .9, 1.3]],
                         [[-.7, .3, .5], [.2, -.1, .4], [.1, -.6, .8]]])
    constants = np.array([[.3, -.7, .2], [-.2, .8, 1.1]])
    averages = constants[:, None]+np.einsum("ci,sij->scj", linear.centroid, matrices)
    actual = fit.gradient(averages)
    np.testing.assert_allclose(actual, np.broadcast_to(matrices[:, None], actual.shape), rtol=0, atol=4e-15)
    np.testing.assert_array_equal(fit.gradient(np.broadcast_to(constants[:, None], averages.shape)), 0)
    moment = linear.covariance[None] @ actual
    np.testing.assert_allclose(moment, linear.covariance[None] @ matrices[:, None], rtol=0, atol=1e-16)
    assert np.min(fit.neighbour_count) >= 3


def test_neighbour_reconstruction_rejects_a_two_dimensional_stencil():
    mesh = box_mesh_3d(np.arange(4.), np.arange(4.), np.array([0., 1.]))
    geo = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    with pytest.raises(ValueError, match="all three spatial directions"):
        CellLeastSquares.from_mesh(mesh, geo["cell_centre"])


def test_weak_moments_recover_constant_curl_of_affine_velocity_on_warped_cells():
    mesh, geo, native, linear = geometry()
    matrix = np.array([[.3, -.2, .7], [.5, -.8, .1], [-.4, .9, 1.3]])
    constant = np.array([.3, -.7, 1.2])
    face_u = constant+geo["face_centre"] @ matrix
    integral_u = (constant+linear.centroid @ matrix)*linear.volume[:, None]
    gamma, moment = weak_curl_moments(native, linear.centroid, integral_u, geo["face_centre"], face_u,
                                      np.broadcast_to(matrix, (mesh["n_faces"], 3, 3)))
    np.testing.assert_allclose(gamma[0], linear.volume[:, None]*curl_affine(matrix), rtol=0, atol=3e-15)
    np.testing.assert_allclose(moment, 0, rtol=0, atol=3e-15)
    gamma, moment = weak_curl_moments(native, linear.centroid, linear.volume[:, None]*constant,
                                      geo["face_centre"], np.broadcast_to(constant, face_u.shape))
    np.testing.assert_allclose(gamma, 0, rtol=0, atol=3e-15)
    np.testing.assert_allclose(moment, 0, rtol=0, atol=3e-15)


def test_shared_weak_trace_preserves_global_stokes_and_first_moment_budget():
    mesh, geo, native, linear = geometry()
    generator = np.random.default_rng(928)
    face_u = generator.normal(size=(2, mesh["n_faces"], 3))
    face_g = generator.normal(size=(2, mesh["n_faces"], 3, 3))
    face_u[:, mesh["n_interior_faces"]:] = 0
    face_g[:, mesh["n_interior_faces"]:] = 0
    integral_u = generator.normal(size=(2, mesh["n_cells"], 3))*linear.volume[None, :, None]
    gamma, moment = weak_curl_moments(native, linear.centroid, integral_u, geo["face_centre"], face_u, face_g)
    np.testing.assert_allclose(gamma.sum(axis=1), 0, rtol=0, atol=4e-15)
    global_moment = moment.sum(axis=1)+np.einsum("ci,scj->sij", linear.centroid, gamma)
    expected = -np.cross(np.eye(3)[None], integral_u.sum(axis=1)[:, None])
    np.testing.assert_allclose(global_moment, expected, rtol=0, atol=4e-15)


def test_shared_reconstructed_faces_are_affine_exact_in_the_interior():
    mesh, geo, _, linear = geometry()
    matrix = np.array([[.3, -.2, .7], [.5, -.8, .1], [-.4, .9, 1.3]])
    constant = np.array([.3, -.7, 1.2])
    fit = CellLeastSquares.from_mesh(mesh, linear.centroid)
    value = constant+linear.centroid @ matrix
    gradient = fit.gradient(value)[0]
    boundary = constant+geo["face_centre"] @ matrix
    actual, derivative = reconstruct_linear_face_velocity(mesh, geo, value, gradient, boundary, cell_positions=linear.centroid)
    np.testing.assert_allclose(actual[0], boundary, rtol=0, atol=3e-15)
    np.testing.assert_allclose(derivative[0, :mesh["n_interior_faces"]],
                               np.broadcast_to(matrix, (mesh["n_interior_faces"], 3, 3)), rtol=0, atol=4e-15)
    np.testing.assert_array_equal(derivative[0, mesh["n_interior_faces"]:], 0)
