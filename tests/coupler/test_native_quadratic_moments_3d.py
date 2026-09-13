"""Independent polynomial, volume-integral and conservation qualifications."""

import numpy as np
import pytest

from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.mesh.rectilinear import box_mesh_3d
from studies.coupler_accuracy.native_cell_integrals_3d import NativeCellIntegration
from studies.coupler_accuracy.native_linear_volume_induction_3d import LinearNativeVolumeSources
from studies.coupler_accuracy.native_quadratic_moments_3d import (
    QuadraticCellLeastSquares,
    reconstruct_quadratic_faces,
    weak_quadratic_curl_moments,
)
from studies.coupler_accuracy.native_volume_induction_3d import NativeVolumeSources


def fixture():
    mesh = box_mesh_3d(np.array([-.7, -.2, .3, 1.1]), np.array([-.4, 0., .5, 1.2]), np.array([-.6, -.1, .2, .8]))
    mesh["vertex_position"] += np.random.default_rng(718).normal(0, .007, mesh["vertex_position"].shape)
    geometry = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    native = NativeVolumeSources.from_mesh(mesh)
    linear = LinearNativeVolumeSources.from_native(native, geometry["cell_centre"])
    generator = np.random.default_rng(821)
    constant, gradient = generator.normal(size=(2, 3)), generator.normal(size=(2, 3, 3))
    hessian = generator.normal(size=(2, 3, 3, 3))
    hessian = (hessian+hessian.swapaxes(1, 2))/2
    return mesh, geometry, native, linear, constant, gradient, hessian


def polynomial(x, constant, gradient, hessian):
    value = constant[:, None]+np.einsum("ni,sij->snj", x, gradient)+.5*np.einsum("ni,nk,sikj->snj", x, x, hessian)
    derivative = gradient[:, None]+np.einsum("nk,sikj->snij", x, hessian)
    return value, derivative


@pytest.mark.parametrize("average_input", [False, True])
def test_quadratic_point_and_average_reconstruction_on_warped_3d_cells(average_input):
    mesh, geometry, _, linear, constant, gradient, hessian = fixture()
    centre = linear.centroid if average_input else geometry["cell_centre"]
    covariance = linear.covariance/linear.volume[:, None, None] if average_input else None
    fit = QuadraticCellLeastSquares.from_mesh(mesh, centre, linear.volume, average_covariance=covariance)
    value, exact_g = polynomial(centre, constant, gradient, hessian)
    if average_input:
        value += .5*np.einsum("nik,sikj->snj", covariance, hessian)
    actual_g, actual_h = fit.derivatives(value)
    np.testing.assert_allclose(actual_g, exact_g, rtol=0, atol=3e-13)
    np.testing.assert_allclose(actual_h, np.broadcast_to(hessian[:, None], actual_h.shape), rtol=0, atol=3e-12)
    cell_integral = fit.cell_integrals(value, actual_g, actual_h, linear.volume, linear.centroid, linear.covariance)
    # Independent signed-tetrahedron quadrature, not the polynomial mean formula.
    integration = NativeCellIntegration.from_mesh(mesh, geometry, np.arange(mesh["n_cells"]))
    expected = np.empty_like(cell_integral)
    for cell in range(mesh["n_cells"]):
        q, weight = integration.rule(cell, 4)
        expected[:, cell] = np.einsum("q,sqj->sj", weight, polynomial(q, constant, gradient, hessian)[0])
    np.testing.assert_allclose(cell_integral, expected, rtol=0, atol=2e-14)
    face_u, face_g = polynomial(geometry["face_centre"], constant, gradient, hessian)
    face_h = np.broadcast_to(hessian[:, None], (2, mesh["n_faces"], 3, 3, 3))
    actual_face = reconstruct_quadratic_faces(mesh, geometry, fit, value, actual_g, actual_h, face_u,
                                               boundary_gradient=face_g, boundary_hessian=face_h)
    for actual, reference in zip(actual_face, (face_u, face_g, face_h), strict=True):
        np.testing.assert_allclose(actual, reference, rtol=0, atol=3e-12)
    zeros = fit.derivatives(np.broadcast_to(constant[:, None], value.shape))
    for array in zeros:
        np.testing.assert_array_equal(array, 0)
    assert fit.condition.max() < 1000 and fit.neighbour_count.min() >= 9


def test_weak_quadratic_curl_and_first_moment_match_independent_volume_quadrature():
    mesh, geometry, native, linear, constant, gradient, hessian = fixture()
    face_u, face_g = polynomial(geometry["face_centre"], constant, gradient, hessian)
    face_h = np.broadcast_to(hessian[:, None], (2, mesh["n_faces"], 3, 3, 3))
    integration = NativeCellIntegration.from_mesh(mesh, geometry, np.arange(mesh["n_cells"]))
    integral, expected_gamma, expected_moment = np.empty((2, mesh["n_cells"], 3)), np.empty((2, mesh["n_cells"], 3)), np.empty((2, mesh["n_cells"], 3, 3))
    for cell in range(mesh["n_cells"]):
        q, weight = integration.rule(cell, 4)
        velocity, derivative = polynomial(q, constant, gradient, hessian)
        curl = np.stack((derivative[:, :, 1, 2]-derivative[:, :, 2, 1], derivative[:, :, 2, 0]-derivative[:, :, 0, 2],
                         derivative[:, :, 0, 1]-derivative[:, :, 1, 0]), axis=-1)
        integral[:, cell] = np.einsum("q,sqj->sj", weight, velocity)
        expected_gamma[:, cell] = np.einsum("q,sqj->sj", weight, curl)
        expected_moment[:, cell] = np.einsum("q,qi,sqj->sij", weight, q-linear.centroid[cell], curl)
    gamma, moment = weak_quadratic_curl_moments(native, linear.centroid, integral, geometry["face_centre"], face_u, face_g, face_h)
    np.testing.assert_allclose(gamma, expected_gamma, rtol=0, atol=5e-15)
    np.testing.assert_allclose(moment, expected_moment, rtol=0, atol=5e-15)


def test_shared_quadratic_faces_preserve_global_stokes_and_first_moment_budget():
    mesh, geometry, native, linear, _, _, _ = fixture()
    generator = np.random.default_rng(272)
    u, g, h = [generator.normal(size=(2, mesh["n_faces"], *shape)) for shape in ((3,), (3, 3), (3, 3, 3))]
    h = (h+h.swapaxes(2, 3))/2
    for trace in (u, g, h):
        trace[:, mesh["n_interior_faces"]:] = 0
    integral = generator.normal(size=(2, mesh["n_cells"], 3))*linear.volume[None, :, None]
    gamma, moment = weak_quadratic_curl_moments(native, linear.centroid, integral, geometry["face_centre"], u, g, h)
    np.testing.assert_allclose(gamma.sum(axis=1), 0, rtol=0, atol=4e-15)
    global_moment = moment.sum(axis=1)+np.einsum("ni,snj->sij", linear.centroid, gamma)
    expected = -np.cross(np.eye(3)[None], integral.sum(axis=1)[:, None])
    np.testing.assert_allclose(global_moment, expected, rtol=0, atol=4e-15)


def test_quadratic_reconstruction_rejects_a_two_dimensional_stencil():
    mesh = box_mesh_3d(np.arange(5.), np.arange(5.), np.array([0., 1.]))
    geometry = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    with pytest.raises(ValueError, match="nine quadratic modes in three dimensions"):
        QuadraticCellLeastSquares.from_mesh(mesh, geometry["cell_centre"], geometry["cell_volume"])
