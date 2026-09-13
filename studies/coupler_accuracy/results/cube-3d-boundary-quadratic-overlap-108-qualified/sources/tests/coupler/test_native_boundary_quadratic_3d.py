"""Qualify the effect and consistency of prescribed 3D boundary observations."""

import numpy as np
import pytest

from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.mesh.rectilinear import box_mesh_3d
from studies.coupler_accuracy.native_boundary_quadratic_3d import BoundaryQuadraticCellLeastSquares
from studies.coupler_accuracy.native_linear_volume_induction_3d import LinearNativeVolumeSources
from studies.coupler_accuracy.native_quadratic_moments_3d import QuadraticCellLeastSquares
from studies.coupler_accuracy.native_velocity_curl_integrals_3d import triangle_rule
from studies.coupler_accuracy.native_volume_induction_3d import NativeVolumeSources


def fixture():
    mesh = box_mesh_3d(np.array([-.7, -.2, .3, 1.1]), np.array([-.4, 0., .5, 1.2]), np.array([-.6, -.1, .2, .8]))
    mesh["vertex_position"] += np.random.default_rng(718).normal(0, .007, mesh["vertex_position"].shape)
    geometry = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    native = NativeVolumeSources.from_mesh(mesh)
    linear = LinearNativeVolumeSources.from_native(native, geometry["cell_centre"])
    generator = np.random.default_rng(982)
    constant, gradient = generator.normal(size=(2, 3)), generator.normal(size=(2, 3, 3))
    hessian = generator.normal(size=(2, 3, 3, 3))
    hessian = (hessian+hessian.swapaxes(1, 2))/2
    return mesh, geometry, native, linear, constant, gradient, hessian


def polynomial(x, constant, gradient, hessian):
    value = constant[:, None]+np.einsum("ni,sij->snj", x, gradient)+.5*np.einsum("ni,nk,sikj->snj", x, x, hessian)
    derivative = gradient[:, None]+np.einsum("nk,sikj->snij", x, hessian)
    return value, derivative


@pytest.mark.parametrize("average_input", [False, True])
def test_prescribed_boundary_fit_preserves_general_quadratic_fields(average_input):
    mesh, geometry, _, linear, constant, gradient, hessian = fixture()
    centre = linear.centroid if average_input else geometry["cell_centre"]
    covariance = linear.covariance/linear.volume[:, None, None] if average_input else None
    fit = BoundaryQuadraticCellLeastSquares.from_mesh(mesh, centre, linear.volume, average_covariance=covariance,
                                                      boundary_faces=np.arange(mesh["n_interior_faces"], mesh["n_faces"]))
    value, exact_g = polynomial(centre, constant, gradient, hessian)
    if average_input:
        value += .5*np.einsum("nik,sikj->snj", covariance, hessian)
    wall = polynomial(fit.boundary_position, constant, gradient, hessian)[0]
    actual_g, actual_h = fit.derivatives(value, wall)
    np.testing.assert_allclose(actual_g, exact_g, rtol=0, atol=3e-13)
    np.testing.assert_allclose(actual_h, np.broadcast_to(hessian[:, None], actual_h.shape), rtol=0, atol=3e-12)
    constant_u = np.broadcast_to(constant[:, None], value.shape)
    constant_b = np.broadcast_to(constant[:, None], wall.shape)
    for derivative in fit.derivatives(constant_u, constant_b):
        np.testing.assert_array_equal(derivative, 0)
    assert np.all(fit.boundary_observation_count > 0)


def test_boundary_fit_is_invariant_to_higher_quadrature_order_for_quadratic_traces():
    mesh, geometry, _, linear, constant, gradient, hessian = fixture()
    faces = np.arange(mesh["n_interior_faces"], mesh["n_faces"])
    # Inconsistent cell observations force a compromise; exact polynomial
    # recovery alone would not detect a sample-count-dependent wall weight.
    value = np.random.default_rng(131).normal(size=(2, mesh["n_cells"], 3))
    fitted = []
    for order in (3, 5):
        fit = BoundaryQuadraticCellLeastSquares.from_mesh(mesh, geometry["cell_centre"], linear.volume,
                                                          boundary_faces=faces, quadrature_order=order)
        boundary = polynomial(fit.boundary_position, constant, gradient, hessian)[0]
        fitted.append(fit.derivatives(value, boundary))
    for low, high in zip(*fitted, strict=True):
        np.testing.assert_allclose(low, high, rtol=0, atol=3e-12)


def test_boundary_data_reduces_independently_integrated_wall_mismatch_and_keeps_bulk_fit():
    mesh, geometry, native, linear, constant, gradient, hessian = fixture()
    face = mesh["n_interior_faces"]
    base = QuadraticCellLeastSquares.from_mesh(mesh, geometry["cell_centre"], linear.volume)
    fit = BoundaryQuadraticCellLeastSquares.from_mesh(mesh, geometry["cell_centre"], linear.volume, boundary_faces=[face])
    value = np.random.default_rng(9827).normal(size=(2, mesh["n_cells"], 3))
    boundary = polynomial(fit.boundary_position, constant, gradient, hessian)[0]
    old_g, old_h = base.derivatives(value)
    new_g, new_h = fit.derivatives(value, boundary)
    untouched = fit.boundary_observation_count == 0
    assert np.any(untouched)
    np.testing.assert_array_equal(new_g[:, untouched], old_g[:, untouched])
    np.testing.assert_array_equal(new_h[:, untouched], old_h[:, untouched])
    # Higher-order independent area integration of the one observed face.
    triangles = native.triangles[native.face_ids == face]
    barycentric, weight = triangle_rule(7)
    q = np.einsum("qi,tij->tqj", barycentric, triangles).reshape(-1, 3)
    area = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1)/2
    weights = (area[:, None]*weight).ravel()
    truth = polynomial(q, constant, gradient, hessian)[0]
    before, after = [], []
    for cell in np.flatnonzero(~untouched):
        d = q-geometry["cell_centre"][cell]
        for g, h, result in ((old_g, old_h, before), (new_g, new_h, after)):
            actual = value[:, cell, None]+np.einsum("qi,sij->sqj", d, g[:, cell])+.5*np.einsum("qi,qk,sikj->sqj", d, d, h[:, cell])
            result.append(np.einsum("q,sqj,sqj->s", weights, actual-truth, actual-truth))
    before, after = np.stack(before), np.stack(after)
    assert np.all(after <= before+1e-13)
    assert np.max(before-after) > .1


def test_boundary_observations_do_not_turn_a_two_dimensional_cell_stencil_into_a_3d_fit():
    mesh = box_mesh_3d(np.arange(5.), np.arange(5.), np.array([0., 1.]))
    geometry = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    with pytest.raises(ValueError, match="nine quadratic modes in three dimensions"):
        BoundaryQuadraticCellLeastSquares.from_mesh(mesh, geometry["cell_centre"], geometry["cell_volume"],
                                                    boundary_faces=np.arange(mesh["n_interior_faces"], mesh["n_faces"]))
