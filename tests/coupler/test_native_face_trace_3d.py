"""Qualify reference-face observations independently of the cropped solve."""

import numpy as np
import pytest

from source.solvers.fvm.assemble.diffusion import assemble_diffusion_term_interior
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.mesh.rectilinear import box_mesh_3d
from source.solvers.fvm.solve.simple_solver import (
    _compute_pressure_face_conductance,
    _pressure_interior_flux_scalar,
)
from studies.coupler_accuracy.native_face_trace_3d import NativeFaceTrace, tangential


def skew_mesh():
    mesh = box_mesh_3d(np.linspace(-1, 1, 5), np.linspace(-1, 1, 4), np.linspace(-1, 1, 3))
    transform = np.array([[1, 0.3, -0.2], [0.1, 1.2, 0.4], [0.2, -0.1, 0.8]])
    mesh["vertex_position"] = mesh["vertex_position"] @ transform.T
    geometry = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    faces = np.arange(mesh["n_interior_faces"])[::2]
    signs = np.where(np.arange(len(faces)) % 2, -1, 1)
    return mesh, geometry, faces, signs


def test_native_face_derivatives_reproduce_affine_3d_field_on_skew_mesh():
    mesh, geo, faces, signs = skew_mesh()
    trace = NativeFaceTrace(mesh, geo, faces, signs)
    jacobian = np.array([[0.2, -0.7, 1.3], [0.6, -0.1, -0.4], [-0.8, 1.2, -0.1]])
    gradp = np.array([0.7, -0.3, 0.9])
    x = geo["cell_centre"]
    velocity = x @ jacobian.T + [0.4, -0.1, 0.6]
    pressure = x @ gradp + 0.3
    result = trace.evaluate(velocity, np.tile(jacobian.T, (len(x), 1, 1)),
                            pressure, np.tile(gradp, (len(x), 1)))
    expected = trace.normal @ jacobian.T
    np.testing.assert_allclose(result["native_flux_velocity_normal_gradient"], expected,
                               rtol=0, atol=2e-14)
    np.testing.assert_allclose(result["native_flux_tangential_gradient"],
                               tangential(expected, trace.normal), rtol=0, atol=2e-14)
    # The pressure kernel adds 1e-12 to its edge length. Keep that actual
    # numerical trace instead of concealing it with a different formula.
    np.testing.assert_allclose(result["native_flux_pressure_normal_gradient"],
                               trace.normal @ gradp, rtol=0, atol=2e-11)
    np.testing.assert_allclose(result["native_face_velocity"],
                               geo["face_centre"][faces] @ jacobian.T + [0.4, -0.1, 0.6],
                               rtol=0, atol=2e-14)
    skew_term = tangential(trace.tangential_displacement @ jacobian.T, trace.normal)
    np.testing.assert_allclose(
        result["native_value_tangential_gradient"] - result["native_flux_tangential_gradient"],
        skew_term / trace.distance[:, None], rtol=0, atol=3e-14,
    )
    assert np.linalg.norm(skew_term) > 0.1


def test_selected_face_traces_match_full_operators_with_variable_scalar_coefficients():
    mesh, geo, faces, signs = skew_mesh()
    rng = np.random.default_rng(1708)
    n = mesh["n_cells"]
    velocity, gradient = rng.normal(size=(n, 3)), rng.normal(size=(n, 3, 3))
    pressure, gradp = rng.normal(size=n), rng.normal(size=(n, 3))
    coefficient = rng.uniform(0.1, 3, size=n)
    trace = NativeFaceTrace(mesh, geo, faces, signs)
    result = trace.evaluate(velocity, gradient, pressure, gradp)
    face_coefficient = trace.interpolate(coefficient)
    for component in range(3):
        flux = assemble_diffusion_term_interior(
            velocity[:, component], gradient[:, :, component], coefficient, mesh, geo,
        )["flux_tf"]
        np.testing.assert_allclose(
            result["native_flux_velocity_normal_gradient"][:, component],
            -flux[faces] * signs / (face_coefficient * trace.area), rtol=2e-14, atol=2e-14,
        )
    flux = np.empty(mesh["n_interior_faces"])
    _pressure_interior_flux_scalar(
        mesh["owners"], mesh["neighbours"], geo["face_interpolation_weight"],
        geo["face_area_vector"], geo["cell_connection_vector"], coefficient, np.zeros((n, 3)),
        gradp, pressure, _compute_pressure_face_conductance(mesh, geo, coefficient),
        np.empty(0), flux,
    )
    np.testing.assert_allclose(result["native_flux_pressure_normal_gradient"],
                               -flux[faces] * signs / (face_coefficient * trace.area),
                               rtol=2e-14, atol=2e-14)
    diffusion_p = assemble_diffusion_term_interior(pressure, gradp, 1.0, mesh, geo)["flux_tf"]
    assert np.linalg.norm(result["native_flux_pressure_normal_gradient"]
                          + diffusion_p[faces] * signs / trace.area) > 0.1


def test_native_face_trace_rejects_external_faces_and_invalid_orientation():
    mesh, geo, faces, signs = skew_mesh()
    with pytest.raises(ValueError, match="interior faces"):
        NativeFaceTrace(mesh, geo, [mesh["n_interior_faces"]], [1])
    with pytest.raises(ValueError, match="orientation"):
        NativeFaceTrace(mesh, geo, faces, signs * 0.5)
