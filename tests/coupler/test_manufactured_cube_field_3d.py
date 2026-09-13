"""Qualify the no-slip 3D manufactured field independently of cell induction."""

import numpy as np
import pytest

from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.mesh.rectilinear import box_mesh_3d
from studies.coupler_accuracy.manufactured_cube_field_3d import (
    NoSlipCubeField,
    native_manufactured_integrals,
)
from studies.coupler_accuracy.native_cell_integrals_3d import NativeCellIntegration
from studies.coupler_accuracy.native_volume_induction_3d import NativeVolumeSources


@pytest.mark.parametrize("width", [0.15, 0.35])
def test_analytic_field_has_no_slip_and_matches_complex_step_curl_in_three_dimensions(width):
    field = NoSlipCubeField(width)
    generator = np.random.default_rng(1911)
    wall = generator.uniform(-0.5, 0.5, (6, 32, 3))
    for side in range(6):
        wall[side, :, side//2] = -0.5 if side % 2 == 0 else 0.5
    np.testing.assert_array_equal(field.velocity(wall), 0)
    np.testing.assert_allclose(np.linalg.norm(field.velocity(field.reference_position[None])), 1, rtol=0, atol=3e-16)
    points = generator.uniform(-0.8, 0.8, (128, 3))
    points = points[np.max(np.abs(points), axis=1) > 0.5]
    derivative_u, derivative_omega = [], []
    for axis in range(3):
        z = points.astype(complex)+1e-30j*np.eye(3)[axis]
        derivative_u.append(field.velocity(z).imag/1e-30)
        derivative_omega.append(field.vorticity(z).imag/1e-30)
    grad_u = np.stack(derivative_u, axis=1)
    grad_omega = np.stack(derivative_omega, axis=1)
    curl = np.column_stack((grad_u[:, 1, 2]-grad_u[:, 2, 1], grad_u[:, 2, 0]-grad_u[:, 0, 2],
                            grad_u[:, 0, 1]-grad_u[:, 1, 0]))
    np.testing.assert_allclose(curl, field.vorticity(points), rtol=3e-12, atol=2e-12)
    np.testing.assert_allclose(np.trace(grad_u, axis1=1, axis2=2), 0, rtol=0, atol=2e-12)
    np.testing.assert_allclose(np.trace(grad_omega, axis1=1, axis2=2), 0, rtol=0, atol=3e-10)
    assert np.all(np.max(np.abs(field.velocity(points)), axis=0) > 1e-3)
    assert np.all(np.max(np.abs(grad_u), axis=(0, 2)) > 1e-2)
    assert field.outer_box_velocity_bound([-5, 10, -5, 5, -5, 5]) < 1e-30


def test_native_surface_integrals_match_independent_volume_rule_on_oblique_cells():
    mesh = box_mesh_3d(np.array([0.48, 0.61, 0.76]), np.array([-0.11, 0.1]), np.array([-0.14, 0.13]))
    transform = np.array([[1., 0.2, -0.1], [0.1, 1., 0.2], [-0.15, 0.1, 1.]])
    mesh["vertex_position"] = mesh["vertex_position"] @ transform.T
    geometry = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    integration = NativeCellIntegration.from_mesh(mesh, geometry, np.arange(mesh["n_cells"]))
    native = NativeVolumeSources.from_mesh(mesh)
    fields = [NoSlipCubeField(0.15), NoSlipCubeField(0.35)]
    actual = native_manufactured_integrals(native, mesh, fields, order=18)
    for cell in range(mesh["n_cells"]):
        points, weight = integration.rule(cell, 18)
        for index, field in enumerate(fields):
            for key, function in (("velocity_integral", field.velocity), ("vorticity_integral", field.vorticity)):
                expected = weight @ function(points)
                np.testing.assert_allclose(actual[key][index, cell], expected, rtol=0, atol=5e-13)


def test_face_quadrature_preserves_exact_zero_boundary_values_on_the_cube():
    mesh = box_mesh_3d(*[np.array([-0.5, 0.5])]*3)
    native = NativeVolumeSources.from_mesh(mesh)
    result = native_manufactured_integrals(native, mesh, [NoSlipCubeField(0.15), NoSlipCubeField(0.35)], order=8)
    for values in result.values():
        np.testing.assert_array_equal(values, 0)
