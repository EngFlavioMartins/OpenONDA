"""Qualify the scoped mixed-outflow experiment before any advancing trial."""

import numpy as np
import pytest

from source.solvers.fvm.assemble import convection
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.mesh.rectilinear import box_mesh_3d
from studies.coupler_accuracy.experimental_mixed_convection import mixed_outflow_linear_upwind


def data():
    mesh = box_mesh_3d(np.linspace(-1, 1, 5), np.linspace(-1, 1, 4), np.linspace(-1, 1, 3))
    mesh["vertex_position"] = mesh["vertex_position"] @ np.array(
        [[1, 0.3, -0.2], [0.1, 1.2, 0.4], [0.2, -0.1, 0.8]]
    ).T
    geo = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    patch = dict(mesh["boundary"][0], velocity_type="normalValueTangentialGradient")
    faces = np.arange(patch["start_face"], patch["start_face"] + patch["n_faces"])
    rng = np.random.default_rng(531)
    n = mesh["n_cells"] + mesh["n_faces"] - mesh["n_interior_faces"]
    scalar = rng.normal(size=n)
    gradient = rng.normal(size=(n, 3))
    flux = rng.uniform(0.1, 0.5, size=mesh["n_faces"])
    flux[faces[::2]] *= -1
    return mesh, geo, patch, faces, scalar, gradient, flux


@pytest.mark.parametrize("total", [False, True])
def test_outflow_has_native_linear_upwind_flux_and_implicit_owner_coefficient(total):
    mesh, geo, patch, faces, scalar, gradient, flux = data()
    component = 1
    args = (scalar, flux, patch, mesh, geo, "linearUpwind", gradient)
    baseline = convection.assemble_convection_term_boundary(*args, component=component,
                                                            include_total_flux=total)
    with mixed_outflow_linear_upwind():
        result = convection.assemble_convection_term_boundary(*args, component=component,
                                                              include_total_flux=total)
        own = mesh["owners"][faces]
        outward = flux[faces] > 0
        correction = np.sum(gradient[own] * (geo["face_centre"][faces] - geo["cell_centre"][own]), axis=1)
        expected = flux[faces] * (scalar[own] + correction)
        assembled = result["flux_cf"] * scalar[own] + result["flux_vf"]
        np.testing.assert_allclose(assembled[outward], expected[outward], rtol=0, atol=2e-16)
        np.testing.assert_array_equal(result["flux_cf"][outward], flux[faces][outward])
        np.testing.assert_array_equal(result["flux_ff"], 0)
        for key in ("flux_cf", "flux_ff", "flux_vf", "flux_tf"):
            if key in result:
                np.testing.assert_array_equal(result[key][~outward], baseline[key][~outward])
        # Change the owner while freezing the deferred gradient. The assembled
        # response must agree with the implicit coefficient on outflow.
        scalar[:mesh["n_cells"]] += 0.2
        perturbed = convection.assemble_convection_term_boundary(*args, component=component)
        np.testing.assert_allclose(perturbed["flux_tf"][outward] - expected[outward],
                                   0.2 * result["flux_cf"][outward], rtol=0, atol=3e-16)


def test_outflow_extrapolation_reproduces_affine_3d_field_on_skew_faces():
    mesh, geo, patch, faces, scalar, gradient, flux = data()
    constant_gradient = np.array([0.7, -1.3, 0.4])
    scalar[:mesh["n_cells"]] = geo["cell_centre"] @ constant_gradient + 0.6
    gradient[:] = constant_gradient
    with mixed_outflow_linear_upwind():
        result = convection.assemble_convection_term_boundary(
            scalar, flux, patch, mesh, geo, "linearUpwind", gradient, component=2,
        )
    outward = flux[faces] > 0
    expected = flux[faces] * (geo["face_centre"][faces] @ constant_gradient + 0.6)
    np.testing.assert_allclose(result["flux_tf"][outward], expected[outward], rtol=0, atol=3e-16)


def test_scope_preserves_other_schemes_and_restores_assembler_after_exception():
    mesh, geo, patch, _, scalar, gradient, flux = data()
    original = convection.assemble_convection_term_boundary
    args = (scalar, flux, patch, mesh, geo, "upwind", gradient)
    expected = original(*args, component=0)
    with pytest.raises(RuntimeError, match="trial failure"), mixed_outflow_linear_upwind():
        actual = convection.assemble_convection_term_boundary(*args, component=0)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])
        raise RuntimeError("trial failure")
    assert convection.assemble_convection_term_boundary is original
