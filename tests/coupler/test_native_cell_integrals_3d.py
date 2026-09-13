"""Analytic geometric and Gaussian checks for native-volume observations."""

import numpy as np
import pytest
from scipy.special import erf

from source.coupler.renewal_projection import gaussian_vorticity_basis
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.mesh.rectilinear import box_mesh_3d
from studies.coupler_accuracy.cell_integral_region_audit_3d import gaussian_surface_flux
from studies.coupler_accuracy.native_cell_integrals_3d import (
    NativeCellIntegration,
    gaussian_cell_integrals,
)


def make_integration(mesh):
    geometry = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    return NativeCellIntegration.from_mesh(mesh, geometry, np.arange(mesh["n_cells"]))


def test_tetra_quadrature_integrates_affine_box_volume_centroid_and_covariance():
    mesh = box_mesh_3d(np.array([-0.2, 0.2]), np.array([-0.3, 0.3]), np.array([-0.4, 0.4]))
    transform = np.array([[1, 0.3, -0.2], [0.1, 1.2, 0.4], [0.2, -0.1, 0.8]])
    offset = np.array([0.7, -0.8, 0.6])
    mesh["vertex_position"] = mesh["vertex_position"] @ transform.T + offset
    integration = make_integration(mesh)
    points, weights = integration.rule(0, 2)
    volume = np.linalg.det(transform) * 0.4 * 0.6 * 0.8
    covariance = transform @ np.diag(np.array([0.4, 0.6, 0.8])**2 / 12) @ transform.T
    np.testing.assert_allclose(weights.sum(), volume, rtol=0, atol=2e-16)
    np.testing.assert_allclose(weights @ points / volume, offset, rtol=0, atol=3e-15)
    np.testing.assert_allclose(np.einsum("q,qi,qj->ij", weights, points-offset, points-offset) / volume,
                               covariance, rtol=0, atol=2e-15)


def test_warped_face_fan_volume_is_kept_separate_from_fvm_aggregate_volume():
    mesh = box_mesh_3d(np.array([0., 1.]), np.array([0., 1.]), np.array([0., 1.]))
    left = np.flatnonzero(np.all(mesh["vertex_position"] == [0, 0, 1], axis=1))[0]
    right = np.flatnonzero(np.all(mesh["vertex_position"] == [1, 0, 1], axis=1))[0]
    extra = len(mesh["vertex_position"])
    mesh["vertex_position"] = np.vstack((mesh["vertex_position"], [0.5, 0, 1.3]))
    mesh["faces"] = [list(face) for face in mesh["faces"]]
    for face in mesh["faces"]:
        for i in range(len(face)):
            if {face[i], face[(i+1) % len(face)]} == {left, right}:
                face.insert(i+1, extra)
                break
    mesh["faces"] = [np.asarray(face, dtype=int) for face in mesh["faces"]]
    integration = make_integration(mesh)
    # A pentagonal roof fan has XY triangle areas .1, .1, .25, .3, .25.
    # Its centre height is 1+0.3/5; the raised edge point belongs to the
    # first two triangles. Integrating their affine roofs gives 1+2*0.3/15.
    np.testing.assert_allclose(integration.polyhedron_volume, [1 + 2*0.3/15], rtol=0, atol=3e-15)
    assert abs(integration.polyhedron_volume[0] - integration.fvm_volume[0]) > 1e-5
    points, weights = integration.rule(0, 3)
    np.testing.assert_allclose(weights.sum(), integration.polyhedron_volume[0], rtol=0, atol=3e-15)
    assert len(points) == len(weights)


@pytest.mark.parametrize("rotate_and_translate", [False, True])
def test_gaussian_integrals_match_independent_erf_box_integral(rotate_and_translate):
    bounds = np.array([[-0.1, 0.1], [-0.3, 0.3], [-0.2, 0.2]])
    mesh = box_mesh_3d(*bounds)
    positions = np.array([[0, 0, 0], [0.1, -0.2, 0.1], [0.3, 0.1, -0.1]])
    radius = np.array([0.12, 0.16, 0.2])
    expected = np.prod((erf((bounds[:, 1] - positions) / radius[:, None])
                        - erf((bounds[:, 0] - positions) / radius[:, None])) / 2, axis=1)
    if rotate_and_translate:
        rotation, _ = np.linalg.qr(np.array([[1., 2., -1.], [0.2, 1., 2.], [0.8, -1., 1.]]))
        assert np.linalg.det(rotation) > 0
        offset = np.array([3.2, -4.3, 1.7])
        mesh["vertex_position"] = mesh["vertex_position"] @ rotation.T + offset
        positions = positions @ rotation.T + offset
    integration = make_integration(mesh)
    result, diagnostic = gaussian_cell_integrals(integration, positions, radius, basis_tolerance=1e-9)
    np.testing.assert_allclose(result.toarray()[0], expected, rtol=0, atol=5e-12)
    for axis in range(3):
        surface = gaussian_surface_flux(integration.tetrahedra[0][:, 1:], positions, radius,
                                        order=16, axis=axis)
        np.testing.assert_allclose(surface, expected, rtol=0, atol=5e-12)
    assert max(diagnostic["successive_basis_change_over_fvm_volume"]) <= 1e-9


def test_gaussian_integrals_add_across_shared_native_faces():
    bounds = (np.array([-0.3, -0.1, 0.1, 0.3]), np.array([-0.2, 0., 0.2]),
              np.array([-0.25, 0.05, 0.25]))
    mesh = box_mesh_3d(*bounds)
    geometry = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    ids = np.random.default_rng(157).permutation(mesh["n_cells"])
    integration = NativeCellIntegration.from_mesh(mesh, geometry, ids)
    positions = np.array([[0.1, 0, 0.05], [-0.15, -0.05, 0.1], [0.4, -0.2, 0.3]])
    radius = np.array([0.15, 0.2, 0.25])
    result, _ = gaussian_cell_integrals(integration, positions, radius, basis_tolerance=1e-9)
    expected = np.prod((erf((integration.upper[:, None, :] - positions) / radius[None, :, None])
                        - erf((integration.lower[:, None, :] - positions) / radius[None, :, None])) / 2, axis=2)
    np.testing.assert_allclose(result.toarray(), expected, rtol=0, atol=5e-12)
    low, high = np.array([b[0] for b in bounds]), np.array([b[-1] for b in bounds])
    whole = np.prod((erf((high - positions) / radius[:, None])
                     - erf((low - positions) / radius[:, None])) / 2, axis=1)
    np.testing.assert_allclose(np.asarray(result.sum(axis=0)).ravel(), whole, rtol=0, atol=2e-11)


def test_integrated_uniform_lattice_reproduces_cell_circulation_without_filter_normalization():
    mesh = box_mesh_3d(*[np.array([-0.5, 0.5])] * 3)
    integration = make_integration(mesh)
    positions = np.stack(np.meshgrid(*[np.arange(-6., 7.)] * 3, indexing="ij"), axis=-1).reshape(-1, 3)
    result, _ = gaussian_cell_integrals(integration, positions, 1.0, basis_tolerance=1e-10)
    np.testing.assert_allclose(result.sum(), 1, rtol=0, atol=2e-11)
    point = gaussian_vorticity_basis(np.zeros((1, 3)), positions, 1.0).sum()
    assert point - 1 > 3e-4
