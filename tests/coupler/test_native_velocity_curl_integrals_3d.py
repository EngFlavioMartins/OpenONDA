"""Continuous-curl Stokes observations, independently checked in 3D volumes."""

import numpy as np

from source.coupler.renewal_projection import gaussian_velocity_operator
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.mesh.rectilinear import box_mesh_3d
from studies.coupler_accuracy.joint_reconstruction_3d import CubePanelResponse
from studies.coupler_accuracy.native_cell_integrals_3d import (
    NativeCellIntegration,
    gaussian_cell_integrals,
)
from studies.coupler_accuracy.native_velocity_curl_integrals_3d import (
    gaussian_radial_velocity_factor,
    gaussian_velocity_curl_volume_integral,
    native_velocity_curl_cell_integrals,
)


def test_radial_factor_matches_existing_velocity_near_self_and_in_algebraic_tail():
    distances = np.r_[0., np.logspace(-10, 3, 31)]
    targets = np.column_stack((distances, np.zeros((len(distances), 2))))
    factor = gaussian_radial_velocity_factor(distances[:, None]**2, np.array([0.2]))[:, 0]
    velocity = gaussian_velocity_operator(targets, np.zeros((1, 3)), 0.2) @ np.array([0., 0., 1.])
    np.testing.assert_allclose(velocity.reshape(-1, 3)[:, 1], distances*factor, rtol=2e-12, atol=2e-15)
    np.testing.assert_allclose(factor[0], 1/(3*np.pi**1.5*0.2**3), rtol=0, atol=2e-15)
    assert factor[-1] > 0


def test_stokes_curl_has_raw_gaussian_trace_and_reproduces_symmetric_cube_integral():
    mesh = box_mesh_3d(*[np.array([-0.25, 0.25])] * 3)
    geo = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    integration = NativeCellIntegration.from_mesh(mesh, geo, [0])
    position, radius = np.zeros((1, 3)), np.array([0.2])
    curl, _ = native_velocity_curl_cell_integrals(mesh, geo, [0], position, radius)
    raw, _ = gaussian_cell_integrals(integration, position, radius)
    np.testing.assert_allclose(curl, (2/3)*raw[0, 0]*np.eye(3), rtol=0, atol=5e-13)
    assert np.linalg.norm(curl-raw[0, 0]*np.eye(3)) > 0.1


def test_stokes_matches_volume_curl_for_all_components_on_sheared_partition():
    mesh = box_mesh_3d(np.array([-0.3, 0., 0.3]), np.array([-0.2, 0.2]), np.array([-0.25, 0.25]))
    transform = np.array([[1., 0.3, -0.2], [0.1, 1.2, 0.4], [0.2, -0.1, 0.8]])
    mesh["vertex_position"] = mesh["vertex_position"] @ transform.T + [0.7, -0.8, 0.6]
    geo = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    ids = np.array([1, 0])
    integration = NativeCellIntegration.from_mesh(mesh, geo, ids)
    positions = np.array([[0.7, -0.8, 0.6], [0.9, -0.7, 0.5], [8.1, -1.2, 2.3]])
    radius = np.array([0.2, 0.3, 0.1])
    curl, diagnostics = native_velocity_curl_cell_integrals(mesh, geo, ids, positions, radius)
    raw, _ = gaussian_cell_integrals(integration, positions, radius)
    for cell in range(len(ids)):
        volume = gaussian_velocity_curl_volume_integral(integration, cell, positions, radius, order=14)
        actual = curl[3*cell:3*cell+3]
        np.testing.assert_allclose(actual, volume, rtol=0, atol=2e-12)
        blocks = actual.reshape(3, len(positions), 3).transpose(1, 0, 2)
        np.testing.assert_allclose(blocks, blocks.transpose(0, 2, 1), rtol=0, atol=2e-12)
        np.testing.assert_allclose(np.trace(blocks, axis1=1, axis2=2), 2*raw.toarray()[cell], rtol=0, atol=3e-12)
        assert np.linalg.norm(blocks[-1]) > 1e-6
        assert raw[cell, -1] == 0
    assert max(diagnostics["cell_summed_remainder_over_fvm_volume"]) <= 1e-9


def test_actual_cube_source_panel_response_has_zero_curl_in_fluid():
    body = CubePanelResponse()
    position = np.array([[0.9, 0.4, 0.2], [-0.8, -0.2, 0.7], [0.3, 0.8, -0.7]])
    strength = np.array([[0.02, -0.04, 0.03], [-0.01, 0.025, -0.015], [0.03, 0.015, 0.04]])
    incident = (gaussian_velocity_operator(body.centres, position, 0.2) @ strength.ravel()).reshape(-1, 3)
    body.panel.solve(np.array([1., 0., 0.]), incident, time=0.5)
    points = np.array([[0.55, 0.2, 0.1], [1.4, 0.4, 0.3], [-0.8, 0.7, -0.3],
                       [0.1, 0.6, -0.4], [0.1, -0.4, -0.6]])
    derivatives = []
    step = 1e-4
    for axis in range(3):
        offset = np.eye(3)[axis] * step
        velocity = body.panel.compute_induced_velocity
        derivatives.append((velocity(points-2*offset)-8*velocity(points-offset)
                            + 8*velocity(points+offset)-velocity(points+2*offset))/(12*step))
    curl = np.column_stack((derivatives[1][:, 2]-derivatives[2][:, 1],
                            derivatives[2][:, 0]-derivatives[0][:, 2],
                            derivatives[0][:, 1]-derivatives[1][:, 0]))
    np.testing.assert_allclose(curl, 0, rtol=0, atol=2e-9)
