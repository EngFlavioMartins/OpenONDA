"""Qualify P1 source geometry, moments and singular induction independently."""

import copy

import numpy as np
import pytest

from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.mesh.rectilinear import box_mesh_3d
from studies.coupler_accuracy.manufactured_cell_moments_3d import native_first_vorticity_moment
from studies.coupler_accuracy.manufactured_cube_field_3d import NoSlipCubeField, native_manufactured_integrals
from studies.coupler_accuracy.native_cell_integrals_3d import NativeCellIntegration
from studies.coupler_accuracy.native_linear_volume_induction_3d import LinearNativeVolumeSources, triangle_potential_moments
from studies.coupler_accuracy.native_velocity_curl_integrals_3d import triangle_rule
from studies.coupler_accuracy.native_volume_induction_3d import NativeVolumeSources, triangle_source_integrals


def oblique_mesh():
    mesh = box_mesh_3d(np.array([.35, .5, .75]), np.array([-.2, .25]), np.array([-.1, .2]))
    matrix = np.array([[1., .1, -.2], [.2, 1., .1], [-.1, .15, 1.]])
    mesh["vertex_position"] = mesh["vertex_position"] @ matrix.T
    # A displacement of one vertex creates nonplanar quadrilateral faces.
    mesh["vertex_position"][0] += [.015, -.008, .012]
    return mesh


def setup(mesh):
    geometry = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    native = NativeVolumeSources.from_mesh(mesh)
    linear = LinearNativeVolumeSources.from_native(native, geometry["cell_centre"])
    integration = NativeCellIntegration.from_mesh(mesh, geometry, np.arange(mesh["n_cells"]))
    return geometry, native, linear, integration


def test_triangle_first_potential_moment_matches_quadrature_and_orientation():
    triangle = np.array([[[-.4, -.2, .1], [.7, -.3, .4], [.1, .8, -.2]]])
    targets = np.array([[.2, .1, .8], [-.3, .2, -.9], [7.1, 2.3, -1.7], [100., 70., -30.]])
    p, moment = triangle_potential_moments(targets, triangle)
    np.testing.assert_allclose(p, triangle_source_integrals(targets, triangle)[0], rtol=0, atol=2e-15)
    barycentric, weight = triangle_rule(48)
    q = barycentric @ triangle[0]
    area = np.linalg.norm(np.cross(triangle[0, 1]-triangle[0, 0], triangle[0, 2]-triangle[0, 0]))/2
    expected = area*np.einsum("q,qd,pq->pd", weight, q-triangle[0].mean(axis=0), 1/np.linalg.norm(targets[:, None]-q, axis=2))
    np.testing.assert_allclose(moment[:, 0], expected, rtol=0, atol=7e-14)
    flipped = triangle_potential_moments(targets, triangle[:, [0, 2, 1]])
    for a, b in zip((p, moment), flipped, strict=True):
        np.testing.assert_allclose(a, b, rtol=0, atol=7e-14)


@pytest.mark.parametrize("point", [[0., 0., 0.], [.5, 0., 0.], [.2, .3, 0.]])
def test_triangle_first_potential_moment_at_vertex_edge_and_interior(point):
    triangle = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
    point, centre = np.asarray(point), triangle[0].mean(axis=0)
    node, weight = np.polynomial.legendre.leggauss(128)
    s, weight = (node+1)/2, weight/2
    expected = np.zeros(3)
    for a, b in zip(triangle[0], np.roll(triangle[0], -1, axis=0), strict=True):
        a, b = a-point, b-point
        cross = np.linalg.norm(np.cross(a, b))
        if cross > 0:
            q = (1-s[:, None])*a+s[:, None]*b
            expected += cross*np.einsum("q,qi->i", weight/np.linalg.norm(q, axis=1), point-centre+.5*q)
    actual = triangle_potential_moments(point[None], triangle)[1][0, 0]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-14)


def test_affine_volume_matches_independent_warped_cell_quadrature_inside_and_outside():
    mesh = oblique_mesh()
    _, native, linear, integration = setup(mesh)
    gradient = np.array([[[.3, -.2, .7], [.5, -.8, .1], [-.4, .9, 1.3]],
                         [[-.2, .4, .7], [-.1, .3, .2], [.7, -.2, .5]]])
    omega = np.array([[.3, -.7, 1.2], [-.2, .5, .7]])
    moment = linear.covariance @ gradient
    coefficients, recovered = linear.coefficients(omega*linear.volume[:, None], moment)
    np.testing.assert_allclose(recovered[0], gradient, rtol=0, atol=2e-15)
    targets = np.array([[1.2, .3, .4], [.4, -.8, .6], [-.7, -.5, -.9]])
    expected = np.zeros_like(targets)
    for cell in range(mesh["n_cells"]):
        q, weight = integration.rule(cell, 20)
        relative = q-linear.centroid[cell]
        covariance = np.einsum("q,qi,qj->ij", weight, relative, relative)
        np.testing.assert_allclose(covariance, linear.covariance[cell], rtol=2e-13, atol=1e-18)
        density = omega[cell]+relative @ gradient[cell]
        delta = targets[:, None]-q
        expected += np.einsum("q,pqd->pd", weight, np.cross(density[None], delta)/(4*np.pi*np.linalg.norm(delta, axis=2)[:, :, None]**3))
    actual = linear.evaluate(targets, coefficients)[:, 0]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-15)
    # Target-apex radial integration is exact for the affine source. The
    # independent angular quadrature includes the integrable interior target.
    target = linear.centroid[0]
    barycentric, weight = triangle_rule(48)
    expected = np.zeros(3)
    for cell in range(mesh["n_cells"]):
        faces = [(native.owners == cell, 1), (native.neighbours == cell, -1)]
        for selected, sign in faces:
            relative = native.triangles[selected]-target
            q = np.einsum("qa,tad->tqd", barycentric, relative)
            determinant = sign*np.einsum("ti,ti->t", relative[:, 0], np.cross(relative[:, 1], relative[:, 2]))
            at_target = omega[cell]+(target-linear.centroid[cell]) @ gradient[cell]
            numerator = np.cross(at_target, q)+.5*np.cross(q @ gradient[cell], q)
            expected -= np.einsum("t,q,tqd->d", determinant/2, weight, numerator/np.linalg.norm(q, axis=2)[:, :, None]**3)/(4*np.pi)
    actual = linear.evaluate(target[None], coefficients)[0, 0]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=5e-14)
    zero = np.zeros_like(moment)
    np.testing.assert_allclose(linear.evaluate(targets, linear.coefficients(omega*linear.volume[:, None], zero)[0])[:, 0],
                               native.evaluate(targets, native.coefficients(omega))[:, 0], rtol=0, atol=2e-16)


def test_affine_source_shared_face_additivity_and_rigid_motion():
    mesh = box_mesh_3d(np.array([-.4, 0., .4]), np.array([-.3, 0., .3]), np.array([-.2, .2]))
    whole = box_mesh_3d(np.array([-.4, .4]), np.array([-.3, .3]), np.array([-.2, .2]))
    _, _, linear, _ = setup(mesh)
    _, _, one, _ = setup(whole)
    omega = np.array([.3, -.7, 1.2])
    gradient = np.array([[.3, -.2, .7], [.5, -.8, .1], [-.4, .9, 1.3]])
    targets = np.array([[0., 0., 0.], [.1, -.1, .05], [1.1, -.2, .7], [2., .3, -.1]])
    coefficient = linear.coefficients((omega+linear.centroid @ gradient)*linear.volume[:, None], linear.covariance @ gradient)[0]
    expected = linear.evaluate(targets, coefficient)[:, 0]
    coefficient_one = one.coefficients((omega+one.centroid @ gradient)*one.volume[:, None], one.covariance @ gradient)[0]
    np.testing.assert_allclose(expected, one.evaluate(targets, coefficient_one)[:, 0], rtol=0, atol=3e-15)
    rotation, _ = np.linalg.qr(np.array([[1., 2., -1.], [.2, 1., 2.], [.8, -1., 1.]]))
    translation = np.array([3.2, -4.3, 1.7])
    moved_mesh = copy.deepcopy(mesh)
    moved_mesh["vertex_position"] = mesh["vertex_position"] @ rotation.T+translation
    _, _, moved, _ = setup(moved_mesh)
    moved_omega = (omega+linear.centroid @ gradient) @ rotation.T
    coefficient_moved = moved.coefficients(moved_omega*moved.volume[:, None], moved.covariance @ (rotation @ gradient @ rotation.T))[0]
    actual = moved.evaluate(targets @ rotation.T+translation, coefficient_moved)[:, 0]
    np.testing.assert_allclose(actual, expected @ rotation.T, rtol=0, atol=3e-15)


def test_surface_first_vorticity_moment_matches_direct_volume_in_3d():
    mesh = oblique_mesh()
    _, native, linear, integration = setup(mesh)
    fields = [NoSlipCubeField(.35), NoSlipCubeField(.15)]
    integral = native_manufactured_integrals(native, mesh, fields, order=18)
    actual = native_first_vorticity_moment(native, linear.centroid, fields, integral["velocity_integral"], order=18)
    for cell in range(mesh["n_cells"]):
        q, weight = integration.rule(cell, 22)
        for index, field in enumerate(fields):
            expected = np.einsum("q,qi,qj->ij", weight, q-linear.centroid[cell], field.vorticity(q))
            np.testing.assert_allclose(actual[index, cell], expected, rtol=2e-13, atol=2e-14)
