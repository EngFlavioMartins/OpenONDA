"""Independent surface, volume and bounded-Helmholtz checks in full 3D."""

import numpy as np
import pytest

from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.mesh.rectilinear import box_mesh_3d
from studies.coupler_accuracy.native_cell_integrals_3d import NativeCellIntegration
from studies.coupler_accuracy.native_velocity_curl_integrals_3d import triangle_rule
from studies.coupler_accuracy.native_volume_induction_3d import (
    NativeVolumeSources,
    piecewise_constant_boundary_completion,
    triangle_source_integrals,
)


def surface_quadrature(points, triangles, order):
    barycentric, weight = triangle_rule(order)
    locations = np.einsum("qa,tad->tqd", barycentric, triangles)
    area = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                   triangles[:, 2]-triangles[:, 0]), axis=1)/2
    delta = points[:, None, None]-locations
    radius = np.linalg.norm(delta, axis=3)
    scalar = np.einsum("t,q,ptq->pt", area, weight, 1/radius)
    vector = np.einsum("t,q,ptqd->ptd", area, weight, delta/radius[:, :, :, None]**3)
    return scalar, vector


def test_triangle_integrals_match_independent_quadrature_orientation_and_gradient():
    triangle = np.array([[[-0.4, -0.2, 0.1], [0.7, -0.3, 0.4], [0.1, 0.8, -0.2]]])
    points = np.array([[0.2, 0.1, 0.8], [-0.3, 0.2, -0.9], [7.1, 2.3, -1.7]])
    actual = triangle_source_integrals(points, triangle)
    expected = surface_quadrature(points, triangle, 40)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a, b, rtol=2e-13, atol=5e-15)
    flipped = triangle_source_integrals(points, triangle[:, [0, 2, 1]])
    for a, b in zip(actual, flipped, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-13, atol=2e-15)
    step = 2e-4
    gradient = []
    for axis in range(3):
        offset = step*np.eye(3)[axis]
        p = lambda x: triangle_source_integrals(x, triangle)[0]
        gradient.append((p(points-2*offset)-8*p(points-offset)
                         + 8*p(points+offset)-p(points+2*offset))/(12*step))
    np.testing.assert_allclose(-np.stack(gradient, axis=2), actual[1], rtol=2e-9, atol=2e-11)


@pytest.mark.parametrize("point", [[0., 0., 0.], [0.5, 0., 0.], [0.2, 0.3, 0.]])
def test_triangle_potential_is_finite_at_vertices_edges_and_interior(point):
    triangle = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
    point = np.asarray(point)
    # Split around the target and integrate the singular radial coordinate
    # exactly. The remaining independent line integral is smooth.
    nodes, weights = np.polynomial.legendre.leggauss(128)
    s, w = (nodes+1)/2, weights/2
    expected = 0.
    for a, b in zip(triangle[0], np.roll(triangle[0], -1, axis=0), strict=True):
        a, b = a-point, b-point
        cross = np.linalg.norm(np.cross(a, b))
        if cross > 0:
            expected += cross*np.sum(w/np.linalg.norm((1-s[:, None])*a+s[:, None]*b, axis=1))
    actual = triangle_source_integrals(point[None], triangle)[0][0, 0]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-14)
    for distance in (1e-6, 1e-10, 1e-14):
        for sign in (-1, 1):
            near = triangle_source_integrals((point+[0, 0, sign*distance])[None], triangle)[0][0, 0]
            assert np.isfinite(near)
            assert abs(near-actual) < 7*distance+2e-14


def warped_box():
    mesh = box_mesh_3d(*[np.array([0., 1.])]*3)
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
    return mesh


def test_native_volume_induction_matches_volume_rule_on_warped_polyhedron():
    mesh = warped_box()
    geometry = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    integration = NativeCellIntegration.from_mesh(mesh, geometry, [0])
    sources = NativeVolumeSources.from_mesh(mesh)
    points = np.array([[1.8, 0.3, 0.4], [0.4, -0.8, 1.6], [-0.7, -0.5, -0.9]])
    omega = np.array([[0.3, -0.7, 1.2]])
    divergence = np.array([0.4])
    actual = sources.evaluate(points, sources.coefficients(omega, divergence))[:, 0]
    locations, weights = integration.rule(0, 24)
    delta = points[:, None]-locations
    expected = np.einsum("q,pqd->pd", weights,
                        (np.cross(omega[0], delta)+divergence[0]*delta)
                        /(4*np.pi*np.linalg.norm(delta, axis=2)[:, :, None]**3))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-14)
    # Interior singularity: target-apex radial integration is exact; integrate
    # only the smooth angular coordinates independently on each face triangle.
    target = geometry["cell_centre"][0]
    barycentric, weight = triangle_rule(32)
    relative = sources.triangles-target
    q = np.einsum("qa,tad->tqd", barycentric, relative)
    determinant = np.einsum("ti,ti->t", relative[:, 0], np.cross(relative[:, 1], relative[:, 2]))
    vector = -np.einsum("t,q,tqd->d", determinant/2, weight,
                        q/np.linalg.norm(q, axis=2)[:, :, None]**3)/(4*np.pi)
    expected = np.cross(omega[0], vector)+divergence[0]*vector
    actual = sources.evaluate(target[None], sources.coefficients(omega, divergence))[0, 0]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-14)


def test_shared_face_partition_additivity_rigid_motion_and_far_field():
    axes = (np.array([-0.4, 0., 0.4]), np.array([-0.3, 0., 0.3]), np.array([-0.2, 0.2]))
    mesh = box_mesh_3d(*axes)
    whole = box_mesh_3d(*[x[[0, -1]] for x in axes])
    sources, one = NativeVolumeSources.from_mesh(mesh), NativeVolumeSources.from_mesh(whole)
    omega = np.array([[0.3, -0.7, 1.2]])
    points = np.array([[0., 0., 0.], [0.1, -0.1, 0.05], [1.1, -0.2, 0.7], [2., 0.3, -0.1]])
    result = sources.evaluate(points, sources.coefficients(np.repeat(omega, mesh["n_cells"], axis=0)))[:, 0]
    expected = one.evaluate(points, one.coefficients(omega))[:, 0]
    np.testing.assert_allclose(result, expected, rtol=0, atol=3e-15)
    np.testing.assert_allclose(result[0], 0, rtol=0, atol=2e-16)
    rotation, _ = np.linalg.qr(np.array([[1., 2., -1.], [0.2, 1., 2.], [0.8, -1., 1.]]))
    offset = np.array([3.2, -4.3, 1.7])
    whole["vertex_position"] = whole["vertex_position"] @ rotation.T+offset
    moved = NativeVolumeSources.from_mesh(whole)
    actual = moved.evaluate(points @ rotation.T+offset, moved.coefficients(omega @ rotation.T))[:, 0]
    np.testing.assert_allclose(actual, expected @ rotation.T, rtol=0, atol=2e-15)
    far = np.array([[100., 70., -30.]])
    actual = one.evaluate(far, one.coefficients(omega))[:, 0]
    leading = np.cross(omega*0.8*0.6*0.4, far)/(4*np.pi*np.linalg.norm(far)**3)
    np.testing.assert_allclose(actual, leading, rtol=3e-5, atol=0)


def test_bounded_helmholtz_recovers_affine_velocity_and_divergence_sign():
    mesh = box_mesh_3d(*[np.array([-0.5, 0.5])]*3)
    sources = NativeVolumeSources.from_mesh(mesh)
    points = np.array([[0., 0., 0.], [0.2, -0.1, 0.3], [-0.25, 0.2, -0.1]])
    constant = np.array([0.4, -0.7, 1.2])
    actual = piecewise_constant_boundary_completion(points, sources.triangles, constant)
    np.testing.assert_allclose(actual, np.broadcast_to(constant, actual.shape), rtol=0, atol=3e-15)
    matrix = np.array([[0.3, -0.2, 0.7], [0.5, -0.8, 0.1], [-0.4, 0.9, 1.3]])
    omega = np.array([[matrix[2, 1]-matrix[1, 2], matrix[0, 2]-matrix[2, 0], matrix[1, 0]-matrix[0, 1]]])
    volume = sources.evaluate(points, sources.coefficients(omega, np.array([np.trace(matrix)])))[:, 0]
    barycentric, weight = triangle_rule(48)
    q = np.einsum("qa,tad->tqd", barycentric, sources.triangles)
    u = q @ matrix.T+constant
    vector_area = np.cross(sources.triangles[:, 1]-sources.triangles[:, 0],
                            sources.triangles[:, 2]-sources.triangles[:, 0])/2
    delta = points[:, None, None]-q
    kernel = delta/(4*np.pi*np.linalg.norm(delta, axis=3)[:, :, :, None]**3)
    normal_dot = np.einsum("ti,tqi->tq", vector_area, u)
    boundary = -np.einsum("q,ptqd->pd", weight, normal_dot[None, :, :, None]*kernel
                           + np.cross(np.cross(vector_area[:, None], u)[None], kernel))
    np.testing.assert_allclose(volume+boundary, points @ matrix.T+constant, rtol=0, atol=3e-13)
