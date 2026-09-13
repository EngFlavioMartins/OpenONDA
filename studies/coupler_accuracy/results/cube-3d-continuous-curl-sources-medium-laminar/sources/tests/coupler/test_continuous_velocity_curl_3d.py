"""Qualify conforming 3D curl, cell integrals, moments and induced physical curl."""

import numpy as np
import pytest

from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.mesh.rectilinear import box_mesh_3d
from studies.coupler_accuracy.continuous_velocity_curl_3d import (
    ContinuousVelocityCurl,
    compact_polynomial_trace,
)
from studies.coupler_accuracy.native_linear_volume_induction_3d import LinearNativeVolumeSources
from studies.coupler_accuracy.native_shared_trace_update_3d import global_first_moment, impulse
from studies.coupler_accuracy.native_velocity_curl_integrals_3d import triangle_rule
from studies.coupler_accuracy.native_volume_induction_3d import NativeVolumeSources


def fixture(cells=3):
    mesh = box_mesh_3d(*[np.linspace(-.5, .5, cells+1)]*3)
    rng = np.random.default_rng(7341)
    mesh["vertex_position"] += rng.normal(0, .004, mesh["vertex_position"].shape)
    geo = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    native = NativeVolumeSources.from_mesh(mesh)
    linear = LinearNativeVolumeSources.from_native(native, geo["cell_centre"])
    continuous = ContinuousVelocityCurl.from_mesh(mesh, linear.centroid)
    return mesh, geo, native, linear, continuous, rng


def compact_state(continuous, rng, nonzero_integral=True):
    trace = rng.normal(size=(2, continuous.cell_offset, 3))
    trace[:, continuous.boundary_nodes] = 0
    integral = rng.normal(size=(2, len(continuous.cell_centroid), 3))*continuous.cell_volume[None, :, None]
    if not nonzero_integral:
        integral[:] = 0
    return continuous.complete_cell_integrals(trace, integral), integral


def test_affine_velocity_and_true_polyhedral_geometry_are_reproduced():
    mesh, geo, native, linear, c, rng = fixture()
    np.testing.assert_allclose(c.cell_volume, linear.volume, rtol=0, atol=3e-17)
    matrices, offset = rng.normal(size=(2, 3, 3)), rng.normal(size=(2, 3))
    velocity = np.einsum("ni,sij->snj", c.position, matrices)+offset[:, None]
    integral = (np.einsum("ni,sij->snj", c.cell_centroid, matrices)+offset[:, None])*c.cell_volume[None, :, None]
    reconstructed = c.complete_cell_integrals(velocity[:, :c.cell_offset], integral)
    np.testing.assert_allclose(reconstructed, velocity, rtol=0, atol=3e-15)
    np.testing.assert_allclose(c.gradient(reconstructed), np.broadcast_to(matrices[:, None], (2, len(c.tetrahedra), 3, 3)), rtol=0, atol=4e-14)
    expected = np.column_stack((matrices[:, 1, 2]-matrices[:, 2, 1], matrices[:, 2, 0]-matrices[:, 0, 2], matrices[:, 0, 1]-matrices[:, 1, 0]))
    np.testing.assert_allclose(c.curl(reconstructed), np.broadcast_to(expected[:, None], (2, len(c.tetrahedra), 3)), rtol=0, atol=4e-14)


def test_compact_curl_has_continuous_normal_trace_and_independent_stokes_moments():
    mesh, geo, native, linear, c, rng = fixture()
    velocity, integral = compact_state(c, rng)
    omega = c.curl(velocity)
    np.testing.assert_allclose(c.velocity_integrals(velocity), integral, rtol=0, atol=8e-17)
    np.testing.assert_allclose(c.normal_curl_jumps(omega), 0, rtol=0, atol=8e-14)
    gamma, moment = c.curl_moments(omega)
    expected_gamma, expected_moment = np.zeros_like(gamma), np.zeros_like(moment)
    barycentric, quadrature = triangle_rule(4)
    triangle = 0
    for face, vertices in enumerate(mesh["faces"]):
        for first, second in zip(vertices, np.roll(vertices, -1), strict=True):
            nodes = np.array([c.n_vertices+face, first, second])
            points = barycentric @ c.position[nodes]
            u = np.einsum("qn,snj->sqj", barycentric, velocity[:, nodes])
            sf = np.cross(c.position[nodes[1]]-c.position[nodes[0]], c.position[nodes[2]]-c.position[nodes[0]])/2
            flux = np.cross(sf[None, None], u)*quadrature[None, :, None]
            for cell, sign in ((native.owners[triangle], 1), (native.neighbours[triangle], -1)):
                if cell >= 0:
                    expected_gamma[:, cell] += sign*flux.sum(axis=1)
                    expected_moment[:, cell] += sign*np.einsum("qi,sqj->sij", points-linear.centroid[cell], flux)
            triangle += 1
    expected_moment -= np.cross(np.eye(3)[None, None], integral[:, :, None])
    np.testing.assert_allclose(gamma, expected_gamma, rtol=0, atol=8e-16)
    np.testing.assert_allclose(moment, expected_moment, rtol=0, atol=2e-16)
    total_q = integral.sum(axis=1)
    h = global_first_moment(linear.centroid, gamma, moment)
    np.testing.assert_allclose(gamma.sum(axis=1), 0, rtol=0, atol=2e-15)
    np.testing.assert_allclose(h, -np.cross(np.eye(3)[None], total_q[:, None]), rtol=0, atol=2e-15)
    np.testing.assert_allclose(impulse(h), total_q, rtol=0, atol=2e-15)


def test_compact_polynomial_nodes_keep_zero_cells_and_boundary_untouched():
    mesh, geo, native, linear, c, rng = fixture(cells=5)
    weight = rng.uniform(.2, 1, mesh["n_cells"])
    weight[np.max(np.abs(geo["cell_centre"]), axis=1) > .3] = 0
    polynomial = [rng.normal(size=(2, mesh["n_faces"])+(3,)*(i+1)) for i in range(3)]
    polynomial[2] = .5*(polynomial[2]+polynomial[2].swapaxes(-2, -3))
    trace, _, _ = compact_polynomial_trace(mesh, geo, c, weight, polynomial)
    integral = np.zeros((2, mesh["n_cells"], 3))
    velocity = c.complete_cell_integrals(trace, integral)
    omega = c.curl(velocity)
    np.testing.assert_array_equal(velocity[:, c.boundary_nodes], 0)
    np.testing.assert_array_equal(omega[:, weight[c.parent] == 0], 0)
    np.testing.assert_allclose(c.velocity_integrals(velocity), 0, rtol=0, atol=1e-17)
    np.testing.assert_allclose(c.normal_curl_jumps(omega), 0, rtol=0, atol=1e-13)
    gamma, moment = c.curl_moments(omega)
    np.testing.assert_allclose(global_first_moment(linear.centroid, gamma, moment), 0, rtol=0, atol=8e-17)
    assert np.linalg.norm(omega) > 1


def finite_difference_curl(evaluate, points, step):
    derivative = []
    for axis in range(3):
        d = step*np.eye(3)[axis]
        derivative.append((evaluate(points+d)-evaluate(points-d))/(2*step))
    return np.stack((derivative[1][..., 2]-derivative[2][..., 1],
                     derivative[2][..., 0]-derivative[0][..., 2],
                     derivative[0][..., 1]-derivative[1][..., 0]), axis=-1)


def test_biot_savart_matches_independent_volume_rule_and_recovers_compact_physical_curl():
    mesh, geo, native, linear, c, rng = fixture(cells=2)
    velocity, _ = compact_state(c, rng, nonzero_integral=False)
    omega = c.curl(velocity[:1])
    source, coefficient = c.active_source(omega)
    outside = np.array([[1.2, .13, -.21], [-.27, -1.3, .11], [.24, .17, 1.4]])
    actual = source.evaluate(outside, coefficient)[:, 0]
    whole = c.source.evaluate(outside, c.source.coefficients(omega))[:, 0]
    np.testing.assert_array_equal(actual, whole)
    nodes, weights = np.polynomial.legendre.leggauss(12)
    a, b, d = np.meshgrid(*[(nodes+1)/2]*3, indexing="ij")
    wa, wb, wd = np.meshgrid(*[weights/2]*3, indexing="ij")
    bary = np.stack((1-a, a*(1-b), a*b*(1-d), a*b*d), axis=-1).reshape(-1, 4)
    qweight = (6*a*a*b*wa*wb*wd).ravel()
    points = np.einsum("qn,tni->tqi", bary, c.position[c.tetrahedra])
    expected = []
    for target in outside:
        r = target-points
        integrand = np.cross(omega[0, :, None], r)/(4*np.pi*np.linalg.norm(r, axis=-1)[..., None]**3)
        expected.append(np.einsum("t,q,tqi->i", c.volume, qweight, integrand))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-14)
    def evaluate(points):
        return source.evaluate(points, coefficient)[:, 0]
    outside_curl = finite_difference_curl(evaluate, outside, 1e-4)
    outside_half = finite_difference_curl(evaluate, outside, 5e-5)
    np.testing.assert_allclose(outside_half, 0, rtol=0, atol=2e-9)
    np.testing.assert_allclose(outside_curl, outside_half, rtol=0, atol=2e-9)
    ids = np.array([7, 43, 117])
    interior = c.centroid[ids]
    interior_curl = finite_difference_curl(evaluate, interior, 1e-5)
    np.testing.assert_allclose(interior_curl, omega[0, ids], rtol=0, atol=3e-6)
    # Identical cell circulation and first moments do not preserve this local
    # curl structure when compressed to an unconstrained affine density.
    gamma, moment = c.curl_moments(omega)
    affine_coefficient, _ = linear.coefficients(gamma, moment)
    affine_curl = finite_difference_curl(lambda points: linear.evaluate(points, affine_coefficient)[:, 0], outside, 5e-5)
    assert np.linalg.norm(affine_curl) > 1e-4


def test_nonconforming_native_edges_and_invalid_data_fail_explicitly():
    mesh, geo, native, linear, c, rng = fixture(cells=1)
    face = list(mesh["faces"][0])
    extra = len(mesh["vertex_position"])
    midpoint = mesh["vertex_position"][face[:2]].mean(axis=0)
    mesh["vertex_position"] = np.vstack((mesh["vertex_position"], midpoint))
    mesh["faces"] = [list(value) for value in mesh["faces"]]
    mesh["faces"][0].insert(1, extra)
    with pytest.raises(ValueError, match="conforming"):
        ContinuousVelocityCurl.from_mesh(mesh, linear.centroid)
    with pytest.raises(ValueError, match="integrals"):
        c.complete_cell_integrals(np.zeros((c.cell_offset, 3)), np.zeros((2, 3)))
    with pytest.raises(ValueError, match="Finite velocities"):
        c.curl(np.full(c.position.shape, np.nan))
