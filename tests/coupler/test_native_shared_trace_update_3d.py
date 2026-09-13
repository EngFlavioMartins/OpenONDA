"""Qualify shared overlap corrections, moment budgets and spatial support."""

import numpy as np
import pytest

from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.mesh.rectilinear import box_mesh_3d
from studies.coupler_accuracy.native_linear_volume_induction_3d import LinearNativeVolumeSources
from studies.coupler_accuracy.native_quadratic_moments_3d import weak_quadratic_curl_moments
from studies.coupler_accuracy.native_shared_trace_update_3d import (
    compact_face_weights,
    global_first_moment,
    impulse,
    shared_trace_update,
)
from studies.coupler_accuracy.native_velocity_curl_integrals_3d import triangle_rule
from studies.coupler_accuracy.native_volume_induction_3d import NativeVolumeSources


def fixture():
    mesh = box_mesh_3d(*[np.linspace(-1, 1, 6)]*3)
    rng = np.random.default_rng(2089)
    mesh["vertex_position"] += rng.normal(0, .003, mesh["vertex_position"].shape)
    geo = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    native = NativeVolumeSources.from_mesh(mesh)
    linear = LinearNativeVolumeSources.from_native(native, geo["cell_centre"])
    weight = rng.uniform(.1, 1, mesh["n_cells"])
    weight[np.max(np.abs(geo["cell_centre"]), axis=1) > .65] = 0
    faces = [rng.normal(size=(2, mesh["n_faces"])+(3,)*(i+1)) for i in range(3)]
    faces[2] = .5*(faces[2]+faces[2].swapaxes(-3, -2))
    base = [np.zeros_like(value) for value in faces]
    integral = rng.normal(size=(2, mesh["n_cells"], 3))*linear.volume[None, :, None]
    return mesh, geo, native, linear, weight, base, faces, integral


def test_weighted_update_matches_independent_triangle_quadrature_and_keeps_zero_cells_untouched():
    mesh, geo, native, linear, weight, base, faces, integral = fixture()
    update = shared_trace_update(mesh, native, linear.centroid, weight, geo["face_centre"], integral, integral, base, faces)
    barycentric, quadrature = triangle_rule(5)
    expected_gamma = np.zeros_like(update.circulation)
    expected_moment = np.zeros_like(update.first_moment)
    for t, triangle in enumerate(native.triangles):
        face = native.face_ids[t]
        if update.face_weight[face] == 0:
            continue
        q = barycentric @ triangle
        d = q-geo["face_centre"][face]
        u = (faces[0][:, face, None]+np.einsum("qd,sdj->sqj", d, faces[1][:, face])
             +.5*np.einsum("qd,qk,sdkj->sqj", d, d, faces[2][:, face]))*update.face_weight[face]
        sf = .5*np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
        flux = np.cross(sf[None, None], u)*quadrature[None, :, None]
        for cell, sign in ((native.owners[t], 1), (native.neighbours[t], -1)):
            if cell >= 0:
                expected_gamma[:, cell] += sign*flux.sum(axis=1)
                expected_moment[:, cell] += sign*np.einsum("qi,sqj->sij", q-linear.centroid[cell], flux)
    np.testing.assert_allclose(update.circulation, expected_gamma, rtol=0, atol=7e-16)
    np.testing.assert_allclose(update.first_moment, expected_moment, rtol=0, atol=3e-16)
    np.testing.assert_array_equal(update.circulation[:, weight == 0], 0)
    np.testing.assert_array_equal(update.first_moment[:, weight == 0], 0)
    assert np.linalg.norm(update.circulation) > .1


def test_equal_cell_integrals_preserve_circulation_and_impulse_without_rescaling():
    mesh, geo, native, linear, weight, base, faces, integral = fixture()
    update = shared_trace_update(mesh, native, linear.centroid, weight, geo["face_centre"], integral, integral, base, faces)
    np.testing.assert_allclose(update.circulation.sum(axis=1), 0, rtol=0, atol=2e-15)
    h = global_first_moment(linear.centroid, update.circulation, update.first_moment)
    np.testing.assert_allclose(h, 0, rtol=0, atol=2e-15)
    np.testing.assert_allclose(impulse(h), 0, rtol=0, atol=2e-15)
    # Weighting completed cell curls instead does not have that cancellation.
    unw_gamma, unw_moment = weak_quadratic_curl_moments(native, linear.centroid, np.zeros_like(integral), geo["face_centre"], *faces)
    naive_gamma, naive_moment = weight[None, :, None]*unw_gamma, weight[None, :, None, None]*unw_moment
    assert np.linalg.norm(naive_gamma.sum(axis=1)) > .1
    assert np.linalg.norm(impulse(global_first_moment(linear.centroid, naive_gamma, naive_moment))) > .1


def test_nonzero_integral_change_has_the_analytic_global_first_moment_and_impulse_budget():
    mesh, geo, native, linear, weight, base, faces, integral = fixture()
    new_integral = integral+linear.volume[None, :, None]*np.array([.7, -.3, .4])
    update = shared_trace_update(mesh, native, linear.centroid, weight, geo["face_centre"], integral, new_integral, base, faces)
    change = np.sum(weight[None, :, None]*(new_integral-integral), axis=1)
    h = global_first_moment(linear.centroid, update.circulation, update.first_moment)
    np.testing.assert_allclose(h, -np.cross(np.eye(3)[None], change[:, None]), rtol=0, atol=2e-15)
    np.testing.assert_allclose(impulse(h), change, rtol=0, atol=2e-15)
    # Taking differences first gives an exact zero update for identical inputs.
    zero = shared_trace_update(mesh, native, linear.centroid, weight, geo["face_centre"], integral, integral, faces, faces)
    np.testing.assert_array_equal(zero.circulation, 0)
    np.testing.assert_array_equal(zero.first_moment, 0)


def test_out_of_range_weights_and_noncompact_boundary_changes_are_rejected():
    mesh, geo, native, linear, weight, base, faces, integral = fixture()
    with pytest.raises(ValueError, match="weights in"):
        compact_face_weights(mesh, weight-2)
    with pytest.raises(ValueError, match="physical boundaries"):
        shared_trace_update(mesh, native, linear.centroid, np.ones_like(weight), geo["face_centre"], integral, integral, base, faces)
