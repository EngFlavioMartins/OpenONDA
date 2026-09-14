"""Independent 3D divergence-theorem and triangle quadrature checks."""

import numpy as np

from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.mesh.rectilinear import box_mesh_3d
from studies.coupler_accuracy.native_flux_moments_3d import (
    cell_flux_moments,
    enforce_face_flux,
    linear_face_flux_moments,
)
from studies.coupler_accuracy.native_volume_induction_3d import NativeVolumeSources


def setup():
    mesh = box_mesh_3d(np.array([-.7, -.2, .3, 1.1]), np.array([-.4, 0., .5, 1.2]), np.array([-.6, -.1, .2, .8]))
    mesh["vertex_position"] += np.random.default_rng(718).normal(0, .007, mesh["vertex_position"].shape)
    geometry = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    native = NativeVolumeSources.from_mesh(mesh)
    return mesh, geometry, native


def quadrature(native, origin, velocity, gradient, normal_shift=None):
    # The symmetric three-point triangle rule integrates quadratic moments.
    barycentric = (np.ones((3, 3)) + 3 * np.eye(3)) / 6
    area_vector = np.cross(native.triangles[:, 1] - native.triangles[:, 0], native.triangles[:, 2] - native.triangles[:, 0]) / 2
    area = np.linalg.norm(area_vector, axis=1)
    relative = np.einsum("qv,tvi->tqi", barycentric, native.triangles - origin[native.face_ids, None])
    value = velocity[native.face_ids, None] + np.einsum("tqi,tij->tqj", relative, gradient[native.face_ids])
    if normal_shift is not None:
        value += normal_shift[native.face_ids, None, None] * (area_vector / area[:, None])[:, None]
    flux = np.einsum("tqi,ti->tq", value, area_vector) / 3
    total, moment = np.zeros(len(origin)), np.zeros_like(origin)
    np.add.at(total, native.face_ids, flux.sum(axis=1))
    np.add.at(moment, native.face_ids, (relative * flux[:, :, None]).sum(axis=1))
    return total, moment


def test_affine_flux_moments_recover_velocity_and_divergence_on_warped_cells():
    mesh, geometry, native = setup()
    origin = geometry["face_centre"]
    cell_origin = geometry["cell_centre"] + [.03, -.02, .01]
    # Independent signed tetrahedron sums, rather than the face-summary volume.
    volume, centre_integral = np.zeros(mesh["n_cells"]), np.zeros((mesh["n_cells"], 3))
    for triangle, own, nei in zip(native.triangles, native.owners, native.neighbours, strict=True):
        for cell, sign in ((own, 1), (nei, -1)):
            if cell < 0:
                continue
            reference = cell_origin[cell]
            dv = sign * np.linalg.det(triangle - reference) / 6
            volume[cell] += dv
            centre_integral[cell] += dv * (reference + triangle.sum(axis=0)) / 4
    centre = centre_integral / volume[:, None]
    constant = np.array([.3, -.7, 1.2])
    for matrix in (np.zeros((3, 3)), np.array([[.3, -.2, .7], [.5, -.8, .1], [-.4, .9, .5]]),
                   np.array([[.3, -.2, .7], [.5, -.8, .1], [-.4, .9, 1.3]])):
        flux, moment = linear_face_flux_moments(native, origin, constant + origin @ matrix,
                                               np.broadcast_to(matrix, (len(origin), 3, 3)))
        net, first = cell_flux_moments(mesh, origin, flux, moment, cell_origin)
        expected = volume[:, None] * (constant + centre @ matrix + np.trace(matrix) * (centre - cell_origin))
        np.testing.assert_allclose(net, volume * np.trace(matrix), rtol=0, atol=1e-15)
        np.testing.assert_allclose(first, expected, rtol=0, atol=1e-15)


def test_random_shared_traces_match_independent_surface_quadrature_and_global_budget():
    mesh, geometry, native = setup()
    rng = np.random.default_rng(829)
    origin = geometry["face_centre"]
    velocity, gradient = rng.normal(size=origin.shape), rng.normal(size=(len(origin), 3, 3))
    flux, moment = linear_face_flux_moments(native, origin, velocity, gradient)
    expected = quadrature(native, origin, velocity, gradient)
    np.testing.assert_allclose(flux, expected[0], rtol=0, atol=5e-16)
    np.testing.assert_allclose(moment, expected[1], rtol=0, atol=1e-16)
    net, first = cell_flux_moments(mesh, origin, flux, moment, geometry["cell_centre"])
    boundary = slice(mesh["n_interior_faces"], None)
    np.testing.assert_allclose(net.sum(), flux[boundary].sum(), rtol=0, atol=3e-15)
    global_moment = (first + geometry["cell_centre"] * net[:, None]).sum(axis=0)
    np.testing.assert_allclose(global_moment, (origin[boundary] * flux[boundary, None] + moment[boundary]).sum(axis=0), rtol=0, atol=3e-15)


def test_normal_flux_correction_has_the_measured_moments_on_warped_faces():
    _, geometry, native = setup()
    rng = np.random.default_rng(792)
    velocity = rng.normal(size=geometry["face_centre"].shape)
    gradient = rng.normal(size=(len(velocity), 3, 3))
    target = rng.normal(size=len(velocity)) * .03
    for offset in (np.zeros(3), np.array([.017, -.023, .01])):
        origin = geometry["face_centre"] + offset
        flux, moment = linear_face_flux_moments(native, origin, velocity, gradient)
        corrected_flux, corrected_moment, delta = enforce_face_flux(native, origin, flux, moment, target)
        expected_flux, expected_moment = quadrature(native, origin, velocity, gradient, delta)
        np.testing.assert_allclose(corrected_flux, expected_flux, rtol=0, atol=5e-16)
        np.testing.assert_allclose(corrected_moment, expected_moment, rtol=0, atol=1e-16)
        # The stored area centroid itself has coordinate-sized roundoff.
        # A nominally zero moment therefore has an eps * |origin| * |flux|
        # error floor, not an absolute zero-based tolerance of 1e-16.
        error = corrected_moment - moment + (target - flux)[:, None] * offset
        scale = np.maximum(1, np.max(np.abs(origin), axis=1)) * np.abs(target - flux)
        bound = 8 * np.finfo(float).eps * (scale[:, None] + np.abs(moment))
        assert np.all(np.abs(error) <= bound)
