"""Qualify sampled native face observations against independent 3D fields."""

import numpy as np
import pytest

from source.solvers.fvm.fields.gradients import compute_gauss_gradient
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.mesh.rectilinear import box_mesh_3d
from studies.coupler_accuracy.native_face_trace_3d import tangential
from studies.coupler_accuracy.native_face_velocity_sampling_3d import InteriorFaceVelocitySampler


def fixture(warp=False):
    mesh = box_mesh_3d(*[np.linspace(-2, 2, 9)]*3)
    for patch in mesh["boundary"]:
        patch["velocity_type"] = "fixedValue"
    mesh["vertex_position"] = mesh["vertex_position"] @ np.array([[1., .25, -.1], [.1, 1.1, .2], [.15, -.2, .9]]).T
    if warp:
        mesh["vertex_position"] += np.random.default_rng(2029).normal(0, .003, mesh["vertex_position"].shape)
    geo = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    faces = np.flatnonzero(np.max(np.abs(geo["face_centre"][:mesh["n_interior_faces"]]), axis=1) < .6)
    signs = np.where(np.arange(len(faces)) % 2, -1, 1)
    return mesh, geo, InteriorFaceVelocitySampler(mesh, geo, faces, signs)


def test_compact_samples_match_full_native_gauss_and_independent_flux_on_warped_cells():
    mesh, geo, sampler = fixture(warp=True)
    rng = np.random.default_rng(1031)
    nt = mesh["n_cells"]+mesh["n_faces"]-mesh["n_interior_faces"]
    velocity = rng.normal(size=(nt, 3))
    actual = sampler.evaluate(velocity[sampler.sample_cells])
    gradient = compute_gauss_gradient(velocity, mesh, geo)
    np.testing.assert_array_equal(actual["cell_gradient"], gradient[sampler.gradient_cells])
    # Direct face formula, independently assembled from the complete field.
    f = sampler.faces
    owner, neighbour = mesh["owners"][f], mesh["neighbours"][f]
    w = geo["face_interpolation_weight"][f]
    sf, edge = geo["face_area_vector"][f], geo["cell_connection_vector"][f]
    length = np.linalg.norm(edge, axis=1)
    direction = edge/length[:, None]
    area = np.linalg.norm(sf, axis=1)
    ef = area**2/np.sum(sf*direction, axis=1)
    transverse = sf-ef[:, None]*direction
    face_grad = w[:, None, None]*gradient[neighbour]+(1-w[:, None, None])*gradient[owner]
    derivative = (ef[:, None]*(velocity[neighbour]-velocity[owner])/length[:, None]
                  +np.einsum("nd,ndc->nc", transverse, face_grad))*sampler.trace.signs[:, None]/area[:, None]
    np.testing.assert_allclose(actual["normal_gradient"], derivative, rtol=0, atol=1e-14)
    np.testing.assert_allclose(actual["tangential_gradient"], tangential(derivative, sampler.trace.normal), rtol=0, atol=1e-14)
    # Changing every unsampled cell/ghost cannot alter any selected trace.
    other = np.ones(nt, dtype=bool)
    other[sampler.sample_cells] = False
    velocity[other] = rng.normal(100, 500, (other.sum(), 3))
    poisoned = compute_gauss_gradient(velocity, mesh, geo)
    np.testing.assert_array_equal(poisoned[sampler.gradient_cells], actual["cell_gradient"])
    assert len(sampler.sample_cells) < mesh["n_cells"]//2


def test_sampled_affine_velocity_recovers_continuous_derivative_in_all_three_directions():
    _, geo, sampler = fixture()
    g = np.array([[.2, -.7, 1.3], [.6, -.1, -.4], [-.8, 1.2, -.1]])
    values = geo["cell_centre"][sampler.sample_cells] @ g+[.4, -.1, .6]
    result = sampler.evaluate(values)
    np.testing.assert_allclose(result["cell_gradient"], np.broadcast_to(g, result["cell_gradient"].shape), rtol=0, atol=3e-14)
    np.testing.assert_allclose(result["normal_gradient"], sampler.trace.normal @ g, rtol=0, atol=3e-14)
    np.testing.assert_allclose(result["face_velocity"], geo["face_centre"][sampler.faces] @ g+[.4, -.1, .6], rtol=0, atol=3e-14)
    assert np.linalg.matrix_rank(sampler.trace.normal) == 3


def test_discrete_and_continuous_observations_converge_quadratically_for_smooth_3d_field():
    k = np.array([[.7, -.2, .3], [.1, .6, -.5], [-.4, .3, .8]])
    errors = []
    for count in (16, 32):
        mesh = box_mesh_3d(*[np.linspace(-2, 2, count+1)]*3)
        for patch in mesh["boundary"]:
            patch["velocity_type"] = "fixedValue"
        geo = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
        faces = np.flatnonzero(np.max(np.abs(geo["face_centre"][:mesh["n_interior_faces"]]), axis=1) < .6)
        sampler = InteriorFaceVelocitySampler(mesh, geo, faces, np.ones(len(faces)))
        value = np.sin(geo["cell_centre"][sampler.sample_cells] @ k)
        result = sampler.evaluate(value)
        face = geo["face_centre"][faces]
        exact = np.cos(face @ k)*(sampler.trace.normal @ k)
        error = result["tangential_gradient"]-tangential(exact, sampler.trace.normal)
        errors.append(np.sqrt(np.mean(np.sum(error**2, axis=1))))
    assert .22 < errors[1]/errors[0] < .28
    assert errors[0] > 1e-4


def test_missing_physical_boundary_observations_and_empty_mesh_are_rejected():
    mesh, geo, _ = fixture()
    ni = mesh["n_interior_faces"]
    boundary_cells = np.unique(mesh["owners"][ni:])
    face = np.flatnonzero(np.isin(mesh["owners"][:ni], boundary_cells))[0]
    with pytest.raises(ValueError, match="physical boundary observations"):
        InteriorFaceVelocitySampler(mesh, geo, [face], [1])
    mesh["boundary"][0]["velocity_type"] = "empty"
    with pytest.raises(ValueError, match="fully 3D"):
        InteriorFaceVelocitySampler(mesh, geo, [face], [1])
