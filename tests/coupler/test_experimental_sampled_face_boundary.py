"""Verify the live query adapter changes only its selected boundary observations."""

import numpy as np
import pytest

from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.mesh.rectilinear import box_mesh_3d
from studies.coupler_accuracy.experimental_sampled_face_boundary import sampled_face_boundary
from studies.coupler_accuracy.native_face_velocity_sampling_3d import InteriorFaceVelocitySampler


def fixture():
    mesh = box_mesh_3d(*[np.linspace(-2, 2, 9)]*3)
    for patch in mesh["boundary"]:
        patch["velocity_type"] = "fixedValue"
    geometry = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    faces = np.flatnonzero(np.max(np.abs(geometry["face_centre"][:mesh["n_interior_faces"]]), axis=1) < .6)
    return InteriorFaceVelocitySampler(mesh, geometry, faces, np.ones(len(faces)))


class VelocityOnlySolver:
    def __init__(self):
        self.queries, self.original_calls = [], 0

    @staticmethod
    def velocity(x):
        return np.sin(x @ np.array([[.7, -.2, .3], [.1, .6, -.5], [-.4, .3, .8]]))

    def compute_velocity_at_points(self, points, **kwargs):
        assert kwargs == {"include_freestream": True, "include_body": True, "zone_mask": None}
        self.queries.append(np.asarray(points).copy())
        return self.velocity(points)

    def compute_velocity_and_tangential_normal_gradient_at_points(self, points, normal, *, particle_spacing):
        self.original_calls += 1
        return self.velocity(points), np.full_like(points, 17.)


@pytest.mark.parametrize("mode", ["native_gradient", "native_both"])
def test_adapter_uses_only_solver_velocity_samples_and_preserves_requested_point_value(mode):
    sampler, solver = fixture(), VelocityOnlySolver()
    points = sampler.geometry["face_centre"][sampler.faces]
    sample_points = sampler.geometry["cell_centre"][sampler.sample_cells]
    original = VelocityOnlySolver.compute_velocity_and_tangential_normal_gradient_at_points
    records = []
    with sampled_face_boundary(sampler, mode=mode, callback=lambda *args: records.append(args), solver_class=VelocityOnlySolver):
        velocity, derivative = solver.compute_velocity_and_tangential_normal_gradient_at_points(points, sampler.trace.normal, particle_spacing=.125)
    expected = sampler.evaluate(solver.velocity(sample_points))
    np.testing.assert_array_equal(derivative, expected["tangential_gradient"])
    np.testing.assert_array_equal(velocity, solver.velocity(points) if mode == "native_gradient" else expected["face_velocity"])
    np.testing.assert_array_equal(solver.queries[0], sample_points)
    assert len(solver.queries) == len(records) == 1
    assert solver.original_calls == (mode == "native_gradient")
    assert VelocityOnlySolver.compute_velocity_and_tangential_normal_gradient_at_points is original
    assert np.max(np.abs(expected["face_velocity"]-solver.velocity(points))) > 1e-4


def test_adapter_rejects_wrong_geometry_and_restores_method_after_exception():
    sampler, solver = fixture(), VelocityOnlySolver()
    points = sampler.geometry["face_centre"][sampler.faces]
    original = VelocityOnlySolver.compute_velocity_and_tangential_normal_gradient_at_points
    with (pytest.raises(ValueError, match="different face set"),
          sampled_face_boundary(sampler, mode="native_both", solver_class=VelocityOnlySolver)):
        solver.compute_velocity_and_tangential_normal_gradient_at_points(points+.001, sampler.trace.normal, particle_spacing=.125)
    assert VelocityOnlySolver.compute_velocity_and_tangential_normal_gradient_at_points is original
    with (pytest.raises(ValueError, match="different face orientations"),
          sampled_face_boundary(sampler, mode="native_gradient", solver_class=VelocityOnlySolver)):
        solver.compute_velocity_and_tangential_normal_gradient_at_points(points, -sampler.trace.normal, particle_spacing=.125)
    assert VelocityOnlySolver.compute_velocity_and_tangential_normal_gradient_at_points is original
    assert not solver.queries
