"""The source potential must integrate 1/r and differentiate to its velocity."""

import numpy as np
import pytest
import taichi as ti
from scipy.special import roots_jacobi

from source.solvers.vpm.boundary_elements.panels.kernels.source_potential import compute_source_potential
from source.solvers.vpm.boundary_elements.panels.kernels.source_velocity import compute_source_velocity


@ti.data_oriented
class PotentialSampler:
    def __init__(self, points, triangle, dtype):
        self.points = ti.Vector.field(3, dtype, shape=len(points))
        self.vertices = ti.Vector.field(3, dtype, shape=3)
        self.potential = ti.field(dtype, shape=len(points))
        self.velocity = ti.Vector.field(3, dtype, shape=len(points))
        self.points.from_numpy(points)
        self.vertices.from_numpy(triangle)

    @ti.kernel
    def evaluate(self):
        v0, v1, v2 = self.vertices[0], self.vertices[1], self.vertices[2]
        normal = (v1-v0).cross(v2-v0).normalized()
        for i in self.points:
            self.potential[i] = compute_source_potential(self.points[i], v0, v1, v2, normal)
            self.velocity[i] = compute_source_velocity(self.points[i], v0, v1, v2, normal)


def quadrature_potential(points, triangle):
    r, wr = roots_jacobi(48, 1, 0)
    s, ws = roots_jacobi(48, 0, 0)
    r, s = (r+1)/2, (s+1)/2
    rr, ss = np.meshgrid(r, s, indexing="ij")
    weights = np.outer(wr/4, ws/2).ravel()
    barycentric = np.column_stack((rr.ravel(), ((1-rr)*ss).ravel(), ((1-rr)*(1-ss)).ravel()))
    points_on_triangle = barycentric @ triangle
    jacobian = np.linalg.norm(np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0]))
    return -jacobian*(1/np.linalg.norm(points[:, None]-points_on_triangle, axis=2)) @ weights/(4*np.pi)


@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_source_potential_matches_integral_and_has_correct_far_field(precision, tmp_path):
    dtype = ti.f64 if precision == "f64" else ti.f32
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=dtype, cpu_max_num_threads=1, offline_cache=False,
            offline_cache_file_path=str(tmp_path / "taichi"))
    try:
        triangle = np.array([[-0.4, -0.2, 0.1], [0.7, -0.3, 0.4], [0.1, 0.8, -0.2]])
        points = np.array([[0.2, 0.1, 0.8], [-0.3, 0.2, -0.9], [7.1, 2.3, -1.7]])
        sampler = PotentialSampler(points, triangle, dtype)
        sampler.evaluate()
        expected = quadrature_potential(points, triangle)
        np.testing.assert_allclose(sampler.potential.to_numpy(), expected,
                                   rtol=5e-6 if precision == "f32" else 2e-13, atol=1e-15)
        reverse = PotentialSampler(points, triangle[[0, 2, 1]], dtype)
        reverse.evaluate()
        np.testing.assert_allclose(reverse.potential.to_numpy(), expected,
                                   rtol=5e-6 if precision == "f32" else 2e-13, atol=1e-15)
    finally:
        ti.reset()


def test_source_potential_gradient_matches_source_velocity_and_vertex_limit(tmp_path):
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1, offline_cache=False,
            offline_cache_file_path=str(tmp_path / "taichi"))
    try:
        triangle = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        points = np.array([[0.2, 0.1, 0.8], [-0.3, 0.2, -0.9]])
        step = 2e-4
        offsets = [step*np.eye(3)[axis] for axis in range(3)]
        all_points = np.vstack([points] + [points+factor*offset for offset in offsets for factor in (-2, -1, 1, 2)]
                               + [triangle[:1]])
        sampler = PotentialSampler(all_points, triangle, ti.f64)
        sampler.evaluate()
        potential = sampler.potential.to_numpy()
        for axis in range(3):
            m2, m1, p1, p2 = potential[2+8*axis:2+8*(axis+1)].reshape(4, 2)
            gradient = (m2-8*m1+8*p1-p2)/(12*step)
            np.testing.assert_allclose(gradient, sampler.velocity.to_numpy()[:2, axis], rtol=0, atol=3e-11)
        # Target at the right triangle vertex: a radial integration leaves
        # integral_0^1 ds / sqrt((1-s)^2+s^2) = sqrt(2)*asinh(1).
        expected = -np.sqrt(2)*np.arcsinh(1)/(4*np.pi)
        np.testing.assert_allclose(potential[-1], expected, rtol=0, atol=2e-15)
    finally:
        ti.reset()
