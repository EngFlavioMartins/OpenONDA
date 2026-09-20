"""Independent three-dimensional quadrature for the source-panel Jacobian."""

import numpy as np
import pytest
from scipy.special import roots_jacobi

from source.solvers.vpm.boundary_elements.panels.kernels.source_gradient import (
    source_panel_gradient,
)
from source.solvers.vpm.core.solver import VPMSolver


@pytest.mark.parametrize("scale", [0.3, 1.0, 7.0])
def test_analytical_source_gradient_matches_surface_integral(scale):
    triangle = scale * np.array([[-0.4, -0.2, 0.1], [0.7, -0.3, 0.4], [0.1, 0.8, -0.2]])
    points = scale * np.array([[0.2, 0.1, 0.8], [-0.3, 0.2, -0.9], [7.1, 2.3, -1.7]])
    r, wr = roots_jacobi(48, 1, 0)
    s, ws = roots_jacobi(48, 0, 0)
    r, s = (r + 1) / 2, (s + 1) / 2
    rr, ss = np.meshgrid(r, s, indexing="ij")
    weights = np.outer(wr / 4, ws / 2).ravel()
    bary = np.column_stack((rr.ravel(), ((1 - rr) * ss).ravel(), ((1 - rr) * (1 - ss)).ravel()))
    delta = points[:, None, :] - (bary @ triangle)[None, :, :]
    radius = np.linalg.norm(delta, axis=2)
    tensor = (
        np.eye(3)[None, None] / radius[:, :, None, None] ** 3
        - 3 * delta[:, :, :, None] * delta[:, :, None, :] / radius[:, :, None, None] ** 5
    )
    area = np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]))
    expected = 0.7 * area * np.einsum("pqij,q->pij", tensor, weights) / (4 * np.pi)
    actual = source_panel_gradient(points, triangle[None], np.array([0.7]))
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-13 / scale)
    # Reversing the triangle orientation must not change its physical source.
    reverse = source_panel_gradient(points, triangle[None, ::-1], np.array([0.7]))
    np.testing.assert_allclose(reverse, actual, rtol=1e-12, atol=1e-14 / scale)
    np.testing.assert_allclose(np.trace(actual, axis1=1, axis2=2), 0, atol=2e-13 / scale)


def test_parallel_targets_preserve_every_bit(monkeypatch):
    rng = np.random.default_rng(41)
    points = rng.normal(size=(8192, 3)) + [3.0, 4.0, 5.0]
    triangles = np.array([[[-0.4, -0.2, 0.1], [0.7, -0.3, 0.4], [0.1, 0.8, -0.2]]])
    monkeypatch.setenv("TI_CPU_MAX_NUM_THREADS", "1")
    serial = source_panel_gradient(points, triangles, [0.7])
    monkeypatch.setenv("TI_CPU_MAX_NUM_THREADS", "4")
    parallel = source_panel_gradient(points, triangles, [0.7])
    np.testing.assert_array_equal(parallel, serial)


def test_fvm_vpm_panel_gradient_avoids_an_f64_taichi_target_kernel():
    """The Metal treecode path must use the host source Jacobian directly."""
    particle_jacobian = np.array(
        [[0.2, -0.1, 0.4], [0.0, 0.3, -0.2], [0.1, 0.5, -0.4]], dtype=np.float64
    )
    panel_jacobian = np.array(
        [[-0.3, 0.2, 0.1], [0.6, -0.2, 0.4], [-0.1, 0.3, 0.2]], dtype=np.float64
    )

    class Panel:
        coupling_scope = "fvm_vpm"

        def compute_induced_velocity(self, points):
            return np.zeros_like(points, dtype=np.float32)

        def compute_induced_velocity_gradient(self, points, time=None):
            del time
            return np.broadcast_to(panel_jacobian, (len(points), 3, 3)).copy()

        def compute_source_velocity_f64(self, points):
            raise AssertionError("The Metal target trace must not launch an f64 Taichi kernel")

    class Physics:
        def compute_target_velocity_and_gradients_consistent(
            self, particles, points, *, include_freestream
        ):
            del particles, include_freestream
            return (
                np.zeros_like(points, dtype=np.float32),
                np.broadcast_to(particle_jacobian, (len(points), 3, 3)).copy(),
            )

    panel = Panel()
    solver = object.__new__(VPMSolver)
    solver.physics = Physics()
    solver.particles = object()
    solver.panel_solver = panel
    solver.vlm_solver = None
    solver.n_sources = 0
    solver.np_dtype = np.float32
    solver.time = 0.0
    solver._pressure_body_induced_fn = panel.compute_induced_velocity
    solver._body_induced_fn = lambda points, time: panel.compute_induced_velocity(points)

    points = np.array([[1.0, 0.1, -0.2], [1.2, -0.3, 0.4]])
    normals = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    expected_jacobian = particle_jacobian + panel_jacobian

    complete = solver._add_nonparticle_target_gradient(
        points,
        np.broadcast_to(particle_jacobian, (len(points), 3, 3)),
        particle_spacing=0.06,
    )
    np.testing.assert_allclose(complete, np.broadcast_to(expected_jacobian, (len(points), 3, 3)))

    _, tangential = solver.compute_velocity_and_tangential_normal_gradient_at_points(
        points, normals, particle_spacing=0.06
    )
    normal_gradient = np.einsum(
        "fij,fj->fi", np.broadcast_to(expected_jacobian, (len(points), 3, 3)), normals
    )
    expected_tangential = (
        normal_gradient - np.einsum("fi,fi->f", normal_gradient, normals)[:, None] * normals
    )
    np.testing.assert_allclose(tangential, expected_tangential)
