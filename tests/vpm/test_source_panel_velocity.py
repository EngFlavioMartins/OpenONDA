import numpy as np
import taichi as ti

from source.solvers.VPM import PanelSolver
from source.solvers.VPM.boundary_elements.panels.kernels.induced_velocity import (
    compute_source_induced_velocity_kernel,
)
from source.solvers.VPM.runtime.backend import reset_taichi_backend


def _cube_triangles() -> np.ndarray:
    faces = [
        ((-0.5, -0.5, -0.5), (0, 0, 1), (0, 1, 0)),
        ((0.5, -0.5, -0.5), (0, 1, 0), (0, 0, 1)),
        ((-0.5, -0.5, -0.5), (1, 0, 0), (0, 0, 1)),
        ((-0.5, 0.5, -0.5), (0, 0, 1), (1, 0, 0)),
        ((-0.5, -0.5, -0.5), (0, 1, 0), (1, 0, 0)),
        ((-0.5, -0.5, 0.5), (1, 0, 0), (0, 1, 0)),
    ]
    triangles = []
    for origin, u, v in faces:
        a = np.asarray(origin, dtype=np.float32)
        u = np.asarray(u, dtype=np.float32)
        v = np.asarray(v, dtype=np.float32)
        triangles.extend(((a, a + u, a + u + v), (a, a + u + v, a + v)))
    return np.asarray(triangles, dtype=np.float32)


def test_square_source_panel_matches_solid_angle():
    reset_taichi_backend()
    ti.init(arch=ti.cpu, default_fp=ti.f32, offline_cache=False)
    try:
        vertices = np.array(
            [
                [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0]],
                [[-0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]],
            ],
            dtype=np.float32,
        )
        normals = np.tile([0.0, 0.0, 1.0], (2, 1)).astype(np.float32)
        strengths = np.ones(2, dtype=np.float32)
        points = np.array([[0.0, 0.0, 2.0]], dtype=np.float32)
        velocity = np.zeros_like(points)

        compute_source_induced_velocity_kernel(vertices, normals, strengths, points, velocity)

        solid_angle = 4.0 * np.arctan(0.25 / (2.0 * np.sqrt(4.5)))
        np.testing.assert_allclose(velocity[0, :2], 0.0, atol=1e-7)
        np.testing.assert_allclose(velocity[0, 2], solid_angle / (4.0 * np.pi), rtol=2e-6)
    finally:
        reset_taichi_backend()


def test_neumann_source_panels_enforce_cube_impermeability():
    reset_taichi_backend()
    ti.init(arch=ti.cpu, default_fp=ti.f32, offline_cache=False)
    try:
        solver = PanelSolver(
            max_panels=16,
            float_dtype="f32",
            linear_solver="SCIPY",
            bc_type="NEUMANN",
        )
        solver._ensure_initialized()
        solver.lattice.add_body("cube", _cube_triangles())
        solver.lattice.update_geometry()
        solver.initialize(force=True)
        freestream = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        solver.solve(freestream, None, 0.0)

        count = solver.lattice.num_panels
        strengths = solver.lattice.source_strengths.to_numpy()[:count]
        areas = solver.lattice.areas.to_numpy()[:count]
        centres = solver.lattice.centers.to_numpy()[:count]
        normals = solver.lattice.normals.to_numpy()[:count]
        points = centres + 0.005 * normals
        velocity = freestream + solver.compute_induced_velocity(points)
        normal_velocity = np.sum(velocity * normals, axis=1)

        assert np.sqrt(np.mean(normal_velocity**2)) < 0.01
        assert abs(np.dot(strengths, areas)) < 1e-6
    finally:
        reset_taichi_backend()
