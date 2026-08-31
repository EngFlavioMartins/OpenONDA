"""Analytic verification of the panel solver against potential flow past a sphere.

The sphere is the reference case with a closed-form incompressible potential
solution, so it pins down the parts of the formulation that a residual check
alone cannot: that the boundary condition actually holds on the surface, that
the surface speed matches ``1.5 U sin(theta)``, that the pressure follows
``1 - 2.25 sin^2(theta)``, that a closed body in potential flow carries no
net force (D'Alembert), and that all of it converges under refinement.
"""

from __future__ import annotations

import numpy as np
import pytest
import taichi

from source.solvers.vpm.boundary_elements.panels.geometry.stl_io import save_stl  # noqa: E402
from source.solvers.vpm.boundary_elements.panels.solver.panel_solver import (  # noqa: E402
    PanelSolver,
)

FREESTREAM_SPEED = 1.0


def _icosphere_triangles(subdivisions: int = 1, radius: float = 1.0) -> np.ndarray:
    """A closed icosphere: near-uniform, near-equilateral triangles.

    Preferred over a UV sphere for convergence work, whose panels degenerate
    into slivers at the poles and pollute the error norm.
    """
    golden = (1.0 + 5.0**0.5) / 2.0
    vertex = np.array(
        [
            [-1, golden, 0],
            [1, golden, 0],
            [-1, -golden, 0],
            [1, -golden, 0],
            [0, -1, golden],
            [0, 1, golden],
            [0, -1, -golden],
            [0, 1, -golden],
            [golden, 0, -1],
            [golden, 0, 1],
            [-golden, 0, -1],
            [-golden, 0, 1],
        ],
        dtype=float,
    )
    vertex /= np.linalg.norm(vertex, axis=1)[:, None]
    face = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ]

    for _ in range(subdivisions):
        vertex, face = _subdivide_on_unit_sphere(vertex, face)

    return (radius * vertex)[np.array(face)]


def _midpoint_on_unit_sphere(
    a: int, b: int, vertices: list, cache: dict[tuple[int, int], int]
) -> int:
    """Index of the edge midpoint of ``a``-``b``, projected onto the sphere."""
    key = (min(a, b), max(a, b))
    if key not in cache:
        point = vertices[a] + vertices[b]
        vertices.append(point / np.linalg.norm(point))
        cache[key] = len(vertices) - 1
    return cache[key]


def _subdivide_on_unit_sphere(vertex: np.ndarray, face: list) -> tuple[np.ndarray, list]:
    """Split every triangle into four, projecting the new vertices outward."""
    vertices = list(vertex)
    cache: dict[tuple[int, int], int] = {}
    subdivided = []
    for a, b, c in face:
        ab = _midpoint_on_unit_sphere(a, b, vertices, cache)
        bc = _midpoint_on_unit_sphere(b, c, vertices, cache)
        ca = _midpoint_on_unit_sphere(c, a, vertices, cache)
        subdivided += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
    return np.array(vertices), subdivided


def _ensure_taichi_cpu() -> None:
    if taichi.lang.impl.get_runtime().prog is None:
        taichi.init(arch=taichi.cpu)


def _solved_sphere(
    tmp_path,
    subdivisions: int,
    linear_solver: str = "SCIPY",
    float_dtype: str = "f64",
) -> PanelSolver:
    _ensure_taichi_cpu()
    triangles = _icosphere_triangles(subdivisions)
    stl_path = tmp_path / f"icosphere_{subdivisions}_{linear_solver}.stl"
    save_stl(str(stl_path), triangles)

    freestream_velocity = np.array([FREESTREAM_SPEED, 0.0, 0.0])
    panel = PanelSolver(
        max_n_panels=triangles.shape[0] + 8,
        float_dtype=float_dtype,
        boundary_condition_type="NEUMANN",
        linear_solver=linear_solver,
        density=1.0,
        freestream_velocity=freestream_velocity,
        coupling_scope="normal",
    )
    panel.add_surface("sphere", str(stl_path))
    panel.solve(freestream_velocity, None, 0.0)
    panel.compute_postprocess(freestream_velocity, freestream_velocity, 1.0)
    return panel


def _surface_state(panel: PanelSolver):
    n = panel.lattice.n_panels
    centre = panel.lattice.panel_centre.to_numpy()[:n]
    normal = panel.lattice.normal.to_numpy()[:n]
    surface_velocity = panel.surface_velocity.to_numpy()[:n]
    pressure_coefficient = panel.lattice.pressure_coefficient.to_numpy()[:n]
    polar_angle = np.arccos(np.clip(centre[:, 0] / np.linalg.norm(centre, axis=1), -1.0, 1.0))
    return normal, surface_velocity, pressure_coefficient, polar_angle


def test_no_penetration_holds_on_the_surface(tmp_path):
    panel = _solved_sphere(tmp_path, subdivisions=2)
    normal, surface_velocity, _, _ = _surface_state(panel)

    normal_velocity = np.einsum("ij,ij->i", surface_velocity, normal)
    residual = float(np.sqrt(np.mean(normal_velocity**2)))
    # What the wall has to cancel: the freestream's own normal component.
    incident = float(np.sqrt(np.mean((normal @ np.array([FREESTREAM_SPEED, 0.0, 0.0])) ** 2)))

    # The solved condition is exactly u.n = 0 at every collocation point, so
    # the evaluated surface field must reproduce it down to arithmetic noise.
    # Stated as a ratio because the achievable floor tracks Taichi's
    # default_fp, not the panel field dtype. A regression where the surface
    # evaluation disagrees with the solved operator (for instance by taking
    # the wrong branch of the source panel's self-term) puts this ratio at
    # order one rather than at round-off.
    assert residual / incident < 1.0e-5


def test_surface_speed_matches_the_analytic_sphere_solution(tmp_path):
    panel = _solved_sphere(tmp_path, subdivisions=2)
    _, surface_velocity, _, polar_angle = _surface_state(panel)

    analytic = 1.5 * FREESTREAM_SPEED * np.sin(polar_angle)
    away_from_stagnation = analytic > 0.2
    speed = np.linalg.norm(surface_velocity, axis=1)
    relative_error = (
        np.abs(speed[away_from_stagnation] - analytic[away_from_stagnation])
        / (analytic[away_from_stagnation])
    )

    assert np.mean(relative_error) < 0.02


def test_pressure_coefficient_matches_the_analytic_sphere_solution(tmp_path):
    panel = _solved_sphere(tmp_path, subdivisions=2)
    _, _, pressure_coefficient, polar_angle = _surface_state(panel)

    analytic = 1.0 - 2.25 * np.sin(polar_angle) ** 2
    away_from_stagnation = np.sin(polar_angle) > 0.15

    assert (
        np.mean(np.abs(pressure_coefficient[away_from_stagnation] - analytic[away_from_stagnation]))
        < 0.05
    )


def test_closed_body_carries_no_net_force(tmp_path):
    panel = _solved_sphere(tmp_path, subdivisions=2)
    n = panel.lattice.n_panels
    total_force = panel.panel_force.to_numpy()[:n].sum(axis=0)

    # D'Alembert's paradox: steady potential flow past a closed body produces
    # zero force. This is an independent check on the pressure integration
    # that no residual or boundary-condition test can provide.
    dynamic_reference = 0.5 * 1.0 * FREESTREAM_SPEED**2 * np.pi
    assert np.linalg.norm(total_force) / dynamic_reference < 1.0e-4


def test_surface_solution_converges_under_refinement(tmp_path):
    errors = []
    for subdivisions in (1, 2):
        panel = _solved_sphere(tmp_path, subdivisions=subdivisions)
        _, surface_velocity, _, polar_angle = _surface_state(panel)
        analytic = 1.5 * FREESTREAM_SPEED * np.sin(polar_angle)
        away = analytic > 0.2
        speed = np.linalg.norm(surface_velocity, axis=1)
        errors.append(float(np.mean(np.abs(speed[away] - analytic[away]) / analytic[away])))

    assert errors[1] < errors[0]


def test_gpu_and_cpu_linear_solvers_agree(tmp_path):
    cpu = _solved_sphere(tmp_path, subdivisions=1, linear_solver="SCIPY")
    gpu = _solved_sphere(tmp_path, subdivisions=1, linear_solver="BICGSTAB_GPU")

    n = cpu.lattice.n_panels
    assert gpu.lattice.n_panels == n
    cpu_strength = cpu.lattice.source_strength.to_numpy()[:n]
    gpu_strength = gpu.lattice.source_strength.to_numpy()[:n]

    # The GPU iterative solver must reach the same solution as the dense
    # direct solve, not merely a self-consistent one. Compared as a
    # normalized norm rather than element-wise: individual source strengths
    # pass through zero around the sphere's equator, where a relative
    # element-wise tolerance is meaningless.
    difference = np.linalg.norm(gpu_strength - cpu_strength) / np.linalg.norm(cpu_strength)
    assert difference < 1.0e-5
    diagnostic = gpu.results["diagnostic_history"][-1]
    assert diagnostic["linear_solver"] == "ProjectedCGLS(Taichi)"
    assert diagnostic["linear_solver_success"]
    assert diagnostic["iterations"] > 0
    assert diagnostic["relative_constraint_residual"] < 1.0e-10


def test_logged_no_penetration_accuracy_tracks_panel_precision(tmp_path):
    """f64 panel arithmetic escapes the default_fp=f32 accuracy ceiling."""
    f32 = _solved_sphere(tmp_path, subdivisions=1, float_dtype="f32")
    f64 = _solved_sphere(tmp_path, subdivisions=1, float_dtype="f64")

    measured = {}
    for dtype, panel in (("f32", f32), ("f64", f64)):
        normal, surface_velocity, _, _ = _surface_state(panel)
        value = float(np.sqrt(np.mean(np.einsum("ij,ij->i", surface_velocity, normal) ** 2)))
        measured[dtype] = value
        logged = panel.results["diagnostic_history"][-1]["no_penetration_residual"]
        assert logged == pytest.approx(value, rel=1e-12, abs=1e-20)

    assert measured["f64"] < 1.0e-10
    assert measured["f32"] > 1.0e-8
