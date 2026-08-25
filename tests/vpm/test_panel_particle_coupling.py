"""The panel body must deflect every active particle, identically and on device."""

from __future__ import annotations

import numpy as np
import pytest

taichi = pytest.importorskip("taichi", reason="VPM requires taichi")

from test_panel_solver_sphere_analytic import _icosphere_triangles  # noqa: E402

from source.solvers.vpm.boundary_elements.panels.geometry.stl_io import save_stl  # noqa: E402
from source.solvers.vpm.boundary_elements.panels.solver.panel_solver import (  # noqa: E402
    PanelSolver,
)


def _ensure_taichi_cpu() -> None:
    if taichi.lang.impl.get_runtime().prog is None:
        taichi.init(arch=taichi.cpu)


def _solved_body(tmp_path, subdivisions: int = 1) -> PanelSolver:
    _ensure_taichi_cpu()
    triangles = _icosphere_triangles(subdivisions)
    stl_path = tmp_path / f"body_{subdivisions}.stl"
    save_stl(str(stl_path), triangles)
    freestream_velocity = np.array([1.0, 0.0, 0.0])
    panel = PanelSolver(
        max_n_panels=triangles.shape[0] + 8,
        float_dtype="f32",
        boundary_condition_type="NEUMANN",
        linear_solver="BICGSTAB_GPU",
        density=1.0,
        freestream_velocity=freestream_velocity,
        coupling_scope="full",
    )
    panel.add_surface("body", str(stl_path))
    panel.solve(freestream_velocity, None, 0.0)
    return panel


def _scatter_shell(n_particles: int, seed: int = 0) -> np.ndarray:
    """Points on a shell outside the unit body, spread over all directions."""
    rng = np.random.RandomState(seed)
    direction = rng.normal(size=(n_particles, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    radius = 1.5 + rng.rand(n_particles) * 2.0
    return (direction * radius[:, None]).astype(np.float32)


def test_every_active_particle_is_deflected(tmp_path):
    panel = _solved_body(tmp_path)
    n_particles = 5000
    position = taichi.Vector.field(3, taichi.f32, shape=n_particles)
    velocity = taichi.Vector.field(3, taichi.f32, shape=n_particles)
    position.from_numpy(_scatter_shell(n_particles))
    velocity.fill(0.0)

    panel.accumulate_induced_velocity_on_field(position, velocity, n_particles)
    induced = velocity.to_numpy()

    # Not "most" particles and not "the first block" — every single active
    # particle must receive a finite, non-zero body-induced velocity.
    assert np.all(np.isfinite(induced))
    assert np.all(np.linalg.norm(induced, axis=1) > 0.0)


def test_deflection_does_not_depend_on_particle_ordering(tmp_path):
    """No injected/retained distinction: index position must not matter."""
    panel = _solved_body(tmp_path)
    n_particles = 2000
    points = _scatter_shell(n_particles, seed=1)

    def induced_for(sample: np.ndarray) -> np.ndarray:
        position = taichi.Vector.field(3, taichi.f32, shape=sample.shape[0])
        velocity = taichi.Vector.field(3, taichi.f32, shape=sample.shape[0])
        position.from_numpy(sample)
        velocity.fill(0.0)
        panel.accumulate_induced_velocity_on_field(position, velocity, sample.shape[0])
        return velocity.to_numpy()

    forward = induced_for(points)
    permutation = np.random.RandomState(2).permutation(n_particles)
    shuffled = induced_for(points[permutation])

    np.testing.assert_allclose(shuffled, forward[permutation], rtol=1e-5, atol=1e-7)


def test_partial_particle_count_leaves_the_tail_untouched(tmp_path):
    """Only the active prefix is written; capacity beyond it stays clean."""
    panel = _solved_body(tmp_path)
    capacity, active = 800, 500
    position = taichi.Vector.field(3, taichi.f32, shape=capacity)
    velocity = taichi.Vector.field(3, taichi.f32, shape=capacity)
    position.from_numpy(_scatter_shell(capacity, seed=3))
    velocity.fill(0.0)

    panel.accumulate_induced_velocity_on_field(position, velocity, active)
    induced = velocity.to_numpy()

    assert np.all(np.linalg.norm(induced[:active], axis=1) > 0.0)
    assert np.all(induced[active:] == 0.0)


def test_device_path_matches_the_host_evaluation(tmp_path):
    """The device kernel must reproduce the host query, not approximate it."""
    panel = _solved_body(tmp_path)
    points = _scatter_shell(1500, seed=4)

    position = taichi.Vector.field(3, taichi.f32, shape=points.shape[0])
    velocity = taichi.Vector.field(3, taichi.f32, shape=points.shape[0])
    position.from_numpy(points)
    velocity.fill(0.0)
    panel.accumulate_induced_velocity_on_field(position, velocity, points.shape[0])

    np.testing.assert_allclose(
        velocity.to_numpy(), panel.compute_induced_velocity(points), rtol=1e-4, atol=1e-6
    )


def test_accumulation_adds_to_existing_velocity(tmp_path):
    """The hook accumulates onto the self-induced field, never overwrites it."""
    panel = _solved_body(tmp_path)
    points = _scatter_shell(400, seed=5)
    n_particles = points.shape[0]
    baseline = np.tile(np.array([[1.0, 2.0, 3.0]], dtype=np.float32), (n_particles, 1))

    position = taichi.Vector.field(3, taichi.f32, shape=n_particles)
    velocity = taichi.Vector.field(3, taichi.f32, shape=n_particles)
    position.from_numpy(points)

    velocity.fill(0.0)
    panel.accumulate_induced_velocity_on_field(position, velocity, n_particles)
    induced_only = velocity.to_numpy()

    velocity.from_numpy(baseline)
    panel.accumulate_induced_velocity_on_field(position, velocity, n_particles)

    np.testing.assert_allclose(velocity.to_numpy(), baseline + induced_only, rtol=1e-5, atol=1e-6)


def test_complex_many_panel_body_is_handled(tmp_path):
    """A refined body must still solve and deflect every particle."""
    panel = _solved_body(tmp_path, subdivisions=3)
    assert panel.lattice.n_panels == 1280
    assert panel.results["diagnostic_history"][-1]["linear_solver_success"]

    n_particles = 3000
    position = taichi.Vector.field(3, taichi.f32, shape=n_particles)
    velocity = taichi.Vector.field(3, taichi.f32, shape=n_particles)
    position.from_numpy(_scatter_shell(n_particles, seed=6))
    velocity.fill(0.0)

    panel.accumulate_induced_velocity_on_field(position, velocity, n_particles)
    induced = velocity.to_numpy()
    assert np.all(np.isfinite(induced))
    assert np.all(np.linalg.norm(induced, axis=1) > 0.0)
