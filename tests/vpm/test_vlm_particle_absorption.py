"""VLM collision/absorption must work against the canonical particle container.

These tests exercise ``VLMSolver.absorb_particles`` with the live ``Particles``
container so that stale field accesses (e.g. ``particles.circulation`` or
``particles.radius``) fail loudly here instead of inside a coupled run.
"""

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.boundary_elements.vlm.config import VLMSetup, VLMSurfaceSetup
from source.solvers.vpm.boundary_elements.vlm.geometry.aircraft import Aircraft, Wing, WingSegment
from source.solvers.vpm.boundary_elements.vlm.solver.vlm_solver import VLMSolver
from source.solvers.vpm.particles.container import Particles


def _plate(*, chord: float = 2.0) -> Aircraft:
    segment = WingSegment(
        uid="segment",
        vertex_position={
            "a": np.array([0.0, 0.0, 0.0]),
            "b": np.array([0.0, 1.0, 0.0]),
            "c": np.array([chord, 1.0, 0.0]),
            "d": np.array([chord, 0.0, 0.0]),
        },
        n_chordwise_panels=4,
        n_spanwise_panels=1,
    )
    wing = Wing(uid="wing")
    wing.add_segment(segment)
    aircraft = Aircraft(uid="plate")
    aircraft.add_wing(wing)
    return aircraft


@pytest.fixture
def taichi_f64():
    """Provide a clean CPU runtime whose default floating-point type is f64."""
    from source.solvers.vpm.runtime.backend import reset_taichi_backend

    reset_taichi_backend()
    ti.init(arch=ti.cpu, default_fp=ti.f64)
    yield
    reset_taichi_backend()


def _seed_particles(n_far: int = 3) -> Particles:
    particles = Particles(max_n_particles=64, float_dtype="f64")
    hitting = np.array([[0.25, 0.5, 0.0]])
    far = np.column_stack(
        (
            np.full(n_far, -5.0),
            np.linspace(0.1, 0.9, n_far),
            np.full(n_far, 3.0),
        )
    )
    position = np.vstack((hitting, far))
    n = len(position)
    particles.add_vortex_particles(
        position=position,
        velocity=np.zeros((n, 3)),
        vortex_strength=np.full((n, 3), [0.0, 0.0, 1.0e-3]),
        core_radius=np.full(n, 0.05),
        particle_volume=np.full(n, 1.0e-3),
        kinematic_viscosity=np.full(n, 1.5e-5),
        eddy_viscosity=np.full(n, 0.0),
        group_id=np.zeros(n, dtype=int),
        zone_id=np.zeros(n, dtype=int),
    )
    return particles


def test_absorb_removes_only_impinging_particles(taichi_f64):
    vlm = VLMSolver(VLMSetup(surfaces=(VLMSurfaceSetup(_plate()),)))
    vlm.generate_mesh()
    particles = _seed_particles()

    removed = vlm.absorb_particles(particles, tolerance=0.05)

    assert removed == 1
    assert particles.n_particles_total == 3

    state = particles.to_numpy_dict() if hasattr(particles, "to_numpy_dict") else None
    if state is None:
        n = particles.n_particles_total
        state = {
            "position": particles.position.to_numpy()[:n],
            "vortex_strength": particles.vortex_strength.to_numpy()[:n],
            "core_radius": particles.core_radius.to_numpy()[:n],
            "eddy_viscosity": particles.eddy_viscosity.to_numpy()[:n],
        }
    position = np.asarray(state["position"])
    assert not np.any(np.all(np.isclose(position, [0.25, 0.5, 0.0], atol=1e-12), axis=1))
    assert np.allclose(np.asarray(state["vortex_strength"])[:, 2], 1.0e-3)
    assert np.allclose(np.asarray(state["core_radius"]), 0.05)


def test_absorb_with_no_hits_leaves_container_untouched(taichi_f64):
    vlm = VLMSolver(VLMSetup(surfaces=(VLMSurfaceSetup(_plate()),)))
    vlm.generate_mesh()
    particles = Particles(max_n_particles=64, float_dtype="f64")
    position = np.array(
        [
            [-5.0, 0.25, 3.0],
            [-5.0, 0.50, 3.0],
            [0.25, 0.50, 0.50],
            [6.0, 0.75, -2.0],
        ]
    )
    n = len(position)
    particles.add_vortex_particles(
        position=position,
        velocity=np.zeros((n, 3)),
        vortex_strength=np.full((n, 3), [0.0, 0.0, 1.0e-3]),
        core_radius=np.full(n, 0.05),
        particle_volume=np.full(n, 1.0e-3),
        kinematic_viscosity=np.full(n, 1.5e-5),
    )
    before = particles.position.to_numpy()[: particles.n_particles_total].copy()

    removed = vlm.absorb_particles(particles, tolerance=0.05)

    assert removed == 0
    assert particles.n_particles_total == 4
    np.testing.assert_array_equal(
        particles.position.to_numpy()[: particles.n_particles_total], before
    )
