"""Device realignment must publish its mutation to host readers and induction."""

import numpy as np
import taichi as ti

from source.solvers.vpm.particles.container import Particles
from source.solvers.vpm.stabilization.operators import StabilizationOperators


def test_realignment_invalidates_cached_strength_and_preserves_norm():
    ti.init(arch=ti.cpu, offline_cache=False)
    try:
        particles = Particles(max_n_particles=8)
        particles.add_vortex_particles(
            position=np.zeros((1, 3)),
            velocity=np.zeros((1, 3)),
            vortex_strength=np.array([[1.0, 0.0, 0.0]]),
            core_radius=np.ones(1),
            particle_volume=np.ones(1),
            kinematic_viscosity=np.zeros(1),
        )
        gradient = np.zeros((8, 3, 3), dtype=np.float32)
        gradient[0, 0, 2] = 1.0  # curl(u) = (0, 1, 0)
        particles.velocity_gradient.from_numpy(gradient)
        before = particles.vortex_strength_cpu().copy()
        revision = particles.state_revision
        operator = StabilizationOperators(ti.f32, 8)
        operator.apply_pedrizzetti_relaxation(particles, 0.3)
        after = particles.vortex_strength_cpu()
        assert particles.state_revision > revision
        assert not np.allclose(after, before)
        np.testing.assert_allclose(
            after[0], np.array([0.7, 0.3, 0.0]) / np.hypot(0.7, 0.3), rtol=1e-6
        )
        np.testing.assert_allclose(np.linalg.norm(after, axis=1), 1.0, rtol=1e-6)
    finally:
        ti.reset()
