"""Host bounds removal preserves every retained field and invalidates caches."""

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.particles.container import Particles


@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("invert", [False, True])
def test_bounds_removal_preserves_fields_and_cache(precision, invert):
    ti.init(arch=ti.cpu, offline_cache=False)
    try:
        dtype = np.float32 if precision == "f32" else np.float64
        particles = Particles(max_n_particles=8, float_dtype=precision)
        positions = np.array([[-2, 0, 0], [-1, 0, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=dtype)
        state = {
            "position": positions,
            "velocity": np.arange(15, dtype=dtype).reshape(5, 3),
            "vortex_strength": np.arange(15, dtype=dtype).reshape(5, 3) * 0.01,
            "core_radius": np.linspace(0.1, 0.2, 5, dtype=dtype),
            "particle_volume": np.linspace(0.01, 0.02, 5, dtype=dtype),
            "kinematic_viscosity": np.linspace(0.001, 0.002, 5, dtype=dtype),
            "eddy_viscosity": np.linspace(0.002, 0.004, 5, dtype=dtype),
            "group_id": np.arange(5, dtype=np.int32),
            "zone_id": np.arange(5, dtype=np.int32) + 5,
            "velocity_gradient": np.arange(45, dtype=dtype).reshape(5, 3, 3),
            "strain_rate": -np.arange(45, dtype=dtype).reshape(5, 3, 3),
        }
        particles.replace_from_numpy(**state)
        particles.position_cpu()
        revision = particles.state_revision
        retained = np.array([1, 2, 3] if invert else [0, 4])
        assert particles.remove_particles_by_bounds(
            [-1, 1, -1, 1, -1, 1], invert_selection=invert
        ) == 5 - len(retained)
        assert particles.state_revision > revision
        assert particles.n_particles_total == len(retained)
        for name, expected in state.items():
            np.testing.assert_array_equal(getattr(particles, name + "_cpu")(), expected[retained])
        revision = particles.state_revision
        assert particles.remove_particles_by_bounds([10, 11, 10, 11, 10, 11]) == 0
        assert particles.state_revision == revision
        assert particles.remove_particles_by_bounds([-3, 3, -3, 3, -3, 3]) == len(retained)
        assert particles.n_particles_total == 0
        assert particles.remove_particles_by_bounds([-3, 3, -3, 3, -3, 3]) == 0
    finally:
        ti.reset()
