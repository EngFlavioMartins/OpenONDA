from types import SimpleNamespace

import numpy as np

from source.solvers.vpm.core.evolution import EvolutionStepper


class _Particles:
    def __init__(self):
        self.position = np.array(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
            dtype=np.float32,
        )
        self.vortex_strength = np.array(
            [[1.0, 0.0, 0.0], [-1.0 + 1.0e-6, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
            dtype=np.float32,
        )
        self.core_radius = np.full(4, 0.1, dtype=np.float32)
        self.particle_volume = np.ones(4, dtype=np.float32)
        self.effective_viscosity = np.full(4, 1.0e-3, dtype=np.float32)

    def __len__(self):
        return len(self.position)

    def position_cpu(self, **_):
        return self.position.copy()

    def vortex_strength_cpu(self, **_):
        return self.vortex_strength.copy()

    def core_radius_cpu(self, **_):
        return self.core_radius.copy()

    def particle_volume_cpu(self, **_):
        return self.particle_volume.copy()


def test_core_spreading_skips_subprecision_moment_correction():
    particles = _Particles()
    updates = []

    def spread(active_particles, time_step_size):
        active_particles.core_radius = np.sqrt(
            active_particles.core_radius**2
            + 4.0 * active_particles.effective_viscosity * time_step_size
        )

    solver = SimpleNamespace(
        particles=particles,
        physics=SimpleNamespace(
            _angular_core_coefficient=1.0 / 3.0,
            core_spreading_diffusion=spread,
        ),
        stretching_conserve_moments=True,
        np_dtype=np.float32,
        core_spreading_correction_relative=np.nan,
        update_particle_vortex_strength=lambda *args: updates.append(args),
    )

    EvolutionStepper(solver)._apply_core_spreading_diffusion(0.01)

    assert updates == []
    assert solver.core_spreading_correction_relative == 0.0
    assert np.all(particles.core_radius > 0.1)
