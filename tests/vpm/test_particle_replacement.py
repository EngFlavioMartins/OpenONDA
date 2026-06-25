import numpy as np

from source.solvers.VPM import Solver, SolverConfig
from source.solvers.VPM.config.backend import reset_taichi_backend
from source.solvers.VPM.config.types import AdvectionConfig, StretchingConfig, ViscousConfig


def test_replace_vortex_particles_matches_uploaded_cloud(tmp_path):
    reset_taichi_backend()
    try:
        solver = Solver(
            SolverConfig(
                processing_unit="CPU",
                stretching=StretchingConfig.disabled(),
                viscous=ViscousConfig(scheme="NONE"),
                advection=AdvectionConfig(scheme="NONE"),
                backup_frequency=0,
                logging_frequency=0,
                backup_directory=str(tmp_path),
                max_particles=16,
            )
        )

        pos0 = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        solver.add_vortex_particles(
            position=pos0,
            velocity=np.zeros((1, 3), dtype=np.float32),
            circulation=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
            radius=np.array([0.1], dtype=np.float32),
            volume=np.array([0.01], dtype=np.float32),
            viscosity=np.zeros(1, dtype=np.float32),
        )

        position = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            dtype=np.float32,
        )
        velocity = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        circulation = np.array(
            [[0.01, 0.02, 0.03], [0.04, 0.05, 0.06], [0.07, 0.08, 0.09]],
            dtype=np.float32,
        )
        radius = np.full(3, 0.15, dtype=np.float32)
        volume = np.full(3, 0.02, dtype=np.float32)
        viscosity = np.full(3, 1.0e-3, dtype=np.float32)
        zone_id = np.array([1, 2, 3], dtype=np.int32)

        solver.replace_vortex_particles(
            position=position,
            velocity=velocity,
            circulation=circulation,
            radius=radius,
            volume=volume,
            viscosity=viscosity,
            zone_id=zone_id,
        )

        assert solver.particles.number_of_particles == 3
        np.testing.assert_allclose(solver.particles_positions, position)
        np.testing.assert_allclose(solver.particles_velocities, velocity)
        np.testing.assert_allclose(solver.particles_circulation, circulation)
        np.testing.assert_allclose(solver.particles_radii, radius)
        np.testing.assert_allclose(solver.particles_volumes, volume)
        np.testing.assert_allclose(solver.particles_viscosities, viscosity)
        np.testing.assert_allclose(solver.particles_vorticities, circulation / volume[:, None])
    finally:
        reset_taichi_backend()
