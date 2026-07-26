import numpy as np

from source.solvers.VPM import Solver, VPMSetup
from source.solvers.VPM.config.backend import reset_taichi_backend
from source.solvers.VPM.config.types import AdvectionConfig, StretchingConfig, ViscousConfig


def test_replace_vortex_particles_matches_uploaded_cloud(tmp_path):
    reset_taichi_backend()
    try:
        solver = Solver(
            VPMSetup(
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
        group_id = np.array([4, 5, 6], dtype=np.int32)
        velocity_gradient = np.arange(27, dtype=np.float32).reshape(3, 3, 3) * 0.01
        strain_rate = np.arange(27, dtype=np.float32).reshape(3, 3, 3) * 0.02

        solver.replace_vortex_particles(
            position=position,
            velocity=velocity,
            circulation=circulation,
            radius=radius,
            volume=volume,
            viscosity=viscosity,
            group_id=group_id,
            zone_id=zone_id,
            velocity_gradient=velocity_gradient,
            strain_rate=strain_rate,
        )

        assert solver.particles.number_of_particles == 3
        assert solver.particles.device_number_of_particles[None] == 3
        np.testing.assert_allclose(solver.particles_positions, position)
        np.testing.assert_allclose(solver.particles_velocities, velocity)
        np.testing.assert_allclose(solver.particles_circulation, circulation)
        np.testing.assert_allclose(solver.particles_radii, radius)
        np.testing.assert_allclose(solver.particles_volumes, volume)
        np.testing.assert_allclose(solver.particles_viscosities, viscosity)
        np.testing.assert_allclose(solver.particles_vorticities, circulation / volume[:, None])
        np.testing.assert_array_equal(solver.particles.group_id_cpu(), group_id)
        np.testing.assert_array_equal(solver.particles.zone_id_cpu(), zone_id)
        np.testing.assert_allclose(solver.particles.velocity_gradient_cpu(), velocity_gradient)
        np.testing.assert_allclose(solver.particles.strain_rate_cpu(), strain_rate)

        solver.replace_vortex_particles(
            position=np.empty((0, 3), dtype=np.float32),
            velocity=np.empty((0, 3), dtype=np.float32),
            circulation=np.empty((0, 3), dtype=np.float32),
            radius=np.empty(0, dtype=np.float32),
            volume=np.empty(0, dtype=np.float32),
            viscosity=np.empty(0, dtype=np.float32),
        )
        assert solver.particles.number_of_particles == 0
        assert solver.particles.device_number_of_particles[None] == 0
    finally:
        reset_taichi_backend()


def test_bounds_removal_uses_compacted_replacement(tmp_path):
    reset_taichi_backend()
    try:
        solver = Solver(
            VPMSetup(
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

        position = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        velocity = np.zeros((3, 3), dtype=np.float32)
        circulation = np.array(
            [[0.0, 0.0, 0.1], [0.0, 0.0, 0.2], [0.0, 0.0, 0.3]],
            dtype=np.float32,
        )
        radius = np.full(3, 0.1, dtype=np.float32)
        volume = np.full(3, 0.01, dtype=np.float32)
        viscosity = np.full(3, 1.0e-5, dtype=np.float32)
        viscosity_turbulent = np.array([0.0, 1.0e-5, 2.0e-5], dtype=np.float32)
        group_id = np.array([10, 11, 12], dtype=np.int32)
        zone_id = np.array([20, 21, 22], dtype=np.int32)
        velocity_gradient = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
        strain_rate = velocity_gradient * 0.5

        solver.replace_vortex_particles(
            position=position,
            velocity=velocity,
            circulation=circulation,
            radius=radius,
            volume=volume,
            viscosity=viscosity,
            viscosity_turbulent=viscosity_turbulent,
            group_id=group_id,
            zone_id=zone_id,
            velocity_gradient=velocity_gradient,
            strain_rate=strain_rate,
        )

        removed = solver.particles.remove_particles_by_bounds([0.5, 1.5, -1.0, 1.0, -1.0, 1.0])

        assert removed == 1
        assert solver.particles.number_of_particles == 2
        assert solver.particles.device_number_of_particles[None] == 2
        np.testing.assert_allclose(solver.particles_positions, position[[0, 2]])
        np.testing.assert_array_equal(solver.particles.group_id_cpu(), group_id[[0, 2]])
        np.testing.assert_array_equal(solver.particles.zone_id_cpu(), zone_id[[0, 2]])
        np.testing.assert_allclose(
            solver.particles.velocity_gradient_cpu(),
            velocity_gradient[[0, 2]],
        )
        np.testing.assert_allclose(solver.particles.strain_rate_cpu(), strain_rate[[0, 2]])
        np.testing.assert_allclose(
            solver.particles.viscosity_effective_cpu(),
            viscosity[[0, 2]] + viscosity_turbulent[[0, 2]],
        )
    finally:
        reset_taichi_backend()
