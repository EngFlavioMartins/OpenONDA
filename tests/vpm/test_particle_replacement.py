import numpy as np
import pytest
import taichi as ti

from source.solvers.VPM import Solver, VPMSetup
from source.solvers.VPM.acceleration import treecode_gpu
from source.solvers.VPM.acceleration.treecode_gpu import TaichiTreecode
from source.solvers.VPM.config.backend import reset_taichi_backend
from source.solvers.VPM.config.types import (
    AdvectionConfig,
    StabilizationConfig,
    StretchingConfig,
    ViscousConfig,
)
from source.solvers.VPM.particles.container import Particles


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


def test_bounds_removal_does_not_depend_on_device_tag_field(tmp_path):
    """Retention must use the downloaded positions, not stale GPU tags."""
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
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            dtype=np.float32,
        )
        zeros = np.zeros((3, 3), dtype=np.float32)
        scalar = np.ones(3, dtype=np.float32)
        solver.replace_vortex_particles(
            position=position,
            velocity=zeros,
            circulation=zeros,
            radius=scalar,
            volume=scalar,
            viscosity=scalar,
        )

        # Reproduce the failed Vulkan outcome: every device tag reads outside.
        solver.particles._removal_tags.fill(0)
        removed = solver.particles.remove_particles_by_bounds(
            [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0], invert_selection=True
        )

        assert removed == 1
        np.testing.assert_array_equal(solver.particles.position_cpu(), position[[0, 2]])
    finally:
        reset_taichi_backend()


def test_bounds_removal_noop_only_downloads_positions(tmp_path, monkeypatch):
    """A no-op retention pass must not transfer or replace the full cloud."""
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
            [[0.0, 0.0, 0.0], [0.5, -0.5, 0.25], [-0.75, 0.75, -0.5]],
            dtype=np.float32,
        )
        zeros = np.zeros((3, 3), dtype=np.float32)
        scalar = np.ones(3, dtype=np.float32)
        solver.replace_vortex_particles(
            position=position,
            velocity=zeros,
            circulation=zeros,
            radius=scalar,
            volume=scalar,
            viscosity=scalar,
        )

        particles = solver.particles
        extract_vector = particles._extract_vector
        position_reads = 0

        def position_only(field, count):
            nonlocal position_reads
            if field is not particles.position:
                raise AssertionError("no-op retention downloaded a non-position vector field")
            position_reads += 1
            return extract_vector(field, count)

        def unexpected_extract(*_args, **_kwargs):
            raise AssertionError("no-op retention downloaded a non-position particle field")

        def unexpected_replace(*_args, **_kwargs):
            raise AssertionError("no-op retention replaced the particle cloud")

        monkeypatch.setattr(particles, "_extract_vector", position_only)
        monkeypatch.setattr(particles, "_extract_scalar", unexpected_extract)
        monkeypatch.setattr(particles, "_extract_matrix", unexpected_extract)
        monkeypatch.setattr(particles, "_extract_int", unexpected_extract)
        monkeypatch.setattr(particles, "replace_from_numpy", unexpected_replace)

        removed = particles.remove_particles_by_bounds(
            [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0], invert_selection=True
        )

        assert removed == 0
        assert position_reads == 1
        assert particles.number_of_particles == 3
    finally:
        reset_taichi_backend()


def test_retention_compacts_vorticity_without_quadratic_reconstruction(tmp_path):
    """Domain retention must preserve stored omega without an O(N^2) rebuild."""
    reset_taichi_backend()
    try:
        solver = Solver(
            VPMSetup(
                processing_unit="CPU",
                stretching=StretchingConfig.disabled(),
                viscous=ViscousConfig(scheme="NONE"),
                advection=AdvectionConfig(scheme="NONE"),
                stabilization=StabilizationConfig.bounded_domain([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]),
                backup_frequency=0,
                logging_frequency=0,
                backup_directory=str(tmp_path),
                max_particles=16,
            )
        )
        position = np.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            dtype=np.float32,
        )
        circulation = np.array(
            [[0.0, 0.0, 0.1], [0.0, 0.0, 0.2], [0.0, 0.0, 0.3]],
            dtype=np.float32,
        )
        volume = np.array([0.01, 0.02, 0.03], dtype=np.float32)
        solver.replace_vortex_particles(
            position=position,
            velocity=np.zeros((3, 3), dtype=np.float32),
            circulation=circulation,
            radius=np.full(3, 0.1, dtype=np.float32),
            volume=volume,
            viscosity=np.full(3, 1.0e-5, dtype=np.float32),
        )

        def unexpected_reconstruction(_particles):
            raise AssertionError("retention must not run direct vorticity reconstruction")

        solver.physics.compute_vorticities = unexpected_reconstruction
        solver.stabilization.apply_retention()

        kept = np.array([0, 2])
        np.testing.assert_array_equal(solver.particles.position_cpu(), position[kept])
        np.testing.assert_allclose(
            solver.particles.vorticity_cpu(),
            circulation[kept] / volume[kept, None],
        )
    finally:
        reset_taichi_backend()


@pytest.mark.gpu
def test_vulkan_chunked_replacement_preserves_distinct_reused_buffers(monkeypatch):
    """A host staging buffer must not be overwritten before its GPU copy ends."""
    reset_taichi_backend()
    try:
        try:
            ti.init(arch=ti.vulkan, default_fp=ti.f32, default_ip=ti.i32)
        except Exception:
            pytest.skip("Vulkan backend unavailable")

        monkeypatch.setattr(Particles, "_COPY_CHUNK_SIZE", 4)
        n = 11
        index = np.arange(n, dtype=np.float32)
        position = np.column_stack((index, index + 100.0, index + 200.0))
        velocity = -position
        circulation = position * 1.0e-3
        radius = 0.1 + index * 0.001
        volume = 0.01 + index * 0.0001
        viscosity = 1.0e-3 + index * 1.0e-6
        viscosity_turbulent = index * 1.0e-5
        group_id = np.arange(n, dtype=np.int32) + 10
        zone_id = np.arange(n, dtype=np.int32) + 20

        particles = Particles(max_particles=32)
        particles.replace_from_numpy(
            position=position,
            velocity=velocity,
            circulation=circulation,
            radius=radius,
            volume=volume,
            viscosity=viscosity,
            viscosity_turbulent=viscosity_turbulent,
            group_id=group_id,
            zone_id=zone_id,
        )

        np.testing.assert_array_equal(particles.position_cpu(), position)
        np.testing.assert_array_equal(particles.velocity_cpu(), velocity)
        np.testing.assert_array_equal(particles.circulation_cpu(), circulation)
        np.testing.assert_array_equal(particles.radius_cpu(), radius)
        np.testing.assert_array_equal(particles.volume_cpu(), volume)
        np.testing.assert_array_equal(particles.viscosity_cpu(), viscosity)
        np.testing.assert_array_equal(particles.viscosity_turbulent_cpu(), viscosity_turbulent)
        np.testing.assert_array_equal(particles.group_id_cpu(), group_id)
        np.testing.assert_array_equal(particles.zone_id_cpu(), zone_id)

        # Full replacements use persistent native arrays that are distinct for
        # every Taichi field.  Sharing one external ndarray across template
        # fields caused circulation/position aliasing in the coupled cube run.
        position_upload = particles._native_vector_uploads[id(particles.position)]
        circulation_upload = particles._native_vector_uploads[id(particles.circulation)]
        assert position_upload is not circulation_upload
        position_download = particles._host_vector_chunks[("download", id(particles.position))]
        circulation_download = particles._host_vector_chunks[
            ("download", id(particles.circulation))
        ]
        assert position_download is not circulation_download
        assert (
            particles._native_scalar_uploads[id(particles.radius)]
            is not particles._native_scalar_uploads[id(particles.volume)]
        )
    finally:
        reset_taichi_backend()


@pytest.mark.gpu
def test_vulkan_treecode_traversal_uses_bounded_batches(monkeypatch):
    """All traversal variants must compile and fill every batch on Vulkan."""
    reset_taichi_backend()
    try:
        try:
            ti.init(arch=ti.vulkan, default_fp=ti.f32, default_ip=ti.i32)
        except Exception:
            pytest.skip("Vulkan backend unavailable")

        monkeypatch.setattr(treecode_gpu, "_TRAVERSAL_BATCH_SIZE", 4)
        rng = np.random.default_rng(71)
        n = 11
        position = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
        circulation = rng.normal(0.0, 0.1, size=(n, 3)).astype(np.float32)
        radius = np.full(n, 0.1, dtype=np.float32)
        tree = TaichiTreecode(
            max_particles=16,
            max_nodes=32,
            theta=0.3,
            kernel_type="GAUSSIAN",
        )
        tree.build(position, circulation, radius, force=True)

        velocity, gradient, strain = tree.compute_velocity_and_gradient(
            np.zeros(3, dtype=np.float32)
        )
        target_velocity = tree.compute_target_velocities(position[:9])
        target_gradient = tree.compute_target_velocity_gradients(position[:9])

        assert velocity.shape == (n, 3)
        assert gradient.shape == (n, 3, 3)
        assert strain.shape == (n, 3, 3)
        assert target_velocity.shape == (9, 3)
        assert target_gradient.shape == (9, 3, 3)
        assert np.all(np.isfinite(velocity))
        assert np.all(np.isfinite(gradient))
        assert np.all(np.isfinite(target_velocity))
        assert np.all(np.isfinite(target_gradient))
    finally:
        reset_taichi_backend()
