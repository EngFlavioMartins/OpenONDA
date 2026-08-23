import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm import VPMSetup, VPMSolver
from source.solvers.vpm.acceleration import treecode_gpu
from source.solvers.vpm.acceleration.treecode_gpu import TaichiTreecode
from source.solvers.vpm.config.types import (
    AdvectionConfig,
    StabilizationConfig,
    StretchingConfig,
    ViscousConfig,
)
from source.solvers.vpm.io.logging import Logging
from source.solvers.vpm.particles.container import Particles
from source.solvers.vpm.runtime.backend import reset_taichi_backend


def test_population_log_reports_physical_operation_not_storage_type(monkeypatch):
    messages = []
    monkeypatch.setattr(Logging, "message", messages.append)
    particles = type("Population", (), {"n_particles_total": 4, "capacity": 16})()

    Particles._log_population(particles, "previous_count=9")

    assert messages == [
        "[VPM][Particles] previous_count=9 count=4 capacity=16 utilization_pct=25.0"
    ]
    assert "array" not in messages[0].lower()


def test_weak_particle_removal_uses_cloud_wide_maximum(tmp_path):
    reset_taichi_backend()
    try:
        solver = VPMSolver(
            VPMSetup(
                compute_device="CPU",
                stretching=StretchingConfig.disabled(),
                viscous=ViscousConfig(scheme="NONE"),
                advection=AdvectionConfig(scheme="NONE"),
                checkpoint_interval_steps=0,
                logging_interval_steps=0,
                checkpoint_directory=str(tmp_path),
                max_n_particles=16,
            )
        )
        vortex_strength = np.array(
            [[0.0, 0.0, 10.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.2], [0.0, 0.0, 0.02]],
            dtype=np.float32,
        )
        solver.replace_vortex_particles(
            position=np.arange(12, dtype=np.float32).reshape(4, 3),
            velocity=np.zeros((4, 3), dtype=np.float32),
            vortex_strength=vortex_strength,
            core_radius=np.ones(4, dtype=np.float32),
            particle_volume=np.ones(4, dtype=np.float32),
            kinematic_viscosity=np.zeros(4, dtype=np.float32),
            group_id=np.array([0, 0, 1, 1], dtype=np.int32),
        )

        removed = solver.remove_weak_particles(5.0)

        assert removed == 2
        np.testing.assert_allclose(solver.particle_vortex_strength, vortex_strength[:2])
        np.testing.assert_array_equal(solver.particles.group_id_cpu(), np.array([0, 0]))
    finally:
        reset_taichi_backend()


def test_replace_vortex_particles_matches_uploaded_cloud(tmp_path):
    reset_taichi_backend()
    try:
        solver = VPMSolver(
            VPMSetup(
                compute_device="CPU",
                stretching=StretchingConfig.disabled(),
                viscous=ViscousConfig(scheme="NONE"),
                advection=AdvectionConfig(scheme="NONE"),
                checkpoint_interval_steps=0,
                logging_interval_steps=0,
                checkpoint_directory=str(tmp_path),
                max_n_particles=16,
            )
        )

        pos0 = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        solver.add_vortex_particles(
            position=pos0,
            velocity=np.zeros((1, 3), dtype=np.float32),
            vortex_strength=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
            core_radius=np.array([0.1], dtype=np.float32),
            particle_volume=np.array([0.01], dtype=np.float32),
            kinematic_viscosity=np.zeros(1, dtype=np.float32),
        )

        position = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            dtype=np.float32,
        )
        velocity = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        vortex_strength = np.array(
            [[0.01, 0.02, 0.03], [0.04, 0.05, 0.06], [0.07, 0.08, 0.09]],
            dtype=np.float32,
        )
        core_radius = np.full(3, 0.15, dtype=np.float32)
        particle_volume = np.full(3, 0.02, dtype=np.float32)
        kinematic_viscosity = np.full(3, 1.0e-3, dtype=np.float32)
        zone_id = np.array([1, 2, 3], dtype=np.int32)
        group_id = np.array([4, 5, 6], dtype=np.int32)
        velocity_gradient = np.arange(27, dtype=np.float32).reshape(3, 3, 3) * 0.01
        strain_rate = np.arange(27, dtype=np.float32).reshape(3, 3, 3) * 0.02

        solver.replace_vortex_particles(
            position=position,
            velocity=velocity,
            vortex_strength=vortex_strength,
            core_radius=core_radius,
            particle_volume=particle_volume,
            kinematic_viscosity=kinematic_viscosity,
            group_id=group_id,
            zone_id=zone_id,
            velocity_gradient=velocity_gradient,
            strain_rate=strain_rate,
        )

        assert solver.particles.n_particles_total == 3
        assert solver.particles.device_n_particles[None] == 3
        np.testing.assert_allclose(solver.particle_position, position)
        np.testing.assert_allclose(solver.particle_velocity, velocity)
        np.testing.assert_allclose(solver.particle_vortex_strength, vortex_strength)
        np.testing.assert_allclose(solver.particle_core_radius, core_radius)
        np.testing.assert_allclose(solver.particle_volume, particle_volume)
        np.testing.assert_allclose(solver.particle_kinematic_viscosity, kinematic_viscosity)
        np.testing.assert_allclose(
            solver.particle_vorticity, vortex_strength / particle_volume[:, None]
        )
        np.testing.assert_array_equal(solver.particles.group_id_cpu(), group_id)
        np.testing.assert_array_equal(solver.particles.zone_id_cpu(), zone_id)
        np.testing.assert_allclose(solver.particles.velocity_gradient_cpu(), velocity_gradient)
        np.testing.assert_allclose(solver.particles.strain_rate_cpu(), strain_rate)

        solver.replace_vortex_particles(
            position=np.empty((0, 3), dtype=np.float32),
            velocity=np.empty((0, 3), dtype=np.float32),
            vortex_strength=np.empty((0, 3), dtype=np.float32),
            core_radius=np.empty(0, dtype=np.float32),
            particle_volume=np.empty(0, dtype=np.float32),
            kinematic_viscosity=np.empty(0, dtype=np.float32),
        )
        assert solver.particles.n_particles_total == 0
        assert solver.particles.device_n_particles[None] == 0
    finally:
        reset_taichi_backend()


def test_bounds_removal_uses_compacted_replacement(tmp_path):
    reset_taichi_backend()
    try:
        solver = VPMSolver(
            VPMSetup(
                compute_device="CPU",
                stretching=StretchingConfig.disabled(),
                viscous=ViscousConfig(scheme="NONE"),
                advection=AdvectionConfig(scheme="NONE"),
                checkpoint_interval_steps=0,
                logging_interval_steps=0,
                checkpoint_directory=str(tmp_path),
                max_n_particles=16,
            )
        )

        position = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        velocity = np.zeros((3, 3), dtype=np.float32)
        vortex_strength = np.array(
            [[0.0, 0.0, 0.1], [0.0, 0.0, 0.2], [0.0, 0.0, 0.3]],
            dtype=np.float32,
        )
        core_radius = np.full(3, 0.1, dtype=np.float32)
        particle_volume = np.full(3, 0.01, dtype=np.float32)
        kinematic_viscosity = np.full(3, 1.0e-5, dtype=np.float32)
        eddy_viscosity = np.array([0.0, 1.0e-5, 2.0e-5], dtype=np.float32)
        group_id = np.array([10, 11, 12], dtype=np.int32)
        zone_id = np.array([20, 21, 22], dtype=np.int32)
        velocity_gradient = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
        strain_rate = velocity_gradient * 0.5

        solver.replace_vortex_particles(
            position=position,
            velocity=velocity,
            vortex_strength=vortex_strength,
            core_radius=core_radius,
            particle_volume=particle_volume,
            kinematic_viscosity=kinematic_viscosity,
            eddy_viscosity=eddy_viscosity,
            group_id=group_id,
            zone_id=zone_id,
            velocity_gradient=velocity_gradient,
            strain_rate=strain_rate,
        )

        removed = solver.particles.remove_particles_by_bounds([0.5, 1.5, -1.0, 1.0, -1.0, 1.0])

        assert removed == 1
        assert solver.particles.n_particles_total == 2
        assert solver.particles.device_n_particles[None] == 2
        np.testing.assert_allclose(solver.particle_position, position[[0, 2]])
        np.testing.assert_array_equal(solver.particles.group_id_cpu(), group_id[[0, 2]])
        np.testing.assert_array_equal(solver.particles.zone_id_cpu(), zone_id[[0, 2]])
        np.testing.assert_allclose(
            solver.particles.velocity_gradient_cpu(),
            velocity_gradient[[0, 2]],
        )
        np.testing.assert_allclose(solver.particles.strain_rate_cpu(), strain_rate[[0, 2]])
        np.testing.assert_allclose(
            solver.particles.effective_viscosity_cpu(),
            kinematic_viscosity[[0, 2]] + eddy_viscosity[[0, 2]],
        )
    finally:
        reset_taichi_backend()


def test_bounds_removal_does_not_depend_on_device_tag_field(tmp_path):
    """Retention must use the downloaded position, not stale GPU tags."""
    reset_taichi_backend()
    try:
        solver = VPMSolver(
            VPMSetup(
                compute_device="CPU",
                stretching=StretchingConfig.disabled(),
                viscous=ViscousConfig(scheme="NONE"),
                advection=AdvectionConfig(scheme="NONE"),
                checkpoint_interval_steps=0,
                logging_interval_steps=0,
                checkpoint_directory=str(tmp_path),
                max_n_particles=16,
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
            vortex_strength=zeros,
            core_radius=scalar,
            particle_volume=scalar,
            kinematic_viscosity=scalar,
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
        solver = VPMSolver(
            VPMSetup(
                compute_device="CPU",
                stretching=StretchingConfig.disabled(),
                viscous=ViscousConfig(scheme="NONE"),
                advection=AdvectionConfig(scheme="NONE"),
                checkpoint_interval_steps=0,
                logging_interval_steps=0,
                checkpoint_directory=str(tmp_path),
                max_n_particles=16,
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
            vortex_strength=zeros,
            core_radius=scalar,
            particle_volume=scalar,
            kinematic_viscosity=scalar,
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
        assert particles.n_particles_total == 3
    finally:
        reset_taichi_backend()


def test_retention_compacts_vorticity_without_quadratic_reconstruction(tmp_path):
    """Domain retention must preserve stored omega without an O(N^2) rebuild."""
    reset_taichi_backend()
    try:
        solver = VPMSolver(
            VPMSetup(
                compute_device="CPU",
                stretching=StretchingConfig.disabled(),
                viscous=ViscousConfig(scheme="NONE"),
                advection=AdvectionConfig(scheme="NONE"),
                stabilization=StabilizationConfig.bounded_domain([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]),
                checkpoint_interval_steps=0,
                logging_interval_steps=0,
                checkpoint_directory=str(tmp_path),
                max_n_particles=16,
            )
        )
        position = np.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            dtype=np.float32,
        )
        vortex_strength = np.array(
            [[0.0, 0.0, 0.1], [0.0, 0.0, 0.2], [0.0, 0.0, 0.3]],
            dtype=np.float32,
        )
        particle_volume = np.array([0.01, 0.02, 0.03], dtype=np.float32)
        solver.replace_vortex_particles(
            position=position,
            velocity=np.zeros((3, 3), dtype=np.float32),
            vortex_strength=vortex_strength,
            core_radius=np.full(3, 0.1, dtype=np.float32),
            particle_volume=particle_volume,
            kinematic_viscosity=np.full(3, 1.0e-5, dtype=np.float32),
        )

        def unexpected_reconstruction(_particles):
            raise AssertionError("retention must not run direct vorticity reconstruction")

        solver.physics.compute_vorticities = unexpected_reconstruction
        solver.stabilization.apply_retention()

        kept = np.array([0, 2])
        np.testing.assert_array_equal(solver.particles.position_cpu(), position[kept])
        np.testing.assert_allclose(
            solver.particles.vorticity_cpu(),
            vortex_strength[kept] / particle_volume[kept, None],
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
        vortex_strength = position * 1.0e-3
        core_radius = 0.1 + index * 0.001
        particle_volume = 0.01 + index * 0.0001
        kinematic_viscosity = 1.0e-3 + index * 1.0e-6
        eddy_viscosity = index * 1.0e-5
        group_id = np.arange(n, dtype=np.int32) + 10
        zone_id = np.arange(n, dtype=np.int32) + 20

        particles = Particles(max_n_particles=32)
        particles.replace_from_numpy(
            position=position,
            velocity=velocity,
            vortex_strength=vortex_strength,
            core_radius=core_radius,
            particle_volume=particle_volume,
            kinematic_viscosity=kinematic_viscosity,
            eddy_viscosity=eddy_viscosity,
            group_id=group_id,
            zone_id=zone_id,
        )

        np.testing.assert_array_equal(particles.position_cpu(), position)
        np.testing.assert_array_equal(particles.velocity_cpu(), velocity)
        np.testing.assert_array_equal(particles.vortex_strength_cpu(), vortex_strength)
        np.testing.assert_array_equal(particles.core_radius_cpu(), core_radius)
        np.testing.assert_array_equal(particles.particle_volume_cpu(), particle_volume)
        np.testing.assert_array_equal(particles.kinematic_viscosity_cpu(), kinematic_viscosity)
        np.testing.assert_array_equal(particles.eddy_viscosity_cpu(), eddy_viscosity)
        np.testing.assert_array_equal(particles.group_id_cpu(), group_id)
        np.testing.assert_array_equal(particles.zone_id_cpu(), zone_id)

        # Full replacements use persistent native arrays that are distinct for
        # every Taichi field.  Sharing one external ndarray across template
        # fields caused vortex-strength/position aliasing in the coupled cube run.
        position_upload = particles._native_vector_uploads[id(particles.position)]
        strength_upload = particles._native_vector_uploads[id(particles.vortex_strength)]
        assert position_upload is not strength_upload
        position_download = particles._host_vector_chunks[("download", id(particles.position))]
        strength_download = particles._host_vector_chunks[
            ("download", id(particles.vortex_strength))
        ]
        assert position_download is not strength_download
        assert (
            particles._native_scalar_uploads[id(particles.core_radius)]
            is not particles._native_scalar_uploads[id(particles.particle_volume)]
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
        vortex_strength = rng.normal(0.0, 0.1, size=(n, 3)).astype(np.float32)
        core_radius = np.full(n, 0.1, dtype=np.float32)
        tree = TaichiTreecode(
            max_n_particles=16,
            max_nodes=32,
            theta=0.3,
            kernel_type="GAUSSIAN",
        )
        tree.build(position, vortex_strength, core_radius, force=True)

        velocity, gradient, strain = tree.compute_velocity_and_gradient(
            np.zeros(3, dtype=np.float32)
        )
        target_velocity = tree.compute_target_velocity(position[:9])
        target_gradient = tree.compute_target_velocity_gradient(position[:9])

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
