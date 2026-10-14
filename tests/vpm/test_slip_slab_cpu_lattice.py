"""The CPU slab must use the same fixed GBD phase as device backends."""

import numpy as np
import pytest
import taichi as ti

from openonda import vpm
from openonda.core.slip_slab_images import slab_image_lattice


def _m4(distance):
    q = np.abs(distance)
    return np.where(
        q <= 1.0,
        1.0 - 2.5 * q**2 + 1.5 * q**3,
        np.where(q <= 2.0, 0.5 * (2.0 - q) ** 2 * (1.0 - q), 0.0),
    )


def _padded_mirror_reference(origin, shape, spacing, source, strength, alpha):
    coordinates = [origin[axis] + spacing * np.arange(shape[axis]) for axis in range(3)]
    result = np.zeros((*shape, 3), dtype=np.float64)
    for plane in (None, -0.24, 0.24):
        position = source.copy()
        gamma = strength.copy()
        if plane is not None:
            position[2] = 2.0 * plane - position[2]
            gamma[:2] *= -1.0
        weights = [_m4((axis - position[d]) / spacing) for d, axis in enumerate(coordinates)]
        result += (
            weights[0][:, None, None, None]
            * weights[1][None, :, None, None]
            * weights[2][None, None, :, None]
            * gamma[None, None, None, :]
        )
    initial = result.copy()
    for _ in range(2):
        padded = np.pad(result, ((1, 1), (1, 1), (1, 1), (0, 0)), mode="edge")
        laplacian = -6.0 * result
        for axis in range(3):
            low = [slice(1, -1)] * 3
            high = [slice(1, -1)] * 3
            low[axis] = slice(0, -2)
            high[axis] = slice(2, None)
            laplacian += padded[tuple(low)] + padded[tuple(high)]
        result = result + alpha * laplacian
    return result, initial


def test_cpu_slab_gbd_configures_fixed_lattice(tmp_path):
    h = 0.08
    solver = vpm.VPMSolver(
        vpm.VPMCase(
            directory=tmp_path,
            name="test",
            numerics=vpm.Numerics(
                time_step_size=0.04,
                compute_device="cpu",
                precision="f32",
                max_n_particles=64,
                domain_bounds=(-0.4, 0.4, -0.4, 0.4, -0.24, 0.24),
                induction=vpm.SlipSlabInduction(vpm.DirectInduction(), z_min=-0.24, z_max=0.24),
                viscous=vpm.ViscousConfig.gbd(
                    particle_spacing=h,
                    padding=3.0,
                    kinematic_viscosity=1.0 / 150.0,
                ),
            ),
        )
    )
    try:
        assert solver.physics._fixed_grid_min is not None
        solver.physics.configure_grid_lattice_anchor((0.0, 0.0, -0.2), h)
        grid_min, _shape = solver.physics._lattice_aligned_bounds(
            np.array([[0.0, 0.0, 0.0]], dtype=np.float32), h, 3.0
        )
        phase = 2.0 * (np.array([-0.24, 0.24]) - grid_min[2]) / h
        np.testing.assert_allclose(phase, np.rint(phase), atol=1.0e-4, rtol=0.0)
    finally:
        solver.close()


def test_active_slab_retains_mirror_and_diffusion_halo_when_cloud_retreats(tmp_path):
    h = 0.08
    solver = vpm.VPMSolver(
        vpm.VPMCase(
            directory=tmp_path,
            name="test",
            numerics=vpm.Numerics(
                time_step_size=0.04,
                compute_device="cpu",
                precision="f32",
                max_n_particles=64,
                domain_bounds=(-0.4, 0.4, -0.4, 0.4, -0.24, 0.24),
                induction=vpm.SlipSlabInduction(vpm.DirectInduction(), z_min=-0.24, z_max=0.24),
                viscous=vpm.ViscousConfig.gbd(
                    particle_spacing=h,
                    padding=3.0,
                    kinematic_viscosity=1.0 / 150.0,
                ),
            ),
        )
    )
    try:
        solver.physics.configure_grid_lattice_anchor((0.0, 0.0, -0.2), h)
        positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        required = (-0.24 - 3 * h, 0.24 + 3 * h)
        grid_min, shape = solver.physics._lattice_aligned_bounds(
            positions, h, 3.0, required_z_bounds=required
        )
        assert grid_min[2] <= required[0] + 1.0e-6
        assert grid_min[2] + (shape[2] - 1) * h >= required[1] - 1.0e-6
        assert shape[2] > 5
        with pytest.raises(ValueError, match="cannot retain the slip-plane"):
            solver.physics._lattice_aligned_bounds(
                positions,
                h,
                3.0,
                required_z_bounds=(-0.24 - 4 * h, 0.24 + 4 * h),
            )
    finally:
        solver.close()


@pytest.mark.parametrize("phase", [-0.24, -0.20])
def test_retreated_slab_diffusion_matches_padded_mirror_reference(tmp_path, phase):
    h = 0.08
    viscosity = 1.0 / 150.0
    dt = 0.08
    solver = vpm.VPMSolver(
        vpm.VPMCase(
            directory=tmp_path,
            name="test",
            numerics=vpm.Numerics(
                time_step_size=dt,
                compute_device="cpu",
                precision="f32",
                max_n_particles=64,
                domain_bounds=(-0.4, 0.4, -0.4, 0.4, -0.24, 0.24),
                induction=vpm.SlipSlabInduction(vpm.DirectInduction(), z_min=-0.24, z_max=0.24),
                viscous=vpm.ViscousConfig.gbd(
                    particle_spacing=h,
                    padding=3.0,
                    kinematic_viscosity=viscosity,
                ),
            ),
        )
    )
    try:
        physics = solver.physics
        physics.configure_grid_lattice_anchor((0.0, 0.0, phase), h)
        assert physics.gbd_diffusion_substep_count(viscosity, dt, h) == 2
        source = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        strength = np.array([0.7, -0.4, 0.9], dtype=np.float32)
        origin, shape = physics._lattice_aligned_bounds(
            source[None, :], h, 3.0, required_z_bounds=(-0.48, 0.48)
        )
        nx, ny, nz = shape
        position_field = ti.Vector.field(3, ti.f32, shape=1)
        strength_field = ti.Vector.field(3, ti.f32, shape=1)
        position_field.from_numpy(source[None, :])
        strength_field.from_numpy(strength[None, :])
        physics._zero_grid_kernel(physics._grid_a, nx, ny, nz)
        physics._m4_scatter_gpu_kernel(
            position_field,
            strength_field,
            physics._grid_a,
            *map(float, origin),
            h,
            nx,
            ny,
            nz,
            0,
            1,
        )
        physics._add_slab_images_kernel(
            physics._grid_a,
            physics._grid_b,
            nx,
            ny,
            nz,
            *slab_image_lattice(origin[2], h, (-0.24, 0.24)),
        )
        alpha = viscosity * (dt / 2.0) / h**2
        physics._laplacian_step_gpu_kernel(
            physics._grid_b,
            physics._grid_a,
            physics._body_mask_grid,
            alpha,
            nx,
            ny,
            nz,
        )
        physics._laplacian_step_gpu_kernel(
            physics._grid_a,
            physics._grid_b,
            physics._body_mask_grid,
            alpha,
            nx,
            ny,
            nz,
        )
        actual = physics._grid_b.to_numpy()[:nx, :ny, :nz]
        reference_origin = origin.astype(np.float64).copy()
        reference_origin[2] -= 3 * h
        reference, _ = _padded_mirror_reference(
            reference_origin,
            (nx, ny, nz + 6),
            h,
            source.astype(np.float64),
            strength.astype(np.float64),
            alpha,
        )
        physical = (origin[2] + h * np.arange(nz) >= -0.24 - 1.0e-6) & (
            origin[2] + h * np.arange(nz) <= 0.24 + 1.0e-6
        )
        np.testing.assert_allclose(
            actual[:, :, physical],
            reference[:, :, 3 : 3 + nz][:, :, physical],
            rtol=0.0,
            atol=2.0e-6,
        )
    finally:
        solver.close()


def test_one_layer_slab_diffusion_is_the_two_dimensional_laplacian(tmp_path):
    h = 0.08
    viscosity = 1.0 / 150.0
    dt = 0.12  # two explicit stages, within the three-cell halo
    planes = (-0.5 * h, 0.5 * h)
    solver = vpm.VPMSolver(
        vpm.VPMCase(
            directory=tmp_path,
            name="test",
            numerics=vpm.Numerics(
                time_step_size=dt,
                compute_device="cpu",
                precision="f32",
                max_n_particles=1024,
                domain_bounds=(-0.8, 0.8, -0.8, 0.8, *planes),
                induction=vpm.SlipSlabInduction(
                    vpm.DirectInduction(), z_min=planes[0], z_max=planes[1]
                ),
                viscous=vpm.ViscousConfig.gbd(
                    particle_spacing=h,
                    padding=3.0,
                    kinematic_viscosity=viscosity,
                ),
            ),
        )
    )
    try:
        physics = solver.physics
        physics.configure_grid_lattice_anchor((0.0, 0.0, 0.0), h)

        # Axial Gaussian vortex on the single particle layer
        x, y = np.meshgrid(h * np.arange(-4, 5), h * np.arange(-4, 5), indexing="ij")
        omega = np.exp(-(x**2 + y**2) / (2 * h) ** 2).ravel()
        count = omega.size
        position = np.column_stack((x.ravel(), y.ravel(), np.zeros(count)))
        strength = np.zeros((count, 3))
        strength[:, 2] = omega * h**3
        solver.add_vortex_particles(
            position, np.zeros((count, 3)), strength, np.full(count, h), np.full(count, h**3)
        )
        result = physics.gbd_diffusion(
            solver.particles,
            time_step_size=dt,
            particle_spacing=h,
            kinematic_viscosity=viscosity,
            regen_threshold=1.0e-30,
            regen_threshold_mode="absolute",
        )

        # The same explicit stages of the five-point Laplacian on the active plane
        grid_min, shape = physics._lattice_aligned_bounds(position, h, 3.0)
        stages = physics.last_gbd_diffusion_substeps
        alpha = viscosity * dt / stages / h**2
        index = np.rint((position[:, :2] - grid_min[:2]) / h).astype(int)
        reference = np.zeros(shape[:2])
        reference[index[:, 0], index[:, 1]] = strength[:, 2]
        for _ in range(stages):
            padded = np.pad(reference, 1, mode="edge")
            reference = reference + alpha * (
                padded[:-2, 1:-1]
                + padded[2:, 1:-1]
                + padded[1:-1, :-2]
                + padded[1:-1, 2:]
                - 4.0 * reference
            )
        nodes = np.rint((result["position"][:, :2] - grid_min[:2]) / h).astype(int)
        np.testing.assert_allclose(result["position"][:, 2], 0.0, atol=1e-7)
        np.testing.assert_allclose(result["vortex_strength"][:, :2], 0.0, atol=1e-12)
        np.testing.assert_allclose(
            result["vortex_strength"][:, 2],
            reference[nodes[:, 0], nodes[:, 1]],
            rtol=1e-5,
            atol=1e-6 * h**3,
        )
        assert stages >= 2 and len(result["position"]) > count
    finally:
        solver.close()
