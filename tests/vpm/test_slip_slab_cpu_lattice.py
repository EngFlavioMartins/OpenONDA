"""The CPU slab must use the same fixed GBD phase as device backends."""

import numpy as np
import pytest
import taichi as ti

from openonda import vpm


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
            backup=vpm.Backup(interval_steps=0),
            numerics=vpm.Numerics(
                time_step_size=0.04,
                compute_device="CPU",
                precision="f32",
                max_n_particles=64,
                domain_bounds=(-0.4, 0.4, -0.4, 0.4, -0.24, 0.24),
                induction=vpm.SlipSlabInduction(vpm.DirectInduction(), z_min=-0.24, z_max=0.24),
                viscous=vpm.ViscousConfig.gbd(
                    particle_spacing=h,
                    padding=3.0,
                    kinematic_viscosity=1.0 / 150.0,
                    max_nodes=64,
                ),
                verbose=False,
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
            backup=vpm.Backup(interval_steps=0),
            numerics=vpm.Numerics(
                time_step_size=0.04,
                compute_device="CPU",
                precision="f32",
                max_n_particles=64,
                domain_bounds=(-0.4, 0.4, -0.4, 0.4, -0.24, 0.24),
                induction=vpm.SlipSlabInduction(vpm.DirectInduction(), z_min=-0.24, z_max=0.24),
                viscous=vpm.ViscousConfig.gbd(
                    particle_spacing=h,
                    padding=3.0,
                    kinematic_viscosity=1.0 / 150.0,
                    max_nodes=64,
                ),
                verbose=False,
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
            backup=vpm.Backup(interval_steps=0),
            numerics=vpm.Numerics(
                time_step_size=dt,
                compute_device="CPU",
                precision="f32",
                max_n_particles=64,
                domain_bounds=(-0.4, 0.4, -0.4, 0.4, -0.24, 0.24),
                induction=vpm.SlipSlabInduction(vpm.DirectInduction(), z_min=-0.24, z_max=0.24),
                viscous=vpm.ViscousConfig.gbd(
                    particle_spacing=h,
                    padding=3.0,
                    kinematic_viscosity=viscosity,
                    max_nodes=64,
                ),
                verbose=False,
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
        for plane in (-0.24, 0.24):
            physics._m4_scatter_slip_image_kernel(
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
                plane,
            )
        alpha = viscosity * (dt / 2.0) / h**2
        physics._laplacian_step_gpu_kernel(
            physics._grid_a,
            physics._grid_b,
            physics._body_mask_grid,
            alpha,
            nx,
            ny,
            nz,
        )
        physics._laplacian_step_gpu_kernel(
            physics._grid_b,
            physics._grid_a,
            physics._body_mask_grid,
            alpha,
            nx,
            ny,
            nz,
        )
        actual = physics._grid_a.to_numpy()[:nx, :ny, :nz]
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
