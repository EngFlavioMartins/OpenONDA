"""Independent face-flux check of the production variable-viscosity GBD kernel."""

import numpy as np
import taichi as ti

from source.solvers.vpm.physics.diffusion.grid import _GridDiffusionMixin


def test_real_gbd_kernel_executes_the_componentwise_variable_heat_operator():
    if ti.lang.impl.get_runtime().prog is None:
        ti.init(arch=ti.cpu, cpu_max_num_threads=1)
    n = 16
    h = 2 * np.pi / n
    axis = h * np.arange(-1, n + 1)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    # ABC velocity equals its curl; all components and coordinates vary.
    omega = np.stack(
        (np.sin(z) + np.cos(y), np.sin(x) + np.cos(z), np.sin(y) + np.cos(x)), axis=-1
    ).astype(np.float32)
    viscosity = (0.001 + 0.0003 * np.cos(x) * np.cos(y) * np.cos(z)).astype(np.float32)
    dt = 0.04 * h**2 / float(viscosity.max())
    shape = viscosity.shape
    src = ti.Vector.field(3, ti.f32, shape=shape)
    dst = ti.Vector.field(3, ti.f32, shape=shape)
    nu = ti.field(ti.f32, shape=shape)
    mask = ti.field(ti.i32, shape=shape)
    src.from_numpy(omega)
    nu.from_numpy(viscosity)
    _GridDiffusionMixin()._laplacian_step_variable_gpu_kernel(src, dst, nu, mask, dt, h, *shape)

    # Assemble each shared face flux once and add its opposite to the neighbour.
    # This checks conservation and the actual discrete PDE without a study helper.
    expected = np.zeros_like(omega, dtype=float)
    for direction in range(3):
        left = [slice(None)] * 3
        right = [slice(None)] * 3
        left[direction], right[direction] = slice(None, -1), slice(1, None)
        left, right = tuple(left), tuple(right)
        flux = (
            0.5
            * (viscosity[left] + viscosity[right])[..., None]
            * (omega[right].astype(float) - omega[left])
            / h**2
        )
        expected[left] += flux
        expected[right] -= flux
    actual = (dst.to_numpy().astype(float) - omega) / dt
    assert float(viscosity.max()) * dt / h**2 < 1 / 12
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 2e-4
    np.testing.assert_allclose(expected.sum(axis=(0, 1, 2)), 0, atol=2e-14, rtol=0)
