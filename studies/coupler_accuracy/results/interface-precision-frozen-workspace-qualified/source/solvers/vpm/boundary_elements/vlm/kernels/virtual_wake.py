"""Finite-core stage transport for a temporary, uninserted newborn wake row."""

import taichi as ti

from ....kernels.base import make_device_vortex_kernels
from ....physics.induction.stretching import stretching_rate


def make_virtual_wake_kernel(kernel_name, dtype):
    """Compile velocity/Jacobian and opposite strip reaction from one source model."""
    functions = make_device_vortex_kernels(kernel_name, dtype=dtype)
    q, zeta = functions["q_"], functions["zeta_"]

    @ti.kernel
    def accumulate(
        position: ti.template(),
        strength: ti.template(),
        core: ti.template(),
        velocity: ti.template(),
        rate: ti.template(),
        gradient_out: ti.template(),
        sources: ti.types.ndarray(dtype=dtype, ndim=2),
        source_strength: ti.types.ndarray(dtype=dtype, ndim=2),
        source_core: ti.types.ndarray(dtype=dtype, ndim=1),
        source_panel: ti.types.ndarray(dtype=ti.i32, ndim=1),
        exchange: ti.template(),
        count: ti.i32,
        n_sources: ti.i32,
        mode: ti.i32,
        publish: ti.i32,
        has_gradient: ti.template(),
    ):
        """Add virtual-row rates and book the negative strength rate to its TE strip."""
        for i in range(count):
            total_velocity = ti.Vector.zero(velocity.dtype, 3)
            total_gradient = ti.Matrix.zero(rate.dtype, 3, 3)
            for j in range(n_sources):
                source = ti.Vector([sources[j, 0], sources[j, 1], sources[j, 2]], dt=velocity.dtype)
                gamma = ti.Vector(
                    [source_strength[j, 0], source_strength[j, 1], source_strength[j, 2]],
                    dt=rate.dtype,
                )
                displacement = position[i] - source
                radius = ti.sqrt(displacement.dot(displacement))
                sigma = 0.5 * (core[i] + source_core[j])
                rho = radius / sigma
                coefficient = zeta(0.0) / (3.0 * sigma**3)
                derivative = ti.cast(0.0, rate.dtype)
                if radius > 1e-10:
                    coefficient = q(rho) / radius**3
                    derivative = zeta(rho) / (sigma**3 * radius**2) - 3.0 * q(rho) / radius**5
                cross = gamma.cross(displacement)
                jacobian = ti.Matrix(
                    [
                        [0.0, -gamma[2], gamma[1]],
                        [gamma[2], 0.0, -gamma[0]],
                        [-gamma[1], gamma[0], 0.0],
                    ],
                    dt=rate.dtype,
                ) * coefficient + derivative * cross.outer_product(displacement)
                total_velocity += coefficient * cross
                total_gradient += jacobian
                if publish:
                    reaction = -stretching_rate(jacobian, strength[i], mode)
                    for axis in ti.static(range(3)):
                        ti.atomic_add(exchange[source_panel[j]][axis], reaction[axis])
            velocity[i] += total_velocity
            rate[i] += stretching_rate(total_gradient, strength[i], mode)
            if ti.static(has_gradient):
                gradient_out[i] += total_gradient

    return accumulate
