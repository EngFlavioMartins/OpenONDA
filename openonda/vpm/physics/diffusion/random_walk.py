"""Random Walk Method (RWM) viscous diffusion.

RWM models diffusion by adding Gaussian random displacements
Dx = eta * sqrt(2*kinematic_viscosity*dt).  O(N), works for any particle distribution; it is a
stochastic method that converges to the exact solution in the many-particle,
small-step limit.

``make_random_walk_kernel`` builds the ``update_position_rwm_kernel`` Taichi kernel, which the
owning physics object binds and calls.

Between free-slip planes a walk that crosses a plane continues as the walk of
the particle's image: the particle is folded back into the slab and an odd
number of reflections reverses its in-plane strength. This is the reflected
Brownian motion of the heat equation with the slab's image symmetry.
"""

from collections.abc import Callable
import functools
from typing import TYPE_CHECKING

import numpy as np
import taichi as ti

from ..induction.base import TaichiDataType

if TYPE_CHECKING:
    from ...particles.container import Particles
    from ..base import PhysicsBase


@functools.cache
def _fold_into_slab_kernel(dtype: TaichiDataType) -> Callable:
    """Device fold of particles into the slab, for position fields of ``dtype``.

    The reference definition is :func:`openonda.core.slip_slab_images.fold_into_slab`.
    """

    @ti.kernel
    def fold(
        position: ti.template(),  # type: ignore[invalid-annotation]
        vortex_strength: ti.template(),  # type: ignore[invalid-annotation]
        count: ti.i32,
        z_min: dtype,
        z_max: dtype,
    ):
        period = 2.0 * (z_max - z_min)
        for i in range(count):
            z = position[i][2]
            if z < z_min or z > z_max:
                phase = (z - z_min) - period * ti.floor((z - z_min) / period)
                if phase > 0.5 * period:
                    position[i][2] = z_min + period - phase
                    vortex_strength[i][0] = -vortex_strength[i][0]
                    vortex_strength[i][1] = -vortex_strength[i][1]
                else:
                    position[i][2] = z_min + phase

    return fold


def apply_random_walk(
    physics: "PhysicsBase",
    particles: "Particles",
    time_step_size: float,
    *,
    random_seed: int,
    accepted_step: int,
) -> None:
    """Apply deterministic counter-based RWM diffusion for one accepted step."""
    N = len(particles)
    if N == 0 or time_step_size <= 0.0:
        return

    # A source-dependent random displacement is the Fokker--Planck operator
    # ``laplacian(nu_eff * omega)``.  The conservative grid schemes instead
    # advance ``div(nu_eff * grad(omega))``.  RWM has no drift/viscosity
    # gradient correction, so refuse variable effective viscosity rather than
    # silently running a different PDE (LES is rejected at case validation).
    stored_viscosity = particles.effective_viscosity_cpu()[:N]
    effective_viscosity = np.asarray(stored_viscosity, dtype=np.float64)
    if not np.all(np.isfinite(effective_viscosity)) or np.any(effective_viscosity < 0.0):
        raise ValueError("RWM requires finite, non-negative effective viscosity")
    reference = float(effective_viscosity[0])
    storage_epsilon = float(np.finfo(stored_viscosity.dtype).eps)
    scale = max(float(np.max(np.abs(effective_viscosity))), abs(reference))
    tolerance = 8.0 * storage_epsilon * scale
    if np.any(np.abs(effective_viscosity - reference) > tolerance):
        raise ValueError(
            "RWM requires spatially uniform effective viscosity; "
            "use DVH or GBD for variable-viscosity diffusion"
        )

    # No temp field is touched, so the temp-field zeroing is skipped.
    physics.update_position_rwm_kernel(
        particles.position,
        particles.effective_viscosity,
        time_step_size,
        N,
        random_seed,
        accepted_step,
    )
    slab = getattr(physics, "_slip_slab_bounds", None)
    if slab is not None:
        _fold_into_slab_kernel(ti.f64 if particles.position.dtype == ti.f64 else ti.f32)(
            particles.position, particles.vortex_strength, N, float(slab[0]), float(slab[1])
        )
    # The Taichi kernels mutate the particle fields in place.
    particles.touch_state()


def make_random_walk_kernel() -> Callable:
    """Create the counter-based Random Walk Method kernel.

    The random variates are a pure function of the declared seed, accepted
    step, particle index, and draw index.  They therefore do not depend on an
    opaque backend RNG cursor and survive fresh processes and restarts exactly.
    """

    @ti.func
    def hash_u32(value: ti.u32) -> ti.u32:
        value ^= value >> 16
        value *= ti.u32(0x7FEB352D)
        value ^= value >> 15
        value *= ti.u32(0x846CA68B)
        value ^= value >> 16
        return value

    @ti.func
    def uniform_01(random_seed: ti.i32, accepted_step: ti.i32, particle: ti.i32, draw: ti.i32):
        key = ti.cast(random_seed, ti.u32)
        key ^= ti.cast(accepted_step, ti.u32) * ti.u32(0x9E3779B9)
        key ^= ti.cast(particle + 1, ti.u32) * ti.u32(0x85EBCA6B)
        key ^= ti.cast(draw + 1, ti.u32) * ti.u32(0xC2B2AE35)
        # Midpoint conversion avoids exact zero and one, which keeps the
        # Box--Muller logarithm finite on every backend.
        return (ti.cast(hash_u32(key), ti.f32) + 0.5) * ti.f32(1.0 / 4294967296.0)

    @ti.kernel
    def update_position_rwm_kernel(
        position: ti.template(),
        viscosities_eff: ti.template(),
        time_step_size: ti.template(),
        n_particles_total: ti.i32,
        random_seed: ti.i32,
        accepted_step: ti.i32,
    ):
        """Apply the Random Walk Method (RWM) with Brownian displacements.

        Uses a stateless integer hash plus a manual Box--Muller transform.  The
        result is reproducible across repeated calls, fresh processes, and
        save/restart continuation for a fixed backend and precision.
        """
        two_pi: ti.f32 = 6.28318530717959
        N = n_particles_total
        for i in range(N):
            effective_viscosity = viscosities_eff[i]
            displacement_factor = ti.sqrt(2.0 * effective_viscosity * time_step_size)
            # Box-Muller transform: two uniform samples -> two standard normals
            u1 = uniform_01(random_seed, accepted_step, i, 0)
            u2 = uniform_01(random_seed, accepted_step, i, 1)
            u3 = uniform_01(random_seed, accepted_step, i, 2)
            u4 = uniform_01(random_seed, accepted_step, i, 3)
            mag1 = ti.sqrt(-2.0 * ti.log(u1))
            mag2 = ti.sqrt(-2.0 * ti.log(u3))
            dx = displacement_factor * mag1 * ti.cos(two_pi * u2)
            dy = displacement_factor * mag1 * ti.sin(two_pi * u2)
            dz = displacement_factor * mag2 * ti.cos(two_pi * u4)
            position[i] += ti.Vector([dx, dy, dz])

    return update_position_rwm_kernel
