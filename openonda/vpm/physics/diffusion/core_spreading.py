"""Core Spreading Method (CSM) viscous diffusion.

CSM models diffusion by expanding the particle core radius.  For the Gaussian
kernel sigma^2(t) = sigma^2(0) + 4*kinematic_viscosity*t, which is equivalent to convolution
with a Gaussian diffusion kernel.  O(N), analytic, unconditionally stable; it
relies on periodic remeshing to prevent excessive core overlap.

``make_core_spreading_kernel`` builds the ``update_radius_csm_kernel`` Taichi kernel, which the
owning physics object binds and calls.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

import taichi as ti

if TYPE_CHECKING:
    from ...particles.container import Particles
    from ..base import PhysicsBase


def apply_core_spreading(
    physics: "PhysicsBase", particles: "Particles", time_step_size: float
) -> None:
    """Apply Core Spreading Method diffusion to ``particles`` over ``time_step_size``."""
    N = len(particles)
    if N == 0 or time_step_size <= 0.0:
        return

    physics.update_radius_csm_kernel(
        particles.core_radius, particles.effective_viscosity, time_step_size, N
    )
    # The kernel mutates a particle field directly; invalidate host snapshots
    # so subsequent diagnostics and public reads observe the grown cores.
    particles.touch_state()


def make_core_spreading_kernel(diffusivity_constant_: Callable) -> Callable:
    """Create the core-spreading kernel for the particle kernel's ``diffusivity_constant_``."""

    @ti.kernel
    def update_radius_csm_kernel(
        core_radius: ti.template(),
        viscosities_eff: ti.template(),
        time_step_size: ti.template(),
        n_particles_total: ti.i32,
    ):
        N = n_particles_total
        for i in range(N):
            effective_viscosity = viscosities_eff[i]
            current_radius = core_radius[i]
            diffusion_term = diffusivity_constant_() * effective_viscosity * time_step_size
            new_rad_sq = current_radius * current_radius + diffusion_term

            core_radius[i] = ti.sqrt(new_rad_sq)

    return update_radius_csm_kernel
