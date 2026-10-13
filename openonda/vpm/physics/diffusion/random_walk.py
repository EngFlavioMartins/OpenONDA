"""Random Walk Method (RWM) viscous diffusion.

RWM models diffusion by adding Gaussian random displacements
Dx = eta * sqrt(2*kinematic_viscosity*dt).  O(N), works for any particle distribution; it is a
stochastic method that converges to the exact solution in the many-particle,
small-step limit.

The concrete kernel call runs on the owning physics object, which supplies the
``update_position_rwm_kernel`` Taichi kernel and its temporary-field
management.

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
    effective_viscosity = np.asarray(particles.effective_viscosity_cpu()[:N], dtype=np.float64)
    if not np.all(np.isfinite(effective_viscosity)) or np.any(effective_viscosity < 0.0):
        raise ValueError("RWM requires finite, non-negative effective viscosity")
    reference = float(effective_viscosity[0])
    storage_dtype = np.asarray(particles.effective_viscosity_cpu()).dtype
    storage_epsilon = (
        float(np.finfo(storage_dtype).eps)
        if np.issubdtype(storage_dtype, np.floating)
        else float(np.finfo(np.float32).eps)
    )
    scale = max(float(np.max(np.abs(effective_viscosity))), abs(reference))
    tolerance = 8.0 * storage_epsilon * scale
    if np.any(np.abs(effective_viscosity - reference) > tolerance):
        raise ValueError(
            "RWM requires spatially uniform effective viscosity; "
            "use DVH or GBD for variable-viscosity diffusion"
        )

    physics._resize_temp_fields(N)
    # No temp field is touched, so the temp-field zeroing is deliberately skipped.
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
