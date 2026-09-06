"""Random Walk Method (RWM) viscous diffusion.

RWM models diffusion by adding Gaussian random displacements
Dx = eta * sqrt(2*kinematic_viscosity*dt).  O(N), works for any particle distribution; it is a
stochastic method that converges to the exact solution in the many-particle,
small-step limit.

The concrete kernel call runs on the owning physics object, which supplies the
``update_position_rwm_kernel`` Taichi kernel and its temporary-field
management.
"""

import numpy as np


def apply_random_walk(
    owner,
    particles,
    time_step_size: float,
    *,
    random_seed: int,
    accepted_step: int,
):
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

    owner._resize_temp_fields(N)
    # No temp field is touched, so the temp-field zeroing is deliberately skipped.
    owner.update_position_rwm_kernel(
        particles.position,
        particles.effective_viscosity,
        time_step_size,
        N,
        random_seed,
        accepted_step,
    )
    # The Taichi kernel mutates the source position field in place.
    particles.touch_state()
