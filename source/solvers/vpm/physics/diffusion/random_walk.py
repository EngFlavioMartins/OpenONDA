"""Random Walk Method (RWM) viscous diffusion.

RWM models diffusion by adding Gaussian random displacements
Dx = eta * sqrt(2*kinematic_viscosity*dt).  O(N), works for any particle distribution; it is a
stochastic method that converges to the exact solution in the many-particle,
small-step limit.

The concrete kernel call runs on the owning physics object, which supplies the
``update_position_rwm_kernel`` Taichi kernel and its temporary-field
management.
"""


def apply_random_walk(owner, particles, time_step_size: float):
    """Apply Random Walk Method diffusion to ``particles`` over ``time_step_size``."""
    N = len(particles)
    if N == 0 or time_step_size <= 0.0:
        return

    owner._resize_temp_fields(N)
    # No temp field is touched, so the temp-field zeroing is deliberately skipped.
    owner.update_position_rwm_kernel(
        particles.position, particles.effective_viscosity, time_step_size, N
    )
