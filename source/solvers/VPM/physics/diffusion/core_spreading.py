"""Core Spreading Method (CSM) viscous diffusion.

CSM models diffusion by expanding the particle core radius.  For the Gaussian
kernel sigma^2(t) = sigma^2(0) + 4*nu*t, which is equivalent to convolution
with a Gaussian diffusion kernel.  O(N), analytic, unconditionally stable; it
relies on periodic remeshing to prevent excessive core overlap.

The concrete kernel call runs on the owning physics object, which supplies the
``update_radius_csm_kernel`` Taichi kernel and its temporary-field management.
"""


def apply_core_spreading(owner, particles, dt: float):
    """Apply Core Spreading Method diffusion to ``particles`` over ``dt``."""
    N = len(particles)
    if N == 0 or dt <= 0.0:
        return

    owner._resize_temp_fields(N)
    owner.update_radius_csm_kernel(particles.radius, particles.viscosity_effective, dt, N)
