"""Diffusion handler for the VPM solver.

:class:`DiffusionPhysics` composes the shared grid mixin with the per-scheme
algorithm modules in this package: core spreading, random walk, and the DVH/GBD
driver methods provided by :class:`_GridDiffusionMixin`.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import taichi as ti

from ...config.constants import MAX_PARTICLES
from ..base import PhysicsBase
from .core_spreading import apply_core_spreading
from .grid import _GridDiffusionMixin
from .random_walk import apply_random_walk


@ti.data_oriented
class DiffusionPhysics(PhysicsBase, _GridDiffusionMixin):
    """
    Diffusion physics handler for viscous effects.

    Implements multiple viscous diffusion schemes with different
    conservation properties and computational costs.
    """

    def __init__(
        self,
        particles_kernel: str = "GAUSSIAN",
        max_particles: int = MAX_PARTICLES,
        accumulator_dtype: ti.types = ti.f32,
    ):
        """Initialize diffusion physics module."""
        super().__init__(particles_kernel, max_particles, accumulator_dtype)
        self._init_grid_diffusion()

    # CORE SPREADING METHOD (CSM)

    def core_spreading_diffusion(self, particles, dt: float):
        """
        Apply viscous diffusion using Core Spreading Method.

        CSM models diffusion by expanding the particle core radius.  For the
        Gaussian kernel:
            sigma^2(t) = sigma^2(0) + 4*nu*t

        which is equivalent to convolution with a Gaussian diffusion kernel.

        Advantages:
        - O(N) computational cost
        - Simple and stable
        - No particle interactions needed

        Disadvantages:
        - Requires periodic remeshing to prevent excessive core overlap
        - Less accurate than direct-interaction methods for non-uniform distributions

        Args:
            particles: Particle container
            dt: Time step size [s]
        """
        apply_core_spreading(self, particles, dt)

    # RANDOM WALK METHOD (RWM)

    def random_walk_method_diffusion(self, particles, dt: float):
        """
        Apply viscous diffusion using Random Walk Method.

        RWM models diffusion by adding Gaussian random displacements:
            Δx = η * sqrt(2nu*dt)

        where η is a normally distributed random vector.

        This is a stochastic method that converges to the exact solution
        in the limit of many particles and small time steps.

        Advantages:
        - O(N) computational cost
        - Simple implementation
        - Works for any particle distribution

        Disadvantages:
        - Statistical noise (requires ensemble averaging)
        - May need very small time steps for accuracy

        Args:
            particles: Particle container
            dt: Time step size [s]
        """
        apply_random_walk(self, particles, dt)

    # VOLUME UPDATE FROM DIVERGENCE
