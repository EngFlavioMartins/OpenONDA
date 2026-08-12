"""
Turbulence module for VPM solver.
==================================
Orchestrates subgrid-scale turbulence models (Smagorinsky eddy viscosity).

The turbulence model does not modify vortex stretching or particle strengths.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026 / Refactored May 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import taichi as ti

from ..config.constants import MAX_PARTICLES, SMAGORINSKY_CONSTANT
from .smagorinsky import SmagorinskyModel


@ti.data_oriented
class ParticlesLES:
    """
    Large Eddy Simulation (LES) turbulence model for VPM.
    Orchestrates a specific SGS eddy-viscosity model (e.g. Smagorinsky).
    """

    def __init__(
        self,
        LES_filter_type: str,
        max_particles: int = MAX_PARTICLES,
        kernel_type: str = "GAUSSIAN",
        cs: float = SMAGORINSKY_CONSTANT,
        ce: float = 1.048,
        accumulator_dtype: ti.types = ti.f32,
    ):
        """
        Initialize the LES turbulence model.

        Selects and instantiates a sub-grid-scale model (e.g.
        :class:`SmagorinskyModel`) based on ``LES_filter_type``, stores the
        particle kernel type, and sets up diagnostic field trackers.

        Args:
            LES_filter_type: Identifier for the SGS model.  Supported values:
                ``"SMAGORINSKY"``, ``"LES_SMAGORINSKY"``.
            max_particles: Maximum number of particles for field allocation.
            kernel_type: Base regularization kernel type (e.g. ``"GAUSSIAN"``).
            cs: Smagorinsky constant (default from ``SMAGORINSKY_CONSTANT``).
            ce: Model dissipation constant (default 1.048).
            accumulator_dtype: Taichi floating-point type used by LES fields.
        """
        self.LES_filter_type = LES_filter_type
        self.max_particles = max_particles
        self.kernel_type = kernel_type.upper()

        if LES_filter_type in ("SMAGORINSKY", "LES_SMAGORINSKY"):
            self.model = SmagorinskyModel(
                max_particles,
                kernel_type,
                cs,
                ce,
                accumulator_dtype,
            )
        else:
            raise ValueError(f"Unknown LES filter type: {LES_filter_type}")

        self.particle_deltas = None
        self._viscosities_t_min_field = ti.field(dtype=accumulator_dtype, shape=())
        self._viscosities_t_max_field = ti.field(dtype=accumulator_dtype, shape=())
        self.viscosities_t_min = 0.0
        self.viscosities_t_max = 0.0
        self.viscosities_t_ratio_min = 0.0
        self.viscosities_t_ratio_max = 0.0

    def __str__(self) -> str:
        return str(self.model)

    @classmethod
    def rebuild(
        cls,
        turbulence_config: object,
        max_particles: int = MAX_PARTICLES,
        kernel_type: str = "GAUSSIAN",
        accumulator_dtype: ti.types = ti.f32,
    ) -> "ParticlesLES":
        model_name = getattr(turbulence_config, "model", "LES_SMAGORINSKY")
        cs = getattr(turbulence_config, "cs", SMAGORINSKY_CONSTANT)
        ce = getattr(turbulence_config, "ce", 1.048)
        return cls(
            LES_filter_type=model_name,
            max_particles=max_particles,
            kernel_type=kernel_type,
            cs=cs,
            ce=ce,
            accumulator_dtype=accumulator_dtype,
        )

    def initialize(self, particles) -> None:
        self.model.initialize(particles)

    def compute(self, particles, dt: float = None):
        """
        Main interface for SGS eddy viscosity computation.
        Delegates to the specific model implementation.
        """
        self.model.compute(particles, dt)
        self.update_turbulence_statistics(particles)

    def update_turbulence_statistics(self, particles):
        N = len(particles)
        if N == 0:
            return
        self._seed_stats_kernel(
            self._viscosities_t_min_field,
            self._viscosities_t_max_field,
            particles.viscosity_turbulent,
        )
        self._update_stats_kernel(
            self._viscosities_t_min_field,
            self._viscosities_t_max_field,
            particles.viscosity_turbulent,
            N,
        )
        self.viscosities_t_min = float(self._viscosities_t_min_field[None])
        self.viscosities_t_max = float(self._viscosities_t_max_field[None])
        if hasattr(self, "viscosity") and self.viscosity > 0.0:
            self.viscosities_t_ratio_min = self.viscosities_t_min / self.viscosity
            self.viscosities_t_ratio_max = self.viscosities_t_max / self.viscosity
        elif N > 0:
            visc = particles.viscosity[0]
            if visc > 0:
                self.viscosity = visc
                self.viscosities_t_ratio_min = self.viscosities_t_min / self.viscosity
                self.viscosities_t_ratio_max = self.viscosities_t_max / self.viscosity

    @ti.kernel
    def _seed_stats_kernel(self, vt_min: ti.template(), vt_max: ti.template(), vt: ti.template()):
        vt_min[None] = vt[0]
        vt_max[None] = vt[0]

    @ti.kernel
    def _update_stats_kernel(
        self,
        vt_min: ti.template(),
        vt_max: ti.template(),
        vt: ti.template(),
        N: ti.i32,
    ):
        # Top-level loop: Taichi only parallelises the outermost for, so nesting
        # this inside `if N > 0` serialised the reduction (measured 0.446 ms vs
        # 0.207 ms at N = 100k).  The seeding is a separate kernel for that reason.
        for i in range(N):
            ti.atomic_min(vt_min[None], vt[i])
            ti.atomic_max(vt_max[None], vt[i])
