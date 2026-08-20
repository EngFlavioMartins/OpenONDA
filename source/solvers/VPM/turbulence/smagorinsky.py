"""Equilibrium Smagorinsky SGS model used by the VPM solver."""

from __future__ import annotations

import taichi as ti

from ..config.constants import MAX_PARTICLES, SMAGORINSKY_CONSTANT


@ti.data_oriented
class SmagorinskyModel:
    """Equilibrium Smagorinsky eddy-viscosity model."""

    def __init__(
        self,
        max_particles: int = MAX_PARTICLES,
        particle_kernel: str = "GAUSSIAN",
        c_s: float = SMAGORINSKY_CONSTANT,
        c_e: float = 1.048,
        accumulator_dtype: ti.types = ti.f32,
    ) -> None:
        self.model_name = "SMAGORINSKY"
        self.max_particles = max_particles
        self.particle_kernel = particle_kernel.upper()
        self.c_s = c_s
        self.c_e = c_e
        self.c_k = (c_s**2 * c_e**0.5) ** (2.0 / 3.0)

        self._filter_width = ti.field(dtype=accumulator_dtype, shape=max_particles)
        self._strain_rate_magnitude = ti.field(dtype=accumulator_dtype, shape=max_particles)

    def initialize(self, particles) -> None:
        """Initialize model state; no additional state is required."""
        del particles

    def compute(
        self,
        particles,
        time_step_size: float | None = None,
    ) -> None:
        """Evaluate eddy and effective viscosity on the current particles."""
        del time_step_size

        n_particles = len(particles)
        if n_particles == 0:
            return

        self._compute_filter_width(
            particles.volume,
            self._filter_width,
            n_particles,
        )
        self._compute_strain_rate_magnitude(
            particles.strain_rate,
            self._strain_rate_magnitude,
            n_particles,
        )
        self._compute_eddy_viscosity(
            self._filter_width,
            self._strain_rate_magnitude,
            n_particles,
            particles.kinematic_viscosity,
            particles.eddy_viscosity,
            particles.effective_viscosity,
            self.c_k,
            self.c_e,
        )

    def __str__(self) -> str:
        return "\n".join(
            [
                "  Model type   : Smagorinsky",
                f"  C_s          : {self.c_s:.4f}",
                f"  C_k          : {self.c_k:.6f}",
                f"  C_e          : {self.c_e:.4f}",
                "  Filter width : V_p^(1/3)",
            ]
        )

    @ti.kernel
    def _compute_filter_width(
        self,
        volume: ti.template(),
        filter_width: ti.template(),
        n_particles: ti.i32,
    ):
        for i in range(n_particles):
            particle_volume = volume[i]
            filter_width[i] = ti.pow(particle_volume, 1.0 / 3.0) if particle_volume > 0.0 else 0.0

    @ti.kernel
    def _compute_strain_rate_magnitude(
        self,
        strain_rate: ti.template(),
        strain_rate_magnitude: ti.template(),
        n_particles: ti.i32,
    ):
        for i in range(n_particles):
            squared_norm = 0.0
            for a in ti.static(range(3)):
                for b in ti.static(range(3)):
                    component = strain_rate[i][a, b]
                    squared_norm += component * component
            strain_rate_magnitude[i] = ti.sqrt(2.0 * squared_norm)

    @ti.kernel
    def _compute_eddy_viscosity(
        self,
        filter_width: ti.template(),
        strain_rate_magnitude: ti.template(),
        n_particles: ti.i32,
        kinematic_viscosity: ti.template(),
        eddy_viscosity: ti.template(),
        effective_viscosity: ti.template(),
        c_k: ti.f32,
        c_e: ti.f32,
    ):
        for i in range(n_particles):
            delta = filter_width[i]
            strain_magnitude = strain_rate_magnitude[i]
            equilibrium_energy = c_k * delta * delta * strain_magnitude * strain_magnitude / c_e
            nu_t = c_k * delta * ti.sqrt(ti.max(equilibrium_energy, 0.0))

            if ti.math.isnan(nu_t) or ti.math.isinf(nu_t):
                nu_t = 0.0

            eddy_viscosity[i] = nu_t
            effective_viscosity[i] = kinematic_viscosity[i] + nu_t
