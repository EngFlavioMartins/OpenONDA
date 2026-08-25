"""Equilibrium Smagorinsky SGS model used by the VPM solver."""

import taichi as ti

from ..config.constants import MAX_N_PARTICLES, SMAGORINSKY_CONSTANT


@ti.data_oriented
class SmagorinskyModel:
    """Equilibrium Smagorinsky eddy-viscosity model."""

    def __init__(
        self,
        max_n_particles: int = MAX_N_PARTICLES,
        particle_kernel: str = "GAUSSIAN",
        smagorinsky_coefficient: float = SMAGORINSKY_CONSTANT,
        subgrid_dissipation_coefficient: float = 1.048,
        accumulator_dtype: ti.types = ti.f32,
    ) -> None:
        self.model_name = "SMAGORINSKY"
        self.max_n_particles = max_n_particles
        self.particle_kernel = particle_kernel.upper()
        self.smagorinsky_coefficient = smagorinsky_coefficient
        self.subgrid_dissipation_coefficient = subgrid_dissipation_coefficient
        self.subgrid_kinetic_energy_coefficient = (
            smagorinsky_coefficient**2 * subgrid_dissipation_coefficient**0.5
        ) ** (2.0 / 3.0)

        self._filter_width = ti.field(dtype=accumulator_dtype, shape=max_n_particles)
        self._strain_rate_magnitude = ti.field(dtype=accumulator_dtype, shape=max_n_particles)

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

        n_particles_total = len(particles)
        if n_particles_total == 0:
            return

        self._compute_filter_width(
            particles.particle_volume,
            self._filter_width,
            n_particles_total,
        )
        self._compute_strain_rate_magnitude(
            particles.strain_rate,
            self._strain_rate_magnitude,
            n_particles_total,
        )
        self._compute_eddy_viscosity(
            self._filter_width,
            self._strain_rate_magnitude,
            n_particles_total,
            particles.kinematic_viscosity,
            particles.eddy_viscosity,
            particles.effective_viscosity,
            self.subgrid_kinetic_energy_coefficient,
            self.subgrid_dissipation_coefficient,
        )

    def report_rows(self) -> list:
        """Return the turbulence-model configuration as log detail rows."""
        return [
            ("model", "Smagorinsky"),
            ("c_s", f"{self.smagorinsky_coefficient:.4f}"),
            ("c_k", f"{self.subgrid_kinetic_energy_coefficient:.6f}"),
            ("c_e", f"{self.subgrid_dissipation_coefficient:.4f}"),
            ("filter width", "V_p^(1/3)"),
        ]

    @ti.kernel
    def _compute_filter_width(
        self,
        particle_volume: ti.template(),
        filter_width: ti.template(),
        n_particles_total: ti.i32,
    ):
        for i in range(n_particles_total):
            local_particle_volume = particle_volume[i]
            filter_width[i] = (
                ti.pow(local_particle_volume, 1.0 / 3.0) if local_particle_volume > 0.0 else 0.0
            )

    @ti.kernel
    def _compute_strain_rate_magnitude(
        self,
        strain_rate: ti.template(),
        strain_rate_magnitude: ti.template(),
        n_particles_total: ti.i32,
    ):
        for i in range(n_particles_total):
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
        n_particles_total: ti.i32,
        kinematic_viscosity: ti.template(),
        eddy_viscosity: ti.template(),
        effective_viscosity: ti.template(),
        subgrid_kinetic_energy_coefficient: ti.f32,
        subgrid_dissipation_coefficient: ti.f32,
    ):
        for i in range(n_particles_total):
            delta = filter_width[i]
            strain_magnitude = strain_rate_magnitude[i]
            equilibrium_energy = (
                subgrid_kinetic_energy_coefficient
                * delta
                * delta
                * strain_magnitude
                * strain_magnitude
                / subgrid_dissipation_coefficient
            )
            computed_eddy_viscosity = (
                subgrid_kinetic_energy_coefficient
                * delta
                * ti.sqrt(ti.max(equilibrium_energy, 0.0))
            )

            if ti.math.isnan(computed_eddy_viscosity) or ti.math.isinf(computed_eddy_viscosity):
                computed_eddy_viscosity = 0.0

            eddy_viscosity[i] = computed_eddy_viscosity
            effective_viscosity[i] = kinematic_viscosity[i] + computed_eddy_viscosity
