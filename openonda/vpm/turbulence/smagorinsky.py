"""Equilibrium Smagorinsky SGS model used by the VPM solver."""

from typing import TYPE_CHECKING

import taichi as ti

from ..config.constants import MAX_N_PARTICLES, SMAGORINSKY_CONSTANT

if TYPE_CHECKING:
    from ..particles.container import Particles


@ti.data_oriented
class SmagorinskyModel:
    """Equilibrium Smagorinsky eddy-viscosity model."""

    def __init__(
        self,
        max_n_particles: int = MAX_N_PARTICLES,
        particle_kernel: str = "gaussian",
        smagorinsky_coefficient: float = SMAGORINSKY_CONSTANT,
        filter_width: float | None = None,
    ) -> None:
        """Configure and allocate the equilibrium Smagorinsky model.

        Parameters
        ----------
        max_n_particles : int, default=MAX_N_PARTICLES
            Fixed particle capacity for model work fields.
        particle_kernel : str, default="gaussian"
            Kernel label used by the surrounding VPM configuration. It is
            normalized to uppercase for reporting.
        smagorinsky_coefficient : float, default=SMAGORINSKY_CONSTANT
            Dimensionless model coefficient ``C_s``.
        filter_width : float or None, default=None
            Fixed LES filter in metres; None uses particle volume^(1/3).

        Notes
        -----
        Construction reads or mutates no particle container. :meth:`compute`
        later writes ``nu_t = (C_s Delta)^2 |S|`` and the effective viscosity
        for the active particle prefix in m²/s, with ``|S| = sqrt(2 S:S)``.
        """
        self.model_name = "smagorinsky"
        self.max_n_particles = max_n_particles
        self.particle_kernel = particle_kernel.lower()
        self.smagorinsky_coefficient = smagorinsky_coefficient
        self.filter_width = filter_width
        self._uses_fixed_filter = filter_width is not None

    def initialize(self, particles: "Particles") -> None:
        """Initialize model state; no additional state is required."""
        del particles

    def compute(
        self,
        particles: "Particles",
        time_step_size: float | None = None,
    ) -> None:
        """Evaluate eddy and effective viscosity on the current particles."""
        del time_step_size

        n_particles_total = len(particles)
        if n_particles_total == 0:
            return
        self._compute_eddy_viscosity(
            particles.particle_volume,
            particles.strain_rate,
            n_particles_total,
            particles.kinematic_viscosity,
            particles.eddy_viscosity,
            particles.effective_viscosity,
            self.smagorinsky_coefficient,
        )

    def report_rows(self) -> list:
        """Return the turbulence-model configuration as log detail rows."""
        return [
            ("model", "Smagorinsky"),
            ("c_s", f"{self.smagorinsky_coefficient:.4f}"),
            (
                "filter width",
                "V_p^(1/3)" if self.filter_width is None else f"{self.filter_width:.8g} m (fixed)",
            ),
        ]

    @ti.kernel
    def _compute_eddy_viscosity(
        self,
        particle_volume: ti.template(),
        strain_rate: ti.template(),
        n_particles_total: ti.i32,
        kinematic_viscosity: ti.template(),
        eddy_viscosity: ti.template(),
        effective_viscosity: ti.template(),
        smagorinsky_coefficient: float,
    ):
        for i in range(n_particles_total):
            delta = 0.0
            if ti.static(self._uses_fixed_filter):
                delta = self.filter_width
            else:
                local_particle_volume = particle_volume[i]
                delta = (
                    ti.pow(local_particle_volume, 1.0 / 3.0) if local_particle_volume > 0.0 else 0.0
                )
            squared_norm = 0.0
            for a in ti.static(range(3)):
                for b in ti.static(range(3)):
                    component = strain_rate[i][a, b]
                    squared_norm += component * component
            computed_eddy_viscosity = (
                smagorinsky_coefficient * smagorinsky_coefficient * delta * delta
            ) * ti.sqrt(2.0 * squared_norm)
            eddy_viscosity[i] = computed_eddy_viscosity
            effective_viscosity[i] = kinematic_viscosity[i] + computed_eddy_viscosity
