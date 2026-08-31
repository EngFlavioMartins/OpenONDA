"""LES turbulence-model orchestration for the VPM solver."""

import taichi as ti

from ..config.constants import MAX_N_PARTICLES, SMAGORINSKY_CONSTANT
from .smagorinsky import SmagorinskyModel


@ti.data_oriented
class ParticlesLES:
    """Evaluate the configured VPM sub-grid eddy-viscosity model."""

    def __init__(
        self,
        model_name: str,
        max_n_particles: int = MAX_N_PARTICLES,
        particle_kernel: str = "GAUSSIAN",
        smagorinsky_coefficient: float = SMAGORINSKY_CONSTANT,
        subgrid_dissipation_coefficient: float = 1.048,
        accumulator_dtype: ti.types = ti.f32,
    ) -> None:
        self.model_name = model_name.upper()
        self.max_n_particles = max_n_particles
        self.particle_kernel = particle_kernel.upper()
        if self.model_name not in {"SMAGORINSKY", "LES_SMAGORINSKY"}:
            raise ValueError(f"Unknown LES model: {model_name!r}")

        self.model = SmagorinskyModel(
            max_n_particles=max_n_particles,
            particle_kernel=particle_kernel,
            smagorinsky_coefficient=smagorinsky_coefficient,
            subgrid_dissipation_coefficient=subgrid_dissipation_coefficient,
            accumulator_dtype=accumulator_dtype,
        )

        self._min_eddy_viscosity_field = ti.field(dtype=accumulator_dtype, shape=())
        self._max_eddy_viscosity_field = ti.field(dtype=accumulator_dtype, shape=())
        self.min_eddy_viscosity = 0.0
        self.max_eddy_viscosity = 0.0
        self.min_eddy_viscosity_ratio = 0.0
        self.max_eddy_viscosity_ratio = 0.0
        self.kinematic_viscosity = 0.0

    def report_rows(self) -> list:
        """Return the active turbulence model's configuration as log detail rows."""
        return self.model.report_rows()

    @classmethod
    def rebuild(
        cls,
        turbulence_config: object,
        max_n_particles: int = MAX_N_PARTICLES,
        particle_kernel: str = "GAUSSIAN",
        accumulator_dtype: ti.types = ti.f32,
    ) -> "ParticlesLES":
        return cls(
            model_name=getattr(turbulence_config, "model", "LES_SMAGORINSKY"),
            max_n_particles=max_n_particles,
            particle_kernel=particle_kernel,
            smagorinsky_coefficient=getattr(
                turbulence_config, "smagorinsky_coefficient", SMAGORINSKY_CONSTANT
            ),
            subgrid_dissipation_coefficient=getattr(
                turbulence_config, "subgrid_dissipation_coefficient", 1.048
            ),
            accumulator_dtype=accumulator_dtype,
        )

    def initialize(self, particles) -> None:
        self.model.initialize(particles)

    def compute(
        self,
        particles,
        time_step_size: float | None = None,
    ) -> None:
        if self.model.smagorinsky_coefficient == 0.0:
            self.min_eddy_viscosity = 0.0
            self.max_eddy_viscosity = 0.0
            self.min_eddy_viscosity_ratio = 0.0
            self.max_eddy_viscosity_ratio = 0.0
            return
        self.model.compute(particles, time_step_size)
        self.update_turbulence_statistics(particles)

    def update_turbulence_statistics(self, particles) -> None:
        n_particles_total = len(particles)
        if n_particles_total == 0:
            return

        self._seed_statistics(
            self._min_eddy_viscosity_field,
            self._max_eddy_viscosity_field,
            particles.eddy_viscosity,
        )
        self._reduce_statistics(
            self._min_eddy_viscosity_field,
            self._max_eddy_viscosity_field,
            particles.eddy_viscosity,
            n_particles_total,
        )
        ti.sync()

        self.min_eddy_viscosity = float(self._min_eddy_viscosity_field[None])
        self.max_eddy_viscosity = float(self._max_eddy_viscosity_field[None])

        kinematic_viscosity = float(particles.kinematic_viscosity[0])
        self.kinematic_viscosity = kinematic_viscosity
        if kinematic_viscosity > 0.0:
            self.min_eddy_viscosity_ratio = self.min_eddy_viscosity / kinematic_viscosity
            self.max_eddy_viscosity_ratio = self.max_eddy_viscosity / kinematic_viscosity

    @ti.kernel
    def _seed_statistics(
        self,
        minimum: ti.template(),
        maximum: ti.template(),
        eddy_viscosity: ti.template(),
    ):
        minimum[None] = eddy_viscosity[0]
        maximum[None] = eddy_viscosity[0]

    @ti.kernel
    def _reduce_statistics(
        self,
        minimum: ti.template(),
        maximum: ti.template(),
        eddy_viscosity: ti.template(),
        n_particles_total: ti.i32,
    ):
        ti.loop_config(serialize=True)
        for i in range(n_particles_total):
            minimum[None] = ti.min(minimum[None], eddy_viscosity[i])
            maximum[None] = ti.max(maximum[None], eddy_viscosity[i])
