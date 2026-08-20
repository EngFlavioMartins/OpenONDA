"""LES turbulence-model orchestration for the VPM solver."""

import taichi as ti

from ..config.constants import MAX_PARTICLES, SMAGORINSKY_CONSTANT
from .smagorinsky import SmagorinskyModel


@ti.data_oriented
class ParticlesLES:
    """Evaluate the configured VPM sub-grid eddy-viscosity model."""

    def __init__(
        self,
        model_name: str,
        max_particles: int = MAX_PARTICLES,
        particle_kernel: str = "GAUSSIAN",
        c_s: float = SMAGORINSKY_CONSTANT,
        c_e: float = 1.048,
        accumulator_dtype: ti.types = ti.f32,
    ) -> None:
        self.model_name = model_name.upper()
        self.max_particles = max_particles
        self.particle_kernel = particle_kernel.upper()

        if self.model_name not in {"SMAGORINSKY", "LES_SMAGORINSKY"}:
            raise ValueError(f"Unknown LES model: {model_name!r}")

        self.model = SmagorinskyModel(
            max_particles=max_particles,
            particle_kernel=particle_kernel,
            c_s=c_s,
            c_e=c_e,
            accumulator_dtype=accumulator_dtype,
        )

        self._eddy_viscosity_min_field = ti.field(dtype=accumulator_dtype, shape=())
        self._eddy_viscosity_max_field = ti.field(dtype=accumulator_dtype, shape=())
        self.eddy_viscosity_min = 0.0
        self.eddy_viscosity_max = 0.0
        self.eddy_viscosity_ratio_min = 0.0
        self.eddy_viscosity_ratio_max = 0.0
        self.kinematic_viscosity = 0.0

    def __str__(self) -> str:
        return str(self.model)

    @classmethod
    def rebuild(
        cls,
        turbulence_config: object,
        max_particles: int = MAX_PARTICLES,
        particle_kernel: str = "GAUSSIAN",
        accumulator_dtype: ti.types = ti.f32,
    ) -> "ParticlesLES":
        return cls(
            model_name=getattr(turbulence_config, "model", "LES_SMAGORINSKY"),
            max_particles=max_particles,
            particle_kernel=particle_kernel,
            c_s=getattr(turbulence_config, "c_s", SMAGORINSKY_CONSTANT),
            c_e=getattr(turbulence_config, "c_e", 1.048),
            accumulator_dtype=accumulator_dtype,
        )

    def initialize(self, particles) -> None:
        self.model.initialize(particles)

    def compute(
        self,
        particles,
        time_step_size: float | None = None,
    ) -> None:
        self.model.compute(particles, time_step_size)
        self.update_turbulence_statistics(particles)

    def update_turbulence_statistics(self, particles) -> None:
        n_particles = len(particles)
        if n_particles == 0:
            return

        self._seed_statistics(
            self._eddy_viscosity_min_field,
            self._eddy_viscosity_max_field,
            particles.eddy_viscosity,
        )
        self._reduce_statistics(
            self._eddy_viscosity_min_field,
            self._eddy_viscosity_max_field,
            particles.eddy_viscosity,
            n_particles,
        )

        self.eddy_viscosity_min = float(self._eddy_viscosity_min_field[None])
        self.eddy_viscosity_max = float(self._eddy_viscosity_max_field[None])

        molecular_viscosity = float(particles.kinematic_viscosity[0])
        self.kinematic_viscosity = molecular_viscosity
        if molecular_viscosity > 0.0:
            self.eddy_viscosity_ratio_min = self.eddy_viscosity_min / molecular_viscosity
            self.eddy_viscosity_ratio_max = self.eddy_viscosity_max / molecular_viscosity

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
        n_particles: ti.i32,
    ):
        for i in range(n_particles):
            ti.atomic_min(minimum[None], eddy_viscosity[i])
            ti.atomic_max(maximum[None], eddy_viscosity[i])
