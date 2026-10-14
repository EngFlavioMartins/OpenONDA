"""LES turbulence-model execution for the VPM solver."""

import math
from typing import TYPE_CHECKING

import taichi as ti

from ..config.constants import MAX_N_PARTICLES, SMAGORINSKY_CONSTANT
from .sigma import SIGMA_CONSTANT, SigmaModel
from .smagorinsky import SmagorinskyModel
from .strength_exchange import StrengthExchange

if TYPE_CHECKING:
    from ..particles.container import Particles


@ti.data_oriented
class ParticlesLES:
    """Evaluate the configured VPM sub-grid eddy-viscosity model.

    The model (:class:`SmagorinskyModel` or :class:`SigmaModel`) writes eddy
    viscosity into the particle container. An optional consistency switch scales
    it by ``s = sin^2(theta) / (sin^2(theta) + sin^2(theta_0))``, with ``theta`` the
    angle between a particle's strength and the resolved vorticity at the particle.
    The transport either adds the eddy viscosity to core spreading or applies it by
    :class:`StrengthExchange` after the coupled update. Reported viscosity values are
    in m²/s; ratios to molecular kinematic viscosity are dimensionless. Host
    statistics describe the last explicit ``update_turbulence_statistics`` call;
    ``compute`` updates physical viscosity without synchronizing for logging.
    """

    consistency_angle: float | None = None
    strength_exchange: StrengthExchange | None = None

    def __init__(
        self,
        model_name: str,
        max_n_particles: int = MAX_N_PARTICLES,
        particle_kernel: str = "GAUSSIAN",
        smagorinsky_coefficient: float = SMAGORINSKY_CONSTANT,
        subgrid_dissipation_coefficient: float = 1.048,
        accumulator_dtype: ti.types = ti.f32,
        filter_width: float | None = None,
        sigma_coefficient: float = SIGMA_CONSTANT,
        consistency_angle: float | None = None,
        subgrid_transport: str = "CORE_SPREADING",
        particle_dtype: ti.types = ti.f32,
    ) -> None:
        """Create a particle LES model and allocate its reduction fields.

        Parameters
        ----------
        model_name : str
            Registered model name: ``"SMAGORINSKY"``, ``"LES_SMAGORINSKY"`` or
            ``"LES_SIGMA"``.
        max_n_particles : int, default=MAX_N_PARTICLES
            Particle capacity used for device reductions.
        particle_kernel : str, default="GAUSSIAN"
            VPM radial kernel used by the wrapped model.
        smagorinsky_coefficient : float, default=SMAGORINSKY_CONSTANT
            Dimensionless Smagorinsky coefficient.
        subgrid_dissipation_coefficient : float, default=1.048
            Dimensionless model coefficient for sub-grid dissipation.
        accumulator_dtype : taichi scalar type, default=ti.f32
            Precision of device reduction fields.
        filter_width : float or None, default=None
            Fixed LES filter in metres; None uses particle volume^(1/3).
        sigma_coefficient : float, default=1.35
            Dimensionless sigma-model constant.
        consistency_angle : float or None, default=None
            Switch reference angle ``theta_0`` in degrees; None disables the switch.
        subgrid_transport : {'CORE_SPREADING', 'STRENGTH_EXCHANGE'}
            How the eddy viscosity acts on the particles.
        particle_dtype : taichi scalar type, default=ti.f32
            Precision of the particle fields, for the exchange work fields.

        Raises
        ------
        ValueError
            If ``model_name`` is not supported.
        """
        self.model_name = model_name.upper()
        self.max_n_particles = max_n_particles
        self.particle_kernel = particle_kernel.upper()
        if self.model_name not in {"SMAGORINSKY", "LES_SMAGORINSKY", "LES_SIGMA"}:
            raise ValueError(f"Unknown LES model: {model_name!r}")
        self.subgrid_transport = subgrid_transport.upper()
        if self.subgrid_transport not in {"CORE_SPREADING", "STRENGTH_EXCHANGE"}:
            raise ValueError(f"Unknown sub-grid transport: {subgrid_transport!r}")
        self.consistency_angle = consistency_angle

        if self.model_name == "LES_SIGMA":
            self.model = SigmaModel(
                max_n_particles=max_n_particles,
                sigma_coefficient=sigma_coefficient,
                accumulator_dtype=accumulator_dtype,
                filter_width=filter_width,
            )
        else:
            self.model = SmagorinskyModel(
                max_n_particles=max_n_particles,
                particle_kernel=particle_kernel,
                smagorinsky_coefficient=smagorinsky_coefficient,
                subgrid_dissipation_coefficient=subgrid_dissipation_coefficient,
                accumulator_dtype=accumulator_dtype,
                filter_width=filter_width,
            )
        self.strength_exchange = (
            StrengthExchange(max_n_particles, particle_dtype)
            if self.subgrid_transport == "STRENGTH_EXCHANGE"
            else None
        )

        self._min_eddy_viscosity_field = ti.field(dtype=accumulator_dtype, shape=())
        self._max_eddy_viscosity_field = ti.field(dtype=accumulator_dtype, shape=())
        self.min_eddy_viscosity = 0.0
        self.max_eddy_viscosity = 0.0
        self.min_eddy_viscosity_ratio = 0.0
        self.max_eddy_viscosity_ratio = 0.0
        self.kinematic_viscosity = 0.0

    @property
    def model_coefficient(self) -> float:
        """Return the active model constant; zero disables the closure."""
        if isinstance(self.model, SigmaModel):
            return self.model.sigma_coefficient
        return self.model.smagorinsky_coefficient

    def report_rows(self) -> list:
        """Return the active turbulence model's configuration as log detail rows."""
        rows = self.model.report_rows()
        if self.consistency_angle is not None:
            rows.append(("consistency switch", f"theta_0 = {self.consistency_angle:g} deg"))
        rows.append(("sub-grid transport", self.subgrid_transport.lower().replace("_", " ")))
        return rows

    @classmethod
    def rebuild(
        cls,
        turbulence_config: object,
        max_n_particles: int = MAX_N_PARTICLES,
        particle_kernel: str = "GAUSSIAN",
        accumulator_dtype: ti.types = ti.f32,
    ) -> "ParticlesLES":
        """Construct LES from a configuration-like object."""
        return cls(
            sigma_coefficient=getattr(turbulence_config, "sigma_coefficient", SIGMA_CONSTANT),
            consistency_angle=getattr(turbulence_config, "consistency_angle", None),
            subgrid_transport=getattr(turbulence_config, "subgrid_transport", "CORE_SPREADING"),
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
            filter_width=getattr(turbulence_config, "filter_width", None),
        )

    def initialize(self, particles: "Particles") -> None:
        """Initialize wrapped-model fields from a particle container."""
        self.model.initialize(particles)

    def compute(
        self,
        particles: "Particles",
        time_step_size: float | None = None,
    ) -> None:
        """Compute eddy viscosity for the current particle state.

        Parameters
        ----------
        particles : Particles
            Mutable particle container whose ``eddy_viscosity`` field is
            updated in m²/s.
        time_step_size : float, optional
            Physical step in seconds for models that use temporal scaling.
        """
        if self.model_coefficient == 0.0:
            self.min_eddy_viscosity = 0.0
            self.max_eddy_viscosity = 0.0
            self.min_eddy_viscosity_ratio = 0.0
            self.max_eddy_viscosity_ratio = 0.0
            return
        self.model.compute(particles, time_step_size)
        if self.consistency_angle is not None or self.strength_exchange is not None:
            reference = (
                math.sin(math.radians(self.consistency_angle)) ** 2
                if self.consistency_angle is not None
                else -1.0
            )
            self._switch_and_transport(
                particles.vortex_strength,
                particles.velocity_gradient,
                particles.kinematic_viscosity,
                particles.eddy_viscosity,
                particles.effective_viscosity,
                reference,
                int(self.strength_exchange is not None),
                len(particles),
            )

    def apply_strength_exchange(self, particles: "Particles", time_step_size: float) -> None:
        """Diffuse strengths with the current eddy viscosity when exchange transport is set."""
        if self.strength_exchange is not None and self.model_coefficient > 0.0:
            self.strength_exchange.apply(particles, time_step_size)

    @ti.kernel
    def _switch_and_transport(
        self,
        vortex_strength: ti.template(),
        velocity_gradient: ti.template(),
        kinematic_viscosity: ti.template(),
        eddy_viscosity: ti.template(),
        effective_viscosity: ti.template(),
        reference: ti.f32,
        exchange: ti.i32,
        count: ti.i32,
    ):
        for i in range(count):
            nu_t = eddy_viscosity[i]
            if reference >= 0.0:
                J = velocity_gradient[i]
                vorticity = ti.Vector([J[2, 1] - J[1, 2], J[0, 2] - J[2, 0], J[1, 0] - J[0, 1]])
                strength = vortex_strength[i]
                strength_sq = strength.dot(strength)
                vorticity_sq = vorticity.dot(vorticity)
                switch = 0.0
                if strength_sq > 1e-30 and vorticity_sq > 1e-30:
                    sine_sq = ti.max(
                        1.0 - strength.dot(vorticity) ** 2 / (strength_sq * vorticity_sq), 0.0
                    )
                    switch = sine_sq / (sine_sq + reference)
                nu_t *= switch
                eddy_viscosity[i] = nu_t
            # With strength exchange the cores spread with the molecular viscosity only.
            effective_viscosity[i] = kinematic_viscosity[i] + (1 - exchange) * nu_t

    def update_turbulence_statistics(self, particles: "Particles") -> None:
        """Sample active viscosity extrema and ratios on the diagnostic cadence.

        Updates host measurements only; particle viscosity is unchanged. The
        physical ``compute`` path does not call this reduction or synchronize
        merely for console output. The sampler requests it when needed.
        """
        n_particles_total = len(particles)
        if n_particles_total == 0:
            self.min_eddy_viscosity = self.max_eddy_viscosity = 0.0
            self.min_eddy_viscosity_ratio = self.max_eddy_viscosity_ratio = 0.0
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
        for i in range(n_particles_total):
            ti.atomic_min(minimum[None], eddy_viscosity[i])
            ti.atomic_max(maximum[None], eddy_viscosity[i])
