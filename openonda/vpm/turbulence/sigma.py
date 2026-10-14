"""Sigma eddy-viscosity model (Nicoud, Toda, Cabrit, Bose & Lee 2011)."""

import math
from typing import TYPE_CHECKING

import taichi as ti

from ..config.constants import MAX_N_PARTICLES

if TYPE_CHECKING:
    from ..particles.container import Particles

SIGMA_CONSTANT = 1.35


@ti.data_oriented
class SigmaModel:
    """Sigma model: ``nu_t = (C Delta)^2 s3 (s1 - s2)(s2 - s3) / s1^2``.

    ``s1 >= s2 >= s3`` are the singular values of the resolved velocity gradient
    and ``Delta`` the filter width. The operator vanishes for two-dimensional
    flow, pure rotation, pure shear and axisymmetric strain, and decays as r^3
    at a wall. It does not vanish on a curved vortex tube, whose hoop strain is a
    third singular value of order u_r/R.
    """

    def __init__(
        self,
        max_n_particles: int = MAX_N_PARTICLES,
        sigma_coefficient: float = SIGMA_CONSTANT,
        accumulator_dtype: ti.types = ti.f32,
        filter_width: float | None = None,
    ) -> None:
        """Configure the model.

        Parameters
        ----------
        max_n_particles : int, default=MAX_N_PARTICLES
            Particle capacity; the model allocates no per-particle fields.
        sigma_coefficient : float, default=1.35
            Dimensionless model constant ``C``.
        accumulator_dtype : Taichi primitive type, default=ti.f32
            Kept for interface parity with :class:`SmagorinskyModel`.
        filter_width : float or None, default=None
            Fixed filter width in metres; None uses particle volume^(1/3).
        """
        del accumulator_dtype
        self.model_name = "SIGMA"
        self.max_n_particles = max_n_particles
        self.sigma_coefficient = sigma_coefficient
        self.filter_width = filter_width
        self._fixed_filter_width = 0.0 if filter_width is None else float(filter_width)

    def initialize(self, particles: "Particles") -> None:
        """Initialize model state; no additional state is required."""
        del particles

    def compute(self, particles: "Particles", time_step_size: float | None = None) -> None:
        """Write eddy and effective viscosity from the particle velocity gradient."""
        del time_step_size
        count = len(particles)
        if count == 0:
            return
        self._compute_eddy_viscosity(
            particles.velocity_gradient,
            particles.particle_volume,
            particles.kinematic_viscosity,
            particles.eddy_viscosity,
            particles.effective_viscosity,
            self.sigma_coefficient,
            self._fixed_filter_width,
            count,
        )

    def report_rows(self) -> list:
        """Return the turbulence-model configuration as log detail rows."""
        return [
            ("model", "Sigma (Nicoud et al. 2011)"),
            ("c_sigma", f"{self.sigma_coefficient:.4f}"),
            (
                "filter width",
                "V_p^(1/3)" if self.filter_width is None else f"{self.filter_width:.8g} m (fixed)",
            ),
        ]

    @ti.kernel
    def _compute_eddy_viscosity(
        self,
        velocity_gradient: ti.template(),
        particle_volume: ti.template(),
        kinematic_viscosity: ti.template(),
        eddy_viscosity: ti.template(),
        effective_viscosity: ti.template(),
        coefficient: ti.f32,
        fixed_filter_width: ti.f32,
        count: ti.i32,
    ):
        for i in range(count):
            # Singular values from the invariants of G = J^T J (Hasan, Ewing &
            # Nicoud's closed form), avoiding an iterative decomposition.
            J = velocity_gradient[i]
            G = J.transpose() @ J
            i1 = G.trace()
            i2 = 0.5 * (i1 * i1 - (G @ G).trace())
            i3 = G.determinant()
            a1 = ti.max(i1 * i1 / 9.0 - i2 / 3.0, 0.0)
            a2 = i1 * i1 * i1 / 27.0 - i1 * i2 / 6.0 + i3 / 2.0
            nu_t = 0.0
            if a1 > 1e-30:
                a3 = ti.acos(ti.min(ti.max(a2 / ti.pow(a1, 1.5), -1.0), 1.0)) / 3.0
                root = 2.0 * ti.sqrt(a1)
                s1 = ti.sqrt(ti.max(i1 / 3.0 + root * ti.cos(a3), 0.0))
                s2 = ti.sqrt(ti.max(i1 / 3.0 - root * ti.cos(math.pi / 3.0 + a3), 0.0))
                s3 = ti.sqrt(ti.max(i1 / 3.0 - root * ti.cos(math.pi / 3.0 - a3), 0.0))
                if s1 > 1e-30:
                    delta = ti.pow(ti.max(particle_volume[i], 0.0), 1.0 / 3.0)
                    if fixed_filter_width > 0.0:
                        delta = fixed_filter_width
                    d_sigma = ti.max(s3 * (s1 - s2) * (s2 - s3) / (s1 * s1), 0.0)
                    nu_t = (coefficient * delta) ** 2 * d_sigma
            eddy_viscosity[i] = nu_t
            effective_viscosity[i] = kinematic_viscosity[i] + nu_t
