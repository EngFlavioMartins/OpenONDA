"""Turbulence-model configuration for the VPM solver."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal

from . import constants as constants_module


@dataclass(frozen=True)
class TurbulenceConfig:
    """Configure the VPM turbulence/LES closure.

    Parameters
    ----------
    model : {'DNS', 'LES_SMAGORINSKY', 'LES_SIGMA', 'INVISCID'}, default='DNS'
        ``DNS`` applies no subgrid closure, ``LES_SMAGORINSKY`` computes an
        equilibrium eddy viscosity, ``LES_SIGMA`` the sigma-model eddy
        viscosity, and ``INVISCID`` disables viscous/SGS modeling at this
        layer. Names are normalized uppercase.
    smagorinsky_coefficient : float, default=0.2
        Non-negative dimensionless ``C_s`` in the filter-width eddy-viscosity
        model.
    subgrid_dissipation_coefficient : float, default=1.048
        Positive dimensionless equilibrium dissipation coefficient ``C_e``.
    filter_width : float or None, default=None
        Fixed LES filter width in metres. None retains the local particle
        volume^(1/3) rule. A fixed width separates the closure scale from
        quadrature changes during spatial refinement or redistribution.
    sigma_coefficient : float, default=1.35
        Non-negative sigma-model constant (Nicoud et al. 2011).
    consistency_angle : float or None, default=None
        Reference angle in degrees, in ``(0, 90)``, of the consistency switch
        ``sin^2(theta) / (sin^2(theta) + sin^2(theta_0))``: ``theta`` is the angle
        between a particle's strength and the resolved vorticity at the particle.
        None applies the eddy viscosity everywhere.
    subgrid_transport : {'CORE_SPREADING', 'STRENGTH_EXCHANGE'}, default='CORE_SPREADING'
        ``CORE_SPREADING`` adds the eddy viscosity to the viscous scheme;
        ``STRENGTH_EXCHANGE`` diffuses strengths between neighbouring particles
        after each step and leaves the cores to the molecular viscosity.

    Attributes
    ----------
    flow_model : {'DNS', 'LES', 'INVISCID'}
        Derived solver category, excluded from the constructor.

    Raises
    ------
    ValueError
        If ``model`` is unsupported or a coefficient violates its range.

    Notes
    -----
    LES updates per-particle eddy viscosity from the accepted velocity-gradient
    field. With core-spreading transport it does not modify circulation
    directly; strength exchange conserves the total strength exactly.
    """

    model: Literal["DNS", "LES_SMAGORINSKY", "LES_SIGMA", "INVISCID"] = "DNS"
    """Turbulence-model identifier."""

    smagorinsky_coefficient: float = constants_module.SMAGORINSKY_CONSTANT
    """Smagorinsky coefficient ``C_s``."""

    subgrid_dissipation_coefficient: float = 1.048
    """SGS dissipation coefficient ``C_e``."""

    filter_width: float | None = None
    """Optional fixed LES filter width in metres."""

    sigma_coefficient: float = 1.35
    """Sigma-model constant ``C``."""

    consistency_angle: float | None = None
    """Consistency-switch reference angle in degrees, or None."""

    subgrid_transport: Literal["CORE_SPREADING", "STRENGTH_EXCHANGE"] = "CORE_SPREADING"
    """How the eddy viscosity acts on the particles."""

    flow_model: str = field(default="DNS", init=False)
    """Solver flow-model category derived from ``model``."""

    def __post_init__(self) -> None:
        model = self.model.upper()
        valid_models = {"DNS", "LES_SMAGORINSKY", "LES_SIGMA", "INVISCID"}
        if model not in valid_models:
            raise ValueError(
                f"Invalid turbulence model {self.model!r}; expected one of {sorted(valid_models)}"
            )
        if not math.isfinite(self.smagorinsky_coefficient) or self.smagorinsky_coefficient < 0.0:
            raise ValueError("smagorinsky_coefficient must be finite and non-negative")
        if (
            not math.isfinite(self.subgrid_dissipation_coefficient)
            or self.subgrid_dissipation_coefficient <= 0.0
        ):
            raise ValueError("subgrid_dissipation_coefficient must be finite and positive")
        if self.filter_width is not None and (
            not math.isfinite(self.filter_width) or self.filter_width <= 0.0
        ):
            raise ValueError("filter_width must be finite and positive or None")
        if not math.isfinite(self.sigma_coefficient) or self.sigma_coefficient < 0.0:
            raise ValueError("sigma_coefficient must be finite and non-negative")
        if self.consistency_angle is not None and not 0.0 < self.consistency_angle < 90.0:
            raise ValueError("consistency_angle must lie in (0, 90) degrees or be None")
        transport = self.subgrid_transport.upper()
        if transport not in {"CORE_SPREADING", "STRENGTH_EXCHANGE"}:
            raise ValueError(
                f"Invalid subgrid_transport {self.subgrid_transport!r}; expected "
                "'CORE_SPREADING' or 'STRENGTH_EXCHANGE'"
            )
        object.__setattr__(self, "subgrid_transport", transport)
        flow_model = {
            "DNS": "DNS",
            "LES_SMAGORINSKY": "LES",
            "LES_SIGMA": "LES",
            "INVISCID": "INVISCID",
        }[model]

        object.__setattr__(self, "model", model)
        object.__setattr__(self, "flow_model", flow_model)

    @property
    def subgrid_kinetic_energy_coefficient(self) -> float:
        """Return dimensionless ``C_k=(C_s²*sqrt(C_e))**(2/3)``."""
        return (self.smagorinsky_coefficient**2 * self.subgrid_dissipation_coefficient**0.5) ** (
            2.0 / 3.0
        )

    @staticmethod
    def dns() -> TurbulenceConfig:
        """Return DNS configuration without an SGS closure."""
        return TurbulenceConfig(model="DNS")

    @staticmethod
    def les_smagorinsky(
        smagorinsky_coefficient: float = constants_module.SMAGORINSKY_CONSTANT,
        subgrid_dissipation_coefficient: float = 1.048,
        filter_width: float | None = None,
        consistency_angle: float | None = None,
        subgrid_transport: str = "CORE_SPREADING",
    ) -> TurbulenceConfig:
        """Return the equilibrium Smagorinsky LES configuration."""
        return TurbulenceConfig(
            model="LES_SMAGORINSKY",
            smagorinsky_coefficient=smagorinsky_coefficient,
            subgrid_dissipation_coefficient=subgrid_dissipation_coefficient,
            filter_width=filter_width,
            consistency_angle=consistency_angle,
            subgrid_transport=subgrid_transport,
        )

    @staticmethod
    def les_sigma(
        sigma_coefficient: float = 1.35,
        consistency_angle: float | None = 3.0,
        subgrid_transport: str = "STRENGTH_EXCHANGE",
        filter_width: float | None = None,
    ) -> TurbulenceConfig:
        """Return the sigma-model LES, by default switched and applied by strength exchange.

        The consistency switch keeps the eddy viscosity off particles whose strength
        is aligned with the resolved vorticity, as in resolved laminar vortices, and
        turns it on where the particle field stops representing the vorticity.
        """
        return TurbulenceConfig(
            model="LES_SIGMA",
            sigma_coefficient=sigma_coefficient,
            consistency_angle=consistency_angle,
            subgrid_transport=subgrid_transport,
            filter_width=filter_width,
        )

    @staticmethod
    def equilibrium_smagorinsky(
        subgrid_kinetic_energy_coefficient: float = 0.094,
        subgrid_dissipation_coefficient: float = 1.048,
    ) -> TurbulenceConfig:
        """Configure Smagorinsky LES from equilibrium coefficients.

        The current implementation uses

        ``C_s = C_k**(3/4) / C_e**(1/4)``.
        """
        if (
            not math.isfinite(subgrid_kinetic_energy_coefficient)
            or subgrid_kinetic_energy_coefficient < 0.0
        ):
            raise ValueError("subgrid_kinetic_energy_coefficient must be finite and non-negative")
        if (
            not math.isfinite(subgrid_dissipation_coefficient)
            or subgrid_dissipation_coefficient <= 0.0
        ):
            raise ValueError("subgrid_dissipation_coefficient must be finite and positive")

        smagorinsky_coefficient = (
            subgrid_kinetic_energy_coefficient**0.75 / subgrid_dissipation_coefficient**0.25
        )
        return TurbulenceConfig.les_smagorinsky(
            smagorinsky_coefficient=smagorinsky_coefficient,
            subgrid_dissipation_coefficient=subgrid_dissipation_coefficient,
        )

    @staticmethod
    def inviscid() -> TurbulenceConfig:
        """Return inviscid turbulence-model configuration."""
        return TurbulenceConfig(model="INVISCID")
