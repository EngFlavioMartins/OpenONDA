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
    model : {'DNS', 'LES_SMAGORINSKY', 'INVISCID'}, default='DNS'
        ``DNS`` applies no subgrid closure, ``LES_SMAGORINSKY`` computes an
        equilibrium eddy viscosity, and ``INVISCID`` disables viscous/SGS
        modeling at this layer. Names are normalized uppercase.
    smagorinsky_coefficient : float, default=0.2
        Non-negative dimensionless ``C_s`` in the filter-width eddy-viscosity
        model.
    subgrid_dissipation_coefficient : float, default=1.048
        Positive dimensionless equilibrium dissipation coefficient ``C_e``.
    filter_width : float or None, default=None
        Fixed LES filter width in metres. None retains the local particle
        volume^(1/3) rule. A fixed width separates the closure scale from
        quadrature changes during spatial refinement or redistribution.

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
    LES updates per-particle eddy viscosity from the accepted strain-rate
    field; it does not directly modify circulation. Viscous configuration must
    select a diffusion path that supports variable effective viscosity (GBD).
    """

    model: Literal["DNS", "LES_SMAGORINSKY", "INVISCID"] = "DNS"
    """Turbulence-model identifier."""

    smagorinsky_coefficient: float = constants_module.SMAGORINSKY_CONSTANT
    """Smagorinsky coefficient ``C_s``."""

    subgrid_dissipation_coefficient: float = 1.048
    """SGS dissipation coefficient ``C_e``."""

    filter_width: float | None = None
    """Optional fixed LES filter width in metres."""

    flow_model: str = field(default="DNS", init=False)
    """Solver flow-model category derived from ``model``."""

    def __post_init__(self) -> None:
        model = self.model.upper()
        valid_models = {"DNS", "LES_SMAGORINSKY", "INVISCID"}
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
        flow_model = {
            "DNS": "DNS",
            "LES_SMAGORINSKY": "LES",
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
    ) -> TurbulenceConfig:
        """Return the equilibrium Smagorinsky LES configuration."""
        return TurbulenceConfig(
            model="LES_SMAGORINSKY",
            smagorinsky_coefficient=smagorinsky_coefficient,
            subgrid_dissipation_coefficient=subgrid_dissipation_coefficient,
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
