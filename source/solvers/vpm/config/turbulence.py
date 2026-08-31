"""Turbulence-model configuration for the VPM solver."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal

from . import constants as constants_module


@dataclass(frozen=True)
class TurbulenceConfig:
    """Configure the VPM turbulence model.

    ``DNS`` applies no sub-grid closure. ``LES_SMAGORINSKY`` uses the current
    equilibrium Smagorinsky closure. ``INVISCID`` disables viscous and SGS
    turbulence modelling at the turbulence-model level.
    """

    model: Literal["DNS", "LES_SMAGORINSKY", "INVISCID"] = "DNS"
    """Turbulence-model identifier."""

    smagorinsky_coefficient: float = constants_module.SMAGORINSKY_CONSTANT
    """Smagorinsky coefficient ``C_s``."""

    subgrid_dissipation_coefficient: float = 1.048
    """SGS dissipation coefficient ``C_e``."""

    vortex_stretching_sfs_coefficient: float = 0.0
    """Constant coefficient of the anisotropic vortex-stretching SFS model.

    A value of zero disables the model.  Positive values apply the
    no-backscatter model within the reformulated VPM equations.
    """

    vortex_stretching_sfs_cutoff: float = 4.0
    """Gaussian support radius of the SFS interaction, in source core radii."""

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
        if (
            not math.isfinite(self.vortex_stretching_sfs_coefficient)
            or self.vortex_stretching_sfs_coefficient < 0.0
        ):
            raise ValueError(
                "vortex_stretching_sfs_coefficient must be finite and non-negative"
            )
        if (
            not math.isfinite(self.vortex_stretching_sfs_cutoff)
            or self.vortex_stretching_sfs_cutoff <= 0.0
        ):
            raise ValueError("vortex_stretching_sfs_cutoff must be finite and positive")

        flow_model = {
            "DNS": "DNS",
            "LES_SMAGORINSKY": "LES",
            "INVISCID": "INVISCID",
        }[model]

        object.__setattr__(self, "model", model)
        object.__setattr__(self, "flow_model", flow_model)

    @property
    def subgrid_kinetic_energy_coefficient(self) -> float:
        """Equivalent equilibrium SGS kinetic-energy coefficient ``C_k``."""
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
        vortex_stretching_sfs_coefficient: float = 0.0,
        vortex_stretching_sfs_cutoff: float = 4.0,
    ) -> TurbulenceConfig:
        """Return the equilibrium Smagorinsky LES configuration."""
        return TurbulenceConfig(
            model="LES_SMAGORINSKY",
            smagorinsky_coefficient=smagorinsky_coefficient,
            subgrid_dissipation_coefficient=subgrid_dissipation_coefficient,
            vortex_stretching_sfs_coefficient=vortex_stretching_sfs_coefficient,
            vortex_stretching_sfs_cutoff=vortex_stretching_sfs_cutoff,
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
