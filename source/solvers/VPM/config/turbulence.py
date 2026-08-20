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

    c_s: float = constants_module.SMAGORINSKY_CONSTANT
    """Smagorinsky coefficient ``C_s``."""

    c_e: float = 1.048
    """SGS dissipation coefficient ``C_e``."""

    flow_model: str = field(default="DNS", init=False)
    """Solver flow-model category derived from ``model``."""

    def __post_init__(self) -> None:
        model = self.model.upper()
        valid_models = {"DNS", "LES_SMAGORINSKY", "INVISCID"}
        if model not in valid_models:
            raise ValueError(
                f"Invalid turbulence model {self.model!r}; expected one of {sorted(valid_models)}"
            )
        if not math.isfinite(self.c_s) or self.c_s < 0.0:
            raise ValueError("c_s must be finite and non-negative")
        if not math.isfinite(self.c_e) or self.c_e <= 0.0:
            raise ValueError("c_e must be finite and positive")

        flow_model = {
            "DNS": "DNS",
            "LES_SMAGORINSKY": "LES",
            "INVISCID": "INVISCID",
        }[model]

        object.__setattr__(self, "model", model)
        object.__setattr__(self, "flow_model", flow_model)

    @property
    def c_k(self) -> float:
        """Equivalent equilibrium SGS kinetic-energy coefficient ``C_k``."""
        return (self.c_s**2 * self.c_e**0.5) ** (2.0 / 3.0)

    @staticmethod
    def dns() -> TurbulenceConfig:
        """Return DNS configuration without an SGS closure."""
        return TurbulenceConfig(model="DNS")

    @staticmethod
    def les_smagorinsky(
        c_s: float = constants_module.SMAGORINSKY_CONSTANT,
        c_e: float = 1.048,
    ) -> TurbulenceConfig:
        """Return the equilibrium Smagorinsky LES configuration."""
        return TurbulenceConfig(
            model="LES_SMAGORINSKY",
            c_s=c_s,
            c_e=c_e,
        )

    @staticmethod
    def equilibrium_smagorinsky(
        c_k: float = 0.094,
        c_e: float = 1.048,
    ) -> TurbulenceConfig:
        """Configure Smagorinsky LES from equilibrium coefficients.

        The current implementation uses

        ``C_s = C_k**(3/4) / C_e**(1/4)``.
        """
        if not math.isfinite(c_k) or c_k < 0.0:
            raise ValueError("c_k must be finite and non-negative")
        if not math.isfinite(c_e) or c_e <= 0.0:
            raise ValueError("c_e must be finite and positive")

        c_s = c_k**0.75 / c_e**0.25
        return TurbulenceConfig.les_smagorinsky(c_s=c_s, c_e=c_e)

    @staticmethod
    def inviscid() -> TurbulenceConfig:
        """Return inviscid turbulence-model configuration."""
        return TurbulenceConfig(model="INVISCID")
