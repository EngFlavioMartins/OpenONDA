"""Subgrid-scale turbulence closure of the FVM solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ._validation import finite_real, lowercase_choice

TURBULENCE_MODELS = frozenset({"dns", "equilibrium_smagorinsky", "smagorinsky"})


@dataclass(frozen=True, slots=True)
class TurbulenceConfig:
    """LES closure and its dimensionless coefficients.

    Parameters
    ----------
    model : {'dns', 'smagorinsky', 'equilibrium_smagorinsky'}, default='dns'
        ``dns`` applies no subgrid closure. The Smagorinsky models add a
        cell-centred eddy viscosity ``nu_t`` in m²/s, and the momentum
        equation uses ``nu + nu_t``.
    smagorinsky_coefficient : float, default=0.17
        Non-negative Smagorinsky coefficient ``C_s``.
    subgrid_kinetic_energy_coefficient : float, default=0.094
        Non-negative equilibrium coefficient ``C_k``.
    subgrid_dissipation_coefficient : float, default=1.048
        Positive equilibrium dissipation coefficient ``C_e``.

    Use the constructors :meth:`dns`, :meth:`smagorinsky` and
    :meth:`equilibrium_smagorinsky`.
    """

    model: Literal["dns", "smagorinsky", "equilibrium_smagorinsky"] = "dns"
    smagorinsky_coefficient: float = 0.17
    subgrid_kinetic_energy_coefficient: float = 0.094
    subgrid_dissipation_coefficient: float = 1.048

    def __post_init__(self) -> None:
        lowercase_choice("TurbulenceConfig.model", self.model, TURBULENCE_MODELS)
        for name, strict in (
            ("smagorinsky_coefficient", False),
            ("subgrid_kinetic_energy_coefficient", False),
            ("subgrid_dissipation_coefficient", True),
        ):
            object.__setattr__(
                self,
                name,
                finite_real(
                    f"TurbulenceConfig.{name}",
                    getattr(self, name),
                    minimum=0.0,
                    strict_minimum=strict,
                ),
            )

    @staticmethod
    def dns() -> TurbulenceConfig:
        """Return direct numerical simulation without a subgrid closure."""
        return TurbulenceConfig(model="dns")

    @staticmethod
    def smagorinsky(smagorinsky_coefficient: float = 0.17) -> TurbulenceConfig:
        """Return the classical Smagorinsky closure."""
        return TurbulenceConfig(
            model="smagorinsky", smagorinsky_coefficient=smagorinsky_coefficient
        )

    @staticmethod
    def equilibrium_smagorinsky(
        subgrid_kinetic_energy_coefficient: float = 0.094,
        subgrid_dissipation_coefficient: float = 1.048,
    ) -> TurbulenceConfig:
        """Return the algebraic-equilibrium Smagorinsky closure.

        The equivalent classical coefficient is
        ``C_s = C_k**(3/4) / C_e**(1/4)``.
        """
        coefficient_k = finite_real(
            "subgrid_kinetic_energy_coefficient", subgrid_kinetic_energy_coefficient, minimum=0.0
        )
        coefficient_e = finite_real(
            "subgrid_dissipation_coefficient",
            subgrid_dissipation_coefficient,
            minimum=0.0,
            strict_minimum=True,
        )
        return TurbulenceConfig(
            model="equilibrium_smagorinsky",
            smagorinsky_coefficient=coefficient_k**0.75 / coefficient_e**0.25,
            subgrid_kinetic_energy_coefficient=coefficient_k,
            subgrid_dissipation_coefficient=coefficient_e,
        )
