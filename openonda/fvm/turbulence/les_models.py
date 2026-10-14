"""Construction of the configured FVM subgrid-scale eddy-viscosity model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .smagorinsky import EquilibriumSmagorinskyModel, SmagorinskyModel

if TYPE_CHECKING:
    from ..config.turbulence import TurbulenceConfig


def create_model(
    config: TurbulenceConfig, mesh_data: dict[str, object], geo_data: dict[str, object]
) -> SmagorinskyModel | EquilibriumSmagorinskyModel | None:
    """Return the eddy-viscosity model of ``config``, or ``None`` for ``dns``."""
    if config.model == "dns":
        return None
    if config.model == "equilibrium_smagorinsky":
        return EquilibriumSmagorinskyModel(
            mesh_data,
            geo_data,
            subgrid_kinetic_energy_coefficient=config.subgrid_kinetic_energy_coefficient,
            subgrid_dissipation_coefficient=config.subgrid_dissipation_coefficient,
        )
    return SmagorinskyModel(
        mesh_data, geo_data, smagorinsky_coefficient=config.smagorinsky_coefficient
    )
