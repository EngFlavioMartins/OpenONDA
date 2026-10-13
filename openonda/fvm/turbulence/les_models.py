"""Construction of the configured FVM subgrid-scale eddy-viscosity model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .smagorinsky import EquilibriumSmagorinsky, Smagorinsky

if TYPE_CHECKING:
    from ..config.types import TurbulenceConfig


def create_model(
    config: TurbulenceConfig, mesh_data: dict[str, object], geo_data: dict[str, object]
) -> Smagorinsky | EquilibriumSmagorinsky | None:
    """Build the configured LES model instance from a :class:`TurbulenceConfig`.

    ``"none"``, ``"iles"`` and ``"dns"`` select no subgrid model;
    ``"smagorinsky"`` and ``"equilibriumsmagorinsky"`` build
    :class:`Smagorinsky` and :class:`EquilibriumSmagorinsky`.

    Args:
        config:   :class:`TurbulenceConfig` instance (may be ``None``).
        mesh_data: Mesh dictionary.
        geo_data:  Geometry dictionary.

    Returns:
        An LES model instance with a ``compute_eddy_viscosity(velocity)`` interface,
        or ``None`` for no-model (ILES/DNS).
    """
    if config is None:
        return None
    name = str(config.model).lower()
    if name in ("none", "", "iles", "dns"):
        return None
    if name in ("equilibriumsmagorinsky", "equilibrium_smagorinsky"):
        return EquilibriumSmagorinsky(
            mesh_data,
            geo_data,
            subgrid_kinetic_energy_coefficient=config.subgrid_kinetic_energy_coefficient,
            subgrid_dissipation_coefficient=config.subgrid_dissipation_coefficient,
        )
    return Smagorinsky(mesh_data, geo_data, smagorinsky_coefficient=config.smagorinsky_coefficient)
