# OpenONDA/__init__.py


from . import solvers

from .utilities import (
    vortex_filament_model, vortex_ring_model, inflating_dipole_model,
    doublet_flow_model, set_initial_condition, set_eulerian_module,
    set_boundary_conditions, lamb_oseen_vpm, vortex_ring_vpm,
    isotropic_turbulence_vpm, get_rectangular_point_distribuition,
    get_2D_rectangular_point_distribuition, get_hexagonal_point_distribution,
    get_cylindrical_point_distribuition
)

__all__ = [
    "solvers",
    "vortex_filament_model", "vortex_ring_model", "inflating_dipole_model",
    "doublet_flow_model", "set_initial_condition", "set_eulerian_module",
    "set_boundary_conditions", "lamb_oseen_vpm", "vortex_ring_vpm",
    "isotropic_turbulence_vpm", "get_rectangular_point_distribuition",
    "get_2D_rectangular_point_distribuition", "get_hexagonal_point_distribution",
    "get_cylindrical_point_distribuition"
]


# Package metadata
__version__ = "0.0.1"
