"""Aggregate imports for VPM configuration and state types.

Subsystem modules remain the canonical definition sites. This module exists only
as a convenient import surface inside the VPM package; it does not provide
legacy-name aliases.
"""

from .advection import AdvectionConfig
from .case import Numerics, RestartState, RunPlan, VPMCase
from .diagnostics import DiagnosticsConfig
from .divergence_relaxation import DivergenceRelaxationConfig
from .filament_refinement import FilamentRefinementConfig
from .health import HealthLimits
from .stabilization import StabilizationConfig
from .state import cached_particle_property, set_flow_model
from .stretching import StretchingConfig
from .turbulence import TurbulenceConfig
from .velocity import VelocityConfig
from .viscous import ViscousConfig

__all__ = [
    "AdvectionConfig",
    "DivergenceRelaxationConfig",
    "DiagnosticsConfig",
    "FilamentRefinementConfig",
    "HealthLimits",
    "Numerics",
    "StabilizationConfig",
    "StretchingConfig",
    "TurbulenceConfig",
    "RestartState",
    "RunPlan",
    "VPMCase",
    "VelocityConfig",
    "ViscousConfig",
    "cached_particle_property",
    "set_flow_model",
]
