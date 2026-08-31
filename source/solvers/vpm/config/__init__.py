"""Configuration API for the Vortex Particle Method solver.

Importing this package does not initialize Taichi. Backend initialization occurs
only when a VPM or VLM solver is constructed.
"""

from .advection import AdvectionConfig
from .artifacts import Backup, Samplers
from .case import Numerics, RestartState, RunPlan, VPMCase
from .diagnostics import DiagnosticsConfig
from .divergence_relaxation import DivergenceRelaxationConfig
from .filament_refinement import FilamentRefinementConfig
from .health import (
    DivergenceLimit,
    FiniteStateCheck,
    GrowthLimit,
    HealthError,
    HealthLimits,
    LagrangianCFLLimit,
    MisalignmentLimit,
    ParticleStrengthLimit,
)
from .setup import PanelBodySetup
from .stabilization import StabilizationConfig
from .stretching import StretchingConfig
from .turbulence import TurbulenceConfig
from .velocity import VelocityConfig
from .viscous import ViscousConfig

__all__ = [
    "AdvectionConfig",
    "Backup",
    "DivergenceRelaxationConfig",
    "DiagnosticsConfig",
    "DivergenceLimit",
    "FiniteStateCheck",
    "FilamentRefinementConfig",
    "Numerics",
    "PanelBodySetup",
    "GrowthLimit",
    "HealthError",
    "HealthLimits",
    "LagrangianCFLLimit",
    "ParticleStrengthLimit",
    "MisalignmentLimit",
    "RestartState",
    "RunPlan",
    "Samplers",
    "StabilizationConfig",
    "StretchingConfig",
    "TurbulenceConfig",
    "VPMCase",
    "VelocityConfig",
    "ViscousConfig",
]
