"""Configuration API for the Vortex Particle Method solver.

Importing this package does not initialize Taichi. Backend initialization occurs
only when a VPM or VLM solver is constructed.
"""

from .case import Numerics, RestartState, RunPlan, VPMCase
from .diagnostics import DiagnosticsConfig
from .divergence_relaxation import DivergenceRelaxationConfig
from .filament_refinement import FilamentRefinementConfig
from .output import Backup, Samplers
from .stabilization import StabilizationConfig
from .state_limits import (
    DivergenceLimit,
    FiniteStateCheck,
    GrowthLimit,
    LagrangianCFLLimit,
    MisalignmentLimit,
    ParticleStateError,
    ParticleStateLimits,
    ParticleStrengthLimit,
)
from .turbulence import TurbulenceConfig
from .viscous import ViscousConfig

__all__ = [
    "Backup",
    "DivergenceRelaxationConfig",
    "DiagnosticsConfig",
    "DivergenceLimit",
    "FiniteStateCheck",
    "FilamentRefinementConfig",
    "Numerics",
    "GrowthLimit",
    "ParticleStateError",
    "ParticleStateLimits",
    "LagrangianCFLLimit",
    "ParticleStrengthLimit",
    "MisalignmentLimit",
    "RestartState",
    "RunPlan",
    "Samplers",
    "StabilizationConfig",
    "TurbulenceConfig",
    "VPMCase",
    "ViscousConfig",
]
