"""Configuration of the vortex particle method solver.

Importing this package does not initialize Taichi; the compute backend is
initialized when a solver is constructed.
"""

from .case import Numerics, RunPlan, VPMCase
from .divergence_relaxation import DivergenceRelaxationConfig
from .filament_refinement import FilamentRefinementConfig
from .logging_config import LoggingConfig
from .output import Backup, Output, Samplers
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
    "DivergenceLimit",
    "DivergenceRelaxationConfig",
    "FilamentRefinementConfig",
    "FiniteStateCheck",
    "GrowthLimit",
    "LagrangianCFLLimit",
    "LoggingConfig",
    "MisalignmentLimit",
    "Numerics",
    "Output",
    "ParticleStateError",
    "ParticleStateLimits",
    "ParticleStrengthLimit",
    "RunPlan",
    "Samplers",
    "StabilizationConfig",
    "TurbulenceConfig",
    "VPMCase",
    "ViscousConfig",
]
