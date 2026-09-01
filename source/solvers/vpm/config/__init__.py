"""Configuration API for the Vortex Particle Method solver.

Importing this package does not initialize Taichi. Backend initialization occurs
only when a VPM or VLM solver is constructed.
"""

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
    "TurbulenceConfig",
    "VPMCase",
    "ViscousConfig",
]
