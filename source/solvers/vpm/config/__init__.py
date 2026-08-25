"""Configuration API for the Vortex Particle Method solver.

Importing this package does not initialize Taichi. Backend initialization occurs
only when a VPM or VLM solver is constructed.
"""

from .advection import AdvectionConfig
from .divergence_relaxation import DivergenceRelaxationConfig
from .filament_refinement import FilamentRefinementConfig
from .setup import PanelBodySetup, VPMSetup
from .stabilization import StabilizationConfig
from .state import ParticlesState, SolverState
from .stretching import StretchingConfig
from .turbulence import TurbulenceConfig
from .velocity import VelocityConfig
from .viscous import ViscousConfig

__all__ = [
    "AdvectionConfig",
    "DivergenceRelaxationConfig",
    "FilamentRefinementConfig",
    "ParticlesState",
    "PanelBodySetup",
    "SolverState",
    "StabilizationConfig",
    "StretchingConfig",
    "TurbulenceConfig",
    "VPMSetup",
    "VelocityConfig",
    "ViscousConfig",
]
