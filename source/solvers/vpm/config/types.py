"""Aggregate imports for VPM configuration and state types.

Subsystem modules remain the canonical definition sites. This module exists only
as a convenient import surface inside the VPM package; it does not provide
legacy-name aliases.
"""

import sys

from .advection import AdvectionConfig
from .divergence_relaxation import DivergenceRelaxationConfig
from .filament_refinement import FilamentRefinementConfig
from .setup import VPMSetup
from .stabilization import StabilizationConfig
from .state import (
    ParticlesState,
    SolverState,
    cached_particle_property,
    set_flow_model,
)
from .stretching import StretchingConfig
from .turbulence import TurbulenceConfig
from .velocity import VelocityConfig
from .viscous import ViscousConfig

sys.tracebacklimit = 0

__all__ = [
    "AdvectionConfig",
    "DivergenceRelaxationConfig",
    "FilamentRefinementConfig",
    "ParticlesState",
    "SolverState",
    "StabilizationConfig",
    "StretchingConfig",
    "TurbulenceConfig",
    "VPMSetup",
    "VelocityConfig",
    "ViscousConfig",
    "cached_particle_property",
    "set_flow_model",
]
