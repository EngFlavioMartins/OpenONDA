"""Configuration types for the VPM solver (compatibility facade).

The type definitions now live in per-subsystem modules under ``config/``:

- ``advection.py``            AdvectionConfig
- ``viscous.py``              ViscousConfig
- ``stretching.py``           StretchingConfig
- ``turbulence.py``           TurbulenceConfig
- ``velocity.py``             VelocityConfig
- ``filament_refinement.py``  FilamentRefinementConfig
- ``divergence_relaxation.py`` DivergenceRelaxationConfig
- ``stabilization.py``        StabilizationConfig
- ``setup.py``                VPMSetup (the umbrella that aggregates every subsystem)
- ``state.py``                SolverState, ParticlesState, CachedParticleProperty, SetFlowModel

This module re-exports them so ``from ...config.types import X`` keeps working;
new code should import from the per-subsystem module directly.
"""

import sys

from .advection import AdvectionConfig
from .divergence_relaxation import DivergenceRelaxationConfig
from .filament_refinement import FilamentRefinementConfig
from .setup import VPMSetup
from .stabilization import StabilizationConfig
from .state import (
    CachedParticleProperty,
    ParticlesState,
    SetFlowModel,
    SolverState,
)
from .stretching import StretchingConfig
from .turbulence import TurbulenceConfig
from .velocity import VelocityConfig
from .viscous import ViscousConfig

sys.tracebacklimit = 0

__all__ = [
    "AdvectionConfig",
    "CachedParticleProperty",
    "DivergenceRelaxationConfig",
    "FilamentRefinementConfig",
    "ParticlesState",
    "SetFlowModel",
    "SolverState",
    "StabilizationConfig",
    "StretchingConfig",
    "TurbulenceConfig",
    "VPMSetup",
    "VelocityConfig",
    "ViscousConfig",
]
