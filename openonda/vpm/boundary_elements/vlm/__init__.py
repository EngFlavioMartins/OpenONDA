"""
Vortex-lattice-method (VLM) subpackage: geometry, solver, kinematics, and VPM
coupling.
"""

# Kinematics classes for moving surfaces
from .config import ForceConfig, VLMConfig, VLMMeshConfig, VLMSurface
from .coupling.kinematics import (
    CompositeVLM,
    HeavingVLM,
    LinearPeriodicVLM,
    ManeuverVLM,
    PitchingVLM,
    RotatingVLM,
    StaticVLM,
    TranslatingVLM,
    VLMKinematics,
)
from .geometry.aircraft import Aircraft, Wing, WingSegment
from .solver.field import BoundSurfaceFieldSettings
from .solver.lattice import VLMLattice
from .solver.vlm_solver import VLMSolver

__all__ = [
    "Aircraft",
    "BoundSurfaceFieldSettings",
    "CompositeVLM",
    "ForceConfig",
    "HeavingVLM",
    "LinearPeriodicVLM",
    "ManeuverVLM",
    "PitchingVLM",
    "RotatingVLM",
    "StaticVLM",
    "TranslatingVLM",
    "VLMConfig",
    "VLMKinematics",
    "VLMLattice",
    "VLMMeshConfig",
    "VLMSolver",
    "VLMSurface",
    "Wing",
    "WingSegment",
]
