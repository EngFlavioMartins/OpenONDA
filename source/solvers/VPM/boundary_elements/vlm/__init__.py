"""
Vortex-lattice-method (VLM) subpackage: geometry, solver, kinematics, and VPM
coupling.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

# Kinematics classes for moving surfaces
from .config import ForceConfig, VLMMeshSetup, VLMSetup, VLMSurfaceSetup
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
from .solver.lattice import VLMLattice
from .solver.vlm_solver import VLMSolver

__all__ = [
    "Aircraft",
    "Wing",
    "WingSegment",
    "VLMSolver",
    "ForceConfig",
    "VLMSetup",
    "VLMSurfaceSetup",
    "VLMMeshSetup",
    "VLMLattice",
    # Kinematics
    "VLMKinematics",
    "StaticVLM",
    "TranslatingVLM",
    "RotatingVLM",
    "ManeuverVLM",
    "HeavingVLM",
    "PitchingVLM",
    "LinearPeriodicVLM",
    "CompositeVLM",
]
