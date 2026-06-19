"""
Initialization module for vlm.
==================
Initialization module for vlm. module.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

# Kinematics classes for moving surfaces
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
from .solver.vlm_solver import ForceConfig, VLMSolver

__all__ = [
    "Aircraft",
    "Wing",
    "WingSegment",
    "VLMSolver",
    "ForceConfig",
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
