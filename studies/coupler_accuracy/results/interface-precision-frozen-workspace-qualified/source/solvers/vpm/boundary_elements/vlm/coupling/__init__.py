"""
VLM coupling subpackage: kinematics drivers used by the integrated VPM solver.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from .kinematics import (
    HeavingVLM,
    ManeuverVLM,
    PitchingVLM,
    RotatingVLM,
    StaticVLM,
    TranslatingVLM,
    VLMKinematics,
)

__all__ = [
    "VLMKinematics",
    "StaticVLM",
    "TranslatingVLM",
    "RotatingVLM",
    "ManeuverVLM",
    "HeavingVLM",
    "PitchingVLM",
]
