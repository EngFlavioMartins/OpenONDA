"""
VLM coupling subpackage: the VPM coupler, kinematics drivers, and wake shedding.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from .coupler import CouplingConfig, VLMVPMCoupler
from .kinematics import (
    HeavingVLM,
    ManeuverVLM,
    PitchingVLM,
    RotatingVLM,
    StaticVLM,
    TranslatingVLM,
    VLMKinematics,
)
from .wake_shedding import VLMWakeShedder

__all__ = [
    "VLMVPMCoupler",
    "CouplingConfig",
    "VLMWakeShedder",
    "VLMKinematics",
    "StaticVLM",
    "TranslatingVLM",
    "RotatingVLM",
    "ManeuverVLM",
    "HeavingVLM",
    "PitchingVLM",
]
