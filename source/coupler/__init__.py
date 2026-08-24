"""
FVM-VPM Coupler module for OpenONDA.
====================================
Hybrid near-field (FVM) / far-field (VPM) simulations: the native FVM
resolves the body and near wake inside a box whose boundary is driven by the
particle field; the FVM-authoritative part of the overlap is replaced from the
absolute FVM cell circulation every step while the outer wake is retained.

The public driver follows the physical coupling sequence. Boundary sampling,
vorticity transfer, checkpointing, and reporting live in focused modules
beside it.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: June 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from .config.types import CouplerSetup
from .factory import create_coupler
from .solver import FVMVPMCoupler

__all__ = [
    "CouplerSetup",
    "FVMVPMCoupler",
    "create_coupler",
]
