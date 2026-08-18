"""
FVM-VPM Coupler module for OpenONDA.
====================================
Hybrid near-field (FVM) / far-field (VPM) simulations: the native FVM
resolves the body and near wake inside a box whose boundary is driven by the
particle field; the near-field vorticity is conservatively handed back to the
particles every step.

The public driver follows the physical coupling sequence. Boundary sampling,
blending, vorticity handoff, conservation, checkpointing, and reporting live
in physically named modules beside it.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: June 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from .config.types import CouplerSetup
from .factory import setup_coupler
from .solver import FVMVPMCoupler

__all__ = [
    "CouplerSetup",
    "FVMVPMCoupler",
    "setup_coupler",
]
