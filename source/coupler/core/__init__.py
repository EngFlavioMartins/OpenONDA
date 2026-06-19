"""
Core subpackage for the Coupler module.
=======================================

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: June 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

try:
    from .solver import FVMVPMCoupler
except ImportError:
    FVMVPMCoupler = None  # type: ignore[assignment,misc]

__all__ = ["FVMVPMCoupler"]
