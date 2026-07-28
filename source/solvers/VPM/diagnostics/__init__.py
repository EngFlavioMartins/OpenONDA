"""
VPM Diagnostics Module
======================
Conservation diagnostics and validation tools for VPM simulations.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: February 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from .conservation import ConservationTracker
from .resolution import discretization_health

__all__ = ["ConservationTracker", "discretization_health"]
