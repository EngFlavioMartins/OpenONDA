"""
VPM Diagnostics Module
======================
Conservation diagnostics and validation tools for VPM simulations.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: February 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from .conservation import ConservationTracker
from .offline import ComputeOfflineDiagnostics, OfflineFlowDiagnostics
from .resolution import discretization_health
from .ring import RING_DIAGNOSTIC_COLUMNS, RingDiagnosticsSampler

__all__ = [
    "ComputeOfflineDiagnostics",
    "ConservationTracker",
    "OfflineFlowDiagnostics",
    "RING_DIAGNOSTIC_COLUMNS",
    "RingDiagnosticsSampler",
    "discretization_health",
]
