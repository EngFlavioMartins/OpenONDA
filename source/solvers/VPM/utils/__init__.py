"""
Utility subpackage: field samplers, analytic flow models, and offline diagnostics.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from .field_samplers import LineSampler, SurfaceSampler
from .flow_models import (
    DoubletFlowVPM,
    IsotropicTurbulenceVPM,
    LambOseenVPM,
    TaylorGreenVortexVPM,
    VortexRingVPM,
)
from .offline_diagnostics import ComputeOfflineDiagnostics, OfflineFlowDiagnostics

__all__ = [
    # Field samplers
    "SurfaceSampler",
    "LineSampler",
    # Offline diagnostics
    "OfflineFlowDiagnostics",
    "ComputeOfflineDiagnostics",
    # Flow models
    "LambOseenVPM",
    "VortexRingVPM",
    "DoubletFlowVPM",
    "TaylorGreenVortexVPM",
    "IsotropicTurbulenceVPM",
]
