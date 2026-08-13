"""Analytic flow-field initializers for VPM.

Canonical vortex solutions (Lamb-Oseen, vortex ring, doublet, Taylor-Green,
isotropic turbulence) that produce particle fields as numpy arrays.  These are
used to initialize a VPM simulation and as verification/validation references.
"""

from .flow_models import (
    DoubletFlowVPM,
    IsotropicTurbulenceVPM,
    LambOseenVPM,
    TaylorGreenVortexVPM,
    VortexRingVPM,
    vortex_ring_centerline,
)

__all__ = [
    "DoubletFlowVPM",
    "IsotropicTurbulenceVPM",
    "LambOseenVPM",
    "TaylorGreenVortexVPM",
    "VortexRingVPM",
    "vortex_ring_centerline",
]
