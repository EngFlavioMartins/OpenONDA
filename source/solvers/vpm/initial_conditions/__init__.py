"""Analytic flow-field initializers for VPM.

Canonical vortex solutions (Lamb-Oseen, vortex ring, doublet, Taylor-Green,
isotropic turbulence) that produce particle fields as numpy arrays.  These are
used to initialize a VPM simulation and as verification/validation references.
"""

from .flow_models import (
    doublet_flow_vpm,
    isotropic_turbulence_vpm,
    lamb_oseen_vpm,
    taylor_green_vortex_vpm,
    vortex_ring_centreline,
    vortex_ring_vpm,
)

__all__ = [
    "doublet_flow_vpm",
    "isotropic_turbulence_vpm",
    "lamb_oseen_vpm",
    "taylor_green_vortex_vpm",
    "vortex_ring_vpm",
    "vortex_ring_centreline",
]
