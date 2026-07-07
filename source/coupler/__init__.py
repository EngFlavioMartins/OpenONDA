"""
FVM-VPM Coupler module for OpenONDA.
====================================
Hybrid near-field (OpenFOAM) / far-field (VPM) simulations: the FVM
resolves the body and near wake inside a box whose boundary is driven by
the particle field; the near-field vorticity is conservatively handed
back to the particles every step.

Organization:
- config/types.py: CouplerSetup (one flat coupling-setup dataclass)
- core/solver.py:  FVMVPMCoupler (the four-step coupling loop)
- core/helpers/:   hand-off, fringe relaxation, case setup, I/O redirection
- diagnostics/:    conservation recovery and validation signals

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: June 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from .config.types import CouplerConfig, CouplerSetup

def __getattr__(name: str):
    if name == "FVMVPMCoupler":
        try:
            from .core.solver import FVMVPMCoupler
        except ImportError:
            print(
                "ERROR: FVMVPMCoupler import failed. "
                "Check Cython extensions / OpenFOAM linkage."
            )
            raise
        return FVMVPMCoupler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "CouplerConfig",
    "CouplerSetup",
    "FVMVPMCoupler",
]
