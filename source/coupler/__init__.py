"""
FVM-VPM Coupler module for OpenONDA.
====================================
Hybrid near-field (OpenFOAM) / far-field (VPM) simulations: the FVM
resolves the body and near wake inside a box whose boundary is driven by
the particle field; the near-field vorticity is conservatively handed
back to the particles every step.

Organization:
- config/types.py: CouplerConfig (one flat, physics-driven dataclass)
- core/solver.py:  FVMVPMCoupler (the four-step coupling loop)
- core/helpers/:   hand-off, fringe relaxation, case setup, I/O redirection
- diagnostics/:    conservation recovery and validation signals

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: June 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from source.solvers.VPM.config.types import StabilizationConfig, ViscousConfig  # noqa: F401

from .config.types import CouplerConfig

try:
    from .core.solver import FVMVPMCoupler
except ImportError:
    print("ERROR: FVMVPMCoupler import failed. Check Cython extensions / OpenFOAM linkage.")
    raise

__all__ = [
    "CouplerConfig",
    "FVMVPMCoupler",
    "StabilizationConfig",
    "ViscousConfig",
]
