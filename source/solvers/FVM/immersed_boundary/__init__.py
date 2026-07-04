"""Immersed Boundary Method (discrete direct forcing) for the FVM solver.

Implements the Pinelli et al. (2010) / Uhlmann (2005) marker-based direct
forcing IBM as adapted to collocated finite-volume PISO solvers by
Constant et al. (docs/literature/Constant2016.pdf).  Design notes in
docs/plans/2026-07-fvm-ibm-design.md.
"""

from .body import ImmersedBody
from .forcing import IBMForcing

__all__ = ["ImmersedBody", "IBMForcing"]
