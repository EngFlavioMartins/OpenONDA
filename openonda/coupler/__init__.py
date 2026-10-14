"""Coupled FVM--VPM solver of OpenONDA.

The FVM resolves the body and the near wake inside a box whose outer boundary
is driven by the particle field; the FVM vorticity renews the particles in the
overlap while the VPM carries the outer wake::

    import openonda.coupler as coupling

    with coupling.CoupledSolver(fvm_case, vpm_case, coupling.CouplerConfig()) as solver:
        solver.run(start_from="latest")
"""

from openonda.core.schedules import EverySteps, EveryTime, FinalOnly
from openonda.core.simulation.forcing import VelocityRamp
from openonda.core.status import RunStatus

from .config import Backup, CouplerConfig
from .solver import CoupledSolver

__all__ = [
    "Backup",
    "CoupledSolver",
    "CouplerConfig",
    "EverySteps",
    "EveryTime",
    "FinalOnly",
    "RunStatus",
    "VelocityRamp",
]
