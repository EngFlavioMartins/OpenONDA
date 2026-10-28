"""Coupled FVM--VPM solver of OpenONDA.

The FVM resolves the body and the near wake inside the FVM domain, whose outer
boundary receives its boundary data from the particle field; the FVM vorticity is
transferred to the particles in the transfer region while the VPM carries the
outer wake::

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
