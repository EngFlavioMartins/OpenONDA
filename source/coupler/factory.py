"""Public construction helper for coupled FVM–VPM simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config.types import CouplerSetup

if TYPE_CHECKING:
    from source.solvers.fvm import FVMSolver
    from source.solvers.vpm import VPMSolver

    from .solver import FVMVPMCoupler


def create_coupler(
    fvm_solver: FVMSolver,
    vpm_solver: VPMSolver | None,
    coupler_setup: CouplerSetup,
) -> FVMVPMCoupler:
    """Connect configured FVM and VPM solvers through ``coupler_setup``.

    The native solvers retain ownership of their physics, mesh, and output
    directories; this function only creates the coupling driver.
    """
    from .solver import FVMVPMCoupler

    return FVMVPMCoupler(fvm_solver, vpm_solver, coupler_setup)


__all__ = ["create_coupler"]
