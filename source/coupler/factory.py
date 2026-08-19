"""Public construction helper for coupled FVM–VPM simulations."""

from __future__ import annotations

from .config.types import CouplerSetup


def create_coupler(fvm_solver, vpm_solver, coupler_setup: CouplerSetup):
    """Connect configured FVM and VPM solvers through ``coupler_setup``.

    The native solvers retain ownership of their physics, mesh, and output
    directories; this function only creates the coupling driver.
    """
    from .solver import FVMVPMCoupler

    return FVMVPMCoupler(fvm_solver, vpm_solver, coupler_setup)


__all__ = ["create_coupler"]
