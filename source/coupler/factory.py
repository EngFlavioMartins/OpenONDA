"""Public construction helper for coupled FVM–VPM simulations."""

from __future__ import annotations

from .config.types import CouplerSetup


def setup_coupler(vpm_solver, fvm_solver, setup: CouplerSetup):
    """Connect configured VPM and FVM solvers through ``setup``.

    The native solvers retain ownership of their physics, mesh, and output
    directories; this function only creates the coupling driver.
    """
    from .solver import FVMVPMCoupler

    return FVMVPMCoupler(vpm_solver, fvm_solver, setup)


__all__ = ["setup_coupler"]
