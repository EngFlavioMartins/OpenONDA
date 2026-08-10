"""Public construction helper for coupled FVM–VPM simulations."""

from __future__ import annotations

from dataclasses import replace

from .config.types import CouplerSetup


def setup_coupler(vpm_solver, fvm_solver, setup: CouplerSetup):
    """Connect configured VPM and FVM solvers through ``setup``.

    Rank ownership, native-case preparation, and dependency injection remain
    internal so case files contain only solver physics and this single
    coupling operation.
    """
    from .core.solver import FVMVPMCoupler

    runtime_setup = setup
    if getattr(fvm_solver, "case_dir", None) is not None:
        runtime_setup = replace(
            setup,
            case_dir=fvm_solver.case_dir,
        )

    FVMVPMCoupler.prepare_case(runtime_setup, vpm_solver=vpm_solver)
    return FVMVPMCoupler.from_solvers(
        runtime_setup,
        fvm_solver=fvm_solver,
        vpm_solver=vpm_solver,
    )


__all__ = ["setup_coupler"]
