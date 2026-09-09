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
    """Construct the driver joining configured FVM and VPM solver instances.

    Parameters
    ----------
    fvm_solver : FVMSolver
        Initialized or constructible native FVM solver. It owns the mesh,
        cell-centred fields, boundary patches, density, viscosity, FVM time
        step, and output directory. Every MPI rank passes its local instance.
    vpm_solver : VPMSolver or None
        VPM solver owning the particle cloud and its macro time step. Rank zero
        supplies the real solver; non-owner MPI ranks pass ``None`` (or the
        inactive-rank placeholder produced by the VPM factory).
    coupler_setup : CouplerSetup
        Coupling-owned transfer, boundary-trace, diagnostic, and backup policy.

    Returns
    -------
    FVMVPMCoupler
        Uninitialized driver that retains references to both solvers. Call
        :meth:`FVMVPMCoupler.initialize`, :meth:`FVMVPMCoupler.run`, or
        :meth:`FVMVPMCoupler.solve` next.

    Raises
    ------
    ValueError
        If ``fvm_solver`` is ``None``. Cross-solver compatibility is checked
        when the returned driver is initialized.

    Notes
    -----
    Neither native solver is copied. Construction creates the coupled solution
    directory and logging redirector on rank zero but does not advance time.

    Examples
    --------
    >>> driver = create_coupler(fvm_solver, vpm_solver, CouplerSetup())
    >>> final_step = driver.run(max_coupling_steps=1)
    """
    from .solver import FVMVPMCoupler

    return FVMVPMCoupler(fvm_solver, vpm_solver, coupler_setup)


__all__ = ["create_coupler"]
