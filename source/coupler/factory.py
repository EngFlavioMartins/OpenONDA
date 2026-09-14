"""Public construction helper for coupled FVM–VPM simulations."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from .config.types import CouplerSetup
from .parallel import collective_phase

if TYPE_CHECKING:
    from source.solvers.fvm import FVMSolver
    from source.solvers.fvm.config import FVMSetup
    from source.solvers.fvm.factory import MeshSource
    from source.solvers.vpm import VPMSolver
    from source.solvers.vpm.config import VPMCase

    from .solver import FVMVPMCoupler


def create_coupler(
    fvm_solver: FVMSolver | FVMSetup,
    vpm_solver: VPMSolver | VPMCase | None,
    coupler_setup: CouplerSetup,
    *,
    mesh: MeshSource | None = None,
    case_dir: str | Path | None = None,
    immersed_bodies=None,
    grid_spacing: float | None = None,
    require_empty_output: bool = False,
) -> FVMVPMCoupler:
    """Construct a coupled driver from configurations or existing solvers.

    Parameters
    ----------
    fvm_solver : FVMSetup or FVMSolver
        An FVM configuration creates the solver and establishes its requested
        MPI runtime internally. Existing solver instances remain supported.
    vpm_solver : VPMCase, VPMSolver or None
        Pair an FVM configuration with a VPM case. The factory constructs VPM
        only on rank zero; callers need no rank checks. With existing solver
        instances, rank zero passes its VPM instance and other ranks pass None.
    coupler_setup : CouplerSetup
        Coupling-owned transfer, boundary-trace, diagnostic, and backup policy.
    mesh : MeshSource or None, optional
        Mesh passed to FVM construction. Only valid with an FVM configuration.
    case_dir : path-like or None, optional
        Directory for both solvers; defaults to the VPM case directory. Only
        valid with an FVM configuration.
    immersed_bodies, grid_spacing : optional
        Immersed geometry passed to the FVM factory, which selects a compatible
        parallel layout and attaches the bodies before coupling construction.
    require_empty_output : bool, default=False
        Reject existing output directories before construction, collectively.

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
    Use the driver as a context manager to close factory-created solvers on
    success or failure. Externally supplied solvers remain caller-owned.
    Construction does not advance time.

    Examples
    --------
    >>> with create_coupler(fvm_setup, vpm_case, CouplerSetup(), mesh=mesh) as driver:
    ...     final_step = driver.run()
    """
    from source.solvers.fvm.config import FVMSetup

    from .solver import FVMVPMCoupler

    if not isinstance(fvm_solver, FVMSetup):
        if require_empty_output or any(
            value is not None for value in (mesh, case_dir, immersed_bodies, grid_spacing)
        ):
            raise ValueError("Mesh, directory and body options require an FVMSetup configuration")
        return FVMVPMCoupler(fvm_solver, vpm_solver, coupler_setup)

    from source.solvers.fvm.factory import create_fvm_solver
    from source.solvers.vpm import VPMSolver
    from source.solvers.vpm.config import VPMCase

    if not isinstance(vpm_solver, VPMCase):
        raise TypeError("An FVMSetup must be paired with a VPMCase")
    directory = Path(case_dir if case_dir is not None else vpm_solver.directory).resolve()
    with ExitStack() as resources:
        body_options = {}
        if immersed_bodies is not None:
            body_options = {"immersed_bodies": immersed_bodies, "grid_spacing": grid_spacing}
        if require_empty_output:
            body_options["require_empty_output"] = True
        fvm = resources.enter_context(
            create_fvm_solver(fvm_solver, case_dir=directory, mesh=mesh, **body_options)
        )
        vpm = None
        construction_error = None
        failure = None
        if fvm.parallel.is_root:
            try:
                vpm = VPMSolver(replace(vpm_solver, directory=directory))
                resources.callback(vpm.close)
            except BaseException as error:
                construction_error = error
                failure = f"{type(error).__name__}: {error}"
        failure = fvm.parallel.bcast(failure)
        if failure is not None:
            if construction_error is not None:
                raise construction_error
            raise RuntimeError(f"VPM construction failed on rank zero: {failure}")

        with collective_phase(fvm.parallel.comm, "driver construction"):
            driver = FVMVPMCoupler(fvm, vpm, coupler_setup)
        driver._owned_resources = resources.pop_all()
        return driver


__all__ = ["create_coupler"]
