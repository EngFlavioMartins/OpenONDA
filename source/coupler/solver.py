"""FVM–VPM coupling driver with VPM boundary conditions and state replacement."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from contextlib import ExitStack
import logging
from numbers import Integral
import os
from pathlib import Path
import time
from typing import TYPE_CHECKING

try:
    from mpi4py import MPI as _MPI

    _mpi4py_comm = _MPI.COMM_WORLD
except ImportError:
    _mpi4py_comm = None

import numpy as np

from source.coupler.backup import (
    BACKUP_DIRECTORY,
    load_coupled_backup,
    publish_vpm_snapshot,
    save_coupled_backup,
)
from source.coupler.boundary import (
    advance_fvm,
    evaluate_vpm_boundary,
    initialize_vpm_boundary_history,
    update_boundary_history_after_replacement,
)
from source.coupler.config.types import CouplerSetup
from source.coupler.consistency import FVMConsistencyBand
from source.coupler.parallel import collective_phase
from source.coupler.reporting import (
    OutputRedirector,
    configure_logging,
    flush_log,
    format_coupler_log,
    format_coupler_step,
    record_step,
    write_run_metadata,
)
from source.coupler.vorticity_transfer import VorticityTransfer

if TYPE_CHECKING:
    from source.solvers.fvm import FVMSolver
    from source.solvers.vpm import VPMSolver

logger = logging.getLogger("coupler")


def _validate_gbd_moment_recovery(
    recovery: Mapping[str, bool | int | float] | None,
    correction_limit: float,
) -> None:
    """Fail fast when the just-completed GBD prune is not trustworthy."""
    if recovery is None:
        return
    numeric_names = (
        "nonzero_node_count",
        "retained_node_count",
        "pruned_node_count",
        "correction_fraction",
        "normalized_vortex_strength_residual",
        "normalized_linear_impulse_residual",
        "normalized_angular_impulse_residual",
    )
    try:
        values = {name: float(recovery[name]) for name in numeric_names}
        values["support_augmented_node_count"] = float(
            recovery.get("support_augmented_node_count", 0)
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("GBD moment-recovery diagnostics are incomplete or invalid") from error
    for name, value in values.items():
        if not np.isfinite(value):
            raise RuntimeError(f"GBD moment-recovery diagnostic {name!r} is non-finite")
    for name in (
        "nonzero_node_count",
        "retained_node_count",
        "pruned_node_count",
        "support_augmented_node_count",
    ):
        if values[name] < 0.0:
            raise RuntimeError(f"GBD moment-recovery diagnostic {name!r} is negative")

    if values["pruned_node_count"] > 0.0 and not bool(recovery.get("applied", False)):
        raise RuntimeError("GBD pruned vortex nodes without conservative moment recovery")
    maximum_residual = max(
        values["normalized_vortex_strength_residual"],
        values["normalized_linear_impulse_residual"],
        values["normalized_angular_impulse_residual"],
    )
    if maximum_residual > 1.0e-5:
        raise RuntimeError("GBD moment recovery exceeds its precision-aware residual tolerance")
    if values["correction_fraction"] > float(correction_limit):
        raise RuntimeError(
            "GBD pruning required an excessive particle-strength correction: "
            f"fraction={values['correction_fraction']:.6e}, "
            f"limit={float(correction_limit):.6e}"
        )


def _world_rank() -> int:
    """Return the launcher rank for OpenMPI, MPICH/PMI, or MVAPICH."""
    for name in (
        "OMPI_COMM_WORLD_RANK",
        "PMI_RANK",
        "PMIX_RANK",
        "MV2_COMM_WORLD_RANK",
        "SLURM_PROCID",
    ):
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    if _mpi4py_comm is not None:
        return int(_mpi4py_comm.Get_rank())
    return 0


class FVMVPMCoupler:
    """Synchronize an inner FVM solution with an outer VPM wake.

    The driver samples the VPM field on an outer FVM patch, subcycles the FVM
    to the next VPM time, then replaces or blends the FVM-authoritative
    vorticity into the particle cloud. It owns the exchange sequence and
    coupled restart state; the injected solvers retain ownership of their
    respective meshes, fields, particles, and numerical models.

    Parameters
    ----------
    fvm_solver : FVMSolver
        Native incompressible FVM solver on this rank. In MPI execution every
        rank supplies its local solver and enters collective calls in the same
        order.
    vpm_solver : VPMSolver or None
        Particle solver on rank zero. Non-owner ranks pass ``None`` or an
        inactive-rank placeholder. The active solver's domain, freestream,
        viscosity, and time step must be compatible with the FVM case.
    coupler_setup : CouplerSetup
        Immutable coupling-owned transfer, boundary, diagnostic, and backup
        policy.

    Attributes
    ----------
    setup : CouplerSetup
        Retained coupling policy.
    fvm_solver : FVMSolver or None
        Adopted solver after :meth:`initialize`; ``None`` beforehand.
    vpm_solver : VPMSolver or None
        Rank-zero adopted particle solver after initialization. It remains
        ``None`` on other MPI ranks.
    vorticity_transfer : VorticityTransfer or None
        Prepared FVM-to-VPM transfer component.
    fvm_consistency_band : FVMConsistencyBand or None
        Optional resolved-scale consistency component.
    fvm_time_step_size, vpm_time_step_size : float or None
        Accepted FVM substep and VPM/coupling-step durations in s.
    n_fvm_substeps : int
        Exact integer number of FVM steps per VPM coupling step.
    end_time : float or None
        Physical FVM end time in s after initialization.
    kinematic_viscosity : float or None
        Shared fluid kinematic viscosity in m²/s.
    density : float or None
        FVM reference density in kg/m³.
    fvm_box : ndarray or None, shape (6,)
        Outer FVM bounds ``(xmin, xmax, ymin, ymax, zmin, zmax)`` in m.
    coupling_diagnostics : list[dict]
        Accepted-step diagnostic records. The list is reset when a run is
        prepared and mutated as coupling steps complete.

    Notes
    -----
    Construction does not advance either solver, but on rank zero it creates
    ``<fvm case>/solution`` and configures coupled logging. :meth:`run` is the
    normal entry point; it calls :meth:`initialize` idempotently. A coupled
    step is persistent state: VPM advance, VPM boundary trace, FVM substeps,
    absolute vorticity transfer, scheduled VPM output, diagnostics, and backup.

    Examples
    --------
    >>> setup = CouplerSetup(coupling_patch="numericalBoundary")
    >>> driver = FVMVPMCoupler(fvm_solver, vpm_solver, setup)
    >>> completed_step = driver.run(max_coupling_steps=2, backup_at_stop=True)
    """

    def __init__(
        self,
        fvm_solver: FVMSolver,
        vpm_solver: VPMSolver | None,
        coupler_setup: CouplerSetup,
    ) -> None:
        """Retain externally configured solvers and prepare coupled output.

        Parameters
        ----------
        fvm_solver : FVMSolver
            FVM instance required on every rank. The object is referenced, not
            copied, and its accepted state will be mutated during a run.
        vpm_solver : VPMSolver or None
            VPM instance required on rank zero and referenced directly. It is
            intentionally absent on other ranks.
        coupler_setup : CouplerSetup
            Validated coupling policy retained by reference.

        Raises
        ------
        ValueError
            If no FVM solver is supplied. Cross-solver constraints are deferred
            to :meth:`initialize`, when mesh and runtime data are available.

        Notes
        -----
        The constructor creates the coupled solution directory and log files
        on rank zero. It does not initialize exchange geometry or advance the
        accepted time of either solver.
        """
        if fvm_solver is None:
            raise ValueError(
                "FVMVPMCoupler requires an FVM solver. Use create_coupler with "
                "FVMSetup, VPMCase and CouplerSetup for automatic MPI ownership."
            )
        self.setup = coupler_setup
        self._owned_resources = ExitStack()
        self._closed = False
        self.case_dir = Path(fvm_solver.case_dir).expanduser().absolute()

        self._injected_fvm = fvm_solver
        self._injected_vpm = vpm_solver

        self._mpi_rank = _world_rank()
        self._is_master = self._mpi_rank == 0

        self.solution_dir = self.case_dir / "solution"
        self._log_handler = None
        if self._is_master:
            self.solution_dir.mkdir(parents=True, exist_ok=True)
            self._log_handler = configure_logging(self.solution_dir, logger)

        if self._is_master:
            self.vpm_redirector = OutputRedirector(
                logfile=str(self.solution_dir / "vpm.log"), append=True
            )
        else:
            self.vpm_redirector = OutputRedirector()  # no-op

        self.vpm_solver: VPMSolver | None = None
        self.fvm_solver: FVMSolver | None = None
        self.vorticity_transfer: VorticityTransfer | None = None
        self.fvm_consistency_band: FVMConsistencyBand | None = None
        self._velocity_boundary_condition_old: np.ndarray | None = None
        self._normal_velocity_boundary_condition_old: np.ndarray | None = None
        self._normal_velocity_boundary_condition: np.ndarray | None = None
        self._tangential_gradient_boundary_condition_old: np.ndarray | None = None
        self._tangential_gradient_boundary_condition: np.ndarray | None = None
        self._kinematic_pressure_gradient_boundary_condition_old: np.ndarray | None = None
        self._kinematic_pressure_gradient_boundary_condition: np.ndarray | None = None
        self._pressure_velocity_snapshot: np.ndarray | None = None
        self._velocity_global_buffer: np.ndarray | None = None
        self._velocity_gradient_global_buffer: np.ndarray | None = None
        self._last_vpm_boundary_condition_flux_diagnostics = {
            "raw_mismatch": 0.0,
            "raw_relative": 0.0,
            "acceptance_limit": 0.0,
            "applied_correction": 0.0,
            "corrected_mismatch": 0.0,
        }
        self._last_fvm_boundary_trace_diagnostics = {
            "mean_velocity_mismatch": 0.0,
            "maximum_velocity_mismatch": 0.0,
            "maximum_normal_velocity_mismatch": 0.0,
            "mean_outflow_velocity_mismatch": 0.0,
            "maximum_outflow_velocity_mismatch": 0.0,
        }
        self.coupling_diagnostics: list[dict] = []
        self._last_transfer_result = None

        self.fvm_time_step_size: float | None = None
        self.vpm_time_step_size: float | None = None
        self.end_time: float | None = None
        self.kinematic_viscosity: float | None = None
        self.density: float | None = None
        self.fvm_box: np.ndarray | None = None
        self.vpm_particle_spacing = float("nan")
        self.vpm_core_radius_ratio = float("nan")
        self.n_fvm_substeps = 1
        self.freestream_velocity = np.array(coupler_setup.freestream_velocity, dtype=np.float64)

    def close(self, *, failure: BaseException | None = None) -> None:
        """Close factory-owned solvers and this driver's log; safe to repeat.

        Externally supplied solvers remain caller-owned. Every MPI rank must
        close its driver because native FVM resources are collective.
        """
        if self._closed:
            return
        try:
            self._owned_resources.__exit__(
                type(failure) if failure is not None else None,
                failure,
                failure.__traceback__ if failure is not None else None,
            )
        finally:
            if self._log_handler is not None:
                logger.removeHandler(self._log_handler)
                self._log_handler.close()
            self._closed = True

    def __enter__(self):
        """Return this live driver for an automatically closed run."""
        if self._closed:
            raise RuntimeError("The coupled driver is closed")
        return self

    def __exit__(self, _exc_type, exc_value, _traceback):
        """Close resources on success or failure without suppressing errors."""
        self.close(failure=exc_value)

    @staticmethod
    def _validate_vpm(vpm, cfg: CouplerSetup, box: np.ndarray, kinematic_viscosity: float) -> None:
        """Validate the injected VPM against the coupling discretization."""
        vsc = vpm.setup.viscous
        if vsc.particle_spacing is None:
            raise ValueError(
                "Coupled VPM runs require viscous.particle_spacing so injected FVM "
                "cell particles and boundary derivatives use the VPM discretization."
            )
        dom = vpm.setup.domain_bounds
        if dom is not None:
            contains = (
                dom[0] <= box[0]
                and dom[1] >= box[1]
                and dom[2] <= box[2]
                and dom[3] >= box[3]
                and dom[4] <= box[4]
                and dom[5] >= box[5]
            )
            if not contains:
                raise ValueError(
                    f"Injected VPM domain {tuple(dom)} does not contain the FVM "
                    f"box {tuple(box)}. The near-body particles the coupler replaces "
                    "would be removed by the VPM's out-of-bounds cull every step. "
                    "Widen domain_bounds (or the VPM stabilization "
                    "remove_particles_by_bounds) to enclose fvm_box."
                )
        vpm_kinematic_viscosity = vsc.kinematic_viscosity
        if (
            vpm_kinematic_viscosity is not None
            and abs(float(vpm_kinematic_viscosity) - kinematic_viscosity) > 1e-12
        ):
            raise ValueError(
                f"Incompatible kinematic viscosity: VPM viscous.kinematic_viscosity="
                f"{float(vpm_kinematic_viscosity):g} but the Eulerian solver uses "
                f"kinematic_viscosity={float(kinematic_viscosity):g}. "
                "The two solvers must model the same fluid."
            )
        bg = np.asarray(vpm.freestream_velocity, dtype=np.float64)
        if not np.allclose(bg, np.asarray(cfg.freestream_velocity), atol=1e-9):
            raise ValueError(
                f"Incompatible freestream: VPM freestream_velocity {tuple(bg)} "
                f"!= coupling freestream_velocity {tuple(cfg.freestream_velocity)}. "
                "The VPM far-field and "
                "the VPM advection frame must agree."
            )

    @staticmethod
    def _derive_n_fvm_substeps(vpm_time_step_size: float, fvm_time_step_size: float) -> int:
        """Return the integer FVM sub-cycle count implied by solver time steps."""
        if fvm_time_step_size <= 0.0:
            raise ValueError(f"FVM time step must be positive, got {fvm_time_step_size!r}.")
        if vpm_time_step_size <= 0.0:
            raise ValueError(f"VPM time step must be positive, got {vpm_time_step_size!r}.")
        ratio = vpm_time_step_size / fvm_time_step_size
        n_fvm_substeps = max(1, int(round(ratio)))
        if not np.isclose(ratio, n_fvm_substeps, rtol=1e-9, atol=1e-12):
            raise ValueError(
                "The VPM time step must be an integer multiple of the FVM time "
                "step for sub-cycling. Got "
                f"vpm_time_step_size={vpm_time_step_size:.12g}, "
                f"fvm_time_step_size={fvm_time_step_size:.12g}, ratio={ratio:.12g}."
            )
        return n_fvm_substeps

    @staticmethod
    def _derive_coupling_step_count(end_time: float, vpm_time_step_size: float) -> int:
        """Return the number of VPM/coupling intervals for a given end time.

        The end time need not be an exact multiple of the VPM step size; the
        count is rounded to the nearest integer, landing on the closest
        coupling-step boundary.

        Args:
            end_time: Requested simulation end time.
            vpm_time_step_size: VPM (coupling) time-step size.

        Returns:
            Integer number of coupling steps.
        """
        if vpm_time_step_size <= 0.0:
            raise ValueError(f"VPM time step must be positive, got {vpm_time_step_size!r}.")
        if end_time < 0.0:
            raise ValueError(f"Coupling end time must be non-negative, got {end_time!r}.")
        return max(0, int(round(end_time / vpm_time_step_size)))

    def _derive_fvm_box(self) -> np.ndarray:
        """Bounds of the coupling patch, from the injected solver's geometry.

        The patch faces lie exactly on the six box planes, so the per-axis
        min/max of the face centres reproduce the box bounds to round-off.
        Collective (all ranks) — the face-geometry getter gathers globally.
        """
        assert self.fvm_solver is not None
        fc = np.asarray(
            self.fvm_solver.get_boundary_face_centre_coordinates(self.setup.coupling_patch),
            dtype=np.float64,
        ).reshape(-1, 3)
        box = None
        error = None
        collective = _mpi4py_comm is not None and _mpi4py_comm.Get_size() > 1
        if self._is_master or not collective:
            if fc.shape[0] == 0:
                error = (
                    f"Coupling patch {self.setup.coupling_patch!r} has no faces on the "
                    "injected Eulerian solver."
                )
            else:
                box = np.array(
                    [
                        fc[:, 0].min(),
                        fc[:, 0].max(),
                        fc[:, 1].min(),
                        fc[:, 1].max(),
                        fc[:, 2].min(),
                        fc[:, 2].max(),
                    ]
                )
        if collective:
            error, box = _mpi4py_comm.bcast((error, box) if self._is_master else None, root=0)
        if error is not None:
            raise ValueError(error)
        return np.asarray(box, dtype=np.float64)

    def _read_fvm_state(self) -> None:
        """Read fluid properties, time integration, and domain from the FVM."""
        assert self.fvm_solver is not None
        fvm_cfg = getattr(self.fvm_solver, "_resolved_setup", self.fvm_solver.setup)
        time_config = self.fvm_solver._time_config
        if time_config.adjustment is not None:
            raise ValueError(
                "FVM-VPM coupling currently requires a fixed FVM time step: "
                "adaptive FVM substeps would no longer partition each immutable "
                "VPM coupling interval exactly. Use MaximumCourantTimeStep only "
                "for standalone FVM and reference-flow runs."
            )
        self.fvm_time_step_size = float(time_config.time_step_size)
        self.end_time = float(time_config.end_time)
        self.kinematic_viscosity = float(
            getattr(self.fvm_solver, "_kinematic_viscosity", fvm_cfg.transport.kinematic_viscosity)
        )
        self.density = float(fvm_cfg.transport.density)
        self.fvm_box = self._derive_fvm_box()
        self.setup.validate_transfer_region_box(self.fvm_box)

    def initialize(self) -> None:
        """Validate both solvers and build immutable coupling geometry.

        The injected FVM solver's configuration owns the FVM step; the
        injected VPM solver's ``time_step_size`` configures the
        coupling/VPM step.  After both are known, the coupler derives
        ``n_fvm_substeps = round(vpm_time_step_size / fvm_time_step_size)``
        internally.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the master has no active VPM solver, the solver time steps do
            not form an integer subcycling ratio, freestream/viscosity/domain
            contracts disagree, adaptive FVM stepping is configured, or a
            configured transfer/consistency region is invalid.
        RuntimeError
            If MPI and FVM execution sizes disagree or a transfer component
            cannot be prepared from the available mesh/particle settings.

        Notes
        -----
        This method reads collective FVM geometry, constructs transfer
        lattices and boundary history, configures VPM diffusion anchors/body
        masks, and resets run diagnostics. It does not advance accepted time.
        It is idempotent: subsequent calls are no-ops once
        :attr:`vorticity_transfer` exists.
        """
        if self.vorticity_transfer is not None:
            return  # already initialized
        cfg = self.setup

        self.fvm_solver = self._injected_fvm

        world_size = 1
        if _mpi4py_comm is not None:
            world_size = int(_mpi4py_comm.Get_size())
        else:
            world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))
        if world_size > 1 and int(self.fvm_solver.n_procs()) == 1:
            raise RuntimeError(
                f"Launched under MPI (world size {world_size}) but the injected "
                "Eulerian solver is serial (n_procs() == 1). Configure a parallel "
                "solver or launch one process."
            )

        self._read_fvm_state()

        candidate_vpm = self._injected_vpm
        inactive_rank = bool(getattr(candidate_vpm, "_openonda_inactive_rank", False))
        self.vpm_solver = candidate_vpm if self._is_master and not inactive_rank else None
        with collective_phase(_mpi4py_comm, "VPM configuration validation"):
            if self._is_master:
                if self.vpm_solver is None:
                    raise ValueError(
                        "vpm_solver is None on the master rank. "
                        "Build the VPM on the master (the Coupler's internal VPM-owner rank)."
                    )
                assert self.fvm_box is not None and self.kinematic_viscosity is not None
                self._validate_vpm(self.vpm_solver, cfg, self.fvm_box, self.kinematic_viscosity)

        assert self.fvm_time_step_size is not None
        vpm_particle_spacing = self.fvm_time_step_size
        vpm_core_radius_ratio = 1.0
        if self._is_master:
            assert self.vpm_solver is not None
            viscous = self.vpm_solver.setup.viscous
            assert viscous.particle_spacing is not None
            vpm_particle_spacing = float(viscous.particle_spacing)
            vpm_core_radius_ratio = float(viscous.core_radius_ratio)
        if _mpi4py_comm is not None and _mpi4py_comm.Get_size() > 1:
            vpm_particle_spacing, vpm_core_radius_ratio = _mpi4py_comm.bcast(
                (vpm_particle_spacing, vpm_core_radius_ratio) if self._is_master else None,
                root=0,
            )
        self.vpm_particle_spacing = float(vpm_particle_spacing)
        self.vpm_core_radius_ratio = float(vpm_core_radius_ratio)

        vpm_time_step_size = self.fvm_time_step_size
        if self._is_master:
            assert self.vpm_solver is not None
            vpm_time_step_size = float(self.vpm_solver.time_step_size)
        if _mpi4py_comm is not None and _mpi4py_comm.Get_size() > 1:
            vpm_time_step_size = float(
                _mpi4py_comm.bcast(vpm_time_step_size if self._is_master else None, root=0)
            )

        self.vpm_time_step_size = vpm_time_step_size
        self.n_fvm_substeps = self._derive_n_fvm_substeps(
            self.vpm_time_step_size, self.fvm_time_step_size
        )
        if cfg.interface_iterations > 1:
            from .interface_iteration import validate_output_schedules

            validate_output_schedules(self)
        if self._is_master:
            logger.info(
                format_coupler_log(
                    "time stepping",
                    ("fvm time step", f"{self.fvm_time_step_size:.4e}", "s"),
                    ("vpm time step", f"{self.vpm_time_step_size:.4e}", "s"),
                    ("fvm substeps per coupling step", self.n_fvm_substeps),
                )
            )

        self.vorticity_transfer = VorticityTransfer(self)
        self.vorticity_transfer.setup(self.fvm_solver)
        if cfg.fvm_consistency_width > 0.0:
            assert self.fvm_box is not None
            self.fvm_consistency_band = FVMConsistencyBand(
                cfg,
                self.fvm_solver,
                coupling_time_step_size=self.vpm_time_step_size,
                fvm_box=self.fvm_box,
            )
        with collective_phase(_mpi4py_comm, "VPM grid configuration"):
            if self._is_master and self.vorticity_transfer._body_bounds is not None:
                assert self.vpm_solver is not None
                self.vpm_solver.physics.configure_body_box(self.vorticity_transfer._body_bounds)
                bounds = np.asarray(self.vorticity_transfer._body_bounds, dtype=np.float64)
                logger.info(
                    format_coupler_log(
                        "vpm diffusion grid",
                        ("solid mask", "box"),
                        ("bounds, x", f"[{bounds[0]:.6g}, {bounds[1]:.6g}]", "m"),
                        ("bounds, y", f"[{bounds[2]:.6g}, {bounds[3]:.6g}]", "m"),
                        ("bounds, z", f"[{bounds[4]:.6g}, {bounds[5]:.6g}]", "m"),
                    )
                )
            if self._is_master:
                anchor = self.vorticity_transfer._lattice_anchor
                if (
                    anchor is None
                    and self.vorticity_transfer._cell_centre is not None
                    and len(self.vorticity_transfer._cell_centre) > 0
                ):
                    anchor = self.vorticity_transfer._cell_centre[0]
                if anchor is not None:
                    assert self.vpm_solver is not None
                    self.vpm_solver.physics.configure_grid_lattice_anchor(
                        anchor, self.vpm_particle_spacing
                    )
                    anchor_text = ", ".join(
                        f"{value:.6g}" for value in np.asarray(anchor, dtype=np.float64)
                    )
                    logger.info(
                        format_coupler_log(
                            "vpm diffusion lattice",
                            ("anchor", f"[{anchor_text}]", "m"),
                            ("spacing", f"{self.vpm_particle_spacing:.6g}", "m"),
                        )
                    )

            if self._is_master:
                logger.info(
                    format_coupler_log(
                        "initial state",
                        ("start", "impulsive"),
                        ("particles", 0),
                    )
                )
                logger.info(format_coupler_log("initialization complete"))

        self._initialize_run_state()

    def apply_vpm(self, callback, *args, **kwargs):
        """Run application instrumentation once with the owned VPM solver.

        Callbacks may inspect the VPM state or install diagnostic hooks. The
        library selects its owner and propagates callback failures. Collective
        FVM field queries belong outside the callback. The result is shared.
        """
        result = None
        with collective_phase(_mpi4py_comm, "VPM application callback"):
            if self._is_master:
                vpm = self.vpm_solver if self.vpm_solver is not None else self._injected_vpm
                result = callback(vpm, *args, **kwargs)
        return _mpi4py_comm.bcast(result, root=0) if _mpi4py_comm is not None else result

    def run(
        self,
        start_step: int = 0,
        restart_from: str | Path | None = None,
        *,
        restart_allowed_config_differences: Collection[str] = (),
        max_coupling_steps: int | None = None,
        backup_at_stop: bool = False,
    ) -> int:
        """Initialize and run a complete or explicitly bounded coupling segment.

        Parameters
        ----------
        start_step : int, default=0
            Previously completed coupling step for an already-restored in-memory
            state. The FVM step must equal ``start_step * n_fvm_substeps`` and
            the master VPM step must equal ``start_step``.
        restart_from : str, pathlib.Path, or None, default=None
            Coupled-backup directory to restore before solving. It is mutually
            exclusive with a non-zero ``start_step``.
        restart_allowed_config_differences : collection[str], default=()
            Exact dotted configuration paths permitted to differ from the
            backup manifest. This is accepted only with ``restart_from``;
            artifact hashes and all unlisted settings remain strict.
        max_coupling_steps : int or None, default=None
            Positive cap on accepted coupling steps performed by this call.
            ``None`` advances to the configured physical end time.
        backup_at_stop : bool, default=False
            Write an atomic coupled backup if a bounded segment stops at a step
            not already covered by scheduled backup cadence.

        Returns
        -------
        int
            Final completed coupling-step index, including any restored prefix.

        Raises
        ------
        TypeError
            If a step limit/index is not an integer.
        ValueError
            If restart arguments conflict, step state is inconsistent, no
            steps remain under a requested limit, or solver configurations are
            incompatible.
        RuntimeError
            If initialization, stepping, transfer, output, or backup fails.

        Notes
        -----
        ``max_coupling_steps`` is an execution limit, not part of the physical
        configuration, so strict same-configuration restarts can continue a
        bounded run without changing its end time. Both injected solvers,
        boundary-history arrays, diagnostics, files, and clocks are mutated.
        """
        if self.vorticity_transfer is None:
            self.initialize()
        if restart_from is not None:
            if start_step:
                raise ValueError("start_step and restart_from are mutually exclusive")
            start_step = self.load_backup(
                restart_from,
                allowed_config_differences=restart_allowed_config_differences,
            )
        elif restart_allowed_config_differences:
            raise ValueError("restart_allowed_config_differences requires restart_from")
        if (
            start_step == 0
            and restart_from is None
            and getattr(self.fvm_solver, "auto_write", False)
        ):
            self.fvm_solver.write_vtk()
        return self.solve(
            start_step=start_step,
            max_coupling_steps=max_coupling_steps,
            backup_at_stop=backup_at_stop,
        )

    @staticmethod
    def _validate_step_limit(max_coupling_steps: int | None) -> int | None:
        if max_coupling_steps is None:
            return None
        if isinstance(max_coupling_steps, bool) or not isinstance(max_coupling_steps, Integral):
            raise TypeError("max_coupling_steps must be a positive integer or None")
        limit = int(max_coupling_steps)
        if limit <= 0:
            raise ValueError("max_coupling_steps must be positive")
        return limit

    def _validate_start_step(self, start_step: int, configured_end_step: int) -> int:
        if isinstance(start_step, bool) or not isinstance(start_step, Integral):
            raise TypeError("start_step must be a non-negative integer")
        step = int(start_step)
        if not 0 <= step <= configured_end_step:
            raise ValueError(f"start_step must lie in [0, {configured_end_step}], got {step}")
        assert self.fvm_solver is not None
        expected_fvm_step = step * self.n_fvm_substeps
        if int(self.fvm_solver.step) != expected_fvm_step:
            raise ValueError(
                "FVM state does not match start_step: "
                f"step={self.fvm_solver.step}, expected={expected_fvm_step}"
            )
        if self._is_master:
            assert self.vpm_solver is not None
            if int(self.vpm_solver.step) != step:
                raise ValueError(
                    "VPM state does not match start_step: "
                    f"step={self.vpm_solver.step}, expected={step}"
                )
        return step

    def solve(
        self,
        start_step: int = 0,
        *,
        max_coupling_steps: int | None = None,
        backup_at_stop: bool = False,
    ) -> int:
        """Advance an initialized coupled state through accepted macro-steps.

        Parameters
        ----------
        start_step : int, default=0
            Number of coupling steps already represented by both solver states.
            At zero, an initial FVM-to-VPM synchronization is performed before
            any time advance.
        max_coupling_steps : int or None, default=None
            Optional positive execution cap for this invocation.
        backup_at_stop : bool, default=False
            Persist a coupled backup at a bounded stop unless that step was
            already saved by the configured cadence.

        Returns
        -------
        int
            Last completed coupling-step index.

        Raises
        ------
        RuntimeError
            If :meth:`initialize` has not prepared the transfer component or a
            numerical/transfer/output operation fails.
        TypeError
            If either step argument has an invalid type.
        ValueError
            If the in-memory solver steps disagree with ``start_step`` or the
            execution cap is invalid.

        Notes
        -----
        Unlike :meth:`run`, this method does not initialize or load a restart.
        Each iteration mutates both accepted solver states and their clocks,
        updates boundary history and transfer diagnostics, executes due VPM
        samplers, and may write logs/backups. MPI ranks must call it
        collectively and in the same order.
        """
        face_geometry, n_steps = self._prepare_run()
        with collective_phase(_mpi4py_comm, "coupled start state"):
            start_step = self._validate_start_step(start_step, n_steps)
        step_limit = self._validate_step_limit(max_coupling_steps)
        if step_limit is not None and start_step == n_steps:
            raise ValueError("No configured coupling steps remain after start_step")
        stop_step = n_steps if step_limit is None else min(n_steps, start_step + step_limit)
        self._n_steps = stop_step
        with collective_phase(_mpi4py_comm, "coupled run metadata"):
            if self._is_master:
                write_run_metadata(self, start_step=start_step, stop_step=stop_step)
                if stop_step < n_steps:
                    logger.info(
                        format_coupler_log(
                            "execution limit",
                            ("start step", f"{start_step:,}"),
                            ("stop step", f"{stop_step:,}"),
                            ("steps this invocation", f"{stop_step - start_step:,}"),
                            ("backup at stop", "enabled" if backup_at_stop else "disabled"),
                        )
                    )
        if start_step == 0:
            # Every VPM interval must start from the FVM state at the same
            # physical time. This first synchronization also supports non-zero
            # user-supplied initial FVM vorticity while preserving any outer
            # particles already present in the VPM cloud.
            initial_result, _ = self._transfer_vorticity_to_vpm(*face_geometry)
            self._last_transfer_result = initial_result
        initialize_vpm_boundary_history(self, *face_geometry)
        assert self.vpm_time_step_size is not None
        for step in range(1 + start_step, stop_step + 1):
            time_end = step * self.vpm_time_step_size
            vpm_time = self._advance_vpm(step, time_end)
            velocity_boundary_condition_old, next_velocity, boundary_time = evaluate_vpm_boundary(
                self, *face_geometry
            )
            if self.setup.interface_iterations > 1:
                from .interface_iteration import advance_iterated_interface

                transfer_result, fvm_time, transfer_time = advance_iterated_interface(
                    self, face_geometry, next_velocity
                )
            else:
                fvm_time = advance_fvm(
                    self, *face_geometry, velocity_boundary_condition_old, next_velocity
                )
                transfer_result, transfer_time = self._transfer_vorticity_to_vpm(*face_geometry)
                update_boundary_history_after_replacement(self, *face_geometry)
            with collective_phase(_mpi4py_comm, "VPM health check and output"):
                if self._is_master:
                    assert self.vpm_solver is not None
                    self.vpm_solver.execute_scheduled_samplers()
            self._last_transfer_result = transfer_result
            record_step(
                self,
                step,
                time_end,
                (vpm_time, boundary_time, fvm_time, transfer_time),
                transfer_result,
                logger=logger,
                comm=_mpi4py_comm,
            )
        backup_was_scheduled = (
            self.setup.backup_interval_steps > 0
            and stop_step > start_step
            and stop_step % self.setup.backup_interval_steps == 0
        )
        if backup_at_stop and stop_step > start_step and not backup_was_scheduled:
            self.save_backup(
                self.solution_dir / BACKUP_DIRECTORY,
                coupling_step=stop_step,
            )
        if self._is_master:
            flush_log(logger)
        return stop_step

    def _initialize_run_state(self) -> None:
        self._step_transfer_stats: dict[str, float | int] | None = None
        self.coupling_diagnostics = []

    def _prepare_run(self):
        """Validate a run and collect the immutable interface geometry."""
        if self.vorticity_transfer is None:
            raise RuntimeError(
                "solve() called before initialize(); call coupler.initialize() "
                "first, or use coupler.run() which does both."
            )
        assert self._is_master == (self.vpm_solver is not None)
        assert self.fvm_solver is not None

        assert self.end_time is not None and self.vpm_time_step_size is not None
        n_steps = self._derive_coupling_step_count(self.end_time, self.vpm_time_step_size)
        patch = self.setup.coupling_patch
        if self._is_master:
            logger.info(
                format_coupler_log(
                    "run",
                    ("coupling steps", f"{n_steps:,}"),
                    ("end time", f"{self.end_time:.6g}", "s"),
                    ("boundary mode", self.setup.boundary_condition_mode),
                    ("coupling patch", patch),
                )
            )

        face_centre = np.asarray(
            self.fvm_solver.get_boundary_face_centre_coordinates(patch), dtype=np.float64
        ).reshape(-1, 3)
        face_normal = np.asarray(
            self.fvm_solver.get_boundary_face_normal(patch), dtype=np.float64
        ).reshape(-1, 3)
        face_area = np.asarray(
            self.fvm_solver.get_boundary_face_area(patch), dtype=np.float64
        ).ravel()
        self._n_steps = n_steps
        return (face_centre, face_normal, face_area), n_steps

    def _advance_vpm(self, step: int, time_end: float) -> float:
        t0 = time.perf_counter()
        with collective_phase(_mpi4py_comm, "VPM advance"):
            if self._is_master:
                assert self.vpm_solver is not None
                with self.vpm_redirector:
                    self.vpm_solver._set_freestream_velocity(self.setup.freestream_velocity)
                logger.info(format_coupler_step(step, self._n_steps, time_end))

                with self.vpm_redirector:
                    self.vpm_solver.advance(defer_output=True)
                self.vpm_solver.synchronize()
                if str(getattr(self.vpm_solver, "viscous_scheme", "")).upper() == "GBD":
                    _validate_gbd_moment_recovery(
                        getattr(
                            self.vpm_solver.physics,
                            "last_gbd_moment_recovery",
                            None,
                        ),
                        self.setup.transfer_discretization_error_limit,
                    )
        return time.perf_counter() - t0

    def _transfer_vorticity_to_vpm(
        self,
        face_centre: np.ndarray,
        _face_normals: np.ndarray,
        _face_area: np.ndarray,
    ):
        """Replace the FVM-authoritative part of the particle cloud."""
        t_transfer = time.perf_counter()
        velocity_global = self._get_velocity_field_buffer()
        gradient_global = self._get_velocity_gradient_field_buffer()
        transfer_result = None
        with collective_phase(_mpi4py_comm, "vorticity transfer"):
            if self._is_master:
                assert self.vpm_solver is not None
                assert self.vorticity_transfer is not None
                vpm = self.vpm_solver
                transfer = self.vorticity_transfer
                n_before = vpm.particles.n_particles_total
                sum_before = (
                    float(np.sum(np.linalg.norm(np.asarray(vpm.particle_vortex_strength), axis=1)))
                    if n_before > 0
                    else 0.0
                )
                transfer_result = transfer.transfer(
                    vpm,
                    velocity=velocity_global,
                    velocity_gradient=gradient_global,
                )
                n_after = vpm.particles.n_particles_total
                sum_after = (
                    float(np.sum(np.linalg.norm(np.asarray(vpm.particle_vortex_strength), axis=1)))
                    if n_after > 0
                    else 0.0
                )
                self._step_transfer_stats = {
                    "n_before": n_before,
                    "n_after": n_after,
                    "sum_before": sum_before,
                    "sum_after": sum_after,
                    "face_count": len(face_centre),
                }
        return transfer_result, time.perf_counter() - t_transfer

    def _get_velocity_field_buffer(self) -> np.ndarray:
        assert self.fvm_solver is not None
        if self._velocity_global_buffer is None:
            self._velocity_global_buffer = np.ascontiguousarray(
                self.fvm_solver.get_velocity_field(), dtype=np.float64
            ).reshape(-1, 3)
        else:
            self.fvm_solver.get_velocity_field_into(self._velocity_global_buffer)
        return self._velocity_global_buffer

    def _get_velocity_gradient_field_buffer(self) -> np.ndarray:
        assert self.fvm_solver is not None
        if self._velocity_gradient_global_buffer is None:
            self._velocity_gradient_global_buffer = np.ascontiguousarray(
                self.fvm_solver.get_velocity_gradient_field(), dtype=np.float64
            ).reshape(-1, 3, 3)
        else:
            self.fvm_solver.get_velocity_gradient_field_into(self._velocity_gradient_global_buffer)
        return self._velocity_gradient_global_buffer

    def save_backup(
        self,
        directory: str | Path,
        *,
        coupling_step: int | None = None,
    ) -> Path:
        """Atomically write both solvers and the coupling boundary history.

        Parameters
        ----------
        directory : str or pathlib.Path
            Destination directory. It is created if needed. A rolling manifest
            and referenced FVM, VPM, XDMF, and boundary-history artifacts are
            stored below it.
        coupling_step : int or None, default=None
            Step label for the checkpoint. ``None`` derives it from the FVM
            accepted step and :attr:`n_fvm_substeps`.

        Returns
        -------
        pathlib.Path
            Backup directory. On rank zero the committed VPM HDF5/XDMF pair is
            also copied into :attr:`solution_dir` as a retained output frame.

        Raises
        ------
        RuntimeError
            If the coupler is not initialized on a rank that owns a required
            solver.
        OSError
            If an artifact cannot be written, synchronized, or committed.

        Notes
        -----
        The FVM save is collective in partitioned execution. The manifest is
        committed last so a visible manifest denotes a complete checkpoint.
        This method writes files but does not change physical time.
        """
        backup = save_coupled_backup(self, directory, coupling_step=coupling_step)
        if self._is_master:
            publish_vpm_snapshot(backup, self.solution_dir)
        return backup

    def load_backup(
        self,
        directory: str | Path,
        *,
        allowed_config_differences: Collection[str] = (),
    ) -> int:
        """Restore a complete coupled checkpoint into initialized solvers.

        Parameters
        ----------
        directory : str or pathlib.Path
            Directory containing the committed coupled ``manifest.json`` and
            all artifacts named by it.
        allowed_config_differences : collection[str], default=()
            Exact recursive configuration paths permitted to differ for a
            controlled restart. All other configuration and artifact hashes
            remain strict.

        Returns
        -------
        int
            Restored coupling-step index, suitable as ``start_step`` for
            :meth:`solve`.

        Raises
        ------
        RuntimeError
            If the coupler is not initialized, the manifest/artifacts are
            incomplete or inconsistent, or restored solver clocks disagree.
        ValueError
            If configuration differs outside the explicit allow-list.
        OSError
            If required files cannot be read.

        Notes
        -----
        This method mutates FVM fields/history, the VPM particle state and
        clock, and all stored VPM boundary-condition history. In partitioned
        execution every rank must participate collectively.
        """
        return load_coupled_backup(
            self,
            directory,
            comm=_mpi4py_comm,
            allowed_config_differences=allowed_config_differences,
        )
