"""Coupled FVM--VPM solver: VPM boundary traces drive the FVM, FVM vorticity renews the particles."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from contextlib import ExitStack
from dataclasses import replace
import logging
import math
from pathlib import Path
import time
from typing import TYPE_CHECKING

import numpy as np

from openonda.core.metadata import write_run_metadata
from openonda.core.restart import reset_run_outputs, select_backup
from openonda.core.schedules import EveryTime
from openonda.core.simulation.parallel import collective_phase
from openonda.core.status import RunStatus
from openonda.runtime import detected_world_size

from .backup import coupled_backup_path, load_coupled_backup, save_coupled_backup
from .boundary import (
    advance_fvm,
    evaluate_vpm_boundary,
    initialize_vpm_boundary_history,
    update_boundary_history_after_replacement,
)
from .config import CouplerConfig
from .reporting import (
    begin_coupling_step,
    close_coupler_logging,
    configure_coupler_logging,
    flush_log,
    format_coupler_log,
    record_step,
)
from .vorticity_transfer import VorticityTransfer

if TYPE_CHECKING:
    from openonda.fvm import FVMCase, FVMSolver
    from openonda.vpm import VPMCase, VPMSolver

logger = logging.getLogger("openonda.coupler")


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


def _n_fvm_substeps(vpm_time_step_size: float, fvm_time_step_size: float) -> int:
    """Return the whole number of FVM steps in one exchange."""
    if fvm_time_step_size <= 0.0 or vpm_time_step_size <= 0.0:
        raise ValueError("The FVM and VPM time-step sizes must be positive")
    ratio = vpm_time_step_size / fvm_time_step_size
    n_substeps = max(1, round(ratio))
    if not math.isclose(ratio, n_substeps, rel_tol=1.0e-9, abs_tol=1.0e-12):
        raise ValueError(
            "The VPM time-step size must be a whole multiple of the FVM time-step size; got "
            f"{vpm_time_step_size:.12g} s and {fvm_time_step_size:.12g} s (ratio {ratio:.12g})"
        )
    return n_substeps


def _validate_cases(fvm_case: FVMCase, vpm_case: VPMCase, config: CouplerConfig) -> None:
    """Check the case settings the two solvers and the coupler must share."""
    from openonda.fvm import FVMCase
    from openonda.vpm import VPMCase

    if not isinstance(fvm_case, FVMCase):
        raise TypeError("CoupledSolver requires an FVMCase")
    if not isinstance(vpm_case, VPMCase):
        raise TypeError("CoupledSolver requires a VPMCase")
    if not isinstance(config, CouplerConfig):
        raise TypeError("CoupledSolver requires a CouplerConfig")
    vpm_solution = Path(vpm_case.directory) / "solution" / (vpm_case.variant or "")
    if fvm_case.solution_directory.resolve() != vpm_solution.resolve():
        raise ValueError("The FVM and VPM cases must share their directory and variant")
    if fvm_case.backup.schedule is not None or vpm_case.backup.schedule is not None:
        raise ValueError(
            "Coupled backups are set by CouplerConfig.backup; "
            "the FVM and VPM cases take no backup schedule"
        )
    if fvm_case.numerics.time_step_control is not None:
        raise ValueError(
            "Coupling requires a fixed FVM time-step size: adaptive FVM steps would not "
            "partition each exchange exactly"
        )
    if fvm_case.run.start_time != 0.0:
        raise ValueError("Coupled runs start at time zero")
    if not math.isclose(fvm_case.run.end_time, vpm_case.run.end_time, rel_tol=1.0e-12):
        raise ValueError(
            f"The FVM end time {fvm_case.run.end_time!r} s differs from the VPM end time "
            f"{vpm_case.run.end_time!r} s"
        )
    numerics = vpm_case.numerics
    if numerics.viscous.particle_spacing is None:
        raise ValueError(
            "Coupled VPM runs require viscous.particle_spacing so the transferred FVM "
            "vorticity and the boundary derivatives use the VPM discretization"
        )
    viscosity = numerics.viscous.kinematic_viscosity
    fvm_viscosity = fvm_case.numerics.transport.kinematic_viscosity
    if viscosity is not None and abs(float(viscosity) - fvm_viscosity) > 1.0e-12:
        raise ValueError(
            f"The VPM kinematic viscosity {float(viscosity):g} m^2/s differs from the FVM "
            f"kinematic viscosity {fvm_viscosity:g} m^2/s"
        )
    if not np.allclose(numerics.freestream_velocity, config.freestream_velocity, atol=1.0e-9):
        raise ValueError(
            f"The VPM freestream velocity {tuple(numerics.freestream_velocity)} differs from "
            f"the coupling freestream velocity {tuple(config.freestream_velocity)}"
        )
    exchange_time_step_size = numerics.time_step_size
    n_substeps = _n_fvm_substeps(exchange_time_step_size, fvm_case.numerics.time_step_size)
    schedule = config.backup.schedule
    if isinstance(schedule, EveryTime):
        for name in ("interval", "start_time"):
            ratio = getattr(schedule, name) / exchange_time_step_size
            if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1.0e-9):
                raise ValueError(
                    f"The coupled backup {name} must be a whole number of exchange time steps"
                )
    if config.interface_iterations > 1:
        from .interface_iteration import validate_output_schedules

        validate_output_schedules(fvm_case, n_substeps, exchange_time_step_size)


def _validate_vpm_domain(vpm_case: VPMCase, config: CouplerConfig, box: np.ndarray) -> None:
    """Check the particle domain, slip planes and fringes against the FVM box."""
    numerics = vpm_case.numerics
    induction = numerics.induction
    if getattr(induction, "method", None) == "slip_slab":
        # The FVM extrusion, transfer region and VPM slab share one span.
        planes = np.array([induction.z_min, induction.z_max], dtype=np.float64)
        tolerance = 1.0e-9 * max(1.0, float(np.max(np.abs(planes))))
        transfer = box if config.transfer_region_bounds is None else config.transfer_region_bounds
        for name, bounds in (("FVM box", box), ("transfer_region_bounds", transfer)):
            z_bounds = np.asarray(bounds, dtype=np.float64)[4:6]
            if np.any(np.abs(z_bounds - planes) > tolerance):
                raise ValueError(
                    f"{name} z {tuple(z_bounds.tolist())} must equal the VPM slip "
                    f"planes {tuple(planes.tolist())}"
                )
    domain = numerics.domain_bounds
    if domain is not None:
        # Projected wall vertices can move an outer face by a few picometres;
        # the VPM domain is specified at the intended plane.
        tolerance = 1.0e-9 * max(1.0, float(np.max(np.abs(box))))
        lower = np.asarray(domain[::2]) <= box[::2] + tolerance
        upper = np.asarray(domain[1::2]) >= box[1::2] - tolerance
        if not (np.all(lower) and np.all(upper)):
            raise ValueError(
                f"The VPM domain {tuple(domain)} does not contain the FVM box {tuple(box)}: "
                "the particles the coupler renews would be removed every step"
            )
    stabilization = numerics.stabilization
    fringe = getattr(stabilization, "outflow_fringe_depth", None)
    if fringe is not None:
        depth = np.asarray(fringe, dtype=np.float64)
        keep = np.asarray(stabilization.remove_particles_by_bounds, dtype=np.float64)
        inner = keep + depth * np.array([1.0, -1.0] * 3)
        lower_overlap = (depth[::2] > 0.0) & (inner[::2] > box[::2])
        upper_overlap = (depth[1::2] > 0.0) & (inner[1::2] < box[1::2])
        if np.any(lower_overlap) or np.any(upper_overlap):
            raise ValueError(
                f"The VPM outflow fringes {tuple(fringe)} reach the FVM box {tuple(box)}; "
                "place each fringe outside the FVM box"
            )


class CoupledSolver:
    """Inner FVM solution coupled to an outer VPM wake.

    At each exchange the particles advance one step, their velocity and
    tangential velocity gradient on the outer FVM patch drive the FVM
    substeps to the same time, and the FVM vorticity replaces the particles
    inside the transfer region. The coupler owns the exchange clock: both
    solvers accept exactly the time :meth:`exchange_time` of each exchange.

    Parameters
    ----------
    fvm_case : FVMCase
        Inner flow, mesh and FVM output. Under MPI every rank constructs the
        coupled solver; the FVM runs on all ranks.
    vpm_case : VPMCase
        Particle wake and its output; the VPM runs on the root rank. Its
        time-step size is the exchange time-step size, a whole multiple of
        the FVM one, and its directory and variant are those of the FVM case.
    config : CouplerConfig
        Overlap transfer, interface iteration and coupled backups.

    Attributes
    ----------
    config : CouplerConfig
        Coupling settings.
    fvm_case, vpm_case : FVMCase, VPMCase
        Cases of the two solvers.
    fvm_solver : FVMSolver
        FVM solver of this rank.
    vpm_solver : VPMSolver or None
        VPM solver on the root rank, ``None`` elsewhere.
    solution_dir : pathlib.Path
        ``solution/`` of the case, below its variant.
    fvm_time_step_size, vpm_time_step_size : float
        FVM substep and exchange time-step sizes in s.
    n_fvm_substeps : int
        FVM steps per exchange.
    coupling_diagnostics : list[dict]
        Diagnostics of the exchanges of this process, also appended to
        ``solution/coupler_diagnostics.jsonl``.

    Notes
    -----
    Coupled backups (``solution/coupled/coupled_{step:06d}/``) replace the
    backups of the two solvers: the cases take no backup schedule and the
    solvers write no backup at their end. Every coupled backup also writes
    the FVM and particle frames of its time. Whenever the VPM writes a
    particle frame (``VPMCase.output``), the FVM writes a frame at the same
    time. The solver is a context manager.

    Examples
    --------
    >>> with CoupledSolver(fvm_case, vpm_case, CouplerConfig()) as solver:
    ...     solver.run(start_from="latest")
    """

    def __init__(self, fvm_case: FVMCase, vpm_case: VPMCase, config: CouplerConfig) -> None:
        """Construct both solvers, validate them together and build the transfer geometry."""
        from openonda.fvm import Backup as FVMBackup
        from openonda.fvm import FVMSolver
        from openonda.vpm import Backup as VPMBackup
        from openonda.vpm import VPMSolver

        _validate_cases(fvm_case, vpm_case, config)
        self.config = config
        self.fvm_case = fvm_case
        self.vpm_case = vpm_case
        self.case_dir = Path(fvm_case.directory).resolve()
        self.solution_dir = fvm_case.solution_directory.resolve()
        self.run_status = RunStatus.not_started
        self.run_failure: BaseException | None = None
        self.interface_trace_correction: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self.coupling_diagnostics: list[dict] = []
        self._closed = False
        self._run_started = False
        self._started = False
        self._restart_loaded = False
        self._frame_step: int | None = None
        self._backup_step: int | None = None
        self._resources = ExitStack()
        self.vpm_solver: VPMSolver | None = None
        self._is_root = True
        try:
            self.fvm_solver: FVMSolver = self._resources.enter_context(
                FVMSolver(replace(fvm_case, backup=FVMBackup(write_at_end=False)))
            )
            self._comm = self.fvm_solver.parallel.comm
            self._is_root = bool(self.fvm_solver.parallel.is_root)
            if self._is_root:
                configure_coupler_logging(self.solution_dir / "coupler.log", console=True)
            with collective_phase(self._comm, "VPM construction"):
                if self._is_root:
                    self.vpm_solver = self._resources.enter_context(
                        VPMSolver(replace(vpm_case, backup=VPMBackup(write_at_end=False)))
                    )
            self._initialize()
            self.run_status = RunStatus.created
            self.write_run_metadata(status=self.run_status)
        except BaseException as error:
            self._resources.__exit__(type(error), error, error.__traceback__)
            if self._is_root:
                close_coupler_logging()
            raise

    def _initialize(self) -> None:
        """Read the shared discretization and build the transfer geometry (collective)."""
        fvm = self.fvm_solver
        world_size = int(self._comm.Get_size()) if self._comm is not None else detected_world_size()
        if world_size > 1 and fvm.n_ranks == 1:
            raise RuntimeError(
                f"Launched under MPI with {world_size} processes but the FVM solver is serial; "
                "set FVMCase.cores or launch one process"
            )
        numerics = self.vpm_case.numerics
        viscous = numerics.viscous
        self.fvm_time_step_size = float(self.fvm_case.numerics.time_step_size)
        self.vpm_time_step_size = float(numerics.time_step_size)
        self.n_fvm_substeps = _n_fvm_substeps(self.vpm_time_step_size, self.fvm_time_step_size)
        self.end_time = float(self.fvm_case.run.end_time)
        self.kinematic_viscosity = float(self.fvm_case.numerics.transport.kinematic_viscosity)
        self.density = float(self.fvm_case.numerics.transport.density)
        self.vpm_particle_spacing = float(viscous.particle_spacing)
        self.vpm_core_radius_ratio = float(viscous.core_radius_ratio)
        grid_spacing = {"gbd": viscous.gbd_grid_spacing, "dvh": viscous.dvh_grid_spacing}.get(
            viscous.scheme
        )
        self.vpm_diffusion_grid_spacing = float(
            self.vpm_particle_spacing if grid_spacing is None else grid_spacing
        )
        self.freestream_velocity = self.config.freestream_velocity_vector
        self._velocity_boundary_condition_old: np.ndarray | None = None
        self._normal_velocity_boundary_condition_old: np.ndarray | None = None
        self._normal_velocity_boundary_condition: np.ndarray | None = None
        self._tangential_gradient_boundary_condition_old: np.ndarray | None = None
        self._tangential_gradient_boundary_condition: np.ndarray | None = None
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
        self._last_transfer_result = None
        self._step_transfer_stats: dict[str, float | int] | None = None

        self.fvm_box = self._gather_fvm_box()
        self.config.validate_transfer_region_box(self.fvm_box)
        _validate_vpm_domain(self.vpm_case, self.config, self.fvm_box)
        if self._is_root:
            backup = self.config.backup.schedule
            logger.info(
                format_coupler_log(
                    "time stepping",
                    ("fvm time step", f"{self.fvm_time_step_size:.4e}", "s"),
                    ("vpm time step", f"{self.vpm_time_step_size:.4e}", "s"),
                    ("fvm substeps per coupling step", self.n_fvm_substeps),
                    ("exchanges", f"{self.n_steps:,}"),
                    ("end time", f"{self.end_time:.6g}", "s"),
                    ("coupled backups", "none" if backup is None else backup.describe()),
                )
            )

        self.vorticity_transfer = VorticityTransfer(self)
        self.vorticity_transfer.setup(fvm)
        with collective_phase(self._comm, "VPM grid configuration"):
            if self._is_root:
                self._configure_vpm_geometry()

    def _gather_fvm_box(self) -> np.ndarray:
        """Return the outer FVM bounds from its non-wall patches (collective)."""
        fvm = self.fvm_solver
        patch = self.config.coupling_patch
        faces = [np.asarray(fvm.gather_boundary_face_centre_coordinates(patch)).reshape(-1, 3)]
        for boundary in self.fvm_case.boundaries:
            if boundary.name != patch and boundary.mesh_type != "wall":
                faces.append(
                    np.asarray(fvm.gather_boundary_face_centre_coordinates(boundary.name)).reshape(
                        -1, 3
                    )
                )
        box = None
        with collective_phase(self._comm, "FVM box"):
            if self._is_root:
                if len(faces[0]) == 0:
                    raise ValueError(f"The coupling patch {patch!r} has no faces")
                points = np.concatenate(faces)
                box = np.ravel(np.column_stack((points.min(axis=0), points.max(axis=0))))
        if self._comm is not None and self._comm.Get_size() > 1:
            box = self._comm.bcast(box, root=0)
        return np.asarray(box, dtype=np.float64)

    def _configure_vpm_geometry(self) -> None:
        """Give the VPM the FVM wall geometry and the transfer lattice phase."""
        vpm = self.vpm_solver
        transfer = self.vorticity_transfer
        boundary = transfer.solid_boundary
        if boundary is not None:
            from .solid import SolidParticleGuard

            vpm.physics.configure_body_classifier(
                boundary.contains,
                revision=boundary.revision,
                query_bounds=boundary.bounds,
                blocks_segments=boundary.blocks_segments,
                geometry_queries=boundary.grid_geometry_queries(),
            )
            guard = SolidParticleGuard(boundary, vpm.physics, self.vpm_particle_spacing, logger)
            vpm.stage_rhs.position_guard = guard.stage
            vpm.stage_rhs.accepted_position_projector = guard.accepted
            logger.info(
                format_coupler_log(
                    "vpm diffusion grid",
                    ("solid mask", "FVM wall geometry"),
                    ("geometry revision", repr(boundary.revision)),
                )
            )
        anchor = transfer._lattice_anchor
        if anchor is not None:
            vpm.physics.configure_grid_lattice_anchor(anchor, self.vpm_diffusion_grid_spacing)
            anchor_text = ", ".join(f"{value:.6g}" for value in np.asarray(anchor))
            logger.info(
                format_coupler_log(
                    "vpm diffusion lattice",
                    ("anchor", f"[{anchor_text}]", "m"),
                    ("spacing", f"{self.vpm_diffusion_grid_spacing:.6g}", "m"),
                )
            )

    # Clock

    @property
    def n_steps(self) -> int:
        """Number of exchanges from time zero to ``RunPlan.end_time``."""
        return self.vpm_case.n_steps

    @property
    def step(self) -> int:
        """Number of completed exchanges."""
        return self.fvm_solver.step // self.n_fvm_substeps

    @property
    def time(self) -> float:
        """Accepted physical time in s."""
        return self.fvm_solver.time

    def exchange_time(self, step: int) -> float:
        """Return the physical time of exchange ``step``, in s.

        Both solvers accept exactly this time at every exchange, so their
        outputs carry identical time values; summing time-step sizes would
        drift by round-off between the two solvers.
        """
        return step * self.vpm_time_step_size

    # Lifecycle

    def __enter__(self) -> CoupledSolver:
        return self

    def __exit__(self, _exc_type, exc_value, _traceback) -> None:
        self.close(failure=exc_value)

    def start_from(self, selection: str | Path = "latest") -> bool:
        """Select the initial state or a coupled backup before the first exchange.

        Parameters
        ----------
        selection : {'latest', 'initial'} or path, default='latest'
            ``latest`` loads the coupled backup with the highest step and
            starts from the initial state when there is none; ``initial``
            deletes the earlier output series and starts from the initial
            state; a path loads that coupled backup.

        Returns
        -------
        bool
            Whether a backup was loaded.
        """
        if selection == "initial":
            if self.step or self._restart_loaded or self._started:
                raise RuntimeError("start_from('initial') requires a new coupled solver")
            with collective_phase(self._comm, "coupled output reset"):
                if self._is_root:
                    reset_run_outputs(self.solution_dir, kind="coupled")
            self.fvm_solver.start_from("initial")
            with collective_phase(self._comm, "VPM output reset"):
                if self._is_root:
                    self.vpm_solver.start_from("initial")
            return False
        selected = None
        with collective_phase(self._comm, "coupled backup selection"):
            if self._is_root:
                selected = select_backup(selection, directory=self.solution_dir, kind="coupled")
        if self._comm is not None and self._comm.Get_size() > 1:
            selected = self._comm.bcast(selected, root=0)
        if selected is None:
            return self.start_from("initial")
        self.load_backup(selected)
        if self._is_root:
            logger.info(f"Continuing from {selected}: exchange {self.step}, time {self.time:.17g}")
        return True

    def run(self, *, start_from: str | Path | None = None) -> None:
        """Run the coupled case to ``RunPlan.end_time`` and close the solver.

        Parameters
        ----------
        start_from : {'latest', 'initial'}, path or None, default=None
            Passed to :meth:`start_from` before the run; ``None`` continues
            from the state in memory.

        Notes
        -----
        A fresh run writes the initial frames and samples of both solvers
        with their ``write_initial_output``; a restarted run does not repeat
        them. The run then advances the exchanges, writes the scheduled output
        and backups, a final coupled backup with ``Backup.write_at_end``, and
        the final frames and final-only samples with ``write_final_output``.
        The run status is recorded in ``solution/coupler_metadata.json``.
        Every MPI rank calls this method.
        """
        if self._run_started:
            raise RuntimeError("CoupledSolver.run() may be called only once")
        self._run_started = True
        self.run_status = RunStatus.running
        failure: BaseException | None = None
        try:
            if start_from is not None:
                self.start_from(start_from)
            restored = self._restart_loaded
            self._start()
            if not restored:
                self._write_initial_output()
                if start_from is not None:
                    self.save_backup()
            self.write_run_metadata(status=self.run_status)
            while self.step < self.n_steps:
                self.advance()
            self._write_final_output()
            self.run_status = RunStatus.completed
        except BaseException as error:
            failure = error
            self.run_status = RunStatus.failed
            self.run_failure = error
        try:
            self.write_run_metadata(status=self.run_status)
        except BaseException as error:
            if failure is None:
                failure = error
                self.run_status = RunStatus.failed
        self.close(failure=failure)
        if failure is not None:
            raise failure.with_traceback(failure.__traceback__)

    def advance(self) -> None:
        """Advance one exchange and write its scheduled output and backup.

        The VPM advances to the next exchange time, its boundary trace drives
        the FVM substeps, the FVM vorticity renews the overlap particles, and
        the due samples, frames and coupled backup are written. Every MPI
        rank calls this method.
        """
        if self._closed:
            raise RuntimeError("The coupled solver is closed")
        self._start()
        step = self.step + 1
        if step > self.n_steps:
            raise RuntimeError("The coupled run has reached its end time")
        if not self._run_started:
            self.run_status = RunStatus.running
        face_geometry = self._face_geometry
        exchange_started = time.perf_counter()
        time_end = self.exchange_time(step)
        with collective_phase(self._comm, "freestream at the exchange end"):
            self._update_freestream(time_end)
        vpm_seconds = self._advance_vpm(step, time_end)
        velocity_old, next_velocity, boundary_seconds = evaluate_vpm_boundary(self, *face_geometry)
        if self.config.interface_iterations > 1:
            from .interface_iteration import advance_iterated_interface

            transfer_result, fvm_seconds, transfer_seconds = advance_iterated_interface(
                self, face_geometry, next_velocity
            )
            boundary_seconds += self._last_interface_iteration_diagnostics["phase_seconds"][
                "boundary_refresh"
            ]
        else:
            fvm_seconds = advance_fvm(self, *face_geometry, velocity_old, next_velocity)
            transfer_result, transfer_seconds = self._transfer_vorticity_to_vpm(*face_geometry)
            refresh_started = time.perf_counter()
            update_boundary_history_after_replacement(self, *face_geometry)
            boundary_seconds += time.perf_counter() - refresh_started
        output_started = time.perf_counter()
        with collective_phase(self._comm, "VPM particle-state check and output"):
            if self._is_root:
                self.vpm_solver.write_step_output()
        frame = self.vpm_case.output.schedule
        if frame is not None and frame.is_due(step, time_end, self.vpm_time_step_size):
            # The VPM wrote a particle frame; the FVM writes one at the same time.
            if not self._fvm_frame_written():
                self.fvm_solver.write_vtk()
            self._frame_step = step
        output_seconds = time.perf_counter() - output_started
        self._last_transfer_result = transfer_result
        record_step(
            self,
            step,
            time_end,
            (vpm_seconds, boundary_seconds, fvm_seconds, transfer_seconds),
            transfer_result,
            logger=logger,
            exchange_started=exchange_started,
            output_seconds=output_seconds,
        )

    def _start(self) -> None:
        """Prepare the first exchange of this process once (collective).

        A fresh state first receives the FVM vorticity at the initial time,
        so every VPM interval starts from the FVM state of the same time.
        """
        if self._started:
            return
        self.interface_trace_correction = None
        fvm = self.fvm_solver
        fvm.apply_velocity_boundaries(fvm.time)
        with collective_phase(self._comm, "initial freestream"):
            self._update_freestream(fvm.time)
        patch = self.config.coupling_patch
        self._face_geometry = (
            np.asarray(fvm.gather_boundary_face_centre_coordinates(patch)).reshape(-1, 3),
            np.asarray(fvm.gather_boundary_face_normal(patch)).reshape(-1, 3),
            np.asarray(fvm.gather_boundary_face_area(patch)).ravel(),
        )
        if self.step == 0 and not self._restart_loaded:
            self._last_transfer_result, _ = self._transfer_vorticity_to_vpm(*self._face_geometry)
        initialize_vpm_boundary_history(self, *self._face_geometry)
        self._started = True

    def _fvm_frame_written(self) -> bool:
        """Whether the FVM frame of the current exchange already exists."""
        if self._frame_step == self.step:
            return True
        schedule = self.fvm_case.output.schedule
        fvm = self.fvm_solver
        return (
            self.step > 0
            and schedule is not None
            and schedule.is_due(fvm.step, fvm.time, self.fvm_time_step_size)
        )

    def _write_frames(self, *, fvm_frame: bool, vpm_frame: bool) -> None:
        """Write the frames of the current exchange; a particle frame brings an FVM frame."""
        if (fvm_frame or vpm_frame) and not self._fvm_frame_written():
            self.fvm_solver.write_vtk()
        self.fvm_solver.flush_output()
        if vpm_frame and self._frame_step != self.step:
            with collective_phase(self._comm, "VPM frame"):
                if self._is_root:
                    self.vpm_solver.write_vtk()
            self._frame_step = self.step

    def _write_initial_output(self) -> None:
        """Write the initial frames and samples of both solvers; a particle frame brings an FVM frame."""
        fvm = self.fvm_solver
        fvm_frame = self.fvm_case.run.initial_output
        vpm_frame = self.vpm_case.run.initial_output
        fvm.write_initial_output()
        if vpm_frame and not fvm_frame:
            fvm.write_vtk()
        fvm.flush_output()
        with collective_phase(self._comm, "VPM initial output"):
            if self._is_root:
                self.vpm_solver.write_initial_output()
        if fvm_frame or vpm_frame:
            self._frame_step = self.step

    def _write_final_output(self) -> None:
        """Write the final backup, frames and samples of both solvers."""
        if self.config.backup.write_at_end and self._backup_step != self.step:
            self.save_backup()
        fvm = self.fvm_solver
        fvm_frame = self.fvm_case.run.final_output
        vpm_frame = self.vpm_case.run.final_output
        if vpm_frame and not fvm_frame and not self._fvm_frame_written():
            fvm.write_vtk()
        fvm.write_final_output()
        fvm.flush_output()
        with collective_phase(self._comm, "VPM final output"):
            if self._is_root:
                self.vpm_solver.write_final_output()
        if fvm_frame or vpm_frame:
            self._frame_step = self.step

    def _backup_due(self, step: int) -> bool:
        schedule = self.config.backup.schedule
        return schedule is not None and schedule.is_due(
            step, self.exchange_time(step), self.vpm_time_step_size
        )

    def save_backup(self, path: str | Path | None = None) -> Path:
        """Save a coupled backup of the current exchange with both frames.

        Parameters
        ----------
        path : str, pathlib.Path or None, default=None
            Backup directory; ``None`` writes
            ``solution/coupled/coupled_{step:06d}``.

        Returns
        -------
        pathlib.Path
            The backup directory, on every rank.

        Notes
        -----
        The backup holds the FVM backup, the VPM backup and the interface
        trace. The FVM and particle frames of the same time are written too.
        Collective under MPI.
        """
        self._start()
        destination = (
            coupled_backup_path(self.solution_dir, self.step) if path is None else Path(path)
        )
        save_coupled_backup(self, destination)
        self._backup_step = self.step
        self._write_frames(fvm_frame=True, vpm_frame=True)
        return destination

    def load_backup(
        self,
        path: str | Path,
        *,
        allowed_config_differences: Collection[str] = (),
        expected_config_differences: Mapping[str, tuple[object, object]] | None = None,
    ) -> None:
        """Restore a coupled backup into both solvers.

        Parameters
        ----------
        path : str or pathlib.Path
            Coupled backup directory written by :meth:`save_backup`.
        allowed_config_differences : collection[str], default=()
            Exact configuration paths allowed to differ from the backup;
            ``vpm.``-prefixed paths are passed to the VPM backup reader.
        expected_config_differences : mapping or None, default=None
            ``path: (stored_value, current_value)`` expectations for the
            allowed differences.

        Notes
        -----
        The output series of both solvers and the coupler diagnostics are
        rewound to the restored time. Collective under MPI.
        """
        self.interface_trace_correction = None
        step = load_coupled_backup(
            self,
            Path(path),
            allowed_config_differences=allowed_config_differences,
            expected_config_differences=expected_config_differences,
        )
        # A fresh run transfers once at the start and once per exchange.
        self.vorticity_transfer.step = step + 1
        self._restart_loaded = True
        self._started = False
        self._frame_step = self._backup_step = step

    def write_run_metadata(self, *, status: RunStatus | str) -> Path:
        """Write ``solution/coupler_metadata.json`` with the configuration and state.

        Returns
        -------
        pathlib.Path
            The metadata file, on every rank.
        """
        destination = self.solution_dir / "coupler_metadata.json"
        with collective_phase(self._comm, "coupler metadata"):
            if self._is_root:
                names = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
                box = dict(zip(names, map(float, self.fvm_box), strict=True))
                write_run_metadata(
                    destination,
                    solver="coupler",
                    case_name=self.fvm_case.name,
                    status=status,
                    configuration=self.config,
                    sections={
                        "coupling": {
                            "fvm_time_step_size": self.fvm_time_step_size,
                            "vpm_time_step_size": self.vpm_time_step_size,
                            "n_fvm_substeps": self.n_fvm_substeps,
                            "end_time": self.end_time,
                            "n_steps": self.n_steps,
                            "kinematic_viscosity": self.kinematic_viscosity,
                            "density": self.density,
                            "fvm_box": box,
                            "particle_spacing": self.vpm_particle_spacing,
                            "core_radius_ratio": self.vpm_core_radius_ratio,
                            "diffusion_grid_spacing": self.vpm_diffusion_grid_spacing,
                            "viscous_scheme": self.vpm_case.numerics.viscous.scheme,
                        },
                        "state": {"step": self.step, "time": float(self.time)},
                    },
                )
        return destination

    def flush_output(self) -> None:
        """Finish the pending file output of both solvers and the coupler log."""
        self.fvm_solver.flush_output()
        if self._is_root:
            self.vpm_solver.flush_output()
            flush_log(logger)

    def close(self, *, failure: BaseException | None = None) -> None:
        """Close both solvers and the coupler log; safe to repeat.

        A coupled solver advanced without :meth:`run` records ``completed``
        when it reached the end time and ``partial`` otherwise. While an
        exception ``failure`` unwinds, cleanup errors are attached to it as
        notes. Every MPI rank closes its solver.
        """
        if self._closed:
            return
        cleanup_error: BaseException | None = None
        if not self._run_started and self._started:
            self.run_status = (
                RunStatus.failed
                if failure is not None
                else RunStatus.completed
                if self.step >= self.n_steps
                else RunStatus.partial
            )
            try:
                self.write_run_metadata(status=self.run_status)
            except BaseException as error:
                cleanup_error = error
        try:
            self._resources.__exit__(
                None if failure is None else type(failure),
                failure,
                None if failure is None else failure.__traceback__,
            )
        except BaseException as error:
            cleanup_error = cleanup_error or error
        finally:
            if self._is_root:
                close_coupler_logging()
            self._closed = True
        if cleanup_error is not None:
            if failure is not None:
                failure.add_note(f"Coupled solver cleanup also failed: {cleanup_error!r}")
            else:
                raise cleanup_error

    # Exchange operations

    def _update_freestream(self, time: float) -> None:
        if self.config.freestream is not None:
            self.freestream_velocity = self.config.freestream.at(time)
            self.vorticity_transfer.config = replace(
                self.config, freestream_velocity=tuple(self.freestream_velocity)
            )
            if self._is_root:
                self.vpm_solver.set_freestream_velocity(self.freestream_velocity)

    def _advance_vpm(self, step: int, time_end: float) -> float:
        started = time.perf_counter()
        with collective_phase(self._comm, "VPM advance"):
            if self._is_root:
                vpm = self.vpm_solver
                vpm.physics.last_solid_projection = {
                    "stage_count": 0,
                    "accepted_count": 0,
                    "stage_displacement_l1": 0.0,
                    "accepted_displacement_l1": 0.0,
                    "stage_impulse_change": np.zeros(3),
                    "accepted_impulse_change": np.zeros(3),
                }
                vpm.set_freestream_velocity(self.freestream_velocity)
                begin_coupling_step(logger, step, self.n_steps, time_end)
                vpm.advance(defer_output=True, time=time_end)
                vpm.synchronize()
                if vpm.viscous_scheme == "gbd":
                    _validate_gbd_moment_recovery(
                        getattr(vpm.physics, "last_gbd_moment_recovery", None),
                        self.config.transfer_discretization_error_limit,
                    )
        return time.perf_counter() - started

    def _transfer_vorticity_to_vpm(
        self,
        face_centre: np.ndarray,
        _face_normals: np.ndarray,
        _face_area: np.ndarray,
    ):
        """Replace the FVM-derived part of the particle cloud."""
        started = time.perf_counter()
        velocity = self._gather_velocity_field()
        gradient = self._gather_velocity_gradient_field()
        gather_seconds = time.perf_counter() - started
        transfer_result = None
        with collective_phase(self._comm, "vorticity transfer"):
            if self._is_root:
                vpm = self.vpm_solver
                n_before = vpm.particles.n_particles_total
                sum_before = (
                    float(np.sum(np.linalg.norm(np.asarray(vpm.particle_vortex_strength), axis=1)))
                    if n_before > 0
                    else 0.0
                )
                transfer_result = self.vorticity_transfer.transfer(
                    vpm, velocity=velocity, velocity_gradient=gradient
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
                    "donor_gather_seconds": gather_seconds,
                }
        return transfer_result, time.perf_counter() - started

    def _gather_velocity_field(self) -> np.ndarray:
        if self._velocity_global_buffer is None:
            self._velocity_global_buffer = np.ascontiguousarray(
                self.fvm_solver.gather_velocity_field(), dtype=np.float64
            ).reshape(-1, 3)
        else:
            self.fvm_solver.gather_velocity_field_into(self._velocity_global_buffer)
        return self._velocity_global_buffer

    def _gather_velocity_gradient_field(self) -> np.ndarray:
        if self._velocity_gradient_global_buffer is None:
            self._velocity_gradient_global_buffer = np.ascontiguousarray(
                self.fvm_solver.gather_velocity_gradient_field(), dtype=np.float64
            ).reshape(-1, 3, 3)
        else:
            self.fvm_solver.gather_velocity_gradient_field_into(
                self._velocity_gradient_global_buffer
            )
        return self._velocity_gradient_global_buffer


__all__ = ["CoupledSolver"]
