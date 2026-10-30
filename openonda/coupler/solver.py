"""Coupled FVM--VPM solver: particle boundary data drive the FVM, FVM vorticity is transferred to the particles."""

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
from openonda.core.schedules import EverySteps
from openonda.core.simulation.parallel import collective_phase
from openonda.core.status import RunStatus
from openonda.fvm.core.time_step import event_aligned_time_step_size
from openonda.runtime import detected_world_size

from .backup import coupled_backup_path, load_coupled_backup, save_coupled_backup
from .boundary import (
    advance_fvm,
    evaluate_vpm_boundary,
    initialize_vpm_boundary_history,
    update_boundary_history_after_replacement,
)
from .config import CouplerConfig
from .frame import MeshFrame
from .reporting import (
    begin_coupling_step,
    close_coupler_logging,
    configure_coupler_logging,
    flush_log,
    format_coupler_log,
    record_step,
)
from .vorticity_transfer import VorticityTransfer, remeshing_extension_length

if TYPE_CHECKING:
    from openonda.fvm import FVMCase, FVMSolver
    from openonda.vpm import VPMCase, VPMSolver

logger = logging.getLogger("openonda.coupler")


def _validate_gbd_moment_correction(
    correction: Mapping[str, bool | int | float] | None,
    correction_limit: float,
) -> None:
    """Stop when the GBD weak-strength removal of the last VPM step failed its moment checks."""
    if correction is None:
        return
    numeric_names = (
        "nonzero_node_count",
        "retained_node_count",
        "removed_node_count",
        "correction_fraction",
        "normalized_vortex_strength_residual",
        "normalized_linear_impulse_residual",
        "normalized_angular_impulse_residual",
    )
    try:
        values = {name: float(correction[name]) for name in numeric_names}
        values["support_augmented_node_count"] = float(
            correction.get("support_augmented_node_count", 0)
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("GBD moment-correction diagnostics are incomplete or invalid") from error
    for name, value in values.items():
        if not np.isfinite(value):
            raise RuntimeError(f"GBD moment-correction diagnostic {name!r} is non-finite")
    for name in (
        "nonzero_node_count",
        "retained_node_count",
        "removed_node_count",
        "support_augmented_node_count",
    ):
        if values[name] < 0.0:
            raise RuntimeError(f"GBD moment-correction diagnostic {name!r} is negative")

    if values["removed_node_count"] > 0.0 and not bool(correction.get("applied", False)):
        raise RuntimeError("GBD removed weak strengths without the moment correction")
    maximum_residual = max(
        values["normalized_vortex_strength_residual"],
        values["normalized_linear_impulse_residual"],
        values["normalized_angular_impulse_residual"],
    )
    if maximum_residual > 1.0e-5:
        raise RuntimeError(
            "the GBD moment correction exceeds its storage-precision residual tolerance"
        )
    if values["correction_fraction"] > float(correction_limit):
        raise RuntimeError(
            "the GBD weak-strength removal needed a moment correction above its limit: "
            f"fraction={values['correction_fraction']:.6e}, "
            f"limit={float(correction_limit):.6e}"
        )


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

    initial_substep_ratio = numerics.time_step_size / fvm_case.numerics.time_step_size
    nominal_fvm_substeps = max(
        1, math.ceil(initial_substep_ratio - 1.0e-12 * max(1.0, initial_substep_ratio))
    )
    if config.max_coupling_iterations > 1 and (
        nominal_fvm_substeps > 1 or fvm_case.numerics.time_step_control is not None
    ):
        fvm_step_schedules = [fvm_case.output.schedule]
        fvm_step_schedules.extend(sampler.schedule for sampler in fvm_case.samplers.samples)
        if any(isinstance(schedule, EverySteps) for schedule in fvm_step_schedules):
            raise ValueError(
                "FVM output and samplers that use EverySteps cannot be combined with multiple "
                "FVM substeps and coupling iterations; use EveryTime schedules"
            )


def _validate_vpm_domain(vpm_case: VPMCase, config: CouplerConfig, fvm_bounds: np.ndarray) -> None:
    """Check the particle domain, slip planes and fringes against the FVM domain bounds."""
    numerics = vpm_case.numerics
    induction = numerics.induction
    if getattr(induction, "method", None) == "slip_slab":
        # The FVM extrusion, transfer region and VPM slab share one span.
        planes = np.array([induction.z_min, induction.z_max], dtype=np.float64)
        tolerance = 1.0e-9 * max(1.0, float(np.max(np.abs(planes))))
        transfer = config.transfer_region(
            fvm_bounds, numerics.viscous.particle_spacing, slip_planes=True
        )
        for name, bounds in (("FVM domain", fvm_bounds), ("transfer_region_bounds", transfer)):
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
        tolerance = 1.0e-9 * max(1.0, float(np.max(np.abs(fvm_bounds))))
        lower = np.asarray(domain[::2]) <= fvm_bounds[::2] + tolerance
        upper = np.asarray(domain[1::2]) >= fvm_bounds[1::2] - tolerance
        if not (np.all(lower) and np.all(upper)):
            raise ValueError(
                f"The VPM domain {tuple(domain)} does not contain the FVM domain {tuple(fvm_bounds)}: "
                "the transfer-lattice particles would be removed every step"
            )
    stabilization = numerics.stabilization
    fringe = getattr(stabilization, "outflow_fringe_depth", None)
    if fringe is not None:
        depth = np.asarray(fringe, dtype=np.float64)
        keep = np.asarray(stabilization.remove_particles_by_bounds, dtype=np.float64)
        inner = keep + depth * np.array([1.0, -1.0] * 3)
        lower_reach = (depth[::2] > 0.0) & (inner[::2] > fvm_bounds[::2])
        upper_reach = (depth[1::2] > 0.0) & (inner[1::2] < fvm_bounds[1::2])
        if np.any(lower_reach) or np.any(upper_reach):
            raise ValueError(
                f"The VPM outflow fringes {tuple(fringe)} reach the FVM domain {tuple(fvm_bounds)}; "
                "place each fringe outside the FVM domain"
            )


class CoupledSolver:
    """Inner FVM solution coupled to an outer VPM wake.

    The FVM CFL controller selects each FVM substep. The VPM interval equals
    the sum of those substep sizes, and the coupler shortens the interval when
    the particle strain limit requires it. Physical-time samples, frames,
    backups and the run horizon remain exact event times.
    Particle velocity and its normal gradient on the outer FVM boundary drive
    the FVM substeps; FVM vorticity then replaces particle vorticity in the
    transfer region.
    On a translating FVM mesh the VPM runs in the frame of the mesh, with the
    freestream relative to the mesh (:class:`~openonda.coupler.frame.MeshFrame`).

    Parameters
    ----------
    fvm_case : FVMCase
        Inner flow, mesh and FVM output. Under MPI every rank constructs the
        coupled solver; the FVM runs on all ranks.
    vpm_case : VPMCase
        Particle wake and its output; the VPM runs on the root rank. Its
        configured time-step size is the maximum coupling interval, and its
        directory and variant are those of the FVM case.
    config : CouplerConfig
        Vorticity transfer, coupling iterations and coupled backups.

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
        Current FVM substep and VPM coupling interval in s.
    n_fvm_substeps : int
        Number of FVM substeps in the current VPM coupling interval.
    frame : MeshFrame
        Frame translating with the FVM mesh, in which the VPM is solved.
    freestream_velocity : numpy.ndarray
        VPM freestream in m/s, relative to the FVM mesh.
    absolute_freestream_velocity : numpy.ndarray
        Inertial-frame freestream in m/s.
    coupling_diagnostics : list[dict]
        Diagnostics of the coupling steps of this process, also appended to
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
        self.boundary_data_correction: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self.coupling_diagnostics: list[dict] = []
        self._closed = False
        self._run_started = False
        self._started = False
        self._restart_loaded = False
        self._frame_step: int | None = None
        self._backup_step: int | None = None
        self._coupling_step = 0
        self._resources = ExitStack()
        self.vpm_solver: VPMSolver | None = None
        self._is_root = True
        try:
            self.fvm_solver: FVMSolver = self._resources.enter_context(
                FVMSolver(replace(fvm_case, backup=FVMBackup(write_at_end=False)))
            )
            self._comm = self.fvm_solver.parallel.comm
            # The VPM runs in the frame translating with the FVM mesh; all
            # coupling geometry is read in that frame.
            self.frame = MeshFrame(self.fvm_solver)
            self._is_root = bool(self.fvm_solver.parallel.is_root)
            if self._is_root:
                configure_coupler_logging(self.solution_dir / "coupler.log", console=True)
            with collective_phase(self._comm, "VPM construction"):
                if self._is_root:
                    self.vpm_solver = self._resources.enter_context(
                        VPMSolver(
                            replace(vpm_case, backup=VPMBackup(write_at_end=False)),
                            external_clock=True,
                        )
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
        initial_substep_ratio = self.vpm_time_step_size / self.fvm_time_step_size
        self.nominal_fvm_substeps_per_vpm_step = max(
            1, math.ceil(initial_substep_ratio - 1.0e-12 * max(1.0, initial_substep_ratio))
        )
        self.n_fvm_substeps = 1
        control = self.fvm_case.numerics.time_step_control
        self.maximum_fvm_time_step_size = (
            float(control.maximum_time_step_size)
            if control is not None and control.maximum_time_step_size is not None
            else self.fvm_time_step_size
            if control is None
            else None
        )
        self.maximum_coupling_time_step_size = self.vpm_time_step_size
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
        self.frame.validate(fvm.time, self.end_time, self.fvm_time_step_size)
        self.freestream_velocity = self.config.freestream_velocity_vector
        self.absolute_freestream_velocity = self.freestream_velocity.copy()
        self._velocity_boundary_condition_old: np.ndarray | None = None
        self._normal_velocity_boundary_condition_old: np.ndarray | None = None
        self._normal_velocity_boundary_condition: np.ndarray | None = None
        self._normal_gradient_boundary_condition_old: np.ndarray | None = None
        self._normal_gradient_boundary_condition: np.ndarray | None = None
        self._gathered_velocity: np.ndarray | None = None
        self._gathered_velocity_gradient: np.ndarray | None = None
        self._last_vpm_boundary_condition_flux_diagnostics = {
            "raw_mismatch": 0.0,
            "raw_relative": 0.0,
            "acceptance_limit": 0.0,
            "applied_correction": 0.0,
            "corrected_mismatch": 0.0,
        }
        self._last_fvm_boundary_data_diagnostics = {
            "mean_velocity_mismatch": 0.0,
            "maximum_velocity_mismatch": 0.0,
            "maximum_normal_velocity_mismatch": 0.0,
            "mean_outflow_velocity_mismatch": 0.0,
            "maximum_outflow_velocity_mismatch": 0.0,
        }
        self._last_transfer_result = None
        self._step_transfer_stats: dict[str, float | int] | None = None

        self.fvm_domain_bounds = self._gather_fvm_domain_bounds()
        self.config.validate_transfer_region(
            self.fvm_domain_bounds,
            self.vpm_particle_spacing,
            getattr(self.vpm_case.numerics.induction, "method", None) == "slip_slab",
        )
        _validate_vpm_domain(self.vpm_case, self.config, self.fvm_domain_bounds)
        if self._is_root:
            backup = self.config.backup.schedule
            logger.info(
                format_coupler_log(
                    "time stepping",
                    ("initial fvm time step", f"{self.fvm_time_step_size:.4e}", "s"),
                    (
                        "maximum fvm time step",
                        "none"
                        if self.maximum_fvm_time_step_size is None
                        else f"{self.maximum_fvm_time_step_size:.4e}",
                        "s",
                    ),
                    ("maximum vpm interval", f"{self.maximum_coupling_time_step_size:.4e}", "s"),
                    ("nominal fvm substeps per vpm step", self.nominal_fvm_substeps_per_vpm_step),
                    ("end time", f"{self.end_time:.6g}", "s"),
                    ("coupled backups", "none" if backup is None else backup.describe()),
                )
            )

        self.vorticity_transfer = VorticityTransfer(self)
        if self.frame.moving:
            # The remeshing region must hold the maximum shared-step drift relative to the mesh.
            relative_speed = self.frame.maximum_relative_speed(
                self._absolute_freestream,
                fvm.time,
                self.end_time,
                self.maximum_coupling_time_step_size,
            )
            self.vorticity_transfer.remeshing_extension_length = remeshing_extension_length(
                [relative_speed, 0.0, 0.0],
                self.maximum_coupling_time_step_size,
                self.vpm_particle_spacing,
            )
        self.vorticity_transfer.setup(fvm)
        with collective_phase(self._comm, "VPM grid configuration"):
            if self._is_root:
                self._configure_vpm_geometry()

    def _gather_fvm_domain_bounds(self) -> np.ndarray:
        """Return the outer FVM bounds from its non-wall patches (collective)."""
        fvm = self.fvm_solver
        patch = self.config.outer_boundary_patch
        names = [patch] + [
            boundary.name
            for boundary in self.fvm_case.boundaries
            if boundary.name != patch and boundary.mesh_type != "wall"
        ]
        faces = [
            self.frame.to_frame(
                np.asarray(fvm.gather_boundary_face_centre_coordinates(name)).reshape(-1, 3)
            )
            for name in names
        ]
        bounds = None
        with collective_phase(self._comm, "FVM domain bounds"):
            if self._is_root:
                if len(faces[0]) == 0:
                    raise ValueError(f"The coupling patch {patch!r} has no faces")
                points = np.concatenate(faces)
                bounds = np.ravel(np.column_stack((points.min(axis=0), points.max(axis=0))))
        if self._comm is not None and self._comm.Get_size() > 1:
            bounds = self._comm.bcast(bounds, root=0)
        return np.asarray(bounds, dtype=np.float64)

    def _configure_vpm_geometry(self) -> None:
        """Give the VPM the FVM wall geometry and the transfer lattice phase."""
        vpm = self.vpm_solver
        transfer = self.vorticity_transfer
        boundary = transfer.solid_boundary
        if boundary is not None:
            from .solid import WallProjection

            vpm.physics.configure_body_classifier(
                boundary.contains,
                revision=boundary.revision,
                query_bounds=boundary.bounds,
                segments_cross_wall=boundary.segments_cross_wall,
                geometry_queries=boundary.grid_geometry_queries(),
            )
            projection = WallProjection(boundary, vpm.physics, self.vpm_particle_spacing, logger)
            vpm.stage_rhs.position_guard = projection.stage
            vpm.stage_rhs.accepted_position_projector = projection.accepted
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
    def step(self) -> int:
        """Number of completed coupling steps."""
        return self._coupling_step

    @property
    def time(self) -> float:
        """Accepted physical time in s."""
        return self.fvm_solver.time

    def coupling_time(self, step: int) -> float:
        """Return the current or next shared physical time, in s."""
        if step == self.step:
            return float(self.time)
        if step == self.step + 1:
            return float(self.time + self.vpm_time_step_size)
        raise ValueError("The coupled clock exposes only the current and next accepted times")

    # Lifecycle

    def __enter__(self) -> CoupledSolver:
        return self

    def __exit__(self, _exc_type, exc_value, _traceback) -> None:
        self.close(failure=exc_value)

    def start_from(self, selection: str | Path = "latest") -> bool:
        """Select the initial state or a coupled backup before the first coupling step.

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
            logger.info(
                f"Continuing from {selected}: coupling step {self.step}, time {self.time:.17g}"
            )
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
        them. The run then advances the coupling steps, writes the scheduled output
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
            while self.time < self.end_time - max(1.0e-14, abs(self.end_time) * 1.0e-12):
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
        """Advance one shared FVM--VPM step and write its due output and backup.

        The VPM and FVM advance by one shared selected step, the boundary data
        of the VPM predictor drive the FVM, the FVM vorticity is transferred to
        the particles of the transfer region, and the due samples, frames and
        coupled backup are written. Every MPI rank calls this method.
        """
        if self._closed:
            raise RuntimeError("The coupled solver is closed")
        self._start()
        step = self.step + 1
        if self.time >= self.end_time - max(1.0e-14, abs(self.end_time) * 1.0e-12):
            raise RuntimeError("The coupled run has reached its end time")
        if not self._run_started:
            self.run_status = RunStatus.running
        face_geometry = self._face_geometry
        interval_started = time.perf_counter()
        self._select_coupling_interval()
        time_end = self.coupling_time(step)
        with collective_phase(self._comm, "free stream at the interval end"):
            self._update_freestream(time_end, self._interval_frame_velocity(step))
        vpm_seconds = self._advance_vpm(step, time_end)
        _, next_velocity, boundary_seconds = evaluate_vpm_boundary(self, *face_geometry)
        if self.config.max_coupling_iterations > 1:
            from .coupling_iteration import advance_coupling_iterations

            transfer_result, fvm_seconds, transfer_seconds = advance_coupling_iterations(
                self, face_geometry, next_velocity
            )
            boundary_seconds += self._last_coupling_iteration_diagnostics["phase_seconds"][
                "boundary_refresh"
            ]
        else:
            fvm_seconds = advance_fvm(self, *face_geometry, next_velocity)
            transfer_result, transfer_seconds = self._transfer_vorticity_to_vpm(*face_geometry)
            refresh_started = time.perf_counter()
            update_boundary_history_after_replacement(self, *face_geometry)
            boundary_seconds += time.perf_counter() - refresh_started
        self._coupling_step = step
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
            interval_started=interval_started,
            output_seconds=output_seconds,
        )

    def _start(self) -> None:
        """Prepare the first coupling step of this process once (collective).

        A fresh state first receives the FVM vorticity at the initial time,
        so every VPM interval starts from the FVM state of the same time.
        """
        if self._started:
            return
        self.boundary_data_correction = None
        fvm = self.fvm_solver
        fvm.apply_velocity_boundaries(fvm.time)
        with collective_phase(self._comm, "initial freestream"):
            self._update_freestream(fvm.time, self.frame.velocity(fvm.time))
        patch = self.config.outer_boundary_patch
        self._face_geometry = (
            self.frame.to_frame(
                np.asarray(fvm.gather_boundary_face_centre_coordinates(patch)).reshape(-1, 3)
            ),
            np.asarray(fvm.gather_boundary_face_normal(patch)).reshape(-1, 3),
            np.asarray(fvm.gather_boundary_face_area(patch)).ravel(),
        )
        if self.step == 0 and not self._restart_loaded:
            self._last_transfer_result, _ = self._transfer_vorticity_to_vpm(*self._face_geometry)
        initialize_vpm_boundary_history(self, *self._face_geometry)
        self._started = True

    def _fvm_frame_written(self) -> bool:
        """Whether the FVM frame of the current coupling step already exists."""
        if self._frame_step == self.step:
            return True
        schedule = self.fvm_case.output.schedule
        fvm = self.fvm_solver
        return (
            self.step > 0
            and schedule is not None
            and schedule.is_due(fvm.step, fvm.time, fvm._accepted_time_step_size)
        )

    def _write_frames(self, *, fvm_frame: bool, vpm_frame: bool) -> None:
        """Write the frames of the current coupling step; a particle frame brings an FVM frame."""
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
            step, self.time, self.fvm_solver._accepted_time_step_size
        )

    def save_backup(self, path: str | Path | None = None) -> Path:
        """Save a coupled backup of the current coupling step with both frames.

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
        The backup holds the FVM backup, the VPM backup and the boundary
        data. The FVM and particle frames of the same time are written too.
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
        self.boundary_data_correction = None
        step = load_coupled_backup(
            self,
            Path(path),
            allowed_config_differences=allowed_config_differences,
            expected_config_differences=expected_config_differences,
        )
        # A fresh run transfers once at the start and once per coupling step.
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
                domain_bounds = dict(zip(names, map(float, self.fvm_domain_bounds), strict=True))
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
                            "fvm_substeps": self.n_fvm_substeps,
                            "maximum_time_step_size": self.maximum_coupling_time_step_size,
                            "maximum_fvm_time_step_size": self.maximum_fvm_time_step_size,
                            "nominal_fvm_substeps_per_vpm_step": (
                                self.nominal_fvm_substeps_per_vpm_step
                            ),
                            "end_time": self.end_time,
                            "kinematic_viscosity": self.kinematic_viscosity,
                            "density": self.density,
                            "fvm_domain_bounds": domain_bounds,
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
                if self.time >= self.end_time - max(1.0e-14, abs(self.end_time) * 1.0e-12)
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

    # Coupling-interval operations

    def _absolute_freestream(self, time: float) -> np.ndarray:
        """Inertial-frame freestream velocity in m/s at ``time``."""
        if self.config.freestream is not None:
            return np.asarray(self.config.freestream.at(time), dtype=np.float64)
        return self.config.freestream_velocity_vector

    def _interval_frame_velocity(self, step: int) -> np.ndarray:
        """Mean mesh velocity over the next VPM interval, in m/s."""
        return self.frame.step_velocity(np.asarray([self.coupling_time(step)]))

    def _coupling_event_schedules(self) -> tuple:
        """Return schedules that must land on VPM coupling times."""
        schedules = []
        if self.config.max_coupling_iterations > 1:
            schedules.extend(self.fvm_solver._run_event_schedules())
        schedules.append(self.vpm_case.logging.schedule)
        if self.vpm_case.output.schedule is not None:
            schedules.append(self.vpm_case.output.schedule)
        if self.config.backup.schedule is not None:
            schedules.append(self.config.backup.schedule)
        schedules.extend(
            sampler.schedule
            for sampler in self.vpm_case.samplers.samples
            if sampler.schedule is not None
        )
        return tuple(schedules)

    def _vpm_strain_time_step_limit(self) -> float:
        """Return a conservative step ceiling from the last accepted VPM strain."""
        vpm = self.vpm_solver
        metrics = vpm._accepted_state_metrics
        maximum = vpm.state_limits.lagrangian_cfl.maximum
        if maximum is None:
            return self.maximum_coupling_time_step_size
        if metrics is None:
            strain_rate = vpm.maximum_lagrangian_strain_rate()
        elif metrics.strain_increment_infinity > 0.0:
            strain_rate = metrics.strain_increment_infinity / vpm.time_step_size
        else:
            strain_rate = 0.0
        if strain_rate <= 0.0:
            return self.maximum_coupling_time_step_size
        return 0.9 * maximum / strain_rate

    def _select_coupling_interval(self) -> float:
        """Select an event-aligned VPM interval from FVM CFL and VPM strain limits."""
        schedules = self._coupling_event_schedules()
        fvm_limit = self.fvm_solver.courant_time_step_limit()
        if self.maximum_fvm_time_step_size is not None:
            fvm_limit = min(fvm_limit, self.maximum_fvm_time_step_size)
        strain_limit = None
        if self._is_root:
            strain_limit = self._vpm_strain_time_step_limit()
        if self._comm is not None and self._comm.Get_size() > 1:
            strain_limit = self._comm.bcast(strain_limit, root=0)

        maximum_interval = self.maximum_coupling_time_step_size
        maximum_interval = min(
            maximum_interval,
            self.nominal_fvm_substeps_per_vpm_step * fvm_limit,
        )
        if strain_limit is not None:
            maximum_interval = min(maximum_interval, float(strain_limit))
        remaining = self.end_time - self.time
        maximum_interval = min(maximum_interval, remaining)
        next_event = min(
            (
                event_time
                for schedule in schedules
                if (event_time := schedule.next_time_after(self.time)) is not None
            ),
            default=self.end_time,
        )
        event_distance = min(remaining, next_event - self.time)
        selected_interval = event_aligned_time_step_size(maximum_interval, event_distance)
        self.fvm_time_step_size = fvm_limit
        self.vpm_time_step_size = selected_interval
        if self._is_root:
            self.vpm_solver.time_step_size = selected_interval
        self.vorticity_transfer.coupling_time_step_size = selected_interval
        return selected_interval

    def _update_freestream(self, time: float, frame_velocity: np.ndarray) -> None:
        """Set the freestream at ``time`` and the VPM freestream relative to the mesh."""
        self.absolute_freestream_velocity = self._absolute_freestream(time)
        if self.config.freestream is not None or self.frame.moving:
            self.freestream_velocity = self.absolute_freestream_velocity - frame_velocity
            self.vorticity_transfer.config = replace(
                self.config, freestream_velocity=tuple(self.freestream_velocity.tolist())
            )
            if self._is_root:
                self.vpm_solver.set_freestream_velocity(self.freestream_velocity)

    def inertial_boundary_velocity(self, velocity: np.ndarray) -> np.ndarray:
        """Convert a VPM velocity, with the VPM freestream, to the inertial frame."""
        return velocity + (self.absolute_freestream_velocity - self.freestream_velocity)

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
                begin_coupling_step(logger, step, None, time_end)
                vpm.advance(defer_output=True, time=time_end)
                vpm.synchronize()
                if vpm.viscous_scheme == "gbd":
                    _validate_gbd_moment_correction(
                        getattr(vpm.physics, "last_gbd_moment_correction", None),
                        self.config.max_moment_correction_fraction,
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
        if self.frame.moving:
            # The particles see the flow relative to the mesh.
            velocity = velocity - self.frame.velocity(self.fvm_solver.time)
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
                    "fvm_gather_seconds": gather_seconds,
                }
        return transfer_result, time.perf_counter() - started

    def _gather_velocity_field(self) -> np.ndarray:
        if self._gathered_velocity is None:
            self._gathered_velocity = np.ascontiguousarray(
                self.fvm_solver.gather_velocity_field(), dtype=np.float64
            ).reshape(-1, 3)
        else:
            self.fvm_solver.gather_velocity_field_into(self._gathered_velocity)
        return self._gathered_velocity

    def _gather_velocity_gradient_field(self) -> np.ndarray:
        if self._gathered_velocity_gradient is None:
            self._gathered_velocity_gradient = np.ascontiguousarray(
                self.fvm_solver.gather_velocity_gradient_field(), dtype=np.float64
            ).reshape(-1, 3, 3)
        else:
            self.fvm_solver.gather_velocity_gradient_field_into(self._gathered_velocity_gradient)
        return self._gathered_velocity_gradient


__all__ = ["CoupledSolver"]
