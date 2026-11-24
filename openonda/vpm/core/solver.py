"""Vortex Particle Method solver.

Provides DNS, LES, and inviscid VPM models with Taichi acceleration,
viscous diffusion, diagnostics, sampling, and restart support.
"""

from collections.abc import Collection, Mapping, Sequence
from contextlib import suppress
import math
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import NDArray
import taichi as ti

from openonda.core.clock import time_tolerance
from openonda.core.errors import NonFiniteSolutionError
from openonda.core.log_style import StepLog
from openonda.core.output_events import OutputEvent
from openonda.core.profiling import Profiler, Stopwatch
from openonda.core.restart import reset_run_outputs
from openonda.core.status import RunStatus

from ..config.case import Numerics, VPMCase
from ..config.stabilization import StabilizationConfig
from ..diagnostics.particle_state import (
    ParticleStateError,
    ParticleStateMetrics,
    check_accepted_particle_state,
    measure_particle_state,
)
from ..diagnostics.resolution import particle_resolution_metrics
from ..io.backup import _BackupIO, rewind_vpm_frames, select_vpm_backup
from ..io.frames import write_solution_frames
from ..io.logging import run_finished, run_header, startup
from ..io.metadata import write_vpm_metadata
from ..kernels.base import make_vortex_kernel
from ..numerics.runge_kutta import RungeKutta
from ..particles.container import FIELD_TRAILING_SHAPES, Particles, inside_box
from ..physics.diffusion.solenoidal_projection import SolenoidalLatticeProjection
from ..physics.engine import PhysicsEngine
from ..physics.evaluation import ParticleFieldEvaluation
from ..physics.induction.base import StageRates, StageState
from ..physics.stage_rhs import (
    ParticleExternalStageContribution,
    StageRHS,
    VLMStageContribution,
)
from ..runtime.backend import (
    reset_taichi_backend,
    start_taichi,
)
from ..runtime.timing import remainder_section
from ..sampling.manager import SamplerManager
from ..stabilization import StabilizationManager
from ..stabilization.context import (
    SolverParticleMutations,
    StabilizationContext,
    StabilizationStepState,
)
from ..turbulence.les import ParticleLES
from ..turbulence.spectral_cusp import SpectralCuspClosure
from ..vlm.output import VLMDiagnostics, VLMLoadingDistribution
from .evolution import EvolutionStepper

if TYPE_CHECKING:
    from ..config.viscous import ViscousConfig
    from ..particles.snapshot import ParticleSnapshot
    from ..vlm.config import VLMConfig

FloatArray: TypeAlias = NDArray[np.float32] | NDArray[np.float64]
ParticleRecord: TypeAlias = dict[str, np.ndarray | np.generic]
_PRESSURE_HIERARCHICAL_OPENING = 0.3


@ti.data_oriented
class VPMSolver:
    """Own and advance one complete Vortex Particle Method simulation.

    The solver holds the mutable particle fields, accepted clock, induction
    backend, Runge--Kutta workspace, viscous/turbulence models, optional VLM
    coupling, diagnostics, samplers, and restart I/O. A particle's
    ``vortex_strength`` is ``Gamma = omega * V`` in m³/s; it is distinct from
    vorticity ``omega`` in 1/s and from core radius ``sigma`` in m.

    Parameters
    ----------
    case : VPMCase
        Immutable construction object containing numerical, physical, output,
        and initial-condition settings. The case is validated before device
        fields are allocated.

    Attributes
    ----------
    time : float
        Accepted physical time in seconds.
    step : int
        Accepted integer step index.
    particles : Particles
        Fixed-capacity mutable device container. Public CPU properties expose
        copies of its active prefix.
    numerics : Numerics
        Immutable numerical configuration used to build the runtime.
    solution_dir, samples_dir : pathlib.Path
        ``solution/`` and ``samples/`` of the case, below ``variant`` when
        the case has one.

    Notes
    -----
    Constructing a solver claims/configures the process Taichi backend and
    creates output directories. :meth:`run` owns the complete run and
    closes resources on success or failure; interactive callers use the
    solver as a context manager and call :meth:`advance`. Numerical state is
    updated in place, while an accepted step is committed only after all
    physical phases pass.
    """

    # Initialization

    def __init__(self, case: VPMCase, *, external_clock: bool = False) -> None:
        """Validate a case and allocate its runtime workspaces.

        Parameters
        ----------
        case : VPMCase
            Immutable VPM configuration. Its particle capacity controls all
            fixed-size device allocations; initial-condition builders are
            invoked by :meth:`run` or the first :meth:`advance`.
        external_clock : bool, default=False
            Let a coupled driver set each accepted physical time and shorten
            the final interval to its own end time. The VPM cannot be run
            independently when this option is enabled.

        Raises
        ------
        TypeError
            If ``case`` is not a :class:`VPMCase`.
        ValueError
            If the selected precision, device, induction backend, or coupled
            configuration is unsupported.
        RuntimeError
            If backend initialization or an owned subsystem fails. Resources
            acquired before the failure are released before the exception is
            propagated.

        Notes
        -----
        Creates ``solution/`` and ``samples/``, configures the
        ``openonda.vpm`` logger, claims the process Taichi runtime, allocates
        the particle and physics fields and writes ``vpm_metadata.json``.
        """
        if not isinstance(case, VPMCase):
            raise TypeError("VPMSolver requires a VPMCase construction object")
        if not isinstance(external_clock, bool):
            raise TypeError("external_clock must be a bool")
        if not external_clock:
            case.run.n_steps(case.numerics.time_step_size)
        self.case = case
        self.numerics = case.numerics
        self.case_dir = Path(case.directory).resolve()
        variant = case.variant or ""
        self.solution_dir = self.case_dir / "solution" / variant
        self.samples_dir = self.case_dir / "samples" / variant
        self._backend_claimed = False
        self._closed = False
        self.time = 0.0
        self.step = 0
        self._restart_details: dict | None = None
        self._restart_loaded = False
        self._restart_output_time: float | None = None
        self._restart_terminal_noop = False
        self._restart_particle_count: int | None = None
        self._initial_conditions_built = False
        self._n_initial_particles = 0
        self._run_initial_step = 0
        self._run_initial_time = 0.0
        self._external_clock = bool(external_clock)
        self._run_final_step = None if self._external_clock else case.n_steps
        self._run_started = False
        self._run_clock: Stopwatch | None = None
        self._last_backup_state: tuple | None = None
        self._last_frame_state: tuple | None = None
        # Key of the accepted state whose transport velocity the particle velocity
        # field holds, between the diagnostics of an output event and its backup.
        self._velocity_valid_for: tuple | None = None
        self._particle_snapshot_buffers: dict = {}
        self._mesh_cleanup_failure: BaseException | None = None
        self._loading_numerical_state = False
        self.run_status = RunStatus.not_started
        self.run_failure: BaseException | None = None
        self._evolution_failure: BaseException | None = None
        self.logger = StepLog(
            "VPM",
            self.solution_dir / "vpm.log",
            case.logging.schedule,
            console=case.logging.console,
            debug=case.logging.mode == "debug",
        )
        try:
            self.solution_dir.mkdir(parents=True, exist_ok=True)
            self.samples_dir.mkdir(parents=True, exist_ok=True)
            self._initialize_numerics(case)
            self._initialize_backend(case.numerics)
            self._initialize_particles_and_physics(case.numerics)
            self._initialize_turbulence(case.numerics)
            self._initialize_solvers(case.numerics)
            self._sampler_manager = SamplerManager(self)
            startup(self)
            write_vpm_metadata(self, status=RunStatus.created)
        except BaseException:
            self._cleanup_failed_construction()
            raise

    def __enter__(self) -> "VPMSolver":
        """Return the solver; :meth:`close` runs when the block exits."""
        return self

    def __exit__(self, _exc_type: object, exc: BaseException | None, _traceback: object) -> None:
        """Close the solver, attaching cleanup errors to a pending exception."""
        self.close(failure=exc)

    def _cleanup_failed_construction(self) -> None:
        """Release every resource acquired before construction failed."""
        with suppress(Exception):
            self.logger.close()
        if self._backend_claimed:
            # Preserve the original construction exception if cleanup fails.
            with suppress(Exception):
                reset_taichi_backend(solver=self)
            self._backend_claimed = False

    @staticmethod
    def synchronize() -> None:
        """Wait until all queued VPM backend work has completed."""
        ti.sync()

    def _initialize_numerics(self, case: VPMCase) -> None:
        """Set the scalar solver state derived from the numerical settings."""
        final_setup = case.numerics
        self.time_step_size = final_setup.time_step_size

        vc = final_setup.viscous

        # RWM accuracy criterion.
        if (
            vc.scheme == "rwm"
            and vc.particle_spacing is not None
            and vc.particle_spacing > 0
            and vc.kinematic_viscosity is not None
            and vc.kinematic_viscosity > 0
        ):
            rwm_max_time_step_size = vc.rwm_accuracy_time_step_size()
            if self.time_step_size > rwm_max_time_step_size * (1.0 + 1e-6):
                self.logger.warning(
                    "RWM time step exceeds the accuracy limit",
                    ("time step", self.time_step_size, "s"),
                    ("accuracy limit", rwm_max_time_step_size, "s"),
                    ("criterion", "particle spacing squared / (4 * kinematic viscosity)"),
                )

        # DVH's compact heat support is a resolved transfer only when
        # beta*R_d²/(4*nu) has elapsed.  Accumulate smaller accepted steps and
        # apply the full physical interval once the lattice can represent it;
        # direct diffusion calls validate the same conditions.
        self._n_steps_per_dvh_diffusion: int = 1
        self._n_steps_since_dvh_diffusion: int = 0
        if (
            vc.scheme == "dvh"
            and vc.dvh_grid_spacing is not None
            and vc.kinematic_viscosity is not None
            and vc.kinematic_viscosity > 0.0
        ):
            required_time_step_size = vc.dvh_required_time_step_size()
            self._n_steps_per_dvh_diffusion = max(
                1,
                int(np.ceil(required_time_step_size / self.time_step_size)),
            )

        self.integrator_tableau = final_setup.integrator
        configured_induction = final_setup.induction
        self.induction = configured_induction.build()
        self.compute_device = case.run.runtime_compute_device or final_setup.compute_device
        self.flow_model = final_setup.turbulence.flow_model
        self.viscous_scheme = final_setup.viscous.scheme
        self._viscous_config = final_setup.viscous
        self.stabilization_config: StabilizationConfig = final_setup.stabilization
        self.state_limits = final_setup.state_limits
        self.particle_kernel = final_setup.particle_kernel

    def _initialize_backend(self, final_setup: Numerics) -> None:
        """Initialize the Taichi backend at the configured precision."""
        self.precision = final_setup.precision
        self.write_precision = self.case.output.precision
        workspace_estimate = getattr(self.induction, "estimated_workspace_bytes", None)
        minimum_pool_bytes = (
            workspace_estimate(final_setup.max_n_particles) + (1 << 30)
            if workspace_estimate is not None
            else 0
        )
        self.compute_device = start_taichi(
            self,
            device=self.compute_device,
            precision=self.precision,
            debug=self.case.logging.mode == "debug",
            random_seed=final_setup.random_seed,
            supported_devices=self.induction.supported_devices,
            minimum_pool_bytes=minimum_pool_bytes,
            logger=self.logger,
        )
        self._backend_claimed = True
        # ``close()`` releases the runtime before the terminal metadata is
        # written, so the metadata records the device resolved here.
        self._backend_name = str(self.compute_device)
        self.logger.report(run_header(self.precision, self._backend_name))
        # Initialization and the initial particles call the same helpers as
        # the run. Their routine records stay out of the log until the first
        # step output; the first requested diagnostics describe the state.
        self.logger.muted = True
        self.compute_dtype = ti.f64 if self.precision == "f64" else ti.f32
        self.accumulator_dtype = self.compute_dtype
        self.np_dtype = np.float64 if self.precision == "f64" else np.float32

    def _initialize_particles_and_physics(self, final_setup: Numerics) -> None:
        """Create particle container, physics engine, source fields, background velocity."""
        max_p = final_setup.max_n_particles
        self.particles = Particles(
            max_n_particles=max_p, float_dtype=self.precision, logger=self.logger
        )
        self.physics = PhysicsEngine(
            particle_kernel=self.particle_kernel,
            max_n_particles=max_p,
            accumulator_dtype=self.accumulator_dtype,
        )

        if hasattr(self.induction, "bind"):
            self.induction.bind(
                self.physics,
                kernel=make_vortex_kernel(self.particle_kernel),
            )
        # Target diagnostics and coupling queries must use the same selected
        # induction method as RK particle stages.  PhysicsBase keeps a
        # direct fallback for standalone users, but a solver-owned backend is
        # used here (including FMM).
        self.physics.induction = self.induction
        if hasattr(self.induction, "estimated_workspace_bytes"):
            self.fmm_workspace_bytes = self.induction.estimated_workspace_bytes(max_p)
        else:
            self.fmm_workspace_bytes = None
        self.integrator = RungeKutta(
            tableau=self.integrator_tableau,
            max_n_particles=max_p,
            dtype=self.accumulator_dtype,
        )
        self._stage_providers = [ParticleExternalStageContribution(self.particles)]
        from ..physics.induction.slip_slab import SlipSlabInduction

        viscous = final_setup.viscous
        self.physics.lattice.core_radius_ratio = float(viscous.core_radius_ratio)
        # Grid diffusion on GPU uses a fixed workspace to avoid repeated allocation.
        vpm_bounds = final_setup.domain_bounds
        scheme = viscous.scheme
        is_grid_diffusion = scheme in {"dvh", "gbd"}
        fixed_grid_required = (
            self.compute_device in {"metal", "vulkan", "cuda"} and is_grid_diffusion
        )
        # The slip-slab image stencil fixes the z-plane phase even on CPU.
        # CPU may still grow the workspace; only its lattice origin is fixed.
        slip_slab_grid = is_grid_diffusion and isinstance(self.induction, SlipSlabInduction)
        if fixed_grid_required:
            self.physics.lattice.require_fixed_grid_allocation(True)
        if fixed_grid_required or slip_slab_grid:
            if scheme == "dvh":
                _grid_h = viscous.dvh_grid_spacing
                _grid_pad = viscous.dvh_domain_padding
            else:
                _grid_h = viscous.gbd_grid_spacing
                _grid_pad = viscous.gbd_domain_padding

            if vpm_bounds is None:
                raise ValueError(
                    "Fixed-lattice DVH/GBD requires domain_bounds so the diffusion "
                    "grid can be allocated once."
                )
            if _grid_h is None:
                raise ValueError(
                    "Fixed-lattice DVH/GBD requires a positive grid spacing so the "
                    "fixed diffusion grid can be pre-allocated."
                )

            self.physics.lattice.configure_max_grid_extent(vpm_bounds, _grid_h, _grid_pad)
        self.solenoidal_projection = (
            SolenoidalLatticeProjection()
            if scheme == "gbd" and viscous.gbd_solenoidal_projection
            else None
        )
        self.particles.set_freestream_velocity(np.array(self.numerics.freestream_velocity))

    def _initialize_turbulence(self, final_setup: Numerics) -> None:
        """Initialize LES turbulence, stretching settings, and diagnostics."""
        max_p = final_setup.max_n_particles
        self.turbulence_model = None
        self.spectral_closure = None
        if final_setup.turbulence.model == "spectral_cusp":
            self.spectral_closure = SpectralCuspClosure(
                kolmogorov_constant=final_setup.turbulence.kolmogorov_constant,
                cutoff_wavenumber=final_setup.turbulence.cutoff_wavenumber,
                use_cuda=self.compute_device == "cuda",
            )
        elif self.flow_model == "les":
            from ..physics.induction.slip_slab import SlipSlabInduction

            if final_setup.turbulence.subgrid_transport == "strength_exchange" and isinstance(
                final_setup.induction, SlipSlabInduction
            ):
                raise ValueError(
                    "subgrid_transport='strength_exchange' has no slip-plane images; "
                    "use 'core_spreading' between slip planes"
                )
            self.turbulence_model = ParticleLES(
                model_name=final_setup.turbulence.model,
                max_n_particles=max_p,
                particle_kernel=self.particle_kernel,
                smagorinsky_coefficient=final_setup.turbulence.smagorinsky_coefficient,
                filter_width=final_setup.turbulence.filter_width,
                sigma_coefficient=final_setup.turbulence.sigma_coefficient,
                consistency_angle=final_setup.turbulence.consistency_angle,
                subgrid_transport=final_setup.turbulence.subgrid_transport,
                accumulator_dtype=self.accumulator_dtype,
                particle_dtype=self.compute_dtype,
            )
        self.field_diagnostics = ParticleFieldEvaluation(
            particle_kernel=self.particle_kernel,
            max_n_particles=max_p,
            accumulator_dtype=self.accumulator_dtype,
        )
        self._flow_integrals: dict = {}
        self._particle_resolution_metrics: dict = {}
        self._accepted_state_metrics: ParticleStateMetrics | None = None
        # Boxes (xmin, xmax, ymin, ymax, zmin, zmax) in m: particles inside do not set the
        # Lagrangian CFL number. The coupler excludes the particles the FVM replaces.
        self.strain_excluded_bounds: tuple[tuple[float, ...], ...] | None = None
        self._particles_removed_this_step = 0
        self._vortex_strength_removed_this_step = np.zeros(3, dtype=self.np_dtype)

    def _initialize_solvers(self, final_setup: Numerics) -> None:
        """Initialize the stabilization master and the optional sub-solvers."""

        # Accepted times and vortex centroids, read by the flow-integral log.
        self._diagnostics_history: dict = {"time": [], "vortex_centroid": []}
        stabilization_state = StabilizationStepState(
            step=self.step,
            time=self.time,
            time_step_size=self.time_step_size,
            vortex_strength_removed=np.zeros(3, dtype=self.np_dtype),
        )
        self.stabilization = StabilizationManager(
            StabilizationContext(
                particles=self.particles,
                physics=self.physics,
                field_diagnostics=self.field_diagnostics,
                config=self.numerics.stabilization,
                compute_dtype=self.compute_dtype,
                np_dtype=self.np_dtype,
                flow_model=self.flow_model,
                state=stabilization_state,
                mutations=SolverParticleMutations(self, stabilization_state),
                logger=self.logger,
                slip_slab=getattr(self.physics, "_slip_slab_bounds", None),
            )
        )
        active = self.stabilization.active_mechanisms()
        if active:
            self.logger.section(
                "stabilization",
                *(("  " + mechanism, "active") for mechanism in active),
            )
        self._initialize_vlm_solver(final_setup)
        # Phase timing waits for the device around every phase, so it is
        # opt-in; the whole-step timer of a normal run adds no device wait.
        self.profiler = Profiler(enabled=self.case.logging.detailed_timing, synchronize=ti.sync)
        self.physics.lattice.profiler = self.profiler
        self.physics.lattice.logger = self.logger
        self.stage_rhs = StageRHS(
            self.induction,
            providers=tuple(self._stage_providers),
            # Core spreading changes every core radius before the first stage.
            retain_strength_rate=self.viscous_scheme != "cs",
            profiler=self.profiler,
        )
        self.wall_time = 0.0
        self.stepper = EvolutionStepper(self)

    def _initialize_vlm_coupling(self) -> None:
        """Configure VLM solver coupling: mesh generation, force config, stability check."""
        self.vlm_solver._wake_kernel = self.induction.kernel
        self.vlm_solver.generate_mesh()
        if getattr(self.vlm_solver, "lattice", None) is not None:
            self.logger.section("vlm", ("panels", f"{self.vlm_solver.lattice.n_panels:,}"))
            self.vlm_solver.check_coupling_stability(
                self.time_step_size, self.numerics.freestream_velocity
            )

    def _initialize_vlm_solver(self, final_setup: Numerics) -> None:
        """Initialize the optional VLM solver with error handling."""
        if final_setup.vlm is None:
            self.vlm_solver = None
        else:
            self._require_consistent_molecular_viscosity(final_setup.viscous, final_setup.vlm)
            from ..vlm.solver import VLMSolver

            self.vlm_solver = VLMSolver(final_setup.vlm, self.logger)

        if self.vlm_solver is not None:
            self._initialize_vlm_coupling()
            # The VLM circulation is solved in the accepted-step coupling
            # phase. Its induced field is sampled at each temporary
            # particle RK position through the common StageRHS boundary.
            # Construct the immutable provider tuple only after optional
            # solvers have initialized.
            self._stage_providers.insert(
                0,
                VLMStageContribution(self.vlm_solver, self.physics),
            )

    @staticmethod
    def _require_consistent_molecular_viscosity(
        viscous_cfg: "ViscousConfig", vlm_setup: "VLMConfig"
    ) -> None:
        """Require one molecular kinematic viscosity in an attached VPM/VLM system."""
        scheme = viscous_cfg.scheme
        if scheme == "none":
            vpm_kinematic_viscosity = 0.0
        else:
            configured_kinematic_viscosity = viscous_cfg.kinematic_viscosity
            if configured_kinematic_viscosity is None:
                raise ValueError(
                    f"VPM viscous scheme {scheme!r} requires kinematic_viscosity "
                    "when a VLM setup is attached"
                )
            vpm_kinematic_viscosity = float(configured_kinematic_viscosity)
        vlm_kinematic_viscosity = float(vlm_setup.kinematic_viscosity)
        if not np.isclose(
            vlm_kinematic_viscosity,
            vpm_kinematic_viscosity,
            rtol=0.0,
            atol=1e-15,
        ):
            raise ValueError(
                "Molecular kinematic viscosity mismatch: the VPM viscous "
                f"scheme {scheme!r} uses {vpm_kinematic_viscosity!r} m^2/s while the "
                "attached VLM setup uses "
                f"{vlm_kinematic_viscosity!r} m^2/s. The VPM owns the "
                "molecular viscosity in a coupled VLM+VPM run; configure both "
                "to the same value."
            )

    # Basic protocol

    # Time stepping

    def advance(self, *, defer_output: bool = False, time: float | None = None) -> None:
        """Advance the VPM solution by one accepted time step.

        The step (velocity and gradient preparation, coupled advection and
        stretching, viscous diffusion, operator splitting and in-step
        stabilization) is carried out by
        :class:`~openonda.vpm.core.evolution.EvolutionStepper`.

        Parameters
        ----------
        defer_output : bool, default=False
            Leave the scheduled output of the accepted state to a later
            :meth:`write_step_output` call, so a coupled driver can first
            transfer vorticity to the particles.
        time : float or None, default=None
            Accepted time in s imposed by a driver that owns the clock, such
            as the FVM--VPM coupler. It needs a solver built with
            ``external_clock=True`` and must equal ``time + time_step_size``
            to round-off. ``None`` advances by the configured time-step size.

        Raises
        ------
        ValueError
            If ``time`` is given to a solver built without ``external_clock``,
            or does not follow the current time by one time step.
        RuntimeError
            If an earlier step failed and no backup was loaded since.
        Exception
            Any numerical, particle-state or stabilization error. The failed
            step leaves the solver unusable until :meth:`load_backup`.
        """
        self._ensure_evolution_usable()
        if not self._initial_conditions_built:
            self._build_initial_conditions()
        self.logger.muted = False
        if time is not None and not self._external_clock:
            raise ValueError("An imposed time requires VPMSolver(case, external_clock=True)")
        if time is not None and not math.isclose(
            time,
            self.time + self.time_step_size,
            rel_tol=1.0e-12,
            abs_tol=1.0e-9 * self.time_step_size,
        ):
            raise ValueError(
                f"Imposed VPM time {time!r} s does not follow {self.time!r} s "
                f"by one time step of {self.time_step_size!r} s"
            )
        if not self._run_started:
            self.run_status = RunStatus.running
        step_clock = Stopwatch()
        try:
            self.stepper.advance(defer_output=defer_output, time=time)
        except BaseException as error:
            self._mark_evolution_failure(error)
            raise
        if time is not None:
            # An imposed time is the base of a later fixed step.
            self._anchor_clock()
        if defer_output:
            return
        self.write_step_output()
        self.logger.section("STEP TIMING", ("Step total wall time", step_clock.seconds, "s"))

    def _ensure_evolution_usable(self) -> None:
        """Reject continuation after a physical step has failed."""
        if self._evolution_failure is not None:
            raise RuntimeError(
                "VPMSolver is terminally invalid after a failed physical step; "
                "load an accepted backup before continuing"
            ) from self._evolution_failure

    def _mark_evolution_failure(self, error: BaseException) -> None:
        """Latch the first physical-step failure; device fields may be partly updated."""
        if self._evolution_failure is None:
            self._evolution_failure = error
        self.run_status = RunStatus.failed
        self.run_failure = self._evolution_failure

    def start_from(self, selection: str | Path = "latest") -> bool:
        """Select the initial state or a backup for the run to ``RunPlan.end_time``.

        Parameters
        ----------
        selection : str or pathlib.Path, default='latest'
            ``"initial"`` starts from the initial conditions and deletes the
            output series of an earlier run; ``"latest"`` loads the latest
            backup in ``solution/vpm/`` or starts from the initial
            conditions when there is none; a path loads that backup.

        Returns
        -------
        bool
            Whether a backup was loaded.
        """
        if selection == "initial":
            if self.step or self.time or self._restart_loaded:
                raise RuntimeError(
                    "start_from='initial' requires a new VPM solver before advancing"
                )
            reset_run_outputs(
                self.solution_dir,
                kind="vpm",
                samples_dir=self.samples_dir,
                sample_names=self._sampler_manager.owned_names(),
            )
            self._sampler_manager.rewind(-1.0e100)
            return False
        path = select_vpm_backup(selection, self.solution_dir)
        if path is None:
            return self.start_from("initial")
        self.load_backup(path)
        self.logger.event(f"Continuing from {path}: step {self.step}, time {self.time:.17g}")
        return True

    def _start_run_from(self, selection: str | Path) -> None:
        """Start a run without loading a backup that already reaches the end time."""
        if selection == "initial":
            self.start_from(selection)
            return
        path = select_vpm_backup(selection, self.solution_dir)
        if path is None:
            self.start_from("initial")
            return
        step, time, _, particle_count = _BackupIO.inspect(self, path)
        if step < self.case.n_steps:
            self.start_from(path)
            return
        self.step = step
        self.time = time
        self._anchor_clock()
        self._restart_loaded = True
        self._restart_terminal_noop = True
        self._restart_particle_count = particle_count
        self._initial_conditions_built = True
        self.logger.event(
            f"Latest backup already reaches the end time: step {step}, time {time:.17g}"
        )

    def run(self, *, start_from: str | Path | None = "latest") -> None:
        """Run the case to ``RunPlan.end_time`` and close the solver.

        Parameters
        ----------
        start_from : {'latest', 'initial'}, path or None, default='latest'
            Passed to :meth:`start_from` before the run; ``None`` continues
            from the state in memory.

        Raises
        ------
        RuntimeError
            If the run was already started or an earlier step failed.
        Exception
            The first evolution, output or finalization error, re-raised
            after the failed-event output, the metadata and the cleanup. With
            ``state_limit_action="stop"`` a nonfinite state instead returns
            with ``run_status="unstable"`` and the error in ``run_failure``.
        """
        if self._external_clock:
            raise RuntimeError("An externally clocked VPM solver must be advanced by its driver")
        if self._run_started:
            raise RuntimeError("VPMSolver.run() may be called only once")
        try:
            if start_from is not None:
                self._start_run_from(start_from)
        except BaseException as error:
            self.close(failure=error)
            raise
        self._run_started = True
        self._run_clock = Stopwatch()
        self._run_initial_step = self.step
        self._run_initial_time = self.time
        remaining_steps = max(0, self._run_final_step - self.step)
        if self._restart_terminal_noop:
            self._run_clock.stop()
            self.run_status = RunStatus.completed
            try:
                run_finished(self, self.run_status)
                write_vpm_metadata(self, status=self.run_status)
            finally:
                self.close()
            return
        status = RunStatus.failed
        failure: BaseException | None = None
        primary_failure: BaseException | None = None
        try:
            self._build_initial_conditions()
            if start_from is not None and not self._restart_loaded:
                self.save_backup()
            self.run_status = RunStatus.running
            write_vpm_metadata(self, status=self.run_status)
            self.logger.muted = False
            state_limit_failure = None
            invalid_state_failure = None
            if self._restart_loaded:
                self._sampler_manager.resume()
            self.write_initial_output()
            for _ in range(remaining_steps):
                try:
                    self.advance()
                except ParticleStateError as error:
                    if self.case.run.state_limit_action == "raise":
                        raise
                    if not error.restartable:
                        invalid_state_failure = error
                    else:
                        state_limit_failure = error
                    break
            if invalid_state_failure is not None:
                status = RunStatus.unstable
                failure = invalid_state_failure
            else:
                try:
                    self._write_end_backup()
                    self.write_final_output()
                except Exception as output_failure:
                    if state_limit_failure is not None:
                        output_failure.add_note(
                            "Final output followed a particle state limit: "
                            f"{type(state_limit_failure).__name__}: {state_limit_failure}"
                        )
                    raise
                if state_limit_failure is not None:
                    status = RunStatus.resolution_lost
                    failure = state_limit_failure
                else:
                    status = RunStatus.completed
        except BaseException as error:
            failure = error
            primary_failure = error
            with suppress(BaseException):
                self._sampler_manager.dispatch(OutputEvent.failed)
        finally:
            self._run_clock.stop()
            if primary_failure is not None:
                status = RunStatus.failed
                failure = primary_failure
            self.run_status = status
            self.run_failure = failure
            for finalizer in (
                lambda: run_finished(self, self.run_status, self.run_failure),
                lambda: write_vpm_metadata(self, status=self.run_status),
                self.close,
            ):
                try:
                    finalizer()
                except BaseException as finalizer_error:
                    if primary_failure is None:
                        primary_failure = finalizer_error
                        self.run_status = RunStatus.failed
                        self.run_failure = finalizer_error

        if primary_failure is not None:
            raise primary_failure.with_traceback(primary_failure.__traceback__)

    def write_initial_output(self) -> None:
        """Write the initial particle frame and the samplers with ``initial=True``.

        :meth:`run` calls this before the first step; a driver that advances
        the solver itself, such as the FVM--VPM coupler, calls it once at the
        start of a fresh run. Both follow ``RunPlan.initial_output``. A state
        restored from a backup already has its initial output, so the call
        does nothing then.
        """
        if self._restart_loaded or not self.case.run.initial_output:
            return
        self._ensure_evolution_usable()
        self._build_initial_conditions()
        self.logger.muted = False
        self._refresh_diagnostics_for_output()
        self.write_vtk()
        self._sampler_manager.dispatch(OutputEvent.initial)

    def write_final_output(self) -> None:
        """Write the final particle frame and the final-only samplers.

        :meth:`run` calls this after the last step; a driver that advances
        the solver itself calls it once at the end of the run. Both follow
        ``RunPlan.final_output``; a frame already written for this state is
        not repeated. A run stopped before its final time also writes every
        sampler that has no sample of the last state.
        """
        if not self.case.run.final_output:
            return
        self._refresh_diagnostics_for_output()
        if self._last_frame_state != self._state_key():
            self.write_vtk()
        if self._reached_end_time():
            self._sampler_manager.dispatch(OutputEvent.final)
        else:
            self._sampler_manager.write_all(OutputEvent.final)

    def _reached_end_time(self) -> bool:
        """Return whether the accepted state is the end of the run.

        A fixed-step solver is at its end after the whole number of steps that
        :meth:`RunPlan.n_steps` accepts for ``end_time``; an externally clocked one
        when its time reaches ``end_time`` to the time tolerance.
        """
        if not self._external_clock:
            return self.step >= self._run_final_step
        final_time = float(self.case.run.end_time)
        return self.time >= final_time - time_tolerance(final_time)

    def _write_end_backup(self) -> None:
        """Write the end-of-run backup unless the accepted state already has one."""
        if not self.case.backup.write_at_end or self._last_backup_state == self._state_key():
            return
        self._refresh_diagnostics_for_output()
        self._velocity_valid_for = self._velocity_key()
        self.save_backup()

    def _build_initial_conditions(self) -> None:
        """Build each declarative initial condition exactly once."""
        if self._initial_conditions_built:
            return
        for initial_condition in self.case.initial_conditions:
            particles = initial_condition.build()
            self.add_vortex_particles(
                position=particles.position,
                velocity=particles.velocity,
                vortex_strength=particles.vortex_strength,
                core_radius=particles.core_radius,
                particle_volume=particles.particle_volume,
                kinematic_viscosity=particles.kinematic_viscosity,
                group_id=particles.group_id,
                zone_id=particles.zone_id,
            )
        self._n_initial_particles = int(self.particles.n_particles_total)
        self._initial_conditions_built = True

    def flush_output(self) -> None:
        """Finish pending file output; VPM output is written synchronously."""
        self.logger.flush()

    @property
    def state_revision(self) -> int:
        """Revision of the particle source state; it changes with every particle update."""
        return int(self.particles.state_revision)

    def close(self, *, failure: BaseException | None = None) -> None:
        """Record the final metadata of manual stepping and release resources.

        A solver advanced without :meth:`run` records ``completed`` when it
        reached the end time and ``partial`` otherwise. While an exception
        ``failure`` unwinds, cleanup errors are attached to it as notes.
        """
        if self._closed:
            return
        primary_failure: BaseException | None = None
        # Only the slab's private CuPy solver is ours; never close a shared induction.
        from ..physics.induction.slip_slab import SlipSlabInduction

        mesh_failure = self._mesh_cleanup_failure
        if mesh_failure is None and type(self.induction) is SlipSlabInduction:
            try:
                self.induction.close_mesh_session()
            except BaseException as error:
                mesh_failure = self._mesh_cleanup_failure = error
        if mesh_failure is not None:
            primary_failure = mesh_failure
        try:
            self.stage_rhs.close()
        except BaseException as error:
            if primary_failure is None:
                primary_failure = error
        for buffer in self._particle_snapshot_buffers.values():
            try:
                buffer.destroy()
            except BaseException as error:
                if primary_failure is None:
                    primary_failure = error
        self._particle_snapshot_buffers = {}
        if not self._run_started and self._initial_conditions_built:
            status = RunStatus.completed if self._reached_end_time() else RunStatus.partial
            if self._evolution_failure is not None or mesh_failure is not None:
                status = RunStatus.failed
            self.run_status = status
            try:
                write_vpm_metadata(self, status=status)
            except BaseException as error:
                if primary_failure is None:
                    primary_failure = error
        self.logger.close()
        if self._backend_claimed and mesh_failure is None:
            try:
                reset_taichi_backend(solver=self)
                self._backend_claimed = False
            except BaseException as error:
                if primary_failure is None:
                    primary_failure = error
        if primary_failure is not None:
            if failure is not None:
                failure.add_note(f"VPM cleanup also failed: {primary_failure!r}")
                return
            raise primary_failure
        self._closed = True

    def _velocity_key(self) -> tuple:
        """Identify the accepted state and transport field of the particle velocity."""
        return (
            self.step,
            self.time,
            self.particles.state_revision,
            self.particles.n_particles_total,
            tuple(self.freestream_velocity),
            id(self.stage_rhs),
            id(self.stage_rhs.position_guard),
            tuple(id(provider) for provider in self.stage_rhs.providers),
        )

    def _refresh_particle_diagnostic_fields(self) -> None:
        """Refresh derived diagnostic fields without changing future evolution."""
        if self.particles.n_particles_total:
            self.stepper._update_velocity_and_gradients()
            self.stepper._update_les_state()

    def _check_accepted_particle_state(self) -> None:
        """Refresh and validate diagnostics for one accepted physical state."""
        self._refresh_particle_diagnostic_fields()
        if self.state_limits.divergence.maximum is not None or (
            self.state_limits.misalignment.maximum_degrees is not None
        ):
            self._update_particle_resolution_metrics()
        self._accepted_state_metrics = check_accepted_particle_state(
            limits=self.state_limits,
            step=self.step,
            time_step_size=self.time_step_size,
            state=measure_particle_state(
                self.particles, self.time_step_size, self.strain_excluded_bounds
            ),
            resolution=self._particle_resolution_metrics,
            previous=self._accepted_state_metrics,
        )
        self.stabilization.lagrangian_cfl = self._accepted_state_metrics.strain_increment_infinity
        self._velocity_valid_for = self._velocity_key()

    def maximum_lagrangian_strain_rate(self) -> float:
        """Return the maximum infinity-norm strain rate of the current particle field, in 1/s."""
        self._refresh_particle_diagnostic_fields()
        strain_rate = float(
            measure_particle_state(
                self.particles, 1.0, self.strain_excluded_bounds
            ).metrics.strain_increment_infinity
        )
        if not np.isfinite(strain_rate) or strain_rate < 0.0:
            raise NonFiniteSolutionError("The particle strain-rate field is not finite")
        return strain_rate

    def _refresh_diagnostics_for_output(self) -> None:
        """Refresh dependencies before framework-owned diagnostics are sampled."""
        self._refresh_particle_diagnostic_fields()
        self._update_all_flow_integrals()

    def write_step_output(self) -> None:
        """Check the accepted state and write its scheduled output.

        Writes the backup, the particle frame and the samplers whose
        schedules select the accepted step. :meth:`advance` calls this unless
        ``defer_output=True``; a coupled driver calls it after exchanging the
        particle cloud.
        """
        self.logger.muted = False
        phase_clock = Stopwatch()
        with remainder_section(self.profiler, "Particle state checks"):
            self._check_accepted_particle_state()
        state_check_seconds = phase_clock.lap()
        with remainder_section(self.profiler, "Flow and VLM diagnostics"):
            if self.vlm_solver is not None:
                VLMDiagnostics.record_vlm_diagnostics(
                    self.vlm_solver,
                    self.particles,
                    self.particle_vortex_strength,
                    self.time,
                    self.samples_dir,
                )
                VLMLoadingDistribution.record_loading_distributions(
                    self.vlm_solver, self.time, self.samples_dir
                )
            if self._sampler_manager.flow_integrals_due():
                self._update_all_flow_integrals()
        diagnostics_seconds = phase_clock.lap()
        with remainder_section(self.profiler, "Backups, frames and samplers"):
            backup_schedule = self.case.backup.schedule
            if backup_schedule is not None and backup_schedule.is_due(
                self.step, self.time, self.time_step_size
            ):
                self.save_backup()
            frame_schedule = self.case.output.schedule
            if frame_schedule is not None and frame_schedule.is_due(
                self.step, self.time, self.time_step_size
            ):
                self.write_vtk()
            # Only the backup of this event reuses the velocity of its diagnostics.
            self._velocity_valid_for = None
            self._sampler_manager.dispatch(OutputEvent.accepted_step)
        self.logger.section(
            "OUTPUT TIMING",
            ("Particle state checks", state_check_seconds, "s"),
            ("Flow and VLM diagnostics", diagnostics_seconds, "s"),
            ("Backups, frames and samplers", phase_clock.lap(), "s"),
        )

    @property
    def phase_seconds(self) -> dict[str, float]:
        """Wall time in s of each phase since the last step started.

        The phases are those of the step and of the output written after it.
        They do not overlap: a phase that contains another holds only its
        own part, so the sum is at most the wall time spent in the solver.

        ``"Induction"`` is the velocity, velocity-gradient and stretching
        evaluation of the particles. ``"Wall projection"`` moves particles
        out of the walls at the Runge--Kutta stages and at the accepted
        state. ``"Runge-Kutta update"`` builds and combines the stages and
        ``"Wall retry snapshot"`` downloads the state a rejected wall
        correction restores. GBD splits into ``"GBD remeshing"`` (particles
        to the grid), ``"GBD wall correction"``, ``"GBD diffusion"`` (the
        grid Laplacian), ``"GBD grid download"`` and ``"GBD moment
        redistribution"`` (the moments of the removed weak nodes);
        ``"Viscous diffusion"`` is the rest of the diffusion operator, such
        as the particle downloads and the regeneration of the particles, and
        all of it for the other schemes. ``"Particle replacement"`` uploads
        the regenerated particles. The stabilization workers, ``"Strain-rate
        tensor"``, ``"LES update"``, ``"LES strength exchange"``,
        ``"External stage rates"`` and ``"VLM coupling"`` have their own
        phases, and the output has ``"Particle state
        checks"``, ``"Flow and VLM diagnostics"`` and ``"Backups, frames
        and samplers"``.

        Empty unless :attr:`LoggingConfig.detailed_timing` is set, because
        each phase waits for the compute device before and after it.
        """
        return self.profiler.step_sections

    def _state_key(self) -> tuple[int, float, int]:
        """Identify the accepted state for once-only output."""
        return (self.step, self.time, self.particles.state_revision)

    @property
    def time_step_size(self) -> float:
        """Fixed time-step size in s."""
        return self._time_step_size

    @time_step_size.setter
    def time_step_size(self, value: float) -> None:
        """Set the time-step size and start the fixed-step clock at the current state."""
        self._time_step_size = value
        self._anchor_clock()

    def _anchor_clock(self) -> None:
        """Make the current accepted state the base of the fixed-step clock."""
        self._clock_base_step = self.step
        self._clock_base_time = self.time

    def _clock_time(self, step: int) -> float:
        """Return the time in s of accepted ``step`` on the fixed-step clock.

        The time is computed from the base state rather than summed step by
        step, so its error is one rounding at every step.
        """
        return self._clock_base_time + (step - self._clock_base_step) * self._time_step_size

    @property
    def elapsed_wall_time(self) -> float:
        """Wall-clock seconds of the run, or of the evolution steps without a run."""
        return self.wall_time if self._run_clock is None else self._run_clock.seconds

    # Particle properties
    @property
    def particle_position(self) -> np.ndarray:
        """Return a copy of active particle positions, shape ``(N, 3)`` in m."""
        return self.particles.position_cpu()

    @property
    def particle_group_id(self) -> np.ndarray:
        """Return int32 particle-group labels, shape ``(N,)``."""
        return self.particles.group_id_cpu()

    @property
    def freestream_velocity(self) -> np.ndarray:
        """Return the uniform background velocity, shape ``(3,)`` in m/s."""
        return self.particles.velocity_background_cpu()

    @property
    def particle_vortex_strength(self) -> np.ndarray:
        """Return particle vortex strengths ``Gamma = omega * V``, shape ``(N, 3)`` in m³/s."""
        return self.particles.vortex_strength_cpu()

    # Flow diagnostics
    def _update_all_flow_integrals(self) -> None:
        """Recompute flow integrals and associated diagnostic histories."""
        self._flow_integrals = self.field_diagnostics.compute_flow_integrals(
            self.particles, self.time
        )
        self._update_particle_resolution_metrics()
        self._record_vortex_centroid_history()
        self._record_time_history()

    def _update_particle_resolution_metrics(self) -> None:
        """Refresh field quality, confirming sampled limit crossings at all blobs.

        A bounded spatial sample is sufficient for routine monitoring, but its
        composition can change as particles move. Before a sampled divergence
        or alignment value can stop the run, evaluate the same metric over every
        distinct blob, retaining the configured threshold and physical state.
        """
        if self.particles.n_particles_total == 0:
            self._particle_resolution_metrics = {}
            return
        # The stored particle vorticity is only reconstructed for backups and
        # initially contains Gamma/V. It is not an accepted-step field. Use
        # curl(u) from the freshly evaluated Jacobian, as the P-relaxation
        # operator does; otherwise the particle-state stop depends on backup cadence.
        vorticity = self.particles.velocity_curl_cpu()
        position = self.particle_position
        strength = self.particle_vortex_strength
        core_radius = self.particles.core_radius_cpu()
        metrics = particle_resolution_metrics(
            position,
            strength,
            core_radius,
            vorticity=vorticity,
            slip_slab=getattr(self.physics, "_slip_slab_bounds", None),
        )
        limits = (
            ("vorticity_divergence_error", self.state_limits.divergence.maximum),
            (
                "vortex_strength_misalignment_degrees",
                self.state_limits.misalignment.maximum_degrees,
            ),
        )
        if any(
            maximum is not None and (not np.isfinite(metrics[name]) or metrics[name] > maximum)
            for name, maximum in limits
        ):
            metrics = particle_resolution_metrics(
                position,
                strength,
                core_radius,
                vorticity=vorticity,
                sample_all=True,
                slip_slab=getattr(self.physics, "_slip_slab_bounds", None),
            )
        self._particle_resolution_metrics = metrics

    def _record_vortex_centroid_history(self) -> None:
        """Record the vortex-strength-magnitude-weighted particle centroid."""
        ParticleFieldEvaluation.record_vortex_centroid_history(
            self._diagnostics_history,
            self.particle_position,
            self.particle_vortex_strength,
        )

    def _record_time_history(self) -> None:
        """Append the accepted time once per step."""
        times = self._diagnostics_history["time"]
        if not times or times[-1] != self.time:
            times.append(self.time)

    @property
    def total_kinetic_energy(self) -> float:
        """Return total kinetic energy per unit density in m⁵/s²."""
        return self._flow_integrals.get("total_kinetic_energy", 0.0)

    @property
    def total_helicity(self) -> float:
        """Return ``integral(u · omega) dV`` per unit density, in m⁴/s²."""
        return self._flow_integrals.get("total_helicity", 0.0)

    @property
    def total_enstrophy(self) -> float:
        """Return ``integral(|omega|²) dV``, in m³/s² (no one-half factor)."""
        return self._flow_integrals.get("total_enstrophy", 0.0)

    @property
    def viscous_kinetic_energy_rate(self) -> float:
        """Return the modeled viscous rate of energy per density, in m⁵/s³."""
        return self._flow_integrals.get("viscous_kinetic_energy_rate", 0.0)

    @property
    def kinetic_energy_rate(self) -> float:
        """Return the signed energy-per-density rate, in m⁵/s³.

        Usually this is a backward difference. Transitions without compatible
        energy history use a viscous estimate; see ``kinetic_energy_rate_source``.
        """
        return self._flow_integrals.get("kinetic_energy_rate", 0.0)

    @property
    def net_vortex_strength(self) -> np.ndarray:
        """Return the total vortex strength, shape ``(3,)`` in m³/s."""
        return self._flow_integrals.get("net_vortex_strength", np.array([0.0, 0.0, 0.0]))

    @property
    def vortex_strength_magnitude_sum(self) -> float:
        """Return the sum of particle-strength vector magnitudes in m³/s."""
        return self._flow_integrals.get("vortex_strength_magnitude_sum", 0.0)

    @property
    def total_linear_impulse(self) -> np.ndarray:
        """Return ``0.5 * sum(x × Gamma)``, shape ``(3,)`` in m⁴/s."""
        # This linear moment needs neither a quadratic field integral nor an
        # energy-history derivative (which may be undefined at a grid switch).
        return 0.5 * np.cross(
            self.particle_position.astype(np.float64),
            self.particle_vortex_strength.astype(np.float64),
        ).sum(axis=0, dtype=np.float64)

    # Particle management
    def _configured_kinematic_viscosity(self, count: int) -> np.ndarray:
        """Broadcast the configured scalar kinematic viscosity to ``count`` particles."""
        kinematic_viscosity = getattr(self._viscous_config, "kinematic_viscosity", None)
        if kinematic_viscosity is not None and kinematic_viscosity > 0:
            return np.full(count, kinematic_viscosity, dtype=self.np_dtype)
        raise ValueError(
            "kinematic_viscosity is required. Either configure "
            "ViscousConfig.kinematic_viscosity or pass an explicit array."
        )

    def remove_all_particles(self) -> None:
        """Remove the complete active cloud and record removed-strength diagnostics.

        Side Effects
        ------------
        Compacts device particle fields, resets the stabilization refinement
        reference, and stores the removed count and vortex strength for diagnostics.
        """
        self._particles_removed_this_step = len(self.particles)
        self._vortex_strength_removed_this_step = self.particles.net_vortex_strength()
        self.stabilization.on_removal(remove_all=True)
        self.particles.remove_all_particles()

    def add_vortex_particles(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        vortex_strength: np.ndarray,
        core_radius: np.ndarray,
        particle_volume: np.ndarray,
        kinematic_viscosity: np.ndarray | None = None,
        eddy_viscosity: np.ndarray | None = None,
        group_id: np.ndarray | None = None,
        zone_id: np.ndarray | None = None,
        velocity_gradient: np.ndarray | None = None,
    ) -> None:
        """Append vortex particles to the active cloud.

        ``position``, ``velocity``, and ``vortex_strength`` have shape
        ``(N, 3)``; ``core_radius`` and ``particle_volume`` have shape ``(N,)``.
        Molecular viscosity may be omitted when it is defined by the viscous
        configuration.

        Parameters
        ----------
        position, velocity, vortex_strength : numpy.ndarray
            Arrays with shape ``(N, 3)`` in m, m/s, and m³/s.
        core_radius, particle_volume : numpy.ndarray
            Arrays with shape ``(N,)`` in m and m³.
        kinematic_viscosity : numpy.ndarray or None, default=None
            Molecular viscosity with shape ``(N,)`` in m²/s. If omitted, a
            configured scalar viscosity is broadcast to the batch.
        eddy_viscosity : numpy.ndarray or None, default=None
            Optional turbulent viscosity, shape ``(N,)``, in m²/s.
        group_id, zone_id : numpy.ndarray or None, default=None
            Optional integer labels, shape ``(N,)``.
        velocity_gradient : numpy.ndarray or None, default=None
            Optional Jacobian field, shape ``(N, 3, 3)``, in 1/s.

        Raises
        ------
        ValueError
            If viscosity is unavailable, arrays are inconsistent/non-finite,
            or the fixed particle capacity is exceeded.

        Side Effects
        ------------
        Appends and copies the batch into device fields and updates stabilization
        refinement reference/reference totals.
        """
        if kinematic_viscosity is None:
            kinematic_viscosity = self._configured_kinematic_viscosity(len(position))

        start = self.particles.n_particles_total
        self.particles.add_vortex_particles(
            position=position,
            velocity=velocity,
            vortex_strength=vortex_strength,
            core_radius=core_radius,
            particle_volume=particle_volume,
            kinematic_viscosity=kinematic_viscosity,
            eddy_viscosity=eddy_viscosity,
            group_id=group_id,
            zone_id=zone_id,
            velocity_gradient=velocity_gradient,
        )
        magnitude = np.linalg.norm(np.asarray(vortex_strength, dtype=np.float64), axis=1)
        self.stabilization.on_add(
            magnitude,
            particle_volume,
            start,
            loading=self._loading_numerical_state,
        )

    def replace_vortex_particles(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        vortex_strength: np.ndarray,
        core_radius: np.ndarray,
        particle_volume: np.ndarray,
        kinematic_viscosity: np.ndarray | None = None,
        eddy_viscosity: np.ndarray | None = None,
        group_id: np.ndarray | None = None,
        zone_id: np.ndarray | None = None,
        velocity_gradient: np.ndarray | None = None,
        strain_rate: np.ndarray | None = None,
        report_removal: bool = True,
    ) -> None:
        """Replace the active particle cloud in one field-upload operation.

        ``report_removal`` should be set to ``False`` by mechanisms that rebuild
        the cloud in place without representing physical removal (for example
        filament refinement), so the removed-this-step diagnostic counters stay
        untouched.

        Parameters
        ----------
        position, velocity, vortex_strength : numpy.ndarray
            Replacement arrays with shape ``(N, 3)`` in m, m/s, and m³/s.
        core_radius, particle_volume : numpy.ndarray
            Replacement arrays with shape ``(N,)`` in m and m³.
        kinematic_viscosity, eddy_viscosity : numpy.ndarray or None
            Viscosity fields with shape ``(N,)`` in m²/s; molecular viscosity
            may be broadcast from the configured scalar.
        group_id, zone_id : numpy.ndarray or None
            Integer labels with shape ``(N,)``.
        velocity_gradient, strain_rate : numpy.ndarray or None
            Optional tensor fields with shape ``(N, 3, 3)`` in 1/s.
        report_removal : bool, default=True
            Include the old cloud in removal diagnostics and stabilization
            bookkeeping.

        Raises
        ------
        ValueError
            If fields are inconsistent/non-finite or exceed capacity.

        Side Effects
        ------------
        Clears/repopulates active device fields, invalidates caches, and
        notifies stabilization of replacement.
        """
        if report_removal:
            vortex_strength_removed = (
                self.particles.net_vortex_strength()
                if len(self.particles) > 0
                else np.zeros(3, dtype=self.np_dtype)
            )
            self._particles_removed_this_step = len(self.particles)
            self._vortex_strength_removed_this_step = vortex_strength_removed

        if kinematic_viscosity is None:
            kinematic_viscosity = self._configured_kinematic_viscosity(len(position))

        self.particles.replace_from_numpy(
            position=position,
            velocity=velocity,
            vortex_strength=vortex_strength,
            core_radius=core_radius,
            particle_volume=particle_volume,
            kinematic_viscosity=kinematic_viscosity,
            eddy_viscosity=eddy_viscosity,
            group_id=group_id,
            zone_id=zone_id,
            velocity_gradient=velocity_gradient,
            strain_rate=strain_rate,
        )
        magnitude = np.linalg.norm(np.asarray(vortex_strength, dtype=np.float64), axis=1)
        self.stabilization.on_replacement(magnitude, particle_volume)

    def capture_particle_snapshot(self, *, slot: str) -> "ParticleSnapshot":
        """Capture a provisional device state in an explicitly named scratch slot.

        Distinct nested state updates must use different slots. Reusing a slot
        invalidates its previous handle; restoring it then fails before mutation.
        This is not a restart or a cache: every call copies current device state.
        """
        from ..particles.snapshot import ParticleSnapshotBuffer

        if not isinstance(slot, str) or not slot:
            raise ValueError("Particle snapshot slot must be a non-empty string")
        buffers = self._particle_snapshot_buffers
        count = int(self.particles.n_particles_total)
        buffer = buffers.get(slot)
        if buffer is None or buffer.capacity < count:
            # A tenth of headroom: a doubled copy of a large particle set may not fit on the device
            capacity = min(self.particles.capacity, max(1, int(np.ceil(1.1 * count))))
            replacement = ParticleSnapshotBuffer(capacity, self.particles._taichi_dtype)
            if buffer is not None:
                buffer.destroy()
            buffer = buffers[slot] = replacement
        return buffer.capture(
            self.particles,
            prepare_refinement_reference=self.stabilization.reference_vortex_strength is not None,
        )

    def restore_particle_snapshot(self, snapshot: "ParticleSnapshot") -> None:
        """Publish a device rollback with the existing replacement semantics."""
        from ..particles.snapshot import ParticleSnapshot

        if not isinstance(snapshot, ParticleSnapshot):
            raise TypeError("Expected a particle snapshot from this solver")
        snapshot._validate(self.particles)
        refinement_reference = (
            snapshot.prepare_refinement_reference()
            if self.stabilization.reference_vortex_strength is not None
            else None
        )
        snapshot.restore(self.particles)
        if refinement_reference is not None:
            self.stabilization.on_replacement(*refinement_reference)

    @staticmethod
    def _validate_particle_property(
        prop_name: str,
        prop_value: np.ndarray,
        n_particles_total: int,
    ) -> np.ndarray:
        """Validate one standard active-particle field before device upload.

        Parameters
        ----------
        prop_name : str
            Standard field name, used to select the expected trailing shape.
        prop_value : array-like
            Candidate values. It is converted to a NumPy array when needed.
        n_particles_total : int
            Active count defining the leading dimension.

        Returns
        -------
        numpy.ndarray
            The validated array. It is not copied here; the device setter owns
            the subsequent conversion/copy.

        Raises
        ------
        ValueError
            If shape is incompatible or any value is NaN/Inf.
        """
        if not isinstance(prop_value, np.ndarray):
            prop_value = np.asarray(prop_value)

        expected_shape = (n_particles_total, *FIELD_TRAILING_SHAPES[prop_name])

        if prop_value.shape != expected_shape:
            raise ValueError(
                f"Property '{prop_name}' has incorrect shape {prop_value.shape}. "
                f"Expected {expected_shape} for {n_particles_total} particles."
            )
        if not np.all(np.isfinite(prop_value)):
            nan_count = int(np.sum(np.isnan(prop_value)))
            inf_count = int(np.sum(np.isinf(prop_value)))
            raise ValueError(
                f"Property '{prop_name}' contains invalid values: "
                f"{nan_count} NaN, {inf_count} Inf. "
                "Cannot set particle properties with non-finite values."
            )
        return prop_value

    def set_particle_properties(self, **properties: object) -> None:
        """Update standard particle fields after validating shape and finiteness.

        Parameters
        ----------
        **properties : object
            Named active fields such as ``position`` ``(N, 3)`` m,
            ``vortex_strength`` ``(N, 3)`` m³/s, scalar fields ``(N,)``, or
            tensors ``(N, 3, 3)`` in 1/s. Unknown names are rejected.

        Raises
        ------
        ValueError
            If no active particles exist, a property name is invalid, a shape
            is wrong, or any value is non-finite.

        Side Effects
        ------------
        Writes the validated fields in place, increments the source-state
        revision once for the batch, and records an update event.
        """
        if not properties:
            return

        invalid = [name for name in properties if name not in FIELD_TRAILING_SHAPES]
        if invalid:
            raise ValueError(
                f"Invalid property name {invalid[0]!r}. "
                f"Valid properties: {sorted(FIELD_TRAILING_SHAPES)}"
            )

        n_particles_total = self.particles.n_particles_total
        if n_particles_total == 0:
            raise ValueError("Cannot set properties: particle system is empty")

        for field_name, value in properties.items():
            validated = self._validate_particle_property(
                field_name,
                value,
                n_particles_total,
            )
            self.particles.set_field(field_name, validated)

        # Field writes may change Biot--Savart sources.  Publish one revision
        # after the validated batch so every derived particle cache is invalid.
        self.particles.touch_state()

        property_names = list(properties)
        self.logger.section(
            "particle fields updated",
            ("fields", f"{len(property_names):,}"),
            *(("  " + name, "updated") for name in property_names),
        )

    # State and restart

    def load_backup(
        self,
        path: str | Path,
        *,
        time_step_size: float | None = None,
        allowed_config_differences: Collection[str] = (),
        expected_config_differences: Mapping[str, tuple[object, object]] | None = None,
    ) -> None:
        """Restore the state of a backup into this solver.

        Parameters
        ----------
        path : str or pathlib.Path
            HDF5 backup written by :meth:`save_backup`.
        time_step_size : float or None, keyword-only
            Continue with this time-step size in s instead of the stored one.
            Every other numerical setting must match. Not available for DVH,
            whose backup holds no physical-time diffusion history.
        allowed_config_differences : collection[str], default=()
            Numerical configuration paths allowed to differ from the backup.
        expected_config_differences : mapping or None, default=None
            ``path: (stored_value, current_value)`` expectations, required for
            structural changes. Use
            ``config.restart_changes.MISSING_CONFIGURATION_VALUE`` for a
            missing key.

        Notes
        -----
        Replaces the particle fields and the clock, clears an earlier step
        failure, and removes sample rows and frames after the restored time.
        The case configuration is unchanged.
        """
        if time_step_size is not None:
            if isinstance(time_step_size, bool) or not isinstance(time_step_size, Real):
                raise TypeError("time_step_size override must be a real number")
            time_step_size = float(time_step_size)
            if not np.isfinite(time_step_size) or time_step_size <= 0.0:
                raise ValueError("time_step_size override must be finite and positive")
            if self.viscous_scheme == "dvh":
                raise ValueError(
                    "changed time_step_size continuation is not supported for DVH: "
                    "the backup does not retain enough physical-time diffusion history"
                )
        path = Path(path)
        _BackupIO.load(
            self,
            path,
            time_step_size=time_step_size,
            allowed_config_differences=allowed_config_differences,
            expected_config_differences=expected_config_differences,
        )
        # The restored clock is the base of the fixed step.
        self._anchor_clock()
        self._evolution_failure = None
        self.run_failure = None
        self._restart_loaded = True
        self._restart_output_time = self.time
        # The backup replaces the particle state; the initial conditions are never rebuilt.
        self._initial_conditions_built = True
        self._sampler_manager.rewind(self.time)
        rewind_vpm_frames(self.solution_dir, self.step, self.time)
        self._last_backup_state = self._state_key()

    def save_backup(self, path: str | Path | None = None) -> Path:
        """Write the restart state of the current accepted state.

        Parameters
        ----------
        path : str, pathlib.Path or None, default=None
            HDF5 file. ``None`` writes ``solution/vpm/vpm_{step:06d}.h5``
            and the run metadata.

        Returns
        -------
        pathlib.Path
            The backup file.
        """
        default = path is None
        if default:
            path = self.solution_dir / "vpm" / f"vpm_{self.step:06d}.h5"
        backup_clock = Stopwatch()
        self._refresh_backup_particle_fields()
        # Serialization downloads the fields anyway; charge the device work here.
        ti.sync()
        preparation_seconds = backup_clock.lap()
        written = _BackupIO.save(self, path)
        self.logger.section(
            "BACKUP TIMING",
            ("Field preparation", preparation_seconds, "s"),
            ("File writing", backup_clock.lap(), "s"),
        )
        if default:
            self._last_backup_state = self._state_key()
            status = RunStatus.running if self._run_started else RunStatus.partial
            write_vpm_metadata(self, status=status)
        return written

    def write_vtk(self) -> Path:
        """Write the particle and vortex-lattice frames of the current state.

        The particle frame is ``solution/vpm/vpm_{step:06d}.vtu``, listed in
        ``solution/vpm.pvd``; an attached lattice adds
        ``solution/vlm/vlm_{step:06d}.vtp`` and ``solution/vlm.pvd``.

        Returns
        -------
        pathlib.Path
            The particle frame.
        """
        # A frame evaluates its own velocity; only a backup reuses the one of the diagnostics.
        self._velocity_valid_for = None
        self._refresh_backup_particle_fields()
        frame = write_solution_frames(self)
        self._last_frame_state = self._state_key()
        return frame

    def _refresh_backup_particle_fields(self) -> None:
        """Evaluate the transport velocity and vorticity written to backups and frames.

        The velocity that the diagnostics of the current output event
        evaluated is reused once for the backup of that event; otherwise the
        complete transport field (including a bound lattice and external
        sources) is evaluated at the accepted time, without publishing
        stretching or exchange rates.
        """
        n_particles = self.particles.n_particles_total
        reuse_velocity = (
            self._velocity_valid_for is not None
            and self._velocity_valid_for == self._velocity_key()
        )
        self._velocity_valid_for = None
        if n_particles > 0 and not reuse_velocity:
            self.stage_rhs.evaluate(
                StageState(
                    position=self.particles.position,
                    vortex_strength=self.particles.vortex_strength,
                    core_radius=self.particles.core_radius,
                    count=n_particles,
                    time=self.time,
                    accepted_revision=self.particles.state_revision,
                ),
                self.time,
                StageRates(
                    velocity=self.particles.velocity,
                    vortex_strength_rate=self.physics.dstr_dt_temp,
                    strength_rate_enabled=False,
                ),
            )
        self.physics.compute_vorticities(self.particles)

    # Particle updates

    def set_freestream_velocity(self, velocity: list[float] | np.ndarray) -> None:
        """Set framework-owned uniform background velocity state [m/s]."""
        velocity_arr = np.array(velocity, dtype=self.np_dtype)

        if velocity_arr.shape != (3,):
            if velocity_arr.size == 3:
                velocity_arr = velocity_arr.flatten()
            else:
                raise ValueError(
                    f"Background velocity must be a 3D vector, got shape {velocity_arr.shape}"
                )

        self.particles.set_freestream_velocity(velocity_arr)

    # Particle control

    def remove_particles_by_bounds(
        self, bounds: Sequence[float], invert_selection: bool = False
    ) -> int:
        """Remove particles inside or outside an axis-aligned bounding box.

        Set ``invert_selection=True`` to keep particles inside the box and remove
        those outside it. Returns the number removed.

        Parameters
        ----------
        bounds : sequence[float]
            ``[xmin, xmax, ymin, ymax, zmin, zmax]`` in m.
        invert_selection : bool, default=False
            Remove outside rather than inside when true.

        Returns
        -------
        int
            Number of removed active particles.

        Raises
        ------
        ValueError
            If ``bounds`` does not contain six values.
        """
        keep_mask = None
        if self.stabilization.reference_vortex_strength is not None:
            inside = inside_box(self.particles.position_cpu(), bounds)
            keep_mask = inside if invert_selection else ~inside

        n_removed = self.particles.remove_particles_by_bounds(
            bounds, invert_selection=invert_selection
        )

        if n_removed > 0:
            self.stabilization.on_removal(keep_mask=keep_mask)
            xmin, xmax, ymin, ymax, zmin, zmax = bounds
            action = "outside" if invert_selection else "inside"
            self.logger.section(
                "particles removed",
                ("particles", f"{n_removed:,}"),
                ("region", f"{action} box"),
                ("bounds, x", f"[{xmin:.6g}, {xmax:.6g}]", "m"),
                ("bounds, y", f"[{ymin:.6g}, {ymax:.6g}]", "m"),
                ("bounds, z", f"[{zmin:.6g}, {zmax:.6g}]", "m"),
            )

        return n_removed
