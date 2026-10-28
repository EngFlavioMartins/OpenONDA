"""Vortex Particle Method solver.

Provides DNS, LES, and inviscid VPM models with Taichi acceleration,
viscous diffusion, diagnostics, sampling, and restart support.
"""

from collections.abc import Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
import logging
import math
from numbers import Real
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import NDArray
import taichi as ti

from openonda.core.log_style import close_logging, configure_logging
from openonda.core.output_events import OutputEvent
from openonda.core.profiling import Profiler
from openonda.core.restart import reset_run_outputs, rewind_vpm_frames, select_backup
from openonda.core.status import RunStatus

from ..boundary_elements.vlm.solver.diagnostics import VLMDiagnostics
from ..boundary_elements.vlm.solver.forces import VLMForceEvaluator
from ..boundary_elements.vlm.solver.loading_distribution import VLMLoadingDistribution
from ..config.case import Numerics, VPMCase
from ..config.stabilization import StabilizationConfig
from ..config.state import set_flow_model
from ..config.state_limits import (
    ParticleStateError,
    ParticleStateMetrics,
    check_accepted_particle_state,
)
from ..coupling import CouplingStepper
from ..diagnostics.particle_state import measure_particle_state
from ..diagnostics.resolution import particle_resolution_metrics
from ..io.backup import _BackupIO
from ..io.frames import write_solution_frames
from ..io.logging import Logging, run_header
from ..io.metadata import write_vpm_metadata
from ..io.physics_events import LoggingPhysicsEventObserver
from ..kernels.base import make_vortex_kernel
from ..numerics.runge_kutta import RungeKutta
from ..particles.container import Particles
from ..physics.engine import PhysicsEngine
from ..physics.evaluation import ParticleFieldEvaluation
from ..physics.induction.base import StageRates, StageState
from ..physics.stage_rhs import (
    AxisymmetricNoSwirlStageProjection,
    ParticleExternalStageContribution,
    StageRHS,
    VLMStageContribution,
)
from ..runtime.backend import (
    acquire_taichi_backend,
    initialize_taichi_backend,
    reset_taichi_backend,
)
from ..sampling.manager import SamplerManager
from ..stabilization import StabilizationManager
from ..stabilization.context import (
    SolverParticleMutations,
    StabilizationContext,
    StabilizationMetrics,
    StabilizationStepState,
)
from ..turbulence.les import ParticleLES
from .evolution import EvolutionStepper

if TYPE_CHECKING:
    from ..boundary_elements.vlm.config import VLMConfig
    from ..config.viscous import ViscousConfig
    from ..particles.snapshot import ParticleSnapshot

FloatArray: TypeAlias = NDArray[np.float32] | NDArray[np.float64]
ParticleRecord: TypeAlias = dict[str, np.ndarray | np.generic]
_PRESSURE_HIERARCHICAL_OPENING = 0.3


@ti.data_oriented
class VPMSolver:
    """Own and advance one complete Vortex Particle Method simulation.

    The solver is the solver of the mutable particle fields, accepted clock,
    induction backend, Runge--Kutta workspace, viscous/turbulence models,
    optional VLM coupling, diagnostics, samplers, and restart I/O. A
    particle's ``vortex_strength`` is the particle-strength/circulation vector
    ``Gamma = omega * V`` in m³/s; it is distinct from vorticity ``omega`` in
    1/s and from core radius ``sigma`` in m.

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

    def __init__(self, case: VPMCase) -> None:
        """Validate a case and allocate its runtime workspaces.

        Parameters
        ----------
        case : VPMCase
            Immutable VPM configuration. Its particle capacity controls all
            fixed-size device allocations; initial-condition builders are
            invoked by :meth:`run` or the first :meth:`advance`.

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
        self.case = case
        self.numerics = case.numerics
        self.case_dir = Path(case.directory).resolve()
        variant = case.variant or ""
        self.solution_dir = self.case_dir / "solution" / variant
        self.samples_dir = self.case_dir / "samples" / variant
        self._backend_claimed = False
        self._closed = False
        acquire_taichi_backend(
            self,
            preferred_backend=case.run.runtime_compute_device or case.numerics.compute_device,
            precision=case.numerics.precision,
        )
        self._backend_claimed = True
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
        self._run_final_step = case.n_steps
        self._run_started = False
        self._run_wall_started_at: float | None = None
        self._run_wall_finished_at: float | None = None
        self._last_backup_state: tuple | None = None
        self._last_frame_state: tuple | None = None
        self._preparing_output = False
        self._prepared_backup_velocity: tuple | None = None
        self._allow_backup_velocity_reuse = False
        self._particle_snapshot_buffers: dict = {}
        self._mesh_cleanup_failure: BaseException | None = None
        self._loading_numerical_state = False
        self.run_status = RunStatus.not_started
        self.run_failure: BaseException | None = None
        self._evolution_failure: BaseException | None = None
        self._configuration_logged = False
        try:
            self.solution_dir.mkdir(parents=True, exist_ok=True)
            self.samples_dir.mkdir(parents=True, exist_ok=True)
            configure_logging(
                "openonda.vpm",
                self.solution_dir / "vpm.log",
                console=case.logging.console,
                level=logging.DEBUG if case.logging.mode == "debug" else logging.INFO,
            )
            Logging.configure(case.logging.schedule, case.numerics.time_step_size)
            self._initialize_numerics(case)
            self._initialize_backend(case.numerics)
            self._initialize_particles_and_physics(case.numerics)
            self._initialize_turbulence(case.numerics)
            self._initialize_solvers(case.numerics)
            self._sampler_manager = SamplerManager(self)
            Logging.startup(self)
            self._configuration_logged = True
            self.write_run_metadata(status=RunStatus.created)
            # Initial particles are added after construction. Keep those
            # mutations out of the event stream; the first requested
            # diagnostics describe their state.
            Logging.set_routine_messages_enabled(False)
        except BaseException:
            self._cleanup_failed_construction()
            raise

    def __enter__(self) -> "VPMSolver":
        """Return the solver; :meth:`close` runs when the block exits."""
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, traceback: object) -> None:
        """Close the solver, attaching cleanup errors to a pending exception."""
        self.close(failure=exc)

    def _cleanup_failed_construction(self) -> None:
        """Release every resource acquired before construction failed."""
        with suppress(Exception):
            close_logging("openonda.vpm")
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
        self._particle_regeneration_pending = False
        axisymmetric_axis = final_setup.axisymmetric_no_swirl_axis
        self.axisymmetric_axis = (
            -1 if axisymmetric_axis is None else {"x": 0, "y": 1, "z": 2}[axisymmetric_axis]
        )
        self._axisymmetric_orbits_validated = False

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
                Logging.warning_record(
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
        self.compute_device = initialize_taichi_backend(
            self.compute_device,
            self.case.logging.mode == "debug",
            self.precision,
            random_seed=final_setup.random_seed,
            supported_devices=getattr(self.induction, "supported_devices", None),
            _minimum_pool_bytes=minimum_pool_bytes,
        )
        # Keep the resolved backend identity independent of the process-global
        # Taichi constant.  ``close()`` releases that global runtime before a
        # terminal metadata is written, so consulting the constant later can
        # incorrectly report ``UNKNOWN`` for an otherwise reproducible run.
        self._backend_name = str(self.compute_device)
        supported_devices = getattr(self.induction, "supported_devices", None)
        if supported_devices is not None and self.compute_device not in supported_devices:
            raise ValueError(
                f"{type(self.induction).__name__} resolved AUTO to unsupported "
                f"compute_device={self.compute_device}; supported devices: "
                f"{sorted(device for device in supported_devices if device != 'auto')}"
            )
        Logging.message(run_header(self.precision, self._backend_name))
        # Initialization can call the same particle/model helpers used at run
        # time.  Suppress their routine event records until the complete,
        # resolved configuration is printed once after initial conditions.
        Logging.set_routine_messages_enabled(False)
        set_flow_model(self, flow_model=self.flow_model)
        self.compute_dtype = ti.f64 if self.precision == "f64" else ti.f32
        self.accumulator_dtype = self.compute_dtype
        self.np_dtype = np.float64 if self.precision == "f64" else np.float32

    def _initialize_particles_and_physics(self, final_setup: Numerics) -> None:
        """Create particle container, physics engine, source fields, background velocity."""
        max_p = final_setup.max_n_particles
        self.particles = Particles(max_n_particles=max_p, float_dtype=self.precision)
        self.physics = PhysicsEngine(
            particle_kernel=self.particle_kernel,
            max_n_particles=max_p,
            accumulator_dtype=self.accumulator_dtype,
            event_observer=LoggingPhysicsEventObserver(),
        )

        if hasattr(self.induction, "bind"):
            self.induction.bind(
                self.physics,
                kernel=make_vortex_kernel(self.particle_kernel),
            )
        # Target diagnostics and coupling queries must use the same selected
        # induction conditions as RK particle stages.  PhysicsBase keeps a
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
        self._stage_providers = [
            ParticleExternalStageContribution(self.particles, self.physics, self)
        ]
        if self.axisymmetric_axis >= 0:
            self._stage_providers.append(
                AxisymmetricNoSwirlStageProjection(
                    self.physics,
                    self.particles.zone_id,
                    self.axisymmetric_axis,
                )
            )
        _visc_cfg = getattr(final_setup, "viscous", None)
        if _visc_cfg is not None and hasattr(self.physics, "core_radius_ratio"):
            self.physics.core_radius_ratio = float(getattr(_visc_cfg, "core_radius_ratio", 2.5))
        # Grid diffusion on GPU uses a fixed workspace to avoid repeated allocation.
        vpm_bounds = final_setup.domain_bounds
        vc = getattr(final_setup, "viscous", None)
        scheme = getattr(vc, "scheme", "").lower() if vc is not None else ""
        is_grid_diffusion = scheme in {"dvh", "gbd"}
        fixed_grid_required = (
            self.compute_device in {"metal", "vulkan", "cuda"} and is_grid_diffusion
        )
        # The slip-slab image stencil fixes the z-plane phase even on CPU.
        # CPU may still grow the workspace; only its lattice origin is fixed.
        slip_slab_grid = (
            is_grid_diffusion
            and hasattr(self.induction, "z_min")
            and hasattr(self.induction, "z_max")
        )
        if fixed_grid_required and hasattr(self.physics, "require_fixed_grid_allocation"):
            self.physics.require_fixed_grid_allocation(True)
        if (fixed_grid_required or slip_slab_grid) and hasattr(
            self.physics, "configure_max_grid_extent"
        ):
            if scheme == "dvh":
                _grid_h = getattr(vc, "dvh_grid_spacing", None)
                _grid_pad = getattr(vc, "dvh_domain_padding", 3.0)
            else:
                _grid_h = getattr(vc, "gbd_grid_spacing", None)
                _grid_pad = getattr(vc, "gbd_domain_padding", 3.0)

            if vpm_bounds is None:
                raise ValueError(
                    "Fixed-lattice DVH/GBD requires domain_bounds so the diffusion "
                    "grid can be allocated once."
                )
            if _grid_h is None or _grid_h <= 0:
                raise ValueError(
                    "Fixed-lattice DVH/GBD requires a positive grid spacing so the "
                    "fixed diffusion grid can be pre-allocated."
                )

            self.physics.configure_max_grid_extent(vpm_bounds, _grid_h, _grid_pad)
        self._source_batch_size = self.physics.max_evaluation_points
        self.source_position = ti.Vector.field(
            3, dtype=self.compute_dtype, shape=self._source_batch_size
        )
        self.source_strength = ti.field(dtype=self.compute_dtype, shape=self._source_batch_size)
        self.source_core_radius = ti.field(dtype=self.compute_dtype, shape=self._source_batch_size)
        self._surface_sources = None
        self.n_sources = 0
        self.particles.set_freestream_velocity(np.array(self.numerics.freestream_velocity))

    def _initialize_turbulence(self, final_setup: Numerics) -> None:
        """Initialize LES turbulence, stretching settings, and diagnostics."""
        max_p = final_setup.max_n_particles
        self.turbulence_model = None
        if self.flow_model == "les":
            from ..physics.induction.slip_slab import SlipSlabInduction

            if final_setup.turbulence.subgrid_transport == "strength_exchange" and isinstance(
                final_setup.induction, SlipSlabInduction
            ):
                raise ValueError(
                    "STRENGTH_EXCHANGE sub-grid transport has no slip-plane images; "
                    "use CORE_SPREADING between slip planes"
                )
            self.turbulence_model = ParticleLES(
                model_name=final_setup.turbulence.model,
                max_n_particles=max_p,
                particle_kernel=self.particle_kernel,
                smagorinsky_coefficient=final_setup.turbulence.smagorinsky_coefficient,
                subgrid_dissipation_coefficient=final_setup.turbulence.subgrid_dissipation_coefficient,
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
            event_observer=LoggingPhysicsEventObserver(),
        )
        self._flow_integrals: dict = {}
        self._particle_resolution_metrics: dict = {}
        self._accepted_state_metrics: ParticleStateMetrics | None = None
        self._body_induced_fn = None
        self._particles_removed_this_step = 0
        self._vortex_strength_removed_this_step = np.zeros(3, dtype=self.np_dtype)

    def _initialize_solvers(self, final_setup: Numerics) -> None:
        """Initialize the stabilization master and the optional sub-solvers."""

        # Time histories consumed by export_diagnostics_csv and the VLM report.
        self._diagnostics_history: dict = {
            "time": [],
            "vpm_net_vortex_strength": [],
            "vpm_vortex_strength_magnitude_sum": [],
            "fvm_net_vortex_strength": [],
            "fvm_vortex_strength_magnitude_sum": [],
            "interpolated_net_vortex_strength": [],
            "interpolated_vortex_strength_magnitude_sum": [],
            "vortex_centroid": [],
            "n_particles_injected": [],
            "n_particle_candidates": [],
            "observed_time_step_size": [],
            "vlm_lift_coefficient": [],
            "vlm_drag_coefficient": [],
            "vlm_bound_vortex_strength_y": [],
            "vlm_wake_vortex_strength_y": [],
            "vlm_max_leading_edge_suction_parameter": [],
            "vlm_n_particles_total": [],
        }
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
                metrics=StabilizationMetrics(),
                slip_slab=getattr(self.physics, "_slip_slab_bounds", None),
            )
        )
        active = self.stabilization.active_mechanisms()
        if active:
            Logging.record(
                "stabilization",
                *(("  " + mechanism, "active") for mechanism in active),
            )
        self._initialize_vlm_solver(final_setup)
        self.stage_rhs = StageRHS(
            self.induction,
            providers=tuple(self._stage_providers),
            strength_enabled=self.flow_model != "potential",
            # Core spreading changes every core radius before the first stage.
            retain_strength_rate=self.viscous_scheme != "cs",
        )
        # Detailed section timing forces a device barrier around every phase.
        # Make that diagnostic opt-in; the whole-step timer remains available in
        # normal production runs without serialising every kernel launch.
        self.profiler = Profiler(enabled=self.case.logging.detailed_timing, synchronize=ti.sync)
        self.wall_time = 0.0
        self.stepper = EvolutionStepper(self)
        self.coupling = CouplingStepper(self)

    def _initialize_vlm_coupling(self) -> None:
        """Configure VLM solver coupling: mesh generation, force config, stability check."""
        self.vlm_solver._wake_kernel = self.induction.kernel
        self.vlm_solver.ensure_mesh_generated()
        if getattr(self.vlm_solver, "lattice", None) is not None:
            Logging.record("vlm", ("panels", f"{self.vlm_solver.lattice.n_panels:,}"))
            self.vlm_solver.check_coupling_stability(
                self.time_step_size, self.numerics.freestream_velocity
            )

    def _initialize_vlm_solver(self, final_setup: Numerics) -> None:
        """Initialize the optional VLM solver with error handling."""
        if final_setup.vlm is None:
            self.vlm_solver = None
        else:
            self._require_consistent_molecular_viscosity(final_setup.viscous, final_setup.vlm)
            from ..boundary_elements.vlm.solver.vlm_solver import VLMSolver

            self.vlm_solver = VLMSolver(final_setup.vlm)

        if self.vlm_solver is not None:
            try:
                self._initialize_vlm_coupling()
                # The VLM circulation is solved in the accepted-step coupling
                # phase. Its induced field is sampled at each temporary
                # particle RK position through the common StageRHS boundary.
                # Construct the immutable provider tuple only after optional
                # solvers have initialized. VLM is first so the axisymmetric
                # provider, when present, remains the final projection.
                self._stage_providers.insert(
                    0,
                    VLMStageContribution(self.vlm_solver, self.physics, self.particles),
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize VLM solver: {e}") from e

    @staticmethod
    def _require_consistent_molecular_viscosity(
        viscous_cfg: "ViscousConfig", vlm_setup: "VLMConfig"
    ) -> None:
        """Require one molecular kinematic viscosity in an attached VPM/VLM system."""
        scheme = getattr(viscous_cfg, "scheme", "none")
        if scheme == "none":
            vpm_kinematic_viscosity = 0.0
        else:
            configured_kinematic_viscosity = getattr(viscous_cfg, "kinematic_viscosity", None)
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

    def __len__(self) -> int:
        """Return the number of particles in the system."""
        return len(self.particles)

    def __getitem__(self, index: int) -> ParticleRecord:
        """Return a CPU snapshot of one active particle record.

        Parameters
        ----------
        index : int
            Zero-based active-particle index; negative indices follow the
            container's normal Python indexing behavior.

        Returns
        -------
        ParticleRecord
            Dictionary of scalar/vector fields in SI units. Values are copies,
            so editing the record does not mutate device state.
        """
        return self.particles[index]

    def __iter__(self) -> Iterator[ParticleRecord]:
        """Yield CPU snapshots for active particles in index order."""
        for i in range(len(self)):
            yield self[i]

    def __str__(self) -> str:
        """Return a formatted string summarizing the solver state."""
        return Logging.solver_summary(self)

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
            as the FVM--VPM coupler. It must equal ``time + time_step_size``
            to round-off. ``None`` advances by the configured time-step size.

        Raises
        ------
        RuntimeError
            If an earlier step failed and no backup was loaded since.
        Exception
            Any numerical, particle-state or stabilization error. The failed
            step leaves the solver unusable until :meth:`load_backup`.
        """
        self._ensure_evolution_usable()
        if not self._initial_conditions_built:
            self._build_initial_conditions()
        self._log_configuration_once()
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
        step_started = perf_counter()
        try:
            self.stepper.advance(defer_output=defer_output, time=time)
        except BaseException as error:
            self._mark_evolution_failure(error)
            raise
        if defer_output:
            return
        self.write_step_output()
        Logging.section("STEP TIMING", ("Step total wall time", perf_counter() - step_started, "s"))

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
        path = select_backup(selection, directory=self.solution_dir, kind="vpm")
        if path is None:
            return self.start_from("initial")
        self.load_backup(path)
        Logging.info(f"Continuing from {path}: step {self.step}, time {self.time:.17g}")
        return True

    def _start_run_from(self, selection: str | Path) -> None:
        """Start a run without loading a backup that already reaches the end time."""
        if selection == "initial":
            self.start_from(selection)
            return
        path = select_backup(selection, directory=self.solution_dir, kind="vpm")
        if path is None:
            self.start_from("initial")
            return
        step, time, _, particle_count = _BackupIO.inspect(self, path)
        if step < self.case.n_steps:
            self.start_from(path)
            return
        self.step = step
        self.time = time
        self._restart_loaded = True
        self._restart_terminal_noop = True
        self._restart_particle_count = particle_count
        self._initial_conditions_built = True
        Logging.info(f"Latest backup already reaches the end time: step {step}, time {time:.17g}")

    def run(self, *, start_from: str | Path | None = None) -> None:
        """Run the case to ``RunPlan.end_time`` and close the solver.

        Parameters
        ----------
        start_from : str, pathlib.Path or None, default=None
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
        if self._run_started:
            raise RuntimeError("VPMSolver.run() may be called only once")
        try:
            if start_from is not None:
                self._start_run_from(start_from)
        except BaseException as error:
            self.close(failure=error)
            raise
        self._run_started = True
        self._run_wall_started_at = perf_counter()
        self._run_wall_finished_at = None
        self._run_initial_step = self.step
        self._run_initial_time = self.time
        remaining_steps = max(0, self._run_final_step - self.step)
        if self._restart_terminal_noop:
            self._run_wall_finished_at = perf_counter()
            self.run_status = RunStatus.completed
            try:
                Logging.run_finished(self, self.run_status)
                self.write_run_metadata(status=self.run_status)
            finally:
                self.close()
            return
        status = RunStatus.failed
        failure: BaseException | None = None
        limit = self.case.run.wall_time_limit_seconds
        deadline = None if limit is None else self._run_wall_started_at + limit
        budget_exhausted = False
        primary_failure: BaseException | None = None
        try:
            self._build_initial_conditions()
            if start_from is not None and not self._restart_loaded:
                self.save_backup()
            self.run_status = RunStatus.running
            self.write_run_metadata(status=self.run_status)
            self._log_configuration_once()
            state_limit_failure = None
            invalid_state_failure = None
            if self._restart_loaded:
                self._sampler_manager.resume()
            self.write_initial_output()
            for _ in range(remaining_steps):
                if deadline is not None and perf_counter() >= deadline:
                    budget_exhausted = True
                    break
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
                if budget_exhausted:
                    status = RunStatus.wall_time_limit
                elif state_limit_failure is not None:
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
            self._run_wall_finished_at = perf_counter()
            if primary_failure is not None:
                status = RunStatus.failed
                failure = primary_failure
            self.run_status = status
            self.run_failure = failure
            for finalizer in (
                lambda: Logging.run_finished(self, self.run_status, self.run_failure),
                lambda: self.write_run_metadata(status=self.run_status),
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
        self._log_configuration_once()
        with self._output_preparation():
            self._refresh_diagnostics_for_output()
            self.write_vtk()
            self._sampler_manager.dispatch(OutputEvent.initial)

    def write_final_output(self) -> None:
        """Write the final particle frame and the final-only samplers.

        :meth:`run` calls this after the last step; a driver that advances
        the solver itself calls it once at the end of the run. Both follow
        ``RunPlan.final_output``; a frame already written for this state is
        not repeated. A run stopped before its final step also writes every
        sampler that has no sample of the last state.
        """
        if not self.case.run.final_output:
            return
        with self._output_preparation():
            self._refresh_diagnostics_for_output()
            if self._last_frame_state != self._state_key():
                self.write_vtk()
            if self.step < self._run_final_step:
                self._sampler_manager.write_all(OutputEvent.final)
            else:
                self._sampler_manager.dispatch(OutputEvent.final)

    def _write_end_backup(self) -> None:
        """Write the end-of-run backup unless the accepted state already has one."""
        if not self.case.backup.write_at_end or self._last_backup_state == self._state_key():
            return
        with self._output_preparation():
            self._refresh_diagnostics_for_output()
            self._allow_backup_velocity_reuse = True
            try:
                self.save_backup()
            finally:
                self._allow_backup_velocity_reuse = False

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
        if self.case.initial_weak_particle_percent > 0.0:
            self.remove_weak_particles(self.case.initial_weak_particle_percent)
        self._n_initial_particles = int(self.particles.n_particles_total)
        self._initial_conditions_built = True

    def write_run_metadata(self, *, status: RunStatus | str) -> Path:
        """Write ``vpm_metadata.json`` with the run status ``status``.

        Returns
        -------
        pathlib.Path
            The metadata file in the solution directory.
        """
        return write_vpm_metadata(
            self, self.solution_dir / "vpm_metadata.json", status=RunStatus(status)
        )

    def flush_output(self) -> None:
        """Finish pending file output; VPM output is written synchronously."""
        for handler in logging.getLogger("openonda.vpm").handlers:
            handler.flush()

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
            status = RunStatus.completed if self.step >= self.case.n_steps else RunStatus.partial
            if self._evolution_failure is not None or mesh_failure is not None:
                status = RunStatus.failed
            self.run_status = status
            try:
                self.write_run_metadata(status=status)
            except BaseException as error:
                if primary_failure is None:
                    primary_failure = error
        close_logging("openonda.vpm")
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

    @contextmanager
    def _output_preparation(self) -> Iterator[None]:
        """Allow one backup to reuse fields within a framework-owned event.

        Nothing survives the event: manual saves, coupling steps, changed
        callbacks and restart loads must evaluate their current transport field.
        Consuming the token before writing also prevents sampler callbacks from
        reusing it for a second save with externally changed boundary data.
        """
        self._preparing_output = True
        self._prepared_backup_velocity = None
        try:
            yield
        finally:
            self._preparing_output = False
            self._prepared_backup_velocity = None

    def _backup_velocity_key(self) -> tuple:
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
        if self._preparing_output:
            self._prepared_backup_velocity = self._backup_velocity_key()

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
            state=measure_particle_state(self.particles, self.time_step_size),
            resolution=self._particle_resolution_metrics,
            previous=self._accepted_state_metrics,
        )
        self.stabilization.lagrangian_cfl = self._accepted_state_metrics.strain_increment_infinity

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
        self._log_configuration_once()
        started = perf_counter()
        with self._output_preparation():
            self._check_accepted_particle_state()
            state_check_finished = perf_counter()
            if self.vlm_solver is not None:
                self._record_vlm_diagnostics()
            if self._sampler_manager.flow_integrals_due():
                self._update_all_flow_integrals()
            diagnostics_finished = perf_counter()
            if self._is_due(self.case.backup.schedule):
                self._allow_backup_velocity_reuse = True
                try:
                    self.save_backup()
                finally:
                    self._allow_backup_velocity_reuse = False
            if self._is_due(self.case.output.schedule):
                self.write_vtk()
            self._sampler_manager.dispatch(OutputEvent.accepted_step)
        Logging.section(
            "OUTPUT TIMING",
            ("Particle state checks", state_check_finished - started, "s"),
            ("Flow and VLM diagnostics", diagnostics_finished - state_check_finished, "s"),
            ("Backups, frames and samplers", perf_counter() - diagnostics_finished, "s"),
        )

    def _is_due(self, schedule: object | None) -> bool:
        """Return whether ``schedule`` selects the accepted state."""
        return schedule is not None and schedule.is_due(self.step, self.time, self.time_step_size)

    def _state_key(self) -> tuple[int, float, int]:
        """Identify the accepted state for once-only output."""
        return (self.step, self.time, self.particles.state_revision)

    @property
    def elapsed_wall_time(self) -> float:
        """Wall-clock seconds of the run, or of the evolution steps without a run."""
        if self._run_wall_started_at is None:
            return self.wall_time
        finished = self._run_wall_finished_at
        return (perf_counter() if finished is None else finished) - self._run_wall_started_at

    # Particle properties
    def _get_particle_field(self, method_name: str) -> np.ndarray:
        """Generic helper to get particle field data via cpu() methods."""
        return getattr(self.particles, f"{method_name}_cpu")()

    @property
    def particle_position(self) -> np.ndarray:
        """Return a copy of active particle positions, shape ``(N, 3)`` in m."""
        return self._get_particle_field("position")

    @property
    def particle_velocity(self) -> np.ndarray:
        """Return a copy of active particle velocities, shape ``(N, 3)`` in m/s."""
        return self._get_particle_field("velocity")

    @property
    def particle_core_radius(self) -> np.ndarray:
        """Return a copy of active core radii, shape ``(N,)`` in m."""
        return self._get_particle_field("core_radius")

    @property
    def particle_volume(self) -> np.ndarray:
        """Return a copy of quadrature volumes, shape ``(N,)`` in m³."""
        return self._get_particle_field("particle_volume")

    @property
    def particle_group_id(self) -> np.ndarray:
        """Return int32 particle-group labels, shape ``(N,)``."""
        return self._get_particle_field("group_id")

    @property
    def particle_zone_id(self) -> np.ndarray:
        """Return int32 spatial-zone labels, shape ``(N,)``."""
        return self._get_particle_field("zone_id")

    @property
    def particle_velocity_gradient(self) -> np.ndarray:
        """Return ``J[i,j] = d u_i/d x_j``, shape ``(N, 3, 3)`` in 1/s."""
        return self._get_particle_field("velocity_gradient")

    @property
    def freestream_velocity(self) -> np.ndarray:
        """Return the uniform background velocity, shape ``(3,)`` in m/s."""
        return self.particles.velocity_background_cpu()

    @property
    def particle_vorticity(self) -> np.ndarray:
        """Return particle vorticity diagnostics, shape ``(N, 3)`` in 1/s."""
        return self._get_particle_field("vorticity")

    @property
    def particle_vortex_strength(self) -> np.ndarray:
        """Return circulation vectors, shape ``(N, 3)`` in m³/s."""
        return self._get_particle_field("vortex_strength")

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
        core_radius = self.particle_core_radius
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
            event_observer=LoggingPhysicsEventObserver(),
        )

    def _record_time_history(self) -> None:
        """Delegate to VLMDiagnostics."""
        ft_hist = self._diagnostics_history.get("time", [])
        observed_time_step_size = self.time - ft_hist[-1] if len(ft_hist) >= 1 else 0.0
        VLMDiagnostics.record_time(self._diagnostics_history, self.time, observed_time_step_size)

    def _record_vlm_diagnostics(self) -> None:
        """Record the lattice loads and loading distributions of the accepted state."""
        VLMDiagnostics.record_vlm_diagnostics(
            self.vlm_solver,
            self.particles,
            self.particle_vortex_strength,
            self._diagnostics_history,
            self.time,
            self.samples_dir,
        )
        VLMLoadingDistribution.record_loading_distributions(
            self.vlm_solver,
            self._diagnostics_history,
            self.time,
            self.samples_dir,
        )

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
        """Return the net circulation vector, shape ``(3,)`` in m³/s."""
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

    @property
    def vortex_centroid(self) -> np.ndarray:
        """Return the global strength-magnitude-weighted centroid, shape ``(3,)`` in m."""
        return self.field_diagnostics.compute_vortex_centroid(self.particles)

    def compute_forces(
        self, density: float | None = None, reference_speed: float | None = None
    ) -> dict[str, np.ndarray | float]:
        """Compute aerodynamic force using the configured VLM method.

        Parameters
        ----------
        density : float or None, default=None
            Fluid density in kg/m³; defaults to the attached VLM case density.
        reference_speed : float or None, default=None
            Speed used for nondimensional force coefficients in m/s. When
            omitted, the configured VLM relative reference velocity is used.

        Returns
        -------
        dict[str, numpy.ndarray or float]
            Force components, coefficients, and the selected method as defined
            by the VLM force evaluator.

        Raises
        ------
        RuntimeError
            If this VPM solver has no attached VLM setup.
        ValueError
            If the attached VLM force method is unsupported.
        """
        if self.vlm_solver is None:
            raise RuntimeError("Force evaluation requires a VLM setup")
        if density is None:
            density = self.vlm_solver.density
        method = self.vlm_solver.force.method

        if method == "kutta_joukowski":
            return self._compute_forces_kutta_joukowski(density, reference_speed)
        else:
            raise ValueError(f"Unknown force method: {method}")

    def _compute_forces_kutta_joukowski(
        self, density: float, reference_speed: float | None
    ) -> dict[str, np.ndarray | float]:
        """Compute force through the VLM Kutta--Joukowski evaluator."""
        return VLMForceEvaluator.compute_kutta_joukowski(
            self.vlm_solver, self.freestream_velocity, density, reference_speed
        )

    # Per-particle diagnostics
    # Field evaluation
    def compute_velocity_at_points(
        self,
        evaluation_position: np.ndarray,
        include_freestream: bool = True,
        zone_mask: np.ndarray | None = None,
        include_body: bool = True,
    ) -> np.ndarray:
        """Evaluate complete velocity at arbitrary target points.

        Parameters
        ----------
        evaluation_position : numpy.ndarray
            Target coordinates, shape ``(N, 3)``, in m.
        include_freestream : bool, default=True
            Include the uniform background velocity in the particle induction.
        zone_mask : numpy.ndarray or None, default=None
            Optional boolean/int mask selecting source particles. Its length
            must match the active particle count.
        include_body : bool, default=True
            Include regularized surface sources and the optional body callback.

        Returns
        -------
        numpy.ndarray
            Velocity vectors, shape ``(N, 3)``, in m/s.
        """
        velocity = self.physics.compute_target_velocity(
            self.particles,
            evaluation_position,
            include_freestream=include_freestream,
            zone_mask=zone_mask,
        )
        return self._add_target_velocity_corrections(
            evaluation_position, velocity, include_body=include_body
        )

    def _add_target_velocity_corrections(
        self,
        evaluation_position: np.ndarray,
        particle_velocity: np.ndarray,
        *,
        include_body: bool,
    ) -> np.ndarray:
        """Add source-particle and body-potential terms to particle induction."""
        points = np.asarray(evaluation_position, dtype=np.float64).reshape(-1, 3)
        velocity = np.array(particle_velocity, dtype=self.np_dtype, copy=True).reshape(-1, 3)
        if len(velocity) != len(points):
            raise ValueError("target velocity and position counts must match")

        vlm = self.vlm_solver
        add_vlm = include_body and vlm is not None and vlm._solved
        if self.n_sources > 0 or add_vlm:
            for start, stop in self.physics._target_batch_slices(len(points)):
                count = stop - start
                positions = self.physics.target_position
                velocities = self.physics.target_velocity
                self.physics.upload_vector_array(points[start:stop], positions, count)
                self.physics.upload_vector_array(velocity[start:stop], velocities, count)
                for source_count in self._surface_source_batches():
                    self.physics.kernels["compute_target_source_velocity_kernel"](
                        positions,
                        self.source_position,
                        self.source_strength,
                        self.source_core_radius,
                        velocities,
                        count,
                        source_count,
                    )
                if add_vlm:
                    vlm.add_stage_velocity(positions, velocities, count, self.time)
                velocity[start:stop] = self.physics.extract_target_velocity(count)

        body_fn = self._body_induced_fn
        if include_body and body_fn is not None:
            velocity = velocity + np.asarray(
                body_fn(points, self.time), dtype=velocity.dtype
            ).reshape(velocity.shape)

        return velocity

    def _nonparticle_target_velocity(self, evaluation_position: np.ndarray) -> np.ndarray:
        """Return only regularized-source and body-potential target velocity."""
        points = np.asarray(evaluation_position, dtype=np.float64).reshape(-1, 3)
        return self._add_target_velocity_corrections(
            points,
            np.zeros((len(points), 3), dtype=self.np_dtype),
            include_body=True,
        )

    def _surface_source_batches(self) -> Iterator[int]:
        """Upload every source through the reusable device workspace."""
        if not self.n_sources:
            return
        positions, strengths, radii = self._surface_sources
        for start in range(0, self.n_sources, self._source_batch_size):
            stop = min(start + self._source_batch_size, self.n_sources)
            count = stop - start
            self.physics.upload_vector_array(positions[start:stop], self.source_position, count)
            self.physics._upload_scalar_array(strengths[start:stop], self.source_strength, count)
            self.physics._upload_scalar_array(radii[start:stop], self.source_core_radius, count)
            yield count

    def compute_velocity_gradient_at_points(
        self, evaluation_position: np.ndarray, *, particle_spacing: float
    ) -> np.ndarray:
        """Evaluate the Jacobian of the body-complete velocity field.

        Vortex-particle induction is differentiated analytically. Regularized
        source and body-callback contributions are differentiated by centred
        differences with a step scaled by the coupling lattice spacing ``particle_spacing``.
        The result has shape ``(N, 3, 3)`` and convention
        ``J[i,j] = d(u_i)/d(x_j)``.

        Parameters
        ----------
        evaluation_position : numpy.ndarray
            Target coordinates, shape ``(N, 3)``, in m.
        particle_spacing : float
            Positive coupling/resolution spacing in m. It scales the centered
            difference used only for auxiliary source/body terms.

        Returns
        -------
        numpy.ndarray
            Complete velocity Jacobian, shape ``(N, 3, 3)``, in 1/s.

        Raises
        ------
        RuntimeError
            If a body callback returns non-finite or incorrectly shaped data.
        """
        points = np.asarray(evaluation_position, dtype=np.float64).reshape(-1, 3)
        # Use the same approximation as the target velocity.  Mixing a
        # direct Jacobian with a treecode velocity is both prohibitively
        # expensive at coupling faces and inconsistent with the boundary data.
        gradient = np.asarray(
            self.physics.compute_target_velocity_gradient_consistent(self.particles, points),
            dtype=np.float64,
        ).reshape(-1, 3, 3)
        return self._add_nonparticle_target_gradient(
            points, gradient, particle_spacing=particle_spacing
        )

    def _add_nonparticle_target_gradient(
        self, points: np.ndarray, particle_gradient: np.ndarray, *, particle_spacing: float
    ) -> np.ndarray:
        """Add finite-difference nonparticle target gradients."""
        gradient = np.asarray(particle_gradient, dtype=np.float64).reshape(-1, 3, 3).copy()
        has_vlm = self.vlm_solver is not None and self.vlm_solver._solved
        if len(points) == 0:
            return gradient

        needs_finite_difference = self.n_sources > 0 or has_vlm or self._body_induced_fn is not None
        if not needs_finite_difference:
            return gradient

        step = max(1.0e-6, 1.0e-3 * float(particle_spacing))
        velocity_at = self._nonparticle_target_velocity
        for axis in range(3):
            offset = np.zeros(3, dtype=np.float64)
            offset[axis] = step
            plus = velocity_at(points + offset)
            minus = velocity_at(points - offset)
            if plus.shape != points.shape or minus.shape != points.shape:
                raise RuntimeError("VPM body-velocity callback returned an invalid shape")
            gradient[:, :, axis] += (plus - minus) / (2.0 * step)
        if not np.all(np.isfinite(gradient)):
            raise RuntimeError("Complete VPM target-gradient evaluation returned non-finite data")
        return gradient

    def compute_velocity_and_gradient_at_points(
        self, evaluation_position: np.ndarray, *, particle_spacing: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate body-complete target velocity and Jacobian together.

        Treecode runs build and traverse the particle hierarchy once, then add
        the regularized-source and body-potential velocity and Jacobian terms.

        Parameters
        ----------
        evaluation_position : numpy.ndarray
            Target coordinates, shape ``(N, 3)``, in m.
        particle_spacing : float
            Coupling spacing in m for centered differences of auxiliary terms.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Complete velocity ``(N, 3)`` in m/s and Jacobian ``(N, 3, 3)`` in
            1/s.
        """
        points = np.asarray(evaluation_position, dtype=np.float64).reshape(-1, 3)
        velocity, gradient = self.physics.compute_target_velocity_and_gradients_consistent(
            self.particles, points, include_freestream=True
        )
        complete_velocity = self._add_target_velocity_corrections(
            points, velocity, include_body=True
        )
        complete_gradient = self._add_nonparticle_target_gradient(
            points, gradient, particle_spacing=particle_spacing
        )
        return complete_velocity, complete_gradient

    def compute_velocity_and_tangential_normal_gradient_at_points(
        self,
        evaluation_position: np.ndarray,
        normal: np.ndarray,
        *,
        particle_spacing: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the velocity and the normal gradient of the tangential velocity at points.

        The mixed FVM boundary condition does not consume the full nine-component
        Jacobian. Particle induction is still evaluated by the configured fused
        target operation. Auxiliary source, VLM, and body-callback terms use
        only two centred samples along each face normal instead of three
        coordinate-direction pairs.

        Parameters
        ----------
        evaluation_position, normal : numpy.ndarray
            Face points and normals, both shape ``(N, 3)``. Positions are in m;
            normals are dimensionless and must be finite/non-zero.
        particle_spacing : float
            Centered-difference spacing in m for auxiliary source/body terms.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Complete velocity ``(N, 3)`` in m/s and tangential part of
            ``J @ n`` with shape ``(N, 3)`` in m/s².

        Raises
        ------
        ValueError
            If normal and point shapes differ or a normal is zero/non-finite.
        """
        points = np.asarray(evaluation_position, dtype=np.float64).reshape(-1, 3)
        face_normals = np.asarray(normal, dtype=np.float64).reshape(-1, 3)
        if face_normals.shape != points.shape:
            raise ValueError("normal count does not match target position")
        normal_magnitude = np.linalg.norm(face_normals, axis=1)
        if np.any(~np.isfinite(face_normals)) or np.any(normal_magnitude <= 0.0):
            raise ValueError("target normal must be finite and non-zero")
        unit_normals = face_normals / normal_magnitude[:, None]

        velocity, gradient = self.physics.compute_target_velocity_and_gradients_consistent(
            self.particles, points, include_freestream=True
        )

        complete_velocity = self._add_target_velocity_corrections(
            points, velocity, include_body=True
        )
        normal_velocity_gradient = np.einsum(
            "fij,fj->fi", np.asarray(gradient, dtype=np.float64).reshape(-1, 3, 3), unit_normals
        )
        has_vlm = self.vlm_solver is not None and self.vlm_solver._solved
        needs_finite_difference = self.n_sources > 0 or has_vlm or self._body_induced_fn is not None
        if needs_finite_difference:
            step = max(1.0e-6, 1.0e-3 * float(particle_spacing))
            velocity_at = self._nonparticle_target_velocity
            plus = velocity_at(points + step * unit_normals)
            minus = velocity_at(points - step * unit_normals)
            if plus.shape != points.shape or minus.shape != points.shape:
                raise RuntimeError("VPM body-velocity callback returned an invalid shape")
            normal_velocity_gradient += (plus - minus) / (2.0 * step)
        tangential = (
            normal_velocity_gradient
            - np.einsum("fi,fi->f", normal_velocity_gradient, unit_normals)[:, None] * unit_normals
        )
        if not np.all(np.isfinite(complete_velocity)) or not np.all(np.isfinite(tangential)):
            raise RuntimeError("Mixed VPM target evaluation returned non-finite data")
        return np.asarray(complete_velocity, dtype=np.float64), tangential

    # Diagnostics
    def _log_configuration_once(self) -> None:
        """Ensure the complete static time-zero configuration is visible once."""
        if self._configuration_logged:
            Logging.set_routine_messages_enabled(True)
            return
        # Lightweight unit doubles created without a VPMCase have no reportable
        # configuration and should retain the numerical facade conditions.
        if not hasattr(self, "case") or not hasattr(self, "setup"):
            return
        Logging.set_routine_messages_enabled(True)
        Logging.message(Logging.solver_info(self), flush=True)
        self._configuration_logged = True

    # Particle management
    def remove_all_particles(self) -> None:
        """Remove the complete active cloud and record removed-strength diagnostics.

        Side Effects
        ------------
        Compacts device particle fields, resets the stabilization refinement
        reference, and stores the removed count and circulation for diagnostics.
        """
        self._particles_removed_this_step = len(self.particles)
        self._vortex_strength_removed_this_step = self.particles.net_vortex_strength()
        self.stabilization.on_removal(remove_all=True)
        self.particles.remove_vortex_particles((), remove_all=True)

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
        Appends and copies the batch into device fields, resets axisymmetric
        orbit validation, and updates stabilization refinement reference/reference totals.
        """
        if kinematic_viscosity is None:
            kinematic_viscosity = getattr(self._viscous_config, "kinematic_viscosity", None)
            if kinematic_viscosity is not None and kinematic_viscosity > 0:
                N = len(position)
                kinematic_viscosity = np.full(N, kinematic_viscosity, dtype=self.np_dtype)
            else:
                raise ValueError(
                    "kinematic_viscosity is required. Either configure "
                    "ViscousConfig.kinematic_viscosity or pass an explicit array."
                )

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
        self._axisymmetric_orbits_validated = False
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
        Clears/repopulates active device fields, invalidates caches, resets
        axisymmetric validation, and notifies stabilization of replacement.
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
            kinematic_viscosity = getattr(self._viscous_config, "kinematic_viscosity", None)
            if kinematic_viscosity is not None and kinematic_viscosity > 0:
                kinematic_viscosity = np.full(
                    len(position), kinematic_viscosity, dtype=self.np_dtype
                )
            else:
                raise ValueError(
                    "viscosity parameter is required.  Either set "
                    "ViscousConfig.kinematic_viscosity or pass an explicit array."
                )

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
        self._axisymmetric_orbits_validated = False
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
        self._axisymmetric_orbits_validated = False
        if refinement_reference is not None:
            self.stabilization.on_replacement(*refinement_reference)

    def load_particle_field(
        self, particle_file_name: str | Path, remove_current_particles: bool = False
    ) -> None:
        """Add the particles of a particle file to the active cloud.

        Parameters
        ----------
        particle_file_name : str or pathlib.Path
            Particle file readable by :meth:`Particles.load_vortex_particles`.
        remove_current_particles : bool, default=False
            Remove the current particles first.
        """
        self.particles.load_vortex_particles(str(particle_file_name), remove_current_particles)
        Logging.info(f"Particle field loaded: {particle_file_name}")

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

        if prop_name in {
            "position",
            "velocity",
            "vortex_strength",
            "vorticity",
        }:
            expected_shape = (n_particles_total, 3)
        elif prop_name in {"velocity_gradient", "strain_rate"}:
            expected_shape = (n_particles_total, 3, 3)
        else:
            expected_shape = (n_particles_total,)

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

        valid_properties = {
            "position",
            "velocity",
            "vortex_strength",
            "vorticity",
            "core_radius",
            "particle_volume",
            "kinematic_viscosity",
            "eddy_viscosity",
            "effective_viscosity",
            "group_id",
            "zone_id",
            "velocity_gradient",
            "strain_rate",
        }
        invalid = [name for name in properties if name not in valid_properties]
        if invalid:
            raise ValueError(
                f"Invalid property name {invalid[0]!r}. "
                f"Valid properties: {sorted(valid_properties)}"
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
        Logging.record(
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
        self._evolution_failure = None
        self.run_failure = None
        self._restart_loaded = True
        self._restart_output_time = self.time
        # The backup replaces the particle state; the initial conditions are never rebuilt.
        self._initial_conditions_built = True
        self._sampler_manager.rewind(self.time)
        rewind_vpm_frames(self.solution_dir, self.step, self.time)
        self._last_backup_state = self._state_key()
        # A growth limit compares adjacent accepted states; the restored
        # state starts a new history.
        self._accepted_state_metrics = None

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
        started = perf_counter()
        self._refresh_backup_particle_fields()
        # Serialization downloads the fields anyway; charge the device work here.
        ti.sync()
        prepared = perf_counter()
        written = _BackupIO.save(self, path)
        Logging.section(
            "BACKUP TIMING",
            ("Field preparation", prepared - started, "s"),
            ("File writing", perf_counter() - prepared, "s"),
        )
        if default:
            self._last_backup_state = self._state_key()
            status = RunStatus.running if self._run_started else RunStatus.partial
            self.write_run_metadata(status=status)
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
        self._refresh_backup_particle_fields()
        frame = write_solution_frames(self)
        self._last_frame_state = self._state_key()
        return frame

    def _refresh_backup_particle_fields(self) -> None:
        """Evaluate the transport velocity and vorticity written to backups and frames.

        A field prepared by the current output event is reused once;
        otherwise the complete transport field (including a bound lattice
        and external sources) is evaluated at the accepted time, without
        publishing stretching or exchange rates.
        """
        n_particles = self.particles.n_particles_total
        prepared = self._prepared_backup_velocity
        self._prepared_backup_velocity = None
        reuse_velocity = (
            self._preparing_output
            and self._allow_backup_velocity_reuse
            and prepared is not None
            and prepared == self._backup_velocity_key()
        )
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

    def write_state(self, path: str | Path) -> Path:
        """Write the active particles to the VTK PolyData file ``path``.

        Returns
        -------
        pathlib.Path
            The written file.
        """
        path = Path(path)
        self.particles.save_vortex_particles(str(path), write_precision=self.write_precision)
        return path

    # Particle updates

    def set_freestream_velocity(self, velocity: list[float] | np.ndarray) -> None:
        """Set framework-owned uniform background velocity state [m/s]."""
        dtype = np.float64 if self.precision == "f64" else np.float32
        velocity_arr = np.array(velocity, dtype=dtype)

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
        if len(bounds) != 6:
            raise ValueError("bounds must be [xmin, xmax, ymin, ymax, zmin, zmax]")

        n_particles_total = self.particles.n_particles_total
        if n_particles_total == 0:
            return 0

        xmin, xmax, ymin, ymax, zmin, zmax = bounds

        keep_mask = None
        if self.stabilization.reference_vortex_strength is not None:
            position = self.particles.position_cpu()
            inside = (
                (xmin <= position[:, 0])
                & (position[:, 0] <= xmax)
                & (ymin <= position[:, 1])
                & (position[:, 1] <= ymax)
                & (zmin <= position[:, 2])
                & (position[:, 2] <= zmax)
            )
            keep_mask = inside if invert_selection else ~inside

        n_removed = self.particles.remove_particles_by_bounds(
            bounds, invert_selection=invert_selection
        )

        if n_removed > 0:
            self.stabilization.on_removal(keep_mask=keep_mask)
            action = "outside" if invert_selection else "inside"
            Logging.record(
                "particles removed",
                ("particles", f"{n_removed:,}"),
                ("region", f"{action} box"),
                ("bounds, x", f"[{xmin:.6g}, {xmax:.6g}]", "m"),
                ("bounds, y", f"[{ymin:.6g}, {ymax:.6g}]", "m"),
                ("bounds, z", f"[{zmin:.6g}, {zmax:.6g}]", "m"),
            )

        return n_removed

    def remove_weak_particles(self, percent: float) -> int:
        """Remove particles below a percentage of maximum strength.

        Parameters
        ----------
        percent : float
            Threshold in ``[0, 100]`` applied to particle-strength vector magnitude.

        Returns
        -------
        int
            Number of removed particles.

        Raises
        ------
        ValueError
            If ``percent`` is outside ``[0, 100]``.
        """
        if percent < 0 or percent > 100:
            raise ValueError("Percent must be between 0 and 100")

        if len(self.particles) == 0:
            return 0

        particles_before = len(self.particles)

        removed_indices = self.particles._remove_weak_particles(
            percent=percent,
        )
        if removed_indices is not None and len(removed_indices) > 0:
            keep = np.ones(particles_before, dtype=bool)
            keep[np.asarray(removed_indices, dtype=np.int64)] = False
            self.stabilization.on_removal(keep_mask=keep)

        if len(self.particles) > 0:
            self.physics.compute_vorticities(self.particles)

        particles_after = len(self.particles)
        particles_removed = particles_before - particles_after

        Logging.particle_cleanup(percent, particles_before, particles_removed, particles_after)

        return particles_removed
