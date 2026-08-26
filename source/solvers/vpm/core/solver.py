"""Vortex Particle Method solver.

Provides DNS, LES, and inviscid VPM models with Taichi acceleration,
viscous diffusion, diagnostics, sampling, and restart support.

Author: Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
License: GPL-3.0-or-later
"""

import os
from pathlib import Path

import numpy as np
import taichi as ti

from source.solvers.vpm.particles.container import Particles
from source.solvers.vpm.turbulence.turbulence import ParticlesLES
from source.write_precision import DEFAULT_WRITE_PRECISION, validate_write_precision

from ..boundary_elements.vlm.solver.diagnostics import VLMDiagnostics
from ..boundary_elements.vlm.solver.forces import VLMForceEvaluator
from ..boundary_elements.vlm.solver.loading_distribution import VLMLoadingDistribution
from ..config.constants import MAX_N_PARTICLES, MAX_SOURCES
from ..config.setup import VPMSetup
from ..config.stabilization import StabilizationConfig
from ..config.state import set_flow_model
from ..coupling import CouplingStepper
from ..diagnostics.resolution import discretization_health
from ..io.checkpoint import CheckpointManager
from ..io.logging import Logging, print_openonda_header
from ..io.runtime_profiler import RuntimeProfiler
from ..io.sampler import SamplerExecutor
from ..io.solver_io import SolverIO
from ..physics.engine import PhysicsEngine
from ..physics.evaluation import ParticleFieldEvaluation
from ..runtime.backend import initialize_taichi_backend, reset_taichi_backend
from ..stabilization import StabilizationManager
from ..stabilization.context import StabilizationContext
from .evolution import EvolutionStepper


@ti.data_oriented
class VPMSolver:
    """Vortex Particle Method solver.

    The solver owns the particle field, time integration, viscous and turbulence
    models, optional boundary-element coupling, diagnostics, sampling, and restart
    state. Configuration is supplied through :class:`VPMSetup`.
    """

    # Initialization

    def __init__(
        self, setup: VPMSetup | None = None, *, case_dir: str | Path | None = None
    ) -> None:
        """Initialize the VPM solver. See VPMSetup for all parameters."""
        self.case_dir = Path("." if case_dir is None else case_dir).resolve()
        final_setup = self._init_setup(setup)
        self._init_io_and_backend(final_setup, final_setup.debug_mode)
        self._init_particles_and_physics(final_setup)
        self._init_turbulence_and_adaptation(final_setup)
        self._init_solvers(final_setup)
        Logging.message(Logging.solver_info(self))

    @staticmethod
    def reset_gpu() -> None:
        """Reset the Taichi runtime and release device allocations.

        Call before constructing a new solver when several VPM cases are run
        sequentially in the same Python process.
        """
        reset_taichi_backend()

    @staticmethod
    def synchronize() -> None:
        """Wait until all queued VPM backend work has completed."""
        ti.sync()

    def _init_setup(self, setup: VPMSetup | None) -> VPMSetup:
        """Validate the setup and initialize scalar solver state."""
        final_setup = setup if setup is not None else VPMSetup.dns_simulation()
        final_setup._validate_config()
        self.setup = final_setup
        self.time_step_size = final_setup.time_step_size
        self.time = final_setup.time
        self.step = final_setup.step
        self._is_particle_regeneration_pending = False
        self.time_integration = final_setup.time_integration.upper()
        self.coupled_max_strain_increment = final_setup.coupled_max_strain_increment
        self.coupled_max_advection_fraction = final_setup.coupled_max_advection_fraction
        self.coupled_max_substeps = final_setup.coupled_max_substeps
        axisymmetric_axis = final_setup.axisymmetric_no_swirl_axis
        self.axisymmetric_axis = (
            -1 if axisymmetric_axis is None else {"x": 0, "y": 1, "z": 2}[axisymmetric_axis]
        )
        self._axisymmetric_orbits_validated = False

        # DVH uses a fixed heat-kernel increment Δt_d = β R_d² / (4ν).
        import math as _math

        self._dvh_time_step_size_info: str | None = None
        self._gbd_time_step_size_info: str | None = None
        self._rwm_time_step_size_info: str | None = None
        vc = final_setup.viscous

        # RWM accuracy criterion.
        if (
            vc.scheme == "RWM"
            and vc.particle_spacing is not None
            and vc.particle_spacing > 0
            and vc.kinematic_viscosity is not None
            and vc.kinematic_viscosity > 0
        ):
            rwm_max_time_step_size = vc.rwm_accuracy_time_step_size()
            if self.time_step_size > rwm_max_time_step_size * (1.0 + 1e-6):
                Logging.warning(
                    f"component=RWM time_step_size_s={self.time_step_size:.4e} "
                    f"accuracy_limit_s={rwm_max_time_step_size:.4e} criterion=h2_over_4nu"
                )
            self._rwm_time_step_size_info = (
                f"RWM accuracy limit particle_spacing²/(4nu) = {rwm_max_time_step_size:.4e} s "
                f"(particle_spacing = {vc.particle_spacing:.3e} m, "
                f"kinematic_viscosity = {vc.kinematic_viscosity:.3e} m²/s)."
            )

        # GBD substeps only its explicit grid Laplacian when this limit is exceeded.
        if vc.scheme == "GBD" and vc.kinematic_viscosity is not None and vc.kinematic_viscosity > 0:
            max_time_step_size = vc.gbd_max_time_step_size()
            self._gbd_time_step_size_info = (
                f"GBD macro-step = {self.time_step_size:.4e} s; "
                f"molecular explicit stage limit = {max_time_step_size:.4e} s."
            )

        # Match the user step to an integer subdivision of the DVH increment.
        self._n_steps_per_dvh_diffusion: int = 1
        self._n_steps_since_dvh_diffusion: int = 0
        if vc.scheme == "DVH" and vc.kinematic_viscosity is not None and vc.kinematic_viscosity > 0:
            from ..physics.diffusion import _DVH_BETA

            diffusion_time_step_size_raw = vc.dvh_required_time_step_size()
            # Avoid noisy floating-point time values.
            magnitude = _math.floor(_math.log10(abs(diffusion_time_step_size_raw)))
            diffusion_time_step_size = round(diffusion_time_step_size_raw, -magnitude + 2)
            user_time_step_size = self.time_step_size
            n_sub = (
                max(1, int(round(diffusion_time_step_size / user_time_step_size)))
                if user_time_step_size > 0
                else 1
            )
            substep_size = diffusion_time_step_size / n_sub
            if abs(user_time_step_size - substep_size) > 1e-6 * max(
                user_time_step_size, substep_size
            ):
                Logging.record(
                    "discrete vortex heat method, time step pinned",
                    ("time step, requested", f"{user_time_step_size:.4e}", "s"),
                    ("time step, applied", f"{substep_size:.4e}", "s"),
                    ("diffusion interval", f"{diffusion_time_step_size:.4e}", "s"),
                    ("steps per diffusion", f"{n_sub:,}"),
                    ("beta", f"{_DVH_BETA:g}"),
                    (
                        "support radius",
                        f"{vc.dvh_support_radius_ratio * vc.dvh_grid_spacing:.4e}",
                        "m",
                    ),
                )
                self.time_step_size = substep_size
            self._n_steps_per_dvh_diffusion = n_sub
            self._dvh_time_step_size_info = (
                f"DVH fires every {n_sub} step(s) (time_step_size = "
                f"Δt_d/{n_sub} = {substep_size:.4e} s, "
                f"Δt_d = β·R_d²/(4nu) = {diffusion_time_step_size:.4e} s)."
            )

        self.advection_scheme = final_setup.advection.scheme
        self.stretching_scheme = final_setup.stretching.scheme
        self.stretching_use_treecode = getattr(final_setup.stretching, "use_treecode", False)
        self.stretching_treecode_theta = getattr(final_setup.stretching, "treecode_theta", 0.3)
        self.stretching_conserve_moments = getattr(
            final_setup.stretching, "conserve_moments", False
        )
        self.stretching_conserve_energy = getattr(final_setup.stretching, "conserve_energy", False)
        self.compute_device = final_setup.compute_device.upper()
        self.flow_model = final_setup.turbulence.flow_model.upper()
        self.viscous_scheme = final_setup.viscous.scheme
        self._viscous_config = final_setup.viscous
        self.stabilization_config: StabilizationConfig = final_setup.stabilization
        self.particle_kernel = final_setup.particle_kernel.upper()
        self.checkpoint_interval_steps = final_setup.checkpoint_interval_steps
        self.logging_interval_steps = final_setup.logging_interval_steps
        self.timing_interval_steps = final_setup.timing_interval_steps
        self.checkpoint_name = final_setup.checkpoint_name
        configured_checkpoint_directory = Path(final_setup.checkpoint_directory)
        if not configured_checkpoint_directory.is_absolute():
            configured_checkpoint_directory = self.case_dir / configured_checkpoint_directory
        self.checkpoint_directory = str(configured_checkpoint_directory.resolve())
        if getattr(final_setup, "clean", False):
            import shutil as _shutil

            _checkpoint_path = Path(self.checkpoint_directory)
            if _checkpoint_path.exists():
                _shutil.rmtree(_checkpoint_path)
        Path(self.checkpoint_directory).mkdir(parents=True, exist_ok=True)
        return final_setup

    def _init_io_and_backend(self, final_setup: VPMSetup, debug_mode: bool) -> None:
        """Set up output redirection, IO, precision, splitter/remesher, Taichi backend."""
        Logging.setup_output_redirection(self)
        self.io = SolverIO(self)
        self.precision = getattr(final_setup, "precision", "f32")
        if self.precision not in ("f32", "f64"):
            raise ValueError(f"precision must be 'f32' or 'f64', got '{self.precision}'")
        self.write_precision = validate_write_precision(
            getattr(final_setup, "write_precision", DEFAULT_WRITE_PRECISION)
        )
        self.checkpoint_store_velocity_gradient = bool(
            getattr(final_setup, "checkpoint_store_velocity_gradient", True)
        )
        self.compute_device = initialize_taichi_backend(
            self.compute_device,
            debug_mode,
            self.precision,
            device_memory_fraction=getattr(final_setup, "device_memory_fraction", 0.5),
            random_seed=final_setup.random_seed,
        )
        print_openonda_header(self.precision)
        set_flow_model(self, flow_model=self.flow_model)
        self.compute_dtype = ti.f64 if self.precision == "f64" else ti.f32
        self.accumulator_dtype = self.compute_dtype
        self.np_dtype = np.float64 if self.precision == "f64" else np.float32

    def _init_particles_and_physics(self, final_setup: VPMSetup) -> None:
        """Create particle container, physics engine, source fields, background velocity."""
        max_p = getattr(final_setup, "max_n_particles", MAX_N_PARTICLES)
        self.particles = Particles(max_n_particles=max_p, float_dtype=self.precision)
        self.physics = PhysicsEngine(
            particle_kernel=self.particle_kernel,
            max_n_particles=max_p,
            accumulator_dtype=self.accumulator_dtype,
            max_evaluation_points=final_setup.max_evaluation_points,
        )

        _vel_cfg = getattr(final_setup, "velocity", None)
        _vel_method = "TREECODE" if (_vel_cfg and _vel_cfg.method == "TREECODE") else "DIRECT"
        _vel_theta = _vel_cfg.theta if _vel_cfg else 0.5
        self.physics.configure_velocity(
            _vel_method,
            _vel_theta,
            multipole_order=getattr(_vel_cfg, "multipole_order", 1),
            sort_particle_targets=getattr(_vel_cfg, "sort_particle_targets", False),
            traversal_block_dim=getattr(_vel_cfg, "traversal_block_dim", 128),
        )

        _visc_cfg = getattr(final_setup, "viscous", None)
        if _visc_cfg is not None and hasattr(self.physics, "core_radius_ratio"):
            self.physics.core_radius_ratio = float(getattr(_visc_cfg, "core_radius_ratio", 2.5))
        if hasattr(self.physics, "configure_body_mask"):
            try:
                bodies = getattr(final_setup, "bodies", ())
                first_body_stl = bodies[0].stl if bodies else None
                self.physics.configure_body_mask(first_body_stl)
            except Exception as exc:
                Logging.warning(f"component=body_mask status=configuration_failed error={exc!r}")

        # Grid diffusion on GPU uses a fixed workspace to avoid repeated allocation.
        vpm_bounds = final_setup.domain_bounds
        vc = getattr(final_setup, "viscous", None)
        scheme = getattr(vc, "scheme", "").upper() if vc is not None else ""
        is_grid_diffusion = scheme in {"DVH", "GBD"}
        fixed_grid_required = (
            self.compute_device in {"METAL", "VULKAN", "CUDA"} and is_grid_diffusion
        )
        if fixed_grid_required and hasattr(self.physics, "require_fixed_grid_allocation"):
            self.physics.require_fixed_grid_allocation(True)
        if fixed_grid_required and hasattr(self.physics, "configure_max_grid_extent"):
            if scheme == "DVH":
                _grid_h = getattr(vc, "dvh_grid_spacing", None)
                _grid_pad = getattr(vc, "dvh_domain_padding", 3.0)
            else:
                _grid_h = getattr(vc, "gbd_grid_spacing", None)
                _grid_pad = getattr(vc, "gbd_domain_padding", 3.0)

            if vpm_bounds is None:
                raise ValueError(
                    "GPU DVH/GBD requires domain_bounds so the diffusion "
                    "grid can be allocated once."
                )
            if _grid_h is None or _grid_h <= 0:
                raise ValueError(
                    "GPU DVH/GBD requires a positive grid spacing so the "
                    "fixed diffusion grid can be pre-allocated."
                )

            self.physics.configure_max_grid_extent(vpm_bounds, _grid_h, _grid_pad)
        self.source_position = ti.Vector.field(3, dtype=self.compute_dtype, shape=MAX_SOURCES)
        self.source_strength = ti.field(dtype=self.compute_dtype, shape=MAX_SOURCES)
        self.source_core_radius = ti.field(dtype=self.compute_dtype, shape=MAX_SOURCES)
        self.n_sources = 0
        if hasattr(self.setup, "freestream_velocity"):
            self.particles.set_freestream_velocity(np.array(self.setup.freestream_velocity))

    def _init_turbulence_and_adaptation(self, final_setup: VPMSetup) -> None:
        """Initialize LES turbulence, stretching settings, and diagnostics."""
        max_p = getattr(final_setup, "max_n_particles", MAX_N_PARTICLES)
        self.turbulence_model = None
        if self.flow_model == "LES":
            self.turbulence_model = ParticlesLES(
                model_name=final_setup.turbulence.model,
                max_n_particles=max_p,
                particle_kernel=self.particle_kernel,
                smagorinsky_coefficient=final_setup.turbulence.smagorinsky_coefficient,
                subgrid_dissipation_coefficient=final_setup.turbulence.subgrid_dissipation_coefficient,
                accumulator_dtype=self.accumulator_dtype,
            )
        self.stretching_enabled = final_setup.stretching.enabled
        self.stretching_mode = final_setup.stretching.mode

        self.field_diagnostics = ParticleFieldEvaluation(
            particle_kernel=self.particle_kernel,
            max_n_particles=max_p,
            accumulator_dtype=self.accumulator_dtype,
        )
        self._flow_integrals: dict = {}
        self._discretization_health: dict = {}
        self._body_induced_fn = None
        self._stretch_time_step_size_warned: bool = False
        self._particles_removed_this_step = 0
        self._vortex_strength_removed_this_step = np.zeros(3, dtype=self.np_dtype)
        # Size of the last core-spreading moment projection, relative to |vortex_strength|.
        self.core_spreading_correction_relative = 0.0

    def _init_solvers(self, final_setup: VPMSetup) -> None:
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
        self.stabilization = StabilizationManager(
            StabilizationContext(
                particles=self.particles,
                physics=self.physics,
                field_diagnostics=self.field_diagnostics,
                config=self.setup.stabilization,
                compute_dtype=self.compute_dtype,
                np_dtype=self.np_dtype,
                flow_model=self.flow_model,
                step=lambda: self.step,
                time=lambda: self.time,
                time_step_size=lambda: self.time_step_size,
                replace_vortex_particles=self.replace_vortex_particles,
                set_particles_properties=self.set_particles_properties,
                remove_particles_by_bounds=self.remove_particles_by_bounds,
                particles_removed=lambda: self._particles_removed_this_step,
                set_particles_removed=lambda value: setattr(
                    self, "_particles_removed_this_step", value
                ),
                vortex_strength_removed=lambda: self._vortex_strength_removed_this_step,
                set_vortex_strength_removed=lambda value: setattr(
                    self, "_vortex_strength_removed_this_step", value
                ),
                domain_bounds_enforced=lambda: self._domain_bounds_enforced_this_step,
                set_domain_bounds_enforced=lambda value: setattr(
                    self, "_domain_bounds_enforced_this_step", bool(value)
                ),
            )
        )
        active = self.stabilization.active_mechanisms()
        if active:
            Logging.record(
                "stabilization",
                *(("  " + mechanism, "active") for mechanism in active),
            )
        self._init_optional_solvers(final_setup)
        # Detailed section timing forces a device barrier around every phase.
        # Make that diagnostic opt-in; the whole-step timer remains available in
        # normal production runs without serialising every kernel launch.
        self.profiler = RuntimeProfiler(
            enabled=os.environ.get("VPM_DETAILED_TIMING", "0") == "1",
            sync=ti.sync,
        )
        self._domain_bounds_enforced_this_step = False
        self.wall_time = 0.0
        # The step algorithm lives in the stepper; this facade drives it.
        self.stepper = EvolutionStepper(self)
        # Panel/VLM coupling orchestration runs inside the step.
        self.coupling = CouplingStepper(self)

    def _setup_vlm_solver(self) -> None:
        """Configure VLM solver coupling: mesh generation, force config, stability check."""
        self.vlm_solver.ensure_mesh_generated()
        if getattr(self.vlm_solver, "lattice", None) is not None:
            Logging.record("vlm", ("panels", f"{self.vlm_solver.lattice.n_panels:,}"))
            self.vlm_solver.check_coupling_stability(
                self.time_step_size, getattr(self.setup, "freestream_velocity", None)
            )

    def _init_optional_solvers(self, final_setup) -> None:
        """Initialize optional sub-solvers (panel, VLM) with error handling."""
        self.panel_solver = getattr(final_setup, "panel_solver", None)
        if self.panel_solver is not None:
            try:
                bodies = getattr(final_setup, "bodies", ())
                lattice = getattr(self.panel_solver, "lattice", None)
                if bodies and (lattice is None or lattice.n_panels == 0):
                    for body in bodies:
                        stl_path = Path(body.stl)
                        if not stl_path.is_absolute():
                            stl_path = self.case_dir / stl_path
                        self.panel_solver.add_surface(
                            uid=body.uid,
                            stl_path=str(stl_path),
                            kinematics=body.kinematics,
                            group_id=body.group_id,
                            translation=body.translation,
                            rotation_degrees=body.rotation_degrees,
                            rotation_centre=body.rotation_centre,
                            reference_area=body.reference_area,
                        )
                self.panel_solver.initialize(force=True)
                scope = getattr(self.panel_solver, "coupling_scope", "full")
                self._pressure_body_induced_fn = self.panel_solver.compute_induced_velocity
                if scope in ("full", "vpm_boundary_condition"):
                    self.set_body_induced_velocity(self.panel_solver.compute_induced_velocity)
                else:
                    self.set_body_induced_velocity(None)
                if scope == "full":
                    # Only "full" deflects particle trajectories, and it does so
                    # at every RK stage, so give it the device-resident hook.
                    self.physics.body_velocity_field = (
                        self.panel_solver.accumulate_induced_velocity_on_field
                    )
                else:
                    self.physics.body_velocity = None
                    self.physics.body_velocity_field = None
            except Exception as e:
                raise RuntimeError(f"Failed to initialize panel solver: {e}") from e

        if final_setup.vlm is None:
            self.vlm_solver = None
        else:
            self._require_consistent_molecular_viscosity(final_setup.viscous, final_setup.vlm)
            from ..boundary_elements.vlm.solver.vlm_solver import VLMSolver

            self.vlm_solver = VLMSolver(final_setup.vlm)

        if self.vlm_solver is not None:
            self._vpm_velocity_at_vlm = None
            self._vlm_velocity_at_vpm = None
            try:
                self._setup_vlm_solver()
            except Exception as e:
                Logging.warning(f"component=VLM status=initialization_failed error={e!r}")

    @staticmethod
    def _require_consistent_molecular_viscosity(viscous_cfg, vlm_setup) -> None:
        """Require one molecular kinematic viscosity in an attached VPM/VLM system."""
        scheme = getattr(viscous_cfg, "scheme", "NONE")
        if scheme == "NONE":
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

    def export_diagnostics_csv(self, filename: str) -> None:
        """Export diagnostics history to CSV for offline analysis."""
        self.io.export_diagnostics_csv(self._diagnostics_history, filename)

    @classmethod
    def from_setup_file(cls, filename: str) -> "VPMSolver":
        """Create a solver from a JSON configuration file."""
        setup = VPMSetup.load_from_file(filename)
        return cls(setup=setup)

    def save_setup(self, filename: str) -> None:
        """Save the current solver configuration to a JSON file."""
        self.io.save_setup(filename)

    # Basic protocol

    def __len__(self) -> int:
        """Return the number of particles in the system."""
        return len(self.particles)

    def __getitem__(self, index: int):
        """Access particle data by index."""
        return self.particles[index]

    def __iter__(self):
        """Iterate over all particles."""
        for i in range(len(self)):
            yield self[i]

    def __str__(self) -> str:
        """Return a formatted string summarizing the solver state."""
        return Logging.solver_summary(self)

    # Time stepping

    def print_timing(self) -> None:
        """Print cumulative runtime-profiler statistics."""
        self.profiler.set_particle_count(self.particles.n_particles_total)
        self.profiler.report()

    def advance(self, *, defer_output: bool = False) -> None:
        """Advance the VPM solution by one time step.

        The step algorithm (velocity/gradient preparation, advection,
        stretching, coupled inviscid integration, viscous diffusion, operator
        splitting, and the in-step stabilization phases) is owned by the
        :class:`~source.solvers.vpm.core.evolution.EvolutionStepper`; this
        facade method delegates to it. Coupled drivers may set
        ``defer_output=True`` and write scheduled output after synchronizing
        the particle state at the new time level.
        """
        self.stepper.advance(defer_output=defer_output)

    def record_diagnostics(self, *, refresh_fields: bool = False) -> None:
        """Evaluate and log diagnostics for the current particle state.

        Set ``refresh_fields=True`` when velocity, gradients, or LES viscosity are
        stale for the current state.
        """
        if refresh_fields:
            self.stepper._update_velocity_and_gradients()
            self.stepper._update_les_state()
            self.stabilization.update_residual_viscosity()
        self._update_all_flow_integrals()
        self.log_diagnostics()

    def log_diagnostics(self) -> None:
        """Log the most recently evaluated flow diagnostics and run samplers."""

        Logging.flow_diagnostics(self)

        if getattr(self.setup, "export_flow_integrals", True):
            self._export_flow_integrals_csv()

        if self.turbulence_model is not None:
            Logging.les_diagnostics(self)

        self._execute_samplers()

    def _export_flow_integrals_csv(self) -> None:
        """Append one row of flow integrals to ``<case_dir>/samples/flow_integrals.csv``.

        Thin wrapper that delegates the CSV export to the ``SolverIO`` manager
        (which owns all exports).
        """
        self.io.export_flow_integrals_csv(self)

    def _execute_samplers(self) -> None:
        """Execute samplers without an explicit schedule at logging cadence."""
        SamplerExecutor.execute(self)

    def execute_scheduled_samplers(self) -> None:
        """Execute due time- or step-scheduled field samplers."""
        SamplerExecutor.execute(self, scheduled_only=True)

    def execute_final_samplers(self) -> None:
        """Execute the final-only samplers declared by the immutable setup."""
        SamplerExecutor.execute(self, self.setup.final_samplers, scheduled_only=None)

    def _prepare_sampler_context(self, sampler_entry, samples_dir):
        """Delegate to SamplerExecutor."""
        return SamplerExecutor._prepare_context(sampler_entry, samples_dir)

    def _save_sampler_output(self, sampler, name_prefix, solution_dir, seq_num):
        """Delegate to SamplerExecutor."""
        SamplerExecutor._save_output(
            sampler,
            self,
            name_prefix,
            solution_dir,
            seq_num,
            self.time,
            self.step,
        )

    def _write_pvd_file(self, output_dir, name_prefix, entries):
        """Delegate to SamplerExecutor."""
        SamplerExecutor._write_pvd(output_dir, name_prefix, entries)

    # Particle properties
    def _get_particle_field(self, method_name: str) -> np.ndarray:
        """Generic helper to get particle field data via cpu() methods."""
        return getattr(self.particles, f"{method_name}_cpu")()

    @property
    def particle_position(self) -> np.ndarray:
        """Particle position with shape ``(N, 3)`` [m]."""
        return self._get_particle_field("position")

    @property
    def particle_velocity(self) -> np.ndarray:
        """Particle velocity with shape ``(N, 3)`` [m/s]."""
        return self._get_particle_field("velocity")

    @property
    def particle_core_radius(self) -> np.ndarray:
        """Particle core radius with shape ``(N,)`` [m]."""
        return self._get_particle_field("core_radius")

    @property
    def particle_volume(self) -> np.ndarray:
        """Particle volume with shape ``(N,)`` [m³]."""
        return self._get_particle_field("particle_volume")

    @property
    def particle_group_id(self) -> np.ndarray:
        """Particle group identifiers with shape ``(N,)``."""
        return self._get_particle_field("group_id")

    @property
    def particle_zone_id(self) -> np.ndarray:
        """Particle zone identifiers with shape ``(N,)``."""
        return self._get_particle_field("zone_id")

    @property
    def particle_kinematic_viscosity(self) -> np.ndarray:
        """Particle molecular kinematic_viscosity with shape ``(N,)`` [m²/s]."""
        return self._get_particle_field("kinematic_viscosity")

    @property
    def particle_eddy_viscosity(self) -> np.ndarray:
        """Particle eddy viscosity with shape ``(N,)`` [m²/s]."""
        return self._get_particle_field("eddy_viscosity")

    @property
    def particle_effective_viscosity(self) -> np.ndarray:
        """Particle effective kinematic_viscosity with shape ``(N,)`` [m²/s]."""
        return self._get_particle_field("effective_viscosity")

    @property
    def particle_velocity_gradient(self) -> np.ndarray:
        """Get particle velocity gradients array."""
        return self._get_particle_field("velocity_gradient")

    @property
    def particle_strain_rate(self) -> np.ndarray:
        """Get particle strain rate tensors array."""
        return self._get_particle_field("strain_rate")

    @property
    def freestream_velocity(self) -> np.ndarray:
        """Uniform background velocity [m/s]."""
        return self.particles.velocity_background_cpu()

    @property
    def particle_vorticity(self) -> np.ndarray:
        """Get particle vorticity array."""
        return self._get_particle_field("vorticity")

    @property
    def particle_vortex_strength(self) -> np.ndarray:
        """Particle vortex-strength vectors with shape ``(N, 3)`` [m³/s]."""
        return self._get_particle_field("vortex_strength")

    # Flow diagnostics
    def _update_all_flow_integrals(self) -> None:
        """Recompute flow integrals and associated diagnostic histories."""
        self._flow_integrals = self.field_diagnostics.compute_flow_integrals(
            self.particles, self.time
        )
        self._update_discretization_health()
        self._record_vortex_centroid_history()
        self._record_time_history()
        self._record_vlm_diagnostics()

    def _update_discretization_health(self) -> None:
        """Refresh particle-resolution and field-quality diagnostics."""
        if not getattr(self.setup, "export_discretization_health", True):
            return
        if self.particles.n_particles_total == 0:
            self._discretization_health = {}
            return
        self._discretization_health = discretization_health(
            self.particle_position,
            self.particle_vortex_strength,
            self.particle_core_radius,
        )

    def _record_vortex_centroid_history(self) -> None:
        """Record the vortex-strength-magnitude-weighted particle centroid."""
        ParticleFieldEvaluation.record_vortex_centroid_history(
            self._diagnostics_history,
            self.particle_position,
            self.particle_vortex_strength,
        )

    def _record_time_history(self) -> None:
        """Delegate to VLMDiagnostics."""
        ft_hist = self._diagnostics_history.get("time", [])
        observed_time_step_size = self.time - ft_hist[-1] if len(ft_hist) >= 1 else 0.0
        VLMDiagnostics.record_time(self._diagnostics_history, self.time, observed_time_step_size)

    def _record_vlm_diagnostics(self) -> None:
        """Delegate to VLMDiagnostics."""
        sample_subdirectory = getattr(self.setup, "sample_subdirectory", None)
        VLMDiagnostics.record_vlm_diagnostics(
            self.vlm_solver,
            self.particles,
            self.particle_vortex_strength,
            self._diagnostics_history,
            self.step,
            self.time,
            self.case_dir,
            sample_subdirectory,
        )
        VLMLoadingDistribution.record_loading_distributions(
            self.vlm_solver,
            self._diagnostics_history,
            self.step,
            self.time,
            self.case_dir,
            sample_subdirectory,
        )

    def _export_vlm_forces_to_csv(
        self,
        forces,
        bound_vortex_strength,
        wake_vortex_strength,
        max_leading_edge_suction_parameter,
        n_particles_total,
    ):
        """Delegate to VLMDiagnostics."""
        VLMDiagnostics.export_forces_csv(
            self.vlm_solver,
            forces,
            bound_vortex_strength,
            wake_vortex_strength,
            max_leading_edge_suction_parameter,
            n_particles_total,
            self.time,
            self.step,
            self.case_dir,
            getattr(self.setup, "sample_subdirectory", None),
        )

    @property
    def total_kinetic_energy(self) -> float:
        """Total kinetic energy per unit density."""
        return self._flow_integrals.get("total_kinetic_energy", 0.0)

    @property
    def total_helicity(self) -> float:
        """Total helicity."""
        return self._flow_integrals.get("total_helicity", 0.0)

    @property
    def total_enstrophy(self) -> float:
        """Total enstrophy."""
        return self._flow_integrals.get("total_enstrophy", 0.0)

    @property
    def viscous_kinetic_energy_rate(self) -> float:
        """Signed viscous contribution to the kinetic-energy rate [J/s]."""
        return self._flow_integrals.get("viscous_kinetic_energy_rate", 0.0)

    @property
    def kinetic_energy_rate(self) -> float:
        """Signed finite-difference kinetic-energy rate [J/s]."""
        return self._flow_integrals.get("kinetic_energy_rate", 0.0)

    @property
    def net_vortex_strength(self) -> np.ndarray:
        """Total particle vortex-strength vector [m³/s]."""
        return self._flow_integrals.get("net_vortex_strength", np.array([0.0, 0.0, 0.0]))

    @property
    def vortex_strength_magnitude_sum(self) -> float:
        """Sum of particle vortex-strength magnitudes [m³/s]."""
        return self._flow_integrals.get("vortex_strength_magnitude_sum", 0.0)

    @property
    def total_linear_impulse(self) -> np.ndarray:
        """Return the current linear impulse, recomputed from the active particle field."""
        integrals = self.field_diagnostics.compute_flow_integrals(
            self.particles, self.time, record_history=False
        )
        return integrals.get("linear_impulse", np.array([0.0, 0.0, 0.0]))

    @property
    def total_angular_impulse(self) -> np.ndarray:
        """Total angular impulse."""
        return self._flow_integrals.get("angular_impulse", np.array([0.0, 0.0, 0.0]))

    @property
    def vortex_centroids_by_group(self) -> dict[int, np.ndarray]:
        """Vortex-strength-magnitude-weighted centroid for each particle group."""
        return self.field_diagnostics.compute_vortex_centroids_by_group(self.particles)

    @property
    def vortex_centroid(self) -> np.ndarray:
        """Global vortex-strength-magnitude-weighted particle centroid."""
        return self.field_diagnostics.compute_vortex_centroid(self.particles)

    def compute_forces(
        self, density: float = 1.225, reference_speed: float | None = None
    ) -> dict[str, np.ndarray | float]:
        """Compute aerodynamic force from the configured VLM model.

        Args:
            density: Fluid density [kg/m³].
            reference_speed: Reference speed [m/s]. Uses the background speed when omitted.

        Returns:
            Force components and the force-evaluation method.
        """
        if self.vlm_solver is None:
            raise RuntimeError("Force evaluation requires a VLM setup")
        method = self.vlm_solver.force.method

        if method == "KUTTA_JOUKOWSKI":
            return self._compute_forces_kutta_joukowski(density, reference_speed)
        else:
            raise ValueError(f"Unknown force method: {method}")

    def _compute_forces_kutta_joukowski(
        self, density: float, reference_speed: float | None
    ) -> dict[str, np.ndarray | float]:
        """Compute forces via the Kutta-Joukowski theorem. Delegates to VLMForceEvaluator."""
        return VLMForceEvaluator.compute_kutta_joukowski(
            self.vlm_solver, self.freestream_velocity, density, reference_speed
        )

    # Per-particle diagnostics
    def compute_kinetic_energies(self) -> np.ndarray:
        """Return per-particle kinetic-energy contributions."""
        return self.field_diagnostics.compute_particles_kinetic_energy(self.particles)

    def compute_helicities(self) -> np.ndarray:
        """Return per-particle helicity contributions."""
        return self.field_diagnostics.compute_particles_helicity(self.particles)

    def compute_enstrophies(self) -> np.ndarray:
        """Return per-particle enstrophy contributions."""
        return self.field_diagnostics.compute_particles_enstrophy(self.particles)

    # Field evaluation
    def compute_vorticity_at_points(self, evaluation_position: np.ndarray) -> np.ndarray:
        """Evaluate vorticity at arbitrary target points.

        Args:
            evaluation_position: Target coordinates with shape ``(N, 3)`` [m].

        Returns:
            Vorticity vectors with shape ``(N, 3)`` [1/s].
        """
        return self.physics.compute_target_vorticity(self.particles, evaluation_position)

    def compute_velocity_at_points(
        self,
        evaluation_position: np.ndarray,
        include_freestream: bool = True,
        zone_mask: np.ndarray | None = None,
        include_body: bool = True,
    ) -> np.ndarray:
        """Evaluate velocity at arbitrary target points.

        Args:
            evaluation_position: Target coordinates with shape ``(N, 3)`` [m].
            include_freestream: Include the uniform background velocity.
            zone_mask: Optional mask selecting contributing particles.
            include_body: Include boundary-element body induction.

        Returns:
            Velocity vectors with shape ``(N, 3)`` [m/s].
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
        velocity = np.asarray(particle_velocity, dtype=self.np_dtype).reshape(-1, 3)
        if len(velocity) != len(points):
            raise ValueError("target velocity and position counts must match")

        if self.n_sources > 0:
            n_targets = len(points)
            self.physics._resize_target_fields(n_targets)
            target_position_field = self.physics.target_position
            target_velocity_field = self.physics.target_velocity

            # Fixed-shape buffers avoid persistent staging allocations.
            self.physics._upload_vector_array(points, target_position_field, n_targets)
            self.physics._upload_vector_array(velocity, target_velocity_field, n_targets)

            self.physics.kernels["compute_target_source_velocity_kernel"](
                target_position_field,
                self.source_position,
                self.source_strength,
                self.source_core_radius,
                target_velocity_field,
                n_targets,
                self.n_sources,
            )
            velocity = self.physics.extract_target_velocity(n_targets)

        body_fn = self._body_induced_fn
        if include_body and body_fn is not None:
            velocity = velocity + np.asarray(body_fn(points), dtype=velocity.dtype).reshape(
                velocity.shape
            )

        return velocity

    def _nonparticle_target_velocity(self, evaluation_position: np.ndarray) -> np.ndarray:
        """Return only regularized-source and body-potential target velocity."""
        points = np.asarray(evaluation_position, dtype=np.float64).reshape(-1, 3)
        return self._add_target_velocity_corrections(
            points,
            np.zeros((len(points), 3), dtype=self.np_dtype),
            include_body=True,
        )

    def set_body_induced_velocity(self, fn) -> None:
        """Set the optional boundary-element velocity callback.

        The callback must map an ``(N, 3)`` point array to an ``(N, 3)`` velocity
        array. Pass ``None`` to disable body induction.
        """
        self._body_induced_fn = fn
        self.physics.body_velocity = fn
        if fn is None:
            # Never leave the device hook installed for a disabled body.
            self.physics.body_velocity_field = None

    def refresh_boundary_element_solution(self) -> None:
        """Make a synchronized panel solution consistent with current particles.

        A ``vpm_boundary_condition`` panel does not participate in particle
        evolution, and a ``full`` panel was last solved against the particle
        state at the top of the VPM step. External couplers may replace the
        particle cloud at fixed physical time, so the panel's harmonic/body
        correction must be re-solved against the replaced state before the
        next boundary trace or advection step evaluates it.
        """
        panel = self.panel_solver
        if panel is None or getattr(panel, "coupling_scope", "full") not in (
            "full",
            "vpm_boundary_condition",
        ):
            return
        panel.refresh_coupled_solution(
            particles=self.particles,
            physics=self.physics,
            freestream_velocity=self.freestream_velocity,
            time=self.time,
        )

    def set_surface_sources(
        self, position: np.ndarray, vortex_strength: np.ndarray, core_radius: np.ndarray
    ) -> None:
        """Set regularized source particles used for body-blockage corrections."""
        self.n_sources = len(position)
        if self.n_sources > MAX_SOURCES:
            Logging.warning(
                f"component=sources requested={self.n_sources} limit={MAX_SOURCES} status=clipped"
            )
            self.n_sources = MAX_SOURCES

        n = self.n_sources
        # Taichi ``from_numpy`` requires the allocated shape.
        position_buffer = np.zeros((MAX_SOURCES, 3), dtype=self.np_dtype)
        str_buf = np.zeros(MAX_SOURCES, dtype=self.np_dtype)
        core_radius_buffer = np.zeros(MAX_SOURCES, dtype=self.np_dtype)
        position_buffer[:n] = np.asarray(position[:n], dtype=self.np_dtype)
        str_buf[:n] = np.asarray(vortex_strength[:n], dtype=self.np_dtype)
        core_radius_buffer[:n] = np.asarray(core_radius[:n], dtype=self.np_dtype)
        self.source_position.from_numpy(position_buffer)
        self.source_strength.from_numpy(str_buf)
        self.source_core_radius.from_numpy(core_radius_buffer)

    def _compute_particle_velocity_gradient_at_points(
        self, evaluation_position: np.ndarray
    ) -> np.ndarray:
        """Evaluate the particle-induced ``∇u`` at arbitrary points.

        Returns an ``(N, 9)`` array in row-major Jacobian order,
        ``J[i, j] = d(u_i)/d(x_j)``.  This kernel differentiates the vortex
        particle velocity; the uniform freestream has zero gradient and the
        optional regularized sources and body-potential callback are not
        differentiated here.
        """
        return self.physics.compute_target_velocity_gradient(self.particles, evaluation_position)

    def compute_velocity_gradient_at_points(
        self, evaluation_position: np.ndarray, *, particle_spacing: float
    ) -> np.ndarray:
        """Evaluate the Jacobian of the body-complete velocity field.

        Vortex-particle induction is differentiated analytically. Regularized
        source and body-callback contributions are differentiated by centred
        differences with a step scaled by the coupling lattice spacing ``particle_spacing``.
        The result has shape ``(N, 3, 3)`` and convention
        ``J[i,j] = d(u_i)/d(x_j)``.
        """
        points = np.asarray(evaluation_position, dtype=np.float64).reshape(-1, 3)
        # Use the same approximation as the target-velocity trace.  In a
        # treecode run, mixing a direct Jacobian with a treecode velocity is
        # both prohibitively expensive at coupling faces and inconsistent with
        # the boundary trace from which the normal velocity is taken.
        if self.physics.velocity_method == "TREECODE":
            gradient = np.asarray(
                self.physics.compute_target_velocity_gradient_hierarchical(
                    self.particles,
                    points,
                    theta=self.physics.velocity_theta,
                ),
                dtype=np.float64,
            ).reshape(-1, 3, 3)
        else:
            gradient = np.asarray(
                self._compute_particle_velocity_gradient_at_points(points), dtype=np.float64
            ).reshape(-1, 3, 3)
        return self._add_nonparticle_target_gradient(
            points, gradient, particle_spacing=particle_spacing
        )

    def _add_nonparticle_target_gradient(
        self, points: np.ndarray, particle_gradient: np.ndarray, *, particle_spacing: float
    ) -> np.ndarray:
        """Differentiate only the source and body corrections by centred differences."""
        gradient = np.asarray(particle_gradient, dtype=np.float64).reshape(-1, 3, 3).copy()
        if (self._body_induced_fn is None and self.n_sources == 0) or len(points) == 0:
            return gradient

        step = max(1.0e-6, 1.0e-3 * float(particle_spacing))
        for axis in range(3):
            offset = np.zeros(3, dtype=np.float64)
            offset[axis] = step
            plus = self._nonparticle_target_velocity(points + offset)
            minus = self._nonparticle_target_velocity(points - offset)
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
        """
        points = np.asarray(evaluation_position, dtype=np.float64).reshape(-1, 3)
        if self.physics.velocity_method == "TREECODE":
            velocity, gradient = self.physics.compute_target_velocity_and_gradients_hierarchical(
                self.particles,
                points,
                theta=self.physics.velocity_theta,
                include_freestream=True,
            )
        else:
            velocity = self.physics.compute_target_velocity(
                self.particles,
                points,
                include_freestream=True,
                zone_mask=None,
            )
            gradient = self.physics.compute_target_velocity_gradient(self.particles, points)
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
        """Return velocity and tangential normal-gradient trace at points.

        The mixed FVM boundary condition does not consume the full nine-component
        Jacobian.  Particle induction is still evaluated by the configured fused
        target operation, while source/body terms use only the two centred samples
        along each face normal instead of three coordinate-direction pairs.
        """
        points = np.asarray(evaluation_position, dtype=np.float64).reshape(-1, 3)
        face_normals = np.asarray(normal, dtype=np.float64).reshape(-1, 3)
        if face_normals.shape != points.shape:
            raise ValueError("normal count does not match target position")
        normal_magnitude = np.linalg.norm(face_normals, axis=1)
        if np.any(~np.isfinite(face_normals)) or np.any(normal_magnitude <= 0.0):
            raise ValueError("target normal must be finite and non-zero")
        unit_normals = face_normals / normal_magnitude[:, None]

        if self.physics.velocity_method == "TREECODE":
            velocity, gradient = self.physics.compute_target_velocity_and_gradients_hierarchical(
                self.particles,
                points,
                theta=self.physics.velocity_theta,
                include_freestream=True,
            )
        else:
            velocity = self.physics.compute_target_velocity(
                self.particles,
                points,
                include_freestream=True,
                zone_mask=None,
            )
            gradient = self.physics.compute_target_velocity_gradient(self.particles, points)

        complete_velocity = self._add_target_velocity_corrections(
            points, velocity, include_body=True
        )
        normal_velocity_gradient = np.einsum(
            "fij,fj->fi", np.asarray(gradient, dtype=np.float64).reshape(-1, 3, 3), unit_normals
        )
        if self._body_induced_fn is not None or self.n_sources > 0:
            step = max(1.0e-6, 1.0e-3 * float(particle_spacing))
            plus = self._nonparticle_target_velocity(points + step * unit_normals)
            minus = self._nonparticle_target_velocity(points - step * unit_normals)
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

    def compute_pressure_gradient_at_points(
        self,
        evaluation_position: np.ndarray,
        density: float = 1.0,
        kinematic_viscosity: float | None = None,
        include_viscous: bool = True,
        include_temporal: bool = True,
        include_freestream: bool = True,
        particle_spacing: float | None = None,
        temporal_method: str = "lagrangian",
        velocity_previous: np.ndarray | None = None,
        time_step_size: float | None = None,
        return_velocity: bool = False,
        treecode_theta: float | None = None,
        include_body: bool = True,
    ) -> dict | tuple[dict, np.ndarray]:
        """Evaluate pressure-gradient terms at arbitrary points.

        The result contains the total pressure gradient and its convective, viscous,
        and temporal contributions. ``temporal_method='eulerian'`` requires
        ``velocity_previous`` and ``time_step_size`` when the temporal term is enabled.
        ``include_body=False`` omits the optional boundary-element velocity from
        the hierarchical pressure evaluation while retaining particles and the
        configured freestream.
        """
        if kinematic_viscosity is None:
            kinematic_viscosity = (
                float(np.mean(self.particle_kinematic_viscosity))
                if self.particles.n_particles_total > 0
                else 1e-5
            )
        if not hasattr(self, "_pressure_physics"):
            from source.solvers.vpm.physics.pressure import PressurePhysics

            self._pressure_physics = PressurePhysics(
                particle_kernel=self.particle_kernel,
                max_n_particles=int(self.setup.max_n_particles),
                accumulator_dtype=self.accumulator_dtype,
            )
        if treecode_theta is not None:
            body_fn = None
            if include_body:
                body_fn = getattr(
                    self,
                    "_pressure_body_induced_fn",
                    self._body_induced_fn,
                )
            return self._pressure_physics.compute_target_pressure_gradient_hierarchical(
                self.particles,
                evaluation_position,
                density=density,
                kinematic_viscosity=kinematic_viscosity,
                include_viscous=include_viscous,
                include_temporal=include_temporal,
                include_freestream=include_freestream,
                temporal_method=temporal_method,
                velocity_previous=velocity_previous,
                time_step_size=time_step_size,
                particle_spacing=particle_spacing,
                return_velocity=return_velocity,
                theta=treecode_theta,
                freestream_velocity=self.freestream_velocity,
                body_fn=body_fn,
            )

        if self.particles.n_particles_total > 0:
            self.physics.compute_velocity_gradients(self.particles)
        return self._pressure_physics.compute_target_pressure_gradient_components(
            self.particles,
            evaluation_position,
            density=density,
            kinematic_viscosity=kinematic_viscosity,
            include_viscous=include_viscous,
            include_temporal=include_temporal,
            laplacian_spacing=particle_spacing,
            include_freestream=include_freestream,
            temporal_method=temporal_method,
            velocity_previous=velocity_previous,
            time_step_size=time_step_size,
            return_velocity=return_velocity,
        )

    # Diagnostics
    def info(self):
        """Print a summary of the solver configuration and current state."""
        info_str = Logging.solver_info(self)
        Logging.message(info_str)

    # Particle management
    def remove_particles(
        self, particle_indices: list[int] | None = None, remove_all: bool = False
    ) -> None:
        """Remove selected particles and track removed vortex strength for diagnostics."""
        if particle_indices is not None and len(particle_indices) > 0:
            # Reduce removed vortex strength on device.
            vortex_strength_removed, _ = self.particles.subset_moments(particle_indices)
            self._particles_removed_this_step = len(particle_indices)
            self._vortex_strength_removed_this_step = vortex_strength_removed

        elif remove_all:
            # Sum removed vortex strength on device.
            vortex_strength_removed = self.particles.net_vortex_strength()

            self._particles_removed_this_step = len(self.particles)
            self._vortex_strength_removed_this_step = vortex_strength_removed

        else:
            self._particles_removed_this_step = 0
            self._vortex_strength_removed_this_step = np.zeros(3)

        # Trim the stabilization lineage references to match the removed set.
        if remove_all:
            self.stabilization.on_removal(remove_all=True)
        elif particle_indices is not None and len(particle_indices) > 0:
            self.stabilization.on_removal(indices=particle_indices)

        self.particles.remove_vortex_particles(indices=particle_indices, remove_all=remove_all)

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
            loading=getattr(self, "_loading_numerical_state", False),
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

    def update_particle_vortex_strength(
        self,
        mask: np.ndarray,
        vortex_strength_increment: np.ndarray,
    ) -> None:
        """Apply an in-place vortex-strength delta to a masked particle subset."""
        self.particles.update_vortex_strength_masked(mask, vortex_strength_increment)

    def notify_external_particle_mutation(self) -> None:
        """Schedule VPM-owned GBD regeneration before the next evolution step."""
        if self.viscous_scheme == "GBD":
            self._is_particle_regeneration_pending = True

    def _print_time_step_validation_summary(self, results: dict) -> None:
        Logging.time_step_validation_summary(results)

    def load_particle_field(
        self, particle_file_name: str, remove_current_particles: bool = False
    ) -> None:
        """Load particle field from file."""
        self.io.load_particle_field(particle_file_name, remove_current_particles)

    def set_time_step_size(self, time_step_size: float) -> None:
        """Set the positive simulation time-step size [s]."""
        if time_step_size <= 0:
            raise ValueError("Time step size must be positive")
        self.time_step_size = time_step_size

    def get_time_step_size(self) -> float:
        """Return the current time-step size [s]."""
        return self.time_step_size

    @staticmethod
    @staticmethod
    def _validate_particle_property(
        prop_name: str,
        prop_value,
        n_particles_total: int,
    ) -> np.ndarray:
        "Validate one canonical particle-property array."
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

    def set_particles_properties(self, **properties) -> None:
        "Update canonical particle fields after validating shape and finiteness."
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

        self.particles._cache_step = -1

        property_names = list(properties)
        Logging.record(
            "particle fields updated",
            ("fields", f"{len(property_names):,}"),
            *(("  " + name, "updated") for name in property_names),
        )

    # State and restart

    def save_state(self, filename: str = "solution/solver_state") -> None:
        """Save a restartable numerical state and its configuration."""

        if checkpoint_dir := os.path.dirname(filename):
            os.makedirs(checkpoint_dir, exist_ok=True)

        self._refresh_checkpoint_particle_fields()
        CheckpointManager.write_checkpoint(self, filename, append_step=False, verbose=False)

        config_file = f"{filename}.config.json"
        CheckpointManager.write_configuration(self, config_file)

        Logging.record(
            "checkpoint saved",
            ("base", str(filename)),
            ("data", f"{filename}.h5"),
            ("visualization", f"{filename}.xdmf"),
            ("configuration", str(config_file)),
        )

    def save_numerical_state(self, filename: str) -> None:
        """Save numerical state for a caller that already owns configuration."""
        self._refresh_checkpoint_particle_fields()
        CheckpointManager.write_checkpoint(self, filename, append_step=False, verbose=False)

    def load_numerical_state(self, filename: str) -> None:
        """Restore numerical state into this configured VPM solver."""
        path = filename if filename.endswith(".h5") else f"{filename}.h5"
        CheckpointManager.load_numerical_state(self, path)

    def write_checkpoint(self, checkpoint_name: str = "checkpoint") -> None:
        """Write the solver state to a specified checkpoint file."""
        self._refresh_checkpoint_particle_fields()
        CheckpointManager.write_checkpoint(self, checkpoint_name, verbose=True)

    def _write_checkpoint(self) -> None:
        """Write a scheduled solver checkpoint when one is due."""
        if not self.io.should_checkpoint():
            return

        self._refresh_checkpoint_particle_fields()

        self.io.write_checkpoint()

    def _refresh_checkpoint_particle_fields(self) -> None:
        """Refresh particle fields that are expected to be available in checkpoints."""
        N = self.particles.n_particles_total
        if N > 50_000:
            return
        if N > 0:
            self.physics.compute_self_induced_velocity(
                self.particles.position,
                self.particles.vortex_strength,
                self.particles.core_radius,
                self.particles.velocity,
                self.particles.velocity_background,
                N,
            )
        self.physics.compute_vorticities(self.particles)
        if self.checkpoint_store_velocity_gradient and self.flow_model != "POTENTIAL":
            self.stepper._update_velocity_gradients()

    @staticmethod
    def continue_from_checkpoint(checkpoint_name: str | None = None) -> "VPMSolver | None":
        """Restore a solver from an HDF5 checkpoint and its saved configuration."""
        if not CheckpointManager.validate_checkpoint(checkpoint_name):
            raise ValueError(f"Checkpoint validation failed for: {checkpoint_name}")

        Logging.record("checkpoint loading", ("base", str(checkpoint_name)))

        try:
            hdf5_file = f"{checkpoint_name}.h5"
            config_file = f"{checkpoint_name}.config.json"

            if not os.path.exists(hdf5_file):
                raise FileNotFoundError(f"Numerical data file not found: {hdf5_file}")
            if not os.path.exists(config_file):
                raise FileNotFoundError(f"Configuration file not found: {config_file}")

            setup = CheckpointManager.load_configuration(config_file)
            restored_solver = VPMSolver(setup=setup)
            CheckpointManager._load_numerical_data(restored_solver, hdf5_file)
        except Exception as e:
            raise RuntimeError(f"Restore failed: {e}") from e

        restored_solver.field_diagnostics.reset_energy_history()

        restored_solver._update_all_flow_integrals()

        Logging.record(
            "checkpoint loaded",
            ("time", f"{restored_solver.time:.6e}", "s"),
            ("step", f"{restored_solver.step:,}"),
            ("particles", f"{restored_solver.particles.n_particles_total:,}"),
            ("backend", str(restored_solver.setup.compute_device)),
        )

        return restored_solver

    def export_state(self, filename: str, **kwargs):
        """Export solver state for visualization and post-processing."""
        self.io.export_state(filename, **kwargs)

    # Particle updates

    def set_freestream_velocity(self, velocity: list[float] | np.ndarray) -> None:
        """Set the uniform background velocity vector [m/s]."""
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

    def set_velocity_override(self, fn) -> None:
        """Set an optional advection-velocity callback evaluated at each RK stage.

        The callback receives particle position and Biot–Savart velocity and returns
        the velocity used for advection. It does not alter the stretching gradient.
        """
        self.physics.velocity_override = fn

    # Particle control

    def remove_particles_by_bounds(self, bounds: list, invert_selection: bool = False) -> int:
        """Remove particles inside or outside an axis-aligned bounding box.

        Set ``invert_selection=True`` to keep particles inside the box and remove
        those outside it. Returns the number removed.
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
        """Remove particles below a fraction of the global maximum strength."""
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
