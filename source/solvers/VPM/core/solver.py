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

from source.solvers.VPM.particles.container import Particles
from source.solvers.VPM.turbulence.turbulence import ParticlesLES

from ..boundary_elements.vlm.solver.diagnostics import VLMDiagnostics
from ..boundary_elements.vlm.solver.forces import VLMForceEvaluator
from ..boundary_elements.vlm.solver.loading_distribution import VLMLoadingDistribution
from ..config.constants import MAX_PARTICLES, MAX_SOURCES
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
        self._particle_regeneration_pending = False
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
                Logging.message(
                    f"[RWM] WARNING: user dt = {self.time_step_size:.4e} s "
                    f"> accuracy limit particle_spacing²/(4nu) = {rwm_max_time_step_size:.4e} s — "
                    f"random displacement √(2nuΔt) exceeds inter-particle spacing particle_spacing; "
                    f"vorticity gradients will be artificially smoothed."
                )
            self._rwm_time_step_size_info = (
                f"RWM accuracy limit particle_spacing²/(4nu) = {rwm_max_time_step_size:.4e} s "
                f"(particle_spacing = {vc.particle_spacing:.3e} m, "
                f"nu = {vc.kinematic_viscosity:.3e} m²/s)."
            )

        # GBD explicit-diffusion stability criterion.
        if vc.scheme == "GBD" and vc.kinematic_viscosity is not None and vc.kinematic_viscosity > 0:
            max_time_step_size = vc.gbd_max_time_step_size()
            if self.time_step_size > max_time_step_size * (1.0 + 1e-6):
                Logging.message(
                    f"[GBD] WARNING: user dt = {self.time_step_size:.4e} s "
                    f"> CFL limit particle_spacing²/(6nu) = {max_time_step_size:.4e} s — "
                    f"explicit Laplacian may be UNSTABLE."
                )
            self._gbd_time_step_size_info = (
                f"GBD fires every step "
                f"(Δt = {self.time_step_size:.4e} s, "
                f"CFL max = {max_time_step_size:.4e} s)."
            )

        # Match the user step to an integer subdivision of the DVH increment.
        self._dvh_substeps: int = 1
        self._dvh_fire_counter: int = 0
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
                Logging.message(
                    f"[DVH] INFO: time step adjusted — "
                    f"user dt = {user_time_step_size:.4e} s → dt = Δt_d/{n_sub} = {substep_size:.4e} s "
                    f"(Δt_d = β·R_d²/(4nu) = {diffusion_time_step_size:.4e} s, β={_DVH_BETA}, "
                    f"R_d = {vc.dvh_support_radius_ratio}·particle_spacing = {vc.dvh_support_radius_ratio * vc.dvh_grid_spacing:.4e} m; "
                    f"DVH fires every {n_sub} step(s))."
                )
                self.time_step_size = substep_size
            self._dvh_substeps = n_sub
            self._dvh_time_step_size_info = (
                f"DVH fires every {n_sub} step(s) (dt = Δt_d/{n_sub} = {substep_size:.4e} s, "
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
        max_p = getattr(final_setup, "max_particles", MAX_PARTICLES)
        self.particles = Particles(max_particles=max_p, float_dtype=self.precision)
        self.physics = PhysicsEngine(
            particle_kernel=self.particle_kernel,
            max_particles=max_p,
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
                self.physics.configure_body_mask(getattr(final_setup, "body_stl", None))
            except Exception as exc:
                Logging.warning(f"Failed to configure DVH body mask: {exc}")

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
        self.source_positions = ti.Vector.field(3, dtype=self.compute_dtype, shape=MAX_SOURCES)
        self.source_strengths = ti.field(dtype=self.compute_dtype, shape=MAX_SOURCES)
        self.source_radii = ti.field(dtype=self.compute_dtype, shape=MAX_SOURCES)
        self.n_sources = 0
        if hasattr(self.setup, "freestream_velocity"):
            self.particles.set_freestream_velocity(np.array(self.setup.freestream_velocity))

    def _init_turbulence_and_adaptation(self, final_setup: VPMSetup) -> None:
        """Initialize LES turbulence, stretching settings, and diagnostics."""
        max_p = getattr(final_setup, "max_particles", MAX_PARTICLES)
        self.turbulence_model = None
        if self.flow_model == "LES":
            self.turbulence_model = ParticlesLES(
                model_name=final_setup.turbulence.model,
                max_particles=max_p,
                particle_kernel=self.particle_kernel,
                c_s=final_setup.turbulence.c_s,
                c_e=final_setup.turbulence.c_e,
                accumulator_dtype=self.accumulator_dtype,
            )
        self.stretching_enabled = final_setup.stretching.enabled
        self.stretching_mode = final_setup.stretching.mode

        self.field_diagnostics = ParticleFieldEvaluation(
            particle_kernel=self.particle_kernel,
            max_particles=max_p,
            accumulator_dtype=self.accumulator_dtype,
        )
        self._flow_integrals: dict = {}
        self._discretization_health: dict = {}
        self._body_induced_fn = None
        self._stretch_time_step_size_warned: bool = False
        self._particles_removed_this_step = 0
        self._vortex_strength_removed_this_step = np.zeros(3, dtype=self.np_dtype)
        # Size of the last core-spreading moment projection, relative to |Gamma|.
        self.core_spreading_correction_relative = 0.0

    def _init_solvers(self, final_setup: VPMSetup) -> None:
        """Initialize the stabilization master and the optional sub-solvers."""

        # Time histories consumed by export_diagnostics_csv and the VLM report.
        self._diagnostics_history: dict = {
            "time": [],
            "vpm_total_vortex_strength": [],
            "vpm_total_vortex_strength_magnitude": [],
            "fvm_total_vortex_strength": [],
            "fvm_total_vortex_strength_magnitude": [],
            "interpolated_total_vortex_strength": [],
            "interpolated_total_vortex_strength_magnitude": [],
            "centroid": [],
            "n_injected": [],
            "n_candidates": [],
            "observed_time_step_size": [],
            "vlm_CL": [],
            "vlm_CD": [],
            "vlm_bound_vortex_strength_y": [],
            "vlm_wake_vortex_strength_y": [],
            "vlm_lesp_max": [],
            "vlm_n_particles": [],
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
            Logging.message("Stabilization: " + ", ".join(active))
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
            Logging.info(f"VLM solver coupled with {self.vlm_solver.lattice.num_panels} panels")
            self.vlm_solver.check_coupling_stability(
                self.time_step_size, getattr(self.setup, "freestream_velocity", None)
            )

    def _init_optional_solvers(self, final_setup) -> None:
        """Initialize optional sub-solvers (panel, VLM) with error handling."""
        self.panel_solver = getattr(final_setup, "panel_solver", None)
        if self.panel_solver is not None:
            try:
                body_stl = getattr(final_setup, "body_stl", None)
                lattice = getattr(self.panel_solver, "lattice", None)
                if body_stl and (lattice is None or lattice.num_panels == 0):
                    self.panel_solver.add_surface("body", body_stl)
                self.panel_solver.initialize(force=True)
                scope = getattr(self.panel_solver, "coupling_scope", "full")
                self._pressure_body_induced_fn = self.panel_solver.compute_induced_velocity
                if scope in ("full", "vpm_bc"):
                    self.set_body_induced_velocity(self.panel_solver.compute_induced_velocity)
                else:
                    self.set_body_induced_velocity(None)
                if scope != "full":
                    self.physics.body_velocity = None
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
                Logging.warning(f"Failed to initialize VLM solver: {e}")

    @staticmethod
    def _require_consistent_molecular_viscosity(viscous_cfg, vlm_setup) -> None:
        """The VPM owns molecular nu when a VLM is attached; values must agree."""
        scheme = getattr(viscous_cfg, "scheme", "NONE")
        if scheme == "NONE":
            vpm_nu = 0.0
        else:
            configured_nu = getattr(viscous_cfg, "kinematic_viscosity", None)
            if configured_nu is None:
                raise ValueError(
                    f"VPM viscous scheme {scheme!r} requires kinematic_viscosity "
                    "when a VLM setup is attached"
                )
            vpm_nu = float(configured_nu)
        vlm_nu = float(vlm_setup.kinematic_viscosity)
        if not np.isclose(vlm_nu, vpm_nu, rtol=0.0, atol=1e-15):
            raise ValueError(
                "Molecular kinematic_viscosity mismatch: the VPM viscous "
                f"scheme {scheme!r} uses {vpm_nu!r} m^2/s while the "
                f"attached VLM setup uses {vlm_nu!r} m^2/s. The VPM owns the "
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
        self.profiler.set_particle_count(self.particles.n_particles)
        self.profiler.report()

    def advance(self) -> None:
        """Advance the VPM solution by one time step.

        The step algorithm (velocity/gradient preparation, advection,
        stretching, coupled inviscid integration, viscous diffusion, operator
        splitting, and the in-step stabilization phases) is owned by the
        :class:`~source.solvers.VPM.core.evolution.EvolutionStepper`; this
        facade method delegates to it.
        """
        self.stepper.advance()

    def record_diagnostics(self, *, refresh_fields: bool = False) -> None:
        """Evaluate and log diagnostics for the current particle state.

        Set ``refresh_fields=True`` when velocity, gradients, or LES viscosity are
        stale for the current state.
        """
        if refresh_fields:
            self.stepper._update_velocity_and_gradients()
            self.stepper._update_LES_state()
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
        """Execute all configured field samplers (delegates to SamplerExecutor)."""
        SamplerExecutor.execute(self)

    def execute_final_samplers(self) -> None:
        """Execute the final-only samplers declared by the immutable setup."""
        SamplerExecutor.execute(self, self.setup.final_samplers)

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
    def particles_positions(self) -> np.ndarray:
        """Particle positions with shape ``(N, 3)`` [m]."""
        return self._get_particle_field("position")

    @property
    def particles_velocities(self) -> np.ndarray:
        """Particle velocities with shape ``(N, 3)`` [m/s]."""
        return self._get_particle_field("velocity")

    @property
    def particle_core_radius(self) -> np.ndarray:
        """Particle core radii with shape ``(N,)`` [m]."""
        return self._get_particle_field("core_radius")

    @property
    def particles_volumes(self) -> np.ndarray:
        """Particle volumes with shape ``(N,)`` [m³]."""
        return self._get_particle_field("volume")

    @property
    def particles_group_ids(self) -> np.ndarray:
        """Particle group identifiers with shape ``(N,)``."""
        return self._get_particle_field("group_id")

    @property
    def particles_zone_ids(self) -> np.ndarray:
        """Particle zone identifiers with shape ``(N,)``."""
        return self._get_particle_field("zone_id")

    @property
    def particles_viscosities(self) -> np.ndarray:
        """Particle molecular viscosities with shape ``(N,)`` [m²/s]."""
        return self._get_particle_field("kinematic_viscosity")

    @property
    def particles_viscosities_t(self) -> np.ndarray:
        """Particle turbulent viscosities with shape ``(N,)`` [m²/s]."""
        return self._get_particle_field("eddy_viscosity")

    @property
    def particles_viscosities_eff(self) -> np.ndarray:
        """Particle effective viscosities with shape ``(N,)`` [m²/s]."""
        return self._get_particle_field("effective_viscosity")

    @property
    def particles_velocity_gradients(self) -> np.ndarray:
        """Get particle velocity gradients array."""
        return self._get_particle_field("velocity_gradient")

    @property
    def particles_strain_rate_tensors(self) -> np.ndarray:
        """Get particle strain rate tensors array."""
        return self._get_particle_field("strain_rate")

    @property
    def freestream_velocity(self) -> np.ndarray:
        """Uniform background velocity [m/s]."""
        return self.particles.velocity_background_cpu()

    @property
    def particles_vorticities(self) -> np.ndarray:
        """Get particle vorticities array."""
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
        self._record_centroid_history()
        self._record_time_history()
        self._record_vlm_diagnostics()

    def _update_discretization_health(self) -> None:
        """Refresh particle-resolution and field-quality diagnostics."""
        if not getattr(self.setup, "export_discretization_health", True):
            return
        if self.particles.n_particles == 0:
            self._discretization_health = {}
            return
        self._discretization_health = discretization_health(
            self.particles_positions,
            self.particle_vortex_strength,
            self.particle_core_radius,
        )

    def _record_centroid_history(self) -> None:
        """Record the vortex-strength-magnitude-weighted particle centroid."""
        ParticleFieldEvaluation.record_centroid_history(
            self._diagnostics_history,
            self.particles_positions,
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
        lesp_max,
        n_p,
    ):
        """Delegate to VLMDiagnostics."""
        VLMDiagnostics.export_forces_csv(
            self.vlm_solver,
            forces,
            bound_vortex_strength,
            wake_vortex_strength,
            lesp_max,
            n_p,
            self.time,
            self.step,
            self.case_dir,
            getattr(self.setup, "sample_subdirectory", None),
        )

    @property
    def total_kinetic_energy(self) -> float:
        """Total kinetic energy per unit density."""
        return self._flow_integrals.get("kinetic_energy", 0.0)

    @property
    def total_helicity(self) -> float:
        """Total helicity."""
        return self._flow_integrals.get("helicity", 0.0)

    @property
    def total_enstrophy(self) -> float:
        """Total enstrophy."""
        return self._flow_integrals.get("enstrophy", 0.0)

    @property
    def vorticity_dissipation_rate(self) -> float:
        """Vorticity-based dissipation diagnostic."""
        return self._flow_integrals.get("vorticity_dissipation_rate", 0.0)

    @property
    def kinetic_energy_dissipation_rate(self) -> float:
        """Finite-difference kinetic-energy decay rate."""
        return self._flow_integrals.get("kinetic_energy_dissipation_rate", 0.0)

    @property
    def total_vortex_strength(self) -> np.ndarray:
        """Total particle vortex-strength vector [m³/s]."""
        return self._flow_integrals.get("vortex_strength", np.array([0.0, 0.0, 0.0]))

    @property
    def total_vortex_strength_magnitude(self) -> float:
        """Magnitude of the total particle vortex-strength vector [m³/s]."""
        return self._flow_integrals.get("vortex_strength_magnitude", 0.0)

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
    def centroids_of_vortex_strength(self) -> dict[int, np.ndarray]:
        """Vortex-strength-magnitude-weighted centroid for each particle group."""
        return self.field_diagnostics.compute_centroids_of_vortex_strength(self.particles)

    @property
    def centroid_of_vortex_strength(self) -> np.ndarray:
        """Global vortex-strength-magnitude-weighted particle centroid."""
        return self.field_diagnostics.compute_centroid_of_vortex_strength(self.particles)

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
    def compute_target_vorticities(self, grid_positions: np.ndarray) -> np.ndarray:
        """Evaluate vorticity at arbitrary target points.

        Args:
            grid_positions: Target coordinates with shape ``(N, 3)`` [m].

        Returns:
            Vorticity vectors with shape ``(N, 3)`` [1/s].
        """
        return self.physics.compute_target_vorticities(self.particles, grid_positions)

    def compute_target_velocities(
        self,
        grid_positions: np.ndarray,
        include_freestream: bool = True,
        zone_mask: np.ndarray | None = None,
        include_body: bool = True,
    ) -> np.ndarray:
        """Evaluate velocity at arbitrary target points.

        Args:
            grid_positions: Target coordinates with shape ``(N, 3)`` [m].
            include_freestream: Include the uniform background velocity.
            zone_mask: Optional mask selecting contributing particles.
            include_body: Include boundary-element body induction.

        Returns:
            Velocity vectors with shape ``(N, 3)`` [m/s].
        """
        velocities = self.physics.compute_target_velocities(
            self.particles,
            grid_positions,
            include_freestream=include_freestream,
            zone_mask=zone_mask,
        )
        return self._add_target_velocity_corrections(
            grid_positions, velocities, include_body=include_body
        )

    def _add_target_velocity_corrections(
        self,
        grid_positions: np.ndarray,
        particle_velocity: np.ndarray,
        *,
        include_body: bool,
    ) -> np.ndarray:
        """Add source-particle and body-potential terms to particle induction."""
        points = np.asarray(grid_positions, dtype=np.float64).reshape(-1, 3)
        velocities = np.asarray(particle_velocity, dtype=self.np_dtype).reshape(-1, 3)
        if len(velocities) != len(points):
            raise ValueError("target velocity and position counts must match")

        if self.n_sources > 0:
            n_targets = len(points)
            self.physics._resize_target_fields(n_targets)
            target_pos_ti = self.physics.target_positions
            target_vel_ti = self.physics.target_velocities

            # Fixed-shape buffers avoid persistent staging allocations.
            self.physics._upload_vector_array(points, target_pos_ti, n_targets)
            self.physics._upload_vector_array(velocities, target_vel_ti, n_targets)

            self.physics.kernels["compute_target_source_velocity_kernel"](
                target_pos_ti,
                self.source_positions,
                self.source_strengths,
                self.source_radii,
                target_vel_ti,
                n_targets,
                self.n_sources,
            )
            velocities = self.physics.extract_target_velocities(n_targets)

        body_fn = self._body_induced_fn
        if include_body and body_fn is not None:
            velocities = velocities + np.asarray(body_fn(points), dtype=velocities.dtype).reshape(
                velocities.shape
            )

        return velocities

    def _nonparticle_target_velocity(self, grid_positions: np.ndarray) -> np.ndarray:
        """Return only regularized-source and body-potential target velocity."""
        points = np.asarray(grid_positions, dtype=np.float64).reshape(-1, 3)
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

    def set_surface_sources(
        self, positions: np.ndarray, strengths: np.ndarray, radii: np.ndarray
    ) -> None:
        """Set regularized source particles used for body-blockage corrections."""
        self.n_sources = len(positions)
        if self.n_sources > MAX_SOURCES:
            Logging.warning(f"Clipping {self.n_sources} sources to {MAX_SOURCES}")
            self.n_sources = MAX_SOURCES

        n = self.n_sources
        # Taichi ``from_numpy`` requires the allocated shape.
        pos_buf = np.zeros((MAX_SOURCES, 3), dtype=self.np_dtype)
        str_buf = np.zeros(MAX_SOURCES, dtype=self.np_dtype)
        rad_buf = np.zeros(MAX_SOURCES, dtype=self.np_dtype)
        pos_buf[:n] = np.asarray(positions[:n], dtype=self.np_dtype)
        str_buf[:n] = np.asarray(strengths[:n], dtype=self.np_dtype)
        rad_buf[:n] = np.asarray(radii[:n], dtype=self.np_dtype)
        self.source_positions.from_numpy(pos_buf)
        self.source_strengths.from_numpy(str_buf)
        self.source_radii.from_numpy(rad_buf)

    def compute_target_velocity_gradients(self, grid_positions: np.ndarray) -> np.ndarray:
        """Evaluate ``∇u`` at arbitrary target points.

        Returns an ``(N, 9)`` array in row-major Jacobian order,
        ``J[i, j] = d(u_i)/d(x_j)``.  This kernel differentiates the vortex
        particle velocity; the uniform freestream has zero gradient and the
        optional regularized sources and body-potential callback are not
        differentiated here.
        """
        return self.physics.compute_target_velocity_gradients(self.particles, grid_positions)

    def compute_complete_target_velocity_gradients(
        self, grid_positions: np.ndarray, *, particle_spacing: float
    ) -> np.ndarray:
        """Evaluate the Jacobian of the body-complete velocity field.

        Vortex-particle induction is differentiated analytically. Regularized
        source and body-callback contributions are differentiated by centred
        differences with a step scaled by the coupling lattice spacing ``particle_spacing``.
        The result has shape ``(N, 3, 3)`` and convention
        ``J[i,j] = d(u_i)/d(x_j)``.
        """
        points = np.asarray(grid_positions, dtype=np.float64).reshape(-1, 3)
        # Use the same approximation as the target-velocity trace.  In a
        # treecode run, mixing a direct Jacobian with a treecode velocity is
        # both prohibitively expensive at coupling faces and inconsistent with
        # the boundary trace from which the normal velocity is taken.
        if self.physics.velocity_method == "TREECODE":
            gradient = np.asarray(
                self.physics.compute_target_velocity_gradients_hierarchical(
                    self.particles,
                    points,
                    theta=self.physics.velocity_theta,
                ),
                dtype=np.float64,
            ).reshape(-1, 3, 3)
        else:
            gradient = np.asarray(
                self.compute_target_velocity_gradients(points), dtype=np.float64
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

    def compute_complete_target_velocity_and_gradients(
        self, grid_positions: np.ndarray, *, particle_spacing: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate body-complete target velocity and Jacobian together.

        Treecode runs build and traverse the particle hierarchy once, then add
        the regularized-source and body-potential velocity and Jacobian terms.
        """
        points = np.asarray(grid_positions, dtype=np.float64).reshape(-1, 3)
        if self.physics.velocity_method == "TREECODE":
            velocity, gradient = self.physics.compute_target_velocity_and_gradients_hierarchical(
                self.particles,
                points,
                theta=self.physics.velocity_theta,
                include_freestream=True,
            )
        else:
            velocity = self.physics.compute_target_velocities(
                self.particles,
                points,
                include_freestream=True,
                zone_mask=None,
            )
            gradient = self.physics.compute_target_velocity_gradients(self.particles, points)
        complete_velocity = self._add_target_velocity_corrections(
            points, velocity, include_body=True
        )
        complete_gradient = self._add_nonparticle_target_gradient(
            points, gradient, particle_spacing=particle_spacing
        )
        return complete_velocity, complete_gradient

    def compute_complete_target_velocity_and_tangential_normal_gradient(
        self,
        grid_positions: np.ndarray,
        normals: np.ndarray,
        *,
        particle_spacing: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the body-complete velocity and tangential ``du/dn`` trace.

        The mixed FVM boundary condition does not consume the full nine-component
        Jacobian.  Particle induction is still evaluated by the configured fused
        target operation, while source/body terms use only the two centred samples
        along each face normal instead of three coordinate-direction pairs.
        """
        points = np.asarray(grid_positions, dtype=np.float64).reshape(-1, 3)
        face_normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
        if face_normals.shape != points.shape:
            raise ValueError("normal count does not match target positions")
        normal_magnitude = np.linalg.norm(face_normals, axis=1)
        if np.any(~np.isfinite(face_normals)) or np.any(normal_magnitude <= 0.0):
            raise ValueError("target normals must be finite and non-zero")
        unit_normals = face_normals / normal_magnitude[:, None]

        if self.physics.velocity_method == "TREECODE":
            velocity, gradient = self.physics.compute_target_velocity_and_gradients_hierarchical(
                self.particles,
                points,
                theta=self.physics.velocity_theta,
                include_freestream=True,
            )
        else:
            velocity = self.physics.compute_target_velocities(
                self.particles,
                points,
                include_freestream=True,
                zone_mask=None,
            )
            gradient = self.physics.compute_target_velocity_gradients(self.particles, points)

        complete_velocity = self._add_target_velocity_corrections(
            points, velocity, include_body=True
        )
        d_u_dn = np.einsum(
            "fij,fj->fi", np.asarray(gradient, dtype=np.float64).reshape(-1, 3, 3), unit_normals
        )
        if self._body_induced_fn is not None or self.n_sources > 0:
            step = max(1.0e-6, 1.0e-3 * float(particle_spacing))
            plus = self._nonparticle_target_velocity(points + step * unit_normals)
            minus = self._nonparticle_target_velocity(points - step * unit_normals)
            if plus.shape != points.shape or minus.shape != points.shape:
                raise RuntimeError("VPM body-velocity callback returned an invalid shape")
            d_u_dn += (plus - minus) / (2.0 * step)
        tangential = d_u_dn - np.einsum("fi,fi->f", d_u_dn, unit_normals)[:, None] * unit_normals
        if not np.all(np.isfinite(complete_velocity)) or not np.all(np.isfinite(tangential)):
            raise RuntimeError("Mixed VPM target evaluation returned non-finite data")
        return np.asarray(complete_velocity, dtype=np.float64), tangential

    def compute_target_pressure_gradients(
        self,
        grid_positions: np.ndarray,
        density: float = 1.0,
        nu: float | None = None,
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
        """Evaluate pressure-gradient terms at arbitrary target points.

        The result contains the total pressure gradient and its convective, viscous,
        and temporal contributions. ``temporal_method='eulerian'`` requires
        ``velocity_previous`` and ``time_step_size`` when the temporal term is enabled.
        ``include_body=False`` omits the optional boundary-element velocity from
        the hierarchical pressure evaluation while retaining particles and the
        configured freestream.
        """
        if nu is None:
            nu = (
                float(np.mean(self.particles_viscosities))
                if self.particles.n_particles > 0
                else 1e-5
            )
        if not hasattr(self, "_pressure_physics"):
            from source.solvers.VPM.physics.pressure import PressurePhysics

            self._pressure_physics = PressurePhysics(
                particle_kernel=self.particle_kernel,
                max_particles=int(self.setup.max_particles),
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
            return self._pressure_physics.compute_target_pressure_gradients_hierarchical(
                self.particles,
                grid_positions,
                density=density,
                nu=nu,
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

        if self.particles.n_particles > 0:
            self.physics.compute_velocity_gradients(self.particles)
        return self._pressure_physics.compute_target_pressure_gradient_components(
            self.particles,
            grid_positions,
            density=density,
            nu=nu,
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
        """Remove selected particles and track removed circulation for diagnostics."""
        if particle_indices is not None and len(particle_indices) > 0:
            # Reduce removed circulation on device.
            circ_removed, _ = self.particles.subset_moments(particle_indices)
            self._particles_removed_this_step = len(particle_indices)
            self._vortex_strength_removed_this_step = circ_removed

        elif remove_all:
            # Sum removed circulation on device.
            circ_removed = self.particles.total_vortex_strength()

            self._particles_removed_this_step = len(self.particles)
            self._vortex_strength_removed_this_step = circ_removed

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
        volume: np.ndarray,
        kinematic_viscosity: np.ndarray | None = None,
        eddy_viscosity: np.ndarray | None = None,
        group_id: np.ndarray | None = None,
        zone_id: np.ndarray | None = None,
        velocity_gradient: np.ndarray | None = None,
    ) -> None:
        """Append vortex particles to the active cloud.

        ``position``, ``velocity``, and ``vortex_strength`` have shape
        ``(N, 3)``; ``core_radius`` and ``volume`` have shape ``(N,)``.
        Molecular viscosity may be omitted when it is defined by the viscous
        configuration.
        """
        if kinematic_viscosity is None:
            nu = getattr(self._viscous_config, "kinematic_viscosity", None)
            if nu is not None and nu > 0:
                N = len(position)
                kinematic_viscosity = np.full(N, nu, dtype=self.np_dtype)
            else:
                raise ValueError(
                    "kinematic_viscosity is required. Either configure "
                    "ViscousConfig.kinematic_viscosity or pass an explicit array."
                )

        start = self.particles.n_particles
        self.particles.add_vortex_particles(
            position=position,
            velocity=velocity,
            vortex_strength=vortex_strength,
            core_radius=core_radius,
            volume=volume,
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
            volume,
            start,
            loading=getattr(self, "_loading_numerical_state", False),
        )

    def replace_vortex_particles(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        vortex_strength: np.ndarray,
        core_radius: np.ndarray,
        volume: np.ndarray,
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
            circ_removed = (
                self.particles.total_vortex_strength()
                if len(self.particles) > 0
                else np.zeros(3, dtype=self.np_dtype)
            )
            self._particles_removed_this_step = len(self.particles)
            self._vortex_strength_removed_this_step = circ_removed

        if kinematic_viscosity is None:
            nu = getattr(self._viscous_config, "kinematic_viscosity", None)
            if nu is not None and nu > 0:
                kinematic_viscosity = np.full(len(position), nu, dtype=self.np_dtype)
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
            volume=volume,
            kinematic_viscosity=kinematic_viscosity,
            eddy_viscosity=eddy_viscosity,
            group_id=group_id,
            zone_id=zone_id,
            velocity_gradient=velocity_gradient,
            strain_rate=strain_rate,
        )
        self._axisymmetric_orbits_validated = False
        magnitude = np.linalg.norm(np.asarray(vortex_strength, dtype=np.float64), axis=1)
        self.stabilization.on_replacement(magnitude, volume)

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
            self._particle_regeneration_pending = True

    def _print_timestep_validation_summary(self, results: dict) -> None:
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
        n_particles: int,
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
            expected_shape = (n_particles, 3)
        elif prop_name in {"velocity_gradient", "strain_rate"}:
            expected_shape = (n_particles, 3, 3)
        else:
            expected_shape = (n_particles,)

        if prop_value.shape != expected_shape:
            raise ValueError(
                f"Property '{prop_name}' has incorrect shape {prop_value.shape}. "
                f"Expected {expected_shape} for {n_particles} particles."
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
            "volume",
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

        n_particles = self.particles.n_particles
        if n_particles == 0:
            raise ValueError("Cannot set properties: particle system is empty")

        for field_name, value in properties.items():
            validated = self._validate_particle_property(
                field_name,
                value,
                n_particles,
            )
            self.particles.set_field(field_name, validated)

        self.particles._cache_step = -1

        property_names = list(properties)
        if len(property_names) == 1:
            Logging.info(f"Updated particle property: {property_names[0]}")
        else:
            Logging.info(
                f"Updated {len(property_names)} particle properties: " + ", ".join(property_names)
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

        Logging.info(f"Complete state saved to: {filename}")
        Logging.message(f"       - {filename}.h5 (numerical data)")
        Logging.message(f"       - {filename}.xdmf (ParaView visualization)")
        Logging.message(f"       - {config_file} (configuration)")

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
        N = self.particles.n_particles
        if N > 50_000:
            return
        if N > 0:
            self.physics.velocity_self(
                self.particles.position,
                self.particles.vortex_strength,
                self.particles.core_radius,
                self.particles.velocity,
                self.particles.velocity_background,
                N,
            )
        self.physics.compute_vorticities(self.particles)
        if self.flow_model != "POTENTIAL":
            self.stepper._update_velocity_gradients()

    @staticmethod
    def continue_from_checkpoint(checkpoint_name: str | None = None) -> "VPMSolver | None":
        """Restore a solver from an HDF5 checkpoint and its saved configuration."""
        if not CheckpointManager.validate_checkpoint(checkpoint_name):
            raise ValueError(f"Checkpoint validation failed for: {checkpoint_name}")

        Logging.message(f"\n{'-' * 60}")
        Logging.info("Resuming simulation from checkpoint:")
        Logging.message(f"       Base filename: {checkpoint_name}")
        Logging.message(f"{'-' * 60}\n")

        try:
            hdf5_file = f"{checkpoint_name}.h5"
            config_file = f"{checkpoint_name}.config.json"
            legacy_config_file = f"{checkpoint_name}_config.json"
            if not os.path.exists(config_file) and os.path.exists(legacy_config_file):
                config_file = legacy_config_file

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

        Logging.message("Simulation successfully restored!")
        Logging.message(f"Flow time: {restored_solver.time:.6f}")
        Logging.message(f"Time step: {restored_solver.step}")
        Logging.message(f"Particles: {restored_solver.particles.n_particles}")
        Logging.message(f"Backend: {restored_solver.setup.compute_device}")

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

        The callback receives particle positions and Biot–Savart velocity and returns
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

        n_particles = self.particles.n_particles
        if n_particles == 0:
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
            Logging.info(
                f"Removed {n_removed} particles {action} "
                f"box [{xmin:.2f}, {xmax:.2f}] × [{ymin:.2f}, {ymax:.2f}] × [{zmin:.2f}, {zmax:.2f}]"
            )

        return n_removed

    def remove_weak_particles(self, percent: float, per_group: bool = True) -> None:
        """Remove particles below the requested relative-strength threshold.

        When ``per_group=True``, the threshold is applied independently to each
        particle group.
        """
        if percent < 0 or percent > 100:
            raise ValueError("Percent must be between 0 and 100")

        if len(self.particles) == 0:
            return 0

        particles_before = len(self.particles)

        removed_indices = self.particles._remove_weak_particles(
            percent=percent,
            per_group=per_group,
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
