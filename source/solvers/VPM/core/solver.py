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
from ..config.backend import initialize_taichi_backend, reset_taichi_backend
from ..config.constants import MAX_PARTICLES, MAX_SOURCES
from ..config.types import SetFlowModel, StabilizationConfig, VPMSetup
from ..coupling import CouplingStepper
from ..diagnostics.resolution import discretization_health
from ..io.backup import BackupSystem
from ..io.logging import Logging, print_openonda_header
from ..io.runtime_profiler import RuntimeProfiler
from ..io.sampler import SamplerExecutor
from ..io.solver_io import SolverIO
from ..physics.engine import PhysicsEngine
from ..physics.evaluation import ParticleFieldEvaluation
from ..stabilization import StabilizationManager
from ..stabilization.context import StabilizationContext
from .evolution import EvolutionStepper


@ti.data_oriented
class Solver:
    """Vortex Particle Method solver.

    The solver owns the particle field, time integration, viscous and turbulence
    models, optional boundary-element coupling, diagnostics, sampling, and restart
    state. Configuration is supplied through :class:`VPMSetup`.
    """

    # Initialization

    def __init__(self, setup: VPMSetup | None = None) -> None:
        """Initialize the VPM solver. See VPMSetup for all parameters."""
        final_config = self._init_config(setup)
        self._init_io_and_backend(final_config, final_config.debug_mode)
        self._init_particles_and_physics(final_config)
        self._init_turbulence_and_adaptation(final_config)
        self._init_solvers(final_config)
        Logging.message(Logging.solver_info(self))

    @staticmethod
    def reset_gpu() -> None:
        """Reset the Taichi runtime and release device allocations.

        Call before constructing a new solver when several VPM cases are run
        sequentially in the same Python process.
        """
        reset_taichi_backend()

    def _init_config(self, setup: VPMSetup | None) -> VPMSetup:
        """Validate the setup and initialize scalar solver state."""
        final_config = setup if setup is not None else VPMSetup.dns_simulation()
        final_config._validate_config()
        self.config = final_config
        self.time_step_size = final_config.time_step_size
        self.flow_time = final_config.flow_time
        self.time_step = final_config.time_step
        self.time_integration = final_config.time_integration.upper()
        self.coupled_max_strain_increment = final_config.coupled_max_strain_increment
        self.coupled_max_advection_fraction = final_config.coupled_max_advection_fraction
        self.coupled_max_substeps = final_config.coupled_max_substeps
        axisymmetric_axis = final_config.axisymmetric_no_swirl_axis
        self.axisymmetric_axis = (
            -1 if axisymmetric_axis is None else {"x": 0, "y": 1, "z": 2}[axisymmetric_axis]
        )
        self._axisymmetric_orbits_validated = False

        # DVH uses a fixed heat-kernel increment Δt_d = β R_d² / (4ν).
        import math as _math

        self._dvh_dt_info: str | None = None
        self._gbd_dt_info: str | None = None
        self._rwm_dt_info: str | None = None
        vc = final_config.viscous

        # RWM accuracy criterion.
        if (
            vc.scheme == "RWM"
            and vc.characteristic_distance is not None
            and vc.characteristic_distance > 0
            and vc.viscosity is not None
            and vc.viscosity > 0
        ):
            dt_max_rwm = vc.rwm_accuracy_dt()
            if self.time_step_size > dt_max_rwm * (1.0 + 1e-6):
                Logging.message(
                    f"[RWM] WARNING: user dt = {self.time_step_size:.4e} s "
                    f"> accuracy limit h²/(4nu) = {dt_max_rwm:.4e} s — "
                    f"random displacement √(2nuΔt) exceeds inter-particle spacing h; "
                    f"vorticity gradients will be artificially smoothed."
                )
            self._rwm_dt_info = (
                f"RWM accuracy limit h²/(4nu) = {dt_max_rwm:.4e} s "
                f"(h = {vc.characteristic_distance:.3e} m, "
                f"nu = {vc.viscosity:.3e} m²/s)."
            )

        # GBD explicit-diffusion stability criterion.
        if vc.scheme == "GBD" and vc.viscosity is not None and vc.viscosity > 0:
            dt_max = vc.gbd_max_dt()
            if self.time_step_size > dt_max * (1.0 + 1e-6):
                Logging.message(
                    f"[GBD] WARNING: user dt = {self.time_step_size:.4e} s "
                    f"> CFL limit h²/(6nu) = {dt_max:.4e} s — "
                    f"explicit Laplacian may be UNSTABLE."
                )
            self._gbd_dt_info = (
                f"GBD fires every step "
                f"(Δt = {self.time_step_size:.4e} s, "
                f"CFL max = {dt_max:.4e} s)."
            )

        # Match the user step to an integer subdivision of the DVH increment.
        self._dvh_substeps: int = 1
        self._dvh_fire_counter: int = 0
        if vc.scheme == "DVH" and vc.viscosity is not None and vc.viscosity > 0:
            from ..physics.diffusion import _DVH_BETA

            dt_d_raw = vc.dvh_required_dt()
            # Avoid noisy floating-point time values.
            magnitude = _math.floor(_math.log10(abs(dt_d_raw)))
            dt_d = round(dt_d_raw, -magnitude + 2)
            user_dt = self.time_step_size
            n_sub = max(1, int(round(dt_d / user_dt))) if user_dt > 0 else 1
            dt_sub = dt_d / n_sub
            if abs(user_dt - dt_sub) > 1e-6 * max(user_dt, dt_sub):
                Logging.message(
                    f"[DVH] INFO: time step adjusted — "
                    f"user dt = {user_dt:.4e} s → dt = Δt_d/{n_sub} = {dt_sub:.4e} s "
                    f"(Δt_d = β·R_d²/(4nu) = {dt_d:.4e} s, β={_DVH_BETA}, "
                    f"R_d = {vc.dvh_rd_ratio}·h = {vc.dvh_rd_ratio * vc.dvh_grid_spacing:.4e} m; "
                    f"DVH fires every {n_sub} step(s))."
                )
                self.time_step_size = dt_sub
            self._dvh_substeps = n_sub
            self._dvh_dt_info = (
                f"DVH fires every {n_sub} step(s) (dt = Δt_d/{n_sub} = {dt_sub:.4e} s, "
                f"Δt_d = β·R_d²/(4nu) = {dt_d:.4e} s)."
            )

        self.advection_scheme = final_config.advection.scheme
        self.stretching_scheme = final_config.stretching.scheme
        self.stretching_use_treecode = getattr(final_config.stretching, "use_treecode", False)
        self.stretching_treecode_theta = getattr(final_config.stretching, "treecode_theta", 0.3)
        self.stretching_conserve_moments = getattr(
            final_config.stretching, "conserve_moments", False
        )
        self.stretching_conserve_energy = getattr(final_config.stretching, "conserve_energy", False)
        self.processing_unit = final_config.processing_unit.upper()
        self.flow_model = final_config.turbulence.flow_model.upper()
        self.viscous_scheme = final_config.viscous.scheme
        self._viscous_config = final_config.viscous
        self.stabilization_config: StabilizationConfig = final_config.stabilization
        self.particles_kernel = final_config.particles_kernel.upper()
        self.backup_frequency = final_config.backup_frequency
        self.logging_frequency = final_config.logging_frequency
        self.timing_frequency = getattr(final_config, "timing_frequency", 0)
        self.backup_file_name = final_config.backup_file_name
        self.backup_directory = getattr(
            final_config,
            "backup_directory",
            getattr(final_config, "solution_name", "solution"),
        )
        if getattr(final_config, "clean", False):
            import shutil as _shutil

            _backup_path = Path(self.backup_directory)
            if _backup_path.exists():
                _shutil.rmtree(_backup_path)
        Path(self.backup_directory).mkdir(parents=True, exist_ok=True)
        return final_config

    def _init_io_and_backend(self, final_config: VPMSetup, debug_mode: bool) -> None:
        """Set up output redirection, IO, precision, splitter/remesher, Taichi backend."""
        Logging.setup_output_redirection(self)
        self.io = SolverIO(self)
        self.precision = getattr(final_config, "precision", "f32")
        if self.precision not in ("f32", "f64"):
            raise ValueError(f"precision must be 'f32' or 'f64', got '{self.precision}'")
        self.processing_unit = initialize_taichi_backend(
            self.processing_unit,
            debug_mode,
            self.precision,
            device_memory_fraction=getattr(final_config, "device_memory_fraction", 0.5),
            random_seed=final_config.random_seed,
        )
        print_openonda_header(self.precision)
        SetFlowModel(self, flow_model=self.flow_model)
        self.compute_dtype = ti.f64 if self.precision == "f64" else ti.f32
        self.accumulator_dtype = self.compute_dtype
        self.np_dtype = np.float64 if self.precision == "f64" else np.float32

    def _init_particles_and_physics(self, final_config: VPMSetup) -> None:
        """Create particle container, physics engine, source fields, background velocity."""
        max_p = getattr(final_config, "max_particles", MAX_PARTICLES)
        self.particles = Particles(max_particles=max_p, float_dtype=self.precision)
        self.physics = PhysicsEngine(
            particles_kernel=self.particles_kernel,
            max_particles=max_p,
            accumulator_dtype=self.accumulator_dtype,
            max_targets=getattr(final_config, "max_targets", 200000),
        )

        _vel_cfg = getattr(final_config, "velocity", None)
        _vel_method = "TREECODE" if (_vel_cfg and _vel_cfg.method == "TREECODE") else "DIRECT"
        _vel_theta = _vel_cfg.theta if _vel_cfg else 0.5
        self.physics.configure_velocity(
            _vel_method,
            _vel_theta,
            multipole_order=getattr(_vel_cfg, "multipole_order", 1),
            sort_particle_targets=getattr(_vel_cfg, "sort_particle_targets", False),
            traversal_block_dim=getattr(_vel_cfg, "traversal_block_dim", 128),
        )

        _visc_cfg = getattr(final_config, "viscous", None)
        if _visc_cfg is not None and hasattr(self.physics, "regen_radius_ratio"):
            self.physics.regen_radius_ratio = float(getattr(_visc_cfg, "regen_radius_ratio", 2.5))
        if hasattr(self.physics, "configure_body_mask"):
            try:
                self.physics.configure_body_mask(getattr(final_config, "body_stl", None))
            except Exception as exc:
                Logging.warning(f"Failed to configure DVH body mask: {exc}")

        # Grid diffusion on GPU uses a fixed workspace to avoid repeated allocation.
        vpm_bounds = getattr(final_config, "vpm_domain_bounds", None)
        vc = getattr(final_config, "viscous", None)
        scheme = getattr(vc, "scheme", "").upper() if vc is not None else ""
        is_grid_diffusion = scheme in {"DVH", "GBD"}
        fixed_grid_required = (
            self.processing_unit in {"METAL", "VULKAN", "CUDA"} and is_grid_diffusion
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
                    "GPU DVH/GBD requires vpm_domain_bounds so the diffusion "
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
        self.num_sources = 0
        if hasattr(self.config, "background_velocity"):
            self.particles.set_background_velocity(np.array(self.config.background_velocity))

    def _init_turbulence_and_adaptation(self, final_config: VPMSetup) -> None:
        """Initialize LES turbulence, stretching settings, and diagnostics."""
        max_p = getattr(final_config, "max_particles", MAX_PARTICLES)
        self.LES = None
        if self.flow_model == "LES":
            self.LES = ParticlesLES(
                LES_filter_type=final_config.turbulence.model,
                max_particles=max_p,
                kernel_type=self.particles_kernel,
                cs=final_config.turbulence.cs,
                ce=final_config.turbulence.ce,
                accumulator_dtype=self.accumulator_dtype,
            )
        self.stretching_enabled = final_config.stretching.enabled
        self.stretching_mode = final_config.stretching.mode

        self.field_diagnostics = ParticleFieldEvaluation(
            particles_kernel=self.particles_kernel,
            max_particles=max_p,
            accumulator_dtype=self.accumulator_dtype,
        )
        self._flow_integrals: dict = {}
        self._discretization_health: dict = {}
        self._body_induced_fn = None
        self._stretch_dt_warned: bool = False
        self._particles_removed_this_step = 0
        self._circulation_removed_this_step = np.zeros(3, dtype=self.np_dtype)
        # Size of the last core-spreading moment projection, relative to |Gamma|.
        self.core_spreading_correction_relative = 0.0

    def _init_solvers(self, final_config: VPMSetup) -> None:
        """Initialize the stabilization master and the optional sub-solvers."""

        # Time histories consumed by export_diagnostics_csv and the VLM report.
        self._diagnostics_history: dict = {
            "time": [],
            "flow_time": [],
            "vpm_total_circ_vec": [],
            "vpm_total_circ_mag": [],
            "fvm_total_circ_vec": [],
            "fvm_total_circ_mag": [],
            "interp_total_circ_vec": [],
            "interp_total_circ_mag": [],
            "centroid": [],
            "n_injected": [],
            "n_candidates": [],
            "observed_dt": [],
            "vlm_CL": [],
            "vlm_CD": [],
            "vlm_gamma_bound_y": [],
            "vlm_gamma_wake_y": [],
            "vlm_lesp_max": [],
            "vlm_n_particles": [],
        }
        self.stabilization = StabilizationManager(
            StabilizationContext(
                particles=self.particles,
                physics=self.physics,
                field_diagnostics=self.field_diagnostics,
                config=self.config.stabilization,
                compute_dtype=self.compute_dtype,
                np_dtype=self.np_dtype,
                flow_model=self.flow_model,
                time_step=lambda: self.time_step,
                flow_time=lambda: self.flow_time,
                time_step_size=lambda: self.time_step_size,
                replace_vortex_particles=self.replace_vortex_particles,
                set_particles_properties=self.set_particles_properties,
                remove_particles_by_bounds=self.remove_particles_by_bounds,
                particles_removed=lambda: self._particles_removed_this_step,
                set_particles_removed=lambda value: setattr(
                    self, "_particles_removed_this_step", value
                ),
                circulation_removed=lambda: self._circulation_removed_this_step,
                set_circulation_removed=lambda value: setattr(
                    self, "_circulation_removed_this_step", value
                ),
            )
        )
        active = self.stabilization.active_mechanisms()
        if active:
            Logging.message("Stabilization: " + ", ".join(active))
        self._init_optional_solvers(final_config)
        # Synchronize asynchronous kernels at profiler boundaries.
        self.profiler = RuntimeProfiler(sync=ti.sync)
        self.simulation_time = 0.0
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
                self.time_step_size, getattr(self.config, "background_velocity", None)
            )

    def _init_optional_solvers(self, final_config) -> None:
        """Initialize optional sub-solvers (panel, VLM) with error handling."""
        self.panel_solver = getattr(final_config, "panel_solver", None)
        if self.panel_solver is not None:
            try:
                body_stl = getattr(final_config, "body_stl", None)
                lattice = getattr(self.panel_solver, "lattice", None)
                if body_stl and (lattice is None or lattice.num_panels == 0):
                    self.panel_solver.add_surface("body", body_stl)
                self.panel_solver.initialize(force=True)
                scope = getattr(self.panel_solver, "coupling_scope", "full")
                self._pressure_body_induced_fn = self.panel_solver.compute_induced_velocity
                if scope in ("full", "donor"):
                    self.set_body_induced_velocity(self.panel_solver.compute_induced_velocity)
                else:
                    self.set_body_induced_velocity(None)
                if scope != "full":
                    self.physics.body_velocity = None
            except Exception as e:
                raise RuntimeError(f"Failed to initialize panel solver: {e}") from e

        if final_config.vlm is None:
            self.vlm_solver = None
        else:
            from ..boundary_elements.vlm.solver.vlm_solver import VLMSolver

            self.vlm_solver = VLMSolver(final_config.vlm)
        if self.vlm_solver is not None:
            self._vpm_velocity_at_vlm = None
            self._vlm_velocity_at_vpm = None
            try:
                self._setup_vlm_solver()
            except Exception as e:
                Logging.warning(f"Failed to initialize VLM solver: {e}")

    def export_diagnostics_csv(self, filename: str) -> None:
        """Export diagnostics history to CSV for offline analysis."""
        self.io.export_diagnostics_csv(self._diagnostics_history, filename)

    @classmethod
    def from_config_file(cls, filename: str) -> "Solver":
        """Create a solver from a JSON configuration file."""
        config = VPMSetup.load_from_file(filename)
        return cls(setup=config)

    def save_config(self, filename: str) -> None:
        """Save the current solver configuration to a JSON file."""
        self.io.save_config(filename)

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
        self.profiler.set_particle_count(self.particles.number_of_particles)
        self.profiler.report()

    def update_state(self) -> None:
        """Advance the VPM solution by one time step.

        The step algorithm (velocity/gradient preparation, advection,
        stretching, coupled inviscid integration, viscous diffusion, operator
        splitting, and the in-step stabilization phases) is owned by the
        :class:`~source.solvers.VPM.core.evolution.EvolutionStepper`; this
        facade method delegates to it.
        """
        self.stepper.update_state()

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

        if getattr(self.config, "export_flow_integrals", True):
            self._export_flow_integrals_csv()

        if self.LES is not None:
            Logging.les_diagnostics(self)

        self._execute_samplers()

    def _export_flow_integrals_csv(self) -> None:
        """Append one row of flow integrals to ``<backup_directory>/samples/flow_integrals.csv``.

        Thin wrapper that delegates the CSV export to the ``SolverIO`` manager
        (which owns all exports).
        """
        self.io.export_flow_integrals_csv(self)

    def _execute_samplers(self) -> None:
        """Execute all configured field samplers (delegates to SamplerExecutor)."""
        SamplerExecutor.execute(self)

    def execute_final_samplers(self) -> None:
        """Execute the final-only samplers declared by the immutable setup."""
        SamplerExecutor.execute(self, self.config.final_samplers)

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
            self.flow_time,
            self.time_step,
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
    def particles_strengths(self) -> np.ndarray:
        """Particle circulation vectors with shape ``(N, 3)`` [m²/s]."""
        return self._get_particle_field("circulation")

    @property
    def particles_radii(self) -> np.ndarray:
        """Particle core radii with shape ``(N,)`` [m]."""
        return self._get_particle_field("radius")

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
        return self._get_particle_field("viscosity")

    @property
    def particles_viscosities_t(self) -> np.ndarray:
        """Particle turbulent viscosities with shape ``(N,)`` [m²/s]."""
        return self._get_particle_field("viscosity_turbulent")

    @property
    def particles_viscosities_eff(self) -> np.ndarray:
        """Particle effective viscosities with shape ``(N,)`` [m²/s]."""
        return self._get_particle_field("viscosity_effective")

    @property
    def particles_velocity_gradients(self) -> np.ndarray:
        """Get particle velocity gradients array."""
        return self._get_particle_field("velocity_gradient")

    @property
    def particles_strain_rate_tensors(self) -> np.ndarray:
        """Get particle strain rate tensors array."""
        return self._get_particle_field("strain_rate")

    @property
    def background_velocity(self) -> np.ndarray:
        """Uniform background velocity [m/s]."""
        return self.particles.velocity_background_cpu()

    @property
    def particles_vorticities(self) -> np.ndarray:
        """Get particle vorticities array."""
        return self._get_particle_field("vorticity")

    @property
    def particles_circulation(self) -> np.ndarray:
        """Alias for :attr:`particles_strengths`."""
        return self._get_particle_field("circulation")

    # Flow diagnostics
    def _update_all_flow_integrals(self) -> None:
        """Recompute flow integrals and associated diagnostic histories."""
        self._flow_integrals = self.field_diagnostics.compute_flow_integrals(
            self.particles, self.flow_time
        )
        self._update_discretization_health()
        self._record_centroid_history()
        self._record_flow_time_history()
        self._record_vlm_diagnostics()

    def _update_discretization_health(self) -> None:
        """Refresh particle-resolution and field-quality diagnostics."""
        if not getattr(self.config, "export_discretization_health", True):
            return
        if self.particles.number_of_particles == 0:
            self._discretization_health = {}
            return
        self._discretization_health = discretization_health(
            self.particles_positions,
            self.particles_circulation,
            self.particles_radii,
        )

    def _record_centroid_history(self) -> None:
        """Compute and record the circulation-weighted centroid to diagnostics history."""
        ParticleFieldEvaluation.record_centroid_history(
            self._diagnostics_history,
            self.particles_positions,
            self.particles_circulation,
        )

    def _record_flow_time_history(self) -> None:
        """Delegate to VLMDiagnostics."""
        ft_hist = self._diagnostics_history.get("flow_time", [])
        observed_dt = self.flow_time - ft_hist[-1] if len(ft_hist) >= 1 else 0.0
        VLMDiagnostics.record_flow_time(self._diagnostics_history, self.flow_time, observed_dt)

    def _record_vlm_diagnostics(self) -> None:
        """Delegate to VLMDiagnostics."""
        sample_subdirectory = getattr(self.config, "sample_subdirectory", None)
        VLMDiagnostics.record_vlm_diagnostics(
            self.vlm_solver,
            self.particles,
            self.particles_strengths,
            self._diagnostics_history,
            self.time_step,
            self.flow_time,
            self.backup_directory,
            sample_subdirectory,
        )
        VLMLoadingDistribution.record_loading_distributions(
            self.vlm_solver,
            self._diagnostics_history,
            self.time_step,
            self.flow_time,
            self.backup_directory,
            sample_subdirectory,
        )

    def _export_vlm_forces_to_csv(self, forces, gamma_bound, gamma_wake, lesp_max, n_p):
        """Delegate to VLMDiagnostics."""
        VLMDiagnostics.export_forces_csv(
            self.vlm_solver,
            forces,
            gamma_bound,
            gamma_wake,
            lesp_max,
            n_p,
            self.flow_time,
            self.time_step,
            self.backup_directory,
            getattr(self.config, "sample_subdirectory", None),
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
    def total_strength(self) -> np.ndarray:
        """Total particle circulation vector."""
        return self._flow_integrals.get("strength", np.array([0.0, 0.0, 0.0]))

    @property
    def total_strength_magnitude(self) -> float:
        """Magnitude of the total particle circulation."""
        return self._flow_integrals.get("strength_magnitude", 0.0)

    @property
    def total_linear_impulse(self) -> np.ndarray:
        """Return the current linear impulse, recomputed from the active particle field."""
        integrals = self.field_diagnostics.compute_flow_integrals(
            self.particles, self.flow_time, record_history=False
        )
        return integrals.get("linear_impulse", np.array([0.0, 0.0, 0.0]))

    @property
    def total_angular_impulse(self) -> np.ndarray:
        """Total angular impulse."""
        return self._flow_integrals.get("angular_impulse", np.array([0.0, 0.0, 0.0]))

    @property
    def centroids_of_circulation(self) -> dict[int, np.ndarray]:
        """Circulation-weighted centroid for each particle group."""
        return self.field_diagnostics.compute_centroids_of_circulation(self.particles)

    @property
    def centroid_of_circulation(self) -> np.ndarray:
        """Global circulation-weighted centroid."""
        return self.field_diagnostics.compute_centroid_of_circulation(self.particles)

    def compute_forces(
        self, density: float = 1.225, V_ref_mag: float | None = None
    ) -> dict[str, np.ndarray | float]:
        """Compute aerodynamic force from the configured VLM model.

        Args:
            density: Fluid density [kg/m³].
            V_ref_mag: Reference speed [m/s]. Uses the background speed when omitted.

        Returns:
            Force components and the force-evaluation method.
        """
        if self.vlm_solver is None:
            raise RuntimeError("Force evaluation requires a VLM setup")
        method = self.vlm_solver.force.method

        if method == "KUTTA_JOUKOWSKI":
            return self._compute_forces_kutta_joukowski(density, V_ref_mag)
        else:
            raise ValueError(f"Unknown force method: {method}")

    def _compute_forces_kutta_joukowski(
        self, density: float, V_ref_mag: float | None
    ) -> dict[str, np.ndarray | float]:
        """Compute forces via the Kutta-Joukowski theorem. Delegates to VLMForceEvaluator."""
        return VLMForceEvaluator.compute_kutta_joukowski(
            self.vlm_solver, self.background_velocity, density, V_ref_mag
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

        if self.num_sources > 0:
            n_targets = len(grid_positions)
            self.physics._resize_target_fields(n_targets)
            target_pos_ti = self.physics.target_positions
            target_vel_ti = self.physics.target_velocities

            # Fixed-shape buffers avoid persistent staging allocations.
            self.physics._upload_vector_array(grid_positions, target_pos_ti, n_targets)
            self.physics._upload_vector_array(velocities, target_vel_ti, n_targets)

            self.physics.kernels["compute_target_source_velocity_kernel"](
                target_pos_ti,
                self.source_positions,
                self.source_strengths,
                self.source_radii,
                target_vel_ti,
                n_targets,
                self.num_sources,
            )
            velocities = self.physics.extract_target_velocities(n_targets)

        body_fn = self._body_induced_fn
        if include_body and body_fn is not None:
            velocities = velocities + np.asarray(
                body_fn(grid_positions), dtype=velocities.dtype
            ).reshape(velocities.shape)

        return velocities

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
        self.num_sources = len(positions)
        if self.num_sources > MAX_SOURCES:
            Logging.warning(f"Clipping {self.num_sources} sources to {MAX_SOURCES}")
            self.num_sources = MAX_SOURCES

        n = self.num_sources
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

        Returns an ``(N, 9)`` array in row-major tensor order.
        """
        return self.physics.compute_target_velocity_gradients(self.particles, grid_positions)

    def compute_target_pressure_gradients(
        self,
        grid_positions: np.ndarray,
        density: float = 1.0,
        nu: float | None = None,
        include_viscous: bool = True,
        include_temporal: bool = True,
        include_freestream: bool = True,
        h: float | None = None,
        temporal_method: str = "lagrangian",
        velocity_previous: np.ndarray | None = None,
        dt: float | None = None,
        return_velocity: bool = False,
        treecode_theta: float | None = None,
        include_body: bool = True,
    ) -> dict | tuple[dict, np.ndarray]:
        """Evaluate pressure-gradient terms at arbitrary target points.

        The result contains the total pressure gradient and its convective, viscous,
        and temporal contributions. ``temporal_method='eulerian'`` requires
        ``velocity_previous`` and ``dt`` when the temporal term is enabled.
        ``include_body=False`` omits the optional boundary-element velocity from
        the hierarchical pressure evaluation while retaining particles and the
        configured freestream.
        """
        if nu is None:
            nu = (
                float(np.mean(self.particles_viscosities))
                if self.particles.number_of_particles > 0
                else 1e-5
            )
        if not hasattr(self, "_pressure_physics"):
            from source.solvers.VPM.physics.pressure import PressurePhysics

            self._pressure_physics = PressurePhysics(
                particles_kernel=self.particles_kernel,
                max_particles=int(self.config.max_particles),
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
                dt=dt,
                h=h,
                return_velocity=return_velocity,
                theta=treecode_theta,
                background_velocity=self.background_velocity,
                body_fn=body_fn,
            )

        if self.particles.number_of_particles > 0:
            self.physics.compute_velocity_gradients(self.particles)
        return self._pressure_physics.compute_target_pressure_gradient_components(
            self.particles,
            grid_positions,
            density=density,
            nu=nu,
            include_viscous=include_viscous,
            include_temporal=include_temporal,
            h_laplacian=h,
            include_freestream=include_freestream,
            temporal_method=temporal_method,
            velocity_previous=velocity_previous,
            dt=dt,
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
            self._circulation_removed_this_step = circ_removed

        elif remove_all:
            # Sum removed circulation on device.
            circ_removed = self.particles.total_circulation()

            self._particles_removed_this_step = len(self.particles)
            self._circulation_removed_this_step = circ_removed

        else:
            self._particles_removed_this_step = 0
            self._circulation_removed_this_step = np.zeros(3)

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
        circulation: np.ndarray,
        radius: np.ndarray,
        volume: np.ndarray,
        viscosity: np.ndarray | None = None,
        viscosity_turbulent: np.ndarray | None = None,
        group_id: np.ndarray | None = None,
        zone_id: np.ndarray | None = None,
        velocity_gradient: np.ndarray | None = None,
    ) -> None:
        """Append vortex particles to the active cloud.

        ``position``, ``velocity``, and ``circulation`` have shape ``(N, 3)``;
        ``radius`` and ``volume`` have shape ``(N,)``. Molecular viscosity may be
        omitted when it is defined by the viscous configuration.
        """
        if viscosity is None:
            nu = getattr(self._viscous_config, "viscosity", None)
            if nu is not None and nu > 0:
                N = len(position)
                viscosity = np.full(N, nu, dtype=self.np_dtype)
            else:
                raise ValueError(
                    "viscosity parameter is required.  Either set "
                    "ViscousConfig.viscosity or pass an explicit array."
                )

        start = self.particles.number_of_particles
        self.particles.add_vortex_particles(
            position=position,
            velocity=velocity,
            circulation=circulation,
            radius=radius,
            volume=volume,
            viscosity=viscosity,
            viscosity_turbulent=viscosity_turbulent,
            group_id=group_id,
            zone_id=zone_id,
            velocity_gradient=velocity_gradient,
        )
        self._axisymmetric_orbits_validated = False
        magnitude = np.linalg.norm(np.asarray(circulation, dtype=np.float64), axis=1)
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
        circulation: np.ndarray,
        radius: np.ndarray,
        volume: np.ndarray,
        viscosity: np.ndarray | None = None,
        viscosity_turbulent: np.ndarray | None = None,
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
                self.particles.total_circulation()
                if len(self.particles) > 0
                else np.zeros(3, dtype=self.np_dtype)
            )
            self._particles_removed_this_step = len(self.particles)
            self._circulation_removed_this_step = circ_removed

        if viscosity is None:
            nu = getattr(self._viscous_config, "viscosity", None)
            if nu is not None and nu > 0:
                viscosity = np.full(len(position), nu, dtype=self.np_dtype)
            else:
                raise ValueError(
                    "viscosity parameter is required.  Either set "
                    "ViscousConfig.viscosity or pass an explicit array."
                )

        self.particles.replace_from_numpy(
            position=position,
            velocity=velocity,
            circulation=circulation,
            radius=radius,
            volume=volume,
            viscosity=viscosity,
            viscosity_turbulent=viscosity_turbulent,
            group_id=group_id,
            zone_id=zone_id,
            velocity_gradient=velocity_gradient,
            strain_rate=strain_rate,
        )
        self._axisymmetric_orbits_validated = False
        magnitude = np.linalg.norm(np.asarray(circulation, dtype=np.float64), axis=1)
        self.stabilization.on_replacement(magnitude, volume)

    def update_particle_circulations(
        self,
        mask: np.ndarray,
        delta_circ: np.ndarray,
    ) -> None:
        """Apply an in-place circulation delta to a masked subset of particles."""
        self.particles.update_circulations_masked(mask, delta_circ)

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
    def _validate_particle_property(prop_name: str, prop_value, N: int) -> np.ndarray:
        """Validate shape and finiteness of a single particle property array."""
        if not isinstance(prop_value, np.ndarray):
            prop_value = np.array(prop_value)
        expected_shape: tuple
        if prop_name in ("positions", "velocities", "strengths", "vorticities"):
            expected_shape = (N, 3)
        elif prop_name in ("grad_u", "Sij"):
            expected_shape = (N, 3, 3)
        else:
            expected_shape = (N,)
        if prop_value.shape != expected_shape:
            raise ValueError(
                f"Property '{prop_name}' has incorrect shape {prop_value.shape}. "
                f"Expected {expected_shape} for {N} particles."
            )
        if not np.all(np.isfinite(prop_value)):
            nan_count = int(np.sum(np.isnan(prop_value)))
            inf_count = int(np.sum(np.isinf(prop_value)))
            raise ValueError(
                f"Property '{prop_name}' contains invalid values: "
                f"{nan_count} NaN, {inf_count} Inf. "
                f"Cannot set particle properties with non-finite values."
            )
        return prop_value

    def set_particles_properties(self, **properties) -> None:
        """Update one or more particle fields after validating names, shapes, and finiteness.

        Supported keys are ``positions``, ``velocities``, ``strengths``,
        ``vorticities``, ``radii``, ``volumes``, ``viscosities``, ``viscosities_t``,
        ``viscosities_eff``, ``group_ids``, ``grad_u``, and ``Sij``.
        """
        if not properties:
            return

        valid_properties = {
            "positions": "position",
            "velocities": "velocity",
            "strengths": "circulation",
            "vorticities": "vorticity",
            "radii": "radius",
            "volumes": "volume",
            "viscosities": "viscosity",
            "viscosities_t": "viscosity_turbulent",
            "viscosities_eff": "viscosity_effective",
            "group_ids": "group_id",
            "grad_u": "velocity_gradient",
            "Sij": "strain_rate",
        }

        for prop_name in properties:
            if prop_name not in valid_properties:
                raise ValueError(
                    f"Invalid property name '{prop_name}'. "
                    f"Valid properties: {list(valid_properties.keys())}"
                )

        N = self.particles.number_of_particles
        if N == 0:
            raise ValueError("Cannot set properties: particle system is empty")

        validated_properties = {}
        for prop_name, prop_value in properties.items():
            field_name = valid_properties[prop_name]
            validated_properties[field_name] = self._validate_particle_property(
                prop_name, prop_value, N
            )

        for field_name, prop_value in validated_properties.items():
            self.particles.set_field(field_name, prop_value)

        self.particles._cached_step = -1

        property_names = list(properties.keys())
        if len(property_names) == 1:
            Logging.info(f"Updated particle property: {property_names[0]}")
        else:
            Logging.info(
                f"Updated {len(property_names)} particle properties: {', '.join(property_names)}"
            )

    # State and restart

    def save_state(self, filename: str = "solution/solver_state") -> None:
        """Save a restartable numerical state and its configuration."""

        if backup_dir := os.path.dirname(filename):
            os.makedirs(backup_dir, exist_ok=True)

        self._refresh_backup_particle_fields()
        BackupSystem.backup_solver(self, filename, append_step=False, verbose=False)

        config_file = f"{filename}.config.json"
        BackupSystem._save_configuration(self, config_file)

        Logging.info(f"Complete state saved to: {filename}")
        Logging.message(f"       - {filename}.h5 (numerical data)")
        Logging.message(f"       - {filename}.xdmf (ParaView visualization)")
        Logging.message(f"       - {config_file} (configuration)")

    def backup_solution(self, backup_file_name: str = "backup") -> None:
        """Back up the solver state to a specified file."""
        self._refresh_backup_particle_fields()
        BackupSystem.backup_solver(self, backup_file_name, verbose=True)

    def _backup_solution(self) -> None:
        """Write a scheduled solver backup when one is due."""
        if not self.io.should_backup():
            return

        self._refresh_backup_particle_fields()

        self.io.backup()

    def _refresh_backup_particle_fields(self) -> None:
        """Refresh particle fields that are expected to be available in backups."""
        N = self.particles.number_of_particles
        if N > 50_000:
            return
        if N > 0:
            self.physics.velocity_self(
                self.particles.position,
                self.particles.circulation,
                self.particles.radius,
                self.particles.velocity,
                self.particles.velocity_background,
                N,
            )
        self.physics.compute_vorticities(self.particles)
        if self.flow_model != "POTENTIAL":
            self.stepper._update_velocity_gradients()

    @staticmethod
    def continue_from_backup(backup_file_name: str | None = None) -> "Solver | None":
        """Restore a solver from an HDF5 backup and its saved configuration."""
        if not BackupSystem.validate_backup(backup_file_name):
            raise ValueError(f"Backup validation failed for: {backup_file_name}")

        Logging.message(f"\n{'-' * 60}")
        Logging.info("Resuming simulation from backup:")
        Logging.message(f"       Base filename: {backup_file_name}")
        Logging.message(f"{'-' * 60}\n")

        try:
            hdf5_file = f"{backup_file_name}.h5"
            config_file = f"{backup_file_name}.config.json"
            legacy_config_file = f"{backup_file_name}_config.json"
            if not os.path.exists(config_file) and os.path.exists(legacy_config_file):
                config_file = legacy_config_file

            if not os.path.exists(hdf5_file):
                raise FileNotFoundError(f"Numerical data file not found: {hdf5_file}")
            if not os.path.exists(config_file):
                raise FileNotFoundError(f"Configuration file not found: {config_file}")

            config = BackupSystem._load_configuration(config_file)
            restored_solver = Solver(setup=config)
            BackupSystem._load_numerical_data(restored_solver, hdf5_file)
        except Exception as e:
            raise RuntimeError(f"Restore failed: {e}") from e

        restored_solver.field_diagnostics.reset_energy_history()

        restored_solver._update_all_flow_integrals()

        Logging.message("Simulation successfully restored!")
        Logging.message(f"Flow time: {restored_solver.flow_time:.6f}")
        Logging.message(f"Time step: {restored_solver.time_step}")
        Logging.message(f"Particles: {restored_solver.particles.number_of_particles}")
        Logging.message(f"Backend: {restored_solver.config.processing_unit}")

        return restored_solver

    def export_state(self, filename: str, **kwargs):
        """Export solver state for visualization and post-processing."""
        self.io.export_state(filename, **kwargs)

    # Particle updates

    def set_background_velocity(self, velocity: list[float] | np.ndarray) -> None:
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

        self.particles.set_background_velocity(velocity_arr)

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

        n_particles = self.particles.number_of_particles
        if n_particles == 0:
            return 0

        xmin, xmax, ymin, ymax, zmin, zmax = bounds

        keep_mask = None
        if self.stabilization.reference_strengths is not None:
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
