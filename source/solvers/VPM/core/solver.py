"""
VPM Solver — Vortex Particle Method implementation.

Provides DNS, LES, and inviscid flow models with GPU acceleration via Taichi,
backup/restore, turbulence modelling, and comprehensive diagnostics.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
License: GPL-3.0-or-later
"""

# =========================================================
# IMPORTS AND DEPENDENCIES
# =========================================================

# Standard library imports
from dataclasses import replace
import os
from pathlib import Path

# Third-party imports
import numpy as np
import taichi as ti

from source.solvers.VPM.particles import create_physics
from source.solvers.VPM.particles.container import Particles
from source.solvers.VPM.turbulence.turbulence import ParticlesLES

# Internal OpenONDA modules
from ..boundary_elements.vlm.solver.diagnostics import VLMDiagnostics
from ..boundary_elements.vlm.solver.forces import VLMForceEvaluator
from ..boundary_elements.vlm.solver.loading_distribution import VLMLoadingDistribution
from ..config.backend import initialize_taichi_backend, reset_taichi_backend
from ..config.constants import MAX_PARTICLES, MAX_SOURCES
from ..config.types import SetFlowModel, SolverConfig, StabilizationConfig
from ..io.backup import BackupSystem
from ..io.logging import Logging, print_openonda_header
from ..io.runtime_profiler import RuntimeProfiler
from ..io.sampler import SamplerExecutor
from ..io.solver_io import SolverIO
from ..physics.evaluation import ParticleFieldEvaluation

class FilteredParticles:
    """Helper class to pass filtered particles to physics kernels."""

    def __init__(self, positions, strengths, radii, count):
        self.positions = positions
        self.strengths = strengths
        self.radii = radii
        self.count = count

    def __len__(self):
        return self.count

# =========================================================
# MAIN VPM SOLVER CLASS
# =========================================================

@ti.data_oriented
class Solver:
    """
    High-performance Vortex Particle Method (VPM) simulator for computational fluid dynamics.

    This class provides a complete VPM implementation supporting:
    - Multiple flow models: DNS, LES (Smagorinsky, Dynamic Smagorinsky), and inviscid
    - GPU acceleration via Taichi for maximum performance
    - Robust time integration methods (Euler, RK2, RK3)
    - Advanced turbulence modeling and viscous diffusion schemes
    - Comprehensive diagnostics and monitoring capabilities
    - Reliable backup/restore functionality
    - Cached Taichi-based total quantities for optimal performance

    Attributes:
          particles (Particles): The particle system containing all particle data
          time_step_size (float): Time increment per simulation step [s]
          flow_time (float): Current physical simulation time [s]
          time_step (int): Current time step index
          flow_model (str): Flow physics model {'DNS', 'LES', 'POTENTIAL'}
          viscous_scheme (str): Viscous modeling scheme {'CS', 'RWM', 'NONE'}
          particles_kernel (str): Particle interaction kernel {'GAUSSIAN', 'WINCKELMANS'}
          time_integration_scheme (str): Time integration method {'EULER', 'RK2', 'RK3'}
          processing_unit (str): Computation backend {'CPU', 'GPU'}
          backup_frequency (int): Save simulation state every N time steps
          simulation_time (float): Total wall-clock time elapsed [s]

    Cached Total Quantities Properties (auto-updated, GPU-optimized):
          total_kinetic_energy (float): Total kinetic energy [J or m²/s²]
          total_helicity (float): Total helicity [m³/s²]
          total_enstrophy (float): Total enstrophy [1/s²]
          vorticity_dissipation_rate (float): Viscous dissipation rate [J/s]
          kinetic_energy_dissipation_rate (float): Energy decay rate [J/s]
          total_strength (np.ndarray): Total vortex strength vector [1/s]
          total_linear_impulse (np.ndarray): Linear impulse vector [m³/s]
          total_angular_impulse (np.ndarray): Angular impulse vector [m⁴/s]
          centroids_of_circulation (Dict): Circulation centroids by group

    Methods are organized into functional groups:
          - Core Simulation: update_state(), time stepping, physics updates
          - Particle Management: add/remove particles, field loading
          - Diagnostics: compute flow properties, monitoring, logging
          - State Management: backup/restore, serialization
          - Field Computation: velocity/vorticity at arbitrary points
          - Properties: convenient access to particle data

    Example:
          >>> # Create solver with cached properties
          >>> solver = Solver(config)
          >>> solver.update_state()  # Updates all cached quantities
          >>>
          >>> # Access properties (computed in Taichi, converted to numpy on demand)
          >>> energy = solver.total_kinetic_energy      # Fast access via cached property
          >>> helicity = solver.total_helicity         # No CPU transfer until accessed
          >>> impulse = solver.total_linear_impulse    # Cached result
          >>>
          >>> energy_alt = solver.compute_total_kinetic_energy()
          >>>
          >>> # Per-particle analysis (transfers to CPU)
          >>> particle_energies = solver.compute_kinetic_energies()  # For detailed analysis
    """

    # INITIALIZATION AND CONFIGURATION

    # NOTE: Taichi initialization is handled by initialize_taichi_backend() which
    # safely handles re-initialization attempts (catches exceptions if already initialized).
    def __init__(self, config: SolverConfig | None = None, **kwargs) -> None:
        """Initialize the VPM solver. See SolverConfig for all parameters."""
        debug_mode: bool = bool(kwargs.pop("debug_mode", False))
        final_config = self._init_config(config, kwargs)
        self._init_io_and_backend(final_config, debug_mode)
        self._init_particles_and_physics(final_config)
        self._init_turbulence_and_adaptation(final_config)
        self._init_diagnostics_and_solvers(final_config)
        Logging.message(Logging.solver_info(self))

    @staticmethod
    def reset_gpu() -> None:
        """Fully reset the Taichi runtime, releasing **all** GPU memory.

        Call this **before** creating a new :class:`Solver` when running
        multiple VPM simulations sequentially in the same Python process.
        After this call every Taichi field, kernel, and ndarray from the
        previous run is invalidated; the next :class:`Solver` constructor
        will re-initialise Taichi from scratch.

        Example::

            from source.solvers.VPM import Solver, SolverConfig

            for case in cases:
                Solver.reset_gpu()           # free all GPU memory
                solver = Solver(config=case)
                for _ in range(num_steps):
                    solver.update_state()

        This prevents the ``Failed to allocate ext arr buffer`` Taichi error
        that occurs when accumulated GPU allocations from a prior run leave
        no room for external-array staging buffers.
        """
        reset_taichi_backend()

    def _init_config(self, config: SolverConfig | None, kwargs: dict) -> SolverConfig:
        """Merge config + kwargs, validate, set scalar attributes, prepare backup dir."""
        if config is None:
            config = SolverConfig.dns_simulation()
        final_config = replace(config, **kwargs) if kwargs else config
        final_config._validate_config()
        self.config = final_config
        self.time_step_size = final_config.time_step_size
        self.flow_time = final_config.flow_time
        self.time_step = final_config.time_step

        # The DVH heat-kernel width is fixed at β·R_d², so each firing advances
        # viscous time by EXACTLY Δt_d = β·R_d²/(4nu), independent of the dt argument.
        import math as _math

        self._dvh_dt_info: str | None = None
        self._gbd_dt_info: str | None = None
        self._cs_dt_info: str | None = None
        self._rwm_dt_info: str | None = None
        vc = final_config.viscous

        # CS: warn if dt exceeds the parabolic CFL bound h²/(4nu).
        # Requires characteristic_distance and viscosity to be set.
        if (
            vc.scheme == "CS"
            and vc.characteristic_distance is not None
            and vc.characteristic_distance > 0
            and vc.viscosity is not None
            and vc.viscosity > 0
        ):
            dt_max_cs = vc.cs_max_dt()
            if self.time_step_size > dt_max_cs * (1.0 + 1e-6):
                Logging.message(
                    f"[CS] WARNING: user dt = {self.time_step_size:.4e} s "
                    f"> stability limit h²/(4nu) = {dt_max_cs:.4e} s — "
                    f"Core Spreading diffusion step may over-diffuse vortex cores."
                )
            self._cs_dt_info = (
                f"CS stability limit h²/(4nu) = {dt_max_cs:.4e} s "
                f"(h = {vc.characteristic_distance:.3e} m, "
                f"nu = {vc.viscosity:.3e} m²/s)."
            )

        # RWM: warn if dt exceeds the accuracy bound h²/(4nu)
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

        # GBD: warn if dt exceeds CFL upper bound h²/(6nu)
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

        # DVH: pin dt = Δt_d so the diffusion operator fires exactly once per
        # step with the correct viscous increment.
        if vc.scheme == "DVH" and vc.viscosity is not None and vc.viscosity > 0:
            from ..physics.diffusion import _DVH_BETA

            dt_d_raw = vc.dvh_required_dt()
            # Round to 3 significant digits for clean time values.
            magnitude = _math.floor(_math.log10(abs(dt_d_raw)))
            dt_d = round(dt_d_raw, -magnitude + 2)
            # Each DVH application advances viscous time by EXACTLY Δt_d
            # (the heat-kernel width is fixed at β·R_d², independent of dt).
            if abs(self.time_step_size - dt_d) > 1e-6 * max(self.time_step_size, dt_d):
                Logging.message(
                    f"[DVH] INFO: time step overridden — "
                    f"user dt = {self.time_step_size:.4e} s → Δt_d = {dt_d:.4e} s "
                    f"(β·R_d²/(4nu), β={_DVH_BETA}, "
                    f"R_d = {vc.dvh_rd_ratio}·h = {vc.dvh_rd_ratio * vc.dvh_grid_spacing:.4e} m)."
                )
                self.time_step_size = dt_d
            self._dvh_dt_info = (
                f"DVH fires every step (Δt = Δt_d = {dt_d:.4e} s, β·R_d²/(4nu) = {dt_d:.4e} s)."
            )

        self.advection_scheme = final_config.advection.scheme
        self.stretching_scheme = final_config.stretching.scheme
        self.processing_unit = final_config.processing_unit.upper()
        self.flow_model = final_config.turbulence.flow_model.upper()
        self.viscous_scheme = final_config.viscous.scheme
        self._viscous_config = final_config.viscous
        self.stabilization_config: StabilizationConfig = final_config.stabilization
        self.particles_kernel = final_config.particles_kernel.upper()
        self.backup_frequency = final_config._backup_frequency_internal
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

    def _init_io_and_backend(self, final_config: SolverConfig, debug_mode: bool) -> None:
        """Set up output redirection, IO, precision, splitter/remesher, Taichi backend."""
        Logging.setup_output_redirection(self)
        self.io = SolverIO(self)
        self.precision = getattr(final_config, "precision", "f32")
        if self.precision not in ("f32", "f64"):
            raise ValueError(f"precision must be 'f32' or 'f64', got '{self.precision}'")
        stab = self.stabilization_config
        self._splitter = None
        if stab.max_core_radius is not None:
            from ..stabilization.splitting import ParticleSplitter

            self._splitter = ParticleSplitter(precision=self.precision)
        self._remesher = None
        if stab.remeshing_frequency is not None:
            from ..stabilization.conservative_remesh import ConservativeRemesher

            self._remesher = ConservativeRemesher(precision=self.precision)
        self.processing_unit = initialize_taichi_backend(
            self.processing_unit,
            debug_mode,
            self.precision,
            device_memory_fraction=getattr(final_config, "device_memory_fraction", 0.5),
        )
        print_openonda_header(self.precision)
        SetFlowModel(self, flow_model=self.flow_model)
        self.compute_dtype = ti.f64 if self.precision == "f64" else ti.f32
        self.accumulator_dtype = self.compute_dtype
        self.np_dtype = np.float64 if self.precision == "f64" else np.float32

    def _init_particles_and_physics(self, final_config: SolverConfig) -> None:
        """Create particle container, physics engine, source fields, background velocity."""
        max_p = getattr(final_config, "max_particles", MAX_PARTICLES)
        self.particles = Particles(max_particles=max_p, float_dtype=self.precision)
        self.physics = create_physics(
            particles_kernel=self.particles_kernel,
            accumulator_dtype=self.accumulator_dtype,
        )

        _vel_cfg = getattr(final_config, "velocity", None)
        _vel_method = "TREECODE" if (_vel_cfg and _vel_cfg.method == "TREECODE") else "DIRECT"
        _vel_theta = _vel_cfg.theta if _vel_cfg else 0.5
        self.physics.configure_velocity(_vel_method, _vel_theta)

        _visc_cfg = getattr(final_config, "viscous", None)
        if _visc_cfg is not None and hasattr(self.physics, "regen_radius_ratio"):
            self.physics.regen_radius_ratio = float(getattr(_visc_cfg, "regen_radius_ratio", 2.5))
        if hasattr(self.physics, "configure_body_mask"):
            try:
                self.physics.configure_body_mask(getattr(final_config, "body_stl", None))
            except Exception as exc:
                Logging.warning(f"Failed to configure DVH body mask: {exc}")

        # Pre-allocate grid to VPM domain size for grid-based diffusion schemes
        vpm_bounds = getattr(final_config, "vpm_domain_bounds", None)
        if vpm_bounds is not None and hasattr(self.physics, "configure_max_grid_extent"):
            vc = getattr(final_config, "viscous", None)
            if vc is not None:
                scheme = getattr(vc, "scheme", "")
                if scheme == "DVH":
                    _grid_h = getattr(vc, "dvh_grid_spacing", None)
                    _grid_pad = getattr(vc, "dvh_domain_padding", 3.0)
                elif scheme == "GBD":
                    _grid_h = getattr(vc, "gbd_grid_spacing", None)
                    _grid_pad = getattr(vc, "gbd_domain_padding", 3.0)
                else:
                    _grid_h = None
                    _grid_pad = 3.0
                if _grid_h is not None and _grid_h > 0:
                    try:
                        self.physics.configure_max_grid_extent(vpm_bounds, _grid_h, _grid_pad)
                    except Exception as exc:
                        Logging.warning(f"Failed to configure grid max extent: {exc}")
        self.particles.register_resize_callback(self.physics._resize_temp_fields)
        self.source_positions = ti.Vector.field(3, dtype=self.compute_dtype, shape=MAX_SOURCES)
        self.source_strengths = ti.field(dtype=self.compute_dtype, shape=MAX_SOURCES)
        self.source_radii = ti.field(dtype=self.compute_dtype, shape=MAX_SOURCES)
        self.num_sources = 0
        if hasattr(self.config, "background_velocity"):
            self.particles.set_background_velocity(np.array(self.config.background_velocity))

    def _init_turbulence_and_adaptation(self, final_config: SolverConfig) -> None:
        """Initialize LES turbulence model, stretching settings, regularizers, and diagnostics."""
        max_p = getattr(final_config, "max_particles", MAX_PARTICLES)
        self.LES = None
        if self.flow_model == "LES":
            self.LES = ParticlesLES(
                LES_filter_type=final_config.turbulence.model,
                max_particles=max_p,
                kernel_type=self.particles_kernel,
                cs=final_config.turbulence.cs,
                ce=final_config.turbulence.ce,
            )
        self.stretching_enabled = final_config.stretching.enabled
        self.stretching_mode = final_config.stretching.mode

        self.field_diagnostics = ParticleFieldEvaluation(
            particles_kernel=self.particles_kernel,
            max_particles=max_p,
            accumulator_dtype=self.accumulator_dtype,
        )
        self._flow_integrals: dict = {}

        # Strength relaxation — Winckelmans/Pedrizzetti direction projection.
        self._strength_relaxation = None
        stabilization = final_config.stabilization
        if stabilization.relaxation_enabled:
            from ..stabilization.strength_relaxation import StrengthRelaxation

            self._strength_relaxation = StrengthRelaxation(
                C=stabilization.relaxation_rate,
                particles_kernel=self.particles_kernel,
                max_particles=max_p,
                precision=self.precision,
                verbose=stabilization.relaxation_verbose,
                seff_min=stabilization.relaxation_seff_min,
                mode=stabilization.relaxation_mode,
                deconv=stabilization.relaxation_deconv,
                gate=stabilization.relaxation_gate,
                rlx=stabilization.relaxation_factor,
                conserve=stabilization.relaxation_conserve,
                constraint=stabilization.relaxation_constraint,
            )

    def _init_diagnostics_and_solvers(self, final_config: SolverConfig) -> None:
        """Build diagnostics history dict, impulse state, and initialize optional solvers."""
        self._diagnostics_history: dict = {
            "time": [],
            "flow_time": [],
            "vpm_total_circ_vec": [],
            "vpm_total_circ_mag": [],
            "ofw_total_circ_vec": [],
            "ofw_total_circ_mag": [],
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
        self._impulse_state: dict = VLMForceEvaluator.make_impulse_state()
        self._init_optional_solvers(final_config)
        # Runtime wall-clock profiler. ``ti.sync`` makes the timing GPU-correct
        # (Taichi kernels are asynchronous); it is shared across all backends.
        self.profiler = RuntimeProfiler(sync=ti.sync)
        self.simulation_time = 0.0

    def _setup_vlm_solver(self) -> None:
        """Configure VLM solver coupling: mesh generation, force config, stability check."""
        self.vlm_solver.ensure_mesh_generated()
        if getattr(self.vlm_solver, "lattice", None) is not None:
            Logging.info(f"VLM solver coupled with {self.vlm_solver.lattice.num_panels} panels")
            if (
                self.config.force.method != "KUTTA_JOUKOWSKI"
                or self.vlm_solver.force.method == "KUTTA_JOUKOWSKI"
            ):
                self.vlm_solver.force = self.config.force
            self.vlm_solver.check_coupling_stability(
                self.time_step_size, getattr(self.config, "background_velocity", None)
            )

    def _init_optional_solvers(self, final_config) -> None:
        """Initialize optional sub-solvers (panel, VLM) with error handling."""
        self.panel_solver = getattr(final_config, "panel_solver", None)
        if self.panel_solver is not None:
            try:
                self.panel_solver.initialize(force=True)
            except Exception as e:
                Logging.warning(f"Failed to initialize panel solver: {e}")

        self.vlm_solver = getattr(final_config, "vlm_solver", None)
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
    def from_config_file(cls, filename: str, **kwargs) -> "Solver":
        """
        Create solver instance from a JSON configuration file.

        Args:
              filename: Path to JSON configuration file containing simulation parameters
              **kwargs: Parameter overrides for the loaded configuration.
                       Any valid solver parameter can be overridden.

        Returns:
              Solver: Fully initialized VPM solver instance ready for simulation

        Example:
              >>> # Load base configuration and override time step
              >>> solver = VPMSolver.from_config_file(
              ...     'turbulence_config.json',
              ...     time_step_size=0.001,
              ...     final_time=10.0
              ... )
              >>>
              >>> # Load configuration with custom flow model
              >>> solver = VPMSolver.from_config_file(
              ...     'base_config.json',
              ...     flow_model='DNS',
              ...     viscosity=1e-6
              ... )

        Notes:
              Configuration file should contain valid JSON with solver parameters.
              See SolverConfig documentation for complete parameter list.
        """
        config = SolverConfig.load_from_file(filename)
        return cls(config=config, **kwargs)

    def save_config(self, filename: str) -> None:
        """Save the current solver configuration to a JSON file."""
        self.io.save_config(filename)

    def update_config(self, **kwargs) -> None:
        """
        Update solver configuration parameters dynamically.

        Args:
              **kwargs: Configuration parameters to update.
                       Parameter names should match SolverConfig attributes.

        Example:
              >>> # Update time stepping parameters
              >>> solver.update_config(
              ...     time_step_size=0.0005,
              ...     final_time=20.0,
              ...     backup_frequency=100
              ... )
              >>>
              >>> # Update numerical settings
              >>> solver.update_config(
              ...     kernel_type='gaussian',
              ...     cutoff_radius=3.0
              ... )

        Warning:
              Changing parameters during simulation may affect conservation
              properties and numerical stability. Best used between simulation
              runs or during initialization phase.
        """
        # Update configuration
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                # Also update the solver attribute if it exists
                if hasattr(self, key):
                    setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")

        # Re-validate configuration
        self.config._validate_config()

        # Handle turbulence configuration changes
        if "turbulence" in kwargs and self.LES is not None:
            self.LES = ParticlesLES.rebuild(
                kwargs["turbulence"],
                getattr(self.config, "max_particles", MAX_PARTICLES),
                self.particles_kernel,
            )

        Logging.message(f"Configuration updated: {list(kwargs.keys())}")

    # MAGIC METHODS AND BASIC OPERATIONS

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

    # CORE SIMULATION AND TIME STEPPING

    def print_timing(self) -> None:
        """Print the cumulative runtime-profiling report.

        Reports, per solver stage, the number of calls, cumulative time, average
        time per call, and percent of total wall-clock time, plus the measured /
        unprofiled split and the per-step average.  Safe to call at any time
        (e.g. after a run, or between steps).  Output is routed through the
        central :class:`Logging` sink, so it matches the rest of the solver.

        See :attr:`profiler` (:class:`RuntimeProfiler`) for the underlying
        statistics and ``profiler.reset()`` to clear them.
        """
        self.profiler.report()

    def update_state(self) -> None:
        """
        Perform one complete time step of the VPM simulation.

        This method orchestrates the complete simulation update including:
        - Time step advancement and timing
        - Velocity field computation
        - Turbulence state updates (for DNS/LES models)
        - Vortex stretching and viscous diffusion
        - Particle advection
        - Diagnostic logging and backup operations (only when needed)

        Note:
              All stages are timed using ti.sync() for accurate GPU profiling.

        Raises:
              RuntimeError: If simulation update fails
        """

        # Advance step counter and print the step header.
        self._advance_time_step()

        # Update particle time step for cache invalidation
        self.particles.time_step = self.time_step

        # The profiler times the whole step (denominator for the report) and each
        # named stage below; ``section`` synchronises the backend around the block.
        with self.profiler.step():
            # 0. VLM COUPLING (Shed wake particles from lifting surfaces)
            if self.vlm_solver is not None:
                with self.profiler.section("VLM coupling"):
                    self._advance_vlm(self.time_step_size)

            # 0.5 PANEL SOLVER COUPLING
            if self.panel_solver is not None:
                with self.profiler.section("Panel coupling"):
                    self._advance_panel()

            # 1. VELOCITY & GRADIENTS (At t_n)
            _adv = (self.config.advection.scheme if self.config.advection else "RK4").upper()
            # Fuse u + ∇u into one tree build + one traversal when both are needed at
            # the same t_n configuration and the velocity carries no contribution the
            # fused kernel does not model (sources/panel) nor post-processing
            # (velocity_override).  The fused pass writes particles.velocity = v(x_n),
            # which the advection integrator then reuses as its first RK stage (k1).
            _fuse_vel_grad = (
                self.flow_model != "POTENTIAL"
                and _adv != "NONE"
                and self.num_sources == 0
                and self.panel_solver is None
                and getattr(self.physics, "velocity_override", None) is None
            )
            if _fuse_vel_grad:
                with self.profiler.section("Velocity + gradients"):
                    self._update_velocity_and_gradients()
            else:
                if _adv == "NONE" or self.num_sources > 0 or self.panel_solver is not None:
                    with self.profiler.section("Velocity"):
                        self._update_velocities()
                with self.profiler.section("Velocity gradients"):
                    self._update_velocity_gradients()

            with self.profiler.section("LES update"):
                self._update_LES_state()

            # 2. CONVECTION (Advection x_n -> x_n+1)
            with self.profiler.section("Advection"):
                self._update_positions(precomputed_k1=_fuse_vel_grad)

            # 3. DIFFUSION & STRETCHING (Update alpha)
            with self.profiler.section("Stretching + diffusion"):
                self._update_strength()

            # 3.5 FLOW INTEGRALS (Recomputed at t_n+1 after advection/strength update)
            _diag_due = self.logging_frequency > 0 and self.time_step % self.logging_frequency == 0
            if _diag_due:
                with self.profiler.section("Flow integrals"):
                    self._update_all_flow_integrals()
            elif self.vlm_solver is not None:
                with self.profiler.section("VLM diagnostics"):
                    self._record_vlm_diagnostics()

            # 4. ADAPTATION (Splitting / Remeshing / Wake Cutoff)
            with self.profiler.section("Adaptation"):
                self._update_adaptation()

            # 5. DIAGNOSTICS & IO
            with self.profiler.section("Backup / IO"):
                self._backup_solution()

        # Print this step's wall time (+ optional breakdown) and keep the
        # public ``simulation_time`` mirror in sync with the profiler.
        self.profiler.report_step()
        self.simulation_time = self.profiler.wall_time

        # Periodic cumulative runtime-profiling report.
        if self.timing_frequency > 0 and self.time_step % self.timing_frequency == 0:
            self.profiler.report()

        # Log flow diagnostics at specified frequency
        if self.logging_frequency > 0 and self.time_step % self.logging_frequency == 0:
            self.log_diagnostics()

    def _advance_time_step(self) -> None:
        """
        Advance the simulation time step and print current state info.

        The full-step wall time is measured by ``self.profiler`` (see
        :meth:`update_state`); this method only advances the counter and prints
        the step header.

        Note: Flow time is calculated as (time_step * time_step_size) and then
        rounded to 12 decimal places to eliminate floating point accumulation errors.
        This ensures consistent, clean time stamps across all output files.
        """

        # Advance time step counter
        self.time_step += 1

        self.flow_time = round(self.time_step * self.time_step_size, 6)

        Logging.message(
            f"\nTime-step: {self.time_step:d}   Flow time: {self.flow_time:0.2E} s",
            flush=True,
        )

    # Update flow diagnostics and log if enabled
    def log_diagnostics(self) -> None:
        """
        Log flow diagnostics if logging is enabled.

        Note: This method uses flow integrals that were already computed
        during the latest update_state() call. It does not recalculate
        them to avoid corrupting the time history used for dE/dt calculation.
        """

        # Print all flow diagnostics (dE/dt, enstrophy, helicity, impulses, etc.)
        Logging.flow_diagnostics(self)

        # Export flow integrals to CSV (append one row per logging event)
        self._export_flow_integrals_csv()

        # Print turbulence information for LES/DNS models (skip for potential flow)
        if self.LES is not None:
            Logging.les_diagnostics(self)

        # Execute field samplers if configured
        self._execute_samplers()

    def _export_flow_integrals_csv(self) -> None:
        """Append one row of flow integrals to ``<backup_directory>/samples/flow_integrals.csv``."""
        import pandas as pd

        samples_dir = Path(self.backup_directory) / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        csv_path = samples_dir / "flow_integrals.csv"

        impulse = self._flow_integrals.get("linear_impulse", np.zeros(3))
        ang_impulse = self._flow_integrals.get("angular_impulse", np.zeros(3))
        strength = self._flow_integrals.get("strength", np.zeros(3))

        row = {
            "time": self.flow_time,
            "step": self.time_step,
            "kinetic_energy": self.total_kinetic_energy,
            "enstrophy": self.total_enstrophy,
            "dEdt": self.kinetic_energy_dissipation_rate,
            "neg_nu_enstrophy": self.vorticity_dissipation_rate,
            "helicity": self.total_helicity,
            "strength_magnitude": self.total_strength_magnitude,
            "strength_x": float(strength[0]),
            "strength_y": float(strength[1]),
            "strength_z": float(strength[2]),
            "impulse_x": float(impulse[0]),
            "impulse_y": float(impulse[1]),
            "impulse_z": float(impulse[2]),
            "angular_impulse_x": float(ang_impulse[0]),
            "angular_impulse_y": float(ang_impulse[1]),
            "angular_impulse_z": float(ang_impulse[2]),
            "n_particles": self.particles.number_of_particles,
        }

        df = pd.DataFrame([row])
        if not csv_path.exists():
            df.to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path, mode="a", header=False, index=False)

    def _execute_samplers(self) -> None:
        """Execute all configured field samplers (delegates to SamplerExecutor)."""
        SamplerExecutor.execute(self)

    def _prepare_sampler_context(self, sampler_entry, samples_dir):
        """Delegate to SamplerExecutor."""
        return SamplerExecutor._prepare_context(sampler_entry, samples_dir)

    def _save_sampler_output(self, sampler, name_prefix, solution_dir, seq_num):
        """Delegate to SamplerExecutor."""
        SamplerExecutor._save_output(
            sampler, self, name_prefix, solution_dir, seq_num, self.flow_time
        )

    def _write_pvd_file(self, output_dir, name_prefix, entries):
        """Delegate to SamplerExecutor."""
        SamplerExecutor._write_pvd(output_dir, name_prefix, entries)

    # PARTICLE PROPERTY ACCESSORS
    def _get_particle_field(self, method_name: str) -> np.ndarray:
        """Generic helper to get particle field data via cpu() methods."""
        return getattr(self.particles, f"{method_name}_cpu")()

    @property
    def particles_positions(self) -> np.ndarray:
        """
        Get particle positions array.

        Returns:
              np.ndarray: Array of shape (N, 3) containing [x, y, z] coordinates
                         for all N particles in the system. Units: [m]

        Example:
              >>> positions = solver.particles_positions
              >>> x_coords = positions[:, 0]  # All x-coordinates
              >>> first_particle_pos = positions[0]  # [x, y, z] of first particle
        """
        return self._get_particle_field("position")

    @property
    def particles_velocities(self) -> np.ndarray:
        """
        Get particle velocities array.

        Returns:
              np.ndarray: Array of shape (N, 3) containing [u, v, w] velocity
                         components for all N particles. Units: [m/s]

        Example:
              >>> velocities = solver.particles_velocities
              >>> u_components = velocities[:, 0]  # All x-velocity components
              >>> particle_speed = np.linalg.norm(velocities[0])  # Speed of first particle
        """
        return self._get_particle_field("velocity")

    @property
    def particles_strengths(self) -> np.ndarray:
        """
        Get particle vortex strengths array.

        Returns:
              np.ndarray: Array of shape (N, 3) containing [ωx, ωy, ωz] vorticity
                         components for all N particles. Units: [1/s]

        Example:
              >>> strengths = solver.particles_strengths
              >>> omega_z = strengths[:, 2]  # All z-vorticity components
              >>> vorticity_magnitude = np.linalg.norm(strengths, axis=1)
        """
        return self._get_particle_field("circulation")

    @property
    def particles_radii(self) -> np.ndarray:
        """
        Get particle core radii array.

        Returns:
              np.ndarray: Array of shape (N,) containing core radius for each
                         particle. Units: [m]

        Example:
              >>> radii = solver.particles_radii
              >>> avg_radius = np.mean(radii)
              >>> max_radius = np.max(radii)
        """
        return self._get_particle_field("radius")

    @property
    def particles_volumes(self) -> np.ndarray:
        """
        Get particle volumes array.

        Returns:
              np.ndarray: Array of shape (N,) containing volume for each
                         particle. Units: [m³]

        Example:
              >>> volumes = solver.particles_volumes
              >>> total_volume = np.sum(volumes)
              >>> volume_distribution = volumes / np.mean(volumes)
        """
        return self._get_particle_field("volume")

    @property
    def particles_group_ids(self) -> np.ndarray:
        """
        Get particle group identifiers array.

        Returns:
              np.ndarray: Array of shape (N,) containing integer group ID
                         for each particle. Used for tracking particle origins
                         and applying group-specific operations.

        Example:
              >>> group_ids = solver.particles_group_ids
              >>> group_0_particles = np.where(group_ids == 0)[0]
              >>> unique_groups = np.unique(group_ids)
        """
        return self._get_particle_field("group_id")

    @property
    def particles_zone_ids(self) -> np.ndarray:
        """
        Get particle zone identifiers array.

        Returns:
            np.ndarray: Array of shape (N,) containing integer zone ID
                        for each particle. Used for spatial zone tracking

        Example:
              >>> zone_ids = solver.particles_zone_ids
              >>> buffer_particles = np.where(zone_ids == 2)[0]  # Buffer/wake particles
              >>> reset_particles = np.where(zone_ids == 1)[0]   # Interior injection
        """
        return self._get_particle_field("zone_id")

    @property
    def particles_viscosities(self) -> np.ndarray:
        """
        Get particle molecular viscosities array.

        Returns:
              np.ndarray: Array of shape (N,) containing molecular viscosity
                         for each particle. Units: [m²/s]

        Example:
              >>> nu = solver.particles_viscosities
              >>> reynolds_number = np.linalg.norm(velocities, axis=1) * radii / nu
        """
        return self._get_particle_field("viscosity")

    @property
    def particles_viscosities_t(self) -> np.ndarray:
        """
        Get particle turbulent viscosities array.

        Returns:
              np.ndarray: Array of shape (N,) containing turbulent viscosity
                         for each particle. Units: [m²/s]

        Notes:
              Turbulent viscosity represents subgrid-scale effects in LES models
              or additional mixing in DNS with turbulence models.
        """
        return self._get_particle_field("viscosity_turbulent")

    @property
    def particles_viscosities_eff(self) -> np.ndarray:
        """
        Get particle effective viscosities array.

        Returns:
              np.ndarray: Array of shape (N,) containing effective viscosity
                         (molecular + turbulent) for each particle. Units: [m²/s]

        Notes:
              Effective viscosity = molecular viscosity + turbulent viscosity.
              This is the total viscosity used in diffusion calculations.
        """
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
        """
        Get the global background velocity (freestream).

        Returns:
              np.ndarray: Background velocity vector [ux, uy, uz] in m/s.
        """
        return self.particles.velocity_background_cpu()

    @property
    def particles_vorticities(self) -> np.ndarray:
        """Get particle vorticities array."""
        return self._get_particle_field("vorticity")

    @property
    def particles_circulation(self) -> np.ndarray:
        """Alias for particle vortex strengths (circulation).

        Historically some code referenced ``solver.particles_circulation``.
        Return the same data as :py:meth:`particles_strengths` for compatibility.
        """
        return self._get_particle_field("circulation")

    # FLOW INTEGRALS AND DIAGNOSTIC PROPERTIES
    def _update_all_flow_integrals(self) -> None:
        """
        Update all flow integral quantities using the field diagnostics module.

        This method computes all flow integral quantities (energy, helicity, enstrophy,
        dissipation rates, impulses) in a single efficient GPU kernel call using the
        ParticleFieldEvaluation class. The energy dissipation rate is computed using
        actual time steps (not assumed constant).
        """
        # Compute all flow integrals, passing current flow_time for accurate dE/dt calculation
        self._flow_integrals = self.field_diagnostics.compute_flow_integrals(
            self.particles, self.flow_time
        )
        self._record_centroid_history()
        self._record_flow_time_history()
        self._record_vlm_diagnostics()

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
        VLMDiagnostics.record_vlm_diagnostics(
            self.vlm_solver,
            self.particles,
            self.particles_strengths,
            self._diagnostics_history,
            self.time_step,
            self.flow_time,
            self.backup_directory,
        )
        VLMLoadingDistribution.record_loading_distributions(
            self.vlm_solver,
            self._diagnostics_history,
            self.time_step,
            self.flow_time,
            self.backup_directory,
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
        )

    @property
    def total_kinetic_energy(self) -> float:
        """
        Get total kinetic energy of the system.

        Returns:
            float: Total kinetic energy [J] or [m²/s²] per unit density
        """
        return self._flow_integrals.get("kinetic_energy", 0.0)

    @property
    def total_helicity(self) -> float:
        """
        Get total helicity of the system.

        Returns:
            float: Total helicity [m³/s²]
        """
        return self._flow_integrals.get("helicity", 0.0)

    @property
    def total_enstrophy(self) -> float:
        """
        Get total enstrophy of the system.

        Returns:
            float: Total enstrophy [1/s²]
        """
        return self._flow_integrals.get("enstrophy", 0.0)

    @property
    def vorticity_dissipation_rate(self) -> float:
        """
        Get vorticity dissipation rate.

        Returns:
            float: Vorticity dissipation rate [J/s]
        """
        return self._flow_integrals.get("vorticity_dissipation_rate", 0.0)

    @property
    def kinetic_energy_dissipation_rate(self) -> float:
        """
        Get kinetic energy dissipation rate computed using finite differences with actual time steps.

        Returns:
            float: Energy dissipation rate [J/s]
        """
        return self._flow_integrals.get("kinetic_energy_dissipation_rate", 0.0)

    @property
    def total_strength(self) -> np.ndarray:
        """
        Get total strength vector of the system.

        Returns:
            np.ndarray: Total strength vector [Γx, Γy, Γz] [1/s]
        """
        return self._flow_integrals.get("strength", np.array([0.0, 0.0, 0.0]))

    @property
    def total_strength_magnitude(self) -> float:
        """
        Get total strength magnitude of the system.

        Returns:
            float: Total strength magnitude [1/s]
        """
        return self._flow_integrals.get("strength_magnitude", 0.0)

    @property
    def total_linear_impulse(self) -> np.ndarray:
        """
        Get total linear impulse of the system.

        Always recomputed from current particle state to avoid cache
        invalidation bugs when particles are added, removed, or remeshed.

        Returns:
            np.ndarray: Linear impulse vector [Ix, Iy, Iy, Iz] [m³/s]
        """
        # Always compute fresh from current particles (no cache)
        integrals = self.field_diagnostics.compute_flow_integrals(
            self.particles, self.flow_time, record_history=False
        )
        return integrals.get("linear_impulse", np.array([0.0, 0.0, 0.0]))

    @property
    def total_angular_impulse(self) -> np.ndarray:
        """
        Get total angular impulse of the system.

        Returns:
            np.ndarray: Angular impulse vector [Lx, Ly, Lz] [m⁴/s]
        """
        return self._flow_integrals.get("angular_impulse", np.array([0.0, 0.0, 0.0]))

    @property
    def centroids_of_circulation(self) -> dict[int, np.ndarray]:
        """
        Get circulation centroids for each particle group.

        Returns:
            Dict[int, np.ndarray]: Dictionary mapping group_id to centroid position vector [x, y, z]
        """
        return self.field_diagnostics.compute_centroids_of_circulation(self.particles)

    @property
    def centroid_of_circulation(self) -> np.ndarray:
        """
        Get global centroid of circulation (weighted by |Γ|).

        Returns:
            np.ndarray: Centroid position vector [x, y, z]
        """
        return self.field_diagnostics.compute_centroid_of_circulation(self.particles)

    def compute_forces(
        self, density: float = 1.225, V_ref_mag: float | None = None
    ) -> dict[str, np.ndarray | float]:
        """
        Compute aerodynamic forces using configured method.

        This is the unified API for force evaluation. The method used depends
        on the solver configuration (config.force.method):

        - 'KUTTA_JOUKOWSKI': Classical pressure integration on bound panels (conventional)
        - 'IMPULSE': Force from impulse time derivative F = -dI/dt (experimental)

        Args:
            density: Fluid density [kg/m³]
            V_ref_mag: Reference velocity magnitude [m/s] (used for K-J method)
                       If None, uses background velocity magnitude

        Returns:
            Dictionary with keys:
            - 'method': str - Method used ('KUTTA_JOUKOWSKI' or 'IMPULSE')
            - 'force': np.ndarray - Force vector [Fx, Fy, Fz] [N]
            - 'Fx', 'Fy', 'Fz': float - Individual force components [N]
            - Additional keys depending on method

        Example::

            >>> # Configure solver for impulse-based forces
            >>> config = SolverConfig(force=ForceConfig.impulse_based(order=2))
            >>> solver = Solver(config=config)
            >>>
            >>> # Compute forces (uses impulse method)
            >>> result = solver.compute_forces(density=1.225)
            >>> print(f"Force: {result['force']}")
            >>> print(f"Method: {result['method']}")
            >>>
            >>> # Or use conventional K-J method
            >>> config = SolverConfig(force=ForceConfig.kutta_joukowski())
            >>> solver = Solver(config=config)
            >>> result = solver.compute_forces(density=1.225)

        Note:
            The method compute_impulse_based_force()
            and compare_force_methods() are still available.
        """
        method = self.config.force.method

        if method == "KUTTA_JOUKOWSKI":
            return self._compute_forces_kutta_joukowski(density, V_ref_mag)
        elif method == "IMPULSE":
            return self._compute_forces_impulse(density)
        else:
            raise ValueError(f"Unknown force method: {method}")

    def _compute_forces_kutta_joukowski(
        self, density: float, V_ref_mag: float | None
    ) -> dict[str, np.ndarray | float]:
        """Compute forces via the Kutta-Joukowski theorem. Delegates to VLMForceEvaluator."""
        return VLMForceEvaluator.compute_kutta_joukowski(
            self.vlm_solver, self.background_velocity, density, V_ref_mag
        )

    def _compute_forces_impulse(self, density: float) -> dict[str, np.ndarray | float]:
        """Compute forces via impulse F = -dI/dt. Delegates to VLMForceEvaluator."""
        return VLMForceEvaluator.compute_impulse(
            self._impulse_state,
            self.vlm_solver,
            self.total_linear_impulse,
            self.flow_time,
            density,
            self.config.force,
        )

    def compare_force_methods(
        self, density: float = 1.225, V_ref_mag: float | None = None
    ) -> dict[str, np.ndarray]:
        """Compare impulse-based vs Kutta-Joukowski forces. Delegates to VLMForceEvaluator."""
        return VLMForceEvaluator.compare_methods(
            self._impulse_state,
            self.vlm_solver,
            self.total_linear_impulse,
            self.background_velocity,
            self.flow_time,
            density,
            self.config.force,
            V_ref_mag,
        )

    # PARTICLE PHYSICS COMPUTATIONS (PER-PARTICLE ANALYSIS)
    def compute_kinetic_energies(self) -> np.ndarray:
        """
        Compute kinetic energy for each particle.

        Returns:
              np.ndarray: Array of shape (N,) containing kinetic energy
                         for each particle. Units: [J] or [m²/s²] per unit density

        Notes:
              Used for detailed per-particle analysis and diagnostics.
        """
        return self.field_diagnostics.compute_particles_kinetic_energy(self.particles)

    def compute_helicities(self) -> np.ndarray:
        """
        Compute helicity for each particle.

        Returns:
              np.ndarray: Array of shape (N,) containing helicity density
                         for each particle. Units: [m/s²]

        Notes:
              Local helicity h_i = v_i · ω_i where v_i is velocity and ω_i is vorticity.
              Measures alignment between velocity and vorticity at each particle.
              Used for detailed per-particle analysis and diagnostics.
        """
        return self.field_diagnostics.compute_particles_helicity(self.particles)

    def compute_enstrophies(self) -> np.ndarray:
        """
        Compute enstrophy for each particle.

        Returns:
              np.ndarray: Array of shape (N,) containing enstrophy density
                         for each particle. Units: [1/s²]

        Notes:
              Enstrophy density Ω_i = (1/2) |ω_i|² where ω_i is vorticity.
              Measures intensity of rotational motion at each particle.
              Used for detailed per-particle analysis and diagnostics.
        """
        return self.field_diagnostics.compute_particles_enstrophy(self.particles)

    # FIELD COMPUTATION AT ARBITRARY POINTS
    def compute_target_vorticities(self, grid_positions: np.ndarray) -> np.ndarray:
        """
        Compute vorticity field at arbitrary spatial points.

        Args:
              grid_positions: Array of shape (N, 3) containing [x, y, z] coordinates
                             where N is number of evaluation points. Units: [m]

        Returns:
              np.ndarray: Vorticity field of shape (N, 3) with [ωx, ωy, ωz] components
                         at each evaluation point. Units: [1/s]

        Example:
              >>> probe_points = np.array([[0.1, 0.2, 0.3],
              ...                          [0.4, 0.5, 0.6]])
              >>> vorticity = solver.compute_target_vorticities(probe_points)
              >>> omega_magnitude = np.linalg.norm(vorticity, axis=1)

        Notes:
              Uses kernel superposition: ω(x) = ∑ᵢ ζ(x - xᵢ, σᵢ) * Γᵢ
              where ζ is vorticity kernel, σᵢ is core radius, Γᵢ is circulation.
        """
        # Delegate to physics layer which handles the kernel call properly
        return self.physics.compute_target_vorticities(self.particles, grid_positions)

    def compute_target_velocities(
        self,
        grid_positions: np.ndarray,
        include_freestream: bool = True,
        zone_mask: np.ndarray | None = None,
        include_body: bool = True,
    ) -> np.ndarray:
        """
        Compute velocity field at arbitrary spatial points.

        Args:
              grid_positions: Array of shape (N, 3) containing [x, y, z] coordinates
                             where N is number of evaluation points. Units: [m]
              include_freestream: If True (default), includes background velocity (U_inf).
                                 If False, returns only particle-induced velocity.
              zone_mask: Optional boolean mask of shape (n_particles,). If provided,
                        only particles where zone_mask[i]=True contribute to the
                        velocity computation. Used for zone-aware BC computation
                        to exclude interior (ZONE_RESET) particles.

        Returns:
              np.ndarray: Velocity field of shape (N, 3) with [u, v, w] components
                         at each evaluation point. Units: [m/s]

        Example:
              >>> # Total velocity (induced + freestream)
              >>> velocities = solver.compute_target_velocities(points)

              >>> # Induced velocity only (for coupling with external solvers)
              >>> v_induced = solver.compute_target_velocities(points, include_freestream=False)

              >>> # Zone-aware: only exterior particles contribute
              >>> exterior_mask = (zone_ids == ZONE_OUTSIDE) | (zone_ids == ZONE_BUFFER)
              >>> v_wake = solver.compute_target_velocities(points, zone_mask=exterior_mask)

        Notes:
              Uses Biot-Savart law: u(x) = ∑ᵢ K(x - xᵢ, σᵢ) × Γᵢ
              where K is velocity kernel, σᵢ is core radius, Γᵢ is circulation.
        """
        # Delegate to physics layer which handles the kernel call properly
        velocities = self.physics.compute_target_velocities(
            self.particles,
            grid_positions,
            include_freestream=include_freestream,
            zone_mask=zone_mask,
        )

        # Add source induction for potential flow body blockage
        if self.num_sources > 0:
            n_targets = len(grid_positions)
            # Reuse pre-allocated target fields from physics layer to avoid frequent memory allocations
            self.physics._resize_target_fields(n_targets)
            target_pos_ti = self.physics.target_positions
            target_vel_ti = self.physics.target_velocities

            # Transfer results to Taichi fields for kernel execution
            target_pos_ti.from_numpy(grid_positions.astype(self.np_dtype))
            target_vel_ti.from_numpy(velocities.astype(self.np_dtype))

            self.physics.kernels["compute_target_source_velocity_kernel"](
                target_pos_ti,
                self.source_positions,
                self.source_strengths,
                self.source_radii,
                target_vel_ti,
                n_targets,
                self.num_sources,
            )
            # Copy results back to NumPy, respecting the actual number of points
            velocities = self.physics.extract_target_velocities(n_targets)

        # Add boundary-element (panel) body induction
        body_fn = getattr(self, "_body_induced_fn", None)
        if include_body and body_fn is not None:
            velocities = velocities + np.asarray(
                body_fn(grid_positions), dtype=velocities.dtype
            ).reshape(velocities.shape)

        return velocities

    def set_body_induced_velocity(self, fn) -> None:
        """Set (or clear) the boundary-element body-induction callback.

        ``fn(points: np.ndarray (N,3)) -> np.ndarray (N,3)`` returns the
        physical velocity induced by the body panel model at ``points``.  Added
        to every ``compute_target_velocities`` result (samplers, donor BC,
        diagnostics) so the VPM field carries the body's irrotational blockage.
        Pass ``None`` to disable.
        """
        self._body_induced_fn = fn

    def set_surface_sources(
        self, positions: np.ndarray, strengths: np.ndarray, radii: np.ndarray
    ) -> None:
        """
        Set surface source particles for body blockage (potential flow correction).

        Args:
              positions: Array of shape (S, 3) with source coordinates [m]
              strengths: Array of shape (S,) with source strengths [m³/s]
              radii: Array of shape (S,) with source core radii [m]
        """
        self.num_sources = len(positions)
        if self.num_sources > MAX_SOURCES:
            Logging.warning(f"Clipping {self.num_sources} sources to {MAX_SOURCES}")
            self.num_sources = MAX_SOURCES

        n = self.num_sources
        # Pad to MAX_SOURCES — Taichi from_numpy requires exact shape match
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
        """
        Compute velocity gradient tensor ∇u at arbitrary spatial points.

        Uses direct VPM kernel evaluation to compute gradU[i,j] = ∂uᵢ/∂xⱼ at each point.

        Args:
              grid_positions: Array of shape (N, 3) containing [x, y, z] coordinates
                             where N is number of evaluation points. Units: [m]

        Returns:
              np.ndarray: Velocity gradient tensors of shape (N, 9) as flat arrays
                         [∂u/∂x, ∂u/∂y, ∂u/∂z, ∂v/∂x, ∂v/∂y, ∂v/∂z, ∂w/∂x, ∂w/∂y, ∂w/∂z]
                         Can be reshaped to (N, 3, 3) for tensor operations

        Example:
              >>> face_centers = ofw.get_boundary_face_center_coordinates("inlet")
              >>> gradU_flat = vpm_solver.compute_target_velocity_gradients(face_centers)
              >>> gradU = gradU_flat.reshape(-1, 3, 3)  # Shape: (N, 3, 3)

        Notes:
              - Direct kernel evaluation (no finite differences)
              - Useful for passing velocity gradients to OpenFOAM
              - Implementation: Taichi kernel in physics module
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
    ) -> dict | tuple[dict, np.ndarray]:
        """Compute pressure gradient and individual components at arbitrary spatial points.

        Uses the full momentum equation:
            ∇p = -ρ [ ∂u/∂t + (u·∇)u - nu∇²u ]

        Args:
              grid_positions: Array of shape (N, 3) with evaluation coordinates [m]
              density: Fluid density [kg/m³]. Default: 1.0
              nu: Kinematic viscosity [m²/s]. If None, uses particle average.
              include_viscous: Include viscous term nu∇²u. Default: True.
              include_temporal: Include unsteady term ∂u/∂t. Default: True.
              include_freestream: Include background velocity. Default: True.
              h: Step size for Laplacian finite differences [m].
                 If None, uses average particle radius.
              temporal_method: 'lagrangian' (particle-based) or 'eulerian' (snapshots).
              velocity_previous: Previous velocity field for Eulerian method [N, 3].
              dt: Time step for Eulerian method [s].
              return_velocity: If True, also return the internally computed u_target.

        Returns:
              If return_velocity is False (default):
                  dict with keys: 'grad_p', 'convective', 'viscous', 'temporal'
              If return_velocity is True:
                  tuple[dict, np.ndarray]: (components_dict, u_target [N, 3])
        """
        from source.solvers.VPM.physics.pressure import PressurePhysics

        if nu is None:
            nu = (
                float(np.mean(self.particles_viscosities))
                if self.particles.number_of_particles > 0
                else 1e-5
            )
        if self.particles.number_of_particles > 0:
            self.physics.compute_velocity_gradients(self.particles)
        if not hasattr(self, "_pressure_physics"):
            self._pressure_physics = PressurePhysics(particles_kernel=self.particles_kernel)
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

    # DIAGNOSTICS AND MONITORING
    def info(self):
        """
        Print comprehensive information about the solver and all submodels.

        This method provides a complete overview of the solver state including:
        - Solver configuration
        - Particle system statistics
        - Physics model details
        - Viscous diffusion model
        - Turbulence model (if LES)
        - Monitoring and I/O settings

        All information is delegated to appropriate classes for maintainability.

        Example:
              >>> solver = Solver(config)
              >>> solver.info()  # Print comprehensive solver information
        """
        info_str = Logging.solver_info(self)
        Logging.message(info_str)

    # PARTICLE MANAGEMENT
    def remove_particles(
        self, particle_indices: list[int] | None = None, remove_all: bool = False
    ) -> None:
        """
        Remove particles from the system with circulation tracking.

        Tracks the total circulation removed for conservation diagnostics.
        This enables validation of Kelvin's theorem under particle removal:
            Γ_expected = Γ_initial - Σ(Γ_removed)

        Args:
              particle_indices: List of particle indices to remove (None for all)
              remove_all: If True, remove all particles
        """
        # Track circulation before removal (for conservation diagnostics)
        if particle_indices is not None and len(particle_indices) > 0:
            # Reduce circulation ΣΓ and impulse 0.5·Σ(r×Γ) of the removed subset
            # entirely on device — only the index list goes up and two 3-vectors
            # come back, instead of downloading every particle's position/strength.
            circ_removed, impulse_removed = self.particles.subset_moments(particle_indices)
            self._particles_removed_this_step = len(particle_indices)
            self._circulation_removed_this_step = circ_removed

            # Accumulate all removals (in case of multiple remove calls between force evaluations)
            self._impulse_state["removed_accumulated"] += impulse_removed

        elif remove_all:
            # Removal calculation for ALL particles — summed on device (ΣΓ) so we
            # don't download every circulation just to reduce it.
            circ_removed = self.particles.total_circulation()

            self._particles_removed_this_step = len(self.particles)
            self._circulation_removed_this_step = circ_removed

            # Clear history on total removal
            self._impulse_state["history"] = []
            self._impulse_state["time_history"] = []
            self._impulse_state["removed_accumulated"] = np.zeros(3)
        else:
            self._particles_removed_this_step = 0
            self._circulation_removed_this_step = np.zeros(3)

        # Perform removal via particle container
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
        """
        Add multiple vortex particles to the system.

        Args:
              position: Particle positions [N, 3]
              velocity: Particle velocities [N, 3]
              circulation: Particle circulation (α = ω·V) [N, 3]
              radius: Particle core radii [N]
              volume: Particle volumes [N]
              viscosity: Molecular viscosities [N] (required)
              viscosity_turbulent: Turbulent viscosities [N] (optional)
              group_id: Particle group identifiers [N] (optional)
              zone_id: Spatial zone identifiers [N] (optional)
              velocity_gradient: Velocity gradient tensors [N, 3, 3] (optional)
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
    ) -> None:
        """Replace the active particle cloud in one field-upload operation."""
        p_circ = self.particles_circulation
        circ_removed = p_circ.sum(axis=0) if len(p_circ) > 0 else np.zeros(3)
        self._particles_removed_this_step = len(self.particles)
        self._circulation_removed_this_step = circ_removed
        self._impulse_state["history"] = []
        self._impulse_state["time_history"] = []
        self._impulse_state["removed_accumulated"] = np.zeros(3)

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
        """
        Set the time step size for the simulation.

        Args:
              time_step_size: New time step size [s]
        """
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
        """
        Set particle properties with validation and cache invalidation.

        This method provides a high-level interface for updating particle properties
        with proper validation (NaN/Inf checks, shape consistency, dtype conversion)
        and cache invalidation. Supports partial updates - only specified properties
        are modified, others remain unchanged.

        Args:
              **properties: Keyword arguments specifying properties to update.
                           Valid property names:
                           - positions: np.ndarray(N, 3) - particle positions [m]
                           - velocities: np.ndarray(N, 3) - particle velocities [m/s]
                           - strengths: np.ndarray(N, 3) - vortex strengths [m²/s]
                           - vorticities: np.ndarray(N, 3) - vorticities [1/s]
                           - radii: np.ndarray(N,) - core radii [m]
                           - volumes: np.ndarray(N,) - volumes [m³]
                           - viscosities: np.ndarray(N,) - molecular viscosities [m²/s]
                           - viscosities_t: np.ndarray(N,) - turbulent viscosities [m²/s]
                           - viscosities_eff: np.ndarray(N,) - effective viscosities [m²/s]
                           - group_ids: np.ndarray(N,) - integer group identifiers
                           - grad_u: np.ndarray(N, 3, 3) - velocity gradient tensors
                           - Sij: np.ndarray(N, 3, 3) - strain rate tensors

        Raises:
              ValueError: If property name is invalid
              ValueError: If array contains NaN or Inf values
              ValueError: If array shape doesn't match particle count
              ValueError: If array dtype is incompatible

        Examples:
              >>> # Update positions only
              >>> new_positions = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
              >>> solver.set_particles_properties(positions=new_positions)

              >>> # Update multiple properties at once
              >>> solver.set_particles_properties(
              ...     velocities=new_velocities,
              ...     strengths=new_strengths,
              ...     radii=new_radii
              ... )

              >>> # Update from FVM coupling (common use case)
              >>> vorticity_from_fvm = fvm_solver.get_circulation_field()
              >>> solver.set_particles_properties(vorticities=vorticity_from_fvm)

        Notes:
              - All validation is performed before any updates (atomic operation)
              - Cache is invalidated after successful update
              - Efficient: only specified properties are transferred to GPU
              - Compatible with Taichi architecture (minimal host↔device transfers)
        """
        if not properties:
            return  # No properties to update

        # Valid particle property names (mapped to internal field names)
        valid_properties = {
            "positions": "positions",
            "velocities": "velocities",
            "strengths": "strengths",
            "vorticities": "vorticities",
            "radii": "radii",
            "volumes": "volumes",
            "viscosities": "viscosities",
            "viscosities_t": "viscosities_t",
            "viscosities_eff": "viscosities_eff",
            "group_ids": "group_ids",
            "grad_u": "gradU",  # Note: internal field name is gradU, not grad_u
            "Sij": "Sij",
        }

        # Validate all property names first
        for prop_name in properties:
            if prop_name not in valid_properties:
                raise ValueError(
                    f"Invalid property name '{prop_name}'. "
                    f"Valid properties: {list(valid_properties.keys())}"
                )

        # Get current particle count
        N = self.particles.number_of_particles
        if N == 0:
            raise ValueError("Cannot set properties: particle system is empty")

        # Validate all arrays before making any changes (atomic operation)
        validated_properties = {}
        for prop_name, prop_value in properties.items():
            field_name = valid_properties[prop_name]
            validated_properties[field_name] = self._validate_particle_property(
                prop_name, prop_value, N
            )

        # All validation passed - now update properties atomically
        for field_name, prop_value in validated_properties.items():
            self.particles.set_field(field_name, prop_value)

        # Invalidate cache since particle data has changed
        self.particles._cached_step = -1

        # Print confirmation
        property_names = list(properties.keys())
        if len(property_names) == 1:
            Logging.info(f"Updated particle property: {property_names[0]}")
        else:
            Logging.info(
                f"Updated {len(property_names)} particle properties: {', '.join(property_names)}"
            )

    # STATE MANAGEMENT AND BACKUP/RESTORE

    def save_state(self, filename: str = "solution/solver_state") -> None:
        """
        Save complete solver state including configuration for restart.

        Unlike regular backups (which only save HDF5 + XDMF per timestep),
        this method saves the full configuration JSON file needed for
        restoring the solver state. Use this when you want to create a
        checkpoint that can be used with `continue_from_backup()`.

        Args:
              filename: Base filename (without extension). Files saved:
                       - {filename}.h5: Numerical state
                       - {filename}.xdmf: ParaView visualization
                       - {filename}_config.json: Solver configuration

        Example:
              >>> # Save checkpoint at specific time
              >>> solver.save_state('solution/checkpoint_t100')
              >>>
              >>> # Later, restore from checkpoint
              >>> solver = Solver.continue_from_backup('solution/checkpoint_t100')
        """
        import os

        # Ensure directory exists
        if backup_dir := os.path.dirname(filename):
            os.makedirs(backup_dir, exist_ok=True)

        # Save numerical data (HDF5) and XDMF
        self._refresh_backup_particle_fields()
        BackupSystem.backup_solver(self, filename, verbose=False)

        # Also save configuration JSON (required for continue_from_backup)
        config_file = f"{filename}_config.json"
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
        """
        Back up the solver state using the IO manager.

        Delegates to self.io.backup() which handles:
        - HDF5 state backup (for restart)
        - VTK export (for visualization)
        - CSV export (aerodynamic loads)
        - VLM export (if applicable)

        All output files are saved to 'solution/' subdirectory.
        """
        if not self.io.should_backup():
            return

        # Ensure particle attributes for visualization/restart are up-to-date.
        self._refresh_backup_particle_fields()

        self.io.backup()

    def _refresh_backup_particle_fields(self) -> None:
        """Refresh particle fields that are expected to be available in backups."""
        N = self.particles.number_of_particles
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
            self._update_velocity_gradients()

    @staticmethod
    def continue_from_backup(backup_file_name: str | None = None) -> "Solver | None":
        """
        Restore the solver state using the robust HDF5-based backup system.

        Args:
              backup_file_name: Path to backup files (without extensions).
                    Since backups are saved to solution/ by default, typically
                    pass "solution/backup_name" (e.g., "solution/simulation").

        Returns:
              solver: Restored solver instance with exact configuration and full precision

        Raises:
              FileNotFoundError: If backup files don't exist
              ValueError: If backup files are corrupted or invalid

        Example:
              >>> # Restore from most recent backup
              >>> solver = Solver.continue_from_backup("solution/rotor_simulation")
        """
        # Validate backup integrity before attempting restore
        if not BackupSystem.validate_backup(backup_file_name):
            raise ValueError(f"Backup validation failed for: {backup_file_name}")

        Logging.message(f"\n{'-' * 60}")
        Logging.info("Resuming simulation from robust backup:")
        Logging.message(f"       Base filename: {backup_file_name}")
        Logging.message(f"{'-' * 60}\n")

        # Restore solver with full precision and exact configuration
        restored_solver = BackupSystem.restore_solver(backup_file_name)

        # Reset energy history since we're starting from a checkpoint
        restored_solver.field_diagnostics.reset_energy_history()

        # Refresh flow integrals after restore
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

    # PARTICLE PHYSICS UPDATE METHODS

    def set_background_velocity(self, velocity: list[float] | np.ndarray) -> None:
        """
        Set the uniform background velocity field for all particles.

        This method allows dynamic modification of the background velocity during
        simulation. The new velocity will be applied in the next time step.

        Args:
              velocity: Velocity vector [ux, uy, uz] in m/s. Can be a list, tuple, or numpy array.

        Example:
              >>> # Set constant background velocity
              >>> solver.set_background_velocity([10.0, 0.0, 0.0])
              >>>
              >>> # Change velocity during simulation
              >>> for step in range(n_steps):
              ...     if step > 50:
              ...         solver.set_background_velocity([20.0, 0.0, 0.0])
              ...     solver.update_state()
        """
        dtype = np.float64 if self.precision == "f64" else np.float32
        velocity_arr = np.array(velocity, dtype=dtype)

        # Validate shape strictly
        if velocity_arr.shape != (3,):
            # Try to flatten if it's (1, 3) or similar, but be strict about 3 elements
            if velocity_arr.size == 3:
                velocity_arr = velocity_arr.flatten()
            else:
                raise ValueError(
                    f"Background velocity must be a 3D vector, got shape {velocity_arr.shape}"
                )

        # Update particle field directly (this is now the source of truth)
        self.config.background_velocity = [float(v) for v in velocity_arr]
        self.particles.set_background_velocity(velocity_arr)

    def set_velocity_override(self, fn) -> None:
        """Set (or clear) the per-stage advection velocity override.

        When *fn* is not None it is called at every RK stage immediately after
        the Biot–Savart evaluation:

            vel_used = fn(pos: np.ndarray (N,3), vel_bs: np.ndarray (N,3))
                       -> np.ndarray (N,3)

        Pass ``None`` to restore pure Biot–Savart transport.  The override
        applies to ADVECTION only; stretching/∇u uses Biot–Savart.
        """
        self.physics.velocity_override = fn

    def _update_velocities(self) -> None:
        """
        Update particle velocities using self-induced velocity computation.

        This method computes the velocity field at each particle location
        due to the influence of all other particles in the system.

        The direct-vs-treecode choice is owned by physics.velocity_self
        (configured once at startup), so there is no method branching here.
        """
        Logging.message(f"Updating particles' velocities, u ({self.physics.velocity_method.lower()})")
        self.physics.velocity_self(
            self.particles.position,
            self.particles.circulation,
            self.particles.radius,
            self.particles.velocity,
            self.particles.velocity_background,
            self.particles.number_of_particles,
        )

        # Add induced velocity from panels using DIRECT solver (more accurate)
        if self.panel_solver is not None:
            # Synchronize before reading particle velocity data that was
            # just written by velocity_self via an asynchronous Taichi kernel.
            ti.sync()
            self.panel_solver.compute_induced_velocity_direct(self.particles)

        # Add induced velocity from source particles (body blockage potential correction)
        if self.num_sources > 0:
            # Calculate induction from sources onto particles
            self.physics.kernels["compute_target_source_velocity_kernel"](
                self.particles.position,
                self.source_positions,
                self.source_strengths,
                self.source_radii,
                self.particles.velocity,
                self.particles.number_of_particles,
                self.num_sources,
            )

    def _update_velocity_gradients(self, announce: bool = True) -> None:
        """
        Update velocity gradient tensors for all particles.

        This computes the velocity gradient tensor ∇u at each particle
        location, which is essential for turbulence modeling and vortex
        stretching calculations. The strain rate tensor Sij is computed
        automatically within compute_velocity_gradient_tensor.

        If treecode is enabled (via VelocityConfig), uses Barnes-Hut
        O(N log N) algorithm. Otherwise uses direct O(N²) summation.

        Skipped for potential flow models.
        """
        use_treecode = bool(self.config.velocity and self.config.velocity.method == "TREECODE")
        theta = self.config.velocity.theta if self.config.velocity else 0.5

        if use_treecode:
            if announce:
                Logging.message(f"Updating velocity gradient tensor, ∇u (treecode, θ={theta})")
            self.physics.compute_velocity_gradients_hierarchical(self.particles, theta=theta)
        else:
            if announce:
                Logging.message("Updating velocity gradient tensor, ∇u")
            self.physics.compute_velocity_gradients(self.particles)

    def _update_velocity_and_gradients(self, announce: bool = True) -> None:
        """Fused u + ∇u at t_n in a single tree build + traversal.

        Writes ``particles.velocity`` (= v(x_n), reused as the advection k1) plus
        ``velocity_gradient`` / ``strain_rate``.  Used in place of a separate
        velocity pass and gradient pass when both are needed at the same
        configuration (the common DNS/LES advection step)."""
        use_treecode = bool(self.config.velocity and self.config.velocity.method == "TREECODE")
        theta = self.config.velocity.theta if self.config.velocity else 0.5
        if use_treecode:
            if announce:
                Logging.message(f"Updating fused u + ∇u (treecode, θ={theta})")
            self.physics.compute_velocity_and_gradient_hierarchical(self.particles, theta=theta)
        else:
            if announce:
                Logging.message("Updating fused u + ∇u (direct)")
            self.physics.compute_velocity_and_gradient(self.particles)

    def _update_LES_state(self, dt: float | None = None) -> None:
        """
        Update turbulence state for DNS/LES models.

        This method computes essential turbulence properties such as strain rate tensors,
        and turbulent viscosities required for the simulation. Statistics are always
        computed for logging purposes.
        """
        if self.flow_model == "LES":
            # Compute enstrophy for enstrophy-based dissipation (used by SFS model)
            self.physics.compute_enstrophy(self.particles)
            self.LES.compute(
                self.particles,
                dt=self.time_step_size if dt is None else dt,
            )

    def _update_strength(self, dt: float | None = None, announce: bool = True) -> None:
        """
        Update particle vortex strengths via diffusion and stretching.

        Order of operations:
          1. Viscous diffusion
          2. Vortex stretching + strength-relaxation projection
        """
        if self.flow_model == "POTENTIAL":
            return

        effective_mode = self._effective_stretching_mode()
        mode_eq = {
            "CLASSICAL": "(ω·∇)u",
            "TRANSPOSED": "(ω·∇)u",
            "MIXED": "½((ω·∇)u + (∇u)ᵀ·ω)",
            "GRADU": "(∇u)ᵀ·ω",
            "RVPM": "(∇u)ᵀ·ω − c_r(ω̂·(∇u)ᵀω̂)ω (rVPM)",
        }.get(effective_mode, f"({effective_mode})")
        if announce:
            Logging.message(f"Updating strengths via {mode_eq}")

        dt = self.time_step_size if dt is None else dt
        self._apply_viscous_diffusion(dt)
        self._apply_stretching_with_relaxation(dt)

    def _effective_stretching_mode(self) -> str:
        """Mode actually used by the stretching step.

        Stabilizers must not change the user-selected stretching formulation.
        """
        return self.stretching_mode

    def _rvpm_params(self) -> dict:
        """rVPM (f, g) parameters from the stretching config."""
        sc = self.config.stretching
        return {
            "rvpm_f": getattr(sc, "rvpm_f", 0.0),
            "rvpm_g": getattr(sc, "rvpm_g", 0.2),
        }

    def _apply_stretching_with_relaxation(self, dt: float) -> None:
        """Vortex stretching followed by the strength-relaxation projection, once per dt."""
        if self.stretching_enabled:
            # Advisory only: warn once if dt exceeds the strain-set stability limit
            # dt_rec = C/σ_max (C = 0.2 stretching-CFL target), the usual source of an
            # explicit-stretching blow-up.  An explicit solver integrates exactly the
            # adopted dt — this never sub-divides or overrides it.
            if not getattr(self, "_stretch_dt_warned", False):
                from ..stabilization.strength_relaxation import max_seff_from_particles

                sigma_max = max_seff_from_particles(self.particles)
                if sigma_max > 0.0:
                    dt_rec = 0.2 / sigma_max
                    if dt > dt_rec:
                        Logging.stretching_dt_warning(dt, dt_rec, sigma_max)
                        self._stretch_dt_warned = True
            self.physics.vortex_stretching(
                self.particles,
                dt=dt,
                scheme=self.stretching_scheme,
                mode=self.stretching_mode,
                **self._rvpm_params(),
            )
            ti.sync()
        if self._strength_relaxation is not None:
            self._strength_relaxation.apply(self.particles, dt)
            ti.sync()

    def _apply_viscous_diffusion(self, dt: float) -> None:
        """Dispatch viscous diffusion by configured scheme."""
        if self.viscous_scheme == "NONE":
            return

        if self.viscous_scheme == "CS":
            Logging.message("Performing viscous diffusion via Core Spreading.")
            self.physics.core_spreading_diffusion(self.particles, dt=dt)
        elif self.viscous_scheme == "RWM":
            Logging.message("Performing viscous diffusion via Random Walk Method.")
            self.physics.random_walk_method_diffusion(self.particles, dt=dt)
        elif self.viscous_scheme in ("DVH", "GBD"):
            # Both schemes fire exactly once per step: DVH applies the fixed
            # Δt_d heat-kernel increment (dt is pinned to Δt_d), GBD scales
            # with dt directly (α = nu·dt/h²).
            new_p = self._apply_grid_diffusion(self._viscous_config, dt)
            if new_p is not None:
                M = len(new_p["position"])
                self.remove_particles(remove_all=True)
                self.add_vortex_particles(
                    position=new_p["position"],
                    velocity=new_p.get("velocity", np.zeros((M, 3), dtype=self.np_dtype)),
                    circulation=new_p["circulation"],
                    radius=new_p["radius"],
                    volume=new_p["volume"],
                    viscosity=new_p.get("viscosity"),
                    viscosity_turbulent=new_p.get("viscosity_turbulent"),
                    zone_id=new_p.get("zone_id", np.zeros(M, dtype=np.int32)),
                    group_id=new_p.get("group_id", np.zeros(M, dtype=np.int32)),
                )
                self._update_velocities()
        ti.sync()

    def _apply_grid_diffusion(self, vc, dt: float):
        """Run DVH or GBD grid-based diffusion; return new particle dict."""
        if self.viscous_scheme == "DVH":
            # In LES mode the per-particle effective viscosity (nu + nu_t) sets
            # each particle's heat-kernel width — otherwise the SGS model would
            # be computed but never act in DVH runs.
            nu_eff = None
            if self.flow_model == "LES":
                N = self.particles.number_of_particles
                if N > 0:
                    nu_eff = self.particles.viscosity_effective.to_numpy()[:N]
            Logging.message(
                f"\tPerforming DVH particle regeneration "
                f"(h={vc.dvh_grid_spacing:.3e}, nu={vc.viscosity:.3e}, "
                f"threshold={vc.dvh_threshold:.2e}"
                + (
                    f", LES nu_eff/nu max={float(nu_eff.max()) / vc.viscosity:.2f}"
                    if nu_eff is not None
                    else ""
                )
                + ")."
            )
            return self.physics.grid_based_diffusion(
                self.particles,
                dt=dt,
                h=vc.dvh_grid_spacing,
                nu=vc.viscosity,
                domain_padding=vc.dvh_domain_padding,
                regen_threshold=vc.dvh_threshold,
                regen_threshold_mode=vc.dvh_threshold_mode,
                rd_ratio=vc.dvh_rd_ratio,
                nu_eff=nu_eff,
                max_nodes=getattr(vc, "dvh_max_nodes", None),
            )
        else:  # GBD
            # In LES mode the per-particle effective viscosity (nu + nu_t) sets
            # the per-node Laplacian coefficient — otherwise the SGS model
            # would be computed but never act in GBD runs (Bug A).  Mirrors
            # the DVH branch above.
            nu_eff = None
            if self.flow_model == "LES":
                N = self.particles.number_of_particles
                if N > 0:
                    nu_eff = self.particles.viscosity_effective.to_numpy()[:N]
            Logging.message(
                f"\tPerforming GBD diffusion"
                f"(h={vc.gbd_grid_spacing:.3e}, nu={vc.viscosity:.3e}, "
                f"threshold={vc.gbd_threshold:.2e}"
                + (
                    f", LES nu_eff/nu max={float(nu_eff.max()) / vc.viscosity:.2f}"
                    if nu_eff is not None
                    else ""
                )
                + ")."
            )
            return self.physics.gbd_diffusion(
                self.particles,
                dt=dt,
                h=vc.gbd_grid_spacing,
                nu=vc.viscosity,
                domain_padding=vc.gbd_domain_padding,
                regen_threshold=vc.gbd_threshold,
                regen_threshold_mode=vc.gbd_threshold_mode,
                nu_eff=nu_eff,
                max_nodes=getattr(vc, "gbd_max_nodes", None),
            )

    def _update_adaptation(self) -> None:
        """
        Perform particle adaptation based on configuration.

        Supports three independent adaptation mechanisms:
        1. max_core_radius - Particle splitting (Winckelmans 1993) when radius exceeds threshold
        2. remove_particles_by_bounds - Remove particles outside spatial bounds
        3. conservative_remeshing - Periodic remeshing with delta correction
        4. weak_removal - Remove particles with negligible strength

        Each feature is only executed if configured (not None).
        """
        if self.flow_model == "POTENTIAL":
            return

        adaptation_performed = False
        cfg = self.stabilization_config

        # 1. Particle Splitting (1-to-2, transverse to the vorticity direction)
        _split_diag_data = None
        if self._splitter is not None and cfg.max_core_radius is not None:
            max_radius = cfg.max_core_radius

            if self._splitter.needs_splitting(self.particles, max_radius):
                # Compute fresh ω at current positions/strengths before split
                self.physics.compute_vorticities(self.particles)
                N_pre = len(self.particles)
                _pos_pre = self.particles.position.to_numpy()[:N_pre].copy()
                _str_pre = self.particles.circulation.to_numpy()[:N_pre].copy()
                _rad_pre = self.particles.radius.to_numpy()[:N_pre].copy()
                _vort_pre = self.particles.vorticity.to_numpy()[:N_pre].copy()
                _mask_split = _rad_pre > max_radius

                stats = self._splitter.split(self.particles, max_radius)
                Logging.message(
                    f"(Stabilization) Splitting: {stats.particles_split} particles -> {stats.particles_created} children "
                    f"({stats.particles_total_after} total)"
                )
                adaptation_performed = True
                if stats.particles_split > 0:
                    _split_diag_data = (_pos_pre, _str_pre, _rad_pre, _vort_pre, _mask_split, stats)

        # 2. Remove particles outside bounds
        if cfg.remove_particles_by_bounds is not None:
            self.remove_particles_by_bounds(cfg.remove_particles_by_bounds, invert_selection=True)
            adaptation_performed = True

        # 3. Conservative Remeshing (periodic remesh + delta correction)
        if cfg.remeshing_frequency is not None and self.time_step % cfg.remeshing_frequency == 0:
            stats = self._remesher.remesh(
                solver=self,
                spacing=cfg.remeshing_spacing,
                bounds=cfg.remeshing_bounds,
                rel_threshold=cfg.remeshing_relative_threshold,
                abs_threshold=cfg.remeshing_absolute_threshold,
                conserve_impulse=cfg.remeshing_conserve_impulse,
                delta_correction=cfg.remeshing_delta_correction,
                impulse_constraint=cfg.remeshing_impulse_constraint,
                particle_radius=cfg.remeshing_radius,
                project_solenoidal=cfg.remeshing_project_solenoidal,
                projection_padding=cfg.remeshing_projection_padding,
            )
            # Recompute flow integrals from the post-remesh particle state so the
            # logged CSV row is consistent with the particle distribution used next step.
            self._update_all_flow_integrals()
            adaptation_performed = True

        # 4. Remove weak particles (only if not already done by grid reinit or remesh)
        elif cfg.weak_threshold_percent is not None and cfg.max_core_radius is None:
            self.remove_weak_particles(percent=cfg.weak_threshold_percent, per_group=cfg.per_group)
            adaptation_performed = True

        # Update vorticity field if any adaptation was performed
        if adaptation_performed:
            self.physics.compute_vorticities(self.particles)
            if _split_diag_data is not None:
                self._diagnose_split(*_split_diag_data)

    def _diagnose_split(self, pos_pre, str_pre, rad_pre, vort_pre, mask_split, stats):
        """Delegate to diagnostics.split_diagnostics.diagnose_split."""
        from ..diagnostics.split_diagnostics import diagnose_split

        diagnose_split(
            self.physics, self.particles, pos_pre, str_pre, rad_pre, vort_pre, mask_split, stats
        )

    def _update_positions(self, dt: float | None = None, precomputed_k1: bool = False) -> None:
        """
        Update particle positions through advection.

        Honors the configured advection scheme (EULER/RK2/RK3/RK4) in a single
        step over the macro time-step.  Every velocity evaluation inside the
        integrator uses the configured velocity method (direct or treecode) via
        physics.velocity_self — no method logic is duplicated here.

        ``precomputed_k1=True`` means ``particles.velocity`` already holds v(x_n)
        (a fused velocity+gradient pass ran at t_n), so the integrator's first
        stage reuses it rather than recomputing it.
        """
        if self.advection_scheme == "NONE":
            return
        self.physics.update_positions(
            self.particles,
            self.time_step_size if dt is None else dt,
            scheme=self.advection_scheme,
            precomputed_k1=precomputed_k1,
        )

    # PANEL-VPM COUPLING

    def _advance_panel(self):
        """
        Robust hybrid panel-VPM coupling algorithm.

        Delegates all panel-related operations (geometry update, boundary conditions,
        solving, and shedding) to the PanelSolver class.
        """
        new_particles = self.panel_solver.advance(
            particles=self.particles,
            physics=self.physics,
            V_inf=self.background_velocity,
            dt=self.time_step_size,
            time=self.flow_time,
            step=self.time_step,
            logging_frequency=self.logging_frequency,
            # Pass density from config if available, else standard air
            density=getattr(self.config, "density", 1.0),
        )
        if new_particles is not None:
            # Map panel solver output keys to add_vortex_particles signature
            n = len(new_particles["points"])
            if n > 0:
                # Get viscosity from config
                visc_cfg = getattr(self.config, "viscous", None)
                nu = getattr(visc_cfg, "viscosity", None) if visc_cfg is not None else None
                if nu is None or nu <= 0:
                    nu = 1e-2  # default viscosity
                viscosity = np.full(n, nu, dtype=self.np_dtype)

                # Convert all arrays to the solver precision to avoid mismatch
                pos = new_particles["points"].astype(self.np_dtype)
                strength = new_particles["strengths"].astype(self.np_dtype)
                rad = new_particles["radii"].astype(self.np_dtype)
                vol = new_particles["volumes"].astype(self.np_dtype)

                self.add_vortex_particles(
                    position=pos,
                    velocity=np.zeros((n, 3), dtype=self.np_dtype),  # shed particles start at rest
                    circulation=strength,
                    radius=rad,
                    volume=vol,
                    viscosity=viscosity,
                )

    # VLM-VPM COUPLING

    # Pattern: similar to _advance_panel above, delegate to VLM solver's advance_coupled method
    def _advance_vlm(self, dt: float) -> None:
        """
        Update VLM-VPM coupling for one time step.

        Delegates all VLM-related operations (velocity computation, solving,
        and wake shedding) to the VLMSolver class.
        """
        if self.vlm_solver is None:
            return

        # Fix: Return taichi arrays directly for better GPU performance
        wake_particles = self.vlm_solver.advance_coupled(
            particles=self.particles,
            physics=self.physics,
            config=self.config,
            dt=dt,
            time_step=self.time_step,
            time=self.flow_time,
        )

        # Fix: Create (or verify if exits) _add_vortex_particles_taichi(): method that accepts taichi arrays directly
        if wake_particles is not None:
            self.add_vortex_particles(**wake_particles)

    # PARTICLES CONTROL

    def remove_particles_by_bounds(self, bounds: list, invert_selection: bool = False) -> int:
        """
        Remove particles based on their position relative to a bounding box.

        Args:
              bounds: [xmin, xmax, ymin, ymax, zmin, zmax] defining the reference box.
                     Use -inf/inf for unbounded dimensions.
              invert_selection: If False (default), remove particles INSIDE the box.
                          If True, remove particles OUTSIDE the box (keep those inside).

        Returns:
              Number of particles removed.

        Examples:
              >>> # Remove particles in far wake (x > 10)
              >>> solver.remove_particles_by_bounds([10, np.inf, -np.inf, np.inf, -np.inf, np.inf])

              >>> # Keep only particles inside VPM domain (remove those outside)
              >>> solver.remove_particles_by_bounds(vpm_domain_bounds, invert_selection=True)
        """
        if len(bounds) != 6:
            raise ValueError("bounds must be [xmin, xmax, ymin, ymax, zmin, zmax]")

        n_particles = self.particles.number_of_particles
        if n_particles == 0:
            return 0

        xmin, xmax, ymin, ymax, zmin, zmax = bounds

        # Use GPU-based removal with invert_selection flag
        n_removed = self.particles.remove_particles_by_bounds(
            bounds, invert_selection=invert_selection
        )

        if n_removed > 0:
            action = "outside" if invert_selection else "inside"
            Logging.info(
                f"Removed {n_removed} particles {action} "
                f"box [{xmin:.2f}, {xmax:.2f}] × [{ymin:.2f}, {ymax:.2f}] × [{zmin:.2f}, {zmax:.2f}]"
            )

        return n_removed

    def remove_weak_particles(self, percent: float, per_group: bool = True) -> None:
        """
        Remove the weakest particles based on their strengths.

        This method is useful for cleaning up particle fields after initialization,
        removing particles with negligible strength to improve computational efficiency.

        Args:
              percent: Percentage of weakest particles to remove (0-100).
                       This is relative to the maximum strength in each group (if per_group=True)
                       or globally (if per_group=False).
              per_group: If True (default), apply threshold independently to each group.
                        This preserves the relative structure of each vortex system.
                        If False, use global threshold which may cause uneven removal
                        if groups have different strength scales.

        Example:
              >>> # Remove weakest 1% from each group independently (recommended)
              >>> solver.remove_weak_particles(percent=1.0, per_group=True)

              >>> # Remove weakest 1% globally (may remove more from weaker groups)
              >>> solver.remove_weak_particles(percent=1.0, per_group=False)

        Note:
              When working with multiple vortex structures (e.g., two vortex rings),
              using per_group=True ensures each structure loses the same percentage
              of particles, preserving their relative strengths and distributions.
        """
        if percent < 0 or percent > 100:
            raise ValueError("Percent must be between 0 and 100")

        # Early return if no particles (prevents Taichi dimension=0 error)
        if len(self.particles) == 0:
            return 0

        # Store count before removal
        particles_before = len(self.particles)

        # Remove weak particles
        self.particles._remove_weak_particles(percent=percent, per_group=per_group)

        # Only compute vorticities if particles remain after removal
        if len(self.particles) > 0:
            self.physics.compute_vorticities(self.particles)  # Force particle field resizing

        # Get count after removal
        particles_after = len(self.particles)
        particles_removed = particles_before - particles_after

        Logging.particle_cleanup(percent, particles_before, particles_removed, particles_after)

        return particles_removed
