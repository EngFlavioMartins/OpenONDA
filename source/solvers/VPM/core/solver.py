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

from source.solvers.VPM.particles import create_physics
from source.solvers.VPM.particles.container import Particles
from source.solvers.VPM.turbulence.turbulence import ParticlesLES

from ..boundary_elements.vlm.solver.diagnostics import VLMDiagnostics
from ..boundary_elements.vlm.solver.forces import VLMForceEvaluator
from ..boundary_elements.vlm.solver.loading_distribution import VLMLoadingDistribution
from ..config.backend import initialize_taichi_backend, reset_taichi_backend
from ..config.constants import MAX_PARTICLES, MAX_SOURCES
from ..config.types import SetFlowModel, StabilizationConfig, VPMSetup
from ..diagnostics.resolution import discretization_health
from ..io.backup import BackupSystem
from ..io.logging import Logging, print_openonda_header
from ..io.runtime_profiler import RuntimeProfiler
from ..io.sampler import SamplerExecutor
from ..io.solver_io import SolverIO
from ..physics.evaluation import ParticleFieldEvaluation
from ..stabilization import StabilizationManager
from ..utils.field_samplers import resolve_samples_dir


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
        self.physics = create_physics(
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
        self.stabilization = StabilizationManager(self)
        active = self.stabilization.active_mechanisms()
        if active:
            Logging.message("Stabilization: " + ", ".join(active))
        self._init_optional_solvers(final_config)
        # Synchronize asynchronous kernels at profiler boundaries.
        self.profiler = RuntimeProfiler(sync=ti.sync)
        self.simulation_time = 0.0

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

        The inviscid update advances particle motion and, when enabled, vortex
        stretching. Viscous diffusion is then applied by operator splitting. Core
        spreading uses symmetric Strang splitting in the coupled integrator.
        """

        self.stabilization.capture_reference_state()

        self._advance_time_step()

        self.particles.time_step = self.time_step
        self._debug_validate_particle_geometry("step entry")

        diagnostics_due = (
            self.logging_frequency > 0 and self.time_step % self.logging_frequency == 0
        )

        with self.profiler.step():
            if self.vlm_solver is not None:
                with self.profiler.section("VLM coupling"):
                    self._advance_vlm(self.time_step_size)

            if self.panel_solver is not None:
                with self.profiler.section("Panel coupling"):
                    self._advance_panel()

            _adv = (self.config.advection.scheme if self.config.advection else "RK3").upper()
            _gradients_required = (
                self.stretching_enabled
                or self.flow_model == "LES"
                or self.time_integration == "COUPLED"
                or self.stabilization_config.stretching_viscosity_coefficient > 0.0
                or (
                    self.stabilization_config.pedrizzetti_relaxation_enabled
                    and self.flow_model != "POTENTIAL"
                )
            )
            _defer_stationary_velocity = (
                _adv == "NONE"
                and not _gradients_required
                and self.vlm_solver is None
                and self.panel_solver is None
                and self.num_sources == 0
                and getattr(self.physics, "velocity_override", None) is None
            )
            _fuse_vel_grad = (
                self.flow_model != "POTENTIAL"
                and _adv != "NONE"
                and _gradients_required
                and self.num_sources == 0
                and self.panel_solver is None
                and getattr(self.physics, "velocity_override", None) is None
            )
            if _fuse_vel_grad:
                with self.profiler.section("Velocity + gradients"):
                    self._update_velocity_and_gradients()
                velocity_k1_ready = True
            else:
                velocity_k1_ready = False
                if not _defer_stationary_velocity and (
                    _adv == "NONE"
                    or not _gradients_required
                    or self.num_sources > 0
                    or self.panel_solver is not None
                ):
                    with self.profiler.section("Velocity"):
                        self._update_velocities()
                    velocity_k1_ready = True
                if _gradients_required:
                    with self.profiler.section("Velocity gradients"):
                        self._update_velocity_gradients()

            with self.profiler.section("LES update"):
                self._update_LES_state()
                self.stabilization.update_residual_viscosity()

            # Relax against the same t_n gradient used by the strength update.
            with self.profiler.section("Pedrizzetti relaxation"):
                self.stabilization.apply_relaxation()

            coupled_update = self.time_integration == "COUPLED" and self.flow_model != "POTENTIAL"

            # Inviscid evolution, followed by viscous diffusion.
            if not coupled_update:
                with self.profiler.section("Advection"):
                    self._update_positions(precomputed_k1=velocity_k1_ready)

            if self.flow_model != "POTENTIAL":
                if self.stretching_enabled:
                    self._announce_strength_update()

                if coupled_update:
                    with self.profiler.section("Coupled advection + stretching"):
                        self._apply_coupled_update_with_subcycling(
                            self.time_step_size, precomputed_velocity_k1=_fuse_vel_grad
                        )
                else:
                    with self.profiler.section("Stretching"):
                        self._apply_stretching(self.time_step_size)
                    self._debug_validate_particle_geometry("stretching")
                    with self.profiler.section("Viscous diffusion"):
                        self._apply_viscous_diffusion(self.time_step_size)
                    self._debug_validate_particle_geometry("viscous diffusion")

                with self.profiler.section("Filament refinement"):
                    self.stabilization.apply_filament_refinement()
                with self.profiler.section("Divergence relaxation"):
                    self.stabilization.apply_divergence_relaxation()
                with self.profiler.section("Conservative regularization"):
                    self.stabilization.apply_regularization()

            if diagnostics_due:
                with self.profiler.section("Flow integrals"):
                    if _defer_stationary_velocity:
                        # Evaluate deferred velocity on the end-of-step state.
                        self._update_velocities()
                    elif (
                        self.flow_model == "LES"
                        or self.stabilization_config.stretching_viscosity_coefficient > 0.0
                    ):
                        # Keep ν_eff consistent with the end-of-step diagnostics.
                        self._update_velocity_and_gradients(announce=False)
                        self._update_LES_state()
                        self.stabilization.update_residual_viscosity()
                    self._update_all_flow_integrals()
            elif self.vlm_solver is not None:
                with self.profiler.section("VLM diagnostics"):
                    self._record_vlm_diagnostics()

            with self.profiler.section("Particle retention"):
                self.stabilization.apply_retention()
            self._debug_validate_particle_geometry("particle retention")

            with self.profiler.section("Backup / IO"):
                self._backup_solution()

        self.profiler.report_step()
        self.simulation_time = self.profiler.wall_time

        if self.timing_frequency > 0 and self.time_step % self.timing_frequency == 0:
            self.profiler.set_particle_count(self.particles.number_of_particles)
            self.profiler.report()

        if self.logging_frequency > 0 and self.time_step % self.logging_frequency == 0:
            self.log_diagnostics()

    def _debug_validate_particle_geometry(self, stage: str) -> None:
        """Validate active particle radii and volumes when stage tracing is enabled."""
        if os.environ.get("VPM_VALIDATE_STAGES", "0") != "1":
            return
        n = self.particles.number_of_particles
        if n == 0:
            return
        radii = self.particles.radius_cpu(use_cache=False)
        volumes = self.particles.volume_cpu(use_cache=False)
        invalid_radii = ~np.isfinite(radii) | (radii <= 0.0)
        invalid_volumes = ~np.isfinite(volumes) | (volumes <= 0.0)
        n_bad_radii = int(np.count_nonzero(invalid_radii))
        n_bad_volumes = int(np.count_nonzero(invalid_volumes))
        Logging.message(
            f"[Integrity:{stage}] N={n} radius=[{np.nanmin(radii):.6e}, "
            f"{np.nanmax(radii):.6e}] bad={n_bad_radii}; "
            f"volume=[{np.nanmin(volumes):.6e}, {np.nanmax(volumes):.6e}] "
            f"bad={n_bad_volumes}"
        )
        if n_bad_radii or n_bad_volumes:
            bad_radius_index = int(np.flatnonzero(invalid_radii)[0]) if n_bad_radii else -1
            bad_volume_index = int(np.flatnonzero(invalid_volumes)[0]) if n_bad_volumes else -1
            raise RuntimeError(
                f"Invalid VPM particle geometry after {stage}: "
                f"radius_bad={n_bad_radii} first={bad_radius_index}, "
                f"volume_bad={n_bad_volumes} first={bad_volume_index}"
            )

    def _advance_time_step(self) -> None:
        """Advance the step counter and physical time."""

        self.time_step += 1

        self.flow_time = round(self.flow_time + self.time_step_size, 12)

        Logging.message(
            f"\nTime-step: {self.time_step:d}   Flow time: {self.flow_time:0.2E} s",
            flush=True,
        )

    def record_diagnostics(self, *, refresh_fields: bool = False) -> None:
        """Evaluate and log diagnostics for the current particle state.

        Set ``refresh_fields=True`` when velocity, gradients, or LES viscosity are
        stale for the current state.
        """
        if refresh_fields:
            self._update_velocity_and_gradients()
            self._update_LES_state()
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
        """Append one row of flow integrals to ``<backup_directory>/samples/flow_integrals.csv``."""
        import pandas as pd

        samples_dir = resolve_samples_dir(
            self.backup_directory,
            getattr(self.config, "sample_subdirectory", None),
        )
        samples_dir.mkdir(parents=True, exist_ok=True)
        csv_path = samples_dir / "flow_integrals.csv"

        impulse = self._flow_integrals.get("linear_impulse", np.zeros(3))
        ang_impulse = self._flow_integrals.get("angular_impulse", np.zeros(3))
        strength = self._flow_integrals.get("strength", np.zeros(3))
        particle_strength = self.particles_circulation
        turbulent_viscosity = self.particles.viscosity_turbulent_cpu()
        effective_viscosity = self.particles.viscosity_effective_cpu()

        row = {
            "time": self.flow_time,
            "step": self.time_step,
            "kinetic_energy": self.total_kinetic_energy,
            "enstrophy": self.total_enstrophy,
            "enstrophy_test": self._flow_integrals.get("enstrophy_test", 0.0),
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
            "max_gamma": float(np.linalg.norm(particle_strength, axis=1).max(initial=0.0)),
            "turbulent_viscosity_mean": float(turbulent_viscosity.mean())
            if len(turbulent_viscosity)
            else 0.0,
            "turbulent_viscosity_max": float(turbulent_viscosity.max(initial=0.0)),
            "effective_viscosity_mean": float(effective_viscosity.mean())
            if len(effective_viscosity)
            else 0.0,
            "effective_viscosity_max": float(effective_viscosity.max(initial=0.0)),
            "invariant_projection_correction_ratio": float(
                self.physics.rate_projection_max_correction_ratio
            ),
        }
        row.update(self._discretization_health)
        row.update(self.stabilization.diagnostics)

        df = pd.DataFrame([row])
        if not csv_path.exists():
            df.to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path, mode="a", header=False, index=False)

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
    ) -> dict | tuple[dict, np.ndarray]:
        """Evaluate pressure-gradient terms at arbitrary target points.

        The result contains the total pressure gradient and its convective, viscous,
        and temporal contributions. ``temporal_method='eulerian'`` requires
        ``velocity_previous`` and ``dt`` when the temporal term is enabled.
        """
        if nu is None:
            nu = (
                float(np.mean(self.particles_viscosities))
                if self.particles.number_of_particles > 0
                else 1e-5
            )
        if treecode_theta is not None:
            if temporal_method != "eulerian":
                raise ValueError("Treecode pressure gradients require temporal_method='eulerian'")
            points = np.asarray(grid_positions, dtype=np.float64).reshape(-1, 3)
            count = len(points)
            targets = points
            if include_viscous:
                if h is None:
                    h = (
                        float(np.mean(self.particles.radius_cpu()))
                        if self.particles.number_of_particles > 0
                        else 1.0
                    )
                offsets = np.eye(3, dtype=np.float64) * float(h)
                targets = np.concatenate(
                    [
                        points,
                        *(points + offsets[j] for j in range(3)),
                        *(points - offsets[j] for j in range(3)),
                    ]
                )
            velocity_samples = self.physics.compute_target_velocities_hierarchical(
                self.particles,
                targets,
                theta=float(treecode_theta),
                include_freestream=include_freestream,
            ).astype(np.float64)
            if self.particles.number_of_particles == 0 and include_freestream:
                velocity_samples[:] = self.background_velocity
            velocity = velocity_samples[:count]
            gradient = self.physics.compute_target_velocity_gradients_hierarchical(
                self.particles, points, theta=float(treecode_theta)
            ).reshape(count, 3, 3)
            body_fn = getattr(
                self,
                "_pressure_body_induced_fn",
                self._body_induced_fn,
            )
            if body_fn is not None:
                velocity_samples += np.asarray(
                    body_fn(targets), dtype=velocity_samples.dtype
                ).reshape(velocity_samples.shape)
                velocity = velocity_samples[:count]
                gradient_h = (
                    float(h)
                    if h is not None
                    else (
                        float(np.mean(self.particles.radius_cpu()))
                        if self.particles.number_of_particles > 0
                        else 0.05
                    )
                )
                for axis in range(3):
                    offset = np.zeros(3, dtype=np.float64)
                    offset[axis] = gradient_h
                    plus = np.asarray(body_fn(points + offset), dtype=np.float64)
                    minus = np.asarray(body_fn(points - offset), dtype=np.float64)
                    gradient[:, :, axis] += (plus - minus) / (2.0 * gradient_h)
            advective = np.einsum("mb,mab->ma", velocity, gradient)
            temporal = np.zeros_like(velocity)
            if include_temporal:
                if velocity_previous is None or dt is None:
                    raise ValueError("Treecode pressure gradients require velocity_previous and dt")
                temporal = (velocity - velocity_previous) / float(dt)
            viscous = np.zeros_like(velocity)
            if include_viscous and nu > 0.0:
                plus = velocity_samples[count : 4 * count].reshape(3, count, 3)
                minus = velocity_samples[4 * count :].reshape(3, count, 3)
                viscous = (
                    float(nu)
                    * np.sum(plus + minus - 2.0 * velocity[None, :, :], axis=0)
                    / float(h) ** 2
                )
            result = {
                "grad_p": density * (-temporal - advective + viscous),
                "convective": -density * advective,
                "viscous": density * viscous,
                "temporal": -density * temporal,
            }
            return (result, velocity) if return_velocity else result

        from source.solvers.VPM.physics.pressure import PressurePhysics

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

        reference_strengths = getattr(self, "_filament_reference_strengths", None)
        reference_lengths = getattr(self, "_filament_reference_lengths", None)
        if reference_strengths is not None and reference_lengths is not None:
            if remove_all:
                self._filament_reference_strengths = np.empty(0, dtype=np.float64)
                self._filament_reference_lengths = np.empty(0, dtype=np.float64)
            elif particle_indices is not None and len(particle_indices) > 0:
                keep = np.ones(len(reference_strengths), dtype=bool)
                keep[np.asarray(particle_indices, dtype=np.int64)] = False
                self._filament_reference_strengths = reference_strengths[keep]
                self._filament_reference_lengths = reference_lengths[keep]

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
        if hasattr(self, "_filament_reference_strengths") and not getattr(
            self,
            "_loading_numerical_state",
            False,
        ):
            added_strength = np.linalg.norm(np.asarray(circulation, dtype=np.float64), axis=1)
            floor = max(
                float(added_strength.max(initial=0.0)) * 1e-12,
                np.finfo(np.float64).tiny,
            )
            if len(self._filament_reference_strengths) != start:
                raise RuntimeError(
                    "filament-refinement lineage state did not match the cloud before insertion"
                )
            self._filament_reference_strengths = np.concatenate(
                (self._filament_reference_strengths, np.maximum(added_strength, floor))
            )
            self._filament_reference_lengths = np.concatenate(
                (
                    self._filament_reference_lengths,
                    np.cbrt(np.asarray(volume, dtype=np.float64)),
                )
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
        if hasattr(self, "_filament_reference_strengths"):
            magnitude = np.linalg.norm(np.asarray(circulation, dtype=np.float64), axis=1)
            floor = max(
                float(magnitude.max(initial=0.0)) * 1e-12,
                np.finfo(np.float64).tiny,
            )
            self._filament_reference_strengths = np.maximum(magnitude, floor)
            self._filament_reference_lengths = np.cbrt(np.asarray(volume, dtype=np.float64))

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
            self._update_velocity_gradients()

    @staticmethod
    def continue_from_backup(backup_file_name: str | None = None) -> "Solver | None":
        """Restore a solver from an HDF5 backup and its saved configuration."""
        if not BackupSystem.validate_backup(backup_file_name):
            raise ValueError(f"Backup validation failed for: {backup_file_name}")

        Logging.message(f"\n{'-' * 60}")
        Logging.info("Resuming simulation from backup:")
        Logging.message(f"       Base filename: {backup_file_name}")
        Logging.message(f"{'-' * 60}\n")

        restored_solver = BackupSystem.restore_solver(backup_file_name)

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

    def _update_velocities(self) -> None:
        """Evaluate self-induced particle velocity and optional body/source contributions."""
        Logging.message(
            f"Updating particles' velocities, u ({self.physics.velocity_method.lower()})"
        )
        self.physics.velocity_self(
            self.particles.position,
            self.particles.circulation,
            self.particles.radius,
            self.particles.velocity,
            self.particles.velocity_background,
            self.particles.number_of_particles,
        )

        if (
            self.panel_solver is not None
            and getattr(self.panel_solver, "coupling_scope", "full") == "full"
        ):
            # The panel solver reads velocity written by an asynchronous Taichi kernel.
            ti.sync()
            self.panel_solver.compute_induced_velocity_direct(self.particles)

        if self.num_sources > 0:
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
        """Evaluate particle velocity gradients with the configured direct or tree method."""
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
        """Evaluate particle velocity and ``∇u`` in one direct pass or tree traversal."""
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
        """Update LES viscosity from the current strain-rate field."""
        if self.flow_model == "LES":
            self.LES.compute(
                self.particles,
                dt=self.time_step_size if dt is None else dt,
            )
            if self.axisymmetric_axis >= 0:
                self._validate_axisymmetric_orbits()
                self.physics.average_axisymmetric_scalar(
                    self.particles.viscosity_turbulent,
                    self.particles.zone_id,
                    len(self.particles),
                )
                self.physics.average_axisymmetric_scalar(
                    self.particles.viscosity_effective,
                    self.particles.zone_id,
                    len(self.particles),
                )

    def _update_strength(self, dt: float | None = None, announce: bool = True) -> None:
        """Advance vortex stretching, then viscous diffusion, over ``dt``."""
        if self.flow_model == "POTENTIAL":
            return

        if announce:
            self._announce_strength_update()

        dt = self.time_step_size if dt is None else dt
        self._apply_stretching(dt)
        self._apply_viscous_diffusion(dt)

    def _announce_strength_update(self) -> None:
        """Log the stretching formulation used for this strength update."""
        effective_mode = self._effective_stretching_mode()
        mode_eq = {
            "DIRECT": "(ω·∇)u",
            "TRANSPOSED": "(ω·∇')u",
            "MIXED": "½((ω·∇)u + (∇u)ᵀ·ω)",
        }.get(effective_mode, f"({effective_mode})")
        Logging.message(f"Updating strengths via {mode_eq}")

    def _effective_stretching_mode(self) -> str:
        """Return the user-selected stretching formulation."""
        return self.stretching_mode

    def _apply_stretching(self, dt: float) -> None:
        """Advance the configured vortex-stretching equation once per ``dt``."""
        if self.stretching_enabled:
            # Warn once when the explicit stretching step exceeds the strain-based target.
            if not self._stretch_dt_warned:
                gradient = self.particles_velocity_gradients
                strain = 0.5 * (gradient + np.swapaxes(gradient, 1, 2))
                sigma_max = (
                    float(np.max(np.abs(np.linalg.eigvalsh(strain)))) if len(strain) else 0.0
                )
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
                use_treecode=self.stretching_use_treecode,
                treecode_theta=self.stretching_treecode_theta,
            )
            ti.sync()

    def _validate_axisymmetric_orbits(self) -> None:
        """Reject malformed orbit IDs before applying rotational stage averages."""
        if self.axisymmetric_axis < 0 or self._axisymmetric_orbits_validated:
            return
        position = self.particles_positions.astype(np.float64)
        orbit_id = self.particles_zone_ids.astype(np.int64)
        group_id = self.particles_group_ids.astype(np.int64)
        count = len(position)
        if count == 0:
            return
        if np.any(orbit_id < 0) or int(orbit_id.max()) >= count:
            raise ValueError(
                "axisymmetric particles require non-negative zone_id orbit labels below "
                "the particle count"
            )

        present = np.unique(orbit_id)
        if not np.array_equal(present, np.arange(len(present))):
            raise ValueError("axisymmetric zone_id orbit labels must be contiguous from zero")

        axis = self.axisymmetric_axis
        b = (axis + 1) % 3
        c = (axis + 2) % 3
        radial = np.hypot(position[:, b], position[:, c])
        scale = max(1.0, float(np.abs(position).max(initial=0.0)))
        geometry_tolerance = 256.0 * np.finfo(self.np_dtype).eps * scale
        angle_tolerance = 512.0 * np.finfo(self.np_dtype).eps
        for orbit in present:
            selected = orbit_id == orbit
            orbit_count = int(np.count_nonzero(selected))
            if orbit_count < 8:
                raise ValueError(
                    f"axisymmetric orbit {orbit} has {orbit_count} particles; at least 8 are required"
                )
            if np.ptp(group_id[selected]) != 0:
                raise ValueError(f"axisymmetric orbit {orbit} spans more than one particle group")
            if (
                np.ptp(position[selected, axis]) > geometry_tolerance
                or np.ptp(radial[selected]) > geometry_tolerance
            ):
                raise ValueError(
                    f"axisymmetric orbit {orbit} does not have one axial and radial coordinate"
                )
            angles = np.sort(
                np.mod(np.arctan2(position[selected, c], position[selected, b]), 2 * np.pi)
            )
            gaps = np.diff(np.append(angles, angles[0] + 2 * np.pi))
            expected_gap = 2.0 * np.pi / orbit_count
            if np.max(np.abs(gaps - expected_gap)) > angle_tolerance:
                raise ValueError(f"axisymmetric orbit {orbit} is not uniformly periodic")

        self._axisymmetric_orbits_validated = True

    def _apply_coupled_advection_stretching(
        self, dt: float, *, precomputed_velocity_k1: bool = False
    ) -> None:
        """Advance positions and strengths at the same Runge--Kutta stages."""
        self._validate_axisymmetric_orbits()
        self.physics.update_positions_and_strengths(
            self.particles,
            dt=dt,
            scheme=self.stretching_scheme,
            mode=self.stretching_mode,
            use_treecode=self.stretching_use_treecode,
            treecode_theta=self.stretching_treecode_theta,
            conserve_moments=self.stretching_conserve_moments,
            conserve_energy=self.stretching_conserve_energy,
            axisymmetric_axis=self.axisymmetric_axis,
            precomputed_velocity_k1=precomputed_velocity_k1,
        )
        ti.sync()

    def _coupled_stable_dt(self, remaining_dt: float) -> float:
        """Return a strain- and displacement-limited coupled substep."""
        grad = self.particles_velocity_gradients
        stable_dt = float(remaining_dt)
        if len(grad):
            strain = 0.5 * (grad + np.swapaxes(grad, 1, 2))
            max_strain = float(np.max(np.abs(np.linalg.eigvalsh(strain))))
            if np.isfinite(max_strain) and max_strain > 0.0:
                stable_dt = min(
                    stable_dt,
                    self.coupled_max_strain_increment / max_strain,
                )

        spacing = getattr(self._viscous_config, "characteristic_distance", None)
        if spacing is not None and spacing > 0.0:
            velocity = self.particles_velocities
            max_speed = float(np.linalg.norm(velocity, axis=1).max()) if len(velocity) else 0.0
            if np.isfinite(max_speed) and max_speed > 0.0:
                stable_dt = min(
                    stable_dt,
                    self.coupled_max_advection_fraction * float(spacing) / max_speed,
                )
        return max(stable_dt, np.finfo(float).eps)

    def _apply_coupled_update_with_subcycling(
        self, dt: float, *, precomputed_velocity_k1: bool
    ) -> None:
        """Advance one macro step without clipping an inadmissible RK increment."""
        self.physics.rate_projection_max_correction_ratio = 0.0
        remaining = float(dt)
        substeps = 0
        reuse_velocity = bool(precomputed_velocity_k1)
        tolerance = 32.0 * np.finfo(float).eps * max(1.0, abs(dt))

        while remaining > tolerance:
            sub_dt = min(remaining, self._coupled_stable_dt(remaining))
            substeps += 1
            if substeps > self.coupled_max_substeps:
                raise RuntimeError(
                    "Coupled VPM step exceeded coupled_max_substeps. "
                    "The particle field is no longer temporally admissible at the "
                    "requested macro dt; reduce dt or refine the particle spacing."
                )

            if self.viscous_scheme == "CS":
                # Symmetric core-spreading split around the coupled inviscid update.
                self._apply_core_spreading_diffusion(0.5 * sub_dt)
                reuse_velocity = False

            target_moments = self._current_kernel_moments()
            self._apply_coupled_advection_stretching(sub_dt, precomputed_velocity_k1=reuse_velocity)
            self._restore_coupled_step_moments(target_moments)

            if self.viscous_scheme == "CS":
                self._apply_core_spreading_diffusion(0.5 * sub_dt)

            remaining -= sub_dt
            if remaining <= tolerance:
                break

            # Refresh stability bounds and, for LES, eddy viscosity before the next substep.
            self._update_velocity_and_gradients()
            self._update_LES_state()
            self.stabilization.update_residual_viscosity()
            reuse_velocity = self.viscous_scheme == "NONE"

        if substeps > 1:
            Logging.message(f"\t[CoupledSubcycling] {substeps} substeps for macro dt={dt:.3e}")

        if self.viscous_scheme in {"RWM", "DVH", "GBD"}:
            self._apply_viscous_diffusion(dt)

    def _current_kernel_moments(self):
        """Return circulation and both impulses for the active blob kernel."""
        if not self.stretching_conserve_moments or len(self.particles) == 0:
            return None
        from ..stabilization.filament_refinement import particle_moments

        return particle_moments(
            self.particles.position_cpu(use_cache=False).astype(np.float64),
            self.particles.circulation_cpu(use_cache=False).astype(np.float64),
            self.particles.radius_cpu(use_cache=False).astype(np.float64),
            angular_core_coefficient=self.physics._angular_core_coefficient,
        )

    def _restore_coupled_step_moments(self, target_moments) -> None:
        """Correct finite-RK drift in the conserved coupled-step moments."""
        if target_moments is None or len(self.particles) == 0:
            return
        from ..stabilization.divergence_relaxation import (
            _MomentNullspace,
            invariant_rows,
        )
        from ..stabilization.filament_refinement import particle_moments

        position = self.particles.position_cpu(use_cache=False).astype(np.float64)
        circulation = self.particles.circulation_cpu(use_cache=False).astype(np.float64)
        radius = self.particles.radius_cpu(use_cache=False).astype(np.float64)
        volume = self.particles.volume_cpu(use_cache=False).astype(np.float64)
        core_coefficient = self.physics._angular_core_coefficient
        current = particle_moments(
            position,
            circulation,
            radius,
            angular_core_coefficient=core_coefficient,
        )
        moment_change = np.concatenate(
            (
                target_moments[0] - current[0],
                target_moments[2] - current[2],
                target_moments[3] - current[3],
            )
        )
        nullspace = _MomentNullspace(
            invariant_rows(
                position,
                radius,
                angular_core_coefficient=core_coefficient,
            ),
            volume,
        )
        correction = nullspace.correction_for_moment_change(moment_change)
        correction_relative = float(
            np.linalg.norm(correction) / max(np.linalg.norm(circulation), np.finfo(float).tiny)
        )
        self.update_particle_circulations(
            np.ones(len(circulation), dtype=bool),
            correction.astype(self.np_dtype),
        )
        self.physics.rate_projection_max_correction_ratio = max(
            self.physics.rate_projection_max_correction_ratio,
            correction_relative,
        )

        uploaded = self.particles.circulation_cpu(use_cache=False).astype(np.float64)
        restored = particle_moments(
            position,
            uploaded,
            radius,
            angular_core_coefficient=core_coefficient,
        )
        scale = max(target_moments[1], np.finfo(float).tiny)
        impulse_scale = max(
            0.5 * float(np.linalg.norm(np.cross(position, circulation), axis=1).sum()),
            np.finfo(float).tiny,
        )
        angular_terms = (
            np.cross(position, np.cross(position, circulation)) / 3.0
            - core_coefficient * radius[:, None] ** 2 * circulation
        )
        angular_scale = max(
            float(np.linalg.norm(angular_terms, axis=1).sum()),
            np.finfo(float).tiny,
        )
        errors = (
            float(np.linalg.norm(restored[0] - target_moments[0])) / scale,
            float(np.linalg.norm(restored[2] - target_moments[2])) / impulse_scale,
            float(np.linalg.norm(restored[3] - target_moments[3])) / angular_scale,
        )
        if max(errors) > 4096.0 * np.finfo(self.np_dtype).eps:
            raise RuntimeError(
                "coupled-step moment projection exceeded its roundoff allowance: "
                f"circulation={errors[0]:.3e}, linear_impulse={errors[1]:.3e}, "
                f"angular_impulse={errors[2]:.3e}"
            )

    def _apply_core_spreading_diffusion(self, dt: float) -> None:
        """Advance Gaussian core spreading and optionally restore configured moments."""
        if dt <= 0.0 or len(self.particles) == 0:
            return
        if not self.stretching_conserve_moments:
            self.physics.core_spreading_diffusion(self.particles, dt)
            return

        from ..stabilization.divergence_relaxation import (
            _MomentNullspace,
            invariant_rows,
        )
        from ..stabilization.filament_refinement import particle_moments

        position = self.particles.position_cpu(use_cache=False).astype(np.float64)
        circulation = self.particles.circulation_cpu(use_cache=False).astype(np.float64)
        radius = self.particles.radius_cpu(use_cache=False).astype(np.float64)
        volume = self.particles.volume_cpu(use_cache=False).astype(np.float64)
        core_coefficient = self.physics._angular_core_coefficient
        before = particle_moments(
            position,
            circulation,
            radius,
            angular_core_coefficient=core_coefficient,
        )

        self.physics.core_spreading_diffusion(self.particles, dt)
        new_radius = self.particles.radius_cpu(use_cache=False).astype(np.float64)
        uncorrected = particle_moments(
            position,
            circulation,
            new_radius,
            angular_core_coefficient=core_coefficient,
        )
        moment_change = np.concatenate(
            (before[0] - uncorrected[0], before[2] - uncorrected[2], before[3] - uncorrected[3])
        )
        nullspace = _MomentNullspace(
            invariant_rows(
                position,
                new_radius,
                angular_core_coefficient=core_coefficient,
            ),
            volume,
        )
        correction = nullspace.correction_for_moment_change(moment_change)
        self.core_spreading_correction_relative = float(
            np.linalg.norm(correction) / max(np.linalg.norm(circulation), np.finfo(float).tiny)
        )
        self.update_particle_circulations(
            np.ones(len(circulation), dtype=bool),
            correction.astype(self.np_dtype),
        )

        uploaded = self.particles.circulation_cpu(use_cache=False).astype(np.float64)
        after = particle_moments(
            position,
            uploaded,
            new_radius,
            angular_core_coefficient=core_coefficient,
        )
        impulse_scale = max(
            0.5 * float(np.linalg.norm(np.cross(position, circulation), axis=1).sum()),
            np.finfo(float).tiny,
        )
        angular_terms = (
            np.cross(position, np.cross(position, circulation)) / 3.0
            - core_coefficient * radius[:, None] ** 2 * circulation
        )
        angular_scale = max(
            float(np.linalg.norm(angular_terms, axis=1).sum()),
            np.finfo(float).tiny,
        )
        errors = {
            "circulation": float(np.linalg.norm(after[0] - before[0]))
            / max(before[1], np.finfo(float).tiny),
            "linear_impulse": float(np.linalg.norm(after[2] - before[2])) / impulse_scale,
            "angular_impulse": float(np.linalg.norm(after[3] - before[3])) / angular_scale,
        }
        roundoff_limit = 4096.0 * np.finfo(self.np_dtype).eps
        if max(errors.values()) > roundoff_limit:
            raise RuntimeError(
                "core-spreading moment projection exceeded its roundoff allowance: "
                + ", ".join(f"{name}={value:.3e}" for name, value in errors.items())
            )

    def _apply_viscous_diffusion(self, dt: float) -> None:
        """Dispatch viscous diffusion by configured scheme."""
        if self.viscous_scheme == "NONE":
            return

        if self.viscous_scheme == "CS":
            Logging.message("Performing viscous diffusion via Core Spreading.")
            self._apply_core_spreading_diffusion(dt)
        elif self.viscous_scheme == "RWM":
            Logging.message("Performing viscous diffusion via Random Walk Method.")
            self.physics.random_walk_method_diffusion(self.particles, dt=dt)
        elif self.viscous_scheme in ("DVH", "GBD"):
            # DVH fires only when its fixed diffusion increment has accumulated.
            if self.viscous_scheme == "DVH" and self._dvh_substeps > 1:
                self._dvh_fire_counter += 1
                if self._dvh_fire_counter < self._dvh_substeps:
                    return
                self._dvh_fire_counter = 0
            new_p = self._apply_grid_diffusion(self._viscous_config, dt)
            if new_p is not None:
                from ..stabilization.divergence_relaxation import (
                    _MomentNullspace,
                    gaussian_invariant_rows,
                )
                from ..stabilization.filament_refinement import gaussian_particle_moments

                old_position = self.particles.position_cpu().astype(np.float64)
                old_circulation = self.particles.circulation_cpu().astype(np.float64)
                old_radius = self.particles.radius_cpu().astype(np.float64)
                old_moments = gaussian_particle_moments(
                    old_position,
                    old_circulation,
                    old_radius,
                )
                M = len(new_p["position"])
                new_position = np.asarray(new_p["position"], dtype=np.float64)
                proposed_circulation = np.asarray(
                    new_p["circulation"],
                    dtype=np.float64,
                )
                new_radius = np.asarray(new_p["radius"], dtype=np.float64)
                new_volume = np.asarray(new_p["volume"], dtype=np.float64)
                proposed_moments = gaussian_particle_moments(
                    new_position,
                    proposed_circulation,
                    new_radius,
                )
                # Preserve circulation and linear impulse without undoing diffusive core growth.
                target_moments = (
                    old_moments[0],
                    old_moments[2],
                    proposed_moments[3],
                )
                moment_change = np.concatenate(
                    (
                        target_moments[0] - proposed_moments[0],
                        target_moments[1] - proposed_moments[2],
                        np.zeros(3, dtype=np.float64),
                    )
                )
                nullspace = _MomentNullspace(
                    gaussian_invariant_rows(new_position, new_radius),
                    new_volume,
                )
                moment_correction = nullspace.correction_for_moment_change(moment_change)
                corrected_circulation = proposed_circulation + moment_correction
                new_p["circulation"] = corrected_circulation.astype(self.np_dtype)
                correction_relative = float(
                    np.linalg.norm(moment_correction)
                    / max(np.linalg.norm(proposed_circulation), np.finfo(float).tiny)
                )
                retention_bounds = self.stabilization_config.remove_particles_by_bounds
                if retention_bounds is not None and M > 0:
                    position = np.asarray(new_p["position"])
                    xmin, xmax, ymin, ymax, zmin, zmax = retention_bounds
                    inside = (
                        (xmin <= position[:, 0])
                        & (position[:, 0] <= xmax)
                        & (ymin <= position[:, 1])
                        & (position[:, 1] <= ymax)
                        & (zmin <= position[:, 2])
                        & (position[:, 2] <= zmax)
                    )
                    if not np.any(inside):
                        lo = position.min(axis=0)
                        hi = position.max(axis=0)
                        raise RuntimeError(
                            "Grid diffusion generated no particles inside the configured "
                            "retention domain; refusing to replace the wake. "
                            f"Generated extent: x=[{lo[0]:.6g}, {hi[0]:.6g}], "
                            f"y=[{lo[1]:.6g}, {hi[1]:.6g}], "
                            f"z=[{lo[2]:.6g}, {hi[2]:.6g}]."
                        )
                self.replace_vortex_particles(
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
                new_circulation = np.asarray(new_p["circulation"], dtype=np.float64)
                new_moments = gaussian_particle_moments(
                    new_position,
                    new_circulation,
                    new_radius,
                )
                strength_scale = max(old_moments[1], np.finfo(float).tiny)
                impulse_scale = max(
                    0.5
                    * float(
                        np.linalg.norm(
                            np.cross(old_position, old_circulation),
                            axis=1,
                        ).sum(dtype=np.float64)
                    ),
                    np.finfo(float).tiny,
                )
                angular_terms = (
                    np.cross(
                        old_position,
                        np.cross(old_position, old_circulation),
                    )
                    / 3.0
                    - old_radius[:, None] ** 2 * old_circulation / 3.0
                )
                angular_scale = max(
                    float(np.linalg.norm(angular_terms, axis=1).sum(dtype=np.float64)),
                    np.finfo(float).tiny,
                )
                errors = {
                    "circulation": float(np.linalg.norm(new_moments[0] - target_moments[0]))
                    / strength_scale,
                    "linear_impulse": float(np.linalg.norm(new_moments[2] - target_moments[1]))
                    / impulse_scale,
                    "angular_impulse": float(np.linalg.norm(new_moments[3] - target_moments[2]))
                    / angular_scale,
                }
                Logging.message(
                    "\t[Grid diffusion audit] "
                    f"N={len(old_position)}->{M}, "
                    f"correction={correction_relative:.3e}, "
                    f"dGamma={errors['circulation']:.3e}, "
                    f"dI={errors['linear_impulse']:.3e}, "
                    f"dA={errors['angular_impulse']:.3e}"
                )
                # Velocity is intentionally left stale; it is recomputed before the next consumer.
        ti.sync()

    def _apply_grid_diffusion(self, vc, dt: float):
        """Run DVH or GBD grid-based diffusion; return new particle dict."""
        # Fall back to particle viscosity when the scheme has no scalar ν.
        nu = vc.viscosity
        if nu is None or nu <= 0.0:
            n_part = self.particles.number_of_particles
            nu = float(self.particles.viscosity_cpu()[:n_part].mean()) if n_part > 0 else 0.0
        if self.viscous_scheme == "DVH":
            # LES uses per-particle effective viscosity for the heat-kernel width.
            nu_eff = None
            if self.flow_model == "LES":
                N = self.particles.number_of_particles
                if N > 0:
                    nu_eff = self.particles.viscosity_effective_cpu()
            Logging.message(
                f"\tPerforming DVH particle regeneration "
                f"(h={vc.dvh_grid_spacing:.3e}, nu={nu:.3e}, "
                f"threshold={vc.dvh_threshold:.2e}"
                + (
                    f", LES nu_eff/nu max={float(nu_eff.max()) / nu:.2f}"
                    if nu_eff is not None and nu > 0.0
                    else ""
                )
                + ")."
            )
            return self.physics.grid_based_diffusion(
                self.particles,
                dt=dt,
                h=vc.dvh_grid_spacing,
                nu=nu,
                domain_padding=vc.dvh_domain_padding,
                regen_threshold=vc.dvh_threshold,
                regen_threshold_mode=vc.dvh_threshold_mode,
                regen_threshold_window=getattr(vc, "regen_threshold_window", 3),
                rd_ratio=vc.dvh_rd_ratio,
                nu_eff=nu_eff,
                max_nodes=getattr(vc, "dvh_max_nodes", None),
                cap_abs_fraction=vc.regen_cap_abs_fraction,
            )
        else:
            # LES uses per-particle effective viscosity in the grid Laplacian.
            nu_eff = None
            if self.flow_model == "LES":
                N = self.particles.number_of_particles
                if N > 0:
                    nu_eff = self.particles.viscosity_effective_cpu()
            Logging.message(
                f"\tPerforming GBD diffusion"
                f"(h={vc.gbd_grid_spacing:.3e}, nu={nu:.3e}, "
                f"threshold={vc.gbd_threshold:.2e}"
                + (
                    f", LES nu_eff/nu max={float(nu_eff.max()) / nu:.2f}"
                    if nu_eff is not None and nu > 0.0
                    else ""
                )
                + ")."
            )
            return self.physics.gbd_diffusion(
                self.particles,
                dt=dt,
                h=vc.gbd_grid_spacing,
                nu=nu,
                domain_padding=vc.gbd_domain_padding,
                regen_threshold=vc.gbd_threshold,
                regen_threshold_mode=vc.gbd_threshold_mode,
                regen_threshold_window=getattr(vc, "regen_threshold_window", 3),
                nu_eff=nu_eff,
                max_nodes=getattr(vc, "gbd_max_nodes", None),
                cap_abs_fraction=vc.regen_cap_abs_fraction,
            )

    def _update_positions(self, dt: float | None = None, precomputed_k1: bool = False) -> None:
        """Advect particles with the configured time integrator.

        A precomputed first-stage velocity may be reused when velocity and gradients
        were evaluated together at the beginning of the step.
        """
        if self.advection_scheme == "NONE":
            return
        self.physics.update_positions(
            self.particles,
            self.time_step_size if dt is None else dt,
            scheme=self.advection_scheme,
            precomputed_k1=precomputed_k1,
        )

    # Panel–VPM coupling

    def _advance_panel(self):
        """Advance panel–VPM coupling and append any shed particles."""
        new_particles = self.panel_solver.advance(
            particles=self.particles,
            physics=self.physics,
            V_inf=self.background_velocity,
            dt=self.time_step_size,
            time=self.flow_time,
            step=self.time_step,
            logging_frequency=self.logging_frequency,
            density=getattr(self.config, "density", 1.0),
        )
        if new_particles is not None:
            n = len(new_particles["points"])
            if n > 0:
                visc_cfg = getattr(self.config, "viscous", None)
                nu = getattr(visc_cfg, "viscosity", None) if visc_cfg is not None else None
                if nu is None or nu <= 0:
                    nu = 1e-2
                viscosity = np.full(n, nu, dtype=self.np_dtype)

                pos = new_particles["points"].astype(self.np_dtype)
                strength = new_particles["strengths"].astype(self.np_dtype)
                rad = new_particles["radii"].astype(self.np_dtype)
                vol = new_particles["volumes"].astype(self.np_dtype)

                self.add_vortex_particles(
                    position=pos,
                    velocity=np.zeros((n, 3), dtype=self.np_dtype),
                    circulation=strength,
                    radius=rad,
                    volume=vol,
                    viscosity=viscosity,
                )

    # VLM–VPM coupling

    def _advance_vlm(self, dt: float) -> None:
        """Advance VLM–VPM coupling and append shed wake particles."""
        if self.vlm_solver is None:
            return

        wake_particles = self.vlm_solver.advance_coupled(
            particles=self.particles,
            physics=self.physics,
            config=self.config,
            dt=dt,
            time_step=self.time_step,
            time=self.flow_time,
        )

        if wake_particles is not None:
            self.add_vortex_particles(**wake_particles)

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

        keep_reference = None
        if hasattr(self, "_filament_reference_strengths"):
            position = self.particles.position_cpu()
            inside = (
                (xmin <= position[:, 0])
                & (position[:, 0] <= xmax)
                & (ymin <= position[:, 1])
                & (position[:, 1] <= ymax)
                & (zmin <= position[:, 2])
                & (position[:, 2] <= zmax)
            )
            keep_reference = inside if invert_selection else ~inside

        n_removed = self.particles.remove_particles_by_bounds(
            bounds, invert_selection=invert_selection
        )

        if n_removed > 0:
            if keep_reference is not None:
                self._filament_reference_strengths = self._filament_reference_strengths[
                    keep_reference
                ]
                self._filament_reference_lengths = self._filament_reference_lengths[keep_reference]
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
        if (
            removed_indices is not None
            and len(removed_indices) > 0
            and hasattr(self, "_filament_reference_strengths")
        ):
            keep = np.ones(particles_before, dtype=bool)
            keep[np.asarray(removed_indices, dtype=np.int64)] = False
            self._filament_reference_strengths = self._filament_reference_strengths[keep]
            self._filament_reference_lengths = self._filament_reference_lengths[keep]

        if len(self.particles) > 0:
            self.physics.compute_vorticities(self.particles)

        particles_after = len(self.particles)
        particles_removed = particles_before - particles_after

        Logging.particle_cleanup(percent, particles_before, particles_removed, particles_after)

        return particles_removed
