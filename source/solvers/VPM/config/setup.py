"""Top-level VPMSetup configuration for the VPM solver.
Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from dataclasses import dataclass, field, fields
import json
from pathlib import Path
import re
import sys
from typing import Any, Literal, Optional

from ..boundary_elements.vlm.config import VLMSetup
from .advection import AdvectionConfig
from .constants import (
    DEFAULT_BACKUP_FILENAME,
    DEFAULT_CUTOFF_RADIUS_FACTOR,
    DEFAULT_TIME_STEP,
    MAX_PARTICLES,
    TREECODE_SUPPORTED_KERNELS,
)
from .divergence_relaxation import DivergenceRelaxationConfig
from .filament_refinement import FilamentRefinementConfig
from .stabilization import StabilizationConfig
from .stretching import StretchingConfig
from .turbulence import TurbulenceConfig
from .velocity import VelocityConfig
from .viscous import ViscousConfig

sys.tracebacklimit = 0


@dataclass(frozen=True)
class VPMSetup:
    """
    Configuration dataclass for the VPM solver.

    This dataclass provides a clean, type-safe, and user-friendly interface
    for configuring all aspects of the VPM simulation. It includes validation,
    sensible defaults, and comprehensive documentation.

    Key benefits:
    - Type hints and validation for all parameters
    - Clear documentation with physical units
    - Sensible defaults for common use cases
    - Easy serialization/deserialization
    - IDE autocompletion support
    """

    # ---- TIME CONTROL ----
    time_step_size: float = DEFAULT_TIME_STEP
    """Time increment per simulation step [s]"""

    time: float = 0.0
    """Initial simulation time [s]"""

    step: int = 0
    """Initial time step number"""

    # ---- PHYSICS CONFIGURATION ----
    time_integration: Literal["FRACTIONAL", "COUPLED"] = "FRACTIONAL"
    """How advection and vortex stretching are time-integrated.

    ``FRACTIONAL`` retains the legacy position-then-strength update.
    ``COUPLED`` treats ``(x, Gamma)`` as one ODE state and evaluates both
    right-hand sides at common RK stages.  It currently supports matching RK2
    or RK3 advection/stretching schemes and core-spreading (or no) viscous
    diffusion.
    """

    coupled_max_strain_increment: float = 0.08
    """Maximum ``dt_sub * ||S||_2`` used by automatic coupled subcycling."""

    coupled_max_advection_fraction: float = 0.25
    """Maximum particle displacement per substep as a fraction of spacing."""

    coupled_max_substeps: int = 128
    """Fail instead of silently filtering physics when a macro step needs more substeps."""

    axisymmetric_no_swirl_axis: Literal["x", "y", "z"] | None = None
    """Preserve an axisymmetric no-swirl manifold during coupled integration.

    When set, particles with the same non-negative ``zone_id`` form one
    complete azimuthal orbit about the selected coordinate axis. Velocity and
    strength rates are rotationally averaged over each orbit at every common
    Runge--Kutta stage. Azimuthal velocity and meridional circulation rates are
    zero on this reflection-symmetric manifold. This is a symmetry-reduced
    discretization, not a post-step filter: the state is advanced only by the
    physically admissible part of the VPM right-hand side. Orbit geometry is
    checked before the first step so a missing or malformed ``zone_id`` cannot
    silently average unrelated particles.
    """

    advection: AdvectionConfig = field(default_factory=AdvectionConfig)
    """Configuration for advection term (scheme, etc.)."""

    stretching: StretchingConfig = field(default_factory=StretchingConfig.transposed)
    """Configuration for vortex stretching term."""

    viscous: ViscousConfig = field(default_factory=ViscousConfig.cs)
    """Configuration for viscous diffusion term."""

    turbulence: TurbulenceConfig = field(default_factory=TurbulenceConfig.dns)
    """Configuration for turbulence modeling (DNS/LES/INVISCID)."""

    stabilization: StabilizationConfig = field(default_factory=StabilizationConfig.disabled)
    """Complete stabilization policy: retention, local operators, refinement,
    divergence relaxation, and conservative regularization."""

    vlm: VLMSetup | None = None
    """Complete VLM definition. ``None`` selects a pure VPM simulation."""

    particles_kernel: Literal[
        "GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"
    ] = "GAUSSIAN"
    """Particle interaction kernel function type.

      ``HIGH_ORDER_GAUSSIAN`` — recommended production kernel: Gaussian with 2nd-order
      polynomial correction (2.5 − ρ²), improves far-field accuracy over plain Gaussian.
      """

    max_particles: int = MAX_PARTICLES
    """Maximum number of particles allowed in the simulation. Default: 500000"""

    max_targets: int = 200000
    """Maximum points in one velocity, vorticity, gradient, or sampler query.

    Taichi fields cannot be released independently. Allocate this capacity once
    when the solver starts so changing query sizes cannot accumulate abandoned
    device fields. Increase it before construction for larger sampler grids.
    """

    # ---- COMPUTATIONAL SETTINGS ----
    processing_unit: Literal["AUTO", "CPU", "VULKAN", "CUDA", "METAL"] = "AUTO"
    """Compute backend. Default ``'AUTO'`` selects the platform GPU and fails
    clearly when no GPU is available; CPU execution must be requested explicitly.

    * **macOS**             → Metal  (Apple's native GPU API)
    * **Linux / Windows**   → CUDA   (if an NVIDIA GPU + driver is found)
    * **Linux / Windows**   → Vulkan (universal fallback for AMD, Intel, NVIDIA)

    Override the automatic selection via the ``OPENONDA_PROCESSING_UNIT`` environment
    variable or by passing an explicit value here:

    * ``'AUTO'``       — best GPU for this platform, without CPU fallback
    * ``'VULKAN'``     — force Vulkan (Linux / Windows)
    * ``'CUDA'``       — force CUDA (requires NVIDIA GPU + driver)
    * ``'METAL'``      — force Metal (macOS only)
    * ``'CPU'``        — software rendering (no GPU required)
    """

    # ---- MONITORING AND DIAGNOSTICS ----
    logging_frequency: int = 0
    """Log flow diagnostics every N time steps (0 = disabled)."""

    export_flow_integrals: bool = True
    """Append computed global diagnostics to ``samples/flow_integrals.csv``."""

    export_discretization_health: bool = True
    """Also record the resolution metrics (particle overlap h/sigma, vorticity
    divergence error, Gamma-omega misalignment) with each diagnostics row.

    Invariant drift cannot detect a conservative-but-unresolved solution; these
    can.  Cost is one k-d tree plus a neighbour sum per diagnostics event, so it
    scales with ``logging_frequency`` and not with the step count."""

    log_mode: Literal["file", "tee", "console"] = "tee"
    """Solver output destination.

      - ``'tee'`` (default) — write to ``<backup_directory>/vpm*.log`` *and* keep
        the caller's console output working.
      - ``'file'`` — redirect to the log file only.  **This rebinds process-wide
        ``sys.stdout``/``sys.stderr`` for the lifetime of the solver**, so every
        ``print`` in the host program and in unrelated libraries is captured too.
        Only choose it for a batch run that owns the whole process; it was the
        default until the 2026-08 audit, where it silently swallowed diagnostics
        that had nothing to do with the solver.
      - ``'console'`` — no log file, streams untouched.

      All three restore the original streams via an ``atexit`` hook."""

    timing_frequency: int = 0
    """Print the cumulative runtime-profiling report every N time steps
    (0 = disabled). The per-step time line is always shown; this only controls
    the periodic ``RuntimeProfiler`` summary. ``VPMSolver.print_timing()`` can be
    called manually at any time regardless of this setting."""

    # ---- BACKUP AND OUTPUT ----
    backup_frequency: int = 0
    """Save simulation state every N time steps (0 = disabled)."""

    backup_file_name: str = DEFAULT_BACKUP_FILENAME
    """Optional infix in backup names, producing prefixes like 'vpm_<name>_*'."""

    backup_directory: str = "solution"
    """Output directory for restart backups; sampled files use the solver-managed root."""

    sample_subdirectory: str | None = None
    """Optional case-relative directory below the solver-managed ``samples/`` root.

    The root location is fixed by :func:`resolve_samples_dir`; this field only
    separates several runs that share one flow root, for example
    ``samples/merging_dvh/``.
    """

    clean: bool = False
    """If True, delete the backup_directory before starting the simulation."""

    # ---- PHYSICS PARAMETERS ----
    cutoff_radius_factor: float = DEFAULT_CUTOFF_RADIUS_FACTOR
    """Cutoff radius multiplier for particle interactions (performance optimization)"""

    precision: Literal["f32", "f64"] = "f32"
    """Floating-point precision of the particle fields.

      - ``'f32'`` (default) — **the supported production precision.**  Fast on
        GPU (Vulkan/CUDA); the 'Atomic add may lose precision' warnings are
        expected and acceptable.
      - ``'f64'`` — **EXPERIMENTAL and not end-to-end.**  It widens the particle
        container and the direct velocity summation, but three parts of the
        pipeline stay f32 regardless, so the delivered accuracy is well short of
        double precision:

          * the LBVH treecode is f32 throughout — fields, multipoles and host
            buffers — so ``velocity=VelocityConfig.treecode(...)`` is rejected
            with ``precision='f64'`` rather than silently downgraded;
          * the velocity-gradient and energy/helicity accumulators in
            ``numerics/kernels_common.py`` are hardcoded ``ti.f32``;
          * the Gaussian ``q``/``g`` use the Abramowitz & Stegun 7.1.26 erf
            approximation, whose max absolute error is 1.4e-7 — above f32 eps
            and nine orders above f64 eps.

        Requires ``processing_unit='CPU'`` (Vulkan/Metal have limited f64
        support).  Treat results as f32-accurate until the remaining paths are
        widened; see docs/reviews/2026-08-vpm-audit.md finding N-3."""

    random_seed: int = 42
    """Seed for Taichi's RNG (default 42).

      Only the Random Walk Method (``ViscousConfig(scheme='RWM')``) consumes it:
      its Brownian displacements are drawn from this stream.  A fixed seed makes
      a single run reproducible, but it also makes every run identical, so an
      RWM ensemble must vary this across members for the stochastic diffusion to
      average out.  Ignored on the Metal backend."""

    device_memory_fraction: float = 0.5
    """Fraction of GPU VRAM reserved for Taichi's internal memory pool (default 0.5).

      The remaining VRAM is available for external-array staging buffers used to
      transfer numpy arrays to/from the GPU.  If your simulation crashes with
      ``Failed to allocate ext arr buffer``, **lower** this value (e.g. 0.3–0.4)
      so that more VRAM is left for staging.

      Clamped to the range [0.1, 0.7].  Values above ~0.7 almost always cause
      allocation failures on Vulkan backends."""

    debug_mode: bool = False
    """Enable Taichi debug checks. This must be selected before solver construction."""

    freestream_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Free-stream velocity vector [ux, uy, uz] in m/s."""

    verbose: bool = True
    """Enable verbose output (print particle shedding info, etc.)."""

    # ---- VELOCITY COMPUTATION ----
    velocity: VelocityConfig | None = None
    """Configuration for velocity field computation (direct vs treecode)."""

    # ---- SOLVER INSTANCES (Dependency Injection) ----
    panel_solver: Any | None = None
    """Panel solver instance for hybrid simulations."""

    # ---- FIELD SAMPLERS ----
    samplers: tuple[Any, ...] | None = None
    """List of field samplers (SurfaceSampler, LineSampler) called at logging_frequency intervals."""

    final_samplers: tuple[Any, ...] | None = None
    """Field samplers executed only when ``VPMSolver.execute_final_samplers()`` is called."""

    body_stl: str | None = None
    """Path to body STL file for field sampler masking and geometry-aware output.
      Used by field samplers to mask interior points and project near-wall velocities.
      Can be absolute or relative to case directory."""

    vpm_domain_bounds: tuple[float, ...] | None = None
    """Optional ``[xmin, xmax, ymin, ymax, zmin, zmax]`` of the VPM domain.
      When set, the DVH diffusion grid is pre-allocated to cover this domain
      (if the memory cost is acceptable) and re-allocation is capped at these
      bounds.  Passed automatically by the coupler."""

    def __post_init__(self):
        """Post-initialization validation and setup."""

        if len(self.freestream_velocity) != 3:
            raise ValueError("freestream_velocity must contain three coordinates")
        object.__setattr__(
            self,
            "freestream_velocity",
            tuple(float(value) for value in self.freestream_velocity),
        )
        if self.samplers is not None:
            object.__setattr__(self, "samplers", tuple(self.samplers))
        if self.final_samplers is not None:
            object.__setattr__(self, "final_samplers", tuple(self.final_samplers))
        if self.vpm_domain_bounds is not None:
            object.__setattr__(self, "vpm_domain_bounds", tuple(self.vpm_domain_bounds))
        output_name = self.backup_file_name.strip()
        if output_name and (
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", output_name) is None
            or output_name.startswith(("vpm_", "vlm_"))
        ):
            raise ValueError(
                "backup_file_name must be a filename-safe infix without a path, "
                "extension, or solver prefix"
            )
        object.__setattr__(self, "backup_file_name", output_name)

        if self.sample_subdirectory is not None:
            sample_path = Path(self.sample_subdirectory)
            if (
                not self.sample_subdirectory
                or sample_path.is_absolute()
                or any(part in {".", ".."} for part in sample_path.parts)
            ):
                raise ValueError(
                    "sample_subdirectory must be a non-empty relative path below samples/"
                )
            object.__setattr__(self, "sample_subdirectory", str(sample_path))

        if self.velocity is None:
            # The GPU LBVH currently implements Gaussian and Winckelmans
            # regularisation. Corrected/super-Gaussian solvers remain fully
            # GPU-capable through exact direct summation instead of failing
            # later during the first advance().
            if self.particles_kernel.upper() in ("GAUSSIAN", "WINCKELMANS"):
                velocity = VelocityConfig.treecode(theta=0.3)
            else:
                velocity = VelocityConfig.direct()
            object.__setattr__(self, "velocity", velocity)

        # Validate precision
        if self.precision not in ("f32", "f64"):
            raise ValueError(f"precision must be 'f32' or 'f64', got '{self.precision}'")

        # Validation
        self._validate_config()

    def _validate_config(self):
        """Comprehensive validation of configuration parameters."""
        # Time step validation
        if self.time_step_size <= 0:
            raise ValueError("time_step_size must be positive")

        integration = self.time_integration.upper()
        if integration not in {"FRACTIONAL", "COUPLED"}:
            raise ValueError("time_integration must be 'FRACTIONAL' or 'COUPLED'")
        if integration == "COUPLED":
            advection_scheme = self.advection.scheme.upper()
            stretching_scheme = self.stretching.scheme.upper()
            if not self.stretching.enabled:
                raise ValueError("COUPLED time integration requires stretching to be enabled")
            if advection_scheme != stretching_scheme or advection_scheme not in {"RK2", "RK3"}:
                raise ValueError(
                    "COUPLED time integration requires matching RK2 or RK3 "
                    "advection and stretching schemes"
                )
            if self.viscous.scheme.upper() not in {"NONE", "CS", "DVH", "GBD"}:
                raise ValueError(
                    "COUPLED time integration supports NONE, CS, DVH, or GBD diffusion"
                )
        elif self.stretching.conserve_moments or self.stretching.conserve_energy:
            raise ValueError(
                "conserve_moments requires COUPLED time integration; conserve_energy does too, "
                "because position and strength rates share one Runge--Kutta stage"
            )
        if self.axisymmetric_no_swirl_axis is not None:
            axis = self.axisymmetric_no_swirl_axis.lower()
            if axis not in {"x", "y", "z"}:
                raise ValueError("axisymmetric_no_swirl_axis must be 'x', 'y', 'z', or None")
            object.__setattr__(self, "axisymmetric_no_swirl_axis", axis)
            if self.time_integration.upper() != "COUPLED":
                raise ValueError("axisymmetric_no_swirl_axis requires COUPLED time integration")
            if self.stabilization.remove_particles_by_bounds is not None:
                raise ValueError(
                    "axisymmetric_no_swirl_axis is incompatible with particle retention"
                )
            if (
                self.stabilization.filament_refinement.enabled
                or self.stabilization.divergence_relaxation.enabled
            ):
                raise ValueError(
                    "axisymmetric_no_swirl_axis is incompatible with refinement or "
                    "divergence relaxation"
                )
        if self.coupled_max_strain_increment <= 0.0:
            raise ValueError("coupled_max_strain_increment must be positive")
        if self.coupled_max_advection_fraction <= 0.0:
            raise ValueError("coupled_max_advection_fraction must be positive")
        if self.coupled_max_substeps < 1:
            raise ValueError("coupled_max_substeps must be at least one")

        # Frequency validation
        if self.logging_frequency < 0:
            raise ValueError("logging_frequency must be non-negative")

        if self.timing_frequency < 0:
            raise ValueError("timing_frequency must be non-negative")

        if self.max_particles < 1:
            raise ValueError("max_particles must be at least 1")

        if self.max_targets < 1:
            raise ValueError("max_targets must be at least 1")

        if self.log_mode not in {"file", "tee", "console"}:
            raise ValueError("log_mode must be 'file', 'tee', or 'console'")

        # Backup frequency validation
        if self.backup_frequency < 0:
            raise ValueError("backup_frequency must be non-negative")

        # Processing unit validation
        valid_processing_units = [
            "AUTO",
            "CPU",
            "VULKAN",
            "CUDA",
            "METAL",
        ]
        if self.processing_unit.upper() not in valid_processing_units:
            raise ValueError(
                f"processing_unit must be one of {valid_processing_units}, got '{self.processing_unit}'"
            )

        # Particles kernel validation
        _kernel_up = self.particles_kernel.upper()
        valid_kernels = ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
        if _kernel_up not in valid_kernels:
            raise ValueError(
                f"particles_kernel must be one of {valid_kernels}, got '{self.particles_kernel}'"
            )
        # The treecode implements only a subset of the regularization kernels
        # (it carries its own Taichi q/zeta).
        _treecode = self.velocity is not None and self.velocity.method == "TREECODE"
        if _treecode and _kernel_up not in TREECODE_SUPPORTED_KERNELS:
            raise ValueError(
                f"particles_kernel='{_kernel_up}' cannot be used with "
                f"velocity method 'TREECODE': the treecode implements only "
                f"{list(TREECODE_SUPPORTED_KERNELS)}. Either switch to "
                f"VelocityConfig.direct() or choose a supported kernel."
            )
        # The treecode's fields, multipoles and host buffers are f32 throughout,
        # so this pairing delivers f32 accuracy under an f64 label.
        if _treecode and self.precision == "f64":
            raise ValueError(
                "precision='f64' cannot be used with velocity method 'TREECODE': "
                "the treecode is f32 end-to-end, so the velocity field would be "
                "single precision regardless. Use VelocityConfig.direct() for an "
                "f64 run, or precision='f32' with the treecode."
            )
        if self.stabilization.filament_refinement.enabled and _kernel_up != "GAUSSIAN":
            raise ValueError(
                "filament refinement currently requires the GAUSSIAN kernel so its "
                "angular-impulse correction and exact transfer audit use the "
                "same declared regularization"
            )
        if self.stabilization.divergence_relaxation.enabled and _kernel_up != "GAUSSIAN":
            raise ValueError("divergence relaxation currently requires the GAUSSIAN kernel")
        if (
            self.stabilization.divergence_relaxation.enabled
            and not self.stabilization.filament_refinement.enabled
        ):
            raise ValueError(
                "divergence relaxation requires filament refinement so the "
                "interpolation solve cannot hide inadequate vortex-line resolution"
            )
        if self.stabilization.regularization_frequency > 0 and _kernel_up != "GAUSSIAN":
            raise ValueError("conservative regularization currently requires GAUSSIAN particles")
        if self.stabilization.regularization_frequency > 0 and (
            self.stabilization.filament_refinement.enabled
            or self.stabilization.divergence_relaxation.enabled
        ):
            raise ValueError(
                "conservative regularization replaces refinement and divergence relaxation"
            )
        if (
            self.stabilization.regularization_max_particles is not None
            and self.stabilization.regularization_max_particles > self.max_particles
        ):
            raise ValueError("regularization_max_particles cannot exceed VPMSetup.max_particles")
        if (
            self.stabilization.filament_refinement.max_particles is not None
            and self.stabilization.filament_refinement.max_particles > self.max_particles
        ):
            raise ValueError(
                "filament-refinement max_particles cannot exceed VPMSetup.max_particles"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary for serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary representation of the solver configuration, including
            all nested dataclass fields serialised as nested dictionaries.
        """
        # Serialize generic logic (assuming dataclasses.asdict or manual)
        # Since we have nested dataclasses, let's use a helper or manual construct

        def _as_dict(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _as_dict(getattr(obj, k)) for k in obj.__dataclass_fields__}
            return obj

        return {
            "time_step_size": self.time_step_size,
            "time": self.time,
            "step": self.step,
            "time_integration": self.time_integration,
            "coupled_max_strain_increment": self.coupled_max_strain_increment,
            "coupled_max_advection_fraction": self.coupled_max_advection_fraction,
            "coupled_max_substeps": self.coupled_max_substeps,
            "axisymmetric_no_swirl_axis": self.axisymmetric_no_swirl_axis,
            "advection": _as_dict(self.advection),
            "stretching": _as_dict(self.stretching),
            "viscous": _as_dict(self.viscous),
            "turbulence": _as_dict(self.turbulence),
            "stabilization": _as_dict(self.stabilization),
            # Runtime geometry and kinematics are not particle-state backup data.
            "vlm": None,
            "particles_kernel": self.particles_kernel,
            "max_particles": self.max_particles,
            "max_targets": self.max_targets,
            "logging_frequency": self.logging_frequency,
            "export_flow_integrals": self.export_flow_integrals,
            "export_discretization_health": self.export_discretization_health,
            "log_mode": self.log_mode,
            "timing_frequency": self.timing_frequency,
            "backup_frequency": self.backup_frequency,
            "backup_file_name": self.backup_file_name,
            "backup_directory": self.backup_directory,
            "sample_subdirectory": self.sample_subdirectory,
            "cutoff_radius_factor": self.cutoff_radius_factor,
            "precision": self.precision,
            "random_seed": self.random_seed,
            "device_memory_fraction": self.device_memory_fraction,
            "processing_unit": self.processing_unit,
            "debug_mode": self.debug_mode,
            "clean": self.clean,
            "freestream_velocity": list(self.freestream_velocity)
            if self.freestream_velocity
            else [0.0, 0.0, 0.0],
            "verbose": self.verbose,
            "velocity": _as_dict(self.velocity) if self.velocity else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VPMSetup":
        """Create configuration from dictionary.

        Returns
        -------
        VPMSetup
            Configuration object reconstructed from the dictionary.
        """
        values = dict(data)
        if "dt" in values and "time_step_size" not in values:
            values["time_step_size"] = values.pop("dt")
        nested_types = {
            "advection": AdvectionConfig,
            "stretching": StretchingConfig,
            "viscous": ViscousConfig,
            "turbulence": TurbulenceConfig,
            "velocity": VelocityConfig,
        }
        for name, config_type in nested_types.items():
            if isinstance(values.get(name), dict):
                values[name] = config_type(**values[name])
        values["stabilization"] = cls._stabilization_from_dict(values)
        return cls(**values)

    @staticmethod
    def _stabilization_from_dict(values: dict[str, Any]) -> StabilizationConfig:
        """Rebuild the stabilization policy, folding in pre-nesting layouts.

        Checkpoints written while filament refinement and divergence relaxation
        were top-level entries are still restorable: their sections are moved
        under ``stabilization`` and consumed from ``values``.
        """
        stabilization = values.pop("stabilization", None)
        if isinstance(stabilization, StabilizationConfig):
            values.pop("filament_refinement", None)
            values.pop("divergence_relaxation", None)
            return stabilization
        section = dict(stabilization) if isinstance(stabilization, dict) else {}
        nested_types = {
            "filament_refinement": FilamentRefinementConfig,
            "divergence_relaxation": DivergenceRelaxationConfig,
        }
        for name, config_type in nested_types.items():
            legacy = values.pop(name, None)
            entry = section.get(name, legacy)
            if isinstance(entry, dict):
                known = {f.name for f in fields(config_type)}
                section[name] = config_type(
                    **{key: value for key, value in entry.items() if key in known}
                )
            elif isinstance(entry, config_type):
                section[name] = entry
            else:
                section.pop(name, None)
        known = {f.name for f in fields(StabilizationConfig)}
        return StabilizationConfig(**{key: value for key, value in section.items() if key in known})

    @staticmethod
    def viscous_flow_simulation(
        time_step_size: float = 0.01,
        freestream_velocity: Any = (0.0, 0.0, 0.0),
        viscous: Optional["ViscousConfig"] = None,
        **kwargs,
    ) -> "VPMSetup":
        """
        Create a standard viscous flow simulation configuration.

        **Default physics:**
          - Stretching: disabled
          - Viscous:    Core Spreading (CS)
          - Turbulence: DNS (no subgrid model)
          - Advection:  RK3 (SSP-RK3)
          - Velocity:   Treecode (θ = 0.5)

        Args:
              time_step_size: Simulation time step [s]
              freestream_velocity: Free-stream velocity [m/s]
              viscous: Viscous scheme configuration (None = CS)
              **kwargs: Additional parameters to override defaults

        Returns:
              VPMSetup: Initialized configuration object

        Examples:
              >>> # Default viscous flow
              >>> config = VPMSetup.viscous_flow_simulation()
              >>> config.time_step_size
              0.01

              >>> config = VPMSetup.viscous_flow_simulation(
              ...     dt=0.005, freestream_velocity=(10.0, 0.0, 0.0)
              ... )
              >>> config.time_step_size
              0.005
        """
        if viscous is None:
            viscous = ViscousConfig.cs()

        stretching = kwargs.pop("stretching", StretchingConfig.disabled())

        return VPMSetup(
            time_step_size=time_step_size,
            stretching=stretching,
            viscous=viscous,
            freestream_velocity=list(freestream_velocity),
            **kwargs,
        )

    @staticmethod
    def dns_simulation(time_step_size: float = 0.01, **kwargs) -> "VPMSetup":
        """
        Create a Direct Numerical Simulation (DNS) configuration.

        **Default physics:**
          - Advection:  RK3 (SSP-RK3)
          - Stretching: TRANSPOSED mode, RK3 (SSP-RK3)
          - Viscous:    Core Spreading (CS)
          - Turbulence: DNS (molecular viscosity only, no SGS model)
          - Velocity:   Treecode (θ = 0.5)
          - Stabilization: disabled

        Args:
              time_step_size: Simulation time step [s]
              **kwargs: Additional parameters to override defaults

        Returns:
              VPMSetup: Initialized configuration object

        Examples:
              >>> # Default DNS
              >>> config = VPMSetup.dns_simulation()
              >>> config.time_step_size
              0.01

              >>> config = VPMSetup.dns_simulation(
              ...     dt=0.001, processing_unit="CPU"
              ... )
              >>> config.time_step_size
              0.001
        """
        # Extract common kwargs if provided to avoid multiple values for the same argument
        stretching = kwargs.pop("stretching", StretchingConfig.transposed())
        viscous = kwargs.pop("viscous", ViscousConfig.cs())
        turbulence = kwargs.pop("turbulence", TurbulenceConfig.dns())

        return VPMSetup(
            time_step_size=time_step_size,
            stretching=stretching,
            viscous=viscous,
            turbulence=turbulence,
            **kwargs,
        )

    @staticmethod
    def les_simulation(time_step_size: float = 0.01, cs: float = 0.17, **kwargs) -> "VPMSetup":
        """
        Create a Large Eddy Simulation (LES) configuration with the static
        Smagorinsky SGS model (k-equilibrium formulation).

        **Default physics:**
          - Advection:  RK3 (SSP-RK3)
          - Stretching: TRANSPOSED mode, RK3 (SSP-RK3)
          - Viscous:    Core Spreading (CS)
          - Turbulence: LES_SMAGORINSKY with C_s = 0.17, C_e = 1.048
          - Velocity:   Treecode (θ = 0.5)
          - Stabilization: disabled

        Args:
              time_step_size: Simulation time step [s]
              cs: Smagorinsky coefficient (default 0.17)
              **kwargs: Additional parameters to override defaults

        Returns:
              VPMSetup: Initialized configuration object

        Examples:
              >>> # Default LES
              >>> config = VPMSetup.les_simulation()
              >>> config.time_step_size
              0.01

              >>> config = VPMSetup.les_simulation(
              ...     dt=0.005, cs=0.15
              ... )
              >>> config.time_step_size
              0.005
        """
        stretching = kwargs.pop("stretching", StretchingConfig.transposed())
        viscous = kwargs.pop("viscous", ViscousConfig.cs())
        # Use les_smagorinsky or similar if available, else fallback
        turbulence = kwargs.pop("turbulence", TurbulenceConfig.les_smagorinsky(cs=cs))

        return VPMSetup(
            time_step_size=time_step_size,
            stretching=stretching,
            viscous=viscous,
            turbulence=turbulence,
            **kwargs,
        )

    def save_to_file(self, filename: str):
        """Save configuration to JSON file."""
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filename: str) -> "VPMSetup":
        """Load configuration from JSON file.

        Returns
        -------
        VPMSetup
            Configuration object deserialised from the JSON file.
        """
        with open(filename) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __str__(self) -> str:
        """Human-readable string representation."""
        flow_model = (
            self.turbulence.flow_model
            if self.turbulence and hasattr(self.turbulence, "flow_model")
            else "Inferred"
        )

        lines = ["VPM Solver Configuration:"]
        lines.append(f"  Flow Model: {flow_model}")
        lines.append("  Time Integration:")
        lines.append(f"    Advection:  {self.advection}")
        lines.append(f"    Stretching: {self.stretching}")
        lines.append(f"    Diffusion:  {self.viscous}")
        lines.append(f"  Processing Unit: {self.processing_unit}")
        lines.append(f"  Time Step Size: {self.time_step_size:.2e} s")
        lines.append(f"  Viscous Scheme: {self.viscous.scheme if self.viscous else 'None'}")
        lines.append(f"  Particle Kernel: {self.particles_kernel}")
        lines.append(f"  Cutoff Factor: {self.cutoff_radius_factor}")
        lines.append(f"  Logging Frequency: {self.logging_frequency} steps")
        lines.append(f"  Backup Frequency: {self.backup_frequency} steps")
        lines.append(f"  Freestream Velocity: {self.freestream_velocity} m/s")
        return "\n".join(lines)


# =========================================================
# UTILITY DECORATORS AND HELPERS
# =========================================================
