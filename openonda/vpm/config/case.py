"""Immutable VPM case: numerics, run plan and the complete case object."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ..boundary_elements.vlm.config import VLMConfig
from ..kernels.registry import kernel_definition, kernel_names
from ..numerics.rk_tableaux import SSPRK3, RKTableau
from ..physics.induction.base import InductionMethod
from ..physics.induction.direct import DirectInduction
from .constants import DEFAULT_CUTOFF_RADIUS_FACTOR, DEFAULT_TIME_STEP, MAX_N_PARTICLES
from .logging_config import LoggingConfig
from .output import Backup, Output, Samplers
from .stabilization import StabilizationConfig
from .state_limits import ParticleStateLimits
from .turbulence import TurbulenceConfig
from .viscous import ViscousConfig

if TYPE_CHECKING:
    from ..initialization import InitialCondition


@dataclass(frozen=True, slots=True)
class Numerics:
    """Immutable numerical and physical controls for a VPM case.

    Runtime clock values and output destinations are intentionally absent.
    ``time_step_size`` is the accepted-step duration in seconds and must be
    positive.  ``precision`` selects the particle compute dtype (``"f32"`` or
    ``"f64"``).  ``integrator`` advances position and vortex strength together,
    while ``induction`` supplies the stage-rate evaluator.

    Attributes
    ----------
    time_step_size : float
        Accepted VPM macro-step in seconds.
    integrator : RKTableau
        Explicit tableau shared by position and vortex-strength updates.
    induction : InductionMethod
        Backend/formulation construction object, cloned/bound for runtime use.
    axisymmetric_no_swirl_axis : {'x', 'y', 'z'} or None
        Optional rotational orbit projection axis.
    viscous, turbulence, stabilization : object
        Diffusion, LES, and accepted-step stabilization settings.
    vlm : VLMConfig or None
        Optional attached vortex-lattice configuration.
    particle_kernel : str
        Radial regularization kernel name.
    max_n_particles : int
        Fixed particle capacity.
    compute_device, precision : str
        Compute device and compute precision.
    random_seed : int
        Deterministic seed for stochastic diffusion/initializers.
    state_limits : ParticleStateLimits
        Accepted-state validation settings.
    freestream_velocity : tuple[float, float, float]
        Uniform background velocity in m/s.
    bodies, domain_bounds : tuple, tuple or None
        Optional body and diffusion-domain configuration. Bounds use
        ``(xmin, xmax, ymin, ymax, zmin, zmax)`` in metres.

    """

    time_step_size: float = DEFAULT_TIME_STEP
    integrator: RKTableau = field(default_factory=SSPRK3)
    induction: InductionMethod = field(default_factory=DirectInduction)
    axisymmetric_no_swirl_axis: Literal["x", "y", "z"] | None = None
    viscous: ViscousConfig = field(default_factory=ViscousConfig.cs)
    turbulence: TurbulenceConfig = field(default_factory=TurbulenceConfig.dns)
    stabilization: StabilizationConfig = field(default_factory=StabilizationConfig.disabled)
    vlm: VLMConfig | None = None
    particle_kernel: Literal["gaussian", "high_order_gaussian", "super_gaussian", "winckelmans"] = (
        "gaussian"
    )
    max_n_particles: int = MAX_N_PARTICLES
    compute_device: Literal["auto", "cpu", "vulkan", "cuda", "metal"] = "auto"
    precision: Literal["f32", "f64"] = "f32"
    random_seed: int = 42
    state_limits: ParticleStateLimits = field(default_factory=ParticleStateLimits)
    cutoff_radius_factor: float = DEFAULT_CUTOFF_RADIUS_FACTOR
    freestream_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    domain_bounds: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.integrator, RKTableau):
            raise TypeError("integrator must be an RKTableau instance")
        if not callable(getattr(self.induction, "evaluate_stage", None)):
            raise TypeError("induction must implement evaluate_stage")
        if not callable(getattr(self.induction, "build", None)):
            raise TypeError("induction must implement build() for solver-local runtime state")
        if isinstance(self.time_step_size, bool) or not isinstance(self.time_step_size, Real):
            raise TypeError("time_step_size must be a real number")
        if not math.isfinite(float(self.time_step_size)) or self.time_step_size <= 0.0:
            raise ValueError("time_step_size must be positive")
        if self.max_n_particles < 1:
            raise ValueError("max_n_particles must be at least one")
        valid_devices = {"auto", "cpu", "vulkan", "cuda", "metal"}
        if self.compute_device not in valid_devices:
            raise ValueError(f"compute_device must be one of {sorted(valid_devices)}")
        if self.precision not in {"f32", "f64"}:
            raise ValueError("precision must be 'f32' or 'f64'")
        kernel = self.particle_kernel
        if kernel not in kernel_names():
            raise ValueError(f"particle_kernel must be one of {sorted(kernel_names())}")
        if kernel != "gaussian" and (
            self.stabilization.regularization_interval_steps > 0
            or self.stabilization.divergence_relaxation.enabled
        ):
            raise ValueError(
                "Conservative regularization and divergence relaxation currently "
                "require GAUSSIAN particles; their reconstruction and invariants "
                "must not be applied to another kernel."
            )
        supported_kernels = getattr(self.induction, "supported_kernels", None)
        if supported_kernels is not None and kernel not in supported_kernels:
            raise ValueError(
                f"{type(self.induction).__name__} does not support particle_kernel={kernel}; "
                f"supported kernels: {sorted(supported_kernels)}"
            )
        if self.viscous.scheme == "cs" and not kernel_definition(kernel).supports_core_spreading:
            spreading = sorted(
                n for n in kernel_names() if kernel_definition(n).supports_core_spreading
            )
            raise ValueError(
                "Core Spreading requires a kernel with a positive normalized second moment; "
                f"{kernel} cancels that moment, so use {'/'.join(spreading)} or DVH/GBD."
            )
        if self.viscous.scheme == "rwm" and self.turbulence.flow_model == "les":
            raise ValueError(
                "RWM is restricted to spatially uniform effective viscosity; "
                "use GBD for LES variable-viscosity diffusion."
            )
        if self.viscous.scheme == "dvh" and self.turbulence.flow_model == "les":
            raise ValueError(
                "DVH requires spatially uniform effective viscosity; "
                "use GBD for LES variable-viscosity diffusion."
            )
        device = self.compute_device
        supported_devices = getattr(self.induction, "supported_devices", None)
        if supported_devices is not None and device not in supported_devices:
            raise ValueError(
                f"{type(self.induction).__name__} does not support compute_device={device}; "
                f"supported devices: {sorted(supported_devices)}"
            )
        if self.precision == "f64" and not getattr(self.induction, "supports_f64", True):
            raise ValueError(f"{type(self.induction).__name__} does not support precision='f64'")
        if self.precision == "f64" and self.viscous.scheme in {"dvh", "gbd"}:
            raise ValueError(
                f"viscous scheme {self.viscous.scheme} uses an f32 diffusion grid; "
                "select precision='f32' or use CS/RWM for a nominal f64 case"
            )
        if len(self.freestream_velocity) != 3:
            raise ValueError("freestream_velocity must contain three components")
        object.__setattr__(
            self,
            "freestream_velocity",
            tuple(float(value) for value in self.freestream_velocity),
        )
        if self.axisymmetric_no_swirl_axis not in {None, "x", "y", "z"}:
            raise ValueError("axisymmetric_no_swirl_axis must be x, y, z, or None")
        if self.domain_bounds is not None:
            if len(self.domain_bounds) != 6:
                raise ValueError("domain_bounds must contain (xmin, xmax, ymin, ymax, zmin, zmax)")
            object.__setattr__(self, "domain_bounds", tuple(float(v) for v in self.domain_bounds))
        if getattr(self.induction, "method", None) == "slip_slab":
            self._validate_slip_slab()

    def _validate_slip_slab(self) -> None:
        """Reject slab settings that disagree on the span or omit the slab images."""
        planes = (float(self.induction.z_min), float(self.induction.z_max))
        tolerance = 1.0e-9 * max(1.0, abs(planes[0]), abs(planes[1]))
        viscous = self.viscous
        if (
            viscous.scheme == "dvh"
            and viscous.dvh_domain_padding < viscous.dvh_support_radius_ratio
        ):
            raise ValueError(
                "Slip-slab DVH needs the complete heat-kernel support beyond both "
                "slip planes: set dvh_domain_padding >= dvh_support_radius_ratio "
                f"({viscous.dvh_support_radius_ratio})"
            )
        if viscous.scheme == "gbd" and viscous.gbd_remeshing_kernel != "m4_prime":
            raise ValueError("Slip-slab GBD requires M4_PRIME remeshing")
        # Every lattice that carries slab images must map nodes onto nodes, so
        # the slip planes fall on its nodes or half nodes.
        stabilization = self.stabilization
        lattices = (
            ("GBD grid", viscous.scheme == "gbd", viscous.gbd_grid_spacing),
            ("DVH grid", viscous.scheme == "dvh", viscous.dvh_grid_spacing),
            (
                "regularization grid",
                stabilization.regularization_interval_steps > 0,
                stabilization.regularization_grid_spacing,
            ),
            (
                "divergence-relaxation grid",
                stabilization.divergence_relaxation.enabled,
                stabilization.divergence_relaxation.grid_spacing,
            ),
        )
        for name, active, spacing in lattices:
            if not active:
                continue
            layers = 2.0 * (planes[1] - planes[0]) / spacing
            if abs(layers - round(layers)) > 1.0e-6 * max(1.0, layers):
                raise ValueError(
                    f"Slip-slab width {planes[1] - planes[0]:.6g} m must be a multiple of half "
                    f"the {name} spacing {spacing:.6g} m"
                )
        if self.domain_bounds is not None and (
            abs(self.domain_bounds[4] - planes[0]) > tolerance
            or abs(self.domain_bounds[5] - planes[1]) > tolerance
        ):
            raise ValueError(
                f"domain_bounds z {self.domain_bounds[4:6]} must equal the slip planes {planes}"
            )

        keep = stabilization.remove_particles_by_bounds
        if keep is not None and (
            keep[4] > planes[0] + tolerance or keep[5] < planes[1] - tolerance
        ):
            raise ValueError(
                f"remove_particles_by_bounds z ({keep[4]:.6g}, {keep[5]:.6g}) would remove "
                f"particles between the slip planes {planes}"
            )
        fringe = stabilization.outflow_fringe_depth
        if fringe is not None and (fringe[4] > 0.0 or fringe[5] > 0.0):
            raise ValueError("outflow_fringe_depth must be zero on the slip-plane z faces")


@dataclass(frozen=True, slots=True)
class RunPlan:
    """Finite solver run.

    The run advances to ``end_time``, which must be a whole number of
    ``Numerics.time_step_size`` steps. The initial output event writes the
    initial particle frame and the samplers with ``initial=True``; the final
    output event writes the final frame and the ``FinalOnly`` samplers.

    Attributes
    ----------
    end_time : float
        Physical end time in s; finite and non-negative.
    initial_output, final_output : bool
        Write the initial and the final output event.
    state_limit_action : {'raise', 'stop'}
        Action when an accepted state crosses a particle state limit.
        ``raise`` propagates the error. ``stop`` writes the final output and
        a backup and returns with status ``resolution_lost``; a nonfinite
        state returns with status ``unstable`` without final output.
    wall_time_limit_seconds : float or None
        Optional positive wall-clock budget, checked between accepted steps.
        A stopped run writes its final output and backup and has status
        ``wall_time_limit``.
    runtime_compute_device : {'auto', 'cpu', 'vulkan', 'cuda', 'metal'} or None
        Compute device for this process only. It overrides
        ``Numerics.compute_device`` without entering the restart
        configuration, and the run metadata records it.
    """

    end_time: float
    initial_output: bool = True
    final_output: bool = True
    state_limit_action: Literal["raise", "stop"] = "raise"
    wall_time_limit_seconds: float | None = None
    runtime_compute_device: Literal["auto", "cpu", "vulkan", "cuda", "metal"] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.end_time, bool) or not isinstance(self.end_time, Real):
            raise TypeError("RunPlan.end_time must be a real number")
        if not math.isfinite(self.end_time) or self.end_time < 0.0:
            raise ValueError("RunPlan.end_time must be finite and non-negative")
        object.__setattr__(self, "end_time", float(self.end_time))
        for name in ("initial_output", "final_output"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"RunPlan.{name} must be a bool")
        if self.state_limit_action not in {"raise", "stop"}:
            raise ValueError("RunPlan.state_limit_action must be 'raise' or 'stop'")
        limit = self.wall_time_limit_seconds
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, Real):
                raise TypeError("RunPlan.wall_time_limit_seconds must be a positive number")
            if not math.isfinite(limit) or limit <= 0:
                raise ValueError("RunPlan.wall_time_limit_seconds must be finite and positive")
        valid_devices = {"auto", "cpu", "vulkan", "cuda", "metal"}
        if self.runtime_compute_device is not None and (
            self.runtime_compute_device not in valid_devices
        ):
            raise ValueError(
                f"RunPlan.runtime_compute_device must be one of {sorted(valid_devices)} or None"
            )

    def n_steps(self, time_step_size: float) -> int:
        """Return the number of steps of size ``time_step_size`` (s) to ``end_time``.

        Raises
        ------
        ValueError
            If ``end_time`` is not a whole number of steps within round-off.
        """
        n_steps = round(self.end_time / time_step_size)
        if not math.isclose(n_steps * time_step_size, self.end_time, rel_tol=1.0e-9, abs_tol=0.0):
            raise ValueError(
                f"RunPlan.end_time {self.end_time!r} s is not a whole number of time steps "
                f"of {time_step_size!r} s"
            )
        return n_steps


@dataclass(frozen=True, slots=True, kw_only=True)
class VPMCase:
    """Complete immutable construction object for one VPM run.

    Construction performs no Taichi allocation or particle insertion; those
    are solver-owned side effects.

    Attributes
    ----------
    name : str
        Non-empty case name, recorded in ``solution/vpm_metadata.json``.
    directory : str or pathlib.Path, default='.'
        Case root; results go to ``solution/`` and ``samples/`` below it.
    variant : str or None, default=None
        Optional sub-directory of ``solution/`` and ``samples/`` for one
        variant of the case.
    numerics : Numerics
        Physical and numerical settings.
    initial_conditions : tuple[InitialCondition, ...]
        Particle builders, evaluated once when the run starts.
    run : RunPlan
        End time and run-control switches; the default has no steps.
    output : Output
        Particle visualization frames.
    samplers : Samplers
        Scientific samplers.
    backup : Backup
        Restart backups.
    logging : LoggingConfig
        Log verbosity and cadence.
    initial_weak_particle_percent : float, default=0
        Remove initial particles whose strength is below this percentage
        (0--100) of the largest particle strength.
    """

    name: str
    directory: str | Path = "."
    variant: str | None = None
    numerics: Numerics
    initial_conditions: tuple[InitialCondition, ...] = ()
    run: RunPlan = field(default_factory=lambda: RunPlan(end_time=0.0))
    output: Output = field(default_factory=Output)
    samplers: Samplers = field(default_factory=Samplers)
    backup: Backup = field(default_factory=Backup)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    initial_weak_particle_percent: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("VPMCase.name must be a non-empty string")
        directory = Path(self.directory)
        if not str(directory).strip():
            raise ValueError("VPMCase.directory must be a non-empty path")
        object.__setattr__(self, "directory", directory)
        if self.variant is not None:
            variant = Path(self.variant)
            if (
                not isinstance(self.variant, str)
                or not self.variant.strip()
                or variant.is_absolute()
                or any(part in {"", ".", ".."} for part in variant.parts)
            ):
                raise ValueError("VPMCase.variant must be None or a relative sub-directory name")
        for name, kind in (
            ("numerics", Numerics),
            ("run", RunPlan),
            ("output", Output),
            ("samplers", Samplers),
            ("backup", Backup),
            ("logging", LoggingConfig),
        ):
            if not isinstance(getattr(self, name), kind):
                raise TypeError(f"VPMCase.{name} must be a {kind.__name__} instance")
        if self.numerics.vlm is not None:
            self._validate_coupled_vlm_output()
        percent = self.initial_weak_particle_percent
        if isinstance(percent, bool) or not isinstance(percent, Real):
            raise TypeError("initial_weak_particle_percent must be a real number")
        if not math.isfinite(percent) or percent < 0.0 or percent > 100.0:
            raise ValueError("initial_weak_particle_percent must be finite and between 0 and 100")
        object.__setattr__(self, "initial_weak_particle_percent", float(percent))
        object.__setattr__(self, "initial_conditions", tuple(self.initial_conditions))

    @property
    def n_steps(self) -> int:
        """Number of accepted steps from time zero to ``RunPlan.end_time``."""
        return self.run.n_steps(self.numerics.time_step_size)

    def _validate_coupled_vlm_output(self) -> None:
        """Keep attached VLM output under the VPM run.

        Standalone ``VLMSolver`` callers retain their own force
        logging and ``VLMSampler`` API. Once VLM is attached to a VPM case,
        accepted-step force/loading records are mandatory in the VPM-owned
        sample directory, and the surface frames follow the particle frames.
        """
        vlm = self.numerics.vlm
        if vlm.logging_interval_steps != 1:
            raise ValueError(
                "attached VLM cannot configure logging_interval_steps; "
                "the VPM solver emits mandatory accepted-step samples"
            )
        if not vlm.sample_surface_forces:
            raise ValueError(
                "attached VLM requires sample_surface_forces=True; "
                "VLM scientific output is mandatory in the VPM sample path"
            )
        if any(surface.sample_forces is False for surface in vlm.surfaces):
            raise ValueError("attached VLM surfaces cannot opt out of mandatory scientific output")
        from ..sampling.vlm import VLMSampler

        if any(isinstance(sample, VLMSampler) for sample in self.samplers.samples):
            raise ValueError(
                "VLMSampler is standalone-only for coupled VPM cases; "
                "surface frames follow the particle frames and VLM tables "
                "use the solver sample path"
            )
