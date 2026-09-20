"""Immutable VPM case construction and mutable run-state types.

The public construction boundary is deliberately small: numerical controls are
specified once, initial conditions are declarative objects, and the run plan
defines the finite simulation lifecycle.  Runtime clocks belong to
``RestartState``, never to the immutable numerical configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from source.write_precision import DEFAULT_WRITE_PRECISION, WritePrecision

from ..boundary_elements.vlm.config import VLMSetup
from ..numerics.rk_tableaux import SSPRK3, RKTableau
from ..physics.induction.base import InductionMethod
from ..physics.induction.direct import DirectInduction
from .artifacts import Backup, Samplers
from .constants import DEFAULT_CUTOFF_RADIUS_FACTOR, DEFAULT_TIME_STEP, MAX_N_PARTICLES
from .diagnostics import DiagnosticsConfig
from .health import HealthLimits, ResourceLimits
from .setup import PanelBodySetup
from .stabilization import StabilizationConfig
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
    induction : InductionMethod or PlanarInduction
        Backend/formulation construction object, cloned/bound for runtime use.
    axisymmetric_no_swirl_axis : {'x', 'y', 'z'} or None
        Optional rotational orbit projection axis.
    viscous, turbulence, stabilization : object
        Diffusion, LES, and accepted-step stabilization policies.
    vlm : VLMSetup or None
        Optional attached vortex-lattice configuration.
    particle_kernel : str
        Radial regularization kernel name.
    max_n_particles, max_evaluation_points : int
        Fixed particle capacity and target-query capacity.
    compute_device, precision, write_precision : str
        Device and compute/write precision policies.
    random_seed : int
        Deterministic seed for stochastic diffusion/initializers.
    diagnostics, health_limits : object
        Diagnostic and accepted-state validation policies.
    freestream_velocity : tuple[float, float, float]
        Uniform background velocity in m/s.
    bodies, domain_bounds : tuple, tuple or None
        Optional body and diffusion-domain configuration. Bounds use
        ``(xmin, xmax, ymin, ymax, zmin, zmax)`` in metres.

    Notes
    -----
    PlanarInduction selects infinite-span Gaussian filaments, planar GBD and
    zero stretching. It requires zero spanwise freestream and rejects VLM,
    panel bodies, axisymmetric projection, LES and three-dimensional
    stabilization/pressure combinations. Its positive represented span
    converts stored strengths (m³/s) to filament circulation (m²/s); see
    PlanarInduction for the complete source-plane and field conventions.
    """

    time_step_size: float = DEFAULT_TIME_STEP
    integrator: RKTableau = field(default_factory=SSPRK3)
    induction: InductionMethod = field(default_factory=DirectInduction)
    axisymmetric_no_swirl_axis: Literal["x", "y", "z"] | None = None
    viscous: ViscousConfig = field(default_factory=ViscousConfig.cs)
    turbulence: TurbulenceConfig = field(default_factory=TurbulenceConfig.dns)
    stabilization: StabilizationConfig = field(default_factory=StabilizationConfig.disabled)
    vlm: VLMSetup | None = None
    particle_kernel: Literal["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"] = (
        "GAUSSIAN"
    )
    max_n_particles: int = MAX_N_PARTICLES
    max_evaluation_points: int = 200_000
    compute_device: Literal["AUTO", "CPU", "VULKAN", "CUDA", "METAL"] = "AUTO"
    precision: Literal["f32", "f64"] = "f32"
    write_precision: WritePrecision = DEFAULT_WRITE_PRECISION
    random_seed: int = 42
    device_memory_fraction: float = 0.5
    debug_mode: bool = False
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    health_limits: HealthLimits = field(default_factory=HealthLimits)
    cutoff_radius_factor: float = DEFAULT_CUTOFF_RADIUS_FACTOR
    freestream_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    verbose: bool = True
    panel_solver: object | None = None
    bodies: tuple[PanelBodySetup, ...] = ()
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
        if self.max_evaluation_points < 1:
            raise ValueError("max_evaluation_points must be at least one")
        valid_devices = {"AUTO", "CPU", "VULKAN", "CUDA", "METAL"}
        if self.compute_device.upper() not in valid_devices:
            raise ValueError(f"compute_device must be one of {sorted(valid_devices)}")
        if self.precision not in {"f32", "f64"}:
            raise ValueError("precision must be 'f32' or 'f64'")
        kernel = self.particle_kernel.upper()
        valid_kernels = {
            "GAUSSIAN",
            "HIGH_ORDER_GAUSSIAN",
            "SUPER_GAUSSIAN",
            "WINCKELMANS",
        }
        if kernel not in valid_kernels:
            raise ValueError(f"particle_kernel must be one of {sorted(valid_kernels)}")
        object.__setattr__(self, "particle_kernel", kernel)
        if kernel != "GAUSSIAN" and (
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
        if self.viscous.scheme == "CS" and kernel in {"HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN"}:
            raise ValueError(
                "Core Spreading requires a kernel with a positive normalized second moment; "
                f"{kernel} cancels that moment, so use GAUSSIAN/WINCKELMANS or DVH/GBD."
            )
        if self.viscous.scheme == "RWM" and self.turbulence.flow_model == "LES":
            raise ValueError(
                "RWM is restricted to spatially uniform effective viscosity; "
                "use GBD for LES variable-viscosity diffusion."
            )
        if self.viscous.scheme == "DVH" and self.turbulence.flow_model == "LES":
            raise ValueError(
                "DVH requires spatially uniform effective viscosity; "
                "use GBD for LES variable-viscosity diffusion."
            )
        if hasattr(self.induction, "planar_span"):
            if self.viscous.scheme not in {"GBD", "NONE"}:
                raise ValueError("Planar induction supports GBD or NONE diffusion")
            if self.viscous.gbd_remeshing_kernel != "M4_PRIME":
                raise ValueError("Planar GBD currently requires M4_PRIME remeshing")
            if self.axisymmetric_no_swirl_axis is not None:
                raise ValueError("Planar induction cannot use axisymmetric projection")
            if self.turbulence.flow_model == "LES":
                raise ValueError("Planar induction currently supports laminar flow only")
            if self.panel_solver is not None or self.bodies or self.vlm is not None:
                raise ValueError("Planar induction cannot be combined with 3D boundary elements")
            if abs(self.freestream_velocity[2]) > 1e-14:
                raise ValueError("Planar induction requires zero spanwise freestream")
            if (
                self.stabilization.regularization_interval_steps > 0
                or self.stabilization.divergence_relaxation.enabled
            ):
                raise ValueError(
                    "Planar induction must not use 3D regularization/divergence relaxation"
                )
        device = self.compute_device.upper()
        supported_devices = getattr(self.induction, "supported_devices", None)
        if supported_devices is not None and device not in supported_devices:
            raise ValueError(
                f"{type(self.induction).__name__} does not support compute_device={device}; "
                f"supported devices: {sorted(supported_devices)}"
            )
        object.__setattr__(self, "compute_device", device)
        object.__setattr__(self, "precision", self.precision.lower())
        if self.precision == "f64" and not getattr(self.induction, "supports_f64", True):
            raise ValueError(f"{type(self.induction).__name__} does not support precision='f64'")
        if self.precision == "f64" and self.viscous.scheme in {"DVH", "GBD"}:
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
        if self.axisymmetric_no_swirl_axis is not None:
            axis = self.axisymmetric_no_swirl_axis.lower()
            if axis not in {"x", "y", "z"}:
                raise ValueError("axisymmetric_no_swirl_axis must be x, y, z, or None")
            object.__setattr__(self, "axisymmetric_no_swirl_axis", axis)
        object.__setattr__(self, "bodies", tuple(self.bodies))
        body_uids = [body.uid for body in self.bodies]
        if len(body_uids) != len(set(body_uids)):
            duplicates = sorted({uid for uid in body_uids if body_uids.count(uid) > 1})
            raise ValueError("Duplicate panel body uid(s): " + ", ".join(duplicates))
        if self.domain_bounds is not None:
            if len(self.domain_bounds) != 6:
                raise ValueError("domain_bounds must contain (xmin, xmax, ymin, ymax, zmin, zmax)")
            object.__setattr__(self, "domain_bounds", tuple(float(v) for v in self.domain_bounds))


@dataclass(frozen=True, slots=True)
class RunPlan:
    """Finite solver lifecycle.

    ``steps`` is the number of accepted VPM steps to execute.  Initial samples
    are dispatched before the first step, final-only samples after the last
    accepted step, and a final restart backup is written on successful
    completion when ``final_backup`` is enabled.

    ``steps`` is required so a finite run cannot acquire an accidental default.
    ``initial_samples`` defaults to ``True`` because explicitly configured
    initial diagnostics are usually part of a reproducible case.  A final
    backup defaults to ``True`` to preserve a restart point after completed or
    deliberately stopped runs; it does not schedule scientific samplers.
    ``health_limit_action``
    controls an accepted state that crosses a configured health limit:
    ``"RAISE"`` preserves the exception behavior, while ``"STOP"`` writes
    terminal samples and a restart before returning with status
    ``"resolution_lost"`` when the state remains finite. A nonfinite state
    returns with status ``"unstable"`` and its failure reason, without final
    scientific samples or a restart of the rejected state.

    Attributes
    ----------
    steps : int
        Number of accepted VPM steps; required and non-negative.
    initial_samples, final_backup : bool
        Framework lifecycle switches for initial scientific output and the
        terminal numerical backup.
    health_limit_action : {'RAISE', 'STOP'}
        Policy when an accepted-state health limit is crossed. ``STOP`` also
        records a nonfinite state as ``unstable`` without serializing it.
    wall_time_limit_seconds : float or None
        Optional positive runtime budget checked between accepted steps.
    resource_limits : ResourceLimits or None
        Optional accepted-step process-resource bounds. A stopped bound uses
        the distinct ``resource_limit`` lifecycle status.
    runtime_compute_device : {'AUTO', 'CPU', 'VULKAN', 'CUDA', 'METAL'} or None
        Optional explicit backend selection for this process invocation. This
        runtime-only override is excluded from ``Numerics`` restart identity;
        the manifest records it so diagnostic/pilot runs remain auditable.
    """

    steps: int
    initial_samples: bool = True
    final_backup: bool = True
    health_limit_action: Literal["RAISE", "STOP"] = "RAISE"
    wall_time_limit_seconds: float | None = None
    resource_limits: ResourceLimits | None = None
    runtime_compute_device: Literal["AUTO", "CPU", "VULKAN", "CUDA", "METAL"] | None = None
    """Optional runtime budget, checked between accepted steps.

    Budget stops save terminal samplers and the configured final backup. They
    have status ``wall_time_limit``, not a numerical health failure. Initial
    construction and sampling count toward the budget; final output can add
    overhead, and an in-flight step is allowed to finish.
    """

    def __post_init__(self) -> None:
        if isinstance(self.steps, bool) or not isinstance(self.steps, int):
            raise TypeError("RunPlan.steps must be an integer")
        if self.steps < 0:
            raise ValueError("RunPlan.steps must be non-negative")
        action = str(self.health_limit_action).upper()
        if action not in {"RAISE", "STOP"}:
            raise ValueError("RunPlan.health_limit_action must be 'RAISE' or 'STOP'")
        object.__setattr__(self, "health_limit_action", action)
        limit = self.wall_time_limit_seconds
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, Real):
                raise TypeError("RunPlan.wall_time_limit_seconds must be a positive number")
            if not math.isfinite(limit) or limit <= 0:
                raise ValueError("RunPlan.wall_time_limit_seconds must be finite and positive")
        if self.resource_limits is not None and not isinstance(
            self.resource_limits, ResourceLimits
        ):
            raise TypeError("RunPlan.resource_limits must be ResourceLimits or None")
        if self.runtime_compute_device is not None:
            device = str(self.runtime_compute_device).upper()
            valid_devices = {"AUTO", "CPU", "VULKAN", "CUDA", "METAL"}
            if device not in valid_devices:
                raise ValueError(
                    f"RunPlan.runtime_compute_device must be one of {sorted(valid_devices)} or None"
                )
            object.__setattr__(self, "runtime_compute_device", device)


@dataclass(slots=True)
class RestartState:
    """Mutable physical clock restored from a numerical backup.

    ``time`` is in seconds and ``step`` is the number of accepted steps.  The
    state is runtime-owned so it cannot accidentally be serialized as part of
    a new case's numerical construction.

    ``time`` is finite physical time in seconds and ``step`` is a non-negative
    accepted-step counter.  The runtime restores this object from numerical
    backup data; users construct it only for advanced interactive runs.

    The object is mutable because the live solver updates it after every
    accepted step; it is not part of immutable case identity.
    """

    time: float = 0.0
    step: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.time, bool) or not isinstance(self.time, Real):
            raise TypeError("RestartState.time must be a real number")
        if not math.isfinite(float(self.time)) or self.time < 0.0:
            raise ValueError("RestartState.time must be finite and non-negative")
        if isinstance(self.step, bool) or not isinstance(self.step, int):
            raise TypeError("RestartState.step must be an integer")
        if self.step < 0:
            raise ValueError("RestartState.step must be non-negative")


@dataclass(frozen=True, slots=True)
class VPMCase:
    """Complete immutable construction object for one VPM run.

    ``initial_conditions`` contains typed flow builders.  Each builder creates
    a particle set during :meth:`VPMSolver.run`, rather than requiring callers
    to unpack particle arrays or mutate a newly constructed solver.

    ``numerics`` is required and is the only numerical construction object.
    ``backup`` owns restart and log destinations, ``samplers`` owns scientific
    samples, and ``run`` supplies the finite lifecycle.
    ``initial_weak_particle_percent`` optionally removes particles below that
    percentage of the assembled cloud's maximum vortex-strength magnitude.
    ``directory`` is the case root (default current directory) below which
    framework-owned artifacts are written. Invalid nested plans or an empty
    directory raise :class:`TypeError` or :class:`ValueError`. ``name`` is an
    optional stable case identifier recorded in solver-owned metadata;
    it does not affect the equations or output paths.

    Attributes
    ----------
    numerics : Numerics
        Required immutable physical/numerical controls.
    initial_conditions : tuple[InitialCondition, ...]
        Declarative builders evaluated once at run/first-advance time.
    backup, samplers : Backup, Samplers
        Restart/log policy and scientific output policy.
    run : RunPlan
        Finite accepted-step lifecycle.
    initial_weak_particle_percent : float
        Optional 0--100 percentage threshold for initial strength pruning.
    directory : str or pathlib.Path
        Case root for framework-owned artifacts.
    name : str or None
        Optional non-empty case identifier stored in ``vpm_metadata.json``.

    Construction performs no Taichi allocation or particle insertion; those are
    solver-owned side effects.
    """

    numerics: Numerics
    initial_conditions: tuple[InitialCondition, ...] = ()
    backup: Backup = field(default_factory=Backup)
    samplers: Samplers = field(default_factory=Samplers)
    run: RunPlan = field(default_factory=lambda: RunPlan(steps=0))
    initial_weak_particle_percent: float = 0.0
    directory: str | Path = "."
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.numerics, Numerics):
            raise TypeError("VPMCase.numerics must be a Numerics instance")
        if not isinstance(self.run, RunPlan):
            raise TypeError("VPMCase.run must be a RunPlan instance")
        if not isinstance(self.backup, Backup):
            raise TypeError("VPMCase.backup must be a Backup instance")
        if not isinstance(self.samplers, Samplers):
            raise TypeError("VPMCase.samplers must be a Samplers instance")
        if self.numerics.vlm is not None:
            self._validate_coupled_vlm_output_contract()
        percent = self.initial_weak_particle_percent
        if isinstance(percent, bool) or not isinstance(percent, Real):
            raise TypeError("initial_weak_particle_percent must be a real number")
        if not math.isfinite(percent) or percent < 0.0 or percent > 100.0:
            raise ValueError("initial_weak_particle_percent must be finite and between 0 and 100")
        object.__setattr__(self, "initial_weak_particle_percent", float(percent))
        object.__setattr__(self, "initial_conditions", tuple(self.initial_conditions))
        directory = Path(self.directory)
        if not str(directory).strip():
            raise ValueError("VPMCase.directory must be a non-empty path")
        object.__setattr__(self, "directory", directory)
        if self.name is not None:
            if not isinstance(self.name, str) or not self.name.strip():
                raise ValueError("VPMCase.name must be None or a non-empty string")
            object.__setattr__(self, "name", self.name.strip())

    def _validate_coupled_vlm_output_contract(self) -> None:
        """Keep attached VLM output under the owning VPM lifecycle.

        Standalone ``VLMSolver`` callers retain their own force
        logging and ``VLMSampler`` API. Once VLM is attached to a VPM case,
        accepted-step force/loading records are mandatory in the VPM-owned
        sample directory, while surface backup companions are emitted only by
        the VPM backup events.
        """
        vlm = self.numerics.vlm
        if vlm.logging_interval_steps != 1:
            raise ValueError(
                "attached VLM cannot configure logging_interval_steps; "
                "the VPM owner emits mandatory accepted-step samples"
            )
        if not vlm.sample_surface_forces:
            raise ValueError(
                "attached VLM requires sample_surface_forces=True; "
                "VLM scientific output is mandatory in the VPM sample path"
            )
        if any(surface.sample_forces is False for surface in vlm.surfaces):
            raise ValueError("attached VLM surfaces cannot opt out of mandatory scientific output")
        from ..io.sampling.vlm import VLMSampler

        if any(isinstance(sample, VLMSampler) for sample in self.samplers.samples):
            raise ValueError(
                "VLMSampler is standalone-only for coupled VPM cases; "
                "surface companions use the VPM backup lifecycle and VLM tables "
                "use the owner sample path"
            )
