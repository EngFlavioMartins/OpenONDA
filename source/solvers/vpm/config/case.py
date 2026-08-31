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
from .advection import AdvectionConfig
from .artifacts import Backup, Samplers
from .constants import DEFAULT_CUTOFF_RADIUS_FACTOR, DEFAULT_TIME_STEP, MAX_N_PARTICLES
from .diagnostics import DiagnosticsConfig
from .health import HealthLimits
from .setup import PanelBodySetup, VPMSetup
from .stabilization import StabilizationConfig
from .stretching import StretchingConfig
from .turbulence import TurbulenceConfig
from .velocity import VelocityConfig
from .viscous import ViscousConfig

if TYPE_CHECKING:
    from ..initialization import InitialCondition


@dataclass(frozen=True, slots=True)
class Numerics:
    """Immutable numerical and physical controls for a VPM case.

    Runtime clock values and output destinations are intentionally absent.
    ``_to_runtime_setup`` is private framework plumbing that adapts this typed
    public object to the still-internal numerical engine configuration.

    ``time_step_size`` is the accepted-step duration in seconds and must be
    positive.  ``precision`` selects the particle compute dtype (``"f32"`` or
    ``"f64"``); f64 cannot currently use treecode velocity evaluation.
    ``particle_kernel`` and ``velocity`` must be compatible, while coupled
    RK2/RK3 integration requires matching advection and stretching schemes.
    Invalid combinations raise :class:`ValueError` during construction.
    """

    time_step_size: float = DEFAULT_TIME_STEP
    time_integration: Literal["FRACTIONAL", "COUPLED"] = "FRACTIONAL"
    axisymmetric_no_swirl_axis: Literal["x", "y", "z"] | None = None
    advection: AdvectionConfig = field(default_factory=AdvectionConfig)
    stretching: StretchingConfig = field(default_factory=StretchingConfig.transposed)
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
    velocity: VelocityConfig | None = None
    panel_solver: object | None = None
    bodies: tuple[PanelBodySetup, ...] = ()
    domain_bounds: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        # The mature validator remains the single source of numerical truth
        # while the compute engine is being separated from its legacy setup
        # carrier.  No runtime or output state is retained by Numerics.
        self._to_runtime_setup()

    def _to_runtime_setup(
        self, backup: Backup | None = None, samplers: Samplers | None = None
    ) -> VPMSetup:
        """Build the private engine configuration for this immutable case."""
        backup = Backup() if backup is None else backup
        samplers = Samplers() if samplers is None else samplers
        return VPMSetup(
            time_step_size=self.time_step_size,
            time_integration=self.time_integration,
            axisymmetric_no_swirl_axis=self.axisymmetric_no_swirl_axis,
            advection=self.advection,
            stretching=self.stretching,
            viscous=self.viscous,
            turbulence=self.turbulence,
            stabilization=self.stabilization,
            vlm=self.vlm,
            particle_kernel=self.particle_kernel,
            max_n_particles=self.max_n_particles,
            max_evaluation_points=self.max_evaluation_points,
            compute_device=self.compute_device,
            precision=self.precision,
            write_precision=self.write_precision,
            random_seed=self.random_seed,
            device_memory_fraction=self.device_memory_fraction,
            debug_mode=self.debug_mode,
            diagnostics=self.diagnostics,
            health_limits=self.health_limits,
            backup=backup,
            samplers=samplers,
            cutoff_radius_factor=self.cutoff_radius_factor,
            freestream_velocity=self.freestream_velocity,
            verbose=self.verbose,
            velocity=self.velocity,
            panel_solver=self.panel_solver,
            bodies=self.bodies,
            domain_bounds=self.domain_bounds,
        )


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
    backup defaults to ``True`` to preserve a restart point after successful
    runs; it does not schedule scientific samplers.
    """

    steps: int
    initial_samples: bool = True
    final_backup: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.steps, bool) or not isinstance(self.steps, int):
            raise TypeError("RunPlan.steps must be an integer")
        if self.steps < 0:
            raise ValueError("RunPlan.steps must be non-negative")


@dataclass(slots=True)
class RestartState:
    """Mutable physical clock restored from a numerical backup.

    ``time`` is in seconds and ``step`` is the number of accepted steps.  The
    state is runtime-owned so it cannot accidentally be serialized as part of
    a new case's numerical construction.

    ``time`` is finite physical time in seconds and ``step`` is a non-negative
    accepted-step counter.  The runtime restores this object from numerical
    backup data; users construct it only for advanced interactive runs.
    """

    time: float = 0.0
    step: int = 0

    def __post_init__(self) -> None:
        if self.time < 0.0:
            raise ValueError("RestartState.time must be non-negative")
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
    directory raise :class:`TypeError` or :class:`ValueError`.
    """

    numerics: Numerics
    initial_conditions: tuple[InitialCondition, ...] = ()
    backup: Backup = field(default_factory=Backup)
    samplers: Samplers = field(default_factory=Samplers)
    run: RunPlan = field(default_factory=lambda: RunPlan(steps=0))
    initial_weak_particle_percent: float = 0.0
    directory: str | Path = "."

    def __post_init__(self) -> None:
        if not isinstance(self.numerics, Numerics):
            raise TypeError("VPMCase.numerics must be a Numerics instance")
        if not isinstance(self.run, RunPlan):
            raise TypeError("VPMCase.run must be a RunPlan instance")
        if not isinstance(self.backup, Backup):
            raise TypeError("VPMCase.backup must be a Backup instance")
        if not isinstance(self.samplers, Samplers):
            raise TypeError("VPMCase.samplers must be a Samplers instance")
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
