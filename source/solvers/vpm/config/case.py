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
from .health import HealthLimits
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
        if self.time_step_size <= 0.0:
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
        supported_kernels = getattr(self.induction, "supported_kernels", None)
        if supported_kernels is not None and kernel not in supported_kernels:
            raise ValueError(
                f"{type(self.induction).__name__} does not support particle_kernel={kernel}; "
                f"supported kernels: {sorted(supported_kernels)}"
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
