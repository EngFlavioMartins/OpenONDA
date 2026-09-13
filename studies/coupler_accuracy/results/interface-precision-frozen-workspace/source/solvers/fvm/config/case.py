"""Public, physics-oriented construction objects for the FVM solver.

``FVMCase`` is the FVM counterpart of :class:`source.solvers.vpm.VPMCase`.
It keeps mesh provenance, numerical intent, run policy, and output ownership
at one construction boundary while the existing ``FVMSetup`` remains
available as the low-level configuration used by the numerical kernels.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import math
from numbers import Real
from pathlib import Path
from typing import Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .scheduling import RunSchedule
from .types import (
    BackupConfig,
    BoundaryConfig,
    ComputeConfig,
    DiscretizationConfig,
    FVMSetup,
    LinearSolverConfig,
    LoggingConfig,
    MaximumCourantTimeStep,
    MeshQualityConfig,
    OutputConfig,
    PimpleControl,
    RunAcceptanceLimits,
    TimeConfig,
    TransportConfig,
    TurbulenceConfig,
)

VelocityInput: TypeAlias = tuple[float, float, float] | list[float] | NDArray[np.float64]
MeshMapping: TypeAlias = dict[str, object]


class _BuildableMesh(Protocol):
    """Structural mesh builder accepted by :class:`FVMCase`."""

    def build(self) -> MeshMapping | tuple[MeshMapping, object]:
        """Return native mesh data, optionally paired with a build report."""

        ...


MeshSource: TypeAlias = (
    str
    | Path
    | MeshMapping
    | _BuildableMesh
    | Callable[[], MeshMapping | tuple[MeshMapping, object]]
)


@dataclass(frozen=True, slots=True)
class InitialFields:
    """Initial cell-centred FVM fields before boundary reconstruction.

    Parameters
    ----------
    velocity : tuple[float, float, float] or numpy.ndarray
        Uniform velocity in m/s with shape ``(3,)`` or one vector per fluid
        cell with shape ``(n_cells, 3)``. The cell count is checked after the
        mesh is materialized. Inputs are copied when stored.
    kinematic_pressure : float
        Uniform initial ``p/rho`` in m²/s². The solver reconstructs pressure
        boundary values from :class:`BoundaryConfig` after accepting the
        interior field.

    Notes
    -----
    Ghost cells and face fluxes are not accepted here. ``FVMSolver`` creates
    them from the mesh and boundary conditions and initializes all time-history
    levels consistently.
    """

    velocity: VelocityInput = (0.0, 0.0, 0.0)
    kinematic_pressure: float = 0.0

    def __post_init__(self) -> None:
        import numpy as np

        velocity = np.asarray(self.velocity, dtype=np.float64)
        # FVMSolver validates the per-cell count once the mesh is known.
        if velocity.shape != (3,) and (velocity.ndim != 2 or velocity.shape[1:] != (3,)):
            raise ValueError("InitialFields.velocity must have shape (3,) or (n_cells, 3)")
        if not np.all(np.isfinite(velocity)):
            raise ValueError("InitialFields.velocity must be finite")
        pressure = self.kinematic_pressure
        if isinstance(pressure, bool) or not isinstance(pressure, Real):
            raise TypeError("InitialFields.kinematic_pressure must be a real scalar")
        if not math.isfinite(float(pressure)):
            raise ValueError("InitialFields.kinematic_pressure must be finite")
        object.__setattr__(
            self, "velocity", tuple(velocity.tolist()) if velocity.ndim == 1 else velocity.copy()
        )
        object.__setattr__(self, "kinematic_pressure", float(pressure))


@dataclass(frozen=True, slots=True)
class Samplers:
    """Immutable collection of FVM sampler objects owned by one case.

    Parameters
    ----------
    samples : tuple[object, ...]
        Samplers implementing a callable ``sample(solver)`` method. The tuple
        is copied so later mutation of the caller's list cannot change case
        configuration. Individual sampler output paths and schedules remain
        owned by the sampler implementation.
    """

    samples: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        samples = tuple(self.samples)
        for sampler in samples:
            if not callable(getattr(sampler, "sample", None)):
                raise TypeError(f"FVM sampler {type(sampler).__name__} must implement sample()")
        object.__setattr__(self, "samples", samples)


@dataclass(frozen=True, slots=True)
class Numerics:
    """Grouped immutable numerical controls for an FVM case.

    Attributes
    ----------
    transport : TransportConfig
        Density and molecular kinematic viscosity in SI units.
    schemes : DiscretizationConfig
        Convection, gradient, and time-discretization choices.
    linear : LinearSolverConfig
        Momentum/pressure solver names, tolerances, and failure policy.
    coupling : PimpleControl
        SIMPLE/PISO/PIMPLE correction counts, relaxation, and acceptance
        tolerances.
    execution : ComputeConfig
        Operator, linear-algebra, MPI, and output execution backends.
    turbulence : TurbulenceConfig or None
        Optional LES closure. ``None`` is laminar/no explicit SGS model.
    acceptance : RunAcceptanceLimits
        Warning/abort thresholds evaluated on candidate states.
    logging : LoggingConfig
        Console, file, and logging cadence policy.
    """

    transport: TransportConfig = field(default_factory=TransportConfig)
    schemes: DiscretizationConfig = field(default_factory=DiscretizationConfig)
    linear: LinearSolverConfig = field(default_factory=LinearSolverConfig)
    coupling: PimpleControl = field(default_factory=PimpleControl)
    execution: ComputeConfig = field(default_factory=ComputeConfig)
    turbulence: TurbulenceConfig | None = None
    acceptance: RunAcceptanceLimits = field(default_factory=RunAcceptanceLimits)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self) -> None:
        expected = {
            "transport": TransportConfig,
            "schemes": DiscretizationConfig,
            "linear": LinearSolverConfig,
            "coupling": PimpleControl,
            "execution": ComputeConfig,
            "acceptance": RunAcceptanceLimits,
            "logging": LoggingConfig,
        }
        for name, cls in expected.items():
            if not isinstance(getattr(self, name), cls):
                raise TypeError(f"Numerics.{name} must be a {cls.__name__} instance")
        if self.turbulence is not None and not isinstance(self.turbulence, TurbulenceConfig):
            raise TypeError("Numerics.turbulence must be a TurbulenceConfig or None")


@dataclass(frozen=True, slots=True)
class RunPlan:
    """Finite physical-time plan for an FVM run.

    Attributes
    ----------
    start_time, end_time : float
        Physical bounds in seconds. `end_time` must exceed `start_time`.
    time_step_size : float
        Initial/fixed step size in seconds. The solver owns the evolving value
        when an adaptive maximum-Courant policy is configured.
    output_schedule : RunSchedule
        Accepted-step/time cadence for visualization output.
    adjustment : MaximumCourantTimeStep or None
        Optional solver-owned maximum-Courant step selection.
    initial_output, final_output : bool
        Whether framework-owned initial and terminal visualization events run.
    """

    end_time: float = 1.0
    time_step_size: float = 0.01
    start_time: float = 0.0
    output_schedule: RunSchedule = field(default_factory=lambda: RunSchedule(every_n_steps=10))
    adjustment: MaximumCourantTimeStep | None = None
    initial_output: bool = True
    final_output: bool = True

    def __post_init__(self) -> None:
        for name in ("start_time", "end_time", "time_step_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"RunPlan.{name} must be a real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"RunPlan.{name} must be finite")
            object.__setattr__(self, name, float(value))
        if self.time_step_size <= 0.0:
            raise ValueError("RunPlan.time_step_size must be positive")
        if self.end_time <= self.start_time:
            raise ValueError("RunPlan.end_time must be greater than start_time")
        if not isinstance(self.output_schedule, RunSchedule):
            raise TypeError("RunPlan.output_schedule must be a RunSchedule")
        if self.adjustment is not None and not isinstance(self.adjustment, MaximumCourantTimeStep):
            raise TypeError("RunPlan.adjustment must be a MaximumCourantTimeStep or None")
        if not isinstance(self.initial_output, bool) or not isinstance(self.final_output, bool):
            raise TypeError("RunPlan initial_output/final_output values must be booleans")


@dataclass(frozen=True, slots=True)
class FVMCase:
    """Complete immutable construction object for one standalone FVM run.

    Parameters
    ----------
    name : str
        Stable case name used in output filenames and solver metadata.
    mesh : MeshSource
        Native mesh dictionary, path to a supported saved mesh, or callable
        mesher returning one. Mesh coordinates use metres; the native topology
        contract is described in :mod:`source.solvers.fvm.mesh.geometry`.
    directory : str or pathlib.Path, default='.'
        Case root for solutions, samples, logs, and solver metadata.
    mesh_quality, numerics, boundaries, initial_conditions, run, output,
    samplers, backup : corresponding configuration objects
        Immutable case-owned policies. Boundary values are m/s, m²/s², m³/s,
        or m²/s according to their field; see their class docstrings.
    cores : int, default=1
        Requested execution size for legacy/coupled setup materialization.

    Notes
    -----
    Construction validates types and scalar ranges but does not allocate the
    solver mesh or device fields. :class:`FVMSolver` materializes the mesh and
    converts this case to the low-level :class:`FVMSetup` exactly once.
    """

    name: str
    mesh: MeshSource
    directory: str | Path = "."
    mesh_quality: MeshQualityConfig = field(default_factory=MeshQualityConfig)
    numerics: Numerics = field(default_factory=Numerics)
    boundaries: tuple[BoundaryConfig, ...] = ()
    initial_conditions: InitialFields = field(default_factory=InitialFields)
    run: RunPlan = field(default_factory=RunPlan)
    output: OutputConfig = field(default_factory=OutputConfig)
    samplers: Samplers = field(default_factory=Samplers)
    backup: BackupConfig = field(default_factory=BackupConfig)
    cores: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("FVMCase.name must be a non-empty string")
        if self.mesh is None:
            raise TypeError("FVMCase.mesh is required")
        if not isinstance(self.mesh_quality, MeshQualityConfig):
            raise TypeError("FVMCase.mesh_quality must be a MeshQualityConfig")
        if not isinstance(self.numerics, Numerics):
            raise TypeError("FVMCase.numerics must be a Numerics instance")
        if not isinstance(self.initial_conditions, InitialFields):
            raise TypeError("FVMCase.initial_conditions must be an InitialFields instance")
        if not isinstance(self.run, RunPlan):
            raise TypeError("FVMCase.run must be a RunPlan instance")
        if not isinstance(self.output, OutputConfig):
            raise TypeError("FVMCase.output must be an OutputConfig instance")
        if not isinstance(self.samplers, Samplers):
            raise TypeError("FVMCase.samplers must be a Samplers instance")
        if not isinstance(self.backup, BackupConfig):
            raise TypeError("FVMCase.backup must be a BackupConfig instance")
        if isinstance(self.cores, bool) or not isinstance(self.cores, int) or self.cores < 1:
            raise ValueError("FVMCase.cores must be a positive integer")
        object.__setattr__(self, "directory", Path(self.directory))
        object.__setattr__(self, "boundaries", tuple(self.boundaries))

    def to_setup(self) -> FVMSetup:
        """Materialize a mutable low-level setup for the numerical core.

        Returns
        -------
        FVMSetup
            A new configuration object containing copied case policy values,
            legacy field names, and the time/output controls consumed by the
            solver factory. It performs no mesh I/O and does not mutate this
            immutable case.
        """
        fields = self.initial_conditions
        return FVMSetup(
            case_name=self.name,
            cores=self.cores,
            mesh=self.mesh_quality,
            execution=self.numerics.execution,
            output=self.output,
            acceptance=self.numerics.acceptance,
            logging=self.numerics.logging,
            backup=self.backup,
            time=TimeConfig(
                time_step_size=self.run.time_step_size,
                start_time=self.run.start_time,
                end_time=self.run.end_time,
                output_schedule=self.run.output_schedule,
                adjustment=self.run.adjustment,
            ),
            schemes=self.numerics.schemes,
            linear=self.numerics.linear,
            pimple=self.numerics.coupling,
            transport=self.numerics.transport,
            boundaries=list(self.boundaries),
            samplers=self.samplers.samples,
            turbulence=self.numerics.turbulence,
            initial_velocity=fields.velocity,
            initial_kinematic_pressure=fields.kinematic_pressure,
        )


# Short names intentionally mirror the VPM public namespace.  They are
# aliases for output/restart policy objects, not alternate solver paths.
Output = OutputConfig
Backup = BackupConfig


__all__ = [
    "Backup",
    "FVMCase",
    "InitialFields",
    "Numerics",
    "Output",
    "RunPlan",
    "Samplers",
]
