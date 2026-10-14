"""Immutable case definition of one FVM simulation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

from openonda.core.schedules import EveryTime
from openonda.core.simulation.forcing import VelocityBoundary

from ..motion.kinematics import MeshMotion
from ._validation import finite_real, vector_or_field
from .boundaries import BoundaryConfig
from .mesh_quality import MeshQualityConfig
from .numerics import Numerics
from .output import Backup, LoggingConfig, Output, Samplers

VelocityInput: TypeAlias = tuple[float, float, float] | list[float] | NDArray[np.float64]
MeshMapping: TypeAlias = dict[str, object]


class BuildableMesh(Protocol):
    """Mesh generator accepted by :class:`FVMCase`."""

    def build(self) -> MeshMapping | tuple[MeshMapping, object]:
        """Return native mesh data, optionally paired with a build report."""
        ...


MeshSource: TypeAlias = (
    str
    | Path
    | MeshMapping
    | BuildableMesh
    | Callable[[], MeshMapping | tuple[MeshMapping, object]]
)


@dataclass(frozen=True, slots=True)
class InitialFields:
    """Initial cell-centred velocity and kinematic pressure.

    Parameters
    ----------
    velocity : array-like, default=(0, 0, 0)
        Uniform ``(3,)`` or per-cell ``(n_cells, 3)`` velocity in m/s. The
        cell count is checked once the mesh is constructed.
    kinematic_pressure : float, default=0.0
        Uniform initial ``p/rho`` in m²/s².

    Notes
    -----
    The solver reconstructs boundary ghost values, face fluxes and the time
    history from these interior fields.
    """

    velocity: VelocityInput = (0.0, 0.0, 0.0)
    kinematic_pressure: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "velocity", vector_or_field("InitialFields.velocity", self.velocity)
        )
        object.__setattr__(
            self,
            "kinematic_pressure",
            finite_real("InitialFields.kinematic_pressure", self.kinematic_pressure),
        )


@dataclass(frozen=True, slots=True)
class RunPlan:
    """Physical-time span of an FVM run.

    Parameters
    ----------
    end_time : float
        Physical end time in s; greater than ``start_time``.
    start_time : float, default=0.0
        Physical time of the initial state in s.
    initial_output : bool, default=True
        Write the initial visualization frame and the initial samples.
    final_output : bool, default=True
        Write the final visualization frame and the final-only samples.
    """

    end_time: float
    start_time: float = 0.0
    initial_output: bool = True
    final_output: bool = True

    def __post_init__(self) -> None:
        for name in ("end_time", "start_time"):
            object.__setattr__(self, name, finite_real(f"RunPlan.{name}", getattr(self, name)))
        if self.end_time <= self.start_time:
            raise ValueError("RunPlan.end_time must be greater than start_time")
        for name in ("initial_output", "final_output"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"RunPlan.{name} must be a boolean")


@dataclass(frozen=True, slots=True, kw_only=True)
class FVMCase:
    """Complete immutable definition of one FVM simulation.

    Parameters
    ----------
    name : str
        Case name used in file names and metadata.
    directory : str or pathlib.Path, default='.'
        Case root. Results go to ``solution/`` and ``samples/`` below it; a
        generated mesh is kept as ``assets/<name>_mesh.npz``.
    variant : str or None, default=None
        Optional sub-directory of ``solution/`` and ``samples/`` for one
        variant of the case.
    numerics : Numerics
        Every setting that changes the computed solution.
    initial_conditions : InitialFields
        Initial velocity and kinematic pressure.
    run : RunPlan
        Physical-time span and initial/final output switches.
    output : Output
        Visualization frames.
    samplers : Samplers
        Scientific samplers.
    backup : Backup
        Restart backups.
    logging : LoggingConfig
        Log records.
    mesh : MeshSource
        Native mesh mapping, a ``.npz`` or Gmsh ``.msh`` file relative to
        ``directory``, or a mesher (an object with ``build()`` or a callable)
        returning the mesh mapping. Coordinates are in m.
    mesh_quality : MeshQualityConfig
        Mesh-quality limits checked at construction.
    boundaries : tuple[BoundaryConfig, ...]
        Boundary condition of each configured patch.
    velocity_boundaries : tuple[VelocityBoundary, ...]
        Time-dependent velocities imposed on named patches.
    cores : int, default=1
        Number of MPI ranks of the run.
    mesh_motion : MeshMotion or None, default=None
        Prescribed rigid-body motion of the whole mesh; ``None`` is static.
    """

    name: str
    directory: str | Path = "."
    variant: str | None = None
    numerics: Numerics = field(default_factory=Numerics)
    initial_conditions: InitialFields = field(default_factory=InitialFields)
    run: RunPlan
    output: Output = field(default_factory=Output)
    samplers: Samplers = field(default_factory=Samplers)
    backup: Backup = field(default_factory=Backup)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    mesh: MeshSource
    mesh_quality: MeshQualityConfig = field(default_factory=MeshQualityConfig)
    boundaries: tuple[BoundaryConfig, ...] = ()
    velocity_boundaries: tuple[VelocityBoundary, ...] = ()
    cores: int = 1
    mesh_motion: MeshMotion | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("FVMCase.name must be a non-empty string")
        if self.variant is not None and (
            not isinstance(self.variant, str)
            or not self.variant.strip()
            or Path(self.variant).name != self.variant
        ):
            raise ValueError("FVMCase.variant must be None or one directory name")
        if self.mesh is None:
            raise TypeError("FVMCase.mesh is required")
        expected = {
            "numerics": Numerics,
            "initial_conditions": InitialFields,
            "run": RunPlan,
            "output": Output,
            "samplers": Samplers,
            "backup": Backup,
            "logging": LoggingConfig,
            "mesh_quality": MeshQualityConfig,
        }
        for name, cls in expected.items():
            if not isinstance(getattr(self, name), cls):
                raise TypeError(f"FVMCase.{name} must be a {cls.__name__} instance")
        if isinstance(self.cores, bool) or not isinstance(self.cores, int) or self.cores < 1:
            raise ValueError("FVMCase.cores must be a positive integer")
        if self.mesh_motion is not None:
            if not isinstance(self.mesh_motion, MeshMotion):
                raise TypeError("FVMCase.mesh_motion must be a MeshMotion or None")
            if self.numerics.pimple.algorithm == "simple":
                raise ValueError(
                    "A moving mesh requires the transient 'piso' or 'pimple' algorithm"
                )
        object.__setattr__(self, "directory", Path(self.directory))
        boundaries = tuple(self.boundaries)
        if any(not isinstance(item, BoundaryConfig) for item in boundaries):
            raise TypeError("FVMCase.boundaries must contain BoundaryConfig objects")
        names = [boundary.name for boundary in boundaries]
        if len(names) != len(set(names)):
            raise ValueError("FVMCase.boundaries must not repeat a patch name")
        object.__setattr__(self, "boundaries", boundaries)
        velocity_boundaries = tuple(self.velocity_boundaries)
        bound_patches: list[str] = []
        for condition in velocity_boundaries:
            if not isinstance(condition, VelocityBoundary):
                raise TypeError("FVMCase.velocity_boundaries must contain VelocityBoundary objects")
            if set(condition.patches) - set(names):
                raise ValueError("FVMCase.velocity_boundaries refers to an unconfigured patch")
            bound_patches.extend(condition.patches)
        if len(set(bound_patches)) != len(bound_patches):
            raise ValueError("Each patch may have only one velocity history")
        object.__setattr__(self, "velocity_boundaries", velocity_boundaries)
        if self.numerics.pimple.algorithm == "simple" and any(
            isinstance(schedule, EveryTime)
            for schedule in (self.output.schedule, self.backup.schedule)
        ):
            raise ValueError(
                "The steady 'simple' algorithm has no physical time; use EverySteps schedules"
            )

    @property
    def solution_directory(self) -> Path:
        """Return ``directory/solution`` or ``directory/solution/<variant>``."""
        root = self.directory / "solution"
        return root if self.variant is None else root / self.variant

    @property
    def samples_directory(self) -> Path:
        """Return ``directory/samples`` or ``directory/samples/<variant>``."""
        root = self.directory / "samples"
        return root if self.variant is None else root / self.variant

    @property
    def mesh_path(self) -> Path:
        """Return the stored native mesh ``directory/assets/<name>_mesh.npz``."""
        return self.directory / "assets" / f"{self.name}_mesh.npz"
