"""Incompressible finite-volume solver of OpenONDA.

A case is one immutable :class:`FVMCase`; :class:`FVMSolver` constructs and
runs it. Meshes are generated with :mod:`openonda.fvm.mesher`::

    import openonda.fvm as fvm
    import openonda.fvm.mesher as msh

    case = fvm.FVMCase(name="box", run=fvm.RunPlan(end_time=1.0), mesh=msh.CartesianMesher(...))
    with fvm.FVMSolver(case) as solver:
        solver.run()
"""

from openonda.core.schedules import EverySteps, EveryTime, FinalOnly
from openonda.core.simulation.forcing import VelocityBoundary, VelocityRamp
from openonda.core.status import RunStatus
from openonda.core.version import __version__

from . import io, mesher
from .config import (
    Backup,
    BoundaryConfig,
    ComputeConfig,
    DiscretizationConfig,
    FVMCase,
    InitialFields,
    LinearSolverConfig,
    LoggingConfig,
    MaximumCourantTimeStep,
    MeshQualityConfig,
    Numerics,
    Output,
    PimpleConfig,
    RunAcceptanceLimits,
    RunPlan,
    Samplers,
    TransportConfig,
    TurbulenceConfig,
)
from .core.solver import AcceptanceLimitError, FVMSolver
from .core.state import FieldState
from .diagnostics.fields import (
    compute_continuity_error,
    compute_enstrophy,
    compute_kinetic_energy,
)
from .diagnostics.step import StepDiagnostics
from .grid_study import analyse_grid_study, update_grid_study
from .io.analysis import AnalysisSnapshot, BoundarySnapshot
from .motion import (
    CompositeMotion,
    HeavingMotion,
    LinearPeriodicMotion,
    ManeuverMotion,
    MeshKinematics,
    MeshMotion,
    PitchingMotion,
    RigidPose,
    RotatingMotion,
    SmoothRampMotion,
    StaticMotion,
    TranslatingMotion,
)
from .sampling import ForceSampler, LineSampler, SurfaceSampler, YPlusSampler
from .solve.linear_interface import LinearSolveResult

__all__ = [
    "AcceptanceLimitError",
    "AnalysisSnapshot",
    "Backup",
    "BoundaryConfig",
    "BoundarySnapshot",
    "CompositeMotion",
    "ComputeConfig",
    "DiscretizationConfig",
    "EverySteps",
    "EveryTime",
    "FVMCase",
    "FVMSolver",
    "FieldState",
    "FinalOnly",
    "ForceSampler",
    "HeavingMotion",
    "InitialFields",
    "LineSampler",
    "LinearPeriodicMotion",
    "LinearSolveResult",
    "LinearSolverConfig",
    "LoggingConfig",
    "ManeuverMotion",
    "MaximumCourantTimeStep",
    "MeshKinematics",
    "MeshMotion",
    "MeshQualityConfig",
    "Numerics",
    "Output",
    "PimpleConfig",
    "PitchingMotion",
    "RigidPose",
    "RotatingMotion",
    "RunAcceptanceLimits",
    "RunPlan",
    "RunStatus",
    "Samplers",
    "SmoothRampMotion",
    "StaticMotion",
    "StepDiagnostics",
    "SurfaceSampler",
    "TranslatingMotion",
    "TransportConfig",
    "TurbulenceConfig",
    "VelocityBoundary",
    "VelocityRamp",
    "YPlusSampler",
    "__version__",
    "analyse_grid_study",
    "compute_continuity_error",
    "compute_enstrophy",
    "compute_kinetic_energy",
    "io",
    "mesher",
    "update_grid_study",
]
