"""Incompressible finite-volume solver of OpenONDA.

A case is one immutable :class:`FVMCase`; :class:`FVMSolver` constructs and
runs it. Meshes are generated with :mod:`openonda.fvm.mesher`::

    import numpy as np
    from openonda import fvm

    x = np.linspace(0.0, 2.0, 17)
    mesh = fvm.mesher.box_mesh_3d(x, x, np.array([0.0, 0.125]), empty_spanwise=True)
    case = fvm.FVMCase(name="box", run=fvm.RunPlan(end_time=1.0), mesh=mesh, boundaries=...)
    with fvm.FVMSolver(case) as solver:
        solver.run()

The first case of ``docs/fvm.md`` gives the boundaries and numerics in full.
"""

from openonda.core.forcing import VelocityRamp
from openonda.core.kinematics import (
    CompositeMotion,
    HeavingMotion,
    LinearPeriodicMotion,
    ManeuverMotion,
    PitchingMotion,
    RigidMotion,
    RigidPose,
    RotatingMotion,
    SmoothRampMotion,
    StaticMotion,
    TranslatingMotion,
)
from openonda.core.schedules import EverySteps, EveryTime, FinalOnly
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
    VelocityBoundary,
)
from .core.solver import AcceptanceLimitError, FVMSolver
from .core.state import FieldState
from .diagnostics.fields import (
    compute_continuity_error,
)
from .diagnostics.step import StepDiagnostics
from .grid_study import analyse_grid_study, update_grid_study
from .io.analysis import AnalysisSnapshot, BoundarySnapshot
from .motion import ElasticMount, MeshMotion
from .sampling import ForceSampler, LineSampler, MotionSampler, SurfaceSampler, YPlusSampler
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
    "ElasticMount",
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
    "MeshMotion",
    "MeshQualityConfig",
    "MotionSampler",
    "Numerics",
    "Output",
    "PimpleConfig",
    "PitchingMotion",
    "RigidMotion",
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
    "io",
    "mesher",
    "update_grid_study",
]
