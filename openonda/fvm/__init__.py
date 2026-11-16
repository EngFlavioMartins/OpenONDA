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

from . import mesher
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
    Numerics,
    Output,
    PimpleConfig,
    RunPlan,
    Samplers,
    TransportConfig,
    TurbulenceConfig,
    VelocityBoundary,
)
from .core.solver import AcceptanceLimitError, FVMSolver
from .diagnostics.fields import (
    compute_continuity_error,
)
from .grid_study import analyse_grid_study, update_grid_study
from .motion import ElasticMount, MeshMotion
from .sampling import ForceSampler, LineSampler, MotionSampler, SurfaceSampler, YPlusSampler

__all__ = [
    "AcceptanceLimitError",
    "Backup",
    "BoundaryConfig",
    "CompositeMotion",
    "ComputeConfig",
    "DiscretizationConfig",
    "ElasticMount",
    "EverySteps",
    "EveryTime",
    "FVMCase",
    "FVMSolver",
    "FinalOnly",
    "ForceSampler",
    "HeavingMotion",
    "InitialFields",
    "LineSampler",
    "LinearSolverConfig",
    "LoggingConfig",
    "ManeuverMotion",
    "MaximumCourantTimeStep",
    "MeshMotion",
    "MotionSampler",
    "Numerics",
    "Output",
    "PimpleConfig",
    "PitchingMotion",
    "RigidMotion",
    "RigidPose",
    "RotatingMotion",
    "RunPlan",
    "Samplers",
    "SmoothRampMotion",
    "StaticMotion",
    "SurfaceSampler",
    "TranslatingMotion",
    "TransportConfig",
    "TurbulenceConfig",
    "VelocityBoundary",
    "VelocityRamp",
    "YPlusSampler",
    "analyse_grid_study",
    "compute_continuity_error",
    "mesher",
    "update_grid_study",
]
