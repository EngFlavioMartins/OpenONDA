"""Public API for the OpenONDA incompressible FVM solver.

Mesh construction lives in :mod:`openonda.fvm.mesher`, separate from the
solver configuration::

    import openonda.fvm as fvm
    import openonda.fvm.mesher as msh

    mesh = msh.CartesianMesher(
        domain=msh.BoxDomain(...),
        surfaces=(msh.STLSurface(...),),
    )
"""

from openonda.core.version import __version__

from . import io, mesher
from .config import (
    Backup,
    BackupConfig,
    BoundaryConfig,
    ComputeConfig,
    DiscretizationConfig,
    FVMCase,
    FVMSetup,
    InitialFields,
    LinearSolverConfig,
    LoggingConfig,
    MaximumCourantTimeStep,
    MeshQualityConfig,
    Numerics,
    Output,
    OutputConfig,
    PimpleControl,
    RunAcceptanceLimits,
    RunPlan,
    RunSchedule,
    Samplers,
    TimeConfig,
    TransportConfig,
    TurbulenceConfig,
)
from .core.solver import FVMSolver
from .core.state import FieldState
from .factory import create_fvm_solver
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
    RotatingMotion,
    SmoothRampMotion,
    StaticMotion,
    TranslatingMotion,
)
from .sampling.fields import LineSampler, SurfaceSampler
from .sampling.forces import (
    ForceSampler,
    YPlusSampler,
)
from .solve.diagnostics import StepDiagnostics
from .solve.linear_interface import LinearSolveResult

__author__ = "OpenONDA Project (translated from uFVM by CFD Group @ AUB)"

from openonda.core.simulation.forcing import VelocityBoundary, VelocityRamp

from .fields.diagnostics import compute_continuity_error, compute_enstrophy, compute_kinetic_energy

__all__ = [
    "VelocityBoundary",
    "VelocityRamp",
    "compute_continuity_error",
    "compute_enstrophy",
    "compute_kinetic_energy",
    "AnalysisSnapshot",
    "BoundarySnapshot",
    "BackupConfig",
    "Backup",
    "BoundaryConfig",
    "CompositeMotion",
    "ComputeConfig",
    "DiscretizationConfig",
    "FVMSetup",
    "FVMCase",
    "InitialFields",
    "FVMSolver",
    "FieldState",
    "ForceSampler",
    "HeavingMotion",
    "LinearPeriodicMotion",
    "LinearSolveResult",
    "LinearSolverConfig",
    "LineSampler",
    "LoggingConfig",
    "ManeuverMotion",
    "MaximumCourantTimeStep",
    "MeshKinematics",
    "MeshMotion",
    "MeshQualityConfig",
    "Numerics",
    "Output",
    "OutputConfig",
    "PimpleControl",
    "PitchingMotion",
    "RotatingMotion",
    "RunAcceptanceLimits",
    "RunPlan",
    "RunSchedule",
    "Samplers",
    "SmoothRampMotion",
    "StaticMotion",
    "StepDiagnostics",
    "SurfaceSampler",
    "TimeConfig",
    "TranslatingMotion",
    "TransportConfig",
    "TurbulenceConfig",
    "YPlusSampler",
    "__version__",
    "create_fvm_solver",
    "analyse_grid_study",
    "io",
    "mesher",
    "update_grid_study",
]
