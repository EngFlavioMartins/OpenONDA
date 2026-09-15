"""Public API for the OpenONDA incompressible FVM solver."""

from source.version import __version__

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
from .sampling.fields import LineSampler, SurfaceSampler
from .sampling.forces import (
    ForceSampler,
    IBMForceSampler,
    YPlusSampler,
)
from .solve import equation_solver
from .solve.diagnostics import StepDiagnostics
from .solve.linear_interface import LinearSolveResult

__author__ = "OpenONDA Project (translated from uFVM by CFD Group @ AUB)"

__all__ = [
    "AnalysisSnapshot",
    "BoundarySnapshot",
    "BackupConfig",
    "Backup",
    "BoundaryConfig",
    "ComputeConfig",
    "DiscretizationConfig",
    "FVMSetup",
    "FVMCase",
    "InitialFields",
    "FVMSolver",
    "FieldState",
    "ForceSampler",
    "IBMForceSampler",
    "LinearSolveResult",
    "LinearSolverConfig",
    "LineSampler",
    "LoggingConfig",
    "MaximumCourantTimeStep",
    "MeshQualityConfig",
    "Numerics",
    "Output",
    "OutputConfig",
    "PimpleControl",
    "RunAcceptanceLimits",
    "RunPlan",
    "RunSchedule",
    "Samplers",
    "StepDiagnostics",
    "SurfaceSampler",
    "TimeConfig",
    "TransportConfig",
    "TurbulenceConfig",
    "YPlusSampler",
    "__version__",
    "create_fvm_solver",
    "analyse_grid_study",
    "equation_solver",
    "io",
    "mesher",
    "update_grid_study",
]
