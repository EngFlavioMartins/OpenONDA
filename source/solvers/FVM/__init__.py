"""Public API for the OpenONDA incompressible FVM solver."""

from source.version import __version__

from . import io
from .config import (
    BoundaryConfig,
    ComputeConfig,
    DiscretizationConfig,
    FVMSetup,
    LinearSolverConfig,
    LoggingConfig,
    MeshMotionConfig,
    MeshQualityConfig,
    OutputConfig,
    PimpleControl,
    RunAcceptancePolicy,
    TimeConfig,
    TransportConfig,
    TurbulenceConfig,
)
from .core.solver import FVMSolver
from .core.state import FieldState
from .factory import create_fvm_solver
from .mesh.adaptive_cartesian import (
    AdaptiveCartesianMesher,
    BoxRefinement,
)
from .sampling.fields import LineSampler, SurfaceSampler
from .sampling.forces import (
    ForceSampler,
    IBMForceSampler,
    YPlusSampler,
)
from .solve import equation_solver
from .solve.contracts import StepDiagnostics
from .solve.linear_interface import LinearSolveResult

__author__ = "OpenONDA Project (translated from uFVM by CFD Group @ AUB)"

__all__ = [
    "AdaptiveCartesianMesher",
    "BoundaryConfig",
    "BoxRefinement",
    "ComputeConfig",
    "DiscretizationConfig",
    "FVMSetup",
    "FVMSolver",
    "FieldState",
    "ForceSampler",
    "IBMForceSampler",
    "LinearSolveResult",
    "LinearSolverConfig",
    "LineSampler",
    "LoggingConfig",
    "MeshMotionConfig",
    "MeshQualityConfig",
    "OutputConfig",
    "PimpleControl",
    "RunAcceptancePolicy",
    "StepDiagnostics",
    "SurfaceSampler",
    "TimeConfig",
    "TransportConfig",
    "TurbulenceConfig",
    "YPlusSampler",
    "__version__",
    "create_fvm_solver",
    "equation_solver",
    "io",
]
