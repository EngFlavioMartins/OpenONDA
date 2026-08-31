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
    RunAcceptanceLimits,
    TimeConfig,
    TransportConfig,
    TurbulenceConfig,
)
from .core.solver import FVMSolver
from .core.state import FieldState
from .factory import create_fvm_solver
from .mesh.adaptive_cartesian import (
    AdaptiveCartesianMesher,
    BoundaryLayerSpec,
    BoxRefinement,
    ExplicitCylinderGridMesher,
)
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
    "AdaptiveCartesianMesher",
    "BoundaryLayerSpec",
    "BoundaryConfig",
    "BoxRefinement",
    "ComputeConfig",
    "DiscretizationConfig",
    "FVMSetup",
    "FVMSolver",
    "FieldState",
    "ExplicitCylinderGridMesher",
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
    "RunAcceptanceLimits",
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
