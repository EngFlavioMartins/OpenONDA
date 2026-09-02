"""Public API for the OpenONDA incompressible FVM solver."""

from source.version import __version__

from . import io
from .config import (
    BackupConfig,
    BoundaryConfig,
    ComputeConfig,
    DiscretizationConfig,
    FVMSetup,
    LinearSolverConfig,
    LoggingConfig,
    MaximumCourantTimeStep,
    MeshMotionConfig,
    MeshQualityConfig,
    OutputConfig,
    PimpleControl,
    RunAcceptanceLimits,
    RunSchedule,
    TimeConfig,
    TransportConfig,
    TurbulenceConfig,
)
from .core.solver import FVMSolver
from .core.state import FieldState
from .factory import create_fvm_solver
from .mesh.adaptive_cartesian import AdaptiveCartesianMesher
from .mesh.boundary_layer import BoundaryLayerSpec
from .mesh.cartesian import (
    BoundaryLayers,
    BoxDomain,
    BoxPatches,
    BoxRefinement,
    CartesianMesher,
    CompositeSizeField,
    ConeRefinement,
    FeatureRefinement,
    LineRefinement,
    SphereRefinement,
    STLSurface,
)
from .mesh.cartesian.report import GenerationReport, SizeReport
from .mesh.cartesian.size_field import SizeField
from .mesh.general_body import GeneralBodyMesher
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
    "BoundaryLayers",
    "BackupConfig",
    "BoundaryLayerSpec",
    "BoxDomain",
    "BoxPatches",
    "BoundaryConfig",
    "BoxRefinement",
    "CartesianMesher",
    "CompositeSizeField",
    "ComputeConfig",
    "DiscretizationConfig",
    "FVMSetup",
    "FVMSolver",
    "FieldState",
    "ConeRefinement",
    "FeatureRefinement",
    "ForceSampler",
    "GeneralBodyMesher",
    "GenerationReport",
    "IBMForceSampler",
    "LinearSolveResult",
    "LineRefinement",
    "LinearSolverConfig",
    "LineSampler",
    "LoggingConfig",
    "MaximumCourantTimeStep",
    "MeshMotionConfig",
    "MeshQualityConfig",
    "OutputConfig",
    "PimpleControl",
    "RunAcceptanceLimits",
    "RunSchedule",
    "StepDiagnostics",
    "SphereRefinement",
    "STLSurface",
    "SizeField",
    "SizeReport",
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
