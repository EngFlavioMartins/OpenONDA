#!/usr/bin/env python3
"""Public API for the OpenONDA incompressible finite-volume solver.

Internal operators remain available through their package paths, for example
``source.solvers.FVM.mesh.geometry``.  They are intentionally not re-exported
here so the stable API is small and unambiguous.
"""

from source.version import __version__

from . import io
from .config.types import (
    BoundaryConfig,
    DynamicMeshConfig,
    ExecutionConfig,
    FVMSetup,
    LinearSolverConfig,
    LogConfig,
    MeshConfig,
    OutputSetup,
    PimpleControl,
    RunAcceptancePolicy,
    SchemesConfig,
    TimeConfig,
    TransportConfig,
    TurbulenceConfig,
    solver_configs_from_case,
)
from .core.solver import Solver
from .core.state import FieldState
from .factory import setup_fvm_solver
from .mesh.adaptive_cartesian import AdaptiveCartesianMesher, BoxRefinement
from .sampling.fields import LineSampler, SurfaceSampler
from .sampling.forces import ForceSampler, IBMForceSampler, YPlusSampler
from .solve import equation_solver
from .solve.contracts import StepDiagnostics
from .solve.linear_interface import LinearSolveResult

__author__ = "OpenONDA Project (translated from uFVM by CFD Group @ AUB)"

__all__ = [
    "Solver",
    "__version__",
    "FVMSetup",
    "setup_fvm_solver",
    "AdaptiveCartesianMesher",
    "BoxRefinement",
    "ExecutionConfig",
    "MeshConfig",
    "LogConfig",
    "OutputSetup",
    "RunAcceptancePolicy",
    "TimeConfig",
    "SchemesConfig",
    "LinearSolverConfig",
    "PimpleControl",
    "ForceSampler",
    "LineSampler",
    "SurfaceSampler",
    "IBMForceSampler",
    "YPlusSampler",
    "solver_configs_from_case",
    "TransportConfig",
    "BoundaryConfig",
    "DynamicMeshConfig",
    "TurbulenceConfig",
    "equation_solver",
    "LinearSolveResult",
    "StepDiagnostics",
    "FieldState",
    "io",
]
