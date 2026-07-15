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
    FVMConfig,
    MeshConfig,
    RunAcceptancePolicy,
    SolverParams,
    TimeConfig,
    TransportConfig,
    TurbulenceConfig,
)
from .core.solver import Solver
from .solve import equation_solver
from .solve.contracts import StepDiagnostics
from .solve.linear_interface import LinearSolveResult

__author__ = "OpenONDA Project (translated from uFVM by CFD Group @ AUB)"

__all__ = [
    "Solver",
    "__version__",
    "FVMConfig",
    "ExecutionConfig",
    "MeshConfig",
    "RunAcceptancePolicy",
    "TimeConfig",
    "SolverParams",
    "TransportConfig",
    "BoundaryConfig",
    "DynamicMeshConfig",
    "TurbulenceConfig",
    "equation_solver",
    "LinearSolveResult",
    "StepDiagnostics",
    "io",
]
