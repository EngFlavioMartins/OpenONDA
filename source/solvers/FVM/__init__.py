#!/usr/bin/env python3
"""
OpenONDA Finite Volume Method (FVM) Solver
===========================================

Python implementation of finite volume method for CFD simulations.

**Translated from uFVM**:
This module is a Python translation of the uFVM (unstructured Finite Volume Method)
MATLAB/Octave solver developed by the CFD Group @ AUB (2018).

Original uFVM: https://github.com/cfd-aub/uFVM
Contact: cfd@aub.edu.lb

**Translation**: 2025, OpenONDA Project
**License**: Same as uFVM (check original repository)

**Acknowledgements**:
- uFVM developers at American University of Beirut
- CFD Group @ AUB for the original MATLAB implementation
- OpenFOAM project for mesh format specifications

**Modules**:
- mesh_io: OpenFOAM mesh file readers
- topology: Element connectivity computation
- geometry: Geometric property calculation
- field_io: OpenFOAM field file parser
- gradients: Gradient computation schemes
- diffusion: Diffusion term assembly
- convection: Convection term assembly
- time_integration: Time stepping schemes
- matrix_assembly: Sparse matrix construction
- equation_solver: Complete equation solver
- momentum: Momentum equation assembly
- simple_solver: SIMPLE algorithm for NS equations
- cavity_utils: Cavity flow utilities

**Usage**:
```python
from source.solvers.FVM import equation_solver

# Solve scalar transport equation
config = {
    'type': 'steady',
    'terms': ['diffusion', 'convection'],
    'phi_initial': phi_0,
    'velocity': U,
    'gamma': 0.01,
    'solver': 'spsolve'
}

solution = equation_solver.solve_scalar_equation(
    config, mesh_data, geo_data, boundaries
)
```

**Status**:
- Scalar transport: regression-tested reference path
- SIMPLE/PIMPLE: experimental incompressible research solvers
- Structured hexahedral meshes: most mature path
- General unstructured/device/distributed execution: under development
- Compressible flow: not implemented

**Verification**:
Core operators and selected integrated cases have regression coverage. See the
readiness plan for the remaining production gates.
"""

__version__ = "1.0.0"
__author__ = "OpenONDA Project (translated from uFVM by CFD Group @ AUB)"
from . import io
from .config.types import (
    BoundaryConfig,
    ExecutionConfig,
    FVMConfig,
    MeshConfig,
    SolverParams,
    TimeConfig,
    TransportConfig,
    TurbulenceConfig,
)
from .core.solver import Solver

__all__ = [
    "Solver",
    "FVMConfig",
    "ExecutionConfig",
    "MeshConfig",
    "TimeConfig",
    "SolverParams",
    "TransportConfig",
    "BoundaryConfig",
    "TurbulenceConfig",
    "mesh_io",
    "topology",
    "geometry",
    "field_io",
    "gradients",
    "field_writer",
    "convection",
    "diffusion",
    "momentum",
    "matrix_assembly",
    "time_integration",
    "simple_solver",
    "pimple_solver",
    "equation_solver",
    "cavity_utils",
    "io",
]
