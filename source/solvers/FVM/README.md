# OpenONDA Finite Volume Method (FVM) Solver

**Python translation of uFVM (unstructured Finite Volume Method)**

## Overview

This module provides finite-volume operators and experimental incompressible
SIMPLE/PIMPLE solvers derived from the uFVM MATLAB/Octave implementation.

## Original Work

**uFVM - unstructured Finite Volume Method**
Developed by: CFD Group @ American University of Beirut
Year: 2018
Contact: cfd@aub.edu.lb

This Python implementation maintains the same structure, algorithms, and methodology as the original uFVM while adapting to Python's ecosystem (NumPy, SciPy).

## Features

### Verified building blocks
- OpenFOAM mesh I/O
- Geometric property computation
- Gradient schemes (Gauss linear)
- Diffusion term assembly
- Convection schemes (upwind, central, deferred correction)
- Time integration (Euler implicit/explicit)
- Sparse matrix assembly
- Linear system solvers (direct spsolve, iterative CG/BiCGSTAB/GMRES, pyAMG multigrid)
- Complete scalar transport equation solver

### Experimental integrated solvers
- Momentum equation assembly
- SIMPLE algorithm (steady-state)
- PIMPLE algorithm (transient; structured hexahedral path is most mature)
- Smagorinsky LES turbulence model
- Cavity flow handling
- Non-orthogonal correction (explicit k-vector; inactive by default)
- Force coefficient computation (Cd, Cl, Cz, Cm)

### Not implemented
- Compressible flow
- Multiphase flow

## Installation

```bash
from source.solvers.FVM import equation_solver
```

## Quick Start

### Scalar Transport Equation

```python
from source.solvers.FVM import equation_solver, mesh_io, topology, geometry

# Load mesh
mesh_data, geo_data, boundaries = load_mesh("path/to/case")

# Configure equation
config = {
    'type': 'steady',
    'terms': ['diffusion', 'convection'],
    'phi_initial': phi_0,
    'velocity': U,
    'gamma': 0.01,
    'rho': 1.0,
    'convection_scheme': 'deferred',
    'solver': 'spsolve'
}

# Solve
solution = equation_solver.solve_scalar_equation(
    config, mesh_data, geo_data, boundaries
)
```

## Modules

| Module | Description | uFVM Source |
|--------|-------------|-------------|
| `mesh_io` | OpenFOAM mesh readers | `src/mesh/cfdRead*.m` |
| `topology` | Element connectivity | `src/mesh/cfdProcess*.m` |
| `geometry` | Geometric properties | `src/mesh/cfdComputeGeometry.m` |
| `field_io` | Field file parser | `src/fields/cfdRead*.m` |
| `gradients` | Gradient computation | `src/fields/Gradient/*.m` |
| `diffusion` | Diffusion term | `src/assemble/Scalar/cfdAssembleDiffusionTerm.m` |
| `convection` | Convection term | `src/assemble/cfdAssembleConvectionTerm.m` |
| `time_integration` | Time stepping | `src/assemble/cfdAssembleTransientTerm*.m` |
| `matrix_assembly` | Matrix construction | `src/assemble/cfdAssembleIntoGlobalMatrix*.m` |
| `equation_solver` | Complete solver | `src/solve/cfdSolveEquation.m` |

## Verification

Core operators have regression tests for geometry, field I/O, gradients,
diffusion, matrix assembly, and manufactured solutions. The test tolerances and
supported cases are the authoritative verification record.

The integrated SIMPLE/PIMPLE solvers remain research backends. General
unstructured, device, and distributed-memory operation must pass the readiness
gates in `docs/plans/2026-07-fvm-3d-pimple-readiness-plan.md` before being
described as production-ready.

## Documentation

- `ACKNOWLEDGEMENTS.md` - Full attribution and credits
- `__init__.py` - Module documentation
- Individual module docstrings - Detailed API documentation

## Citation

If you use this code, please cite both:

1. **Original uFVM**:
   ```
   uFVM - unstructured Finite Volume Method Solver
   CFD Group, American University of Beirut, 2018
   ```

2. **This Translation**:
   ```
   OpenONDA FVM Module
   Python translation of uFVM, 2025
   ```

## License

Same license as original uFVM (check uFVM repository).

## Contact

**For this Python translation**: OpenONDA Project, 2025
**For original uFVM**: cfd@aub.edu.lb

## Acknowledgements

Special thanks to the uFVM development team at AUB for creating an excellent educational and research tool that made this translation possible.
