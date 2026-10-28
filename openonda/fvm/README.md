# Finite-volume solver

For examples, boundary conditions, units, output, and parallel execution,
see the [FVM guide](../../docs/fvm.md). The public API is
`from openonda import fvm`.

The implementation is organized by numerical task:

- `config/`: the case (`FVMCase`), numerical settings, boundary conditions,
  output and turbulence settings.
- `mesh/`: mesh generation, import, geometry, and partitioning.
- `fields/` and `schemes/`: field reconstruction and discrete operators.
- `solve/`: momentum and pressure equations, SIMPLE/PISO/PIMPLE, linear solvers.
- `core/`: the solver, its state, time stepping, and parallel execution.
- `diagnostics/`: derived fields (Courant number, vorticity, forces, y+) and
  step records.
- `motion/`: rigid motion of the whole mesh (`MeshMotion`, `RigidMeshMotion`); the motion classes are the shared ones in `openonda.core.kinematics`.
- `sampling/`: force, y+, line and surface samplers.
- `coupling/`: field gathers and boundary-condition setters for the FVM–VPM coupler.

See [tests and limitations](../../docs/validation/fvm_qualification.md)
before choosing a numerical backend or interpreting a new case.
