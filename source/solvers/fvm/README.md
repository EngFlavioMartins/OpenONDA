# Finite-volume solver

For examples, boundary conditions, units, output, and parallel execution,
see the [FVM guide](../../../docs/fvm.md). The public API is
`from openonda import fvm`.

The implementation is organized by numerical task:

- `config/`: fluid properties, boundary conditions, schemes, and run settings.
- `mesh/`: mesh generation, import, geometry, and partitioning.
- `fields/` and `schemes/`: field reconstruction and discrete operators.
- `solve/`: momentum and pressure equations, SIMPLE/PISO/PIMPLE, linear solvers.
- `core/`: solver state, time stepping, and parallel execution.
- `sampling/`: online sampling and replay of saved fields.
- `coupling/`: field exchange with the FVM–VPM driver.

See [tests and limitations](../../../docs/validation/fvm_qualification.md)
before choosing a numerical backend or interpreting a new case.
