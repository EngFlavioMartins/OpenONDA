# Finite-volume solver

For examples, boundary conditions, units, output, and parallel execution,
see the [FVM guide](../../docs/fvm.md). The public API is
`from openonda import fvm`.

The implementation is organized by numerical task:

- `config/`: the case (`FVMCase`), numerical settings, boundary conditions,
  output and turbulence settings.
- `mesh/`: mesh generation, import, geometry, and partitioning.
- `fields/`: gradients and boundary-face values of the fields.
- `schemes/`: TVD flux limiters and boundary-condition checks.
- `assemble/`: the momentum equation and its convection, diffusion and
  time-integration terms, and the sparse matrix assembly.
- `solve/`: the PIMPLE algorithm, the pressure equation, and linear solvers.
- `turbulence/`: subgrid-scale eddy-viscosity models.
- `core/`: the solver, its state, time stepping, and parallel execution.
- `io/`: backups, ParaView output, run logs and analysis files.
- `utils/`: row-wise vector algebra.
- `diagnostics/`: derived fields (Courant number, vorticity, forces, y+) and
  step records.
- `motion/`: motion of the whole mesh, prescribed (`MeshMotion` with a motion from `openonda.core.kinematics`) or elastically mounted (`ElasticMount`).
- `sampling/`: force, y+, line and surface samplers.
- `coupling/`: field gathers and boundary-condition setters for the FVM–VPM coupler.
