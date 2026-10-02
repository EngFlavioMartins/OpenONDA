# Choose and run a tutorial

Use the [FVM guide](fvm.md) for mesh-based viscous flow, the [VPM/VLM guide](vpm.md) for particle wakes and lifting surfaces, or the [coupling guide](coupling.md) for a resolved near-body region with a particle outer wake. Case guides below give the physical parameters, model choices, and comparisons.

## First run

After [installation](installation.md):

```bash
openonda tutorial list
openonda tutorial run fvm/taylor_green --workspace ./first-flow
openonda tutorial plot fvm/taylor_green --workspace ./first-flow
```

To edit inputs before running, copy a case without launching it:

```bash
openonda tutorial create vpm/vortex_ring ./ring-workspace
cd ring-workspace/tutorials/vpm/02_vortex_ring
```

Edit `setup.py`: geometry and lengths in m, velocities in m/s, kinematic viscosity in m²/s, times in s, mesh or particle spacing in m, and dimensionless model coefficients. Change resolution and timestep together when checking convergence. Each case's README explains its reference quantities.

From the case directory:

```bash
python setup.py       # Start or resume the default case.
./allcontinue.sh      # Start or resume the listed cases.
./allrun.sh           # Run the case's launcher; see cleanup behavior below.
./allplot.sh          # Plot PNG and PDF, where this launcher is provided.
```

**Most `allrun.sh` launchers clean generated results first.** The coupled cylinder, coupled cube, and their FVM reference launchers preserve results and resume instead. `openonda tutorial run` follows the selected case's `allrun.sh`. Use `allcontinue.sh` to preserve and extend a run. Increase the configured end time or total VPM steps to advance a completed case; see [continuation](continuation.md). `./allclean.sh` deletes generated output. Plotting is separate from simulation; use `./allplot.sh png` or `./allplot.sh pdf` for one format.

## FVM cases

| Tutorial identifier | Physical setup and reference |
| --- | --- |
| [`fvm/taylor_green`](../tutorials/fvm/taylor_green/README.md) | Periodic vortex decay; exact viscous solution. |
| [`fvm/boundary_layer`](../tutorials/fvm/boundary_layer/README.md) | Laminar flat plate; Blasius profiles and skin friction. |
| [`fvm/step_profile`](../tutorials/fvm/step_profile/README.md) | Backward-facing step; separation and reattachment. |
| [`fvm/cylinder_ibm`](../tutorials/fvm/cylinder_ibm/README.md) | Immersed cylinder; forces and wake shedding. |
| [`fvm/cube_flow`](../tutorials/fvm/cube_flow/README.md) | Square cylinder; wall-resolved wake. |
| [`fvm/airfoil_flow`](../tutorials/fvm/airfoil_flow/README.md) | Airfoil geometry and surface pressure. |
| [`fvm/cartesian_mesher`](../tutorials/fvm/cartesian_mesher/README.md) | STL-based mesh, refinement regions, and boundary patches. |

## VPM/VLM cases

| Tutorial identifier | Physical setup and reference |
| --- | --- |
| [`vpm/lamb_oseen_vortex`](../tutorials/vpm/01_lamb_oseen_vortex/README.md) | Viscous vortex, dipole, and merging; diffusion comparisons. |
| [`vpm/vortex_ring`](../tutorials/vpm/02_vortex_ring/README.md) | Toroidal particle distribution; ring motion and viscous decay. |
| [`vpm/vortex_interactions`](../tutorials/vpm/03_vortex_interactions/readme.md) | Interacting rings; stretching and stabilization variants. |
| [`vpm/flat_plate`](../tutorials/vpm/04_flat_plate/README.md) | Finite lifting surface; startup loading and shed wake. |
| [`vpm/delta_wing`](../tutorials/vpm/05_delta_wing/README.md) | Two heaving wings; unsteady loading and wake interaction. |
| [`vpm/rotor_flow`](../tutorials/vpm/06_rotor_flow/README.md) | Wind-turbine wake; blade loading and diffusion choices. |
| [`vpm/quadcopter`](../tutorials/vpm/07_quadcopter/README.md) | Four rotors in climb; wake interaction and thrust. |

## Coupled cases

| Tutorial identifier | Physical setup and reference |
| --- | --- |
| [`coupled_fvm_vpm/cylinder_shedding_flow`](../tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/README.md) | Cylinder wake at $Re=150$; free-slip span and particle renewal. |
| [`coupled_fvm_vpm/cube_flow`](../tutorials/coupled_fvm_vpm/02_cube_flow/README.md) | Square-cylinder wake; FVM–VPM transfer comparison. |
| [`coupled_fvm_vpm/naca4412_flow`](../tutorials/coupled_fvm_vpm/03_naca4412_flow/README.md) | Finite-span NACA 4412 at $Re=1000$ and $10^\circ$ incidence. |
| [`coupled_fvm_vpm/cylinder_shedding_flow/reference_flow`](../tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/reference_flow/README.md) | Larger FVM cylinder domain; force and profile reference. |
| [`coupled_fvm_vpm/cube_flow/reference_flow`](../tutorials/coupled_fvm_vpm/02_cube_flow/reference_flow/README.md) | Larger FVM square-cylinder domain; grid study and force reference. |

Reference directories have their own `setup.py` and launchers. Their meshes and domains differ from the coupled near-field meshes; compare the specified force normalization and averaging intervals. See each guide for grid sizes and arguments.

Set `cores` in an FVM setup to request MPI workers; the solver launches them. Open root `solution/*.pvd` files in ParaView or use case plotters for force histories and field comparisons. See [saved fields](solution_layout.md) and [visualization](visualization.md).
