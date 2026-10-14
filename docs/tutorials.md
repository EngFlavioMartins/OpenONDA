# Learning with the tutorials

The tutorials are the main way to learn OpenONDA. As in OpenFOAM, each case is a small, self-contained directory that sets up one physical problem. Run it as it is, compare with the reference it ships, then change it. Work through them roughly in the order below; each step adds a few new solver features. The case directories are under `tutorials/fvm`, `tutorials/vpm` and `tutorials/coupled_fvm_vpm`.

## Anatomy of a case

```text
case/
├── setup.py        physical constants, mesh or particles, solver settings
├── allrun.sh       clean, then run the case (and its variants)
├── allcontinue.sh  resume from the latest backup
├── allplot.sh      write the figures
├── allclean.sh     delete solution/, samples/ and figures/
├── assets/         geometry, initial conditions, plot scripts, ParaView state
└── README.md       the physics, the parameters and what to look for
```

`setup.py` begins with the physical inputs in SI units: lengths in m, velocities in m/s, kinematic viscosity in m²/s, times in s. Below them, one function builds the solver configuration, and the last lines run it. To make your own variant, copy the directory and edit the constants:

```bash
cp -r tutorials/vpm/02_vortex_ring ~/ring && cd ~/ring
./allrun.sh && ./allplot.sh
```

When checking convergence, refine the spacing and the time step together. Results go to `solution/` and `samples/`, figures to `figures/`; see [saved results](solution_layout.md), [continuation](continuation.md) and [visualization](visualization.md).

## FVM: viscous flow on a mesh

| Case | Physics | You learn |
| --- | --- | --- |
| [Taylor–Green vortex](../tutorials/fvm/taylor_green/README.md) | Decaying periodic vortices; exact solution | Cyclic boundaries, time and convection schemes, numerical dissipation |
| [Flat-plate boundary layer](../tutorials/fvm/boundary_layer/README.md) | Laminar flat plate; Blasius theory | Stretched wall grids, inlet/outlet/wall, skin friction |
| [Backward-facing step](../tutorials/fvm/step_profile/README.md) | Backward-facing step | Separation, reattachment, adaptive time step |
| [Square cylinder](../tutorials/fvm/cube_flow/README.md) | Square-cylinder shedding | Force coefficients, Strouhal number |
| [STL body](../tutorials/fvm/cartesian_mesher/README.md) | Flow around an STL body | Octree meshing, refinement regions, slip walls |
| [NACA 0012 airfoil](../tutorials/fvm/airfoil_flow/README.md) | NACA 0012 airfoil | Far-field boundaries, surface pressure |

## VPM and VLM: vortex particles and lifting surfaces

| Case | Physics | You learn |
| --- | --- | --- |
| [Lamb–Oseen vortices](../tutorials/vpm/01_lamb_oseen_vortex/README.md) | Viscous vortex, dipole and merger | Building particle fields; CS, RWM, DVH and GBD diffusion |
| [Vortex ring](../tutorials/vpm/02_vortex_ring/README.md) | Perturbed vortex ring | Stretching schemes, LES, state limits |
| [Leapfrogging rings](../tutorials/vpm/03_vortex_interactions/readme.md) | Leapfrogging rings | Long runs: DNS, selective LES and strength relaxation against an LBM reference |
| [Flat plate](../tutorials/vpm/04_flat_plate/README.md) | Inclined flat plate | Lifting surfaces, startup loads, wake shedding, moving frames |
| [Heaving delta wings](../tutorials/vpm/05_delta_wing/README.md) | Two heaving wings | Prescribed motion, wake–wing interaction |
| [Wind turbine](../tutorials/vpm/06_rotor_flow/README.md) | Wind turbine | Rotating blades, power and thrust, LES wake, filament splitting |
| [Quadcopter](../tutorials/vpm/07_quadcopter/README.md) | Four rotors in climb | Interacting rotor wakes, the Winckelmans kernel |

## Coupled FVM–VPM

| Case | Physics | You learn |
| --- | --- | --- |
| [Cylinder wake](../tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/README.md) | Laminar shedding at $Re=150$ | Coupling setup, planar flow with one particle layer, outflow fringes, a sensitivity study |
| [Cube wake](../tutorials/coupled_fvm_vpm/02_cube_flow/README.md) | Separated flow at $Re=1000$ | Three-dimensional coupling, LES in both solvers |

Each coupled case has a `reference_flow/` subdirectory that solves the same flow with the FVM alone on a large domain; the cube reference also shows a grid-convergence study and MPI parallel runs.

## Variants

Several cases run variants from one `setup.py`, listed one per line in `allrun.sh`: diffusion schemes (`python setup.py vortex GBD`), stretching schemes (`--variant dns_transposed`), stabilization mechanisms (`python setup.py baseline`), plate angles (`--angle -5`), grid levels (`--name grid_h0045 -h 0.045`), or coupling parameters (`--box-margin 1.4`). Run a single variant by copying its line. Each variant writes and resumes its own results.
