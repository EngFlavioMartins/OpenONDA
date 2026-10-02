# OpenONDA

<p align="center">
  <img src="docs/logos/Logo_V7_Color.png" alt="OpenONDA logo" width="900">
</p>

**OpenONDA (Operator for Numerical Design and Aerodynamics)** simulates incompressible viscous flows, vortex wakes, wings, and rotors with finite-volume (FVM), vortex-particle (VPM), and vortex-lattice (VLM) methods.

<p align="center">
  <img src="tutorials/vpm/05_delta_wing/assets/delta_wing_30fps.gif"
       alt="Looping visualization of the wake from two heaving delta wings"
       width="960">
</p>
<p align="center"><em>Wake from the <a href="tutorials/vpm/05_delta_wing/README.md">two heaving delta wings</a> tutorial.</em></p>

## Choose a model

- **FVM:** resolve velocity and pressure on a mesh, including viscous flow around walls and immersed bodies.
- **VPM/VLM:** evolve vorticity on particles; use lifting surfaces to generate wing and rotor wakes.
- **FVM–VPM:** resolve the flow near a body with FVM and carry its outer wake with particles.

## Installation

On Linux x86-64 or macOS, clone the repository and install:

```bash
git clone --depth 1 --branch development https://github.com/EngFlavioMartins/OpenONDA.git
cd OpenONDA
source install.sh
```

The installer creates and activates the **OpenONDA** Conda environment, installs the checkout editably, and supplies solver and plotting dependencies. In a new terminal, run `conda activate OpenONDA`. See [installation](docs/installation.md) for requirements.

## Examples

Run these short examples as Python scripts in a writable directory. Inputs use SI units: lengths in m, time in s, velocity in m/s, kinematic viscosity in m²/s, and circulation in m²/s.

### FVM: periodic uniform flow

This flow has $U=1$ m/s and $\nu=0.01$ m²/s. Opposite faces are periodic; `empty` span faces make the case two-dimensional. See [FVM physics, meshes, and boundaries](docs/fvm.md).

```python
from openonda import fvm

case = fvm.FVMCase(
    name="uniform-flow",
    directory="fvm-example",
    mesh=fvm.mesher.periodic_square_mesh(16),
    numerics=fvm.Numerics(
        transport=fvm.TransportConfig(kinematic_viscosity=0.01),
    ),
    boundaries=(
        fvm.BoundaryConfig.cyclic("xmin", "xmax"),
        fvm.BoundaryConfig.cyclic("xmax", "xmin"),
        fvm.BoundaryConfig.cyclic("ymin", "ymax"),
        fvm.BoundaryConfig.cyclic("ymax", "ymin"),
        fvm.BoundaryConfig.empty("zmin"),
        fvm.BoundaryConfig.empty("zmax"),
    ),
    initial_conditions=fvm.InitialFields(velocity=(1.0, 0.0, 0.0)),
    run=fvm.RunPlan(end_time=0.05, time_step_size=0.01),
)
with fvm.FVMSolver(case) as solver:
    solver.run()
```

### VPM: viscous vortex ring

The ring has radius $R=1$ m, physical vortex-core radius $a=0.2$ m, circulation $\Gamma=1$ m²/s, and $\nu=0.001$ m²/s. `spacing` sets particle resolution; `core_radius_ratio` sets numerical smoothing relative to spacing. Core spreading models viscous diffusion. See [particle distributions and diffusion](docs/vpm.md) and the [vortex-ring tutorial](tutorials/vpm/02_vortex_ring/README.md).

```python
from openonda import vpm

ring = vpm.VortexRing(
    radius=1.0, vortex_core_radius=0.2, circulation=1.0,
    kinematic_viscosity=0.001,
    distribution=vpm.ToroidalDistribution(
        ring_radius=1.0, tube_radius=0.4, spacing=0.2, core_radius_ratio=1.5,
    ),
)
case = vpm.VPMCase(
    directory="vpm-example",
    numerics=vpm.Numerics(
        time_step_size=0.01, compute_device="CPU", max_n_particles=5000,
        viscous=vpm.ViscousConfig.cs(
            kinematic_viscosity=0.001, particle_spacing=0.2,
        ),
    ),
    initial_conditions=(ring,),
    run=vpm.RunPlan(steps=5),
)
vpm.VPMSolver(case).run()
```

Open `fvm-example/solution/fvm.pvd` or `vpm-example/solution/vpm.pvd` in ParaView to inspect the flow.

## Tutorials

```bash
openonda tutorial list
openonda tutorial run fvm/taylor_green --workspace ./first-flow
openonda tutorial plot fvm/taylor_green --workspace ./first-flow
```

Choose a physical problem in the [tutorial index](docs/tutorials.md), then edit its `setup.py` for geometry, viscosity, resolution, and run duration. Each case guide links to the relevant model definitions.

## Documentation

- [FVM](docs/fvm.md): equations, units, meshes, boundary conditions, and turbulence.
- [VPM/VLM](docs/vpm.md): vortex strength, particle distributions, diffusion, LES, and lifting surfaces.
- [FVM–VPM coupling](docs/coupling.md): transfer regions, interface conditions, and particle resolution.
- [Tutorials](docs/tutorials.md): choose, configure, and run a physical case.
- [Continuing simulations](docs/continuation.md): resume or extend a run.
- [Results](docs/solution_layout.md) and [visualization](docs/visualization.md): locate fields and compare them.
- [Installation](docs/installation.md) and [numerical references](source/solvers/vpm/REFERENCES.md).

Report problems through [GitHub issues](https://github.com/EngFlavioMartins/OpenONDA/issues); contributors can use the [test guide](tests/README.md). License: [GPL-3.0-or-later](license). Research citation: [citation.cff](citation.cff).
