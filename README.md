# OpenONDA

<p align="center">
  <img src="docs/logos/Logo_V7_Color.png" alt="OpenONDA logo" width="900">
</p>

**OpenONDA is a Python library for simulating fluid flow and aerodynamics.**
It combines finite-volume, vortex-particle, and vortex-lattice methods to study
viscous flows, wakes, wings, rotors, and interacting vortices.

<p align="center">
  <img src="tutorials/vpm/05_delta_wing/assets/delta_wing_30fps.gif"
       alt="Looping visualization of the wake from two heaving delta wings"
       width="960">
</p>

<p align="center"><em>Vortex wake from an archived run of the <a href="tutorials/vpm/05_delta_wing/README.md">two heaving delta wings</a> tutorial.</em></p>

## Main capabilities

- **FVM:** incompressible flow, native mesh generation, immersed bodies, and sampling.
- **VPM/VLM:** particle-based wake evolution and aerodynamic lifting surfaces, with CPU and supported GPU backends.
- **Hybrid coupling:** native FVM near-field flow coupled to a VPM outer domain.

## Installation

On Linux x86-64 or macOS, clone the repository and install:

```bash
git clone --depth 1 --branch development https://github.com/EngFlavioMartins/OpenONDA.git
cd OpenONDA
source install.sh
```

The installer creates and activates the **OpenONDA** Conda environment, installs the checkout editably, and supplies solver and plotting dependencies. In a new terminal, run `conda activate OpenONDA`. See [installation](docs/installation.md) for requirements.

## Two small examples

Run either snippet as a Python script from any writable directory. These are
coarse, short examples to illustrate the API; all inputs use SI units.

### FVM: uniform flow in a periodic box

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

### VPM: a viscous vortex ring

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
    ),
    initial_conditions=(ring,),
    run=vpm.RunPlan(steps=5),
)
vpm.VPMSolver(case).run()
```

Open `fvm-example/solution/fvm.pvd` or `vpm-example/solution/vpm.pvd`
in ParaView. See the solver guides below for boundary conditions, sampling,
GPU induction, and restart.

## Tutorials

```bash
openonda tutorial list
openonda tutorial create vpm/vortex_ring ./ring-workspace
```

Working samples and restart checkpoints remain local. Published result snapshots
are published losslessly as versioned release assets, with manifests under each case's `assets/results`;
`allplot.sh` restores them in a fresh clone. The pip package contains tutorial
inputs and scripts; result archives are downloaded separately from GitHub releases.

Each case includes its own instructions and input assets. See the [tutorial guide](docs/tutorials.md) for running and editing local cases.

## Documentation

- [FVM solver guide](docs/fvm.md)
- [VPM solver guide](docs/vpm.md)
- [FVM--VPM coupling guide](docs/coupling.md)
- [Solution-output layout](docs/solution_layout.md)
- [Plotting and ParaView](docs/visualization.md)
- [VPM numerical references](source/solvers/vpm/REFERENCES.md)
- [Tutorials and hybrid examples](docs/tutorials.md)
- [Installation and optional tools](docs/installation.md)
- Command-line help: `openonda --help` and `openonda api fvm.FVMCase`

## Development and contributing

The installation already includes development tools. See the
[test index](tests/README.md) for verification commands.
Report problems through [GitHub issues](https://github.com/EngFlavioMartins/OpenONDA/issues).

## AI assistance

Human maintainers wrote the physics, numerical methods, and base architecture. AI tools assist with docstrings, documentation, code review and debugging. Human maintainers are responsible for final review and publication. All code in this repository were reviewed by a human.

OpenONDA is licensed under [GPL-3.0-or-later](license).
For research use, see [citation.cff](citation.cff).
