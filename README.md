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

- **FVM:** resolve velocity and pressure on a mesh, including viscous flow around walls.
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

### FVM: uniform flow through a box

This flow has $U=1$ m/s and $\nu=0.01$ m²/s. It enters at `inlet`, leaves at `outlet` and slips along the sides; `empty` span faces make the case two-dimensional. See [FVM physics, meshes, and boundaries](docs/fvm.md).

```python
import numpy as np

from openonda import fvm

x = np.linspace(0.0, 2.0, 17)  # m, 16 cells along x and y
mesh = fvm.mesher.box_mesh_3d(x, x, np.array([0.0, 0.125]), empty_spanwise=True)

case = fvm.FVMCase(
    name="uniform-flow",
    directory="fvm-example",
    mesh=mesh,
    numerics=fvm.Numerics(
        transport=fvm.TransportConfig(kinematic_viscosity=0.01),
    ),
    boundaries=(
        fvm.BoundaryConfig.inlet("inlet", [1.0, 0.0, 0.0]),
        fvm.BoundaryConfig.outlet("outlet"),
        fvm.BoundaryConfig.slip("ymin"),
        fvm.BoundaryConfig.slip("ymax"),
        fvm.BoundaryConfig.empty("zmin"),
        fvm.BoundaryConfig.empty("zmax"),
    ),
    initial_conditions=fvm.InitialFields(velocity=(1.0, 0.0, 0.0)),
    run=fvm.RunPlan(end_time=0.05, time_step_size=0.01),
)
with fvm.FVMSolver(case) as solver:
    solver.run()
```

### VPM: viscous vortex column

The Gaussian column has circulation $\Gamma=1$ m²/s, core radius $a=0.35$ m and $\nu=0.01$ m²/s. An initial condition is any object whose `build()` returns a `vpm.VortexParticleSet`: here particles sit on a lattice of spacing $h$ with strengths $\boldsymbol{\Gamma}_p=\boldsymbol{\omega}V_p$ and core radius $1.2h$. Core spreading models viscous diffusion. See [particle distributions and diffusion](docs/vpm.md) and the [vortex-ring tutorial](tutorials/vpm/02_vortex_ring/README.md).

```python
import numpy as np

from openonda import vpm

h, a, nu = 0.2, 0.35, 0.01  # particle spacing [m], core radius [m], viscosity [m²/s]


class GaussianColumn:
    """Vortex column of unit circulation along z, sampled on a cubic lattice."""

    def build(self):
        x = np.arange(-1.0, 1.0 + h / 2, h)
        position = np.stack(np.meshgrid(x, x, x, indexing="ij"), axis=-1).reshape(-1, 3)
        n = len(position)
        radius_squared = position[:, 0] ** 2 + position[:, 1] ** 2
        strength = np.zeros((n, 3))
        strength[:, 2] = np.exp(-radius_squared / a**2) / (np.pi * a**2) * h**3  # omega * V
        return vpm.VortexParticleSet(
            position=position,
            velocity=np.zeros((n, 3)),
            vortex_strength=strength,
            core_radius=np.full(n, 1.2 * h),
            particle_volume=np.full(n, h**3),
            kinematic_viscosity=np.full(n, nu),
            spacing=h,
        )


case = vpm.VPMCase(
    directory="vpm-example",
    numerics=vpm.Numerics(
        time_step_size=0.01,
        compute_device="CPU",
        max_n_particles=5000,
        viscous=vpm.ViscousConfig.cs(kinematic_viscosity=nu, particle_spacing=h),
    ),
    initial_conditions=(GaussianColumn(),),
    run=vpm.RunPlan(steps=2),
)
vpm.VPMSolver(case).run()
```

Open `fvm-example/solution/fvm.pvd` or `vpm-example/solution/vpm.pvd` in ParaView to inspect the flow.

## Tutorials

```bash
cd tutorials/fvm/taylor_green
./allrun.sh
./allplot.sh
```

Choose a physical problem in the [tutorial index](docs/tutorials.md), then edit its `setup.py` for geometry, viscosity, resolution, and run duration. Each case guide links to the relevant model definitions.

## Documentation

- [Learning with the tutorials](docs/tutorials.md): the recommended starting point.
- [Finite-volume flow](docs/fvm.md): equations, meshing, boundary conditions, schemes and turbulence models.
- [Vortex particles and lifting surfaces](docs/vpm.md): particle fields, induction, diffusion, LES, stabilization, planar flow and wings.
- [FVM–VPM coupling](docs/coupling.md): combining a resolved near field with a particle wake.
- [Continuing simulations](docs/continuation.md), [saved results](docs/solution_layout.md) and [visualization](docs/visualization.md).
- [Installation](docs/installation.md) and [numerical references](openonda/vpm/REFERENCES.md).

The complete manual is also available as a single PDF, `docs/OpenONDA_manual.pdf`; rebuild it with `python docs/manual/build_manual.py` (requires pandoc and XeLaTeX).

Report problems through [GitHub issues](https://github.com/EngFlavioMartins/OpenONDA/issues); contributors can use the [test guide](tests/README.md). License: [GPL-3.0-or-later](license). Research citation: [citation.cff](citation.cff).
